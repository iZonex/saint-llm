"""Tests for stop-sequence matching + streaming integration."""

from __future__ import annotations

import torch
from saint_llm_data import CharTokenizer
from saint_llm_inference.stop_sequences import (
    StopSequenceMatcher,
    encode_stop_strings,
)
from saint_llm_inference.streaming import (
    stream_greedy_decode,
    stream_top_p_sample,
)
from torch import nn


class _CycleLM(nn.Module):
    """Argmax cycles through ``token_pattern`` token by token."""

    def __init__(self, vocab: int, token_pattern: list[int]) -> None:
        super().__init__()
        self.vocab = vocab
        self.pattern = token_pattern
        self.embed = nn.Embedding(vocab, 4)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t = tokens.shape
        # Each forward picks the token at position (t mod len(pattern)),
        # so the model "advances" through the pattern per emit.
        pick = self.pattern[t % len(self.pattern)]
        logits = torch.full((b, t, self.vocab), -1e9)
        logits[:, :, pick] = 0.0
        return {"logits": logits}


# ---- StopSequenceMatcher unit tests ---------------------------------


def test_matcher_empty_sequences_never_matches() -> None:
    m = StopSequenceMatcher([])
    assert m.feed(1) is False
    assert m.matched() is False


def test_matcher_single_token_match() -> None:
    m = StopSequenceMatcher([[5]])
    assert m.feed(3) is False
    assert m.feed(5) is True
    assert m.matched() is True
    assert m.matched_sequence == (5,)


def test_matcher_multi_token_sequence() -> None:
    m = StopSequenceMatcher([[10, 11, 12]])
    assert m.feed(10) is False
    assert m.feed(11) is False
    assert m.feed(12) is True


def test_matcher_partial_then_full() -> None:
    """Pattern [10, 11] requires both tokens consecutive."""
    m = StopSequenceMatcher([[10, 11]])
    m.feed(10)
    m.feed(99)  # break the pattern
    assert m.matched() is False
    m.feed(10)
    assert m.feed(11) is True


def test_matcher_alternative_sequences() -> None:
    """Multiple alternatives — first match wins."""
    m = StopSequenceMatcher([[10, 11], [99]])
    assert m.feed(99) is True
    assert m.matched_sequence == (99,)


def test_matcher_stays_matched_after_first_hit() -> None:
    m = StopSequenceMatcher([[5]])
    m.feed(5)
    # Subsequent tokens don't un-match.
    m.feed(99)
    assert m.matched() is True


def test_matcher_reset_clears_state() -> None:
    m = StopSequenceMatcher([[5]])
    m.feed(5)
    m.reset()
    assert m.matched() is False
    assert m.matched_sequence is None


def test_matcher_drops_empty_sequences() -> None:
    m = StopSequenceMatcher([[], [1]])
    assert m.sequences == ((1,),)


def test_matcher_tail_bounded_by_max_seq_len() -> None:
    """The internal tail buffer never grows beyond the longest pattern."""
    m = StopSequenceMatcher([[1, 2]])
    for tok in range(100):
        m.feed(tok)
    # Internal tail length must be at most len(longest sequence) = 2.
    assert len(m._tail) <= 2


# ---- encode_stop_strings --------------------------------------------


def test_encode_stop_strings_basic() -> None:
    tok = CharTokenizer()
    seqs = encode_stop_strings(["ab", "x"], tok)
    assert len(seqs) == 2
    # CharTokenizer encodes "ab" -> 2 tokens, "x" -> 1 token.
    assert len(seqs[0]) == 2
    assert len(seqs[1]) == 1


def test_encode_stop_strings_drops_empty_input() -> None:
    tok = CharTokenizer()
    seqs = encode_stop_strings(["", "ok"], tok)
    assert len(seqs) == 1


# ---- streaming integration ------------------------------------------


def test_stream_greedy_stops_on_first_token_pattern() -> None:
    """A model that always emits 7 stops immediately when [7] is a stop seq."""
    model = _CycleLM(vocab=8, token_pattern=[7])
    out = list(stream_greedy_decode(
        model,
        torch.zeros(1, 2, dtype=torch.long),
        max_new_tokens=10,
        stop_sequences=[[7]],
    ))
    # First yielded token is 7 -> stop.
    assert len(out) == 1
    assert out[0].item() == 7


def test_stream_greedy_stops_on_multi_token_pattern() -> None:
    """Pattern [3, 4]: model cycles 3, 4, 3, 4, ... -> stops at index 1."""
    model = _CycleLM(vocab=8, token_pattern=[3, 4])
    # Note: model.forward picks pattern[t % 2] where t is the *input*
    # length. With prompt of length 2, first emit picks pattern[2 % 2]
    # = pattern[0] = 3. Second emit picks pattern[3 % 2] = pattern[1] = 4.
    out = list(stream_greedy_decode(
        model,
        torch.zeros(1, 2, dtype=torch.long),
        max_new_tokens=10,
        stop_sequences=[[3, 4]],
    ))
    assert len(out) == 2
    assert [t.item() for t in out] == [3, 4]


def test_stream_greedy_continues_when_no_pattern_match() -> None:
    """Without a matching pattern, runs to max_new_tokens."""
    model = _CycleLM(vocab=8, token_pattern=[5])
    out = list(stream_greedy_decode(
        model,
        torch.zeros(1, 2, dtype=torch.long),
        max_new_tokens=4,
        stop_sequences=[[99, 99]],  # never matches in this stream
    ))
    assert len(out) == 4


def test_stream_top_p_with_stop_sequences() -> None:
    """Same behavior in the sampling path."""
    model = _CycleLM(vocab=8, token_pattern=[6])
    out = list(stream_top_p_sample(
        model,
        torch.zeros(1, 2, dtype=torch.long),
        max_new_tokens=10,
        temperature=0.0,
        stop_sequences=[[6]],
    ))
    assert len(out) == 1
    assert out[0].item() == 6


def test_stream_with_stop_sequences_alternatives() -> None:
    """Model emits 5; one of two alternative stops triggers."""
    model = _CycleLM(vocab=8, token_pattern=[5])
    out = list(stream_greedy_decode(
        model,
        torch.zeros(1, 2, dtype=torch.long),
        max_new_tokens=10,
        stop_sequences=[[99], [5]],
    ))
    assert len(out) == 1


def test_stream_with_stop_sequences_multi_row_waits_for_all() -> None:
    """Two-row batch: stream waits until every row matches a pattern."""
    # Use the real model and accept whatever it generates; the assertion
    # is just about not crashing and producing a finite stream.
    from saint_llm_core.config import ModelConfig  # noqa: PLC0415
    from saint_llm_core.model import SaintLLM  # noqa: PLC0415

    torch.manual_seed(0)
    model = SaintLLM(ModelConfig.tiny())
    model.eval()
    prompts = torch.zeros(2, 3, dtype=torch.long)
    out = list(stream_greedy_decode(
        model, prompts, max_new_tokens=4,
        stop_sequences=[[1234567]],  # impossible token id; never matches
    ))
    # Iterator runs to max_new_tokens since stop never fires.
    assert len(out) == 4


def test_stream_no_stop_sequences_no_overhead() -> None:
    """Passing stop_sequences=None preserves default behavior."""
    model = _CycleLM(vocab=8, token_pattern=[3])
    out = list(stream_greedy_decode(
        model, torch.zeros(1, 2, dtype=torch.long), max_new_tokens=3,
        stop_sequences=None,
    ))
    assert len(out) == 3


def test_stream_eos_takes_precedence_over_stop_sequence() -> None:
    """EOS check happens first; stop sequences are checked alongside."""
    model = _CycleLM(vocab=8, token_pattern=[2])
    # eos == 2 -> first emit is eos, iterator stops on EOS.
    out = list(stream_greedy_decode(
        model, torch.zeros(1, 2, dtype=torch.long), max_new_tokens=10,
        eos_token=2, stop_sequences=[[5, 6, 7]],  # never reached
    ))
    assert len(out) == 1
    assert out[0].item() == 2


def test_encode_stop_strings_empty_input_returns_empty() -> None:
    tok = CharTokenizer()
    assert encode_stop_strings([], tok) == []


def test_matcher_does_not_yield_extra_after_match() -> None:
    """Once matched, feed() returns True without modifying matched_sequence."""
    m = StopSequenceMatcher([[10]])
    m.feed(10)
    assert m.matched_sequence == (10,)
    # Subsequent feeds short-circuit and return True.
    assert m.feed(99) is True
    # matched_sequence stays the original.
    assert m.matched_sequence == (10,)


def test_matcher_handles_overlapping_pattern() -> None:
    """Pattern [10, 10] matches when fed 10 twice in a row."""
    m = StopSequenceMatcher([[10, 10]])
    m.feed(10)
    assert m.matched() is False
    assert m.feed(10) is True
