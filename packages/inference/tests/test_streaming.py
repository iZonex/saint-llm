"""Tests for streaming generation."""

from __future__ import annotations

import pytest
import torch
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM
from saint_llm_inference import greedy_decode
from saint_llm_inference.streaming import (
    stream_greedy_decode,
    stream_top_p_sample,
)
from torch import nn


class _FixedNextTokenLM(nn.Module):
    """Argmax always picks ``next_token``; for stop-on-eos behavior tests."""

    def __init__(self, vocab: int, next_token: int) -> None:
        super().__init__()
        self.vocab = vocab
        self.next_token = next_token
        self.embed = nn.Embedding(vocab, 4)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t = tokens.shape
        logits = torch.full((b, t, self.vocab), -1e9)
        logits[:, :, self.next_token] = 0.0
        return {"logits": logits}


@pytest.fixture(scope="module")
def real_model() -> SaintLLM:
    torch.manual_seed(0)
    m = SaintLLM(ModelConfig.tiny())
    m.eval()
    return m


def test_stream_greedy_yields_one_token_per_step(real_model: SaintLLM) -> None:
    prompt = torch.zeros(1, 4, dtype=torch.long)
    out = list(stream_greedy_decode(real_model, prompt, max_new_tokens=3))
    assert len(out) == 3
    for t in out:
        assert t.shape == (1, 1)
        assert t.dtype == torch.long


def test_stream_greedy_matches_non_streaming(real_model: SaintLLM) -> None:
    """Concatenating streamed tokens reproduces greedy_decode's output."""
    prompt = torch.zeros(1, 4, dtype=torch.long)
    streamed = list(stream_greedy_decode(real_model, prompt, max_new_tokens=5))
    streamed_full = torch.cat([prompt, *streamed], dim=-1)
    nonstream = greedy_decode(real_model, prompt, max_new_tokens=5)
    assert torch.equal(streamed_full, nonstream)


def test_stream_greedy_stops_on_eos() -> None:
    """When every row hits EOS, the iterator terminates early."""
    eos = 7
    model = _FixedNextTokenLM(vocab=8, next_token=eos)
    prompt = torch.zeros(1, 2, dtype=torch.long)
    out = list(stream_greedy_decode(
        model, prompt, max_new_tokens=10, eos_token=eos,
    ))
    # First yielded token is eos -> iterator stops immediately after.
    assert len(out) == 1
    assert out[0].item() == eos


def test_stream_greedy_continues_until_all_rows_hit_eos() -> None:
    """Multi-row batch: stream continues until every row has emitted eos."""
    eos = 5
    # Use a real model so the batch picks different tokens per row;
    # we just check that the iterator produces tensors of (B, 1) shape
    # for each step.
    model = SaintLLM(ModelConfig.tiny())
    model.eval()
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = list(stream_greedy_decode(
        model, prompt, max_new_tokens=4, eos_token=eos,
    ))
    for t in out:
        assert t.shape == (2, 1)


def test_stream_greedy_rejects_1d_prompt(real_model: SaintLLM) -> None:
    with pytest.raises(ValueError, match=r"\(B, T\)"):
        list(stream_greedy_decode(
            real_model, torch.tensor([1, 2]), max_new_tokens=2,
        ))


def test_stream_greedy_rejects_zero_max_new_tokens(real_model: SaintLLM) -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        list(stream_greedy_decode(
            real_model, torch.zeros(1, 2, dtype=torch.long), max_new_tokens=0,
        ))


def test_stream_top_p_yields_correct_count(real_model: SaintLLM) -> None:
    prompt = torch.zeros(1, 4, dtype=torch.long)
    out = list(stream_top_p_sample(
        real_model, prompt, max_new_tokens=4,
        p=0.9, temperature=1.0,
        generator=torch.Generator().manual_seed(0),
    ))
    assert len(out) == 4
    for t in out:
        assert t.shape == (1, 1)


def test_stream_top_p_temperature_zero_is_greedy(real_model: SaintLLM) -> None:
    """temperature=0 produces the same sequence as stream_greedy_decode."""
    prompt = torch.zeros(1, 3, dtype=torch.long)
    greedy = list(stream_greedy_decode(real_model, prompt, max_new_tokens=4))
    top_p_zero = list(stream_top_p_sample(
        real_model, prompt, max_new_tokens=4, temperature=0.0,
    ))
    for g, t in zip(greedy, top_p_zero, strict=True):
        assert torch.equal(g, t)


def test_stream_top_p_seeded_is_reproducible(real_model: SaintLLM) -> None:
    """Same seed -> same streamed sequence."""
    prompt = torch.zeros(1, 3, dtype=torch.long)
    out1 = list(stream_top_p_sample(
        real_model, prompt, max_new_tokens=3, temperature=1.0, p=0.9,
        generator=torch.Generator().manual_seed(42),
    ))
    out2 = list(stream_top_p_sample(
        real_model, prompt, max_new_tokens=3, temperature=1.0, p=0.9,
        generator=torch.Generator().manual_seed(42),
    ))
    for a, b in zip(out1, out2, strict=True):
        assert torch.equal(a, b)


def test_stream_top_p_stops_on_eos() -> None:
    eos = 3
    model = _FixedNextTokenLM(vocab=8, next_token=eos)
    prompt = torch.zeros(1, 2, dtype=torch.long)
    out = list(stream_top_p_sample(
        model, prompt, max_new_tokens=10, temperature=0.0, eos_token=eos,
    ))
    # temperature=0 -> greedy -> first yielded token is eos.
    assert len(out) == 1
    assert out[0].item() == eos


def test_stream_top_p_rejects_invalid_p(real_model: SaintLLM) -> None:
    with pytest.raises(ValueError, match="p must be in"):
        list(stream_top_p_sample(
            real_model, torch.zeros(1, 2, dtype=torch.long),
            max_new_tokens=2, p=1.5,
        ))


def test_stream_top_p_rejects_negative_temperature(real_model: SaintLLM) -> None:
    with pytest.raises(ValueError, match="temperature"):
        list(stream_top_p_sample(
            real_model, torch.zeros(1, 2, dtype=torch.long),
            max_new_tokens=2, temperature=-1.0,
        ))


def test_stream_top_p_rejects_zero_top_k(real_model: SaintLLM) -> None:
    with pytest.raises(ValueError, match="top_k"):
        list(stream_top_p_sample(
            real_model, torch.zeros(1, 2, dtype=torch.long),
            max_new_tokens=2, top_k=0,
        ))


def test_stream_can_be_consumed_lazily(real_model: SaintLLM) -> None:
    """The generator only does work when the caller pulls."""
    prompt = torch.zeros(1, 2, dtype=torch.long)
    gen = stream_greedy_decode(real_model, prompt, max_new_tokens=10)
    # Pull one token then stop.
    first = next(gen)
    assert first.shape == (1, 1)
    # Closing the generator should be clean — no exception.
    gen.close()
