"""Tests for logit-bias utilities + decoder integration."""

from __future__ import annotations

import torch
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM
from saint_llm_inference.logit_bias import (
    apply_logit_bias,
    forbid_tokens_bias,
    force_token_bias,
)
from saint_llm_inference.streaming import (
    stream_greedy_decode,
    stream_top_p_sample,
)
from torch import nn

# ---- pure-function tests --------------------------------------------


def test_apply_bias_none_is_noop() -> None:
    logits = torch.randn(2, 8)
    out = apply_logit_bias(logits, None)
    assert torch.equal(out, logits)


def test_apply_bias_empty_dict_is_noop() -> None:
    logits = torch.randn(2, 8)
    out = apply_logit_bias(logits, {})
    assert torch.equal(out, logits)


def test_apply_bias_shifts_indexed_tokens() -> None:
    logits = torch.zeros(1, 8)
    out = apply_logit_bias(logits, {3: 5.0, 6: -2.0})
    assert out[0, 3].item() == 5.0
    assert out[0, 6].item() == -2.0
    # Other entries unchanged.
    assert out[0, 0].item() == 0.0


def test_apply_bias_silently_drops_oob_ids() -> None:
    logits = torch.zeros(1, 4)
    out = apply_logit_bias(logits, {2: 1.0, 99: 1.0})  # 99 out of range
    assert out[0, 2].item() == 1.0
    # No crash, no change for other valid entries.
    assert out[0, 0].item() == 0.0


def test_apply_bias_returns_new_tensor() -> None:
    """Bias application is non-destructive."""
    logits = torch.zeros(1, 4)
    apply_logit_bias(logits, {0: 1.0})
    assert logits[0, 0].item() == 0.0


def test_force_token_bias_helper() -> None:
    bias = force_token_bias(7, magnitude=10.0)
    assert bias == {7: 10.0}


def test_forbid_tokens_bias_helper() -> None:
    bias = forbid_tokens_bias([1, 3, 5], magnitude=-5.0)
    assert bias == {1: -5.0, 3: -5.0, 5: -5.0}


# ---- streaming-decoder integration ----------------------------------


def _model() -> SaintLLM:
    torch.manual_seed(0)
    m = SaintLLM(ModelConfig.tiny())
    m.eval()
    return m


def test_stream_greedy_force_token_picks_biased_id() -> None:
    """With +inf bias on token 5, every step emits token 5."""
    model = _model()
    prompt = torch.zeros(1, 4, dtype=torch.long)
    out = list(stream_greedy_decode(
        model, prompt, max_new_tokens=3,
        logit_bias={5: 1e9},
    ))
    for t in out:
        assert t.item() == 5


def test_stream_greedy_forbid_specific_id() -> None:
    """Forbidding the otherwise-argmax token forces a different pick."""
    model = _model()
    prompt = torch.zeros(1, 4, dtype=torch.long)
    # First, capture the unbiased greedy pick.
    unbiased = next(stream_greedy_decode(
        model, prompt, max_new_tokens=1,
    )).item()
    # Now forbid that token.
    biased = next(stream_greedy_decode(
        model, prompt, max_new_tokens=1,
        logit_bias={unbiased: -1e9},
    )).item()
    assert biased != unbiased


def test_stream_top_p_force_token_picks_biased_id() -> None:
    model = _model()
    prompt = torch.zeros(1, 4, dtype=torch.long)
    out = list(stream_top_p_sample(
        model, prompt, max_new_tokens=2,
        temperature=1.0, p=0.9,
        logit_bias={3: 1e9},
        generator=torch.Generator().manual_seed(0),
    ))
    for t in out:
        assert t.item() == 3


def test_stream_no_bias_unchanged() -> None:
    """With logit_bias=None the output matches a call without the kwarg."""
    model = _model()
    prompt = torch.zeros(1, 3, dtype=torch.long)
    a = list(stream_greedy_decode(
        model, prompt, max_new_tokens=3, logit_bias=None,
    ))
    b = list(stream_greedy_decode(model, prompt, max_new_tokens=3))
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)


def test_stream_bias_changes_distribution_in_top_p() -> None:
    """Strong bias toward a specific token produces more emissions of it."""
    model = _model()
    prompt = torch.zeros(1, 3, dtype=torch.long)
    target = 4

    counts_unbiased = 0
    for seed in range(20):
        out = list(stream_top_p_sample(
            model, prompt, max_new_tokens=4,
            temperature=1.0, p=0.95,
            generator=torch.Generator().manual_seed(seed),
        ))
        counts_unbiased += sum(1 for t in out if t.item() == target)

    counts_biased = 0
    for seed in range(20):
        out = list(stream_top_p_sample(
            model, prompt, max_new_tokens=4,
            temperature=1.0, p=0.95,
            logit_bias={target: 5.0},
            generator=torch.Generator().manual_seed(seed),
        ))
        counts_biased += sum(1 for t in out if t.item() == target)

    # Biased run should emit `target` more often than unbiased.
    assert counts_biased > counts_unbiased


def test_stream_force_then_eos_combination() -> None:
    """Force the EOS token via bias to terminate immediately."""
    model = _model()
    eos = 7
    prompt = torch.zeros(1, 4, dtype=torch.long)
    out = list(stream_greedy_decode(
        model, prompt, max_new_tokens=10,
        eos_token=eos, logit_bias=force_token_bias(eos),
    ))
    # First yielded token is forced eos -> stream stops after one yield.
    assert len(out) == 1
    assert out[0].item() == eos


def test_stream_bias_with_stop_sequences_compose() -> None:
    """Bias + stop_sequences compose: bias the stop-sequence token directly."""
    model = _model()
    target = 5
    prompt = torch.zeros(1, 4, dtype=torch.long)
    out = list(stream_greedy_decode(
        model, prompt, max_new_tokens=10,
        logit_bias={target: 1e9},
        stop_sequences=[[target]],
    ))
    # Bias forces target every step; stop_sequences fires on first emit.
    assert len(out) == 1
    assert out[0].item() == target


class _FreeLM(nn.Module):
    """Uniform logits over vocab; sampling decisions driven entirely by bias."""

    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, 4)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t = tokens.shape
        return {"logits": torch.zeros(b, t, self.vocab)}


def test_uniform_model_with_bias_picks_biased_argmax() -> None:
    """When logits are uniform, the biased token is the unique argmax."""
    model = _FreeLM(vocab=8)
    out = list(stream_greedy_decode(
        model, torch.zeros(1, 2, dtype=torch.long),
        max_new_tokens=3, logit_bias={4: 1.0},
    ))
    for t in out:
        assert t.item() == 4
