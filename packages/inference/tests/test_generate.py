"""Decoding loops: shape, dtype, eos, determinism, top-k semantics."""

from __future__ import annotations

import pytest
import torch
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM
from saint_llm_inference import greedy_decode, top_k_sample, top_p_sample
from saint_llm_inference.generate import _filter_top_p


@pytest.fixture(scope="module")
def model() -> SaintLLM:
    torch.manual_seed(0)
    m = SaintLLM(ModelConfig.tiny())
    m.eval()
    return m


def test_greedy_output_shape(model: SaintLLM) -> None:
    prompt = torch.zeros(2, 4, dtype=torch.long)
    out = greedy_decode(model, prompt, max_new_tokens=6)
    assert out.shape == (2, 10)
    assert out.dtype == torch.long


def test_greedy_preserves_prompt_prefix(model: SaintLLM) -> None:
    prompt = torch.tensor([[3, 7, 1, 5]])
    out = greedy_decode(model, prompt, max_new_tokens=4)
    assert torch.equal(out[:, :4], prompt)


def test_greedy_is_deterministic(model: SaintLLM) -> None:
    prompt = torch.zeros(1, 4, dtype=torch.long)
    a = greedy_decode(model, prompt, max_new_tokens=8)
    b = greedy_decode(model, prompt, max_new_tokens=8)
    assert torch.equal(a, b)


def test_greedy_eos_short_circuits(model: SaintLLM) -> None:
    """Pass an eos token equal to the model's argmax; loop should still produce
    full max_new_tokens-wide output (eos-padded tail)."""
    prompt = torch.zeros(1, 4, dtype=torch.long)
    # Find what greedy picks for the first new token, use that as eos.
    out_no_eos = greedy_decode(model, prompt, max_new_tokens=1)
    eos = int(out_no_eos[0, -1].item())
    out_eos = greedy_decode(model, prompt, max_new_tokens=8, eos_token=eos)
    assert out_eos.shape == (1, 12)
    # Tail past first emission should be eos-padded.
    assert (out_eos[0, 4:] == eos).all()


def test_greedy_rejects_non_2d_prompt(model: SaintLLM) -> None:
    with pytest.raises(ValueError, match="must be 2D"):
        greedy_decode(model, torch.zeros(4, dtype=torch.long), max_new_tokens=2)


def test_top_k_output_shape(model: SaintLLM) -> None:
    prompt = torch.zeros(2, 4, dtype=torch.long)
    out = top_k_sample(model, prompt, max_new_tokens=6, k=10, temperature=1.0)
    assert out.shape == (2, 10)
    assert out.dtype == torch.long


def test_top_k_preserves_prompt(model: SaintLLM) -> None:
    prompt = torch.tensor([[3, 7, 1, 5]])
    out = top_k_sample(model, prompt, max_new_tokens=4, k=10, temperature=1.0)
    assert torch.equal(out[:, :4], prompt)


def test_top_k_deterministic_with_generator(model: SaintLLM) -> None:
    prompt = torch.zeros(1, 4, dtype=torch.long)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a = top_k_sample(model, prompt, max_new_tokens=6, k=10, temperature=0.7, generator=g1)
    b = top_k_sample(model, prompt, max_new_tokens=6, k=10, temperature=0.7, generator=g2)
    assert torch.equal(a, b)


def test_top_k_zero_temperature_falls_back_to_greedy(model: SaintLLM) -> None:
    prompt = torch.zeros(1, 4, dtype=torch.long)
    greedy_out = greedy_decode(model, prompt, max_new_tokens=6)
    sample_out = top_k_sample(model, prompt, max_new_tokens=6, k=10, temperature=0.0)
    assert torch.equal(greedy_out, sample_out)


def test_top_k_only_picks_from_top_k_tokens(model: SaintLLM) -> None:
    """Sample 50 generations with k=2; every emitted token must be one of
    the top-2 logits at that position."""
    prompt = torch.zeros(1, 4, dtype=torch.long)
    g = torch.Generator().manual_seed(0)

    with torch.no_grad():
        baseline_logits = model(prompt)["logits"][0, -1]
        topk_indices = baseline_logits.topk(2).indices.tolist()

    samples: set[int] = set()
    for _ in range(50):
        out = top_k_sample(model, prompt, max_new_tokens=1, k=2, temperature=1.0, generator=g)
        samples.add(int(out[0, -1].item()))
    assert samples.issubset(set(topk_indices))
    assert samples  # at least one sample drawn


def test_top_k_rejects_invalid_args(model: SaintLLM) -> None:
    prompt = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="k must be positive"):
        top_k_sample(model, prompt, max_new_tokens=2, k=0)
    with pytest.raises(ValueError, match="temperature must be"):
        top_k_sample(model, prompt, max_new_tokens=2, k=10, temperature=-0.5)


def test_top_k_rejects_non_2d_prompt(model: SaintLLM) -> None:
    with pytest.raises(ValueError, match="must be 2D"):
        top_k_sample(model, torch.zeros(4, dtype=torch.long), max_new_tokens=2, k=10)


def test_filter_top_p_keeps_at_least_top_token() -> None:
    """Even with p=0, the highest-probability token must survive."""
    logits = torch.tensor([[0.1, 5.0, 0.2, 0.3]])
    out = _filter_top_p(logits, p=0.0)
    finite = torch.isfinite(out)
    assert finite[0].sum() == 1
    assert finite[0, 1].item()  # the argmax position


def test_filter_top_p_keeps_full_distribution_at_p_one() -> None:
    logits = torch.randn(2, 8)
    out = _filter_top_p(logits, p=1.0)
    assert torch.equal(out, logits)


def test_filter_top_p_monotone_in_p() -> None:
    """Higher p → at least as many tokens kept."""
    torch.manual_seed(0)
    logits = torch.randn(1, 32)
    kept_low = torch.isfinite(_filter_top_p(logits, p=0.3)).sum().item()
    kept_high = torch.isfinite(_filter_top_p(logits, p=0.9)).sum().item()
    assert kept_low <= kept_high


def test_top_p_output_shape(model: SaintLLM) -> None:
    prompt = torch.zeros(2, 4, dtype=torch.long)
    out = top_p_sample(model, prompt, max_new_tokens=6, p=0.9)
    assert out.shape == (2, 10)
    assert out.dtype == torch.long


def test_top_p_deterministic_with_generator(model: SaintLLM) -> None:
    prompt = torch.zeros(1, 4, dtype=torch.long)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = top_p_sample(model, prompt, max_new_tokens=6, p=0.95, generator=g1)
    b = top_p_sample(model, prompt, max_new_tokens=6, p=0.95, generator=g2)
    assert torch.equal(a, b)


def test_top_p_zero_temperature_falls_back_to_greedy(model: SaintLLM) -> None:
    prompt = torch.zeros(1, 4, dtype=torch.long)
    greedy_out = greedy_decode(model, prompt, max_new_tokens=6)
    sample_out = top_p_sample(model, prompt, max_new_tokens=6, p=0.9, temperature=0.0)
    assert torch.equal(greedy_out, sample_out)


def test_top_p_combined_with_top_k(model: SaintLLM) -> None:
    """top_p + top_k together must restrict to top_k AND nucleus."""
    prompt = torch.zeros(1, 4, dtype=torch.long)
    g = torch.Generator().manual_seed(0)

    with torch.no_grad():
        baseline = model(prompt)["logits"][0, -1]
    topk_indices = set(baseline.topk(3).indices.tolist())

    samples: set[int] = set()
    for _ in range(50):
        out = top_p_sample(
            model, prompt, max_new_tokens=1, p=0.99, top_k=3, temperature=1.0, generator=g,
        )
        samples.add(int(out[0, -1].item()))
    assert samples.issubset(topk_indices)


def test_top_p_rejects_invalid_args(model: SaintLLM) -> None:
    prompt = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="p must be in"):
        top_p_sample(model, prompt, max_new_tokens=2, p=0.0)
    with pytest.raises(ValueError, match="p must be in"):
        top_p_sample(model, prompt, max_new_tokens=2, p=1.5)
    with pytest.raises(ValueError, match="temperature must be"):
        top_p_sample(model, prompt, max_new_tokens=2, p=0.9, temperature=-0.1)
    with pytest.raises(ValueError, match="top_k must be positive"):
        top_p_sample(model, prompt, max_new_tokens=2, p=0.9, top_k=0)
    with pytest.raises(ValueError, match="must be 2D"):
        top_p_sample(model, torch.zeros(4, dtype=torch.long), max_new_tokens=2, p=0.9)
