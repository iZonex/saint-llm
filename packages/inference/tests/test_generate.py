"""Decoding loops: shape, dtype, eos, determinism, top-k semantics."""

from __future__ import annotations

import pytest
import torch
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM
from saint_llm_inference import greedy_decode, top_k_sample


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
