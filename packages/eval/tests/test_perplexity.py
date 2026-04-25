"""Perplexity utility: shape, finiteness, ignore_index, training-decreases-PPL."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM
from saint_llm_eval import compute_perplexity, compute_perplexity_streaming


@pytest.fixture(scope="module")
def model() -> SaintLLM:
    torch.manual_seed(0)
    m = SaintLLM(ModelConfig.tiny())
    m.eval()
    return m


def test_compute_perplexity_returns_finite_scalar(model: SaintLLM) -> None:
    token_ids = torch.zeros(1, 16, dtype=torch.long)
    ppl = compute_perplexity(model, token_ids)
    assert isinstance(ppl, float)
    assert math.isfinite(ppl)
    assert ppl > 0.0


def test_compute_perplexity_rejects_too_short_sequence(model: SaintLLM) -> None:
    with pytest.raises(ValueError, match="at least 2 tokens"):
        compute_perplexity(model, torch.zeros(1, 1, dtype=torch.long))


def test_compute_perplexity_rejects_non_2d(model: SaintLLM) -> None:
    with pytest.raises(ValueError, match="must be 2D"):
        compute_perplexity(model, torch.zeros(16, dtype=torch.long))


def test_compute_perplexity_matches_exp_cross_entropy(model: SaintLLM) -> None:
    """Result must equal exp of the same cross-entropy loss the training step uses."""
    cfg = ModelConfig.tiny()
    token_ids = torch.zeros(1, 8, dtype=torch.long)
    with torch.no_grad():
        out = model(token_ids)
        logits = out["logits"]
        flat_logits = logits[:, :-1, :].reshape(-1, cfg.vocab_size).to(torch.float32)
        flat_labels = token_ids[:, 1:].reshape(-1)
        ce = F.cross_entropy(flat_logits, flat_labels).item()
    ppl = compute_perplexity(model, token_ids)
    assert ppl == pytest.approx(math.exp(ce), rel=1.0e-5)


def test_streaming_matches_single_batch_when_one_input(model: SaintLLM) -> None:
    token_ids = torch.zeros(1, 16, dtype=torch.long)
    single = compute_perplexity(model, token_ids)
    streamed = compute_perplexity_streaming(model, [token_ids])
    assert streamed == pytest.approx(single, rel=1.0e-5)


def test_streaming_skips_too_short_batches(model: SaintLLM) -> None:
    """A 1-token batch in the list must be silently ignored."""
    long_batch = torch.zeros(1, 16, dtype=torch.long)
    short_batch = torch.zeros(1, 1, dtype=torch.long)
    streamed = compute_perplexity_streaming(model, [long_batch, short_batch])
    only_long = compute_perplexity_streaming(model, [long_batch])
    assert streamed == pytest.approx(only_long, rel=1.0e-5)


def test_streaming_returns_inf_when_no_valid_batches(model: SaintLLM) -> None:
    streamed = compute_perplexity_streaming(model, [torch.zeros(1, 1, dtype=torch.long)])
    assert streamed == math.inf


def test_training_decreases_perplexity(model: SaintLLM) -> None:
    """Train a fresh model on a fixed sequence; held-out PPL on that sequence
    must drop. Validates that the perplexity utility tracks training signal."""
    cfg = ModelConfig.tiny()
    torch.manual_seed(1)
    fresh = SaintLLM(cfg)
    optim = torch.optim.AdamW(fresh.parameters(), lr=3.0e-3)
    token_ids = torch.randint(0, cfg.vocab_size, (1, 16))

    fresh.eval()
    ppl_before = compute_perplexity(fresh, token_ids)

    fresh.train()
    for _ in range(8):
        out = fresh(token_ids)
        loss = F.cross_entropy(
            out["logits"][:, :-1].reshape(-1, cfg.vocab_size),
            token_ids[:, 1:].reshape(-1),
        )
        optim.zero_grad()
        loss.backward()
        optim.step()

    fresh.eval()
    ppl_after = compute_perplexity(fresh, token_ids)
    assert math.isfinite(ppl_before) and math.isfinite(ppl_after)
    assert ppl_after < ppl_before, f"PPL did not decrease: {ppl_before} -> {ppl_after}"
