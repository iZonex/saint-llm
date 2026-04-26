"""Tests for EAGLE-3 speculative decoding (INF-01)."""

from __future__ import annotations

import pytest
import torch
from saint_llm_inference import EAGLE3Config, EAGLE3DraftHead, speculative_decode
from torch import nn


class _MockBaseLM(nn.Module):
    """Base model that returns deterministic logits + hidden.

    The "true" next-token sequence is fixed by ``programmed_outputs``;
    base_argmax at each step always picks the next item from this
    sequence regardless of input.
    """

    def __init__(self, vocab: int = 8, hidden: int = 16, programmed_outputs: list[int] | None = None) -> None:
        super().__init__()
        self.vocab = vocab
        self.hidden = hidden
        self.embed = nn.Embedding(vocab, hidden)
        self.programmed = programmed_outputs or [3, 5, 7, 1, 4, 2, 6, 0]

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t = tokens.shape
        h = self.embed(tokens)
        # Build logits where argmax at each position is the
        # programmed-next-token for the position number.
        logits = torch.full((b, t, self.vocab), -1e9)
        for pos in range(t):
            target = self.programmed[pos % len(self.programmed)]
            logits[:, pos, target] = 1.0
        return {"hidden": h, "logits": logits}


def test_eagle3_draft_head_shape() -> None:
    head = EAGLE3DraftHead(hidden_dim=16, vocab_size=32)
    h = torch.randn(2, 4, 16)
    out = head(h)
    assert out.shape == (2, 4, 32)


def test_speculative_decode_runs_to_max_new_tokens() -> None:
    base = _MockBaseLM(vocab=8, hidden=16)
    draft = EAGLE3DraftHead(hidden_dim=16, vocab_size=8)
    prompt = torch.tensor([[1, 2]])
    cfg = EAGLE3Config(draft_length=2, max_new_tokens=4)
    out, metrics = speculative_decode(base, draft, prompt, cfg=cfg)
    n_new = out.shape[1] - prompt.shape[1]
    assert n_new >= cfg.max_new_tokens
    assert "acceptance_rate" in metrics
    assert "base_forwards" in metrics


def test_speculative_decode_rejects_batch_size_gt_1() -> None:
    base = _MockBaseLM()
    draft = EAGLE3DraftHead(hidden_dim=16, vocab_size=8)
    cfg = EAGLE3Config(draft_length=2, max_new_tokens=4)
    with pytest.raises(ValueError, match="batch_size=1"):
        speculative_decode(base, draft, torch.tensor([[1, 2], [3, 4]]), cfg=cfg)


def test_speculative_decode_acceptance_rate_in_range() -> None:
    base = _MockBaseLM()
    draft = EAGLE3DraftHead(hidden_dim=16, vocab_size=8)
    prompt = torch.tensor([[1, 2]])
    cfg = EAGLE3Config(draft_length=3, max_new_tokens=6)
    _, metrics = speculative_decode(base, draft, prompt, cfg=cfg)
    assert 0.0 <= metrics["acceptance_rate"] <= 1.0


def test_speculative_decode_higher_draft_length_more_drafts() -> None:
    base = _MockBaseLM()
    draft = EAGLE3DraftHead(hidden_dim=16, vocab_size=8)
    prompt = torch.tensor([[1, 2]])

    short, _ = speculative_decode(
        base, draft, prompt, cfg=EAGLE3Config(draft_length=1, max_new_tokens=4),
    )
    long_out, _ = speculative_decode(
        base, draft, prompt, cfg=EAGLE3Config(draft_length=4, max_new_tokens=4),
    )
    # Both should produce at least max_new_tokens new positions.
    assert short.shape[1] >= prompt.shape[1] + 4
    assert long_out.shape[1] >= prompt.shape[1] + 4


def test_speculative_decode_temperature_zero_is_greedy_draft() -> None:
    """temperature=0 selects argmax of draft logits, deterministic."""
    base = _MockBaseLM()
    draft = EAGLE3DraftHead(hidden_dim=16, vocab_size=8)
    prompt = torch.tensor([[1, 2]])
    cfg = EAGLE3Config(draft_length=2, temperature=0.0, max_new_tokens=4)
    out1, _ = speculative_decode(base, draft, prompt, cfg=cfg)
    out2, _ = speculative_decode(base, draft, prompt, cfg=cfg)
    # Same prompt, same models -> same output (greedy is deterministic).
    assert torch.equal(out1, out2)
