"""Tests for GRPO trainer driver helpers."""

from __future__ import annotations

import pytest
import torch
from saint_llm_posttraining import (
    GRPOConfig,
    RolloutBatch,
    build_rollout_batch,
    gather_token_logprobs,
    grpo_train_step,
    score_rollouts,
)
from torch import nn


class _TinyLM(nn.Module):
    """Mini LM that produces `(logits, ...)` shape — sufficient for the driver."""

    def __init__(self, vocab: int = 8, hidden: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.embed(tokens)
        return {"logits": self.head(h)}


def _synth_batch(*, bg: int = 4, t: int = 6, vocab: int = 8) -> RolloutBatch:
    torch.manual_seed(0)
    tokens = torch.randint(0, vocab, (bg, t))
    mask = torch.zeros(bg, t, dtype=torch.long)
    mask[:, 3:] = 1
    old = torch.randn(bg, t).detach()
    ref = torch.randn(bg, t).detach()
    rewards = torch.tensor([0.0, 1.0, 0.5, 0.8])[:bg].clone()
    return RolloutBatch(
        tokens=tokens, response_mask=mask,
        old_logprobs=old, ref_logprobs=ref, rewards=rewards,
    )


def test_score_rollouts_applies_reward_fn() -> None:
    out = score_rollouts(
        prompts=["a", "b", "c"],
        completions=["A", "B", "C"],
        reward_fn=lambda p, c: float(len(c)),
    )
    torch.testing.assert_close(out, torch.tensor([1.0, 1.0, 1.0]))


def test_score_rollouts_validates_lengths() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        score_rollouts(
            prompts=["a"], completions=["x", "y"],
            reward_fn=lambda p, c: 0.0,
        )


def test_build_rollout_batch_with_ref_uses_separate_logprobs() -> None:
    actor = _TinyLM(vocab=8, hidden=16)
    ref = _TinyLM(vocab=8, hidden=16)
    tokens = torch.randint(0, 8, (3, 5))
    mask = torch.ones(3, 5, dtype=torch.long)
    rewards = torch.tensor([0.0, 0.5, 1.0])
    batch = build_rollout_batch(
        actor=actor, ref=ref, tokens=tokens,
        response_mask=mask, rewards=rewards,
    )
    assert batch.tokens.shape == (3, 5)
    assert batch.old_logprobs.shape == (3, 5)
    assert batch.ref_logprobs.shape == (3, 5)
    # ref and actor are different random init; their logprobs differ.
    assert not torch.allclose(batch.old_logprobs, batch.ref_logprobs)


def test_build_rollout_batch_without_ref_zero_kl() -> None:
    actor = _TinyLM(vocab=8)
    tokens = torch.randint(0, 8, (2, 4))
    mask = torch.ones(2, 4, dtype=torch.long)
    rewards = torch.tensor([0.0, 1.0])
    batch = build_rollout_batch(
        actor=actor, ref=None, tokens=tokens,
        response_mask=mask, rewards=rewards,
    )
    # ref logprobs == old → KL = 0 in the loss path.
    torch.testing.assert_close(batch.old_logprobs, batch.ref_logprobs)


def test_grpo_train_step_runs_optimizer() -> None:
    actor = _TinyLM(vocab=8, hidden=16)
    optimizer = torch.optim.SGD(actor.parameters(), lr=1e-3)
    batch = _synth_batch(bg=4, t=6, vocab=8)
    cfg = GRPOConfig(group_size=4)

    weight_pre = actor.head.weight.detach().clone()
    out = grpo_train_step(actor=actor, optimizer=optimizer, batch=batch, cfg=cfg)
    weight_post = actor.head.weight.detach()

    assert torch.isfinite(out.loss).item()
    # Optimizer.step actually changed weights.
    assert not torch.allclose(weight_pre, weight_post)
    assert out.n_kept_groups >= 1


def test_grpo_train_step_drops_zero_variance_group() -> None:
    """All-equal-reward group is filtered by Dynamic Sampling."""
    actor = _TinyLM(vocab=8)
    optimizer = torch.optim.SGD(actor.parameters(), lr=1e-3)
    bg, t, vocab = 4, 5, 8
    tokens = torch.randint(0, vocab, (bg, t))
    mask = torch.ones(bg, t, dtype=torch.long)
    # First group of 2 has variance, second group all-equal.
    rewards = torch.tensor([0.0, 1.0, 0.5, 0.5])
    old = gather_token_logprobs(_TinyLM(vocab=vocab)(tokens)["logits"], tokens).detach()
    batch = RolloutBatch(
        tokens=tokens, response_mask=mask,
        old_logprobs=old, ref_logprobs=old, rewards=rewards,
    )
    cfg = GRPOConfig(group_size=2)
    out = grpo_train_step(actor=actor, optimizer=optimizer, batch=batch, cfg=cfg)
    assert out.n_kept_groups == 1


def test_grpo_train_step_zero_loss_when_all_groups_dropped() -> None:
    """If every group has zero variance, no update happens."""
    actor = _TinyLM(vocab=8)
    optimizer = torch.optim.SGD(actor.parameters(), lr=1e-3)
    bg, t, vocab = 2, 5, 8
    tokens = torch.randint(0, vocab, (bg, t))
    mask = torch.ones(bg, t, dtype=torch.long)
    rewards = torch.tensor([0.5, 0.5])  # one group, all-equal
    old = torch.zeros(bg, t)
    batch = RolloutBatch(
        tokens=tokens, response_mask=mask,
        old_logprobs=old, ref_logprobs=old, rewards=rewards,
    )
    cfg = GRPOConfig(group_size=2)
    weight_pre = actor.head.weight.detach().clone()
    out = grpo_train_step(actor=actor, optimizer=optimizer, batch=batch, cfg=cfg)
    weight_post = actor.head.weight.detach()
    assert out.n_kept_groups == 0
    # Weights unchanged because we didn't backprop.
    torch.testing.assert_close(weight_pre, weight_post)
