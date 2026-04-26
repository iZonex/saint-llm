"""Tests for the Tree-GRPO trainer driver."""

from __future__ import annotations

import pytest
import torch
from saint_llm_posttraining import GRPOConfig
from saint_llm_posttraining.tree_grpo_trainer import (
    TreeGRPOTrainerStep,
    build_tree_rollout_batch,
    tree_dynamic_sampling_mask,
    tree_grpo_train_step,
)
from torch import nn


class _MockLM(nn.Module):
    def __init__(self, vocab: int = 16, hidden: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.head(self.embed(tokens))}


def test_build_tree_rollout_batch_attaches_parent_idx() -> None:
    actor = _MockLM()
    tokens = torch.randint(0, 16, (4, 6))
    response_mask = torch.zeros(4, 6, dtype=torch.long)
    response_mask[:, 3:] = 1
    rewards = torch.tensor([0.5, 0.7, 0.1, 0.9])
    parent_idx = torch.tensor([0, 0, 1, 1])
    batch = build_tree_rollout_batch(
        actor=actor, ref=None, tokens=tokens,
        response_mask=response_mask, rewards=rewards, parent_idx=parent_idx,
    )
    assert torch.equal(batch.parent_idx, parent_idx)
    assert batch.tokens.shape == (4, 6)
    # ref_logprobs == old_logprobs when ref is None.
    assert torch.equal(batch.ref_logprobs, batch.old_logprobs)


def test_build_tree_rollout_batch_with_ref_uses_ref_logprobs() -> None:
    actor = _MockLM()
    ref = _MockLM()
    tokens = torch.randint(0, 16, (2, 4))
    response_mask = torch.ones(2, 4, dtype=torch.long)
    rewards = torch.tensor([0.0, 1.0])
    parent_idx = torch.tensor([0, 0])
    batch = build_tree_rollout_batch(
        actor=actor, ref=ref, tokens=tokens,
        response_mask=response_mask, rewards=rewards, parent_idx=parent_idx,
    )
    # Different models -> different logprobs.
    assert not torch.equal(batch.ref_logprobs, batch.old_logprobs)


def test_tree_dynamic_sampling_mask_drops_zero_variance_groups() -> None:
    rewards = torch.tensor([1.0, 1.0, 0.5, 0.7])
    parent_idx = torch.tensor([0, 0, 1, 1])
    keep = tree_dynamic_sampling_mask(rewards, parent_idx)
    # Group 0 has zero variance -> dropped.
    assert keep[:2].tolist() == [False, False]
    # Group 1 has nonzero variance -> kept.
    assert keep[2:].tolist() == [True, True]


def test_tree_dynamic_sampling_mask_keeps_singleton_groups() -> None:
    """Singletons have no variance but no harmful effect either."""
    rewards = torch.tensor([5.0, 1.0, 2.0])
    parent_idx = torch.tensor([0, 1, 1])
    keep = tree_dynamic_sampling_mask(rewards, parent_idx)
    assert keep.tolist() == [True, True, True]


def test_tree_dynamic_sampling_mask_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="matching shape"):
        tree_dynamic_sampling_mask(
            torch.tensor([1.0, 2.0]), torch.tensor([0, 0, 1]),
        )


def test_tree_grpo_train_step_returns_step_dataclass_and_runs() -> None:
    torch.manual_seed(0)
    actor = _MockLM()
    optimizer = torch.optim.SGD(actor.parameters(), lr=1e-2)
    tokens = torch.randint(0, 16, (4, 6))
    response_mask = torch.zeros(4, 6, dtype=torch.long)
    response_mask[:, 3:] = 1
    rewards = torch.tensor([0.1, 0.2, 0.5, 0.9])
    parent_idx = torch.tensor([0, 0, 1, 1])
    batch = build_tree_rollout_batch(
        actor=actor, ref=None, tokens=tokens,
        response_mask=response_mask, rewards=rewards, parent_idx=parent_idx,
    )
    step = tree_grpo_train_step(
        actor=actor, optimizer=optimizer, batch=batch,
        cfg=GRPOConfig(group_size=2, kl_coef=0.0),
    )
    assert isinstance(step, TreeGRPOTrainerStep)
    assert torch.isfinite(step.loss)
    assert step.n_kept_groups == 2


def test_tree_grpo_train_step_filters_zero_variance_groups_before_loss() -> None:
    """Zero-variance sibling group dropped; surviving groups still trained."""
    torch.manual_seed(0)
    actor = _MockLM()
    optimizer = torch.optim.SGD(actor.parameters(), lr=1e-2)
    tokens = torch.randint(0, 16, (4, 6))
    response_mask = torch.ones(4, 6, dtype=torch.long)
    rewards = torch.tensor([1.0, 1.0, 0.2, 0.8])  # group 0 zero-variance
    parent_idx = torch.tensor([0, 0, 1, 1])
    batch = build_tree_rollout_batch(
        actor=actor, ref=None, tokens=tokens,
        response_mask=response_mask, rewards=rewards, parent_idx=parent_idx,
    )
    step = tree_grpo_train_step(
        actor=actor, optimizer=optimizer, batch=batch,
        cfg=GRPOConfig(group_size=2, kl_coef=0.0),
    )
    # Only one group survives.
    assert step.n_kept_groups == 1


def test_tree_grpo_train_step_returns_zero_loss_when_all_groups_dropped() -> None:
    """Pathological batch: all groups zero-variance -> zero loss returned."""
    torch.manual_seed(0)
    actor = _MockLM()
    optimizer = torch.optim.SGD(actor.parameters(), lr=1e-2)
    tokens = torch.randint(0, 16, (2, 4))
    response_mask = torch.ones(2, 4, dtype=torch.long)
    rewards = torch.tensor([0.5, 0.5])
    parent_idx = torch.tensor([0, 0])
    batch = build_tree_rollout_batch(
        actor=actor, ref=None, tokens=tokens,
        response_mask=response_mask, rewards=rewards, parent_idx=parent_idx,
    )
    step = tree_grpo_train_step(
        actor=actor, optimizer=optimizer, batch=batch,
        cfg=GRPOConfig(group_size=2, kl_coef=0.0),
    )
    assert step.n_kept_groups == 0
    assert step.loss.item() == 0.0


def test_tree_grpo_train_step_actually_updates_actor_params() -> None:
    """After one step, actor params should differ from initial state."""
    torch.manual_seed(0)
    actor = _MockLM()
    optimizer = torch.optim.SGD(actor.parameters(), lr=1.0)
    initial_head = actor.head.weight.detach().clone()
    tokens = torch.randint(0, 16, (4, 6))
    response_mask = torch.ones(4, 6, dtype=torch.long)
    rewards = torch.tensor([0.1, 0.9, 0.3, 0.7])
    parent_idx = torch.tensor([0, 0, 1, 1])
    batch = build_tree_rollout_batch(
        actor=actor, ref=None, tokens=tokens,
        response_mask=response_mask, rewards=rewards, parent_idx=parent_idx,
    )
    tree_grpo_train_step(
        actor=actor, optimizer=optimizer, batch=batch,
        cfg=GRPOConfig(group_size=2, kl_coef=0.0),
    )
    assert not torch.equal(actor.head.weight, initial_head)


def test_tree_grpo_train_step_with_ref_includes_kl_term() -> None:
    """When ref is set + kl_coef>0, loss should reflect KL."""
    torch.manual_seed(0)
    actor = _MockLM()
    ref = _MockLM()
    optimizer = torch.optim.SGD(actor.parameters(), lr=1e-3)
    tokens = torch.randint(0, 16, (2, 4))
    response_mask = torch.ones(2, 4, dtype=torch.long)
    rewards = torch.tensor([0.0, 1.0])
    parent_idx = torch.tensor([0, 0])
    batch = build_tree_rollout_batch(
        actor=actor, ref=ref, tokens=tokens,
        response_mask=response_mask, rewards=rewards, parent_idx=parent_idx,
    )
    step = tree_grpo_train_step(
        actor=actor, optimizer=optimizer, batch=batch,
        cfg=GRPOConfig(group_size=2, kl_coef=0.5),
    )
    assert "kl_penalty" in step.metrics
    assert torch.isfinite(step.loss)
