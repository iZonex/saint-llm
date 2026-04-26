"""Tests for Tree-GRPO (RL-05) — sibling-group advantages + tree surrogate."""

from __future__ import annotations

import pytest
import torch
from saint_llm_posttraining import (
    GRPOConfig,
    RolloutBatch,
    compute_group_advantage,
    grpo_loss,
)
from saint_llm_posttraining.tree_grpo import (
    TreeRolloutBatch,
    compute_tree_advantage,
    tree_grpo_loss,
)


def _tree_batch(
    rewards: list[float],
    parent_idx: list[int],
    *,
    seq_len: int = 4,
    vocab: int = 16,
) -> tuple[TreeRolloutBatch, torch.Tensor]:
    """Build a small TreeRolloutBatch + matching logits tensor."""
    n = len(rewards)
    tokens = torch.randint(0, vocab, (n, seq_len))
    response_mask = torch.zeros(n, seq_len, dtype=torch.long)
    response_mask[:, seq_len // 2 :] = 1  # second half is response
    old_lp = torch.zeros(n, seq_len)
    ref_lp = torch.zeros(n, seq_len)
    batch = TreeRolloutBatch(
        tokens=tokens,
        response_mask=response_mask,
        old_logprobs=old_lp,
        ref_logprobs=ref_lp,
        rewards=torch.tensor(rewards, dtype=torch.float32),
        parent_idx=torch.tensor(parent_idx, dtype=torch.long),
    )
    logits = torch.randn(n, seq_len, vocab)
    return batch, logits


def test_compute_tree_advantage_normalizes_within_sibling_group() -> None:
    """Two children sharing a parent get (r - mean) / (std + eps)."""
    rewards = torch.tensor([2.0, 4.0])
    parent = torch.tensor([0, 0])
    adv = compute_tree_advantage(rewards, parent)
    # Mean=3, std=1 -> [-1, +1].
    assert torch.allclose(adv, torch.tensor([-1.0, 1.0]), atol=1e-5)


def test_compute_tree_advantage_singleton_group_is_zero() -> None:
    """A parent with one child has no relative signal -> advantage 0."""
    rewards = torch.tensor([5.0])
    parent = torch.tensor([0])
    adv = compute_tree_advantage(rewards, parent)
    assert adv.item() == 0.0


def test_compute_tree_advantage_pools_roots_together() -> None:
    """All parent_idx == -1 -> roots form one synthetic group."""
    rewards = torch.tensor([1.0, 3.0, 5.0])
    parent = torch.tensor([-1, -1, -1])
    adv = compute_tree_advantage(rewards, parent)
    # Mean=3, std≈sqrt(8/3)
    expected_mean = adv.mean().item()
    assert abs(expected_mean) < 1e-5
    # The relative ordering is preserved.
    assert adv[0] < adv[1] < adv[2]


def test_compute_tree_advantage_unbiased_skips_std() -> None:
    rewards = torch.tensor([2.0, 4.0])
    parent = torch.tensor([0, 0])
    adv = compute_tree_advantage(rewards, parent, unbiased=True)
    assert torch.allclose(adv, torch.tensor([-1.0, 1.0]))


def test_compute_tree_advantage_separate_groups_dont_cross_normalize() -> None:
    """Two parents -> two groups; advantages computed independently."""
    rewards = torch.tensor([1.0, 2.0, 100.0, 200.0])
    parent = torch.tensor([0, 0, 1, 1])
    adv = compute_tree_advantage(rewards, parent)
    # Each pair normalizes to [-1, +1].
    assert torch.allclose(adv[:2], torch.tensor([-1.0, 1.0]), atol=1e-5)
    assert torch.allclose(adv[2:], torch.tensor([-1.0, 1.0]), atol=1e-5)


def test_compute_tree_advantage_matches_flat_grpo_when_all_share_parent() -> None:
    """Flat GRPO with group_size=N matches Tree-GRPO with one shared parent."""
    rewards = torch.tensor([1.0, 3.0, 5.0, 7.0])
    parent = torch.tensor([0, 0, 0, 0])
    tree_adv = compute_tree_advantage(rewards, parent)
    flat_adv = compute_group_advantage(rewards, group_size=4)
    assert torch.allclose(tree_adv, flat_adv, atol=1e-5)


def test_compute_tree_advantage_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="must be 1-D"):
        compute_tree_advantage(
            torch.tensor([[1.0, 2.0]]), torch.tensor([0, 0]),
        )


def test_compute_tree_advantage_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="matching length"):
        compute_tree_advantage(
            torch.tensor([1.0, 2.0]), torch.tensor([0, 0, 0]),
        )


def test_tree_grpo_loss_finite_and_returns_metrics() -> None:
    torch.manual_seed(0)
    batch, logits = _tree_batch(
        rewards=[1.0, 2.0, 3.0, 4.0],
        parent_idx=[0, 0, 1, 1],
    )
    cfg = GRPOConfig(group_size=2)
    loss, metrics = tree_grpo_loss(logits, batch, cfg=cfg)
    assert torch.isfinite(loss)
    assert "pg_loss" in metrics
    assert "kl_penalty" in metrics
    assert "n_sibling_groups" in metrics
    assert int(metrics["n_sibling_groups"].item()) == 2


def test_tree_grpo_loss_grad_flows() -> None:
    torch.manual_seed(0)
    batch, logits = _tree_batch(
        rewards=[1.0, 2.0],
        parent_idx=[0, 0],
    )
    logits.requires_grad_(True)
    cfg = GRPOConfig(group_size=2)
    loss, _ = tree_grpo_loss(logits, batch, cfg=cfg)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_tree_grpo_loss_matches_flat_grpo_when_uniform_groups() -> None:
    """When every parent has the same sibling count, tree-GRPO = flat GRPO.

    Build a flat RolloutBatch and a tree batch with parent_idx
    grouped identically; loss values must match.
    """
    torch.manual_seed(42)
    n = 4
    seq = 6
    vocab = 16
    tokens = torch.randint(0, vocab, (n, seq))
    response_mask = torch.zeros(n, seq, dtype=torch.long)
    response_mask[:, 3:] = 1
    old_lp = torch.zeros(n, seq)
    ref_lp = torch.zeros(n, seq)
    rewards = torch.tensor([1.0, 3.0, 2.0, 4.0])
    logits = torch.randn(n, seq, vocab)

    flat = RolloutBatch(
        tokens=tokens,
        response_mask=response_mask,
        old_logprobs=old_lp,
        ref_logprobs=ref_lp,
        rewards=rewards,
    )
    tree = TreeRolloutBatch(
        tokens=tokens,
        response_mask=response_mask,
        old_logprobs=old_lp,
        ref_logprobs=ref_lp,
        rewards=rewards,
        # First two share parent 0, second two share parent 1.
        parent_idx=torch.tensor([0, 0, 1, 1]),
    )
    cfg = GRPOConfig(group_size=2, kl_coef=0.0)
    loss_flat, _ = grpo_loss(logits, flat, cfg=cfg)
    loss_tree, _ = tree_grpo_loss(logits, tree, cfg=cfg)
    assert torch.allclose(loss_flat, loss_tree, atol=1e-5)


def test_tree_grpo_loss_singleton_groups_contribute_zero_advantage() -> None:
    """Singleton siblings: advantage zero -> the singleton's PG term vanishes."""
    torch.manual_seed(0)
    batch, logits = _tree_batch(
        rewards=[10.0, 5.0, 5.0],
        parent_idx=[0, 1, 1],  # rollout 0 is alone; 1 and 2 share parent 1
    )
    cfg = GRPOConfig(group_size=2, kl_coef=0.0)
    advantages = compute_tree_advantage(batch.rewards, batch.parent_idx)
    assert advantages[0].item() == 0.0
    # Sibling pair (5, 5) has zero variance -> their advantage is also 0.
    assert torch.allclose(advantages[1:], torch.zeros(2), atol=1e-5)
    loss, _ = tree_grpo_loss(logits, batch, cfg=cfg)
    # All advantages zero -> PG objective is zero.
    assert abs(loss.item()) < 1e-5
