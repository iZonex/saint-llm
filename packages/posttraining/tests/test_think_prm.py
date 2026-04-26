"""Tests for ThinkPRM Process Reward Model (RL-10)."""

from __future__ import annotations

import pytest
import torch
from saint_llm_posttraining.think_prm import (
    StepRewardHead,
    ThinkPRMConfig,
    compute_step_rewards,
    find_step_boundaries,
    gather_step_logits,
    gather_step_scores,
    prm_filter_mask,
    step_prm_loss,
)


def _tiny_cfg() -> ThinkPRMConfig:
    return ThinkPRMConfig(hidden_dim=8, intermediate=16)


def test_step_reward_head_shape() -> None:
    head = StepRewardHead(_tiny_cfg())
    h = torch.randn(2, 5, 8)
    out = head(h)
    assert out.shape == (2, 5, 1)


def test_step_reward_head_default_intermediate() -> None:
    cfg = ThinkPRMConfig(hidden_dim=16, intermediate=0)
    head = StepRewardHead(cfg)
    assert head.fc1.out_features == 32  # 2 * hidden_dim


def test_find_step_boundaries_basic() -> None:
    tokens = torch.tensor([
        [1, 2, 99, 3, 4, 99, 5],
        [99, 1, 2, 3, 4, 5, 6],
    ])
    pos = find_step_boundaries(tokens, boundary_id=99)
    # Row 0: boundaries at 2, 5
    assert pos[0, 0].item() == 2
    assert pos[0, 1].item() == 5
    assert pos[0, 2].item() == -1  # padding
    # Row 1: boundary at 0
    assert pos[1, 0].item() == 0
    assert pos[1, 1].item() == -1


def test_find_step_boundaries_no_boundary_returns_padding() -> None:
    tokens = torch.tensor([[1, 2, 3]])
    pos = find_step_boundaries(tokens, boundary_id=99)
    assert (pos == -1).all()


def test_find_step_boundaries_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match=r"\(B, T\)"):
        find_step_boundaries(torch.tensor([1, 2, 3]), boundary_id=1)


def test_gather_step_logits_returns_per_step_with_mask() -> None:
    head = StepRewardHead(_tiny_cfg())
    hidden = torch.randn(2, 5, 8)
    positions = torch.tensor([[1, 3, -1], [0, -1, -1]])
    logits, valid = gather_step_logits(hidden, head, positions)
    assert logits.shape == (2, 3)
    assert valid.tolist() == [[True, True, False], [True, False, False]]
    # Padded positions must read zero so callers can sum safely.
    assert logits[0, 2].item() == 0.0
    assert logits[1, 1].item() == 0.0


def test_gather_step_scores_returns_probabilities_in_unit_interval() -> None:
    head = StepRewardHead(_tiny_cfg())
    hidden = torch.randn(1, 4, 8)
    positions = torch.tensor([[1, 3]])
    probs, _ = gather_step_scores(hidden, head, positions)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_gather_step_logits_rejects_non_3d_hidden() -> None:
    head = StepRewardHead(_tiny_cfg())
    with pytest.raises(ValueError, match=r"\(B, T, D\)"):
        gather_step_logits(torch.randn(4, 8), head, torch.tensor([[1]]))


def test_gather_step_logits_rejects_batch_mismatch() -> None:
    head = StepRewardHead(_tiny_cfg())
    with pytest.raises(ValueError, match="batch mismatch"):
        gather_step_logits(
            torch.randn(2, 4, 8), head, torch.tensor([[1], [2], [3]]),
        )


def test_step_prm_loss_zero_when_no_active_steps() -> None:
    logits = torch.zeros(2, 3)
    labels = torch.zeros(2, 3)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    loss = step_prm_loss(logits, labels, mask)
    assert loss.item() == 0.0


def test_step_prm_loss_decreases_with_correct_predictions() -> None:
    """Loss should be lower for predictions matching labels."""
    mask = torch.ones(1, 3, dtype=torch.bool)
    labels = torch.tensor([[1.0, 0.0, 1.0]])
    confident_correct = torch.tensor([[5.0, -5.0, 5.0]])
    confident_wrong = torch.tensor([[-5.0, 5.0, -5.0]])
    loss_correct = step_prm_loss(confident_correct, labels, mask)
    loss_wrong = step_prm_loss(confident_wrong, labels, mask)
    assert loss_correct.item() < loss_wrong.item()


def test_step_prm_loss_respects_mask() -> None:
    """Padded positions must not contribute to the loss."""
    logits = torch.tensor([[10.0, 10.0, 10.0]])
    labels = torch.tensor([[1.0, 1.0, 0.0]])  # last would penalize if unmasked
    mask = torch.tensor([[True, True, False]])
    loss = step_prm_loss(logits, labels, mask)
    # Only the first two count; both are confident-correct (label=1, logit=10)
    # so the loss should be tiny.
    assert loss.item() < 1e-3


def test_step_prm_loss_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must share shape"):
        step_prm_loss(
            torch.zeros(2, 3),
            torch.zeros(2, 4),
            torch.zeros(2, 3, dtype=torch.bool),
        )


def test_step_prm_loss_pos_weight_increases_loss_for_missed_positives() -> None:
    """pos_weight > 1 amplifies the cost of false negatives."""
    mask = torch.ones(1, 2, dtype=torch.bool)
    labels = torch.tensor([[1.0, 1.0]])
    logits = torch.tensor([[-5.0, -5.0]])  # confident wrong on positives
    loss_default = step_prm_loss(logits, labels, mask)
    loss_weighted = step_prm_loss(logits, labels, mask, pos_weight=4.0)
    assert loss_weighted.item() > loss_default.item()


def test_compute_step_rewards_end_to_end() -> None:
    head = StepRewardHead(_tiny_cfg())
    tokens = torch.tensor([[1, 2, 99, 3, 99, 4]])
    hidden = torch.randn(1, 6, 8)
    probs, mask = compute_step_rewards(hidden, head, tokens, boundary_id=99)
    assert probs.shape == (1, 6)
    assert mask[0, 0].item() and mask[0, 1].item()
    assert not mask[0, 2].item()  # padded


def test_prm_filter_mask_keeps_good_traces() -> None:
    probs = torch.tensor([[0.9, 0.7, 0.0], [0.9, 0.2, 0.0]])
    valid = torch.tensor([[True, True, False], [True, True, False]])
    keep = prm_filter_mask(probs, valid, min_score=0.5)
    assert keep.tolist() == [True, False]


def test_prm_filter_mask_keeps_traces_with_no_boundaries() -> None:
    """Trace with no scored steps -> no signal to drop on."""
    probs = torch.zeros(2, 3)
    valid = torch.zeros(2, 3, dtype=torch.bool)
    keep = prm_filter_mask(probs, valid, min_score=0.5)
    assert keep.tolist() == [True, True]


def test_prm_filter_mask_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must share shape"):
        prm_filter_mask(
            torch.zeros(2, 3), torch.zeros(2, 4, dtype=torch.bool), min_score=0.5,
        )


def test_step_reward_head_grad_flows() -> None:
    head = StepRewardHead(_tiny_cfg())
    hidden = torch.randn(1, 4, 8)
    tokens = torch.tensor([[1, 99, 2, 99]])
    probs, mask = compute_step_rewards(hidden, head, tokens, boundary_id=99)
    labels = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    # Use BCE on the gathered logits — but compute_step_rewards returns
    # probs; here we just check loss is finite + grads exist.
    logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))
    loss = step_prm_loss(logits, labels, mask)
    loss.backward()
    assert head.fc1.weight.grad is not None
    assert torch.isfinite(head.fc1.weight.grad).all()
