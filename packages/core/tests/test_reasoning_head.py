"""Tests for adaptive thinking effort head (RL-07 / ADR-0017)."""

from __future__ import annotations

import pytest
import torch
from saint_llm_core.multimodal.reasoning_head import (
    EFFORT_TIER_NAMES,
    EffortConfig,
    EffortRouter,
    effort_id_to_token,
    effort_tier_to_id,
)


def test_effort_config_defaults() -> None:
    cfg = EffortConfig()
    assert cfg.n_tiers == 5
    assert cfg.hidden == 64
    assert cfg.enabled is False


def test_effort_router_zero_init_produces_zero_outputs() -> None:
    """Per ADR-0017, zero-init means budget=0 and terminate_logit=0
    until RL training fills the head."""
    cfg = EffortConfig()
    head = EffortRouter(hidden_dim=128, cfg=cfg)
    pooled = torch.randn(3, 128)
    tier = torch.tensor([0, 2, 4])
    out = head(pooled, tier)
    assert out["budget"].shape == (3,)
    assert out["terminate_logit"].shape == (3,)
    torch.testing.assert_close(out["budget"], torch.zeros(3))
    torch.testing.assert_close(out["terminate_logit"], torch.zeros(3))


def test_effort_router_grad_flows_when_trained() -> None:
    """After perturbing fuse weight, gradients flow through the head."""
    cfg = EffortConfig()
    head = EffortRouter(hidden_dim=64, cfg=cfg)
    # Perturb fuse weight so the path through ReLU survives.
    with torch.no_grad():
        head.fuse.weight.fill_(0.1)
        head.fuse.bias.fill_(1.0)  # forces ReLU into linear region
        head.budget_head.weight.fill_(0.5)
    pooled = torch.randn(2, 64, requires_grad=True)
    tier = torch.tensor([1, 3])
    out = head(pooled, tier)
    out["budget"].sum().backward()
    # Gradient flows through fuse to pooled.
    assert pooled.grad is not None
    assert pooled.grad.abs().sum().item() > 0
    assert head.fuse.weight.grad is not None
    assert head.budget_head.weight.grad is not None


def test_effort_router_rejects_out_of_range_tier() -> None:
    head = EffortRouter(hidden_dim=32, cfg=EffortConfig(n_tiers=5))
    pooled = torch.zeros(1, 32)
    with pytest.raises(ValueError, match="must be in"):
        head(pooled, torch.tensor([5]))
    with pytest.raises(ValueError, match="must be in"):
        head(pooled, torch.tensor([-1]))


def test_effort_router_rejects_non_1d_tier() -> None:
    head = EffortRouter(hidden_dim=32, cfg=EffortConfig())
    with pytest.raises(ValueError, match="must be 1D"):
        head(torch.zeros(1, 32), torch.tensor([[0]]))


def test_effort_router_budget_is_non_negative() -> None:
    """Budget head goes through ReLU; outputs are guaranteed >= 0."""
    cfg = EffortConfig()
    head = EffortRouter(hidden_dim=64, cfg=cfg)
    # Force a negative bias to verify ReLU clamps it.
    with torch.no_grad():
        head.budget_head.bias.fill_(-100.0)
    pooled = torch.zeros(2, 64)
    out = head(pooled, torch.tensor([0, 4]))
    assert (out["budget"] >= 0).all().item()


def test_effort_tier_names_match_claude_4_7_convention() -> None:
    assert EFFORT_TIER_NAMES == ("low", "medium", "high", "xhigh", "max")


def test_effort_tier_to_id_roundtrip() -> None:
    assert effort_tier_to_id("low") == 0
    assert effort_tier_to_id("MEDIUM") == 1  # case-insensitive
    assert effort_tier_to_id("max") == 4


def test_effort_tier_to_id_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown effort tier"):
        effort_tier_to_id("turbo")


def test_effort_id_to_token_format() -> None:
    assert effort_id_to_token(0) == "<|effort:0|>"
    assert effort_id_to_token(4) == "<|effort:4|>"


def test_effort_id_to_token_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        effort_id_to_token(5)


def test_effort_router_after_perturbation_distinguishes_tiers() -> None:
    """When the tier embedding is non-zero, different tiers produce
    different fused vectors (and different outputs)."""
    cfg = EffortConfig()
    head = EffortRouter(hidden_dim=32, cfg=cfg)
    # Inject distinguishable tier embeddings: tier i gets row of all `i`.
    with torch.no_grad():
        for i in range(cfg.n_tiers):
            head.tier_emb.weight[i].fill_(float(i + 1))  # 1, 2, 3, 4, 5
        head.fuse.weight.fill_(1.0)
        head.fuse.bias.fill_(0.0)
        head.budget_head.weight.fill_(1.0)
        head.budget_head.bias.fill_(0.0)
    pooled = torch.zeros(5, 32)
    tier = torch.arange(5)
    out = head(pooled, tier)
    # All five outputs distinct (proportional to tier index).
    budgets = out["budget"].tolist()
    assert len(set(budgets)) == 5
    # Monotone: higher tier → larger budget.
    assert budgets == sorted(budgets)
