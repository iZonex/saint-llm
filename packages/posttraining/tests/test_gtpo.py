"""Tests for GTPO entropy-weighted token policy optimization (RL-06)."""

from __future__ import annotations

import math

import pytest
import torch
from saint_llm_posttraining import (
    GRPOConfig,
    RolloutBatch,
    grpo_loss,
)
from saint_llm_posttraining.gtpo import (
    GTPOConfig,
    entropy_weight,
    gtpo_loss,
    per_token_entropy,
)


def _flat_batch(
    rewards: list[float],
    *,
    seq_len: int = 6,
    vocab: int = 16,
) -> tuple[RolloutBatch, torch.Tensor]:
    n = len(rewards)
    tokens = torch.randint(0, vocab, (n, seq_len))
    response_mask = torch.zeros(n, seq_len, dtype=torch.long)
    response_mask[:, seq_len // 2 :] = 1
    old_lp = torch.zeros(n, seq_len)
    ref_lp = torch.zeros(n, seq_len)
    batch = RolloutBatch(
        tokens=tokens,
        response_mask=response_mask,
        old_logprobs=old_lp,
        ref_logprobs=ref_lp,
        rewards=torch.tensor(rewards, dtype=torch.float32),
    )
    logits = torch.randn(n, seq_len, vocab)
    return batch, logits


def test_per_token_entropy_uniform_distribution_is_log_v() -> None:
    """Uniform logits over V tokens -> entropy ~ log(V)."""
    logits = torch.zeros(1, 4, 8)  # uniform softmax
    h = per_token_entropy(logits)
    expected = math.log(8)
    assert torch.allclose(
        h, torch.full_like(h, expected), atol=1e-5,
    )


def test_per_token_entropy_one_hot_is_zero() -> None:
    """One token has all probability mass -> entropy 0."""
    logits = torch.full((1, 1, 8), -1e6)
    logits[0, 0, 3] = 100.0
    h = per_token_entropy(logits)
    assert h.item() < 1e-3


def test_per_token_entropy_is_detached() -> None:
    """Entropy weights flow through autograd as constants, not derivatives."""
    logits = torch.randn(1, 3, 8, requires_grad=True)
    h = per_token_entropy(logits)
    assert not h.requires_grad


def test_per_token_entropy_rejects_non_3d() -> None:
    with pytest.raises(ValueError, match=r"\(B, T, V\)"):
        per_token_entropy(torch.randn(1, 8))


def test_entropy_weight_alpha_zero_is_constant_one_on_response() -> None:
    """alpha=0 -> every response position weight = 1; non-response = 0."""
    entropy = torch.rand(2, 5)
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
    w = entropy_weight(entropy, mask, alpha=0.0)
    # Inside response: weight 1.0; outside: 0.0.
    assert w[0, :2].sum().item() == 0.0
    assert torch.allclose(w[0, 2:], torch.ones(3))
    assert w[1, 0].item() == 0.0
    assert torch.allclose(w[1, 1:3], torch.ones(2))
    assert w[1, 3:].sum().item() == 0.0


def test_entropy_weight_alpha_one_zeroes_min_entropy_position() -> None:
    """alpha=1 + a position with zero entropy -> weight zero there."""
    entropy = torch.tensor([[0.0, 1.0, 2.0]])  # max=2
    mask = torch.ones(1, 3, dtype=torch.long)
    w = entropy_weight(entropy, mask, alpha=1.0, normalize_per_row=True)
    assert w[0, 0].item() == 0.0
    assert w[0, 1].item() == 0.5
    assert w[0, 2].item() == 1.0


def test_entropy_weight_normalize_per_row() -> None:
    """Each row's max response-entropy normalizes its weights to [0, 1]."""
    entropy = torch.tensor([[0.0, 4.0, 8.0], [0.0, 1.0, 2.0]])
    mask = torch.ones(2, 3, dtype=torch.long)
    w = entropy_weight(entropy, mask, alpha=1.0, normalize_per_row=True)
    # Row 0 max = 8, row 1 max = 2. After w_t = ent / max:
    # row 0 -> [0, 0.5, 1]; row 1 -> [0, 0.5, 1].
    assert torch.allclose(w[0], torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(w[1], torch.tensor([0.0, 0.5, 1.0]))


def test_entropy_weight_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError, match="alpha"):
        entropy_weight(
            torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.long), alpha=1.5,
        )


def test_entropy_weight_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="share shape"):
        entropy_weight(
            torch.zeros(1, 3), torch.ones(1, 4, dtype=torch.long), alpha=0.5,
        )


def test_gtpo_loss_alpha_zero_matches_grpo_loss() -> None:
    """At alpha=0 GTPO should produce the exact same loss as GRPO."""
    torch.manual_seed(0)
    batch, logits = _flat_batch(rewards=[1.0, 2.0, 3.0, 4.0])
    grpo_cfg = GRPOConfig(group_size=2, kl_coef=0.0)
    gtpo_cfg = GTPOConfig(grpo=grpo_cfg, entropy_alpha=0.0)
    l_grpo, _ = grpo_loss(logits, batch, cfg=grpo_cfg)
    l_gtpo, _ = gtpo_loss(logits, batch, cfg=gtpo_cfg)
    assert torch.allclose(l_grpo, l_gtpo, atol=1e-5)


def test_gtpo_loss_finite_with_default_alpha() -> None:
    torch.manual_seed(0)
    batch, logits = _flat_batch(rewards=[0.5, 1.0, 1.5, 2.0])
    cfg = GTPOConfig(grpo=GRPOConfig(group_size=2), entropy_alpha=0.5)
    loss, metrics = gtpo_loss(logits, batch, cfg=cfg)
    assert torch.isfinite(loss)
    assert "mean_entropy_weight" in metrics
    assert "mean_entropy_nats" in metrics


def test_gtpo_loss_grad_flows_through_logits() -> None:
    torch.manual_seed(0)
    batch, logits = _flat_batch(rewards=[0.5, 1.0])
    logits.requires_grad_(True)
    cfg = GTPOConfig(grpo=GRPOConfig(group_size=2), entropy_alpha=0.5)
    loss, _ = gtpo_loss(logits, batch, cfg=cfg)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_gtpo_loss_alpha_one_dampens_forced_tokens() -> None:
    """Pure-entropy weighting should reduce loss magnitude vs alpha=0."""
    torch.manual_seed(0)
    batch, logits = _flat_batch(rewards=[0.0, 1.0, 0.0, 1.0])
    cfg_grpo = GTPOConfig(grpo=GRPOConfig(group_size=2, kl_coef=0.0), entropy_alpha=0.0)
    cfg_full = GTPOConfig(grpo=GRPOConfig(group_size=2, kl_coef=0.0), entropy_alpha=1.0)
    loss_grpo, _ = gtpo_loss(logits, batch, cfg=cfg_grpo)
    loss_full, _ = gtpo_loss(logits, batch, cfg=cfg_full)
    # With alpha=1 some tokens contribute zero -> the loss magnitude
    # should be smaller (or equal in pathological cases) than alpha=0.
    assert abs(loss_full.item()) <= abs(loss_grpo.item()) + 1e-6


def test_gtpo_config_rejects_alpha_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="entropy_alpha"):
        GTPOConfig(entropy_alpha=-0.1)
    with pytest.raises(ValueError, match="entropy_alpha"):
        GTPOConfig(entropy_alpha=1.5)


def test_gtpo_loss_metrics_include_mean_entropy_weight() -> None:
    torch.manual_seed(0)
    batch, logits = _flat_batch(rewards=[1.0, 2.0])
    cfg = GTPOConfig(grpo=GRPOConfig(group_size=2), entropy_alpha=0.5)
    _, metrics = gtpo_loss(logits, batch, cfg=cfg)
    weight_val = metrics["mean_entropy_weight"].item()
    # alpha=0.5: weights are at least 0.5 (constant base) and at most 1.0.
    assert 0.5 - 1e-6 <= weight_val <= 1.0 + 1e-6
