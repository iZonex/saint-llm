"""Tests for GRPO scaffolding.

Hand-checked math: advantage normalization, surrogate clipping at the
boundary, the unbiased KL estimator, gradient masking. Synthetic
``logits`` are small enough (V=4..8) that we can compute references with
``F.log_softmax`` directly inside the test.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from saint_llm_posttraining import (
    GRPOConfig,
    RolloutBatch,
    compute_group_advantage,
    gather_token_logprobs,
    grpo_loss,
)


def test_compute_group_advantage_zero_mean_unit_std() -> None:
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
    adv = compute_group_advantage(rewards, group_size=4)
    g0, g1 = adv[:4], adv[4:]
    torch.testing.assert_close(g0.mean(), torch.tensor(0.0), atol=1e-5, rtol=0)
    torch.testing.assert_close(g1.mean(), torch.tensor(0.0), atol=1e-5, rtol=0)
    # Biased std (divide by n) → after normalization, unbiased=False std == 1.
    torch.testing.assert_close(g0.std(unbiased=False), torch.tensor(1.0), atol=1e-5, rtol=0)
    torch.testing.assert_close(g1.std(unbiased=False), torch.tensor(1.0), atol=1e-5, rtol=0)


def test_compute_group_advantage_handles_zero_variance_group() -> None:
    """All-equal rewards in a group → ~0 advantage, no NaN."""
    rewards = torch.tensor([5.0, 5.0, 5.0, 5.0])
    adv = compute_group_advantage(rewards, group_size=4)
    assert torch.isfinite(adv).all()
    # mean=5, std=0 → (r - mean) / eps = 0 / eps = 0.
    torch.testing.assert_close(adv, torch.zeros(4), atol=1e-5, rtol=0)


def test_compute_group_advantage_rejects_misaligned_size() -> None:
    rewards = torch.zeros(5)
    with pytest.raises(ValueError, match="not divisible"):
        compute_group_advantage(rewards, group_size=4)


def test_compute_group_advantage_rejects_nonpositive_group_size() -> None:
    rewards = torch.zeros(4)
    with pytest.raises(ValueError, match="group_size must be positive"):
        compute_group_advantage(rewards, group_size=0)


def test_gather_token_logprobs_matches_log_softmax_gather() -> None:
    torch.manual_seed(0)
    B, T, V = 2, 5, 8
    logits = torch.randn(B, T, V)
    tokens = torch.randint(0, V, (B, T))

    out = gather_token_logprobs(logits, tokens)
    assert out.shape == (B, T)
    # Position 0 is always zero by convention.
    assert (out[:, 0] == 0).all()
    # Positions 1..T-1 should equal log p(tokens[t] | prefix < t).
    log_probs = F.log_softmax(logits, dim=-1)
    expected = log_probs[:, :-1].gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(out[:, 1:], expected)


def test_gather_token_logprobs_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="logits must be"):
        gather_token_logprobs(torch.zeros(3, 4), torch.zeros(3, 4, dtype=torch.long))
    with pytest.raises(ValueError, match="must match"):
        gather_token_logprobs(torch.zeros(2, 5, 8), torch.zeros(3, 5, dtype=torch.long))


def _make_rollout(
    *,
    bg: int = 4,
    seq_len: int = 6,
    response_start: int = 3,
    vocab: int = 8,
) -> tuple[torch.Tensor, RolloutBatch]:
    """Synthetic rollout: bg=B*G rows, response is the trailing chunk.

    Returns ``(logits, batch)`` where logits are random and batch fills
    the convention (response_mask, old/ref logprobs at the response
    positions, scalar reward per row).
    """
    torch.manual_seed(7)
    logits = torch.randn(bg, seq_len, vocab, requires_grad=True)
    tokens = torch.randint(0, vocab, (bg, seq_len))
    mask = torch.zeros(bg, seq_len, dtype=torch.long)
    mask[:, response_start:] = 1

    # Old / ref logprobs: derive from another random logits tensor so the
    # ratio is meaningful. Mask non-response positions to 0.
    old_logits = torch.randn(bg, seq_len, vocab)
    ref_logits = torch.randn(bg, seq_len, vocab)
    old_logprobs = gather_token_logprobs(old_logits, tokens).detach()
    ref_logprobs = gather_token_logprobs(ref_logits, tokens).detach()

    rewards = torch.tensor([0.1, 0.5, 0.9, 0.2])[:bg].clone()

    batch = RolloutBatch(
        tokens=tokens,
        response_mask=mask,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
        rewards=rewards,
    )
    return logits, batch


def test_grpo_loss_returns_scalar_and_finite_metrics() -> None:
    logits, batch = _make_rollout(bg=4, seq_len=6, response_start=3, vocab=8)
    cfg = GRPOConfig(group_size=4, clip_eps=0.2, kl_coef=0.04)
    loss, metrics = grpo_loss(logits, batch, cfg=cfg)

    assert loss.dim() == 0
    assert torch.isfinite(loss).item()
    for k, v in metrics.items():
        assert torch.isfinite(v).item(), f"metric {k} is not finite"


def test_grpo_loss_zero_when_policies_equal_and_advantage_zero() -> None:
    """If new == old (ratio=1) and advantages all zero, pg term is zero;
    if also new == ref, KL term is zero, so total loss is zero.
    """
    bg, seq_len, vocab = 4, 6, 8
    torch.manual_seed(0)
    logits = torch.randn(bg, seq_len, vocab, requires_grad=True)
    tokens = torch.randint(0, vocab, (bg, seq_len))
    mask = torch.zeros(bg, seq_len, dtype=torch.long)
    mask[:, 3:] = 1

    # Force ratio==1 by using new logprobs as old logprobs.
    new_logprobs = gather_token_logprobs(logits, tokens).detach()
    # Force KL==0 by using new as ref.
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])  # all equal → advantage = 0

    batch = RolloutBatch(
        tokens=tokens,
        response_mask=mask,
        old_logprobs=new_logprobs,
        ref_logprobs=new_logprobs,
        rewards=rewards,
    )
    cfg = GRPOConfig(group_size=4)
    loss, metrics = grpo_loss(logits, batch, cfg=cfg)

    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(metrics["pg_loss"], torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(metrics["kl_penalty"], torch.tensor(0.0), atol=1e-6, rtol=0)


def test_grpo_loss_clip_engages_when_ratio_exceeds_bound() -> None:
    """Force a large ratio (very different old vs new) and verify clip_frac > 0."""
    bg, seq_len, vocab = 4, 6, 8
    torch.manual_seed(1)
    logits = torch.randn(bg, seq_len, vocab, requires_grad=True)
    tokens = torch.randint(0, vocab, (bg, seq_len))
    mask = torch.zeros(bg, seq_len, dtype=torch.long)
    mask[:, 3:] = 1

    # Very different old policy → ratios scattered far from 1.
    bad_old_logits = torch.randn(bg, seq_len, vocab) * 5.0
    old_logprobs = gather_token_logprobs(bad_old_logits, tokens).detach()
    new_logprobs = gather_token_logprobs(logits, tokens).detach()

    batch = RolloutBatch(
        tokens=tokens,
        response_mask=mask,
        old_logprobs=old_logprobs,
        ref_logprobs=new_logprobs,
        rewards=torch.tensor([0.0, 1.0, 0.0, 1.0]),
    )
    cfg = GRPOConfig(group_size=4, clip_eps=0.05)  # tight clip
    _, metrics = grpo_loss(logits, batch, cfg=cfg)
    assert metrics["clip_frac"].item() > 0.0


def test_grpo_loss_kl_penalty_is_nonnegative() -> None:
    """The Schulman unbiased KL estimator is always ≥ 0 — verify."""
    logits, batch = _make_rollout(bg=4, seq_len=6, response_start=3, vocab=8)
    cfg = GRPOConfig(group_size=4)
    _, metrics = grpo_loss(logits, batch, cfg=cfg)
    assert metrics["kl_penalty"].item() >= 0.0


def test_grpo_loss_grad_flows_only_through_response_positions() -> None:
    """Grad on logits should be zero on prompt positions (mask=0) and
    nonzero somewhere on response positions (mask=1).

    The loss depends on logits via two paths: the new_logprobs ratio AND
    the KL penalty. Both are masked by response_mask, so prompt-position
    logits should receive no gradient.
    """
    bg, seq_len, vocab = 4, 6, 8
    torch.manual_seed(2)
    logits = torch.randn(bg, seq_len, vocab, requires_grad=True)
    tokens = torch.randint(0, vocab, (bg, seq_len))
    response_start = 3
    mask = torch.zeros(bg, seq_len, dtype=torch.long)
    mask[:, response_start:] = 1

    old_logprobs = gather_token_logprobs(torch.randn(bg, seq_len, vocab), tokens).detach()
    ref_logprobs = gather_token_logprobs(torch.randn(bg, seq_len, vocab), tokens).detach()

    batch = RolloutBatch(
        tokens=tokens,
        response_mask=mask,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
        rewards=torch.tensor([0.1, 0.5, 0.9, 0.2]),
    )
    cfg = GRPOConfig(group_size=4)
    loss, _ = grpo_loss(logits, batch, cfg=cfg)
    loss.backward()
    assert logits.grad is not None

    # gather_token_logprobs at output index t reads logits[:, t-1, :].
    # response_mask = 1 at t in [response_start, seq_len) → gradient lives
    # on logits indices [response_start - 1, seq_len - 1).
    grad_norm = logits.grad.norm(dim=-1)  # (bg, seq_len)
    # Indices [0 .. response_start - 2] never feed any unmasked position.
    if response_start - 1 > 0:
        assert torch.allclose(grad_norm[:, : response_start - 1], torch.zeros(bg, response_start - 1))
    # Response-feeding indices are [response_start - 1, seq_len - 1]; grad must be nonzero somewhere.
    assert (grad_norm[:, response_start - 1 : seq_len - 1] > 0).any()


def test_grpo_loss_constant_advantage_reduces_to_negative_advantage_minus_kl() -> None:
    """Sanity check: when ratio==1 everywhere, the surrogate objective is
    exactly ``A * 1 == A``, so pg_loss per completion = -A_i. With KL=0 and
    advantage normalized in-group, pg_loss.mean() should equal -mean(A) = 0.
    """
    bg, seq_len, vocab = 4, 6, 8
    torch.manual_seed(3)
    logits = torch.randn(bg, seq_len, vocab, requires_grad=True)
    tokens = torch.randint(0, vocab, (bg, seq_len))
    mask = torch.zeros(bg, seq_len, dtype=torch.long)
    mask[:, 3:] = 1
    new_logprobs = gather_token_logprobs(logits, tokens).detach()

    batch = RolloutBatch(
        tokens=tokens,
        response_mask=mask,
        old_logprobs=new_logprobs,
        ref_logprobs=new_logprobs,
        rewards=torch.tensor([0.0, 1.0, 2.0, 3.0]),
    )
    cfg = GRPOConfig(group_size=4, kl_coef=0.0)
    loss, _ = grpo_loss(logits, batch, cfg=cfg)
    # Group-normalized advantage has mean 0 across the 4 rows;
    # pg_loss per row = -A_i; mean across rows = 0.
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-5, rtol=0)
