"""Tests for the Muon optimizer."""

from __future__ import annotations

import pytest
import torch
from saint_llm_optim import Muon


def test_step_changes_param() -> None:
    p = torch.nn.Parameter(torch.randn(8, 16))
    opt = Muon([p], lr=1.0e-2)
    p_before = p.detach().clone()
    loss = (p * 2).sum()
    loss.backward()
    opt.step()
    assert not torch.allclose(p, p_before)


def test_rejects_1d_params() -> None:
    p = torch.nn.Parameter(torch.randn(8))
    opt = Muon([p])
    p.grad = torch.ones_like(p)
    with pytest.raises(ValueError, match="2D matrix params"):
        opt.step()


def test_skips_params_without_grad() -> None:
    p = torch.nn.Parameter(torch.randn(8, 16))
    opt = Muon([p])
    p_before = p.detach().clone()
    opt.step()  # no grad set → no-op
    assert torch.equal(p, p_before)


def test_momentum_buffer_initialized_lazily() -> None:
    p = torch.nn.Parameter(torch.randn(8, 16))
    opt = Muon([p])
    assert "momentum_buffer" not in opt.state[p]
    p.grad = torch.randn_like(p)
    opt.step()
    assert "momentum_buffer" in opt.state[p]
    assert opt.state[p]["momentum_buffer"].shape == p.shape


def test_rms_rescale_magnitude() -> None:
    """After NS the orthogonalized matrix has Frobenius norm ≈ sqrt(min(n,m)).
    After RMS rescale by sqrt(max(n,m)) * γ, the entry-wise RMS should ≈ γ.
    """
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 8))
    opt = Muon([p], lr=1.0, momentum=0.0, weight_decay=0.0, rms_rescale=0.18)
    p.grad = torch.randn_like(p)
    p_before = p.detach().clone()
    opt.step()
    # update applied = -lr * O_t. With lr=1.0, change = -O_t.
    delta = (p_before - p).detach()
    rms = delta.pow(2).mean().sqrt().item()
    assert abs(rms - 0.18) < 0.05, f"Update RMS {rms} != γ=0.18"


def test_decoupled_weight_decay_shrinks_param_when_no_grad() -> None:
    p = torch.nn.Parameter(torch.ones(8, 16))
    opt = Muon([p], lr=0.1, weight_decay=0.5)
    p.grad = torch.zeros_like(p)
    opt.step()
    # After WD: p ← p * (1 - lr*wd) = p * (1 - 0.05) = 0.95
    assert torch.allclose(p, torch.full_like(p, 0.95), atol=1.0e-5)


def test_converges_on_linear_regression() -> None:
    """Train W to fit y = X @ W_target via MSE — Muon should drive loss to near-zero."""
    torch.manual_seed(0)
    n_features = 8
    W_target = torch.randn(n_features, n_features)

    W = torch.nn.Parameter(torch.randn(n_features, n_features) * 0.1)
    # Larger lr because Muon's RMS-rescaled update is small (entrywise ~lr*γ ≈ lr*0.18).
    opt = Muon([W], lr=0.05, momentum=0.9, weight_decay=0.0)

    losses: list[float] = []
    for _ in range(500):
        x = torch.randn(64, n_features)
        y = x @ W_target
        pred = x @ W
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} → {losses[-1]}"
    # 500 steps at this scale should drive loss well under 1.0 — convergence proof.
    assert losses[-1] < 1.0, f"Loss did not converge sufficiently: {losses[0]} → {losses[-1]}"


def test_invalid_hyperparameters_raise() -> None:
    p = torch.nn.Parameter(torch.randn(4, 4))
    with pytest.raises(ValueError, match="Invalid lr"):
        Muon([p], lr=-1.0)
    with pytest.raises(ValueError, match="Invalid momentum"):
        Muon([p], momentum=1.5)
    with pytest.raises(ValueError, match="Invalid weight_decay"):
        Muon([p], weight_decay=-0.1)


def test_state_persists_across_steps() -> None:
    p = torch.nn.Parameter(torch.randn(8, 16))
    opt = Muon([p], lr=1.0e-3, momentum=0.9)
    for _ in range(3):
        p.grad = torch.randn_like(p)
        opt.step()
    buf = opt.state[p]["momentum_buffer"]
    assert buf.abs().sum().item() > 0.0  # buffer accumulated something
