"""LR schedule shape: warmup ramp, peak at boundary, cosine decay floor."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
import torch
from saint_llm_training import warmup_cosine_schedule
from torch import nn


def _make_optim(peak_lr: float = 1.0) -> torch.optim.Optimizer:
    model = nn.Linear(2, 2)
    return torch.optim.SGD(model.parameters(), lr=peak_lr)


def _trace_lrs(optimizer: torch.optim.Optimizer, scheduler: object, steps: int) -> list[float]:
    lrs: list[float] = []
    for _ in range(steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()  # type: ignore[attr-defined]
    lrs.append(optimizer.param_groups[0]["lr"])
    return lrs


def test_warmup_ramps_from_near_zero_to_peak() -> None:
    opt = _make_optim(peak_lr=1.0)
    sched = warmup_cosine_schedule(opt, warmup_steps=10, total_steps=100)
    lrs = _trace_lrs(opt, sched, 11)
    # Initial LR should be near zero (start_factor=1e-8).
    assert lrs[0] < 0.01
    # By the warmup boundary it should be ~peak (1.0 here).
    assert lrs[10] == pytest.approx(1.0, rel=1.0e-3)


def test_warmup_is_monotone_increasing() -> None:
    opt = _make_optim()
    sched = warmup_cosine_schedule(opt, warmup_steps=8, total_steps=100)
    lrs = _trace_lrs(opt, sched, 8)
    for a, b in pairwise(lrs):
        assert b >= a


def test_cosine_decays_to_min_ratio_floor() -> None:
    """At total_steps the LR should equal min_lr_ratio * peak_lr."""
    opt = _make_optim(peak_lr=2.0)
    total = 100
    warmup = 10
    ratio = 0.1
    sched = warmup_cosine_schedule(
        opt, warmup_steps=warmup, total_steps=total, min_lr_ratio=ratio,
    )
    for _ in range(total):
        sched.step()
    final_lr = opt.param_groups[0]["lr"]
    assert final_lr == pytest.approx(2.0 * ratio, rel=1.0e-3)


def test_min_ratio_zero_decays_to_zero() -> None:
    opt = _make_optim(peak_lr=1.0)
    sched = warmup_cosine_schedule(
        opt, warmup_steps=5, total_steps=50, min_lr_ratio=0.0,
    )
    for _ in range(50):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1.0e-6)


def test_warmup_zero_uses_pure_cosine() -> None:
    """warmup_steps=0 → CosineAnnealingLR alone (no SequentialLR wrapping)."""
    opt = _make_optim()
    sched = warmup_cosine_schedule(opt, warmup_steps=0, total_steps=50)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_invalid_warmup_negative() -> None:
    opt = _make_optim()
    with pytest.raises(ValueError, match="warmup_steps must be"):
        warmup_cosine_schedule(opt, warmup_steps=-1, total_steps=10)


def test_invalid_warmup_exceeds_total() -> None:
    opt = _make_optim()
    with pytest.raises(ValueError, match="< total_steps"):
        warmup_cosine_schedule(opt, warmup_steps=10, total_steps=5)


def test_invalid_min_lr_ratio() -> None:
    opt = _make_optim()
    with pytest.raises(ValueError, match="min_lr_ratio"):
        warmup_cosine_schedule(opt, warmup_steps=2, total_steps=10, min_lr_ratio=-0.1)
    with pytest.raises(ValueError, match="min_lr_ratio"):
        warmup_cosine_schedule(opt, warmup_steps=2, total_steps=10, min_lr_ratio=1.5)


def test_lr_curve_shape_is_smooth() -> None:
    """No discontinuities at the warmup→cosine boundary larger than ~1% of peak."""
    opt = _make_optim(peak_lr=1.0)
    sched = warmup_cosine_schedule(opt, warmup_steps=20, total_steps=200)
    lrs = _trace_lrs(opt, sched, 200)
    # At the boundary (step 20 → 21), LR should still be ~peak with tiny step down.
    boundary_step = lrs[20] - lrs[21]
    assert abs(boundary_step) < 0.02
    # And the cosine half-period midpoint should be at half of (peak + floor)/2 ≈ 0.5.
    midpoint = lrs[20 + (200 - 20) // 2]
    expected_mid = (1.0 + 0.1) / 2  # default min_lr_ratio=0.1
    assert math.isclose(midpoint, expected_mid, rel_tol=0.05)
