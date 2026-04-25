"""Learning-rate schedules.

The standard "linear warmup → cosine decay" pattern composed from torch's
built-in primitives. Matches what every modern LLM trainer does (DeepSeek-V3,
Llama, Mistral, etc.); we just spell it as one call so the trainer setup
isn't littered with SequentialLR + milestones boilerplate.
"""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)


def warmup_cosine_schedule(
    optimizer: Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LRScheduler:
    """Linear warmup from ~0 to peak LR over ``warmup_steps``, then cosine
    decay to ``min_lr_ratio * peak_lr`` over the remaining budget.

    Args:
        optimizer: target optimizer; the schedule reads its initial param-group
            LRs as the peak.
        warmup_steps: number of optimizer steps over which the LR ramps up.
            Must be ``>= 0`` and ``< total_steps``.
        total_steps: total expected optimizer steps. Cosine ``T_max`` =
            ``total_steps - warmup_steps``.
        min_lr_ratio: floor of the cosine decay, expressed as a fraction of
            the peak LR. ``0.0`` decays all the way to zero, ``1.0`` is no
            decay at all.

    Returns:
        a SequentialLR wrapping a LinearLR + CosineAnnealingLR pair.
    """
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be ≥ 0; got {warmup_steps}")
    if warmup_steps >= total_steps:
        raise ValueError(
            f"warmup_steps={warmup_steps} must be < total_steps={total_steps}",
        )
    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError(f"min_lr_ratio must be in [0, 1]; got {min_lr_ratio}")

    peak_lrs = [g["lr"] for g in optimizer.param_groups]
    eta_min = min(peak_lrs) * min_lr_ratio if peak_lrs else 0.0

    if warmup_steps == 0:
        # SequentialLR refuses an empty stage; return cosine alone.
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)

    warmup = LinearLR(
        optimizer,
        start_factor=1.0e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=eta_min,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )
