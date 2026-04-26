"""OPD — On-Policy Distillation with multi-teacher pooling.

The student samples its own rollouts (the *on-policy* part) and is
trained to match an aggregated teacher distribution at each response
position. With one teacher this reduces to standard logit
distillation; with multiple teachers it pools them — the typical
recipe for combining domain specialists (math + code + agent) into a
generalist student without re-running every teacher's training.

Pipeline:

    1. Sample response with the student.
    2. Forward the same (prompt + response) tokens through every
       teacher in the pool to gather their logits at the response
       positions.
    3. Aggregate the teachers (mean / weighted / max) into a single
       reference distribution.
    4. Compute temperature-scaled forward KL
       :math:`KL(student || teacher_agg)` on response tokens only.

This module ships the *math* and a thin step helper. Sampling, reward
gating, and rollout filtering live in the trainer driver — same split
as GRPO.

Aggregation modes:

* ``"mean"`` — equal-weight mean of teacher distributions.
* ``"weighted"`` — convex-weighted mean (caller supplies weights summing
  to 1; ``weights`` shape ``(K,)``).
* ``"max"`` — per-token argmax over teachers' top-1 token (sharp
  consensus). Use when one teacher's specialty clearly dominates
  the prompt; degrades to single-teacher behavior when one weight
  is much larger than the rest.

Reference:
    Yang et al. 2024 (multi-teacher distillation)
    Hinton et al. 2015 (KD baseline)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class OPDConfig:
    """OPD hyperparameters.

    Attributes:
        temperature:    softens both student and teacher distributions
            before the KL. Standard KD trick — higher T -> softer
            targets -> stronger emphasis on relative rankings of
            non-top tokens. Default 1.0 = no softening.
        aggregation:    how to pool teacher distributions ("mean",
            "weighted", "max"). Default "mean".
        kl_coef:        scalar weight on the KL term in the final
            loss. Useful when OPD is mixed with another objective
            (e.g. SFT CE). Default 1.0.
    """

    temperature: float = 1.0
    aggregation: Literal["mean", "weighted", "max"] = "mean"
    kl_coef: float = 1.0


def aggregate_teacher_logits(
    teacher_logits: Sequence[Tensor],
    *,
    weights: Tensor | None = None,
    mode: Literal["mean", "weighted", "max"] = "mean",
) -> Tensor:
    """Pool ``K`` teacher logit tensors into one reference distribution.

    All teachers must share shape ``(B, T, V)``.

    * ``mode="mean"``: arithmetic mean of softmaxed teacher
      probabilities, then re-logged. Equivalent to a uniform-weight
      mixture distribution.
    * ``mode="weighted"``: same as ``"mean"`` but with caller-supplied
      ``weights`` of shape ``(K,)`` summing to 1.
    * ``mode="max"``: per-token take the teacher whose top-1 logit is
      largest. Hard mixture-of-experts: each token follows whichever
      teacher is most confident at that position.

    Returns ``(B, T, V)`` log-probabilities. Working in log-space
    keeps the KL math numerically stable.
    """
    if not teacher_logits:
        raise ValueError("teacher_logits is empty")
    shapes = {tuple(t.shape) for t in teacher_logits}
    if len(shapes) != 1:
        raise ValueError(
            f"all teachers must share shape; got {shapes}",
        )
    if any(t.dim() != 3 for t in teacher_logits):
        raise ValueError(
            f"teachers must be (B, T, V); got dims {[t.dim() for t in teacher_logits]}",
        )

    if mode == "max":
        # Take the teacher whose max-logit is largest at each (b, t).
        stacked_logits = torch.stack(list(teacher_logits), dim=0)  # (K, B, T, V)
        max_per_teacher = stacked_logits.max(dim=-1).values  # (K, B, T)
        chosen = max_per_teacher.argmax(dim=0)  # (B, T)
        chosen_idx = chosen.unsqueeze(0).unsqueeze(-1).expand(
            1, *stacked_logits.shape[1:],
        )
        gathered = torch.gather(stacked_logits, 0, chosen_idx).squeeze(0)
        return F.log_softmax(gathered, dim=-1)

    teacher_log_probs = torch.stack(
        [F.log_softmax(t, dim=-1) for t in teacher_logits], dim=0,
    )  # (K, B, T, V)

    if mode == "mean":
        w = torch.full(
            (teacher_log_probs.shape[0],),
            1.0 / teacher_log_probs.shape[0],
            device=teacher_log_probs.device,
            dtype=teacher_log_probs.dtype,
        )
    elif mode == "weighted":
        if weights is None:
            raise ValueError("weighted aggregation requires 'weights'")
        if weights.dim() != 1 or weights.shape[0] != teacher_log_probs.shape[0]:
            raise ValueError(
                f"weights shape {tuple(weights.shape)} must match "
                f"({teacher_log_probs.shape[0]},)",
            )
        if not torch.isclose(
            weights.sum(), torch.tensor(1.0, dtype=weights.dtype),
        ):
            raise ValueError(
                f"weights must sum to 1; got sum={weights.sum().item()}",
            )
        w = weights.to(teacher_log_probs.dtype).to(teacher_log_probs.device)
    else:
        raise ValueError(f"unknown aggregation mode: {mode!r}")

    # Mixture log-prob: log(sum_k w_k * exp(log p_k)) using logsumexp for stability.
    log_w = torch.log(w + 1e-12).view(-1, 1, 1, 1)
    return torch.logsumexp(teacher_log_probs + log_w, dim=0)


def opd_kl_loss(
    student_logits: Tensor,
    teacher_log_probs: Tensor,
    response_mask: Tensor,
    *,
    temperature: float = 1.0,
) -> tuple[Tensor, int]:
    """Forward KL ``KL(student || teacher_agg)`` on response positions.

    Args:
        student_logits:    ``(B, T, V)`` raw student logits.
        teacher_log_probs: ``(B, T, V)`` aggregated teacher log-probs
            (output of :func:`aggregate_teacher_logits`).
        response_mask:     ``(B, T)`` long — 1 at response positions.
        temperature:       softens both distributions before the KL.

    Returns ``(loss, n_active_positions)``. Loss is zero (no NaN) when
    no positions are active.
    """
    if student_logits.shape != teacher_log_probs.shape:
        raise ValueError(
            "student / teacher logits must share shape; got "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_log_probs.shape)}",
        )
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive; got {temperature}")

    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    # Re-soften the (already log-probed) teacher distribution at the
    # same temperature. Multiplying log-probs by 1/T and renormalising
    # is equivalent to softmax(z/T) on the source logits.
    t_log_probs = F.log_softmax(teacher_log_probs / temperature, dim=-1)

    # Forward KL: sum_v p_s * (log p_s - log p_t).
    s_probs = s_log_probs.exp()
    kl_per_pos = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)

    mask = response_mask.to(kl_per_pos.dtype)
    masked = kl_per_pos * mask
    n_active = int(mask.sum().item())
    denom = mask.sum().clamp(min=1.0)
    # Standard KD scaling: multiply by T^2 so gradients don't shrink
    # at high temperatures.
    return (masked.sum() / denom) * (temperature**2), n_active


def opd_step(
    student: torch.nn.Module,
    teachers: Sequence[torch.nn.Module],
    tokens: Tensor,
    response_mask: Tensor,
    *,
    cfg: OPDConfig,
    teacher_weights: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Run one OPD step: forward all models, aggregate, compute KL.

    Args:
        student:         the student module being trained. Called as
            ``student(tokens)`` returning ``{"logits": ...}``.
        teachers:        sequence of teacher modules; same call shape.
        tokens:          ``(B, T)`` long — prompt + on-policy response.
        response_mask:   ``(B, T)`` long.
        cfg:             :class:`OPDConfig`.
        teacher_weights: required iff ``cfg.aggregation == "weighted"``.

    Returns ``(loss, metrics)``. The student's forward pass is part of
    the autograd graph; the teachers' forwards are not (they're
    detached so the teachers stay frozen).
    """
    student_out = student(tokens)
    student_logits = student_out["logits"]
    if not isinstance(student_logits, Tensor):
        raise TypeError("student logits must be a Tensor")

    with torch.no_grad():
        teacher_logits_list = []
        for teacher in teachers:
            t_out = teacher(tokens)
            t_logits = t_out["logits"]
            if not isinstance(t_logits, Tensor):
                raise TypeError("teacher logits must be a Tensor")
            teacher_logits_list.append(t_logits)
        teacher_log_probs = aggregate_teacher_logits(
            teacher_logits_list,
            weights=teacher_weights,
            mode=cfg.aggregation,
        )

    kl, n_active = opd_kl_loss(
        student_logits,
        teacher_log_probs,
        response_mask,
        temperature=cfg.temperature,
    )
    loss = cfg.kl_coef * kl

    with torch.no_grad():
        metrics = {
            "kl": kl.detach(),
            "n_active": torch.tensor(n_active, device=loss.device),
            "n_teachers": torch.tensor(len(teachers), device=loss.device),
        }
    return loss, metrics
