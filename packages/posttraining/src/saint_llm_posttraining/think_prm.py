"""ThinkPRM — Process Reward Model for chain-of-thought (RL-10).

A *Process Reward Model* (PRM) scores each step of a reasoning trace
as good or bad, in contrast to an *Outcome Reward Model* (ORM) that
only scores the final answer. The dense per-step signal is the right
shape for:

* **SFT trace filtering** — drop traces with bad intermediate steps
  even when their final answer happens to be correct.
* **Dense GRPO rewards** — give credit at the step that introduced
  the error rather than smearing the failure across all tokens.
* **Best-of-N answer verification** — pick among candidates by
  agreement of step scores rather than only final-answer correctness.

ThinkPRM (Liu et al. 2025) trains a small score head on top of the
base LM's hidden states. Each reasoning step is delimited by a
boundary token (e.g. ``"\\n\\n"`` or a dedicated step-end token);
the score head reads the hidden state at the boundary position and
predicts whether the *preceding* step was correct.

This module ships the algorithmic core:

* :class:`StepRewardHead` — small MLP over LM hidden states.
* :func:`find_step_boundaries` — locate step-end positions in a
  token sequence given a boundary-token ID.
* :func:`gather_step_logits` / :func:`gather_step_scores` —
  pull per-step logits/probabilities at the boundaries.
* :func:`step_prm_loss` — masked BCE-with-logits loss for training.
* :func:`compute_step_rewards` — convenience inference: hidden
  states + boundaries → per-step probability tensor.
* :func:`prm_filter_mask` — pure-python helper deciding whether a
  trace should be kept based on per-step scores + threshold.

The module is model-agnostic: plug it on top of any backbone that
exposes hidden states ``(B, T, D)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ThinkPRMConfig:
    """Configuration for the step-reward head.

    Attributes:
        hidden_dim:    LM residual dim that feeds the head.
        intermediate:  Hidden width of the head's MLP. Default 2*hidden.
        threshold:     Sigmoid cutoff for "good step" classification.
            Used by :func:`prm_filter_mask`. Default 0.5.
        keep_min_step_score: Lowest per-step probability allowed when
            filtering traces. A trace is kept iff every step's
            probability >= this. Default 0.4 — a lenient threshold so
            v0.0 trace filtering doesn't over-prune.
    """

    hidden_dim: int
    intermediate: int = 0  # 0 -> use 2*hidden_dim
    threshold: float = 0.5
    keep_min_step_score: float = 0.4

    def effective_intermediate(self) -> int:
        return self.intermediate if self.intermediate > 0 else 2 * self.hidden_dim


class StepRewardHead(nn.Module):
    """Small MLP scoring a per-position hidden state.

    Input shape ``(..., hidden_dim)``; output ``(..., 1)`` (logits).
    Apply :func:`torch.sigmoid` to obtain a per-step probability.
    """

    def __init__(self, cfg: ThinkPRMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d_inter = cfg.effective_intermediate()
        self.fc1 = nn.Linear(cfg.hidden_dim, d_inter)
        self.fc2 = nn.Linear(d_inter, 1)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(hidden)))


def find_step_boundaries(
    tokens: Tensor, boundary_id: int,
) -> Tensor:
    """Locate step-end positions in a ``(B, T)`` token tensor.

    Returns a long tensor of shape ``(B, T)`` where each row contains
    boundary positions left-padded with ``-1`` for unused slots. The
    returned tensor's max useful width is ``max_steps`` across the
    batch — callers can ``.max()`` over the row to discard padding.

    Edge cases:
    * if a row has no boundary token, the row is all ``-1`` (no steps).
    * positions where ``tokens == boundary_id`` are themselves the
      boundary; the score head reads the hidden state at this position.
    """
    if tokens.dim() != 2:
        raise ValueError(f"tokens must be (B, T); got shape {tuple(tokens.shape)}")
    b, t = tokens.shape
    out = torch.full((b, t), -1, dtype=torch.long, device=tokens.device)
    for bi in range(b):
        pos = torch.nonzero(tokens[bi] == boundary_id, as_tuple=False).flatten()
        out[bi, : pos.numel()] = pos
    return out


def gather_step_logits(
    hidden: Tensor,
    head: StepRewardHead,
    step_positions: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run the head over hidden states at ``step_positions``.

    Args:
        hidden:         ``(B, T, D)`` LM hidden states.
        head:           :class:`StepRewardHead`.
        step_positions: ``(B, S)`` long, with ``-1`` padding.

    Returns:
        ``(logits, valid_mask)``:
        * ``logits``     — ``(B, S)`` float; entries at padded slots
          are zero (don't read them).
        * ``valid_mask`` — ``(B, S)`` bool; True where the slot is a
          real step boundary.
    """
    if hidden.dim() != 3:
        raise ValueError(f"hidden must be (B, T, D); got {tuple(hidden.shape)}")
    b, t, d = hidden.shape
    if step_positions.shape[0] != b:
        raise ValueError(
            f"batch mismatch: hidden B={b}, step_positions B={step_positions.shape[0]}",
        )

    valid = step_positions >= 0
    safe_pos = step_positions.clamp(min=0)
    flat_idx = (
        torch.arange(b, device=hidden.device).unsqueeze(-1) * t + safe_pos
    )  # (B, S)
    flat_hidden = hidden.reshape(b * t, d)
    gathered = flat_hidden.index_select(0, flat_idx.flatten()).view(
        b, step_positions.shape[1], d,
    )
    logits = head(gathered).squeeze(-1)  # (B, S)
    logits = logits.masked_fill(~valid, 0.0)
    return logits, valid


def gather_step_scores(
    hidden: Tensor,
    head: StepRewardHead,
    step_positions: Tensor,
) -> tuple[Tensor, Tensor]:
    """Same as :func:`gather_step_logits` but returns sigmoid probabilities."""
    logits, valid = gather_step_logits(hidden, head, step_positions)
    probs = torch.sigmoid(logits)
    probs = probs.masked_fill(~valid, 0.0)
    return probs, valid


def step_prm_loss(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
    *,
    pos_weight: float | None = None,
) -> Tensor:
    """Masked BCE-with-logits loss for step-level supervision.

    Args:
        logits:     ``(B, S)`` float — output of :func:`gather_step_logits`.
        labels:     ``(B, S)`` float in {0, 1} — 1 = good step, 0 = bad.
        mask:       ``(B, S)`` bool — True for real boundaries (from the
            ``valid_mask`` returned by gather).
        pos_weight: optional class-imbalance reweighting passed straight
            to :func:`F.binary_cross_entropy_with_logits`.

    Returns:
        Mean BCE loss over masked positions. If no positions are
        active the loss is zero (no NaN).
    """
    if logits.shape != labels.shape or logits.shape != mask.shape:
        raise ValueError(
            "logits, labels, mask must share shape; got "
            f"{tuple(logits.shape)}, {tuple(labels.shape)}, {tuple(mask.shape)}",
        )
    pw = (
        torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
        if pos_weight is not None
        else None
    )
    elementwise = F.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction="none", pos_weight=pw,
    )
    elementwise = elementwise * mask.float()
    denom = mask.float().sum().clamp(min=1.0)
    return elementwise.sum() / denom


def compute_step_rewards(
    hidden: Tensor,
    head: StepRewardHead,
    tokens: Tensor,
    boundary_id: int,
) -> tuple[Tensor, Tensor]:
    """Inference convenience: hidden + tokens -> per-step probabilities.

    Returns ``(probs, mask)`` where ``probs[b, s]`` is the probability
    of step ``s`` in row ``b`` being a good step (0.0 at padded slots),
    and ``mask`` flags real entries.
    """
    positions = find_step_boundaries(tokens, boundary_id)
    return gather_step_scores(hidden, head, positions)


def prm_filter_mask(
    probs: Tensor, valid: Tensor, *, min_score: float,
) -> Tensor:
    """Per-row decision: True = keep this trace; False = drop.

    A trace is kept iff every *real* step's probability is >= ``min_score``.
    Rows with no boundaries are kept by default (no information to
    drop on).
    """
    if probs.shape != valid.shape:
        raise ValueError("probs and valid must share shape")
    bad = (probs < min_score) & valid
    any_bad = bad.any(dim=-1)
    return ~any_bad
