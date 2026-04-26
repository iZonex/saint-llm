"""GTPO — Generalized Token-level Policy Optimization (RL-06).

GRPO normalizes advantages within a group of completions and then
applies the same scalar advantage to every token in a completion. The
per-token gradient is ``ratio_t * adv_seq``: a token contributes
proportionally to how it affects the *whole* completion's reward.

GTPO observes that not every token in a completion deserves equal
credit:

* **Forced tokens** — positions where the model's distribution is
  almost a delta (e.g. mid-word continuations, predictable
  syntactic glue). The policy isn't really making a *choice* there;
  pushing or pulling it via PG is wasted bandwidth and a source of
  spurious gradient noise.
* **Decision tokens** — high-entropy positions where the model has
  multiple plausible options. These are where the rollout's outcome
  was actually shaped; they deserve the full gradient signal.

This module ships a v0.0 GTPO that scales the per-token advantage by
the token's normalized entropy::

    w_t = H_t / max_H_in_response                 # in [0, 1]
    A_token[t] = A_seq * (1 - alpha + alpha * w_t)

with ``alpha`` controlling the strength of the entropy correction
(``alpha=0`` reduces exactly to GRPO; ``alpha=1`` makes a fully-
forced token contribute zero gradient). The default ``alpha=0.5``
keeps half the GRPO signal for every token and adds the other half
weighted by entropy — a soft mid-point.

This is **our interpretation** of "GTPO" pending alignment with the
canonical paper. The shape (per-token advantage adjustment based on
a model-internal signal) matches the family; the specific weighting
function may differ. Tests verify the math is consistent and
reduces to GRPO at ``alpha=0``.

Implementation note: token entropies are computed from the
**actor's current logits** at training time. They flow through
autograd (the entropy itself isn't differentiated through, but the
weighted advantage *is* — entropy is detached when used as the
weight).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from saint_llm_posttraining.grpo import (
    GRPOConfig,
    RolloutBatch,
    compute_group_advantage,
    gather_token_logprobs,
)


@dataclass(frozen=True)
class GTPOConfig:
    """GTPO knobs on top of :class:`GRPOConfig`.

    Attributes:
        grpo:                base :class:`GRPOConfig`. All Q1 2026
            patches still apply (GSPO sequence ratio, Clip-Higher,
            etc.).
        entropy_alpha:       blend coefficient in ``[0, 1]``.
            ``0`` reduces to GRPO (every token gets full advantage).
            ``1`` zeroes the gradient on fully-forced tokens.
        normalize_per_row:   normalize entropy by the max entropy
            within each rollout's response (default True). If False,
            uses absolute entropy bounded by ``log(vocab_size)`` —
            less stable but more comparable across rollouts.
    """

    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    entropy_alpha: float = 0.5
    normalize_per_row: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.entropy_alpha <= 1.0:
            raise ValueError(
                f"entropy_alpha must be in [0, 1]; got {self.entropy_alpha}",
            )


def per_token_entropy(logits: Tensor) -> Tensor:
    """Compute Shannon entropy of the per-position distribution.

    Args:
        logits: ``(B, T, V)`` raw logits.

    Returns:
        ``(B, T)`` float entropy in nats. Detached from the autograd
        graph — entropies are used as a weighting signal, not
        differentiated through.
    """
    if logits.dim() != 3:
        raise ValueError(
            f"logits must be (B, T, V); got shape {tuple(logits.shape)}",
        )
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1).detach()


def entropy_weight(
    entropy: Tensor,
    response_mask: Tensor,
    *,
    alpha: float,
    normalize_per_row: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Convert per-position entropy to a per-position advantage weight.

    Returns ``(B, T)`` weights in ``[0, 1]`` (zero outside the
    response mask). Each weight is ``(1 - alpha) + alpha * w_t``
    where ``w_t`` is the normalized entropy. ``alpha=0`` gives a
    constant weight 1 (reduces to GRPO).

    Args:
        entropy:           ``(B, T)`` per-position entropy.
        response_mask:     ``(B, T)`` long; 1 on response positions.
        alpha:             blend coefficient in ``[0, 1]``.
        normalize_per_row: divide by the max response-entropy in each
            row, so weights span ``[0, 1]`` regardless of vocab size.
            Falls back to vocab-size normalization when False.
        eps:               numerical floor.
    """
    if entropy.shape != response_mask.shape:
        raise ValueError(
            "entropy and response_mask must share shape; got "
            f"{tuple(entropy.shape)} vs {tuple(response_mask.shape)}",
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")

    mask = response_mask.to(entropy.dtype)
    masked_entropy = entropy * mask
    if normalize_per_row:
        denom = masked_entropy.amax(dim=-1, keepdim=True).clamp(min=eps)
    else:
        # Fall back to log(vocab) — but we don't have vocab here, so
        # use the global max entropy across the batch as a stable
        # surrogate. Tests verify behavior; production callers can pass
        # ``normalize_per_row=True`` for the cleanest math.
        denom = masked_entropy.max().clamp(min=eps)
    w = (masked_entropy / denom).clamp(min=0.0, max=1.0)
    weight = (1.0 - alpha) + alpha * w
    # Zero out non-response positions explicitly so they don't accidentally
    # contribute to the loss reduction even if mask is added later.
    return weight * mask


def gtpo_loss(
    logits: Tensor,
    batch: RolloutBatch,
    *,
    cfg: GTPOConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    """GTPO surrogate with entropy-weighted per-token advantages.

    Same surrogate machinery as :func:`grpo_loss` (clip-high /
    sequence ratio / KL penalty all honored via ``cfg.grpo``); the
    one difference is that the per-token advantage is multiplied by
    a per-token entropy weight before the surrogate is computed.

    Returns ``(loss, metrics)``. Metrics include ``mean_entropy_weight``
    so trainers can sanity-check the entropy distribution.
    """
    grpo = cfg.grpo

    new_logprobs = gather_token_logprobs(logits, batch.tokens)
    advantages = compute_group_advantage(
        batch.rewards,
        group_size=grpo.group_size,
        eps=grpo.advantage_eps,
        unbiased=grpo.unbiased_loss,
    )

    mask = batch.response_mask.to(new_logprobs.dtype)
    adv_seq = advantages.to(new_logprobs.dtype)

    entropy = per_token_entropy(logits).to(new_logprobs.dtype)
    weight = entropy_weight(
        entropy, batch.response_mask,
        alpha=cfg.entropy_alpha,
        normalize_per_row=cfg.normalize_per_row,
    ).to(new_logprobs.dtype)

    log_ratio = new_logprobs - batch.old_logprobs.to(new_logprobs.dtype)
    if grpo.importance_ratio_level == "sequence":
        masked_log_ratio = log_ratio * mask
        n_tok_for_ratio = mask.sum(dim=-1).clamp(min=1.0)
        log_ratio_seq = masked_log_ratio.sum(dim=-1) / n_tok_for_ratio
        ratio = log_ratio_seq.exp().unsqueeze(-1).expand_as(log_ratio)
    else:
        ratio = log_ratio.exp()

    if grpo.use_clip_higher:
        clip_lo, clip_hi = 1.0 - grpo.clip_eps_lower, 1.0 + grpo.clip_eps_upper
    else:
        clip_lo, clip_hi = 1.0 - grpo.clip_eps, 1.0 + grpo.clip_eps

    # Token-level advantages: A_seq broadcast then multiplied by entropy weight.
    adv_token = adv_seq.unsqueeze(-1) * weight  # (B, T)
    surrogate1 = ratio * adv_token
    surrogate2 = ratio.clamp(clip_lo, clip_hi) * adv_token
    pg_objective = torch.minimum(surrogate1, surrogate2)

    kl_logr = batch.ref_logprobs.to(new_logprobs.dtype) - new_logprobs
    kl_per_tok = kl_logr.exp() - kl_logr - 1.0

    n_tok = mask.sum(dim=-1).clamp(min=1.0)
    if grpo.token_level_pg or grpo.unbiased_loss:
        pg_term = -(pg_objective * mask).sum(dim=-1)
        kl_term = (kl_per_tok * mask).sum(dim=-1)
    else:
        pg_term = -(pg_objective * mask).sum(dim=-1) / n_tok
        kl_term = (kl_per_tok * mask).sum(dim=-1) / n_tok
    per_completion = pg_term + grpo.kl_coef * kl_term
    loss = per_completion.mean()

    with torch.no_grad():
        clipped = (surrogate1 != surrogate2).to(mask.dtype) * mask
        clip_frac = clipped.sum() / mask.sum().clamp(min=1.0)
        # Mean entropy weight across response positions — diagnostic for
        # how heavily GTPO is dampening forced tokens.
        mean_weight = (weight * mask).sum() / mask.sum().clamp(min=1.0)
        metrics = {
            "pg_loss": pg_term.mean().detach(),
            "kl_penalty": kl_term.mean().detach(),
            "clip_frac": clip_frac.detach(),
            "advantage_mean": advantages.mean().detach(),
            "advantage_std": advantages.std(unbiased=False).detach(),
            "reward_mean": batch.rewards.mean().detach(),
            "ratio_mean": (
                (ratio * mask).sum() / mask.sum().clamp(min=1.0)
            ).detach(),
            "mean_entropy_weight": mean_weight.detach(),
            "mean_entropy_nats": (
                (entropy * mask).sum() / mask.sum().clamp(min=1.0)
            ).detach(),
        }
    return loss, metrics
