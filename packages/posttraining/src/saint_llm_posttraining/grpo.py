"""GRPO — Group Relative Policy Optimization with Q1 2026 patches.

The original DeepSeekMath GRPO (group-normalized advantage + clipped
surrogate + Schulman unbiased KL) is preserved as the default; on top
of it ``GRPOConfig`` exposes the Q1 2026 frontier patches per the
ADRs accepted in ``docs/adr/``:

* **GSPO** (ADR-0014, RL-01) — sequence-level importance ratio with
  length normalization. Required for MoE training stability since
  routing changes between rollout and update inflate per-token ratio
  variance. ``importance_ratio_level="sequence"``.
* **DAPO** (ADR-0015, RL-02) — four orthogonal patches: Clip-Higher
  (decoupled upper/lower clip), Dynamic Sampling (drop zero-variance
  groups), Token-Level PG (sum tokens not mean responses), Overlong
  Reward Shaping (soft-penalty band).
* **Dr.GRPO + CE-GPPO** (ADR-pending RL-03) — unbiased loss tweaks:
  optional removal of length-normalization and per-group std
  normalization.

This module provides the *math*. Rollout generation, reward functions,
old/ref policy snapshotting, and rollout filtering before this loss is
called all live in the trainer driver (Workstream B).

References:
    DeepSeekMath §4 (Shao et al., 2024)
    DeepSeek-R1 §2 (Guo et al., 2025)
    GSPO arXiv 2507.18071 (Qwen team)
    DAPO arXiv 2503.14476 (ByteDance Seed)
    Dr.GRPO arXiv 2503.20783
    Schulman 2020 "Approximating KL Divergence"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class GRPOConfig:
    """GRPO hyperparameters with Q1 2026 patch knobs.

    The default values reproduce vanilla GRPO (matches existing test
    semantics). Production v0.2 overrides via ADR recommendations:

    * ``importance_ratio_level="sequence"`` — GSPO (ADR-0014).
    * ``use_clip_higher=True``, ``clip_eps_upper=0.28``,
      ``clip_eps_lower=0.20`` — DAPO Clip-Higher (ADR-0015).
    * ``token_level_pg=True`` — DAPO Token-Level PG (ADR-0015).
    * ``overlong_reward_band=(0.95, 1.0)`` — DAPO Overlong Reward
      Shaping (ADR-0015).
    * ``unbiased_loss=True`` — Dr.GRPO unbiased loss tweaks (RL-03):
      removes the per-completion length normalization and the
      per-group std normalization in advantage computation.

    Dynamic Sampling is implemented as a separate filter
    :func:`dynamic_sampling_mask` because the trainer driver needs to
    re-sample rollouts when groups are dropped, not just zero them.

    Attributes:
        group_size:    G — completions per prompt. Larger G -> lower
            advantage variance, higher rollout cost.
        clip_eps:      symmetric PPO clip width when
            ``use_clip_higher=False``. Defaults to 0.2 (PPO standard).
        clip_eps_upper / clip_eps_lower: asymmetric clip bounds when
            ``use_clip_higher=True``. DAPO recipe: 0.28 / 0.20.
        use_clip_higher: if True, use asymmetric clip; otherwise
            symmetric ``clip_eps``.
        kl_coef:       beta — strength of KL-to-reference penalty.
            DeepSeekMath sweeps in {0.01, 0.04}.
        advantage_eps: numerical floor for the std denominator.
        importance_ratio_level: "token" (vanilla GRPO) or "sequence"
            (GSPO, length-normalized geometric-mean ratio).
        token_level_pg: DAPO Token-Level Policy Gradient — sum the
            surrogate over tokens (no per-completion length divide),
            then mean over completions. Default False = legacy
            mean-over-tokens-then-mean-over-completions.
        overlong_reward_band: (start, end) as fractions of max length;
            soft-penalty band where reward decays linearly from full
            to zero. ``None`` disables (use raw reward).
        unbiased_loss: Dr.GRPO patches — drop length-norm AND
            per-group std-norm. Slight bias reduction at no cost.
    """

    group_size: int = 8
    clip_eps: float = 0.2
    clip_eps_upper: float = 0.28
    clip_eps_lower: float = 0.20
    use_clip_higher: bool = False
    kl_coef: float = 0.04
    advantage_eps: float = 1e-8
    importance_ratio_level: Literal["token", "sequence"] = "token"
    token_level_pg: bool = False
    overlong_reward_band: tuple[float, float] | None = None
    unbiased_loss: bool = False


@dataclass(frozen=True)
class RolloutBatch:
    """One GRPO rollout, flattened across (prompt, completion) pairs.

    Layout: rows ``[b*G : (b+1)*G]`` are the G completions for prompt b.
    The training script is responsible for keeping that layout when it
    samples — every helper here assumes it.

    Attributes:
        tokens:        ``(B*G, T)`` long — prompt + response, padded.
        response_mask: ``(B*G, T)`` long — 1 at positions whose target
            token is part of the response (and therefore counts toward
            the loss), 0 on prompt/pad.
        old_logprobs:  ``(B*G, T)`` float — ``log π_old(tokens[t] |
            tokens[<t])`` evaluated *under the actor at rollout time*
            (the snapshot frozen at the start of this PPO step). Position
            0 is filled with 0; it is always masked out.
        ref_logprobs:  ``(B*G, T)`` float — same but under the reference
            policy (typically the SFT init).
        rewards:       ``(B*G,)`` float — scalar reward per completion.
    """

    tokens: Tensor
    response_mask: Tensor
    old_logprobs: Tensor
    ref_logprobs: Tensor
    rewards: Tensor


def compute_group_advantage(
    rewards: Tensor,
    *,
    group_size: int,
    eps: float = 1e-8,
    unbiased: bool = False,
) -> Tensor:
    """Group-normalize a flat ``(B*G,)`` reward tensor.

    Default returns ``(r - mean_g) / (std_g + eps)`` (DeepSeekMath
    GRPO). When ``unbiased=True`` (Dr.GRPO, ADR-pending RL-03) returns
    just ``r - mean_g`` without per-group std normalization, which
    avoids the bias toward easier groups documented in arXiv 2503.20783.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive; got {group_size}")
    bg = rewards.shape[0]
    if bg % group_size != 0:
        raise ValueError(
            f"rewards length {bg} is not divisible by group_size {group_size}",
        )
    rewards_grp = rewards.view(-1, group_size)
    mean = rewards_grp.mean(dim=-1, keepdim=True)
    if unbiased:
        adv = rewards_grp - mean
    else:
        std = rewards_grp.std(dim=-1, unbiased=False, keepdim=True)
        adv = (rewards_grp - mean) / (std + eps)
    return adv.view(-1)


def dynamic_sampling_mask(
    rewards: Tensor,
    *,
    group_size: int,
) -> Tensor:
    """DAPO Dynamic Sampling — drop groups with zero advantage variance.

    Returns a ``(B*G,)`` bool mask: True for rows in groups whose rewards
    have non-zero variance (i.e. groups with at least one differing
    reward). All-correct or all-wrong groups are masked out.

    The trainer driver is responsible for re-sampling rollouts when
    groups are dropped — this function only flags them.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive; got {group_size}")
    bg = rewards.shape[0]
    if bg % group_size != 0:
        raise ValueError(
            f"rewards length {bg} is not divisible by group_size {group_size}",
        )
    rewards_grp = rewards.view(-1, group_size)
    # Variance > 0 iff group is non-degenerate.
    keep_group = rewards_grp.var(dim=-1, unbiased=False) > 0.0
    return keep_group.unsqueeze(-1).expand(-1, group_size).reshape(-1)


def overlong_reward_shape(
    rewards: Tensor,
    response_lengths: Tensor,
    *,
    max_length: int,
    band: tuple[float, float] | None,
) -> Tensor:
    """DAPO Overlong Reward Shaping — soft-penalty band near max length.

    For ``band=(a, b)`` with ``0 < a < b <= 1``: rewards are unchanged
    while ``length / max_length <= a``, then linearly decay to zero
    multiplier at ``length / max_length >= b``. ``band=None`` returns
    rewards unchanged.

    Args:
        rewards: ``(B*G,)`` raw rewards.
        response_lengths: ``(B*G,)`` actual response token counts.
        max_length: training-time maximum response length.
        band: ``(start_frac, end_frac)`` or ``None``.

    Returns:
        Shaped ``(B*G,)`` rewards.
    """
    if band is None:
        return rewards
    a, b = band
    if not (0.0 < a < b <= 1.0):
        raise ValueError(f"band must satisfy 0 < a < b <= 1; got {band}")
    frac = response_lengths.to(rewards.dtype) / max(max_length, 1)
    # Linear decay multiplier in [0, 1].
    multiplier = ((b - frac) / (b - a)).clamp(min=0.0, max=1.0)
    return rewards * multiplier


def gather_token_logprobs(logits: Tensor, tokens: Tensor) -> Tensor:
    """Per-position log-prob of the *next* token under ``logits``.

    ``logits`` is ``(B, T, V)`` raw model output. ``tokens`` is ``(B, T)``.
    Returns ``(B, T)`` where index ``t >= 1`` holds
    ``log π(tokens[:, t] | tokens[:, :t])`` and index 0 is filled with
    zero (the first position is always masked out — there is no
    conditioning context for it).

    Implementation: log-softmax then gather at the shifted target.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be (B, T, V); got shape {tuple(logits.shape)}")
    if tokens.shape != logits.shape[:2]:
        raise ValueError(
            f"tokens shape {tuple(tokens.shape)} must match logits[:, :, 0] shape "
            f"{tuple(logits.shape[:2])}",
        )

    log_probs = F.log_softmax(logits, dim=-1)
    shifted = log_probs[:, :-1]
    targets = tokens[:, 1:]
    gathered = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    pad = torch.zeros(gathered.shape[0], 1, dtype=gathered.dtype, device=gathered.device)
    return torch.cat([pad, gathered], dim=1)


def grpo_loss(
    logits: Tensor,
    batch: RolloutBatch,
    *,
    cfg: GRPOConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Clipped surrogate + per-token KL penalty with optional Q1 2026 patches.

    Default behavior reproduces vanilla GRPO; toggling ``cfg`` flags
    enables the patches:

    * ``importance_ratio_level="sequence"`` (GSPO, ADR-0014) — replaces
      per-token ratio with a length-normalized geometric-mean sequence
      ratio computed over the response tokens. Required for MoE
      stability.
    * ``use_clip_higher=True`` (DAPO Clip-Higher, ADR-0015) — uses
      asymmetric clip ``(1 - clip_eps_lower, 1 + clip_eps_upper)``
      instead of the symmetric ``clip_eps`` band.
    * ``token_level_pg=True`` (DAPO Token-Level PG, ADR-0015) — sums
      the surrogate over tokens within a sequence (no per-completion
      length divide), then means over completions.
    * ``unbiased_loss=True`` (Dr.GRPO, RL-03) — uses ``compute_group_
      advantage(unbiased=True)`` and skips the per-completion length
      normalization (similar to ``token_level_pg``).

    Returns ``(loss, metrics)``. ``metrics`` is a dict of detached
    scalars suitable for logging.
    """
    new_logprobs = gather_token_logprobs(logits, batch.tokens)
    advantages = compute_group_advantage(
        batch.rewards,
        group_size=cfg.group_size,
        eps=cfg.advantage_eps,
        unbiased=cfg.unbiased_loss,
    )

    mask = batch.response_mask.to(new_logprobs.dtype)
    adv_seq = advantages.to(new_logprobs.dtype)

    # Per-token log-ratio (GSPO computes a length-normalized sequence
    # ratio from the same per-token signal).
    log_ratio = new_logprobs - batch.old_logprobs.to(new_logprobs.dtype)

    if cfg.importance_ratio_level == "sequence":
        # GSPO: r_seq = exp((1/|R|) * sum_t log_ratio_t * mask_t).
        # Geometric-mean ratio, broadcast back to per-token positions.
        masked_log_ratio = log_ratio * mask
        n_tok_for_ratio = mask.sum(dim=-1).clamp(min=1.0)
        log_ratio_seq = masked_log_ratio.sum(dim=-1) / n_tok_for_ratio
        ratio = log_ratio_seq.exp().unsqueeze(-1).expand_as(log_ratio)
    else:
        ratio = log_ratio.exp()

    if cfg.use_clip_higher:
        clip_lo, clip_hi = 1.0 - cfg.clip_eps_lower, 1.0 + cfg.clip_eps_upper
    else:
        clip_lo, clip_hi = 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps
    adv_tok = adv_seq.unsqueeze(-1)
    surrogate1 = ratio * adv_tok
    surrogate2 = ratio.clamp(clip_lo, clip_hi) * adv_tok
    pg_objective = torch.minimum(surrogate1, surrogate2)

    kl_logr = batch.ref_logprobs.to(new_logprobs.dtype) - new_logprobs
    kl_per_tok = kl_logr.exp() - kl_logr - 1.0

    n_tok = mask.sum(dim=-1).clamp(min=1.0)
    if cfg.token_level_pg or cfg.unbiased_loss:
        # Token-level PG (DAPO) or Dr.GRPO unbiased: sum tokens, no
        # per-completion length normalization. Then mean across the
        # batch in the final reduction.
        pg_term = -(pg_objective * mask).sum(dim=-1)
        kl_term = (kl_per_tok * mask).sum(dim=-1)
    else:
        pg_term = -(pg_objective * mask).sum(dim=-1) / n_tok
        kl_term = (kl_per_tok * mask).sum(dim=-1) / n_tok
    per_completion = pg_term + cfg.kl_coef * kl_term
    loss = per_completion.mean()

    with torch.no_grad():
        clipped = (surrogate1 != surrogate2).to(mask.dtype) * mask
        clip_frac = clipped.sum() / mask.sum().clamp(min=1.0)
        metrics = {
            "pg_loss": pg_term.mean().detach(),
            "kl_penalty": kl_term.mean().detach(),
            "clip_frac": clip_frac.detach(),
            "advantage_mean": advantages.mean().detach(),
            "advantage_std": advantages.std(unbiased=False).detach(),
            "reward_mean": batch.rewards.mean().detach(),
            "ratio_mean": ((ratio * mask).sum() / mask.sum().clamp(min=1.0)).detach(),
        }
    return loss, metrics
