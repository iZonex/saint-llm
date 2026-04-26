"""Tree-GRPO — tree-structured rollouts with sibling-group advantages (RL-05).

Vanilla GRPO samples G flat completions per prompt and normalizes
advantages within that group. Tree-GRPO generalizes: each rollout
carries a *parent index*, so the rollout pool forms a tree (one
root per prompt; each non-root rollout extends some parent's
partial completion). Advantages are normalized within *sibling
groups* — rollouts that share a parent — instead of fixed-size flat
groups.

Why this matters:

* **Variable branching.** Different parents can have different child
  counts (mid-traces are often expanded more aggressively than dead
  ends). Vanilla GRPO can't express this — every group must be the
  same size.
* **Per-step credit.** When paired with :class:`StepRewardHead`
  (ThinkPRM, RL-10), the parent of a rollout that introduces a bad
  step gets penalized via its child's negative advantage, while the
  good-prefix sibling gets pushed up.
* **Search-tree RL.** The output of an MCTS-style rollout policy is
  exactly this tree shape; Tree-GRPO is the natural fit for training
  on such samples without flattening.

This module ships:

* :class:`TreeRolloutBatch` — like :class:`RolloutBatch` plus a
  ``parent_idx`` field.
* :func:`compute_tree_advantage` — sibling-group normalization. Roots
  (``parent_idx == -1``) form one synthetic group together; this
  matches vanilla GRPO when every rollout is a root.
* :func:`tree_grpo_loss` — same surrogate as :func:`grpo_loss` but
  feeds advantages from :func:`compute_tree_advantage` instead of the
  fixed-group-size path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from saint_llm_posttraining.grpo import (
    GRPOConfig,
    gather_token_logprobs,
)


@dataclass(frozen=True)
class TreeRolloutBatch:
    """Tree-structured rollout batch.

    Same fields as :class:`RolloutBatch` plus ``parent_idx`` indicating
    each rollout's parent in the tree. ``parent_idx[i] == -1`` marks a
    root.

    Attributes:
        tokens:        ``(N, T)`` long — prompt + response, padded.
        response_mask: ``(N, T)`` long — 1 at positions whose target
            token is part of the response, 0 elsewhere.
        old_logprobs:  ``(N, T)`` float — log π_old at each position.
        ref_logprobs:  ``(N, T)`` float — log π_ref at each position.
        rewards:       ``(N,)`` float — scalar reward per rollout.
        parent_idx:    ``(N,)`` long — index of each rollout's parent
            in this batch, or ``-1`` for roots.
    """

    tokens: Tensor
    response_mask: Tensor
    old_logprobs: Tensor
    ref_logprobs: Tensor
    rewards: Tensor
    parent_idx: Tensor


def compute_tree_advantage(
    rewards: Tensor,
    parent_idx: Tensor,
    *,
    eps: float = 1e-8,
    unbiased: bool = False,
) -> Tensor:
    """Sibling-group advantage normalization.

    Group rollouts by their ``parent_idx`` value (roots are pooled
    together as one synthetic group with parent ``-1``). Within each
    group, return ``(r - mean_g) / (std_g + eps)`` (default) or
    ``r - mean_g`` when ``unbiased=True`` (Dr.GRPO patch).

    Singleton groups (parent with only one child) get advantage 0 —
    no relative signal is available, so the rollout doesn't push or
    pull the policy on the cross-group axis.

    Args:
        rewards:    ``(N,)`` float scalar rewards.
        parent_idx: ``(N,)`` long parent indices; ``-1`` = root.
        eps:        numerical floor on the std denominator.
        unbiased:   skip per-group std normalization (Dr.GRPO).

    Returns:
        ``(N,)`` float advantages in rollout order.
    """
    if rewards.dim() != 1 or parent_idx.dim() != 1:
        raise ValueError(
            "rewards and parent_idx must be 1-D; got "
            f"{tuple(rewards.shape)}, {tuple(parent_idx.shape)}",
        )
    if rewards.shape != parent_idx.shape:
        raise ValueError(
            "rewards and parent_idx must have matching length; got "
            f"{rewards.shape[0]}, {parent_idx.shape[0]}",
        )

    advantages = torch.zeros_like(rewards)
    unique_parents = torch.unique(parent_idx)
    for p in unique_parents.tolist():
        rows = (parent_idx == p).nonzero(as_tuple=False).flatten()
        if rows.numel() == 1:
            # Singleton sibling group -> no relative signal.
            advantages[rows] = 0.0
            continue
        group_rewards = rewards[rows]
        mean = group_rewards.mean()
        if unbiased:
            advantages[rows] = group_rewards - mean
        else:
            std = group_rewards.std(unbiased=False)
            advantages[rows] = (group_rewards - mean) / (std + eps)
    return advantages


def tree_grpo_loss(
    logits: Tensor,
    batch: TreeRolloutBatch,
    *,
    cfg: GRPOConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Tree-structured analog of :func:`grpo_loss`.

    Identical surrogate + KL math; advantages come from
    :func:`compute_tree_advantage` instead of fixed-group-size
    :func:`compute_group_advantage`. All Q1 2026 patches in
    ``GRPOConfig`` (GSPO sequence-ratio, Clip-Higher, Token-Level PG,
    Dr.GRPO unbiased) are honored.

    Returns ``(loss, metrics)`` with the same keys as ``grpo_loss``,
    plus ``n_sibling_groups`` for diagnostics.
    """
    new_logprobs = gather_token_logprobs(logits, batch.tokens)
    advantages = compute_tree_advantage(
        batch.rewards,
        batch.parent_idx,
        eps=cfg.advantage_eps,
        unbiased=cfg.unbiased_loss,
    )

    mask = batch.response_mask.to(new_logprobs.dtype)
    adv_seq = advantages.to(new_logprobs.dtype)

    log_ratio = new_logprobs - batch.old_logprobs.to(new_logprobs.dtype)
    if cfg.importance_ratio_level == "sequence":
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
            "ratio_mean": (
                (ratio * mask).sum() / mask.sum().clamp(min=1.0)
            ).detach(),
            "n_sibling_groups": torch.tensor(
                int(torch.unique(batch.parent_idx).numel()),
                device=loss.device,
            ),
        }
    return loss, metrics
