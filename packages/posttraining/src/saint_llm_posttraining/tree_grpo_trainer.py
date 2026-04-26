"""Tree-GRPO trainer — driver-side glue for tree-structured rollouts.

Mirrors :mod:`grpo_trainer` for :class:`TreeRolloutBatch`:

* :func:`build_tree_rollout_batch` — gathers ``old_logprobs`` /
  ``ref_logprobs`` per rollout, attaches the parent-index vector,
  returns a :class:`TreeRolloutBatch` ready for the loss.
* :func:`tree_grpo_train_step` — forward + :func:`tree_grpo_loss` +
  backward + optimizer step. Drops sibling groups whose rewards have
  zero variance (same intuition as DAPO Dynamic Sampling for flat
  groups) so we don't waste a backprop on a degenerate batch.

Sampling and reward scoring still live with the caller — Tree-GRPO is
agnostic about the search policy that produced the tree (MCTS, beam,
greedy expand) so we don't bake one in.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from saint_llm_posttraining.grpo import GRPOConfig, gather_token_logprobs
from saint_llm_posttraining.tree_grpo import (
    TreeRolloutBatch,
    tree_grpo_loss,
)


@dataclass
class TreeGRPOTrainerStep:
    """One Tree-GRPO update on a batch of tree-structured rollouts."""

    loss: Tensor
    metrics: dict[str, Tensor]
    n_kept_groups: int


def _per_token_logprobs_under(
    model: torch.nn.Module, tokens: Tensor,
) -> Tensor:
    out = model(tokens)
    logits = out["logits"]
    if not isinstance(logits, Tensor):
        raise TypeError("model output['logits'] must be a Tensor")
    return gather_token_logprobs(logits, tokens)


@torch.no_grad()
def build_tree_rollout_batch(
    *,
    actor: torch.nn.Module,
    ref: torch.nn.Module | None,
    tokens: Tensor,
    response_mask: Tensor,
    rewards: Tensor,
    parent_idx: Tensor,
) -> TreeRolloutBatch:
    """Compute old / ref logprobs and assemble a :class:`TreeRolloutBatch`.

    ``ref=None`` zeroes the KL term by setting ``ref_logprobs ==
    old_logprobs`` (same convention as :func:`build_rollout_batch`).
    """
    actor.eval()
    old_logprobs = _per_token_logprobs_under(actor, tokens)
    if ref is not None:
        ref.eval()
        ref_logprobs = _per_token_logprobs_under(ref, tokens)
    else:
        ref_logprobs = old_logprobs.clone()
    return TreeRolloutBatch(
        tokens=tokens,
        response_mask=response_mask,
        old_logprobs=old_logprobs.detach(),
        ref_logprobs=ref_logprobs.detach(),
        rewards=rewards,
        parent_idx=parent_idx,
    )


def tree_dynamic_sampling_mask(
    rewards: Tensor, parent_idx: Tensor,
) -> Tensor:
    """DAPO Dynamic Sampling adapted to variable-size sibling groups.

    Returns a 1-D bool mask: True for rollouts whose sibling group has
    non-zero reward variance. Singleton groups are kept (no signal but
    no harm — :func:`compute_tree_advantage` returns zero advantage so
    their PG term vanishes; we keep them in case the caller still
    wants the KL term to apply).
    """
    if rewards.shape != parent_idx.shape:
        raise ValueError("rewards and parent_idx must have matching shape")
    keep = torch.ones_like(rewards, dtype=torch.bool)
    for p in torch.unique(parent_idx).tolist():
        rows = (parent_idx == p).nonzero(as_tuple=False).flatten()
        if rows.numel() <= 1:
            continue
        if rewards[rows].var(unbiased=False).item() == 0.0:
            keep[rows] = False
    return keep


def tree_grpo_train_step(
    *,
    actor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TreeRolloutBatch,
    cfg: GRPOConfig,
) -> TreeGRPOTrainerStep:
    """One Tree-GRPO update: drop zero-variance sibling groups, then step.

    Returns a :class:`TreeGRPOTrainerStep` with the scalar loss, the
    metrics dict, and the count of distinct sibling groups that
    survived dynamic-sampling filtering.
    """
    keep = tree_dynamic_sampling_mask(batch.rewards, batch.parent_idx)
    if not keep.any():
        zero = torch.zeros((), device=batch.rewards.device, requires_grad=True)
        return TreeGRPOTrainerStep(
            loss=zero,
            metrics={"reward_mean": batch.rewards.mean().detach()},
            n_kept_groups=0,
        )

    if not keep.all():
        idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        batch = TreeRolloutBatch(
            tokens=batch.tokens[idx],
            response_mask=batch.response_mask[idx],
            old_logprobs=batch.old_logprobs[idx],
            ref_logprobs=batch.ref_logprobs[idx],
            rewards=batch.rewards[idx],
            parent_idx=batch.parent_idx[idx],
        )

    actor.train()
    out = actor(batch.tokens)
    logits = out["logits"]
    if not isinstance(logits, Tensor):
        raise TypeError("model output['logits'] must be a Tensor")
    loss, metrics = tree_grpo_loss(logits, batch, cfg=cfg)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return TreeGRPOTrainerStep(
        loss=loss.detach(),
        metrics=metrics,
        n_kept_groups=int(torch.unique(batch.parent_idx).numel()),
    )
