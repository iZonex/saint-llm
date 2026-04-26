"""GRPO trainer — rollout generation + reward + reference KL.

Bridges the math layer (:mod:`grpo`) with a concrete training step:

1. Sample G completions per prompt from the **actor** (the model being
   trained).
2. Score each completion via a user-supplied :class:`RewardFn`.
3. Compute per-token log-probs under the actor (frozen snapshot ==
   ``π_old``) and the **reference policy** (typically the SFT init).
4. Build a :class:`RolloutBatch` and call :func:`grpo_loss`.
5. Backprop + step the actor's optimizer.

Rollout generation uses the existing greedy/top-p sampler; the
sampler's KV-cache path makes per-step inference cheap. Reward
functions are user-supplied callables ``(prompt, completion) -> float``
so domain-specific scorers (math correctness, code execution,
GRM judge) plug in directly.

The rollout/snapshot orchestration is intentionally minimal — this is
a v0.0 driver. Production GRPO drivers add async rollout pools,
distributed reward scoring, and reference-policy offload (Workstream
F territory).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from saint_llm_posttraining.grpo import (
    GRPOConfig,
    RolloutBatch,
    dynamic_sampling_mask,
    gather_token_logprobs,
    grpo_loss,
)


class RewardFn(Protocol):
    """Score a completion against its prompt."""

    def __call__(self, prompt: str, completion: str) -> float: ...


@dataclass
class GRPOTrainerStep:
    """One GRPO update on a batch of prompts.

    Fields are mostly metrics for logging; ``loss`` is the scalar that
    was backpropped.
    """

    loss: Tensor
    metrics: dict[str, Tensor]
    n_kept_groups: int


def _per_token_logprobs_under(
    model: torch.nn.Module,
    tokens: Tensor,
) -> Tensor:
    """Compute log-probs under ``model`` for each position ``t >= 1``.

    Returns ``(B, T)`` aligned with :func:`grpo.gather_token_logprobs`
    so position 0 is filled with zero.
    """
    out = model(tokens)
    logits = out["logits"]
    if not isinstance(logits, Tensor):
        raise TypeError("model output['logits'] must be a Tensor")
    return gather_token_logprobs(logits, tokens)


@torch.no_grad()
def build_rollout_batch(
    *,
    actor: torch.nn.Module,
    ref: torch.nn.Module | None,
    tokens: Tensor,
    response_mask: Tensor,
    rewards: Tensor,
) -> RolloutBatch:
    """Compute old / ref logprobs and assemble a :class:`RolloutBatch`.

    Caller has already done rollout generation (sampling completions)
    and reward scoring; this helper wires the per-token logprob
    extractions for the loss.

    ``ref=None`` means "no reference policy" — ``ref_logprobs`` are
    set equal to ``old_logprobs`` so the KL term is exactly zero.
    Useful for early SFT-style RL warmup or ablation.
    """
    actor.eval()
    old_logprobs = _per_token_logprobs_under(actor, tokens)
    if ref is not None:
        ref.eval()
        ref_logprobs = _per_token_logprobs_under(ref, tokens)
    else:
        ref_logprobs = old_logprobs.clone()
    return RolloutBatch(
        tokens=tokens,
        response_mask=response_mask,
        old_logprobs=old_logprobs.detach(),
        ref_logprobs=ref_logprobs.detach(),
        rewards=rewards,
    )


def grpo_train_step(
    *,
    actor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    cfg: GRPOConfig,
) -> GRPOTrainerStep:
    """One GRPO update: forward + loss + backward + optimizer.step.

    DAPO Dynamic Sampling (when ``cfg`` has it via the math layer's
    in-loss patches) is applied via a pre-filter on the rollout
    rewards: groups with zero variance are dropped before the loss.
    The trainer driver is responsible for re-sampling rollouts to
    refill the batch when groups are dropped — this helper just
    reports ``n_kept_groups`` so the caller can track it.

    Returns a :class:`GRPOTrainerStep` with the scalar loss and the
    metrics dict for logging.
    """
    keep = dynamic_sampling_mask(batch.rewards, group_size=cfg.group_size)
    if not keep.any():
        # Pathological batch: no usable groups. Return zero loss; the
        # caller's training loop should pick a different rollout batch.
        zero = torch.zeros((), device=batch.rewards.device, requires_grad=True)
        return GRPOTrainerStep(
            loss=zero,
            metrics={"reward_mean": batch.rewards.mean().detach()},
            n_kept_groups=0,
        )

    if not keep.all():
        # Filter rollouts in-place by index selection.
        idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        batch = RolloutBatch(
            tokens=batch.tokens[idx],
            response_mask=batch.response_mask[idx],
            old_logprobs=batch.old_logprobs[idx],
            ref_logprobs=batch.ref_logprobs[idx],
            rewards=batch.rewards[idx],
        )

    actor.train()
    out = actor(batch.tokens)
    logits = out["logits"]
    if not isinstance(logits, Tensor):
        raise TypeError("model output['logits'] must be a Tensor")
    loss, metrics = grpo_loss(logits, batch, cfg=cfg)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return GRPOTrainerStep(
        loss=loss.detach(),
        metrics=metrics,
        n_kept_groups=int(keep.view(-1, cfg.group_size).any(dim=-1).sum().item()),
    )


def score_rollouts(
    *,
    prompts: list[str],
    completions: list[str],
    reward_fn: RewardFn,
) -> Tensor:
    """Apply a reward function over a parallel list of (prompt, completion).

    Returns a ``(B*G,)`` float tensor on CPU. Caller moves to device.
    """
    if len(prompts) != len(completions):
        raise ValueError(
            f"prompts/completions length mismatch: {len(prompts)} vs {len(completions)}",
        )
    return torch.tensor(
        [reward_fn(p, c) for p, c in zip(prompts, completions, strict=True)],
        dtype=torch.float32,
    )
