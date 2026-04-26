"""Critique-GRPO — natural-language critique inside the rollout loop (RL-04, ADR-0016).

Extends standard GRPO with a critique-then-refine step before final
rollout scoring. Mechanism:

1. Generate G rollouts as in standard GRPO.
2. For each rollout below ``critique_threshold`` reward, ask a critic
   to explain in natural language *why* it failed.
3. Generate a refined rollout conditioned on the original prompt +
   critique. The refined rollout replaces the failed one before
   advantage computation and the standard GRPO loss.

Reported gains on Qwen2.5-Math-7B / Qwen3-8B: +15-22% Pass@1 over
GRPO on hard reasoning. Breaks the GRPO plateau where outcome-only
rewards can't distinguish correct reasoning from lucky tokens.

This module ships the *math + orchestration helpers*:

* :class:`CritiqueGRPOConfig` — knobs (threshold, attempts, critic
  identity).
* :class:`CritiqueRequest` — bookkeeping for one rollout-needing-critique.
* :func:`needs_critique_mask` — per-rollout filter mask (reward below
  threshold).
* :func:`make_critique_prompt` / :func:`make_refinement_prompt` —
  deterministic string builders for the critic / refined-rollout
  prompts. Default templates match the ADR-0016 starting point and can
  be overridden per-domain.

The actual policy.act() / sampling loop lives in the trainer driver
(Workstream B). This module is policy-agnostic.

Reference: arXiv 2506.03106 (Critique-GRPO).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from saint_llm_posttraining.grpo import GRPOConfig


@dataclass(frozen=True)
class CritiqueGRPOConfig:
    """Critique-GRPO orchestration knobs on top of vanilla GRPO.

    Attributes:
        grpo:                  base GRPO config (loss, clip, KL, group
            size, all the patches from ADR-0014..0017).
        enabled:               master switch. ``False`` reproduces
            standard GRPO with no critique step.
        critique_threshold:    reward below this triggers critique.
            Recipe value: 0.5 for binary correctness rewards. Set
            higher to be more selective.
        critique_max_attempts: cap on the refinement loop. ``1`` is
            the published recipe (single critique → single refined
            rollout). Higher allows recursive critique.
        critic_self_play:      when True, the actor model itself does
            critique (cheaper but risks blind-spot loops). When False,
            a separate ``critic_policy`` (in the trainer driver) is
            expected.
    """

    grpo: GRPOConfig
    enabled: bool = False
    critique_threshold: float = 0.5
    critique_max_attempts: int = 1
    critic_self_play: bool = True


@dataclass(frozen=True)
class CritiqueRequest:
    """One rollout flagged for critique-then-refine.

    Attributes:
        rollout_idx:  index into the rollout batch (B*G layout).
        original_prompt: text seen by the actor.
        failed_completion: the actor's failed rollout text.
        reward: scalar reward the rollout received.
    """

    rollout_idx: int
    original_prompt: str
    failed_completion: str
    reward: float


# Default critique prompt template — keeps the critic's response short
# so the refined rollout has room within max_length. Override per
# domain (Math gets formal-error-language; Code gets execution
# diagnostics; Agent gets tool-usage critique).
DEFAULT_CRITIQUE_TEMPLATE = (
    "The following completion received a low reward. Diagnose the "
    "specific reasoning error and explain what should have been done "
    "differently. Be brief — 2-3 sentences.\n\n"
    "Problem: {prompt}\n"
    "Failed completion: {completion}\n"
    "Reward: {reward}\n\n"
    "Critique:"
)


DEFAULT_REFINEMENT_TEMPLATE = (
    "{prompt}\n\n"
    "A previous attempt failed because: {critique}\n\n"
    "Try again, addressing the issue:"
)


def needs_critique_mask(
    rewards: Tensor,
    *,
    threshold: float,
) -> Tensor:
    """Return a ``(B*G,)`` bool mask: True for rollouts below threshold.

    Used by the trainer driver to pick which completions go to the
    critic instead of straight to advantage computation.
    """
    return rewards < threshold


def make_critique_prompt(
    *,
    prompt: str,
    failed_completion: str,
    reward: float,
    template: str = DEFAULT_CRITIQUE_TEMPLATE,
) -> str:
    """Build the critic's input prompt from a failed rollout.

    The default template asks for a concise diagnosis. Templates may
    use the placeholders ``{prompt}``, ``{completion}``, ``{reward}``.
    """
    return template.format(
        prompt=prompt,
        completion=failed_completion,
        reward=reward,
    )


def make_refinement_prompt(
    *,
    prompt: str,
    critique: str,
    template: str = DEFAULT_REFINEMENT_TEMPLATE,
) -> str:
    """Build the refined-rollout's input prompt.

    The default template appends the critique to the original prompt
    and asks for a retry. Templates may use the placeholders
    ``{prompt}``, ``{critique}``.
    """
    return template.format(prompt=prompt, critique=critique)


def collect_critique_requests(
    *,
    rewards: Tensor,
    prompts: list[str],
    completions: list[str],
    threshold: float,
) -> list[CritiqueRequest]:
    """Pick out rollouts below ``threshold`` and pair them with their
    prompt + completion text.

    The trainer driver typically holds prompts and completions as
    parallel Python lists alongside the tensor rewards; this helper
    keeps the indexing logic in one place. Returns the rollouts the
    critic should look at; the driver calls the critic + refined
    actor on each, then writes the new completions back into the
    same indices before the standard GRPO loss runs.
    """
    if len(prompts) != rewards.shape[0]:
        raise ValueError(
            f"prompts length {len(prompts)} != rewards length {rewards.shape[0]}",
        )
    if len(completions) != rewards.shape[0]:
        raise ValueError(
            f"completions length {len(completions)} != rewards length {rewards.shape[0]}",
        )
    mask = needs_critique_mask(rewards, threshold=threshold)
    indices = torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist()
    return [
        CritiqueRequest(
            rollout_idx=int(i),
            original_prompt=prompts[i],
            failed_completion=completions[i],
            reward=float(rewards[i].item()),
        )
        for i in indices
    ]
