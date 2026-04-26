"""SpecialistPipeline — high-level SFT + GRPO driver.

The post-training package ships a full algorithmic stack (SFT loss,
GRPO step, rollout generation, reward fns, ...) but the higher-level
driver that ties them into one orchestrator was missing. This module
provides :class:`SpecialistPipeline` — a thin façade that exposes two
methods:

* :meth:`SpecialistPipeline.sft_step` — single SFT update on a packed
  batch.
* :meth:`SpecialistPipeline.rl_step` — single GRPO update from a list
  of prompt strings + a reward function. Internally:
  generate rollouts, decode, score, build the rollout batch, and call
  :func:`grpo_train_step`.

Both methods share the same actor + optimizer, so a real specialist
recipe alternates phases::

    pipeline = SpecialistPipeline(actor=..., ref=..., optimizer=..., tokenizer=...)
    for batch in sft_loader:
        pipeline.sft_step(batch)
    for prompts in rl_prompt_loader:
        pipeline.rl_step(prompts, reward_fn=my_reward)

Loss curves, group rewards, and KL traces are returned by each method
for the caller to log. The pipeline doesn't own dataloaders, schedule,
or wandb plumbing — those stay in the experiment script.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from saint_llm_posttraining.grpo import GRPOConfig
from saint_llm_posttraining.grpo_trainer import (
    GRPOTrainerStep,
    RewardFn,
    build_rollout_batch,
    grpo_train_step,
    score_rollouts,
)
from saint_llm_posttraining.rollout_generator import (
    RolloutGenConfig,
    decode_rollouts_for_reward,
    generate_grpo_rollouts,
)
from saint_llm_posttraining.sft import (
    SFTPackedBatch,
    sft_cross_entropy,
)


@dataclass
class SFTStepOutput:
    """Result of one :meth:`SpecialistPipeline.sft_step`."""

    loss: Tensor
    n_response_tokens: int


@dataclass
class RLStepOutput:
    """Result of one :meth:`SpecialistPipeline.rl_step`."""

    train_step: GRPOTrainerStep
    rewards: Tensor
    n_rollouts: int


class SpecialistPipeline:
    """Tie actor + optimizer + reward fn into a runnable specialist trainer.

    Args:
        actor:        the model being trained.
        ref:          frozen reference policy for the GRPO KL term.
            ``None`` zeros the KL (warmup / ablation).
        optimizer:    optimizer over ``actor.parameters()``.
        encode:       callable ``str -> list[int]`` for prompts. Used
            by :meth:`rl_step` to tokenize prompts and by
            :meth:`decode` round-tripping. Typically
            ``tokenizer.encode``.
        decode:       callable ``list[int] -> str`` for completions.
            Used to feed the reward function.
        eos_token:    optional EOS token ID used by the rollout
            generator's response masking.
    """

    def __init__(
        self,
        *,
        actor: torch.nn.Module,
        ref: torch.nn.Module | None,
        optimizer: torch.optim.Optimizer,
        encode: Callable[[str], Sequence[int]],
        decode: Callable[[Sequence[int]], str],
        eos_token: int | None = None,
    ) -> None:
        self.actor = actor
        self.ref = ref
        self.optimizer = optimizer
        self._encode = encode
        self._decode = decode
        self._eos = eos_token

    # ----- SFT phase ------------------------------------------------

    def sft_step(self, batch: SFTPackedBatch) -> SFTStepOutput:
        """One forward + backward + optimizer step on a packed SFT batch."""
        self.actor.train()
        out = self.actor(batch.tokens)
        loss = sft_cross_entropy(out, batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        n_active = int(batch.loss_mask[:, 1:].sum().item())
        return SFTStepOutput(loss=loss.detach(), n_response_tokens=n_active)

    # ----- RL phase -------------------------------------------------

    def rl_step(
        self,
        prompts: Sequence[str],
        *,
        reward_fn: RewardFn,
        cfg: GRPOConfig,
        rollout_cfg: RolloutGenConfig,
    ) -> RLStepOutput:
        """One GRPO update from prompt strings + a reward function.

        Steps:
        1. Encode prompts to token IDs (right-pad to common length).
        2. Sample G completions per prompt via :func:`generate_grpo_rollouts`.
        3. Decode the response slice of each completion.
        4. Score with ``reward_fn``.
        5. Build a :class:`RolloutBatch` and call :func:`grpo_train_step`.

        Args:
            prompts:      sequence of prompt strings (one per example;
                G completions are sampled per prompt).
            reward_fn:    callable ``(prompt, completion) -> float``.
            cfg:          :class:`GRPOConfig` for the loss math.
            rollout_cfg:  :class:`RolloutGenConfig` for sampling.

        Returns:
            :class:`RLStepOutput` with the trainer-step metrics.
        """
        if not prompts:
            raise ValueError("prompts is empty; nothing to train on")
        if rollout_cfg.group_size != cfg.group_size:
            raise ValueError(
                f"rollout_cfg.group_size ({rollout_cfg.group_size}) must "
                f"match cfg.group_size ({cfg.group_size})",
            )

        prompt_ids = self._encode_prompts(prompts)
        # Override the rollout generator's eos_token with the pipeline's
        # default if the rollout cfg didn't set one.
        effective_rollout_cfg = (
            rollout_cfg if rollout_cfg.eos_token is not None or self._eos is None
            else _replace_eos(rollout_cfg, self._eos)
        )

        tokens, response_mask = generate_grpo_rollouts(
            self.actor, prompt_ids, effective_rollout_cfg,
        )
        prompt_len = prompt_ids.shape[1]
        completions = decode_rollouts_for_reward(
            tokens, self._decode, prompt_len=prompt_len,
        )
        # Replicate prompt strings across the group axis so the parallel
        # list matches the rollout layout (B*G).
        prompts_replicated: list[str] = []
        for p in prompts:
            prompts_replicated.extend([p] * effective_rollout_cfg.group_size)

        rewards = score_rollouts(
            prompts=prompts_replicated,
            completions=completions,
            reward_fn=reward_fn,
        ).to(tokens.device)

        batch = build_rollout_batch(
            actor=self.actor,
            ref=self.ref,
            tokens=tokens,
            response_mask=response_mask,
            rewards=rewards,
        )
        train_step = grpo_train_step(
            actor=self.actor, optimizer=self.optimizer, batch=batch, cfg=cfg,
        )
        return RLStepOutput(
            train_step=train_step,
            rewards=rewards.detach(),
            n_rollouts=tokens.shape[0],
        )

    def _encode_prompts(self, prompts: Sequence[str]) -> Tensor:
        """Encode + right-pad prompt strings to a common length."""
        ids = [list(self._encode(p)) for p in prompts]
        if not ids or any(not row for row in ids):
            raise ValueError("at least one encoded prompt is empty")
        max_len = max(len(row) for row in ids)
        pad_id = 0  # The model's embedding at pad_id is whatever the
        # caller's tokenizer maps to; we right-pad with 0 by convention
        # since the prompt mask will keep these out of the loss anyway.
        device = next(self.actor.parameters()).device
        out = torch.full(
            (len(ids), max_len), pad_id, dtype=torch.long, device=device,
        )
        for i, row in enumerate(ids):
            out[i, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        return out


def _replace_eos(cfg: RolloutGenConfig, eos: int) -> RolloutGenConfig:
    """Return a copy of ``cfg`` with ``eos_token`` set."""
    return RolloutGenConfig(
        group_size=cfg.group_size,
        max_new_tokens=cfg.max_new_tokens,
        sampler=cfg.sampler,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        eos_token=eos,
        pad_token=cfg.pad_token,
        seed=cfg.seed,
    )
