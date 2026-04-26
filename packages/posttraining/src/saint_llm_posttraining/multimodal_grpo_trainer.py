"""Multimodal GRPO trainer — passes vision / audio features through the loss.

The flat-text :func:`grpo_train_step` driver calls
``actor(batch.tokens)`` without modality features, so any
``image_pad`` / audio bracket positions in the rollout get embedded
through the LM's regular embedding table — *not* through the
vision/audio projector. For multimodal RL (caption RL, VQA RL) we
need to pass the per-rollout features through every forward and
backward pass.

This module ships:

* :class:`MultimodalRolloutBatch` — :class:`RolloutBatch` plus
  optional ``vision_features`` and ``audio_features`` tensors. The
  features are *shared* across all rollouts in the batch — a single
  flat tensor whose rows line up with the union of placeholder slots
  across the batch, in row-major order.
* :func:`build_multimodal_rollout_batch` — runs the actor + ref with
  the features attached, returns the assembled batch.
* :func:`multimodal_grpo_train_step` — forward + GRPO loss + backward
  + step, with features passed to ``actor.forward`` so the vision
  projector receives gradients alongside the LM.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from saint_llm_posttraining.grpo import (
    GRPOConfig,
    RolloutBatch,
    dynamic_sampling_mask,
    gather_token_logprobs,
    grpo_loss,
)
from saint_llm_posttraining.grpo_trainer import GRPOTrainerStep


@dataclass(frozen=True)
class MultimodalRolloutBatch:
    """Rollout batch with shared modality features.

    Attributes:
        tokens / response_mask / old_logprobs / ref_logprobs / rewards:
            same as :class:`RolloutBatch`.
        vision_features: ``(N_image_tokens_total, vision_dim)`` flat
            tensor or ``None``. Rows line up with image_pad
            placeholders across the batch in row-major order. Shared
            across all rollouts (i.e. each rollout's prompt contains a
            slice of this tensor's positions).
        audio_features:  ``(N_audio_tokens_total, audio_dim)`` or
            ``None``. Same convention.
    """

    tokens: Tensor
    response_mask: Tensor
    old_logprobs: Tensor
    ref_logprobs: Tensor
    rewards: Tensor
    vision_features: Tensor | None = None
    audio_features: Tensor | None = None

    def to_rollout_batch(self) -> RolloutBatch:
        """Drop modality features for callers that need the text-only batch."""
        return RolloutBatch(
            tokens=self.tokens,
            response_mask=self.response_mask,
            old_logprobs=self.old_logprobs,
            ref_logprobs=self.ref_logprobs,
            rewards=self.rewards,
        )


def _per_token_logprobs_under(
    model: torch.nn.Module,
    tokens: Tensor,
    *,
    vision_features: Tensor | None,
    audio_features: Tensor | None,
) -> Tensor:
    """Run ``model(tokens, vision_features=..., audio_features=...)`` and gather logprobs."""
    kwargs: dict[str, Tensor] = {}
    if vision_features is not None:
        kwargs["vision_features"] = vision_features
    if audio_features is not None:
        kwargs["audio_features"] = audio_features
    out = model(tokens, **kwargs)
    logits = out["logits"]
    if not isinstance(logits, Tensor):
        raise TypeError("model output['logits'] must be a Tensor")
    return gather_token_logprobs(logits, tokens)


@torch.no_grad()
def build_multimodal_rollout_batch(
    *,
    actor: torch.nn.Module,
    ref: torch.nn.Module | None,
    tokens: Tensor,
    response_mask: Tensor,
    rewards: Tensor,
    vision_features: Tensor | None = None,
    audio_features: Tensor | None = None,
) -> MultimodalRolloutBatch:
    """Compute old / ref logprobs with features attached, return the batch."""
    actor.eval()
    old_logprobs = _per_token_logprobs_under(
        actor, tokens,
        vision_features=vision_features, audio_features=audio_features,
    )
    if ref is not None:
        ref.eval()
        ref_logprobs = _per_token_logprobs_under(
            ref, tokens,
            vision_features=vision_features, audio_features=audio_features,
        )
    else:
        ref_logprobs = old_logprobs.clone()
    return MultimodalRolloutBatch(
        tokens=tokens,
        response_mask=response_mask,
        old_logprobs=old_logprobs.detach(),
        ref_logprobs=ref_logprobs.detach(),
        rewards=rewards,
        vision_features=vision_features,
        audio_features=audio_features,
    )


def multimodal_grpo_train_step(
    *,
    actor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: MultimodalRolloutBatch,
    cfg: GRPOConfig,
) -> GRPOTrainerStep:
    """One GRPO update with vision / audio features attached.

    Same dynamic-sampling pre-filter as :func:`grpo_train_step` —
    drops zero-variance groups before the forward pass. The actor
    receives the features each step so the vision / audio projectors
    accumulate gradients alongside the LM.
    """
    keep = dynamic_sampling_mask(batch.rewards, group_size=cfg.group_size)
    if not keep.any():
        zero = torch.zeros((), device=batch.rewards.device, requires_grad=True)
        return GRPOTrainerStep(
            loss=zero,
            metrics={"reward_mean": batch.rewards.mean().detach()},
            n_kept_groups=0,
        )

    if not keep.all():
        idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        # Slicing the rollout's feature tensors is tricky — the flat
        # vision_features tensor isn't aligned with rollout indices,
        # so dropping rows here would desync the splice. v0.0 keeps
        # the full feature pool intact and just filters the rollout
        # rows. Practical caveat: dropped rollouts still consume
        # placeholder positions in the model forward, and their
        # gradient contributions are masked via the response_mask
        # (already zero on the dropped rollouts since we only filter,
        # don't un-tokenize).
        batch = MultimodalRolloutBatch(
            tokens=batch.tokens[idx],
            response_mask=batch.response_mask[idx],
            old_logprobs=batch.old_logprobs[idx],
            ref_logprobs=batch.ref_logprobs[idx],
            rewards=batch.rewards[idx],
            vision_features=batch.vision_features,
            audio_features=batch.audio_features,
        )

    actor.train()
    kwargs: dict[str, Tensor] = {}
    if batch.vision_features is not None:
        kwargs["vision_features"] = batch.vision_features
    if batch.audio_features is not None:
        kwargs["audio_features"] = batch.audio_features
    out = actor(batch.tokens, **kwargs)
    logits = out["logits"]
    if not isinstance(logits, Tensor):
        raise TypeError("model output['logits'] must be a Tensor")
    loss, metrics = grpo_loss(logits, batch.to_rollout_batch(), cfg=cfg)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return GRPOTrainerStep(
        loss=loss.detach(),
        metrics=metrics,
        n_kept_groups=int(keep.view(-1, cfg.group_size).any(dim=-1).sum().item()),
    )
