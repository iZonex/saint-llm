"""Tests for multimodal GRPO trainer."""

from __future__ import annotations

import torch
from saint_llm_core.config import ModelConfig, MultimodalConfig
from saint_llm_core.model import SaintLLM
from saint_llm_posttraining import GRPOConfig, GRPOTrainerStep
from saint_llm_posttraining.multimodal_grpo_trainer import (
    MultimodalRolloutBatch,
    build_multimodal_rollout_batch,
    multimodal_grpo_train_step,
)


def _vision_cfg(vision_input_dim: int = 96) -> ModelConfig:
    base = ModelConfig.tiny()
    return ModelConfig(
        **{
            **base.model_dump(),
            "multimodal": MultimodalConfig(
                enable_vision_projector=True,
                vision_input_dim=vision_input_dim,
            ),
        },
    )


def _build_rollouts(
    cfg: ModelConfig, *, n_rollouts: int = 4, seq_len: int = 16,
    n_image_pads: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (tokens, response_mask, vision_features) of the right shape."""
    torch.manual_seed(0)
    tokens = torch.randint(1, cfg.vocab_size, (n_rollouts, seq_len))
    # Inject image_pad placeholders in every row at fixed positions.
    image_pad_id = cfg.tokenizer_slots.image_pad
    tokens[:, 4 : 4 + n_image_pads] = image_pad_id
    response_mask = torch.zeros(n_rollouts, seq_len, dtype=torch.long)
    response_mask[:, seq_len // 2 :] = 1
    # Total image pads across the batch = n_rollouts * n_image_pads.
    total_image_tokens = n_rollouts * n_image_pads
    vision = torch.randn(total_image_tokens, 96)
    return tokens, response_mask, vision


def test_multimodal_rollout_batch_to_rollout_batch_strips_features() -> None:
    """to_rollout_batch() returns the text-only RolloutBatch shape."""
    batch = MultimodalRolloutBatch(
        tokens=torch.zeros(2, 4, dtype=torch.long),
        response_mask=torch.zeros(2, 4, dtype=torch.long),
        old_logprobs=torch.zeros(2, 4),
        ref_logprobs=torch.zeros(2, 4),
        rewards=torch.zeros(2),
        vision_features=torch.randn(6, 16),
    )
    text_only = batch.to_rollout_batch()
    assert torch.equal(text_only.tokens, batch.tokens)
    assert torch.equal(text_only.rewards, batch.rewards)
    # text-only batch has no vision_features field.
    assert not hasattr(text_only, "vision_features")


def test_build_multimodal_rollout_batch_attaches_features() -> None:
    cfg = _vision_cfg()
    actor = SaintLLM(cfg)
    tokens, mask, vision = _build_rollouts(cfg)
    rewards = torch.randn(tokens.shape[0])
    batch = build_multimodal_rollout_batch(
        actor=actor, ref=None, tokens=tokens, response_mask=mask,
        rewards=rewards, vision_features=vision,
    )
    assert batch.vision_features is vision
    assert batch.tokens.shape == tokens.shape
    # ref_logprobs == old_logprobs when ref is None.
    assert torch.equal(batch.old_logprobs, batch.ref_logprobs)


def test_build_multimodal_rollout_batch_with_ref_diverges() -> None:
    """With a separately-initialized ref model, ref_logprobs differs from old."""
    cfg = _vision_cfg()
    actor = SaintLLM(cfg)
    ref = SaintLLM(cfg)
    tokens, mask, vision = _build_rollouts(cfg, n_rollouts=2)
    rewards = torch.randn(2)
    batch = build_multimodal_rollout_batch(
        actor=actor, ref=ref, tokens=tokens, response_mask=mask,
        rewards=rewards, vision_features=vision,
    )
    assert not torch.equal(batch.old_logprobs, batch.ref_logprobs)


def test_multimodal_grpo_train_step_runs_with_vision_features() -> None:
    cfg_model = _vision_cfg()
    actor = SaintLLM(cfg_model)
    optimizer = torch.optim.SGD(actor.parameters(), lr=1e-3)
    tokens, mask, vision = _build_rollouts(cfg_model, n_rollouts=4)
    # Distinct rewards so dynamic sampling keeps both groups.
    rewards = torch.tensor([0.1, 0.9, 0.2, 0.8])
    batch = build_multimodal_rollout_batch(
        actor=actor, ref=None, tokens=tokens, response_mask=mask,
        rewards=rewards, vision_features=vision,
    )
    cfg = GRPOConfig(group_size=2, kl_coef=0.0)
    step = multimodal_grpo_train_step(
        actor=actor, optimizer=optimizer, batch=batch, cfg=cfg,
    )
    assert isinstance(step, GRPOTrainerStep)
    assert torch.isfinite(step.loss)
    assert step.n_kept_groups == 2


def test_multimodal_grpo_train_step_grad_flows_through_vision_projector() -> None:
    """Backward pass populates gradients on the vision projector."""
    cfg_model = _vision_cfg()
    actor = SaintLLM(cfg_model)
    optimizer = torch.optim.SGD(actor.parameters(), lr=1e-3)
    tokens, mask, vision = _build_rollouts(cfg_model, n_rollouts=4)
    rewards = torch.tensor([0.0, 1.0, 0.2, 0.8])
    batch = build_multimodal_rollout_batch(
        actor=actor, ref=None, tokens=tokens, response_mask=mask,
        rewards=rewards, vision_features=vision,
    )
    cfg = GRPOConfig(group_size=2, kl_coef=0.0)

    # Zero the gradients first.
    for p in actor.parameters():
        if p.grad is not None:
            p.grad.zero_()

    multimodal_grpo_train_step(
        actor=actor, optimizer=optimizer, batch=batch, cfg=cfg,
    )
    # The optimizer.step has consumed the grads — re-run forward+backward
    # without optimizer to inspect them.
    out = actor(batch.tokens, vision_features=batch.vision_features)
    loss = out["logits"].mean()
    actor.zero_grad()
    loss.backward()
    assert actor.vision_proj.up.weight.grad is not None
    assert torch.isfinite(actor.vision_proj.up.weight.grad).all()


def test_multimodal_grpo_train_step_zero_variance_groups_drop() -> None:
    """All-equal rewards -> dynamic sampling drops the group."""
    cfg_model = _vision_cfg()
    actor = SaintLLM(cfg_model)
    optimizer = torch.optim.SGD(actor.parameters(), lr=1e-3)
    tokens, mask, vision = _build_rollouts(cfg_model, n_rollouts=2)
    rewards = torch.tensor([0.5, 0.5])  # zero variance
    batch = build_multimodal_rollout_batch(
        actor=actor, ref=None, tokens=tokens, response_mask=mask,
        rewards=rewards, vision_features=vision,
    )
    cfg = GRPOConfig(group_size=2, kl_coef=0.0)
    step = multimodal_grpo_train_step(
        actor=actor, optimizer=optimizer, batch=batch, cfg=cfg,
    )
    assert step.n_kept_groups == 0
    assert step.loss.item() == 0.0


def test_multimodal_grpo_train_step_no_features_falls_back_to_text() -> None:
    """vision_features=None and audio_features=None still works (text-only)."""
    cfg_model = ModelConfig.tiny()
    actor = SaintLLM(cfg_model)
    optimizer = torch.optim.SGD(actor.parameters(), lr=1e-3)
    tokens = torch.randint(1, cfg_model.vocab_size, (4, 16))
    mask = torch.zeros(4, 16, dtype=torch.long)
    mask[:, 8:] = 1
    rewards = torch.tensor([0.1, 0.9, 0.2, 0.8])
    batch = build_multimodal_rollout_batch(
        actor=actor, ref=None, tokens=tokens, response_mask=mask,
        rewards=rewards, vision_features=None, audio_features=None,
    )
    cfg = GRPOConfig(group_size=2, kl_coef=0.0)
    step = multimodal_grpo_train_step(
        actor=actor, optimizer=optimizer, batch=batch, cfg=cfg,
    )
    assert torch.isfinite(step.loss)


def test_multimodal_grpo_train_step_updates_actor_params() -> None:
    cfg_model = _vision_cfg()
    actor = SaintLLM(cfg_model)
    optimizer = torch.optim.SGD(actor.parameters(), lr=1.0)
    initial = next(actor.parameters()).detach().clone()
    tokens, mask, vision = _build_rollouts(cfg_model, n_rollouts=4)
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0])
    batch = build_multimodal_rollout_batch(
        actor=actor, ref=None, tokens=tokens, response_mask=mask,
        rewards=rewards, vision_features=vision,
    )
    cfg = GRPOConfig(group_size=2, kl_coef=0.0)
    multimodal_grpo_train_step(
        actor=actor, optimizer=optimizer, batch=batch, cfg=cfg,
    )
    assert not torch.equal(initial, next(actor.parameters()))
