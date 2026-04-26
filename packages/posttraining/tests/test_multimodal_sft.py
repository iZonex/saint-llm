"""Tests for multimodal SFT step + loss."""

from __future__ import annotations

import pytest
import torch
from saint_llm_core.config import ModelConfig, MultimodalConfig
from saint_llm_core.model import SaintLLM
from saint_llm_data import MultimodalPackedBatch
from saint_llm_posttraining import (
    MultimodalSFTOutput,
    multimodal_sft_loss,
    multimodal_sft_step,
)


def _vision_enabled_cfg(vision_input_dim: int = 96) -> ModelConfig:
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


def _build_batch(
    tokens: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    vision_features: torch.Tensor | None = None,
    audio_features: torch.Tensor | None = None,
) -> MultimodalPackedBatch:
    return MultimodalPackedBatch(
        tokens=tokens,
        loss_mask=loss_mask,
        segment_ids=torch.zeros_like(tokens),
        vision_features=vision_features,
        audio_features=audio_features,
    )


def test_multimodal_sft_loss_zero_when_no_active_positions() -> None:
    logits = torch.randn(1, 6, 100)
    batch = _build_batch(
        tokens=torch.zeros(1, 6, dtype=torch.long),
        loss_mask=torch.zeros(1, 6, dtype=torch.long),
    )
    loss, n_active = multimodal_sft_loss(logits, batch)
    assert loss.item() == 0.0
    assert n_active == 0


def test_multimodal_sft_loss_counts_only_target_positions() -> None:
    """Mask at position t+1 controls whether step t's loss counts."""
    vocab = 8
    tokens = torch.tensor([[0, 1, 2, 3, 4]])
    # Mask: 0 0 0 1 1 → CE at t=2 (target=3) and t=3 (target=4).
    loss_mask = torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.long)
    logits = torch.full((1, 5, vocab), -10.0)
    # Push CE to ~0 at the active positions: predict the right token.
    logits[0, 2, 3] = 10.0
    logits[0, 3, 4] = 10.0
    batch = _build_batch(tokens=tokens, loss_mask=loss_mask)
    loss, n_active = multimodal_sft_loss(logits, batch)
    assert n_active == 2
    assert loss.item() < 1e-3


def test_multimodal_sft_loss_rejects_non_3d_logits() -> None:
    batch = _build_batch(
        tokens=torch.zeros(1, 4, dtype=torch.long),
        loss_mask=torch.zeros(1, 4, dtype=torch.long),
    )
    with pytest.raises(ValueError, match=r"\(B, T, V\)"):
        multimodal_sft_loss(torch.randn(1, 4), batch)


def test_multimodal_sft_step_runs_with_vision_features() -> None:
    """Model forward + loss returns finite scalar with vision features."""
    torch.manual_seed(0)
    cfg = _vision_enabled_cfg()
    model = SaintLLM(cfg)
    model.train()

    t = 16
    n_image_pads = 3
    tokens = torch.randint(1, cfg.vocab_size, (1, t))
    tokens[0, 4 : 4 + n_image_pads] = cfg.tokenizer_slots.image_pad
    loss_mask = torch.zeros(1, t, dtype=torch.long)
    loss_mask[0, 8:] = 1  # response from position 8 onward
    vision = torch.randn(n_image_pads, 96)

    batch = _build_batch(tokens=tokens, loss_mask=loss_mask, vision_features=vision)
    result = multimodal_sft_step(model, batch)

    assert isinstance(result, MultimodalSFTOutput)
    assert result.loss.dim() == 0
    assert torch.isfinite(result.loss)
    assert result.n_response_tokens > 0
    assert result.logits_for_aux.shape == (1, t, cfg.vocab_size)


def test_multimodal_sft_step_grad_flows_through_vision_path() -> None:
    """Backward pass populates gradients on the vision projector."""
    torch.manual_seed(0)
    cfg = _vision_enabled_cfg()
    model = SaintLLM(cfg)
    model.train()

    t = 16
    n_image_pads = 4
    tokens = torch.randint(1, cfg.vocab_size, (1, t))
    tokens[0, 4 : 4 + n_image_pads] = cfg.tokenizer_slots.image_pad
    loss_mask = torch.zeros(1, t, dtype=torch.long)
    loss_mask[0, 9:] = 1
    vision = torch.randn(n_image_pads, 96)

    batch = _build_batch(tokens=tokens, loss_mask=loss_mask, vision_features=vision)
    out = multimodal_sft_step(model, batch)
    out.loss.backward()

    # The vision projector should have gradients populated.
    assert model.vision_proj.up.weight.grad is not None
    assert torch.isfinite(model.vision_proj.up.weight.grad).all()


def test_multimodal_sft_step_text_only_falls_back_to_plain_lm() -> None:
    """No vision/audio features -> step still works (model ignores splice)."""
    torch.manual_seed(0)
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    model.train()

    t = 16
    tokens = torch.randint(1, cfg.vocab_size, (1, t))
    loss_mask = torch.zeros(1, t, dtype=torch.long)
    loss_mask[0, 8:] = 1
    batch = _build_batch(tokens=tokens, loss_mask=loss_mask)

    result = multimodal_sft_step(model, batch)
    assert torch.isfinite(result.loss)
    assert result.n_response_tokens == int(loss_mask[0, 1:].sum().item())


def test_multimodal_sft_step_two_consecutive_calls_loss_decreases() -> None:
    """One optimizer step on the same batch should reduce the loss."""
    torch.manual_seed(0)
    cfg = _vision_enabled_cfg()
    model = SaintLLM(cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    t = 16
    n_image_pads = 2
    tokens = torch.randint(1, cfg.vocab_size, (1, t))
    tokens[0, 4 : 4 + n_image_pads] = cfg.tokenizer_slots.image_pad
    loss_mask = torch.zeros(1, t, dtype=torch.long)
    loss_mask[0, 8:] = 1
    vision = torch.randn(n_image_pads, 96)
    batch = _build_batch(tokens=tokens, loss_mask=loss_mask, vision_features=vision)

    loss_before = multimodal_sft_step(model, batch).loss
    opt.zero_grad()
    loss_before.backward()
    opt.step()
    loss_after = multimodal_sft_step(model, batch).loss
    assert loss_after.item() < loss_before.item()
