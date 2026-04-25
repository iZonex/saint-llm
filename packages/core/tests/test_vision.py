"""Vision encoder + DeepStack fusion + end-to-end SaintLLM-with-vision integration."""

from __future__ import annotations

import pytest
import torch
from saint_llm_core import ModelConfig, SaintLLM
from saint_llm_core.config import MultimodalConfig
from saint_llm_core.multimodal import (
    FakeViT,
    VisionEncoderConfig,
    VisionTokenizer,
    deepstack_fuse,
)


def test_vision_encoder_config_output_dim() -> None:
    cfg = VisionEncoderConfig(encoder_hidden_dim=1152, deepstack_layers=(-1, -5, -9), deepstack_enabled=True)
    assert cfg.output_dim == 3 * 1152

    cfg2 = VisionEncoderConfig(encoder_hidden_dim=1152, deepstack_enabled=False)
    assert cfg2.output_dim == 1152


def test_fake_vit_output_shapes() -> None:
    cfg = VisionEncoderConfig(encoder_hidden_dim=64, image_size=28, patch_size=14)
    fake = FakeViT(cfg, n_layers=4)
    pixel_values = torch.randn(2, 3, 28, 28)
    out = fake(pixel_values)
    n_patches = (28 // 14) ** 2
    assert out["last_hidden_state"].shape == (2, n_patches, 64)
    assert len(out["hidden_states"]) == 5  # n_layers + 1 (embedding)


def test_deepstack_fuse_concat_dim() -> None:
    cfg = VisionEncoderConfig(encoder_hidden_dim=32, image_size=28, patch_size=14, deepstack_layers=(-1, -3))
    fake = FakeViT(cfg, n_layers=8)
    out = fake(torch.randn(1, 3, 28, 28))
    fused = deepstack_fuse(out["hidden_states"], cfg.deepstack_layers)
    n_patches = (28 // 14) ** 2
    assert fused.shape == (1, n_patches, 64)  # 2 layers * 32 dim


def test_vision_tokenizer_wraps_fake_vit() -> None:
    cfg = VisionEncoderConfig(encoder_hidden_dim=32, image_size=28, patch_size=14, deepstack_layers=(-1, -2, -3))
    fake = FakeViT(cfg, n_layers=8)
    tokenizer = VisionTokenizer(cfg, fake)
    out = tokenizer(torch.randn(2, 3, 28, 28))
    n_patches = (28 // 14) ** 2
    assert out.shape == (2, n_patches, cfg.output_dim)
    assert cfg.output_dim == 96  # 3 * 32


def test_saintllm_with_vision_features_forward() -> None:
    """End-to-end: tokens with <|image_pad|> + vision_features replaces them in residual stream."""
    base = ModelConfig.tiny()
    cfg = ModelConfig(
        **{**base.model_dump(), "multimodal": MultimodalConfig(
            enable_vision_projector=True,
            vision_input_dim=96,
        )},
    )
    model = SaintLLM(cfg)
    model.eval()

    t = 16
    n_image_tokens = 4
    token_ids = torch.randint(0, cfg.vocab_size, (1, t))
    token_ids[0, 4:8] = cfg.tokenizer_slots.image_pad
    assert (token_ids == cfg.tokenizer_slots.image_pad).sum().item() == n_image_tokens

    vision_features = torch.randn(n_image_tokens, 96)
    with torch.no_grad():
        out = model(token_ids, vision_features=vision_features)
    assert out["logits"].shape == (1, t, cfg.vocab_size)
    assert torch.isfinite(out["logits"]).all()


def test_saintllm_vision_features_size_mismatch_raises() -> None:
    base = ModelConfig.tiny()
    cfg = ModelConfig(
        **{**base.model_dump(), "multimodal": MultimodalConfig(
            enable_vision_projector=True,
            vision_input_dim=96,
        )},
    )
    model = SaintLLM(cfg)
    model.eval()

    t = 16
    token_ids = torch.randint(0, cfg.vocab_size, (1, t))
    token_ids[0, 4:8] = cfg.tokenizer_slots.image_pad
    wrong_features = torch.randn(7, 96)  # need 4, not 7
    with pytest.raises(ValueError, match="must match"):
        model(token_ids, vision_features=wrong_features)


def test_saintllm_no_vision_features_when_disabled() -> None:
    """If vision projector disabled, vision_features are silently ignored — text path unaffected."""
    cfg = ModelConfig.tiny()
    assert cfg.multimodal.enable_vision_projector is False
    model = SaintLLM(cfg)
    model.eval()
    t = 16
    token_ids = torch.randint(0, cfg.vocab_size, (2, t))
    with torch.no_grad():
        out = model(token_ids, vision_features=torch.randn(99, 1152))
    assert out["logits"].shape == (2, t, cfg.vocab_size)


def test_vision_tokens_carry_through_is_visual_mask() -> None:
    """When token IDs include <|image_pad|>, is_visual mask gets auto-derived for CSA indexer."""
    base = ModelConfig.tiny()
    cfg = ModelConfig(
        **{**base.model_dump(), "multimodal": MultimodalConfig(
            enable_vision_projector=True,
            vision_input_dim=96,
        )},
    )
    model = SaintLLM(cfg)
    model.eval()
    t = 16
    token_ids = torch.randint(0, cfg.vocab_size, (1, t))
    token_ids[0, 4:8] = cfg.tokenizer_slots.image_pad
    vision_features = torch.randn(4, 96)
    with torch.no_grad():
        out = model(token_ids, vision_features=vision_features)
    assert out["logits"].shape == (1, t, cfg.vocab_size)


@pytest.mark.gpu
@pytest.mark.slow
def test_real_siglip2_loads_and_runs() -> None:
    """Integration: real SigLIP-2-SO400M from HF Hub. Skipped unless -m gpu and slow."""
    pytest.importorskip("transformers")
    from saint_llm_core.multimodal import SigLIP2Wrapper, VisionTokenizer

    cfg = VisionEncoderConfig()
    encoder = SigLIP2Wrapper(cfg)
    tokenizer = VisionTokenizer(cfg, encoder)
    pixel_values = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    out = tokenizer(pixel_values)
    assert out.shape[-1] == cfg.output_dim
    assert out.shape[0] == 1
