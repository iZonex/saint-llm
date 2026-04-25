"""Audio encoder + Voxtral downsample + end-to-end SaintLLM-with-audio integration."""

from __future__ import annotations

import pytest
import torch

from saint_llm_core import ModelConfig, SaintLLM
from saint_llm_core.config import MultimodalConfig
from saint_llm_core.multimodal import (
    AudioEncoderConfig,
    AudioTokenizer,
    FakeWhisperEncoder,
)


def test_audio_encoder_config_defaults_match_voxtral_recipe() -> None:
    cfg = AudioEncoderConfig()
    assert cfg.encoder_hidden_dim == 1280
    assert cfg.n_mel_bins == 128
    assert cfg.downsample_factor == 4
    # 100 Hz mel → 50 Hz post-Whisper → 12.5 Hz post-Voxtral downsample
    assert cfg.output_rate_hz == 12.5


def test_fake_whisper_2x_temporal_pool() -> None:
    cfg = AudioEncoderConfig(encoder_hidden_dim=64, n_mel_bins=80)
    enc = FakeWhisperEncoder(cfg)
    mel = torch.randn(2, 80, 100)  # 1 s of mel at 100 Hz
    out = enc(mel)
    assert out.shape == (2, 50, 64)  # 2× pool


def test_audio_tokenizer_4x_downsample_clean_division() -> None:
    cfg = AudioEncoderConfig(encoder_hidden_dim=64, n_mel_bins=80, downsample_factor=4)
    tok = AudioTokenizer(cfg, FakeWhisperEncoder(cfg))
    mel = torch.randn(2, 80, 200)  # 200 mel → 100 (whisper) → 25 (4× downsample)
    out = tok(mel)
    assert out.shape == (2, 25, 64)


def test_audio_tokenizer_pads_non_divisible_length() -> None:
    cfg = AudioEncoderConfig(encoder_hidden_dim=64, n_mel_bins=80, downsample_factor=4)
    tok = AudioTokenizer(cfg, FakeWhisperEncoder(cfg))
    mel = torch.randn(1, 80, 50)  # 50 → 25 → pad to 28 → 7 audio tokens
    out = tok(mel)
    assert out.shape == (1, 7, 64)


def test_audio_tokenizer_no_downsample_when_factor_one() -> None:
    cfg = AudioEncoderConfig(encoder_hidden_dim=64, n_mel_bins=80, downsample_factor=1)
    tok = AudioTokenizer(cfg, FakeWhisperEncoder(cfg))
    mel = torch.randn(1, 80, 100)
    out = tok(mel)
    assert out.shape == (1, 50, 64)  # only Whisper's 2× pool


def test_audio_tokenizer_output_at_125hz_matches_30s_audio_budget() -> None:
    """Voxtral parity: 30 s @ 16 kHz → ~375 audio tokens at 12.5 Hz output."""
    cfg = AudioEncoderConfig(encoder_hidden_dim=64, n_mel_bins=128, downsample_factor=4)
    tok = AudioTokenizer(cfg, FakeWhisperEncoder(cfg))
    # 30 s × 100 Hz mel = 3000 frames
    mel = torch.randn(1, 128, 3000)
    out = tok(mel)
    # 3000 → 1500 (Whisper 2×) → 375 (4× downsample)
    assert out.shape == (1, 375, 64)


def test_saintllm_with_audio_features_forward() -> None:
    base = ModelConfig.tiny()
    cfg = ModelConfig(
        **{**base.model_dump(), "multimodal": MultimodalConfig(
            enable_audio_projector=True,
            audio_input_dim=64,
        )},
    )
    model = SaintLLM(cfg)
    model.eval()

    t = 16
    n_audio_tokens = 5
    token_ids = torch.randint(0, 500, (1, t))
    # Insert audio-region tokens (in tiny config, audio range is [501, 505]).
    token_ids[0, 4:9] = cfg.tokenizer_slots.audio_start  # 501
    audio_features = torch.randn(n_audio_tokens, 64)

    with torch.no_grad():
        out = model(token_ids, audio_features=audio_features)
    assert out["logits"].shape == (1, t, cfg.vocab_size)
    assert torch.isfinite(out["logits"]).all()


def test_saintllm_audio_features_size_mismatch_raises() -> None:
    base = ModelConfig.tiny()
    cfg = ModelConfig(
        **{**base.model_dump(), "multimodal": MultimodalConfig(
            enable_audio_projector=True,
            audio_input_dim=64,
        )},
    )
    model = SaintLLM(cfg)
    model.eval()
    t = 16
    token_ids = torch.randint(0, 500, (1, t))
    token_ids[0, 4:9] = cfg.tokenizer_slots.audio_start
    wrong = torch.randn(3, 64)  # need 5
    with pytest.raises(ValueError, match="must match"):
        model(token_ids, audio_features=wrong)


def test_saintllm_no_audio_features_when_disabled() -> None:
    cfg = ModelConfig.tiny()
    assert cfg.multimodal.enable_audio_projector is False
    model = SaintLLM(cfg)
    model.eval()
    t = 16
    token_ids = torch.randint(0, 500, (2, t))
    with torch.no_grad():
        out = model(token_ids, audio_features=torch.randn(99, 1280))
    assert out["logits"].shape == (2, t, cfg.vocab_size)


def test_saintllm_combined_vision_and_audio() -> None:
    """Vision + audio in the same forward — proves both modality paths compose."""
    base = ModelConfig.tiny()
    cfg = ModelConfig(
        **{**base.model_dump(), "multimodal": MultimodalConfig(
            enable_vision_projector=True,
            vision_input_dim=64,
            enable_audio_projector=True,
            audio_input_dim=64,
        )},
    )
    model = SaintLLM(cfg)
    model.eval()

    t = 16
    token_ids = torch.randint(0, 500, (1, t))
    token_ids[0, 2:5] = cfg.tokenizer_slots.image_pad
    token_ids[0, 8:11] = cfg.tokenizer_slots.audio_start
    vision_features = torch.randn(3, 64)
    audio_features = torch.randn(3, 64)

    with torch.no_grad():
        out = model(token_ids, vision_features=vision_features, audio_features=audio_features)
    assert out["logits"].shape == (1, t, cfg.vocab_size)


@pytest.mark.gpu
@pytest.mark.slow
def test_real_whisper_loads_and_runs() -> None:
    """Integration: real Whisper-large-v3 from HF Hub. Skipped unless -m gpu and slow."""
    pytest.importorskip("transformers")
    from saint_llm_core.multimodal import AudioTokenizer, WhisperLargeV3Wrapper

    cfg = AudioEncoderConfig()
    encoder = WhisperLargeV3Wrapper(cfg)
    tok = AudioTokenizer(cfg, encoder)
    # Whisper expects fixed 30 s × 100 Hz = 3000-frame mel.
    mel = torch.randn(1, cfg.n_mel_bins, 3000)
    out = tok(mel)
    assert out.shape[-1] == cfg.encoder_hidden_dim
    assert out.shape[1] == 375  # 12.5 Hz × 30 s
