"""Tests for multimodal-aware decoding."""

from __future__ import annotations

import pytest
import torch
from saint_llm_core.config import ModelConfig, MultimodalConfig
from saint_llm_core.model import SaintLLM
from saint_llm_inference import greedy_decode
from saint_llm_inference.multimodal_generate import (
    MultimodalSlots,
    build_multimodal_prompt_ids,
    multimodal_greedy_decode,
)


def _vision_enabled_cfg() -> ModelConfig:
    base = ModelConfig.tiny()
    return ModelConfig(
        **{
            **base.model_dump(),
            "multimodal": MultimodalConfig(
                enable_vision_projector=True,
                vision_input_dim=96,
            ),
        },
    )


def _audio_enabled_cfg() -> ModelConfig:
    base = ModelConfig.tiny()
    return ModelConfig(
        **{
            **base.model_dump(),
            "multimodal": MultimodalConfig(
                enable_audio_projector=True,
                audio_input_dim=64,
            ),
        },
    )


def _slots() -> MultimodalSlots:
    base = ModelConfig.tiny()
    return MultimodalSlots(
        image_pad=base.tokenizer_slots.image_pad,
        audio_start=base.tokenizer_slots.audio_start,
        audio_end=base.tokenizer_slots.audio_end,
    )


def test_build_prompt_image_only() -> None:
    ids = build_multimodal_prompt_ids(
        text_ids_before=[1, 2, 3],
        text_ids_after=[7, 8],
        slots=_slots(),
        n_image_pads=4,
    )
    flat = ids[0].tolist()
    assert flat[:3] == [1, 2, 3]
    assert flat[3:7] == [_slots().image_pad] * 4
    assert flat[7:] == [7, 8]


def test_build_prompt_audio_brackets_correctly() -> None:
    ids = build_multimodal_prompt_ids(
        text_ids_before=[1],
        text_ids_after=[9],
        slots=_slots(),
        n_audio_tokens=3,
    )
    flat = ids[0].tolist()
    s = _slots()
    # 1 + audio_start + [audio_start]*3 + audio_end + 9 = 7 tokens
    assert flat == [1, s.audio_start, s.audio_start, s.audio_start, s.audio_start, s.audio_end, 9]


def test_build_prompt_text_only_returns_just_text() -> None:
    ids = build_multimodal_prompt_ids(
        text_ids_before=[5, 6],
        text_ids_after=[7],
        slots=_slots(),
    )
    assert ids.shape == (1, 3)
    assert ids[0].tolist() == [5, 6, 7]


def test_build_prompt_returns_long_dtype_and_shape() -> None:
    ids = build_multimodal_prompt_ids(
        text_ids_before=[1],
        text_ids_after=[2],
        slots=_slots(),
        n_image_pads=2,
    )
    assert ids.dtype == torch.long
    assert ids.shape == (1, 4)  # 1 + 2 image pads + 1


def test_multimodal_greedy_decode_text_only_matches_greedy() -> None:
    """vision/audio = None => same output as plain greedy_decode."""
    torch.manual_seed(0)
    model = SaintLLM(ModelConfig.tiny())
    model.eval()
    prompt = torch.randint(0, ModelConfig.tiny().vocab_size, (1, 6))
    out_mm = multimodal_greedy_decode(model, prompt, max_new_tokens=4)
    out_text = greedy_decode(model, prompt, max_new_tokens=4)
    assert torch.equal(out_mm, out_text)


def test_multimodal_greedy_decode_with_vision_features() -> None:
    """Vision features pass through; output shape is correct."""
    torch.manual_seed(0)
    cfg = _vision_enabled_cfg()
    model = SaintLLM(cfg)
    model.eval()

    n_image_pads = 3
    prompt = build_multimodal_prompt_ids(
        text_ids_before=[1, 2],
        text_ids_after=[3, 4],
        slots=_slots(),
        n_image_pads=n_image_pads,
    )
    vision_features = torch.randn(n_image_pads, 96)
    out = multimodal_greedy_decode(
        model, prompt, max_new_tokens=4, vision_features=vision_features,
    )
    assert out.shape == (1, prompt.shape[1] + 4)
    assert out.dtype == torch.long
    # Original prompt prefix preserved.
    assert torch.equal(out[:, : prompt.shape[1]], prompt)


def test_multimodal_greedy_decode_with_audio_features() -> None:
    torch.manual_seed(0)
    cfg = _audio_enabled_cfg()
    model = SaintLLM(cfg)
    model.eval()

    n_audio = 2
    prompt = build_multimodal_prompt_ids(
        text_ids_before=[1],
        text_ids_after=[3],
        slots=_slots(),
        n_audio_tokens=n_audio,
    )
    # Audio features must equal the count of slot IDs in [audio_start, audio_end]
    # spliced into the prompt — that's 1 (open) + n_audio (body) + 1 (close).
    n_audio_slots = int(((prompt >= _slots().audio_start) & (prompt <= _slots().audio_end)).sum().item())
    audio_features = torch.randn(n_audio_slots, 64)
    out = multimodal_greedy_decode(
        model, prompt, max_new_tokens=3, audio_features=audio_features,
    )
    assert out.shape == (1, prompt.shape[1] + 3)


def test_multimodal_greedy_decode_rejects_1d_prompt() -> None:
    cfg = _vision_enabled_cfg()
    model = SaintLLM(cfg)
    model.eval()
    with pytest.raises(ValueError, match=r"\(B, T\)"):
        multimodal_greedy_decode(model, torch.tensor([1, 2, 3]), max_new_tokens=2)


def test_multimodal_greedy_decode_eos_stops_early() -> None:
    torch.manual_seed(0)
    model = SaintLLM(ModelConfig.tiny())
    model.eval()
    prompt = torch.zeros(1, 4, dtype=torch.long)
    out = multimodal_greedy_decode(
        model, prompt, max_new_tokens=8, eos_token=0,
    )
    # Output should still be the full requested width (eos may pad).
    assert out.shape == (1, 12)


def test_multimodal_greedy_decode_feature_count_mismatch_raises() -> None:
    """Mismatched feature row count triggers SaintLLM's error."""
    cfg = _vision_enabled_cfg()
    model = SaintLLM(cfg)
    model.eval()
    prompt = build_multimodal_prompt_ids(
        text_ids_before=[1],
        text_ids_after=[2],
        slots=_slots(),
        n_image_pads=3,
    )
    wrong = torch.randn(5, 96)  # need 3
    with pytest.raises(ValueError, match="must match"):
        multimodal_greedy_decode(
            model, prompt, max_new_tokens=2, vision_features=wrong,
        )


def test_multimodal_greedy_decode_deterministic() -> None:
    """Same prompt + features + seed => same output."""
    torch.manual_seed(0)
    cfg = _vision_enabled_cfg()
    model = SaintLLM(cfg)
    model.eval()
    prompt = build_multimodal_prompt_ids(
        text_ids_before=[1, 2],
        text_ids_after=[3],
        slots=_slots(),
        n_image_pads=2,
    )
    features = torch.randn(2, 96)
    out1 = multimodal_greedy_decode(model, prompt, max_new_tokens=3, vision_features=features)
    out2 = multimodal_greedy_decode(model, prompt, max_new_tokens=3, vision_features=features)
    assert torch.equal(out1, out2)
