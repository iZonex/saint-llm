"""Tests for MultimodalChatSession + streaming with multimodal features."""

from __future__ import annotations

import pytest
import torch
from saint_llm_core.config import ModelConfig, MultimodalConfig
from saint_llm_core.model import SaintLLM
from saint_llm_data import CharTokenizer
from saint_llm_data.multimodal import TokenSlots
from saint_llm_inference.chat_session import GenerationConfig
from saint_llm_inference.multimodal_chat_session import MultimodalChatSession
from saint_llm_inference.streaming import (
    stream_greedy_decode,
    stream_top_p_sample,
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


def _slots() -> TokenSlots:
    return TokenSlots(
        image_pad=ModelConfig.tiny().tokenizer_slots.image_pad,
        audio_start=ModelConfig.tiny().tokenizer_slots.audio_start,
        audio_end=ModelConfig.tiny().tokenizer_slots.audio_end,
    )


def _session(*, system: str = "") -> MultimodalChatSession:
    torch.manual_seed(0)
    cfg = _vision_cfg()
    model = SaintLLM(cfg)
    return MultimodalChatSession(
        model=model,
        tokenizer=CharTokenizer(),
        slots=_slots(),
        system=system,
    )


# ---- streaming with multimodal features -----------------------------


def test_stream_greedy_passes_vision_features() -> None:
    """Decoder runs end-to-end with vision_features."""
    torch.manual_seed(0)
    cfg = _vision_cfg()
    model = SaintLLM(cfg)
    model.eval()

    n_image = 3
    prompt = torch.zeros(1, 8, dtype=torch.long)
    prompt[0, 2 : 2 + n_image] = cfg.tokenizer_slots.image_pad
    vision = torch.randn(n_image, 96)
    out = list(stream_greedy_decode(
        model, prompt, max_new_tokens=3, vision_features=vision,
    ))
    assert len(out) == 3
    for t in out:
        assert t.shape == (1, 1)


def test_stream_top_p_passes_vision_features_reproducibly() -> None:
    torch.manual_seed(0)
    cfg = _vision_cfg()
    model = SaintLLM(cfg)
    model.eval()

    prompt = torch.zeros(1, 6, dtype=torch.long)
    prompt[0, 2:5] = cfg.tokenizer_slots.image_pad
    vision = torch.randn(3, 96)
    out1 = list(stream_top_p_sample(
        model, prompt, max_new_tokens=3, temperature=1.0, p=0.9,
        vision_features=vision,
        generator=torch.Generator().manual_seed(42),
    ))
    out2 = list(stream_top_p_sample(
        model, prompt, max_new_tokens=3, temperature=1.0, p=0.9,
        vision_features=vision,
        generator=torch.Generator().manual_seed(42),
    ))
    for a, b in zip(out1, out2, strict=True):
        assert torch.equal(a, b)


def test_stream_without_features_unchanged() -> None:
    """No features = identical behavior to text-only stream."""
    torch.manual_seed(0)
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    model.eval()
    prompt = torch.zeros(1, 4, dtype=torch.long)
    out_with = list(stream_greedy_decode(
        model, prompt, max_new_tokens=3,
        vision_features=None, audio_features=None,
    ))
    out_without = list(stream_greedy_decode(
        model, prompt, max_new_tokens=3,
    ))
    for a, b in zip(out_with, out_without, strict=True):
        assert torch.equal(a, b)


# ---- MultimodalChatSession ------------------------------------------


def test_session_starts_with_system_when_provided() -> None:
    session = _session(system="be helpful")
    assert len(session.history) == 1
    assert session.history[0].role == "system"


def test_add_user_appends_turn() -> None:
    session = _session()
    session.add_user("hello")
    assert session.history[-1].role == "user"
    assert session.history[-1].content == "hello"


def test_add_user_with_image_attaches_features() -> None:
    session = _session()
    img = torch.randn(2, 96)
    session.add_user_with_image("see this", img)
    turn = session.history[-1]
    assert turn.role == "user"
    assert len(turn.image_features) == 1
    assert torch.equal(turn.image_features[0], img)


def test_add_user_with_image_accepts_tuple() -> None:
    session = _session()
    img1 = torch.randn(2, 96)
    img2 = torch.randn(3, 96)
    session.add_user_with_image("see these", (img1, img2))
    assert len(session.history[-1].image_features) == 2


def test_add_user_with_image_rejects_empty_text() -> None:
    session = _session()
    with pytest.raises(ValueError, match="non-empty"):
        session.add_user_with_image("", torch.randn(2, 96))


def test_add_user_with_image_rejects_empty_features() -> None:
    session = _session()
    with pytest.raises(ValueError, match="at least one"):
        session.add_user_with_image("see", ())


def test_session_respond_with_image_returns_string() -> None:
    """End-to-end: session generates a response when given an image."""
    session = _session()
    session.add_user_with_image("describe", torch.randn(2, 96))
    out = session.respond(cfg=GenerationConfig(
        max_new_tokens=3, temperature=0.0,
    ))
    assert isinstance(out, str)


def test_session_respond_appends_assistant_turn() -> None:
    session = _session()
    session.add_user_with_image("describe", torch.randn(2, 96))
    session.respond(cfg=GenerationConfig(max_new_tokens=2, temperature=0.0))
    assert session.history[-1].role == "assistant"


def test_session_stream_response_yields_text() -> None:
    session = _session()
    session.add_user_with_image("hi", torch.randn(1, 96))
    fragments = list(session.stream_response(cfg=GenerationConfig(
        max_new_tokens=3, temperature=0.0,
    )))
    assert all(isinstance(f, str) for f in fragments)


def test_session_text_only_user_works() -> None:
    """Sessions support pure-text user turns alongside image turns."""
    session = _session()
    session.add_user("just text")
    out = session.respond(cfg=GenerationConfig(
        max_new_tokens=2, temperature=0.0,
    ))
    assert isinstance(out, str)


def test_session_multi_turn_with_image() -> None:
    """Image attached to first turn, text-only follow-up works."""
    session = _session(system="helpful")
    session.add_user_with_image("first", torch.randn(2, 96))
    session.respond(cfg=GenerationConfig(max_new_tokens=2, temperature=0.0))
    session.add_user("again")
    session.respond(cfg=GenerationConfig(max_new_tokens=2, temperature=0.0))
    # System + 2*(user, assistant) = 5 turns.
    assert len(session.history) == 5


def test_session_stream_rejects_empty_history() -> None:
    session = _session()
    with pytest.raises(ValueError, match="empty"):
        list(session.stream_response())


def test_session_reset_clears_history() -> None:
    session = _session(system="helpful")
    session.add_user("hi")
    session.reset()
    assert session.history == []


def test_session_seeded_temperature_reproducible() -> None:
    """Same seed -> same response across two sessions with the same state."""
    img = torch.randn(2, 96)
    s1 = _session()
    s2 = _session()
    s1.add_user_with_image("describe", img)
    s2.add_user_with_image("describe", img)
    cfg = GenerationConfig(max_new_tokens=4, temperature=1.0, top_p=0.9, seed=42)
    r1 = s1.respond(cfg=cfg)
    r2 = s2.respond(cfg=cfg)
    assert r1 == r2


def test_session_effort_tier_does_not_crash() -> None:
    session = _session()
    session.add_user_with_image("solve", torch.randn(2, 96))
    out = session.respond(
        cfg=GenerationConfig(max_new_tokens=2, temperature=0.0),
        effort_tier=2,
    )
    assert isinstance(out, str)
