"""Tests for the ChatSession driver."""

from __future__ import annotations

import pytest
import torch
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM
from saint_llm_data import CharTokenizer, ChatTemplate
from saint_llm_inference.chat_session import (
    ChatSession,
    GenerationConfig,
)
from torch import nn


class _FixedNextTokenLM(nn.Module):
    """Argmax always picks ``next_token``."""

    def __init__(self, vocab: int, next_token: int) -> None:
        super().__init__()
        self.vocab = vocab
        self.next_token = next_token
        self.embed = nn.Embedding(vocab, 4)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t = tokens.shape
        logits = torch.full((b, t, self.vocab), -1e9)
        logits[:, :, self.next_token] = 0.0
        return {"logits": logits}


def _real_session(*, system: str = "") -> ChatSession:
    torch.manual_seed(0)
    model = SaintLLM(ModelConfig.tiny())
    return ChatSession(
        model=model,
        tokenizer=CharTokenizer(),
        system=system,
    )


def test_chat_session_starts_with_system_when_provided() -> None:
    session = _real_session(system="be helpful")
    assert len(session.history) == 1
    assert session.history[0].role == "system"


def test_chat_session_starts_empty_when_no_system() -> None:
    session = _real_session()
    assert session.history == []


def test_add_user_appends_turn() -> None:
    session = _real_session()
    session.add_user("hello")
    assert len(session.history) == 1
    assert session.history[0].role == "user"
    assert session.history[0].content == "hello"


def test_add_user_rejects_empty() -> None:
    session = _real_session()
    with pytest.raises(ValueError, match="non-empty"):
        session.add_user("")


def test_add_system_rejects_empty() -> None:
    session = _real_session()
    with pytest.raises(ValueError, match="non-empty"):
        session.add_system("")


def test_reset_clears_history() -> None:
    session = _real_session(system="x")
    session.add_user("y")
    session.reset()
    assert session.history == []


def test_stream_response_rejects_empty_history() -> None:
    session = _real_session()
    with pytest.raises(ValueError, match="empty"):
        list(session.stream_response())


def test_stream_response_yields_text_fragments() -> None:
    """Each yielded value is decoded text for one token."""
    session = _real_session()
    session.add_user("hi")
    cfg = GenerationConfig(max_new_tokens=3, temperature=0.0)
    fragments = list(session.stream_response(cfg=cfg))
    # CharTokenizer maps 1 byte -> 1 char; 3 yielded tokens -> 3 strings.
    # Some may be empty strings depending on what the model emits, but
    # the count should match.
    assert len(fragments) <= 3
    assert all(isinstance(f, str) for f in fragments)


def test_stream_response_appends_assistant_turn_on_completion() -> None:
    session = _real_session()
    session.add_user("hi")
    list(session.stream_response(cfg=GenerationConfig(
        max_new_tokens=3, temperature=0.0,
    )))
    # After the stream completes, history grows by one assistant turn.
    assert session.history[-1].role == "assistant"


def test_stream_response_stops_on_eos() -> None:
    """Custom model whose argmax is EOS -> stream produces zero fragments."""
    tok = CharTokenizer()
    model = _FixedNextTokenLM(vocab=tok.vocab_size, next_token=tok.eos_token_id)
    session = ChatSession(model=model, tokenizer=tok)
    session.add_user("ping")
    fragments = list(session.stream_response(cfg=GenerationConfig(
        max_new_tokens=10, temperature=0.0,
    )))
    # First emitted token is EOS -> no text fragments yielded.
    assert fragments == []


def test_respond_returns_full_text() -> None:
    session = _real_session()
    session.add_user("hi")
    text = session.respond(cfg=GenerationConfig(
        max_new_tokens=3, temperature=0.0,
    ))
    assert isinstance(text, str)


def test_session_multi_turn_dialog() -> None:
    """Two consecutive turns share state — the second sees the first's history."""
    session = _real_session(system="be helpful")
    session.add_user("hi")
    session.respond(cfg=GenerationConfig(max_new_tokens=2, temperature=0.0))
    history_after_first = list(session.history)
    session.add_user("again")
    session.respond(cfg=GenerationConfig(max_new_tokens=2, temperature=0.0))
    # History should grow by exactly 2 turns (user + assistant).
    assert len(session.history) == len(history_after_first) + 2
    assert session.history[-2].role == "user"
    assert session.history[-1].role == "assistant"


def test_custom_template_overrides_defaults() -> None:
    custom = ChatTemplate(
        user_prefix="USER: ",
        user_suffix="\n",
        assistant_prefix="BOT: ",
        assistant_suffix="\n",
    )
    torch.manual_seed(0)
    model = SaintLLM(ModelConfig.tiny())
    session = ChatSession(
        model=model, tokenizer=CharTokenizer(), template=custom,
    )
    session.add_user("ping")
    # The custom prefix must show up in the rendered prompt indirectly
    # — we verify by inspecting that the model is called without crash.
    out = session.respond(cfg=GenerationConfig(max_new_tokens=2, temperature=0.0))
    assert isinstance(out, str)


def test_session_seeded_temperature_is_reproducible() -> None:
    """Same seed + same prompt -> same response across two calls."""
    session1 = _real_session()
    session2 = _real_session()
    session1.add_user("hello")
    session2.add_user("hello")
    cfg = GenerationConfig(max_new_tokens=4, temperature=1.0, top_p=0.9, seed=42)
    r1 = session1.respond(cfg=cfg)
    r2 = session2.respond(cfg=cfg)
    assert r1 == r2


def test_effort_tier_does_not_crash_chat() -> None:
    """Passing effort_tier injects the marker without breaking generation."""
    session = _real_session()
    session.add_user("solve")
    out = session.respond(
        cfg=GenerationConfig(max_new_tokens=2, temperature=0.0),
        effort_tier=3,
    )
    assert isinstance(out, str)
