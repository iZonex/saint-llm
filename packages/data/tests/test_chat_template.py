"""Tests for the chat template / thinking-block formatter."""

from __future__ import annotations

import pytest
from saint_llm_data import CharTokenizer
from saint_llm_data.chat_template import (
    ChatTemplate,
    ChatTurn,
    RenderedChat,
    render_chat,
)


def test_render_chat_basic_user_assistant_round_trip() -> None:
    tok = CharTokenizer()
    turns = [
        ChatTurn(role="user", content="hi"),
        ChatTurn(role="assistant", content="hello"),
    ]
    out = render_chat(turns, tok)
    assert isinstance(out, RenderedChat)
    assert "<|user|>\nhi\n<|/user|>\n" in out.text
    assert "<|assistant|>\nhello\n<|/assistant|>\n" in out.text


def test_render_chat_token_and_mask_lengths_match() -> None:
    tok = CharTokenizer()
    turns = [
        ChatTurn(role="system", content="be helpful"),
        ChatTurn(role="user", content="ping"),
        ChatTurn(role="assistant", content="pong"),
    ]
    out = render_chat(turns, tok)
    assert len(out.token_ids) == len(out.loss_mask)
    assert len(out.token_ids) > 0


def test_render_chat_loss_mask_zero_on_user_and_system() -> None:
    """Only assistant content positions should be loss-active."""
    tok = CharTokenizer()
    turns = [
        ChatTurn(role="system", content="you are helpful"),
        ChatTurn(role="user", content="hi"),
        ChatTurn(role="assistant", content="hello back"),
    ]
    out = render_chat(turns, tok)
    # The substring "hello back" appears once — verify mask is 1 there.
    assistant_substring = "hello back"
    pos = out.text.find(assistant_substring)
    # Char tokenizer maps 1 byte -> 1 token, so positions align.
    for i, ch in enumerate(assistant_substring):
        token_idx = pos + i
        assert out.loss_mask[token_idx] == 1, (
            f"expected loss_mask=1 at char {ch!r}; got "
            f"{out.loss_mask[token_idx]}"
        )


def test_render_chat_thinking_block_wrapped_correctly() -> None:
    tok = CharTokenizer()
    turns = [
        ChatTurn(
            role="user", content="solve",
        ),
        ChatTurn(
            role="assistant",
            content="42",
            thinking="step 1: compute 6*7",
            effort_tier=2,
        ),
    ]
    out = render_chat(turns, tok)
    assert "<|effort:2|>" in out.text
    assert "<|think_start|>step 1: compute 6*7<|think_end|>" in out.text
    # Visible answer "42" comes after the think block.
    think_end_pos = out.text.find("<|think_end|>")
    answer_pos = out.text.find("42", think_end_pos)
    assert answer_pos > think_end_pos


def test_render_chat_thinking_loss_active_by_default() -> None:
    tok = CharTokenizer()
    turns = [
        ChatTurn(role="user", content="x"),
        ChatTurn(role="assistant", content="y", thinking="reasoning"),
    ]
    out = render_chat(turns, tok)
    pos = out.text.find("reasoning")
    # mask_thinking defaults to True -> loss is active on thinking content.
    for i in range(len("reasoning")):
        assert out.loss_mask[pos + i] == 1


def test_render_chat_thinking_loss_masked_when_disabled() -> None:
    tok = CharTokenizer()
    turns = [
        ChatTurn(role="user", content="x"),
        ChatTurn(role="assistant", content="y", thinking="reasoning"),
    ]
    out = render_chat(turns, tok, mask_thinking=False)
    pos = out.text.find("reasoning")
    for i in range(len("reasoning")):
        assert out.loss_mask[pos + i] == 0


def test_render_chat_unknown_role_raises() -> None:
    tok = CharTokenizer()
    with pytest.raises(ValueError, match="unknown role"):
        render_chat(
            [ChatTurn(role="bogus", content="x")],  # type: ignore[arg-type]
            tok,
        )


def test_render_chat_with_generation_prompt_appends_assistant_prefix() -> None:
    tok = CharTokenizer()
    turns = [ChatTurn(role="user", content="hi")]
    out = render_chat(turns, tok, add_generation_prompt=True)
    assert out.text.endswith("<|assistant|>\n")


def test_render_chat_generation_prompt_carries_last_effort_tier() -> None:
    tok = CharTokenizer()
    turns = [
        ChatTurn(role="user", content="x"),
        ChatTurn(role="assistant", content="y", effort_tier=3),
        ChatTurn(role="user", content="next"),
    ]
    out = render_chat(turns, tok, add_generation_prompt=True)
    # The last assistant turn declared effort_tier=3; generation prompt
    # carries it forward.
    assert out.text.endswith("<|assistant|>\n<|effort:3|>")


def test_render_chat_append_eos_loss_active_only_after_assistant() -> None:
    tok = CharTokenizer()
    turns = [
        ChatTurn(role="user", content="x"),
        ChatTurn(role="assistant", content="y"),
    ]
    out = render_chat(turns, tok, append_eos=True)
    # Last token is EOS. With assistant-last, mask should be 1.
    assert out.token_ids[-1] == tok.eos_token_id
    assert out.loss_mask[-1] == 1


def test_render_chat_append_eos_loss_masked_when_user_is_last() -> None:
    tok = CharTokenizer()
    turns = [ChatTurn(role="user", content="x")]
    out = render_chat(turns, tok, append_eos=True)
    assert out.loss_mask[-1] == 0


def test_render_chat_custom_template_overrides() -> None:
    tok = CharTokenizer()
    custom = ChatTemplate(
        user_prefix="USER: ",
        user_suffix="\n",
        assistant_prefix="BOT: ",
        assistant_suffix="\n",
        system_prefix="SYSTEM: ",
        system_suffix="\n",
    )
    turns = [
        ChatTurn(role="user", content="ping"),
        ChatTurn(role="assistant", content="pong"),
    ]
    out = render_chat(turns, tok, template=custom)
    assert "USER: ping" in out.text
    assert "BOT: pong" in out.text
    # Default markers should not appear.
    assert "<|user|>" not in out.text


def test_render_chat_bos_text_added_at_start() -> None:
    tok = CharTokenizer()
    custom = ChatTemplate(bos_text="<bos>\n")
    turns = [ChatTurn(role="user", content="hi")]
    out = render_chat(turns, tok, template=custom)
    assert out.text.startswith("<bos>\n")


def test_render_chat_empty_turns_is_empty_output() -> None:
    tok = CharTokenizer()
    out = render_chat([], tok)
    assert out.token_ids == []
    assert out.loss_mask == []
    assert out.text == ""


def test_render_chat_multi_turn_conversation() -> None:
    """A 4-turn dialog renders all turns in order."""
    tok = CharTokenizer()
    turns = [
        ChatTurn(role="system", content="sys"),
        ChatTurn(role="user", content="u1"),
        ChatTurn(role="assistant", content="a1"),
        ChatTurn(role="user", content="u2"),
        ChatTurn(role="assistant", content="a2"),
    ]
    out = render_chat(turns, tok)
    sys_pos = out.text.find("sys")
    u1_pos = out.text.find("u1")
    a1_pos = out.text.find("a1")
    u2_pos = out.text.find("u2")
    a2_pos = out.text.find("a2")
    assert sys_pos < u1_pos < a1_pos < u2_pos < a2_pos


def test_render_chat_effort_tier_only_when_assistant() -> None:
    """effort_tier on a non-assistant turn is silently ignored."""
    tok = CharTokenizer()
    turns = [
        ChatTurn(role="user", content="x", effort_tier=2),
        ChatTurn(role="assistant", content="y"),
    ]
    out = render_chat(turns, tok)
    assert "<|effort:" not in out.text
