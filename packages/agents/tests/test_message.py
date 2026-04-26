"""Tests for Message + ToolCall dataclass invariants."""

from __future__ import annotations

import pytest
from saint_llm_agents import Message, ToolCall


def test_message_is_frozen() -> None:
    msg = Message(role="user", content="hi")
    with pytest.raises((AttributeError, TypeError)):
        msg.content = "changed"  # type: ignore[misc]


def test_tool_call_default_arguments_is_empty_dict() -> None:
    call = ToolCall(id="c1", name="search")
    assert dict(call.arguments) == {}


def test_tool_role_requires_tool_call_id() -> None:
    with pytest.raises(ValueError, match="tool_call_id"):
        Message(role="tool", content="result")


def test_tool_calls_only_on_assistant_role() -> None:
    call = ToolCall(id="c1", name="search", arguments={"q": "x"})
    with pytest.raises(ValueError, match="only assistant messages"):
        Message(role="user", content="hi", tool_calls=(call,))


def test_assistant_with_tool_calls_is_valid() -> None:
    call = ToolCall(id="c1", name="search", arguments={"q": "x"})
    msg = Message(role="assistant", content="", tool_calls=(call,))
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "search"


def test_tool_message_carries_tool_call_id() -> None:
    msg = Message(role="tool", content="42", tool_call_id="c1")
    assert msg.tool_call_id == "c1"


def test_message_with_optional_name_field() -> None:
    msg = Message(role="assistant", content="hi", name="researcher")
    assert msg.name == "researcher"
