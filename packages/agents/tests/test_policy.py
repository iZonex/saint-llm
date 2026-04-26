"""Tests for MockPolicy + CallablePolicy adapters."""

from __future__ import annotations

import pytest
from saint_llm_agents import CallablePolicy, Message, MockPolicy, ToolCall, ToolSpec


def test_mock_policy_returns_responses_in_order() -> None:
    a = Message(role="assistant", content="first")
    b = Message(role="assistant", content="second")
    policy = MockPolicy([a, b])

    assert policy.act([]) is a
    assert policy.remaining == 1
    assert policy.act([]) is b
    assert policy.remaining == 0


def test_mock_policy_raises_when_exhausted() -> None:
    policy = MockPolicy([Message(role="assistant", content="only")])
    policy.act([])
    with pytest.raises(StopIteration):
        policy.act([])


def test_mock_policy_ignores_messages_and_tools_arg() -> None:
    """It's a mock — it shouldn't introspect inputs."""
    policy = MockPolicy([Message(role="assistant", content="ok")])
    fake_history = [Message(role="user", content="anything")]
    fake_tools = (ToolSpec(name="add", description=""),)
    out = policy.act(fake_history, tools=fake_tools)
    assert out.content == "ok"


def test_callable_policy_passes_messages_and_tools_through() -> None:
    seen: dict[str, object] = {}

    def fn(messages, *, tools):
        seen["messages"] = messages
        seen["tools"] = tools
        return Message(role="assistant", content="echoed")

    policy = CallablePolicy(fn)
    history = [Message(role="user", content="ping")]
    tools = (ToolSpec(name="t", description=""),)
    out = policy.act(history, tools=tools)

    assert out.content == "echoed"
    assert seen["messages"] is history
    assert seen["tools"] is tools


def test_mock_policy_can_emit_tool_calls() -> None:
    """A mocked assistant may include tool_calls so agent loop tests work."""
    call = ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})
    policy = MockPolicy([Message(role="assistant", content="", tool_calls=(call,))])
    out = policy.act([])
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "add"
