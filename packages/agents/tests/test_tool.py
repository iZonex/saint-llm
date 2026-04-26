"""Tests for ToolSpec + ToolRegistry execution semantics."""

from __future__ import annotations

import pytest
from saint_llm_agents import (
    FunctionTool,
    Message,
    ToolCall,
    ToolRegistry,
    ToolSpec,
)


def _add_tool() -> FunctionTool:
    spec = ToolSpec(
        name="add",
        description="add two integers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    )
    return FunctionTool(spec=spec, fn=lambda a, b: a + b)


def test_register_and_get() -> None:
    reg = ToolRegistry()
    tool = _add_tool()
    reg.register(tool)
    assert "add" in reg
    assert len(reg) == 1
    assert reg.get("add") is tool


def test_duplicate_registration_rejected() -> None:
    reg = ToolRegistry()
    reg.register(_add_tool())
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_add_tool())


def test_get_unknown_tool_raises_key_error() -> None:
    reg = ToolRegistry()
    with pytest.raises(KeyError, match="not registered"):
        reg.get("nope")


def test_specs_returns_all_registered() -> None:
    reg = ToolRegistry()
    reg.register(_add_tool())
    spec_b = ToolSpec(name="echo", description="echo back")
    reg.register(FunctionTool(spec=spec_b, fn=lambda **kw: kw))
    names = {s.name for s in reg.specs()}
    assert names == {"add", "echo"}


def test_execute_success_returns_tool_message() -> None:
    reg = ToolRegistry()
    reg.register(_add_tool())
    msg = reg.execute(ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3}))
    assert isinstance(msg, Message)
    assert msg.role == "tool"
    assert msg.tool_call_id == "c1"
    assert msg.content == "5"


def test_execute_unknown_tool_returns_error_message() -> None:
    reg = ToolRegistry()
    msg = reg.execute(ToolCall(id="c2", name="ghost"))
    assert msg.role == "tool"
    assert msg.tool_call_id == "c2"
    assert "ERROR" in msg.content
    assert "ghost" in msg.content


def test_execute_captures_tool_exception_into_error_message() -> None:
    reg = ToolRegistry()

    def boom(**_kw: object) -> object:
        raise RuntimeError("kaboom")

    reg.register(FunctionTool(spec=ToolSpec(name="boom", description=""), fn=boom))
    msg = reg.execute(ToolCall(id="c3", name="boom"))
    assert msg.role == "tool"
    assert msg.tool_call_id == "c3"
    assert "ERROR: RuntimeError: kaboom" in msg.content


def test_execute_none_return_becomes_empty_string() -> None:
    reg = ToolRegistry()
    reg.register(
        FunctionTool(spec=ToolSpec(name="noop", description=""), fn=lambda: None),
    )
    msg = reg.execute(ToolCall(id="c4", name="noop"))
    assert msg.content == ""


def test_contains_only_for_strings() -> None:
    reg = ToolRegistry()
    reg.register(_add_tool())
    assert "add" in reg
    assert 42 not in reg  # type: ignore[operator]
