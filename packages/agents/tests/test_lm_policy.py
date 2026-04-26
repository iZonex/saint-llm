"""Tests for TextLMPolicy."""

from __future__ import annotations

import pytest
from saint_llm_agents import (
    Agent,
    FunctionTool,
    Message,
    ToolCall,
    ToolRegistry,
    ToolSpec,
)
from saint_llm_agents.lm_policy import TextLMPolicy


def test_text_only_output_returns_assistant_message_without_tool_calls() -> None:
    policy = TextLMPolicy(lambda history: "Hello there!")
    msg = policy.act([Message(role="user", content="hi")])
    assert msg.role == "assistant"
    assert msg.content == "Hello there!"
    assert msg.tool_calls == ()


def test_xml_tool_call_parsed_into_tool_calls_field() -> None:
    text = (
        "Sure, let me check.\n"
        '<tool_call name="search" id="c1">{"query": "weather"}</tool_call>'
    )
    policy = TextLMPolicy(lambda history: text)
    msg = policy.act([Message(role="user", content="weather?")])
    assert msg.role == "assistant"
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "search"
    assert msg.tool_calls[0].arguments == {"query": "weather"}


def test_json_fenced_tool_call_parsed() -> None:
    text = (
        "Calling:\n"
        '```json\n{"name": "ping", "arguments": {}}\n```'
    )
    policy = TextLMPolicy(lambda history: text, parse_format="json")
    msg = policy.act([Message(role="user", content="ping?")])
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "ping"


def test_parse_format_none_skips_parsing() -> None:
    """fmt='none' returns the raw text without ever parsing tool calls."""
    text = '<tool_call name="x" id="1">{}</tool_call>'
    policy = TextLMPolicy(lambda history: text, parse_format="none")
    msg = policy.act([Message(role="user", content="hi")])
    assert msg.tool_calls == ()
    assert msg.content == text


def test_strict_mode_propagates_malformed_block_errors() -> None:
    bad = '<tool_call name="x" id="1">not-json</tool_call>'
    policy = TextLMPolicy(lambda history: bad, strict=True)
    with pytest.raises(ValueError, match="invalid JSON"):
        policy.act([Message(role="user", content="hi")])


def test_lenient_mode_drops_malformed_blocks() -> None:
    """Default lenient mode silently skips bad blocks."""
    text = (
        '<tool_call name="ok" id="1">{"a": 1}</tool_call>'
        '<tool_call name="bad" id="2">not-json</tool_call>'
    )
    policy = TextLMPolicy(lambda history: text)
    msg = policy.act([Message(role="user", content="hi")])
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "ok"


def test_generate_must_return_string() -> None:
    policy = TextLMPolicy(lambda history: 12345)  # type: ignore[arg-type, return-value]
    with pytest.raises(TypeError, match="must return str"):
        policy.act([Message(role="user", content="x")])


def test_policy_passes_full_history_to_generate() -> None:
    """The generate callable sees the entire history sequence."""
    captured: list[Message] = []

    def gen(history):  # type: ignore[no-untyped-def]
        captured.extend(history)
        return "response"

    history = [
        Message(role="system", content="be helpful"),
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
        Message(role="user", content="again"),
    ]
    TextLMPolicy(gen).act(history)
    assert captured == history


def test_policy_drives_agent_loop_with_tool_call() -> None:
    """End-to-end: TextLMPolicy + Agent + ToolRegistry produces tool-result roundtrip."""
    add_spec = ToolSpec(
        name="add",
        description="add two integers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    )
    registry = ToolRegistry()
    registry.register(FunctionTool(spec=add_spec, fn=lambda a, b: a + b))

    # Policy alternates: first turn calls add(2, 3); second turn answers.
    call_count = [0]

    def gen(history):  # type: ignore[no-untyped-def]
        call_count[0] += 1
        if call_count[0] == 1:
            return '<tool_call name="add" id="c1">{"a": 2, "b": 3}</tool_call>'
        return "The answer is 5."

    agent = Agent(
        name="calc", policy=TextLMPolicy(gen), tools=registry,
    )
    history = agent.run(user_message="what is 2 + 3?")
    # History should contain user, assistant (tool_call), tool result, assistant (final).
    assistant_finals = [
        m for m in history
        if m.role == "assistant" and not m.tool_calls
    ]
    tool_results = [m for m in history if m.role == "tool"]
    assert len(tool_results) == 1
    assert tool_results[0].content == "5"
    assert len(assistant_finals) == 1
    assert "5" in assistant_finals[0].content


def test_policy_with_no_history_still_works() -> None:
    policy = TextLMPolicy(lambda history: "starting fresh")
    msg = policy.act([])
    assert msg.content == "starting fresh"
    assert msg.tool_calls == ()


def test_multiple_tool_calls_in_one_turn() -> None:
    text = (
        '<tool_call name="a" id="1">{"x": 1}</tool_call>'
        '<tool_call name="b" id="2">{"y": 2}</tool_call>'
    )
    policy = TextLMPolicy(lambda history: text)
    msg = policy.act([Message(role="user", content="batch")])
    assert len(msg.tool_calls) == 2
    assert [c.name for c in msg.tool_calls] == ["a", "b"]
    # All entries are ToolCall instances.
    for c in msg.tool_calls:
        assert isinstance(c, ToolCall)


def test_act_signature_accepts_tools_kwarg() -> None:
    """Mirrors the Policy Protocol: act(messages, *, tools=...)."""
    policy = TextLMPolicy(lambda history: "ok")
    spec = ToolSpec(name="x", description="x", parameters={})
    msg = policy.act([Message(role="user", content="hi")], tools=(spec,))
    assert msg.role == "assistant"
