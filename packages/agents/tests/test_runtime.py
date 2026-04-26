"""Tests for the round-robin multi-agent Runtime."""

from __future__ import annotations

import pytest
from saint_llm_agents import (
    Agent,
    CallablePolicy,
    FunctionTool,
    Message,
    MockPolicy,
    Runtime,
    ToolCall,
    ToolRegistry,
    ToolSpec,
)


def _echo_agent(name: str, *, prefix: str = "") -> Agent:
    """Agent whose policy echoes back the last user message with a prefix."""

    def fn(messages, *, tools):
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        return Message(role="assistant", content=f"{prefix}{last_user}")

    return Agent(name=name, policy=CallablePolicy(fn))


def test_runtime_requires_at_least_two_agents() -> None:
    only = _echo_agent("solo")
    with pytest.raises(ValueError, match="at least 2"):
        Runtime([only])


def test_runtime_rejects_duplicate_names() -> None:
    a = _echo_agent("twin")
    b = _echo_agent("twin")
    with pytest.raises(ValueError, match="must be unique"):
        Runtime([a, b])


def test_send_routes_to_specified_agent_inbox() -> None:
    a = _echo_agent("alice")
    b = _echo_agent("bob")
    rt = Runtime([a, b])
    rt.send(to="bob", content="hi")
    assert rt.inbox_for("bob")[0].content == "hi"
    assert rt.inbox_for("alice") == []


def test_send_to_unknown_agent_raises() -> None:
    rt = Runtime([_echo_agent("a"), _echo_agent("b")])
    with pytest.raises(KeyError, match="charlie"):
        rt.send(to="charlie", content="hi")


def test_two_agent_ping_pong() -> None:
    """alice echoes "alice:<x>"; bob echoes "bob:<x>". After 4 ticks
    the transcript shows both agents replying to each other in turn.
    """
    alice = _echo_agent("alice", prefix="alice:")
    bob = _echo_agent("bob", prefix="bob:")
    rt = Runtime([alice, bob])
    rt.send(to="alice", content="ping")

    rt.tick()  # alice processes "ping" → "alice:ping" → routed to bob
    rt.tick()  # bob processes "alice:ping" → "bob:alice:ping" → routed to alice
    rt.tick()  # alice processes "bob:alice:ping" → "alice:bob:alice:ping"
    rt.tick()  # bob processes "alice:bob:alice:ping" → "bob:alice:bob:alice:ping"

    contents = [m.content for m in rt.transcript]
    assert contents == [
        "ping",
        "alice:ping",
        "bob:alice:ping",
        "alice:bob:alice:ping",
        "bob:alice:bob:alice:ping",
    ]


def test_run_advances_cursor_and_returns_transcript() -> None:
    rt = Runtime([_echo_agent("a", prefix="A:"), _echo_agent("b", prefix="B:")])
    rt.send(to="a", content="x")
    transcript = rt.run(max_ticks=2)
    assert len(transcript) == 3  # 1 user + 2 assistants
    assert rt.cursor == 0  # 2 ticks on 2 agents wraps back to 0


def test_three_agent_round_robin() -> None:
    a = _echo_agent("a", prefix="A:")
    b = _echo_agent("b", prefix="B:")
    c = _echo_agent("c", prefix="C:")
    rt = Runtime([a, b, c])
    rt.send(to="a", content="0")

    for _ in range(3):
        rt.tick()

    contents = [m.content for m in rt.transcript]
    # a("0") → "A:0" routed to b
    # b("A:0") → "B:A:0" routed to c
    # c("B:A:0") → "C:B:A:0" routed to a
    assert contents == ["0", "A:0", "B:A:0", "C:B:A:0"]


def test_tool_results_stay_in_producing_agents_history() -> None:
    """An agent that calls a tool: the tool result is in its own
    history (and global transcript) but is NOT forwarded to the next
    agent's inbox — only assistant content crosses the boundary.
    """
    call = ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})
    reg = ToolRegistry()
    reg.register(
        FunctionTool(
            spec=ToolSpec(name="add", description=""),
            fn=lambda a, b: a + b,
        ),
    )
    # alice calls a tool then says "done"; bob just echoes.
    alice = Agent(
        name="alice",
        policy=MockPolicy(
            [
                Message(role="assistant", content="thinking", tool_calls=(call,)),
                Message(role="assistant", content="done"),
            ],
        ),
        tools=reg,
    )
    bob = _echo_agent("bob", prefix="bob:")

    rt = Runtime([alice, bob])
    rt.send(to="alice", content="please add")

    rt.tick()  # alice: assistant("thinking", calls add) + tool("5"). routes "thinking" to bob.
    rt.tick()  # bob: echoes "thinking" → "bob:thinking". routes to alice.
    rt.tick()  # alice: produces "done". routes to bob.

    # alice's history should contain assistant + tool + user(from bob) + assistant
    alice_roles = [m.role for m in alice.history]
    assert alice_roles.count("tool") == 1
    # bob never received the tool result, only the assistant content.
    bob_inbound_contents = [m.content for m in bob.history if m.role == "user"]
    assert "5" not in bob_inbound_contents  # tool result didn't leak
    assert "thinking" in bob_inbound_contents


def test_run_rejects_zero_max_ticks() -> None:
    rt = Runtime([_echo_agent("a"), _echo_agent("b")])
    with pytest.raises(ValueError, match="max_ticks must be positive"):
        rt.run(max_ticks=0)


def test_inbox_for_returns_a_copy() -> None:
    """Mutating the returned list does not affect runtime state."""
    rt = Runtime([_echo_agent("a"), _echo_agent("b")])
    rt.send(to="a", content="hi")
    snapshot = rt.inbox_for("a")
    snapshot.clear()
    assert len(rt.inbox_for("a")) == 1


def test_empty_assistant_content_is_not_forwarded() -> None:
    """A tool-only assistant message (empty content) shouldn't dump an
    empty user message into the next agent's inbox."""
    call = ToolCall(id="c1", name="noop", arguments={})
    reg = ToolRegistry()
    reg.register(FunctionTool(spec=ToolSpec(name="noop", description=""), fn=lambda: "x"))
    alice = Agent(
        name="alice",
        policy=MockPolicy([Message(role="assistant", content="", tool_calls=(call,))]),
        tools=reg,
    )
    bob = _echo_agent("bob")

    rt = Runtime([alice, bob])
    rt.send(to="alice", content="go")
    rt.tick()

    assert rt.inbox_for("bob") == []
