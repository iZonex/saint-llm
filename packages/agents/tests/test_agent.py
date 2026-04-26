"""Tests for the Agent step / run loop including tool execution."""

from __future__ import annotations

import pytest
from saint_llm_agents import (
    Agent,
    CallablePolicy,
    FunctionTool,
    Message,
    MockPolicy,
    ToolCall,
    ToolRegistry,
    ToolSpec,
)


def _registry_with_add() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        FunctionTool(
            spec=ToolSpec(
                name="add",
                description="add",
                parameters={
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            ),
            fn=lambda a, b: a + b,
        ),
    )
    return reg


def test_agent_with_system_prompt_seeds_history() -> None:
    policy = MockPolicy([Message(role="assistant", content="ack")])
    agent = Agent(name="a1", policy=policy, system="You are helpful.")
    assert len(agent.history) == 1
    assert agent.history[0].role == "system"
    assert agent.history[0].content == "You are helpful."


def test_agent_step_appends_inbox_then_assistant() -> None:
    policy = MockPolicy([Message(role="assistant", content="hi back")])
    agent = Agent(name="a1", policy=policy)
    inbox = [Message(role="user", content="hi")]
    produced = agent.step(inbox)

    assert len(produced) == 1
    assert produced[0].role == "assistant"
    assert produced[0].content == "hi back"
    assert agent.history == [*inbox, produced[0]]


def test_agent_stamps_name_when_policy_did_not() -> None:
    policy = MockPolicy([Message(role="assistant", content="x")])
    agent = Agent(name="alice", policy=policy)
    [out] = agent.step([Message(role="user", content="ping")])
    assert out.name == "alice"


def test_agent_preserves_explicit_policy_name() -> None:
    """If the policy already named the message, don't overwrite it."""
    policy = MockPolicy([Message(role="assistant", content="x", name="custom")])
    agent = Agent(name="alice", policy=policy)
    [out] = agent.step([Message(role="user", content="ping")])
    assert out.name == "custom"


def test_agent_executes_tool_calls_and_appends_results() -> None:
    call = ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})
    responses = [
        Message(role="assistant", content="", tool_calls=(call,)),
        Message(role="assistant", content="answer is 5"),
    ]
    policy = MockPolicy(responses)
    agent = Agent(name="a1", policy=policy, tools=_registry_with_add())

    produced = agent.step([Message(role="user", content="add 2 and 3")])
    # First step: assistant with tool call, then tool result.
    assert len(produced) == 2
    assert produced[0].role == "assistant"
    assert produced[1].role == "tool"
    assert produced[1].content == "5"
    assert produced[1].tool_call_id == "c1"


def test_agent_run_loops_until_no_more_tool_calls() -> None:
    call = ToolCall(id="c1", name="add", arguments={"a": 4, "b": 1})
    policy = MockPolicy(
        [
            Message(role="assistant", content="thinking", tool_calls=(call,)),
            Message(role="assistant", content="final: 5"),
        ],
    )
    agent = Agent(name="a1", policy=policy, tools=_registry_with_add())
    history = agent.run(user_message="please add", max_steps=5)

    assert any(m.role == "tool" and m.content == "5" for m in history)
    final = history[-1]
    assert final.role == "assistant"
    assert "final" in final.content


def test_agent_run_respects_max_steps() -> None:
    """If the policy keeps emitting tool calls, max_steps caps the loop."""
    call = ToolCall(id="c1", name="add", arguments={"a": 1, "b": 1})
    # Three tool-call rounds; max_steps=2 stops after the second.
    policy = MockPolicy(
        [Message(role="assistant", content="", tool_calls=(call,))] * 3,
    )
    agent = Agent(name="a1", policy=policy, tools=_registry_with_add())
    agent.run(user_message="loop", max_steps=2)

    # 1 user + 2 assistants + 2 tool results = 5 history entries (no system).
    assert sum(1 for m in agent.history if m.role == "assistant") == 2


def test_agent_step_rejects_non_assistant_policy_output() -> None:
    """A misbehaving policy that returns role='user' must blow up loudly."""
    policy = CallablePolicy(lambda *_a, **_k: Message(role="user", content="oops"))
    agent = Agent(name="a1", policy=policy)
    with pytest.raises(ValueError, match="must be 'assistant'"):
        agent.step([Message(role="user", content="x")])


def test_agent_run_rejects_zero_max_steps() -> None:
    policy = MockPolicy([Message(role="assistant", content="")])
    agent = Agent(name="a1", policy=policy)
    with pytest.raises(ValueError, match="max_steps must be positive"):
        agent.run(user_message="hi", max_steps=0)


def test_two_agents_can_be_composed_manually() -> None:
    """Without a Runtime yet, two agents communicate by passing messages
    from one's output into the other's inbox.
    """
    a_policy = MockPolicy([Message(role="assistant", content="hello bob")])
    b_policy = CallablePolicy(
        lambda messages, *, tools: Message(
            role="assistant",
            content=f"received: {messages[-1].content}",
        ),
    )

    alice = Agent(name="alice", policy=a_policy)
    bob = Agent(name="bob", policy=b_policy)

    alice_out = alice.step([Message(role="user", content="say hi to bob")])
    # Convert alice's assistant message into a user message for bob (the
    # convention: cross-agent traffic looks like a user turn to the receiver).
    bob_inbox = [Message(role="user", content=alice_out[0].content)]
    bob_out = bob.step(bob_inbox)

    assert bob_out[0].content == "received: hello bob"
    assert bob_out[0].name == "bob"
