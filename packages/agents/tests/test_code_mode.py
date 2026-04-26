"""Tests for Code Mode runtime (ADR-0013 / AGT-01)."""

from __future__ import annotations

import pytest
from saint_llm_agents import (
    CallablePolicy,
    CodeModePolicy,
    ExecutionResult,
    FunctionTool,
    InProcessExecutor,
    Message,
    MockPolicy,
    ToolRegistry,
    ToolSpec,
    generate_api_stub,
)


def _registry_with_math() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(FunctionTool(
        spec=ToolSpec(
            name="add",
            description="add two ints",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "first"},
                    "b": {"type": "integer", "description": "second"},
                },
                "required": ["a", "b"],
            },
        ),
        fn=lambda a, b: a + b,
    ))
    reg.register(FunctionTool(
        spec=ToolSpec(name="ping", description="say pong", parameters={}),
        fn=lambda: "pong",
    ))
    return reg


# ----- generate_api_stub ----------------------------------------------------


def test_generate_api_stub_empty_registry() -> None:
    stub = generate_api_stub([], namespace="tools")
    assert stub.names == ()
    assert "class tools" in stub.source


def test_generate_api_stub_emits_function_per_tool() -> None:
    specs = list(_registry_with_math().specs())
    stub = generate_api_stub(specs)
    assert set(stub.names) == {"add", "ping"}
    # Both function names appear in the source.
    assert "def add(" in stub.source
    assert "def ping(" in stub.source
    # Tool descriptions become docstrings.
    assert "add two ints" in stub.source
    assert "say pong" in stub.source


def test_generate_api_stub_renders_param_metadata() -> None:
    specs = list(_registry_with_math().specs())
    stub = generate_api_stub(specs)
    # Add tool's params should appear in the docstring with type hints.
    assert "a (integer)" in stub.source
    assert "b (integer)" in stub.source
    # Required marker present.
    assert "(required)" in stub.source


def test_generate_api_stub_no_params_documents_so() -> None:
    specs = list(_registry_with_math().specs())
    stub = generate_api_stub(specs)
    assert "No parameters." in stub.source


# ----- InProcessExecutor ----------------------------------------------------


def test_in_process_executor_runs_simple_tool_call() -> None:
    reg = _registry_with_math()
    executor = InProcessExecutor()
    code = "RESULT = tools.add(a=2, b=3)"
    result = executor.execute(code, registry=reg)
    assert result.ok
    assert result.return_value == 5
    assert result.tool_calls == [{"name": "add", "arguments": {"a": 2, "b": 3}}]


def test_in_process_executor_captures_stdout() -> None:
    reg = _registry_with_math()
    code = "print('hello'); RESULT = tools.ping()"
    result = InProcessExecutor().execute(code, registry=reg)
    assert result.ok
    assert "hello" in result.stdout
    assert result.return_value == "pong"


def test_in_process_executor_handles_unknown_tool() -> None:
    reg = _registry_with_math()
    code = "RESULT = tools.nonexistent()"
    result = InProcessExecutor().execute(code, registry=reg)
    assert not result.ok
    assert result.error_type == "AttributeError"
    assert "nonexistent" in (result.error_message or "")


def test_in_process_executor_catches_runtime_error() -> None:
    reg = _registry_with_math()
    code = "raise ValueError('boom')"
    result = InProcessExecutor().execute(code, registry=reg)
    assert not result.ok
    assert result.error_type == "ValueError"
    assert "boom" in (result.error_message or "")


def test_in_process_executor_no_result_assignment() -> None:
    """If RESULT is never assigned, return_value stays None."""
    reg = _registry_with_math()
    result = InProcessExecutor().execute("x = 1", registry=reg)
    assert result.ok
    assert result.return_value is None


def test_in_process_executor_sequential_tool_calls() -> None:
    """Multi-step Code Mode: one snippet calls multiple tools."""
    reg = _registry_with_math()
    code = (
        "x = tools.add(a=1, b=2)\n"
        "y = tools.add(a=x, b=10)\n"
        "RESULT = y\n"
    )
    result = InProcessExecutor().execute(code, registry=reg)
    assert result.ok
    assert result.return_value == 13
    assert len(result.tool_calls) == 2


# ----- CodeModePolicy -------------------------------------------------------


def test_code_mode_policy_executes_actor_snippet() -> None:
    """Mock actor returns a snippet; CodeModePolicy executes it and
    summarizes the result as an assistant message."""
    reg = _registry_with_math()
    actor = MockPolicy([
        Message(role="assistant", content="RESULT = tools.add(a=4, b=5)"),
    ])
    policy = CodeModePolicy(base_policy=actor, registry=reg)

    out = policy.act([Message(role="user", content="add 4 and 5")])
    assert out.role == "assistant"
    assert "9" in out.content  # return value rendered into content


def test_code_mode_policy_injects_api_stub() -> None:
    """The actor sees a system message containing the API stub."""
    reg = _registry_with_math()
    seen_messages: list[Message] = []

    def _capturing_actor(messages, *, tools):
        seen_messages.extend(messages)
        return Message(role="assistant", content="RESULT = 0")

    actor = CallablePolicy(_capturing_actor)
    policy = CodeModePolicy(base_policy=actor, registry=reg)
    policy.act([Message(role="user", content="hi")])

    sys_msgs = [m for m in seen_messages if m.role == "system"]
    assert any("class tools" in m.content for m in sys_msgs)
    assert any("def add(" in m.content for m in sys_msgs)


def test_code_mode_policy_surfaces_execution_error() -> None:
    reg = _registry_with_math()
    actor = MockPolicy([
        Message(role="assistant", content="raise RuntimeError('explicit fail')"),
    ])
    policy = CodeModePolicy(base_policy=actor, registry=reg)
    out = policy.act([Message(role="user", content="run something")])
    assert "RuntimeError" in out.content
    assert "explicit fail" in out.content


def test_code_mode_policy_rejects_non_assistant_actor_output() -> None:
    reg = _registry_with_math()

    actor = CallablePolicy(
        lambda *_a, **_kw: Message(role="user", content="oops"),
    )
    policy = CodeModePolicy(base_policy=actor, registry=reg)
    with pytest.raises(ValueError, match="must return assistant"):
        policy.act([Message(role="user", content="x")])


def test_code_mode_policy_uses_custom_executor() -> None:
    """A custom Executor returning a sentinel is honored."""
    reg = _registry_with_math()
    sentinel = ExecutionResult(
        ok=True, return_value="from-mock-executor", tool_calls=[],
    )

    class _MockExecutor:
        def execute(self, code, *, registry):
            return sentinel

    actor = MockPolicy([Message(role="assistant", content="anything")])
    policy = CodeModePolicy(base_policy=actor, registry=reg, executor=_MockExecutor())
    out = policy.act([Message(role="user", content="x")])
    assert "from-mock-executor" in out.content
