"""Tests for the A2A (Agent-to-Agent) protocol adapter."""

from __future__ import annotations

import threading

import pytest
from saint_llm_agents import (
    Agent,
    CallablePolicy,
    Message,
    MockPolicy,
)
from saint_llm_agents.a2a import (
    A2AClient,
    A2AClientError,
    A2AServer,
    InMemoryTaskStore,
    send_to_agent,
)
from saint_llm_agents.mcp.channel import InMemoryJsonRpcChannel, pair_channels


def _spawn_server(agent: Agent) -> tuple[InMemoryJsonRpcChannel, threading.Thread, A2AServer]:
    client_ch, server_ch = pair_channels()
    server = A2AServer(agent)
    thread = threading.Thread(target=server.serve, args=(server_ch,), daemon=True)
    thread.start()
    return client_ch, thread, server


def _stop(channel: InMemoryJsonRpcChannel, thread: threading.Thread) -> None:
    channel.close()
    thread.join(timeout=2.0)


def _echo_agent(name: str = "echo") -> Agent:
    """Build a tiny Agent whose policy echoes the last user message."""
    def echo(history, *, tools):  # type: ignore[no-untyped-def]
        del tools
        last_user = next(
            (m.content for m in reversed(history) if m.role == "user"),
            "",
        )
        return Message(role="assistant", content=f"echo: {last_user}")

    return Agent(name=name, policy=CallablePolicy(echo))


def test_a2a_send_task_returns_completed_task() -> None:
    client_ch, thread, _ = _spawn_server(_echo_agent())
    try:
        client = A2AClient(client_ch)
        task = client.send_task("hello world")
        assert task["status"] == "completed"
        assert task["message"] == "hello world"
        assert task["artifacts"][0]["text"] == "echo: hello world"
        assert task["error"] is None
    finally:
        _stop(client_ch, thread)


def test_a2a_send_task_assigns_id_when_omitted() -> None:
    client_ch, thread, _ = _spawn_server(_echo_agent())
    try:
        client = A2AClient(client_ch)
        task = client.send_task("ping")
        assert isinstance(task["id"], str)
        assert task["id"] != ""
    finally:
        _stop(client_ch, thread)


def test_a2a_send_task_uses_caller_supplied_id() -> None:
    client_ch, thread, _ = _spawn_server(_echo_agent())
    try:
        client = A2AClient(client_ch)
        task = client.send_task("ping", task_id="my-id-123")
        assert task["id"] == "my-id-123"
    finally:
        _stop(client_ch, thread)


def test_a2a_send_task_threads_session_id() -> None:
    client_ch, thread, _ = _spawn_server(_echo_agent())
    try:
        client = A2AClient(client_ch)
        task = client.send_task("ping", session_id="sess-7")
        assert task["sessionId"] == "sess-7"
    finally:
        _stop(client_ch, thread)


def test_a2a_get_task_returns_stored_task() -> None:
    client_ch, thread, _ = _spawn_server(_echo_agent())
    try:
        client = A2AClient(client_ch)
        sent = client.send_task("yo", task_id="t1")
        fetched = client.get_task("t1")
        assert fetched == sent
    finally:
        _stop(client_ch, thread)


def test_a2a_get_task_unknown_id_returns_error() -> None:
    client_ch, thread, _ = _spawn_server(_echo_agent())
    try:
        client = A2AClient(client_ch)
        with pytest.raises(A2AClientError, match="-32001"):
            client.get_task("does-not-exist")
    finally:
        _stop(client_ch, thread)


def test_a2a_cancel_task_marks_canceled_only_when_in_flight() -> None:
    client_ch, thread, _ = _spawn_server(_echo_agent())
    try:
        client = A2AClient(client_ch)
        # Send finishes synchronously -> task already completed.
        completed = client.send_task("done", task_id="t1")
        assert completed["status"] == "completed"
        result = client.cancel_task("t1")
        # Already-completed tasks can't be canceled retroactively.
        assert result["status"] == "completed"
    finally:
        _stop(client_ch, thread)


def test_a2a_cancel_unknown_task_returns_error() -> None:
    client_ch, thread, _ = _spawn_server(_echo_agent())
    try:
        client = A2AClient(client_ch)
        with pytest.raises(A2AClientError, match="-32001"):
            client.cancel_task("nope")
    finally:
        _stop(client_ch, thread)


def test_a2a_send_empty_message_rejects_with_invalid_params() -> None:
    client_ch, thread, _ = _spawn_server(_echo_agent())
    try:
        client = A2AClient(client_ch)
        with pytest.raises(A2AClientError, match="-32602"):
            client.send_task("")
    finally:
        _stop(client_ch, thread)


def test_a2a_method_not_found_returns_jsonrpc_error() -> None:
    client_ch, thread, _ = _spawn_server(_echo_agent())
    try:
        client = A2AClient(client_ch)
        with pytest.raises(A2AClientError, match="-32601"):
            client.request("does/not/exist", {})
    finally:
        _stop(client_ch, thread)


def test_a2a_ping_works() -> None:
    client_ch, thread, _ = _spawn_server(_echo_agent())
    try:
        client = A2AClient(client_ch)
        assert client.request("ping", {}) == {}
    finally:
        _stop(client_ch, thread)


def test_a2a_agent_info_includes_name_and_tools() -> None:
    client_ch, thread, _ = _spawn_server(_echo_agent("greeter"))
    try:
        client = A2AClient(client_ch)
        info = client.agent_info()
        assert info["name"] == "greeter"
        assert "protocolVersion" in info
        assert info["serverInfo"]["name"] == "saint-llm-a2a"
    finally:
        _stop(client_ch, thread)


def test_send_to_agent_returns_artifact_text() -> None:
    """Convenience helper: send + read first text artifact in one call."""
    client_ch, thread, _ = _spawn_server(_echo_agent())
    try:
        text = send_to_agent(client_ch, "hi there")
        assert text == "echo: hi there"
    finally:
        _stop(client_ch, thread)


def test_a2a_failing_agent_marks_task_failed() -> None:
    """If agent.run raises, the task ends in 'failed' with error string."""
    def bad(history, *, tools):  # type: ignore[no-untyped-def]
        del history, tools
        raise RuntimeError("policy crashed")

    agent = Agent(name="broken", policy=CallablePolicy(bad))
    client_ch, thread, _ = _spawn_server(agent)
    try:
        client = A2AClient(client_ch)
        task = client.send_task("anything")
        assert task["status"] == "failed"
        assert "policy crashed" in task["error"]
    finally:
        _stop(client_ch, thread)


def test_a2a_uses_external_task_store() -> None:
    store = InMemoryTaskStore()
    agent = _echo_agent()
    client_ch, server_ch = pair_channels()
    server = A2AServer(agent, store=store)
    thread = threading.Thread(target=server.serve, args=(server_ch,), daemon=True)
    thread.start()
    try:
        client = A2AClient(client_ch)
        client.send_task("hello", task_id="mine")
        # External store should now hold the task.
        assert len(store) == 1
        assert store.get("mine") is not None
    finally:
        client_ch.close()
        thread.join(timeout=2.0)


def test_a2a_with_real_mockpolicy_agent() -> None:
    """A2A works with the canonical MockPolicy too — not just CallablePolicy."""
    agent = Agent(
        name="mock",
        policy=MockPolicy([Message(role="assistant", content="hello back")]),
    )
    client_ch, thread, _ = _spawn_server(agent)
    try:
        client = A2AClient(client_ch)
        task = client.send_task("hello")
        assert task["artifacts"][0]["text"] == "hello back"
    finally:
        _stop(client_ch, thread)
