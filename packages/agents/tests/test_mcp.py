"""Tests for the MCP (Model Context Protocol) adapter."""

from __future__ import annotations

import io
import threading
from typing import Any

import pytest
from saint_llm_agents import FunctionTool, ToolRegistry, ToolSpec
from saint_llm_agents.mcp import (
    InMemoryJsonRpcChannel,
    MCPServer,
    StdioJsonRpcChannel,
    mcp_client_tools,
    mcp_list_tools,
    pair_channels,
)
from saint_llm_agents.mcp.client import MCPClientError, _MCPClient


def _registry_with_add() -> ToolRegistry:
    spec = ToolSpec(
        name="add",
        description="add two ints",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    )
    reg = ToolRegistry()
    reg.register(FunctionTool(spec=spec, fn=lambda a, b: a + b))
    return reg


def _spawn_server(reg: ToolRegistry) -> tuple[InMemoryJsonRpcChannel, threading.Thread, MCPServer]:
    client_ch, server_ch = pair_channels()
    server = MCPServer(reg)
    thread = threading.Thread(target=server.serve, args=(server_ch,), daemon=True)
    thread.start()
    return client_ch, thread, server


def _stop_server(channel: InMemoryJsonRpcChannel, thread: threading.Thread) -> None:
    channel.close()
    thread.join(timeout=2.0)


def test_mcp_initialize_returns_protocol_and_capabilities() -> None:
    client_ch, thread, _ = _spawn_server(_registry_with_add())
    try:
        client = _MCPClient(client_ch)
        result = client.request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0.0"},
        })
        assert "protocolVersion" in result
        assert "tools" in result["capabilities"]
        assert result["serverInfo"]["name"] == "saint-llm-agents"
    finally:
        _stop_server(client_ch, thread)


def test_mcp_tools_list_exposes_registry() -> None:
    client_ch, thread, _ = _spawn_server(_registry_with_add())
    try:
        specs = mcp_list_tools(client_ch)
        assert len(specs) == 1
        assert specs[0].name == "add"
        assert "two ints" in specs[0].description
        assert specs[0].parameters["required"] == ["a", "b"]
    finally:
        _stop_server(client_ch, thread)


def test_mcp_tools_call_invokes_registry_tool() -> None:
    client_ch, thread, _ = _spawn_server(_registry_with_add())
    try:
        tools = mcp_client_tools(client_ch)
        assert len(tools) == 1
        result = tools[0](a=2, b=3)
        assert result == "5"
    finally:
        _stop_server(client_ch, thread)


def test_mcp_tools_call_unknown_tool_returns_is_error() -> None:
    client_ch, thread, _ = _spawn_server(ToolRegistry())
    try:
        client = _MCPClient(client_ch)
        client.initialize()
        result = client.request("tools/call", {"name": "missing", "arguments": {}})
        assert result["isError"] is True
        assert "missing" in result["content"][0]["text"]
    finally:
        _stop_server(client_ch, thread)


def test_mcp_tools_call_tool_exception_is_serialised_as_error_text() -> None:
    reg = ToolRegistry()
    reg.register(FunctionTool(
        spec=ToolSpec(name="boom", description="explodes", parameters={}),
        fn=lambda: (_ for _ in ()).throw(RuntimeError("nope")),
    ))
    client_ch, thread, _ = _spawn_server(reg)
    try:
        tools = mcp_client_tools(client_ch)
        text = tools[0]()
        assert text.startswith("ERROR: RuntimeError")
        assert "nope" in text
    finally:
        _stop_server(client_ch, thread)


def test_mcp_method_not_found_returns_jsonrpc_error() -> None:
    client_ch, thread, _ = _spawn_server(ToolRegistry())
    try:
        client = _MCPClient(client_ch)
        with pytest.raises(MCPClientError, match="-32601"):
            client.request("does/not/exist", {})
    finally:
        _stop_server(client_ch, thread)


def test_mcp_resources_and_prompts_return_empty_lists() -> None:
    client_ch, thread, _ = _spawn_server(_registry_with_add())
    try:
        client = _MCPClient(client_ch)
        client.initialize()
        assert client.request("resources/list", {}) == {"resources": []}
        assert client.request("prompts/list", {}) == {"prompts": []}
    finally:
        _stop_server(client_ch, thread)


def test_mcp_notification_does_not_get_response() -> None:
    """The 'initialized' notification has no id; server must stay silent."""
    client_ch, server_ch = pair_channels()
    server = MCPServer(ToolRegistry())
    server.handle_one(
        {"jsonrpc": "2.0", "method": "notifications/initialized"}, server_ch,
    )
    # Nothing should have been written to the server's outbox.
    assert client_ch._inbox.empty()


def test_mcp_client_name_prefix_namespaces_remote_tools() -> None:
    client_ch, thread, _ = _spawn_server(_registry_with_add())
    try:
        tools = mcp_client_tools(client_ch, name_prefix="external.")
        assert tools[0].spec.name == "external.add"
        assert tools[0](a=1, b=4) == "5"
    finally:
        _stop_server(client_ch, thread)


def test_mcp_ping_works() -> None:
    client_ch, thread, _ = _spawn_server(ToolRegistry())
    try:
        client = _MCPClient(client_ch)
        assert client.request("ping", {}) == {}
    finally:
        _stop_server(client_ch, thread)


def test_stdio_channel_round_trip() -> None:
    """StdioJsonRpcChannel: encode/decode round-trip across two file objects."""
    a_to_b = io.StringIO()
    b_to_a = io.StringIO()
    side_a = StdioJsonRpcChannel(reader=b_to_a, writer=a_to_b)
    msg: dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    side_a.send(msg)
    a_to_b.seek(0)
    side_b = StdioJsonRpcChannel(reader=a_to_b, writer=b_to_a)
    received = side_b.receive()
    assert received == msg


def test_stdio_channel_returns_none_at_eof() -> None:
    ch = StdioJsonRpcChannel(reader=io.StringIO(""), writer=io.StringIO())
    assert ch.receive() is None
