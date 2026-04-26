"""MCP client — consume an external MCP server's tools as local Tools.

Workflow:

* Open a :class:`JsonRpcChannel` to the remote server (stdio, HTTP, etc.).
* :func:`mcp_list_tools` performs the ``initialize`` handshake and the
  ``tools/list`` query, returning the remote :class:`ToolSpec` set.
* :func:`mcp_client_tools` wraps each remote spec in a local
  :class:`FunctionTool` whose ``__call__`` issues ``tools/call`` and
  returns the textual content (raising on protocol-level errors).

The client is fully synchronous: each request blocks for its response.
The server is single-threaded, so request IDs are issued sequentially
from a private counter.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from saint_llm_agents.mcp.channel import JsonRpcChannel
from saint_llm_agents.mcp.server import MCP_PROTOCOL_VERSION
from saint_llm_agents.tool import FunctionTool, Tool, ToolSpec


class MCPClientError(RuntimeError):
    """Raised when the remote server returns a JSON-RPC error."""


class _MCPClient:
    """Internal request/response driver. One per channel."""

    def __init__(self, channel: JsonRpcChannel) -> None:
        self._channel = channel
        self._next_id = 1
        self._initialized = False

    def initialize(self, *, client_name: str = "saint-llm-agents") -> None:
        if self._initialized:
            return
        self.request("initialize", {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": client_name, "version": "0.0.1"},
        })
        self.notify("notifications/initialized", {})
        self._initialized = True

    def request(
        self, method: str, params: Mapping[str, Any] | None = None,
    ) -> Any:
        msg_id = self._next_id
        self._next_id += 1
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            msg["params"] = dict(params)
        self._channel.send(msg)
        while True:
            reply = self._channel.receive()
            if reply is None:
                raise MCPClientError(f"channel closed before reply to {method}")
            # Skip notifications / mismatched IDs (we don't expect any).
            if reply.get("id") != msg_id:
                continue
            if "error" in reply:
                err = reply["error"]
                raise MCPClientError(
                    f"{method}: code={err.get('code')} {err.get('message')}",
                )
            return reply.get("result")

    def notify(
        self, method: str, params: Mapping[str, Any] | None = None,
    ) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = dict(params)
        self._channel.send(msg)


def mcp_list_tools(channel: JsonRpcChannel) -> tuple[ToolSpec, ...]:
    """Initialize the channel and return the remote tool specs."""
    client = _MCPClient(channel)
    client.initialize()
    result = client.request("tools/list", {})
    raw_tools = result.get("tools", []) if isinstance(result, Mapping) else []
    return tuple(
        ToolSpec(
            name=raw["name"],
            description=raw.get("description", ""),
            parameters=raw.get("inputSchema", {}),
        )
        for raw in raw_tools
    )


def mcp_client_tools(
    channel: JsonRpcChannel,
    *,
    name_prefix: str = "",
) -> tuple[Tool, ...]:
    """Return the remote server's tools as local :class:`FunctionTool` objects.

    Args:
        channel:     a connected JSON-RPC channel to the remote server.
        name_prefix: optional prefix to namespace remote tools (e.g.
            ``"github."``) when merging multiple MCP servers into one
            local registry. Empty by default.

    Each returned tool, when called, issues a ``tools/call`` request and
    returns the joined ``content`` text. Protocol errors raise
    :class:`MCPClientError`; tool-level errors (``isError: true``) are
    returned as the text result so the caller / agent can read them.
    """
    client = _MCPClient(channel)
    client.initialize()
    result = client.request("tools/list", {})
    raw_tools = result.get("tools", []) if isinstance(result, Mapping) else []

    tools: list[Tool] = []
    for raw in raw_tools:
        remote_name = raw["name"]
        local_name = f"{name_prefix}{remote_name}" if name_prefix else remote_name
        spec = ToolSpec(
            name=local_name,
            description=raw.get("description", ""),
            parameters=raw.get("inputSchema", {}),
        )
        tools.append(FunctionTool(
            spec=spec,
            fn=_make_remote_caller(client, remote_name),
        ))
    return tuple(tools)


def _make_remote_caller(client: _MCPClient, remote_name: str) -> Any:
    def call(**kwargs: Any) -> str:
        result = client.request("tools/call", {
            "name": remote_name, "arguments": kwargs,
        })
        if not isinstance(result, Mapping):
            return ""
        parts = [
            str(item.get("text", ""))
            for item in result.get("content", [])
            if isinstance(item, Mapping) and item.get("type") == "text"
        ]
        return "\n".join(parts)

    return call
