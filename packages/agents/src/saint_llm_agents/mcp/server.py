"""MCP server — wraps a :class:`ToolRegistry` and serves it via JSON-RPC.

Implements the minimum of the MCP spec needed for tool exposure:

* ``initialize`` — handshake; replies with server capabilities and
  protocol version.
* ``initialized`` notification — client signals it's ready (we accept
  but don't gate further requests on it).
* ``tools/list`` — list registered tools as MCP ``Tool`` objects.
* ``tools/call`` — invoke a tool by name with arguments; reply with
  the textual result.
* ``ping`` — round-trip check.

Notifications (no ``id``) are processed silently; requests get a
JSON-RPC response with the same ``id``. Unknown methods return
``-32601 Method not found`` per JSON-RPC. Tool errors are surfaced as
``isError: true`` in the ``tools/call`` result content (per MCP spec)
rather than as protocol-level errors.

Resources and prompts are out of scope for v0.0; if a client requests
them we return an empty list. That keeps the server compatible with
clients that probe for both even when only tools are wired up.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from saint_llm_agents.mcp.channel import JsonRpcChannel
from saint_llm_agents.message import ToolCall
from saint_llm_agents.tool import ToolRegistry

logger = logging.getLogger(__name__)

# MCP protocol version we implement. Clients quoting older versions
# still work — the server replies with its own version and the client
# decides whether to proceed.
MCP_PROTOCOL_VERSION = "2024-11-05"

# JSON-RPC error codes per spec.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class MCPServer:
    """Serve a :class:`ToolRegistry` over an :class:`JsonRpcChannel`.

    Args:
        registry:    the tool registry to expose.
        server_name: human-readable name reported in ``initialize``.
        server_version: server version string in ``initialize``.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        server_name: str = "saint-llm-agents",
        server_version: str = "0.0.1",
    ) -> None:
        self.registry = registry
        self.server_name = server_name
        self.server_version = server_version
        self._initialized = False

    def serve(self, channel: JsonRpcChannel) -> None:
        """Loop reading requests until the channel closes."""
        while True:
            msg = channel.receive()
            if msg is None:
                return
            self.handle_one(msg, channel)

    def handle_one(
        self, msg: Mapping[str, Any], channel: JsonRpcChannel,
    ) -> None:
        """Handle a single inbound message. Public so tests can drive it."""
        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        if method is None:
            # JSON-RPC response or malformed; ignore — server doesn't issue
            # client-bound requests yet.
            return

        is_notification = msg_id is None

        try:
            result = self._dispatch(method, params)
        except _MCPError as exc:
            if not is_notification:
                channel.send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": exc.code, "message": exc.message},
                })
            return
        except Exception as exc:
            logger.exception("mcp server: handler crashed for %s", method)
            if not is_notification:
                channel.send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": INTERNAL_ERROR,
                        "message": f"internal error: {type(exc).__name__}",
                    },
                })
            return

        if is_notification:
            return
        channel.send({"jsonrpc": "2.0", "id": msg_id, "result": result})

    def _dispatch(self, method: str, params: Mapping[str, Any]) -> Any:  # noqa: PLR0911
        if method == "initialize":
            return self._handle_initialize(params)
        if method in ("initialized", "notifications/initialized"):
            self._initialized = True
            return None
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": self._tools_list()}
        if method == "tools/call":
            return self._tools_call(params)
        if method == "resources/list":
            return {"resources": []}
        if method == "prompts/list":
            return {"prompts": []}
        raise _MCPError(METHOD_NOT_FOUND, f"method not found: {method}")

    def _handle_initialize(self, params: Mapping[str, Any]) -> dict[str, Any]:
        # Echo the client's requested protocol version back when reasonable;
        # otherwise we offer ours and let the client decide. For simplicity
        # always reply with our pinned version.
        del params
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
            "serverInfo": {
                "name": self.server_name,
                "version": self.server_version,
            },
        }

    def _tools_list(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for spec in self.registry.specs():
            schema = dict(spec.parameters) if spec.parameters else {
                "type": "object", "properties": {},
            }
            tools.append({
                "name": spec.name,
                "description": spec.description,
                "inputSchema": schema,
            })
        return tools

    def _tools_call(self, params: Mapping[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str):
            raise _MCPError(INVALID_PARAMS, "tools/call requires 'name'")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, Mapping):
            raise _MCPError(INVALID_PARAMS, "'arguments' must be an object")

        if name not in self.registry:
            return {
                "content": [{"type": "text", "text": f"unknown tool: {name}"}],
                "isError": True,
            }

        msg = self.registry.execute(
            ToolCall(id="mcp", name=name, arguments=dict(arguments)),
        )
        is_error = msg.content.startswith("ERROR:")
        return {
            "content": [{"type": "text", "text": msg.content}],
            "isError": is_error,
        }


class _MCPError(Exception):
    """Internal exception carrying a JSON-RPC error code + message."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
