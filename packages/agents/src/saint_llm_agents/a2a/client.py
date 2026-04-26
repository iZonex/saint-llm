"""A2A client — submit tasks to a remote agent server.

Synchronous request/reply over a :class:`JsonRpcChannel`.  ``send_task``
issues ``tasks/send`` and returns the completed task in one
round-trip (the v0.0 server runs the agent inline). For pollable async
operation use ``get_task`` after dispatch.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from saint_llm_agents.mcp.channel import JsonRpcChannel


class A2AClientError(RuntimeError):
    """Raised when the remote A2A server returns a JSON-RPC error."""


class A2AClient:
    """Thin synchronous A2A client. One per channel."""

    def __init__(self, channel: JsonRpcChannel) -> None:
        self._channel = channel
        self._next_id = 1

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
                raise A2AClientError(
                    f"channel closed before reply to {method}",
                )
            if reply.get("id") != msg_id:
                continue
            if "error" in reply:
                err = reply["error"]
                raise A2AClientError(
                    f"{method}: code={err.get('code')} {err.get('message')}",
                )
            return reply.get("result")

    def send_task(
        self,
        message: str,
        *,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Issue ``tasks/send`` and return the resulting task dict."""
        params: dict[str, Any] = {"message": message}
        if session_id is not None:
            params["sessionId"] = session_id
        if task_id is not None:
            params["id"] = task_id
        result = self.request("tasks/send", params)
        if not isinstance(result, Mapping):
            raise A2AClientError("tasks/send returned non-object result")
        return dict(result)

    def get_task(self, task_id: str) -> dict[str, Any]:
        result = self.request("tasks/get", {"id": task_id})
        if not isinstance(result, Mapping):
            raise A2AClientError("tasks/get returned non-object result")
        return dict(result)

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        result = self.request("tasks/cancel", {"id": task_id})
        if not isinstance(result, Mapping):
            raise A2AClientError("tasks/cancel returned non-object result")
        return dict(result)

    def agent_info(self) -> dict[str, Any]:
        result = self.request("agent/info", {})
        if not isinstance(result, Mapping):
            raise A2AClientError("agent/info returned non-object result")
        return dict(result)


def send_to_agent(
    channel: JsonRpcChannel,
    message: str,
    *,
    session_id: str | None = None,
) -> str:
    """Convenience: open a client, send one task, return the artifact text.

    For multi-task or polling-style workflows use :class:`A2AClient`
    directly.
    """
    client = A2AClient(channel)
    task = client.send_task(message, session_id=session_id)
    artifacts = task.get("artifacts") or []
    parts = [
        str(item.get("text", ""))
        for item in artifacts
        if isinstance(item, Mapping) and item.get("type") == "text"
    ]
    return "\n".join(parts)
