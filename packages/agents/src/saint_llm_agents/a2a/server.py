"""A2A server — wraps an :class:`Agent` and serves it via JSON-RPC.

Implements the minimum set of A2A spec methods:

* ``tasks/send``    — create + run a new task. Returns the completed
  task in a single round-trip (synchronous; no worker pool).
* ``tasks/get``     — fetch a task by ID.
* ``tasks/cancel``  — mark a task canceled (no effect on already-
  completed tasks).
* ``ping``          — round-trip check.

Task lifecycle states (per the A2A spec): ``submitted`` →
``working`` → ``completed`` / ``failed`` / ``canceled``.

Resources / push-notifications / streaming subscribe are out of scope
for v0.0; ``tasks/send`` runs the agent synchronously so the client
sees ``working``-then-``completed`` collapsed into a single reply.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from saint_llm_agents.agent import Agent
from saint_llm_agents.mcp.channel import JsonRpcChannel
from saint_llm_agents.message import Message

logger = logging.getLogger(__name__)

A2A_PROTOCOL_VERSION = "2025-04-01"

# JSON-RPC error codes per spec.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# A2A-specific application error codes (in the JSON-RPC -32000 range).
TASK_NOT_FOUND = -32001

TaskStatus = Literal["submitted", "working", "completed", "failed", "canceled"]


@dataclass
class A2ATask:
    """One A2A task record.

    Attributes:
        id:        unique task ID.
        sessionId: optional client-supplied conversation grouping ID.
        status:    one of submitted / working / completed / failed /
            canceled.
        message:   the inbound user message text.
        artifacts: list of dicts describing the agent's output. For
            v0.0 each artifact has ``{"type": "text", "text": ...}``.
        error:     populated when status is ``failed``.
    """

    id: str
    sessionId: str | None
    status: TaskStatus
    message: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sessionId": self.sessionId,
            "status": self.status,
            "message": self.message,
            "artifacts": list(self.artifacts),
            "error": self.error,
        }


class InMemoryTaskStore:
    """Dict-backed task index. Production deployments swap for a DB."""

    def __init__(self) -> None:
        self._tasks: dict[str, A2ATask] = {}

    def put(self, task: A2ATask) -> None:
        self._tasks[task.id] = task

    def get(self, task_id: str) -> A2ATask | None:
        return self._tasks.get(task_id)

    def __len__(self) -> int:
        return len(self._tasks)


class A2AServer:
    """Serve a :class:`Agent` over an :class:`JsonRpcChannel`.

    Args:
        agent:          the Agent to run on incoming tasks.
        store:          task storage; defaults to in-memory.
        max_steps:      cap passed to ``agent.run`` per task.
        server_name:    name reported on initialize-style probes.
        server_version: version string.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        store: InMemoryTaskStore | None = None,
        max_steps: int = 8,
        server_name: str = "saint-llm-a2a",
        server_version: str = "0.0.1",
    ) -> None:
        self.agent = agent
        self.store = store if store is not None else InMemoryTaskStore()
        self.max_steps = max_steps
        self.server_name = server_name
        self.server_version = server_version

    def serve(self, channel: JsonRpcChannel) -> None:
        while True:
            msg = channel.receive()
            if msg is None:
                return
            self.handle_one(msg, channel)

    def handle_one(
        self, msg: Mapping[str, Any], channel: JsonRpcChannel,
    ) -> None:
        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        if method is None:
            return

        is_notification = msg_id is None

        try:
            result = self._dispatch(method, params)
        except _A2AError as exc:
            if not is_notification:
                channel.send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": exc.code, "message": exc.message},
                })
            return
        except Exception as exc:
            logger.exception("a2a server: handler crashed for %s", method)
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

    def _dispatch(self, method: str, params: Mapping[str, Any]) -> Any:
        if method == "ping":
            return {}
        if method == "agent/info":
            return self._agent_info()
        if method == "tasks/send":
            return self._tasks_send(params)
        if method == "tasks/get":
            return self._tasks_get(params)
        if method == "tasks/cancel":
            return self._tasks_cancel(params)
        raise _A2AError(METHOD_NOT_FOUND, f"method not found: {method}")

    def _agent_info(self) -> dict[str, Any]:
        return {
            "name": self.agent.name,
            "tools": [
                {"name": s.name, "description": s.description}
                for s in self.agent.tools.specs()
            ],
            "protocolVersion": A2A_PROTOCOL_VERSION,
            "serverInfo": {
                "name": self.server_name,
                "version": self.server_version,
            },
        }

    def _tasks_send(self, params: Mapping[str, Any]) -> dict[str, Any]:
        message = params.get("message")
        if not isinstance(message, str) or not message:
            raise _A2AError(
                INVALID_PARAMS, "tasks/send requires non-empty 'message'",
            )
        session_id = params.get("sessionId")
        task_id = str(params.get("id") or uuid.uuid4())
        task = A2ATask(
            id=task_id,
            sessionId=session_id if isinstance(session_id, str) else None,
            status="working",
            message=message,
        )
        self.store.put(task)

        try:
            history = self.agent.run(
                user_message=message, max_steps=self.max_steps,
            )
        except Exception as exc:
            task.status = "failed"
            task.error = f"{type(exc).__name__}: {exc}"
            return task.to_dict()

        # The artifact is the last assistant response (the agent's
        # "answer" — the message that emitted no further tool calls).
        artifact_text = _extract_final_response(history)
        task.artifacts = [{"type": "text", "text": artifact_text}]
        task.status = "completed"
        return task.to_dict()

    def _tasks_get(self, params: Mapping[str, Any]) -> dict[str, Any]:
        task_id = params.get("id")
        if not isinstance(task_id, str):
            raise _A2AError(INVALID_PARAMS, "tasks/get requires 'id'")
        task = self.store.get(task_id)
        if task is None:
            raise _A2AError(TASK_NOT_FOUND, f"unknown task: {task_id}")
        return task.to_dict()

    def _tasks_cancel(self, params: Mapping[str, Any]) -> dict[str, Any]:
        task_id = params.get("id")
        if not isinstance(task_id, str):
            raise _A2AError(INVALID_PARAMS, "tasks/cancel requires 'id'")
        task = self.store.get(task_id)
        if task is None:
            raise _A2AError(TASK_NOT_FOUND, f"unknown task: {task_id}")
        if task.status not in ("completed", "failed", "canceled"):
            task.status = "canceled"
        return task.to_dict()


def _extract_final_response(history: list[Message]) -> str:
    """Pull the last assistant message that has no pending tool_calls."""
    for msg in reversed(history):
        if msg.role == "assistant" and not msg.tool_calls:
            return msg.content
    # Fallback — return the last assistant message even if it had tool
    # calls, so the client at least sees something.
    for msg in reversed(history):
        if msg.role == "assistant":
            return msg.content
    return ""


class _A2AError(Exception):
    """Internal exception carrying a JSON-RPC error code + message."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
