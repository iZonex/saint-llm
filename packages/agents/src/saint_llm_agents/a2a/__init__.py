"""A2A (Agent-to-Agent) protocol adapter for saint-llm agents.

While MCP is "agent <-> tool", A2A is "agent <-> agent": one agent
asks another to perform a task and waits for the result. The protocol
is JSON-RPC 2.0 with three core methods:

* ``tasks/send`` — submit a new task (a message addressed to the
  remote agent). The server runs the agent until it produces an
  assistant response with no further tool calls and returns the
  task with status ``"completed"`` plus an artifact holding the
  agent's output.
* ``tasks/get`` — fetch a previously-created task by ID.
* ``tasks/cancel`` — mark a task as canceled (best-effort; the
  server is single-threaded so this only matters across calls).

Tasks are persisted in a per-server :class:`InMemoryTaskStore` keyed
by task ID. The server's ``tasks/send`` runs the agent inline (no
async pool yet — v0.0.1 polish item) so the client gets a completed
task on its first round-trip.

Reuses :class:`JsonRpcChannel` from :mod:`saint_llm_agents.mcp.channel`
so the same transport (in-memory pair / stdio / etc.) works for both.
"""

from saint_llm_agents.a2a.client import A2AClient, A2AClientError, send_to_agent
from saint_llm_agents.a2a.server import (
    A2AServer,
    A2ATask,
    InMemoryTaskStore,
    TaskStatus,
)

__all__ = [
    "A2AClient",
    "A2AClientError",
    "A2AServer",
    "A2ATask",
    "InMemoryTaskStore",
    "TaskStatus",
    "send_to_agent",
]
