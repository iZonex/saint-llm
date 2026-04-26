"""Transport channels for MCP JSON-RPC.

MCP messages are JSON-RPC 2.0 objects framed one-per-line over the
chosen transport. This module abstracts the framing so server / client
code is independent of stdio vs in-memory queues.

Three concrete channels:

* :class:`InMemoryJsonRpcChannel` — pair of asyncio-free queues; tests
  use :func:`pair_channels` to wire a client and server in the same
  process.
* :class:`StdioJsonRpcChannel` — line-delimited JSON over stdin/stdout
  (the standard MCP transport for local servers).
* Custom subclasses can wrap sockets, pipes, or websockets.

The channel API is intentionally minimal: send one JSON object,
receive one JSON object. Both halves are blocking; concurrency is the
caller's responsibility.
"""

from __future__ import annotations

import json
import queue
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import IO, Any


class JsonRpcChannel(ABC):
    """Abstract bidirectional JSON-RPC channel."""

    @abstractmethod
    def send(self, message: Mapping[str, Any]) -> None:
        """Send one JSON-RPC object."""

    @abstractmethod
    def receive(self) -> Mapping[str, Any] | None:
        """Receive one JSON-RPC object, or None if the channel is closed."""

    def close(self) -> None:  # noqa: B027
        """Best-effort close; default is a no-op."""


class InMemoryJsonRpcChannel(JsonRpcChannel):
    """Two queues: ``inbox`` reads, ``outbox`` writes.

    Use :func:`pair_channels` to construct a matched (client, server)
    pair where each one's outbox is the other's inbox.
    """

    def __init__(
        self,
        inbox: queue.Queue[Mapping[str, Any] | None],
        outbox: queue.Queue[Mapping[str, Any] | None],
    ) -> None:
        self._inbox = inbox
        self._outbox = outbox
        self._closed = False

    def send(self, message: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("channel is closed")
        self._outbox.put(dict(message))

    def receive(self) -> Mapping[str, Any] | None:
        if self._closed:
            return None
        msg = self._inbox.get()
        if msg is None:
            self._closed = True
            return None
        return msg

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._outbox.put(None)


def pair_channels() -> tuple[InMemoryJsonRpcChannel, InMemoryJsonRpcChannel]:
    """Build a matched ``(client_side, server_side)`` channel pair."""
    a_to_b: queue.Queue[Mapping[str, Any] | None] = queue.Queue()
    b_to_a: queue.Queue[Mapping[str, Any] | None] = queue.Queue()
    client = InMemoryJsonRpcChannel(inbox=b_to_a, outbox=a_to_b)
    server = InMemoryJsonRpcChannel(inbox=a_to_b, outbox=b_to_a)
    return client, server


class StdioJsonRpcChannel(JsonRpcChannel):
    """Line-delimited JSON over a pair of file objects.

    The default reads stdin and writes stdout — the MCP server's
    standard local transport. Tests can pass arbitrary text streams.
    """

    def __init__(
        self,
        reader: IO[str] | None = None,
        writer: IO[str] | None = None,
    ) -> None:
        self._reader = reader if reader is not None else sys.stdin
        self._writer = writer if writer is not None else sys.stdout

    def send(self, message: Mapping[str, Any]) -> None:
        line = json.dumps(message, separators=(",", ":"))
        self._writer.write(line)
        self._writer.write("\n")
        self._writer.flush()

    def receive(self) -> Mapping[str, Any] | None:
        line = self._reader.readline()
        if not line:
            return None
        return json.loads(line)
