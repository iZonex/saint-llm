"""Message format for inter-agent and agent↔tool communication.

Mirrors the OpenAI / Anthropic chat-tool-use convention: every turn is a
:class:`Message` with a role tag (``user`` / ``assistant`` / ``system`` /
``tool``). Assistant messages may carry a tuple of :class:`ToolCall`;
``tool`` messages reference the call they answer via ``tool_call_id``.

Frozen dataclasses so they're hashable and safe to share across agents.
``arguments`` is intentionally a plain ``dict[str, Any]`` rather than a
typed payload — JSON Schema validation lives in the tool layer where it
belongs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """A request from an assistant to invoke one tool.

    Attributes:
        id:        unique identifier so the runtime can pair a result
            message back to this call. Caller assigns; conventionally
            short opaque tokens like ``"call_a3f9"``.
        name:      tool name as registered in :class:`ToolRegistry`.
        arguments: keyword arguments passed to the tool. The runtime is
            responsible for validating these against the tool's
            JSON Schema before executing.
    """

    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    """One conversation turn.

    Attributes:
        role:         conversation role. Standard values are ``"user"``,
            ``"assistant"``, ``"system"``, and ``"tool"``. Multi-agent
            scenarios may also use the sender's agent name as the role.
        content:      text content. Empty string allowed (an assistant
            message that *only* invokes tools will have empty content
            and a non-empty ``tool_calls``).
        tool_calls:   tools the assistant wants the runtime to invoke.
            Only meaningful when ``role == "assistant"``.
        tool_call_id: which assistant tool call this message answers.
            Required when ``role == "tool"``.
        name:         optional sender name. Useful in multi-agent runs
            where multiple distinct assistants share the same role tag.
    """

    role: str
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError("tool messages must reference a tool_call_id")
        if self.tool_calls and self.role != "assistant":
            raise ValueError(
                f"only assistant messages may carry tool_calls; got role={self.role!r}",
            )
