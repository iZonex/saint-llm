"""Tool registry for agent runtime.

A :class:`Tool` is anything callable with a :class:`ToolSpec` describing
its name, description, and JSON Schema for its parameters. The
:class:`ToolRegistry` keeps a name→tool map and provides a
:meth:`ToolRegistry.execute` that runs a :class:`ToolCall` end-to-end:
look up the tool, invoke with the call's arguments, package the return
value (or any exception text) into a ``role="tool"`` :class:`Message`.

Validation strategy: the registry does *not* validate arguments against
the tool's JSON Schema by default. Tools that need strict validation
should do it themselves (e.g. via pydantic). The registry's job is just
to dispatch and capture errors; making validation pluggable is a v0.2
concern.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from saint_llm_agents.message import Message, ToolCall


@dataclass(frozen=True)
class ToolSpec:
    """Tool metadata + JSON Schema for parameters.

    Mirrors the OpenAI/Anthropic tool-call schema so the same ``ToolSpec``
    can be serialized for any external policy backend (OpenAI, Anthropic)
    or our own model's DSML XML emit.

    Attributes:
        name:        unique identifier — collision-checked at registration.
        description: human-readable, fed into the model prompt so it can
            decide *when* to call this tool.
        parameters:  JSON Schema dict (``{"type": "object", "properties":
            {...}, "required": [...]}``). Empty dict means "no
            parameters".
    """

    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


class Tool(Protocol):
    """Protocol every tool implements.

    Tools expose a ``spec`` attribute and are themselves callable with
    keyword arguments. The return value is stringified by the registry
    when packaged into a tool message.
    """

    spec: ToolSpec

    def __call__(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class FunctionTool:
    """Convenience wrapper: any Python callable plus a :class:`ToolSpec`.

    Example::

        spec = ToolSpec(name="add", description="add two ints",
                       parameters={"type": "object",
                                   "properties": {"a": {"type": "integer"},
                                                  "b": {"type": "integer"}},
                                   "required": ["a", "b"]})
        add = FunctionTool(spec=spec, fn=lambda a, b: a + b)
    """

    spec: ToolSpec
    fn: Callable[..., Any]

    def __call__(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


class ToolRegistry:
    """Name→tool map with a single-call execute helper."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if name in self._tools:
            raise ValueError(f"tool {name!r} already registered")
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"tool {name!r} not registered")
        return self._tools[name]

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(t.spec for t in self._tools.values())

    def execute(self, call: ToolCall) -> Message:
        """Invoke ``call.name`` with ``call.arguments``; package result.

        Returns a ``role="tool"`` :class:`Message` whose ``content`` is
        ``str(result)`` on success or ``"ERROR: <type>: <text>"`` on any
        exception. The runtime never re-raises tool errors — a misbehaving
        tool produces an error message the model can read and react to.
        """
        try:
            tool = self.get(call.name)
            result = tool(**dict(call.arguments))
            content = str(result) if result is not None else ""
        except Exception as exc:
            content = f"ERROR: {type(exc).__name__}: {exc}"
        return Message(role="tool", content=content, tool_call_id=call.id)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools
