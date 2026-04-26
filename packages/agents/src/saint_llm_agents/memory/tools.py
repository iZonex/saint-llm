"""FunctionTool wrappers exposing a :class:`MemoryStore` to an agent.

:func:`memory_tools` builds four tools — ``memory_view``,
``memory_store``, ``memory_recall``, ``memory_delete`` — sharing a
single store. Drop them in a :class:`ToolRegistry` and the agent can
read/write its own notes between turns.
"""

from __future__ import annotations

from typing import Any

from saint_llm_agents.memory.store import MemoryEntry, MemoryStore
from saint_llm_agents.tool import FunctionTool, Tool, ToolRegistry, ToolSpec


def memory_tools(
    store: MemoryStore,
    *,
    name_prefix: str = "memory_",
) -> tuple[Tool, ...]:
    """Return four tools wrapping ``store`` for agent use.

    Args:
        store:       the backing :class:`MemoryStore`.
        name_prefix: prefix for the generated tool names. Default
            ``"memory_"`` yields ``memory_view`` etc.
    """
    return (
        _build_view_tool(store, name_prefix),
        _build_store_tool(store, name_prefix),
        _build_recall_tool(store, name_prefix),
        _build_delete_tool(store, name_prefix),
    )


def register_memory_tools(
    registry: ToolRegistry,
    store: MemoryStore,
    *,
    name_prefix: str = "memory_",
) -> None:
    """Convenience: build the four tools and register them in one shot."""
    for tool in memory_tools(store, name_prefix=name_prefix):
        registry.register(tool)


def _build_view_tool(store: MemoryStore, prefix: str) -> FunctionTool:
    spec = ToolSpec(
        name=f"{prefix}view",
        description=(
            "Read a memory entry by key. Returns the stored value, or "
            "the literal text 'no entry' if the key is unknown."
        ),
        parameters={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    )

    def _call(*, key: str, **_: Any) -> str:
        entry = store.view(key)
        return entry.value if entry is not None else "no entry"

    return FunctionTool(spec=spec, fn=_call)


def _build_store_tool(store: MemoryStore, prefix: str) -> FunctionTool:
    spec = ToolSpec(
        name=f"{prefix}store",
        description=(
            "Insert or overwrite a memory entry. Keys must be "
            "alphanumeric (with '.', '-', '_'). Returns 'ok'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
        },
    )

    def _call(*, key: str, value: str, **_: Any) -> str:
        store.store(key, value)
        return "ok"

    return FunctionTool(spec=spec, fn=_call)


def _build_recall_tool(store: MemoryStore, prefix: str) -> FunctionTool:
    spec = ToolSpec(
        name=f"{prefix}recall",
        description=(
            "Search memory entries by substring. Returns up to 'limit' "
            "matches as 'key: value' lines, newest first; empty string "
            "if no match."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    )

    def _call(*, query: str, limit: int = 5, **_: Any) -> str:
        matches = store.recall(query, limit=int(limit))
        if not matches:
            return ""
        return "\n".join(_format_entry(e) for e in matches)

    return FunctionTool(spec=spec, fn=_call)


def _build_delete_tool(store: MemoryStore, prefix: str) -> FunctionTool:
    spec = ToolSpec(
        name=f"{prefix}delete",
        description=(
            "Remove a memory entry by key. Returns 'deleted' if the key "
            "existed, 'no entry' otherwise."
        ),
        parameters={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    )

    def _call(*, key: str, **_: Any) -> str:
        return "deleted" if store.delete(key) else "no entry"

    return FunctionTool(spec=spec, fn=_call)


def _format_entry(entry: MemoryEntry) -> str:
    # Single-line render so the model can scan multiple matches in one read.
    one_line = entry.value.replace("\n", " ").strip()
    return f"{entry.key}: {one_line}"
