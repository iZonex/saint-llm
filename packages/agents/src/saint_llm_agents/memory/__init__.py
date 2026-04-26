"""Agent memory tool — persistent key/value notes the model can read & write.

Mirrors Anthropic's memory-tool pattern: the agent calls a small set of
tools (``memory_view`` / ``memory_store`` / ``memory_recall`` /
``memory_delete``) backed by a :class:`MemoryStore`. Two backends
ship out of the box:

* :class:`InMemoryMemoryStore` — dict-backed; lifetime = process.
* :class:`FileMemoryStore` — flat directory of files, one per key.
  Survives restarts and is human-inspectable on disk.

:func:`memory_tools` returns a tuple of :class:`FunctionTool` objects
ready to register in a :class:`ToolRegistry`.
"""

from saint_llm_agents.memory.store import (
    FileMemoryStore,
    InMemoryMemoryStore,
    MemoryEntry,
    MemoryStore,
)
from saint_llm_agents.memory.tools import memory_tools, register_memory_tools

__all__ = [
    "FileMemoryStore",
    "InMemoryMemoryStore",
    "MemoryEntry",
    "MemoryStore",
    "memory_tools",
    "register_memory_tools",
]
