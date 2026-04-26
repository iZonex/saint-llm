"""Tests for the agent memory tool."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from saint_llm_agents import ToolCall, ToolRegistry
from saint_llm_agents.memory import (
    FileMemoryStore,
    InMemoryMemoryStore,
    MemoryStore,
    memory_tools,
    register_memory_tools,
)


@pytest.fixture(params=["memory", "file"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> MemoryStore:
    if request.param == "memory":
        return InMemoryMemoryStore()
    return FileMemoryStore(tmp_path / "mem")


def test_store_and_view_round_trip(store: MemoryStore) -> None:
    entry = store.store("note1", "hello world")
    assert entry.key == "note1"
    assert entry.value == "hello world"
    fetched = store.view("note1")
    assert fetched is not None
    assert fetched.value == "hello world"


def test_view_missing_returns_none(store: MemoryStore) -> None:
    assert store.view("does_not_exist") is None


def test_store_overwrites_existing(store: MemoryStore) -> None:
    store.store("k", "v1")
    store.store("k", "v2")
    fetched = store.view("k")
    assert fetched is not None
    assert fetched.value == "v2"


def test_list_returns_newest_first(store: MemoryStore) -> None:
    store.store("a", "first")
    time.sleep(0.005)  # ensure distinct timestamps even on coarse clocks
    store.store("b", "second")
    items = store.list()
    assert len(items) == 2
    assert items[0].key == "b"
    assert items[1].key == "a"


def test_delete_returns_existed_flag(store: MemoryStore) -> None:
    store.store("k", "v")
    assert store.delete("k") is True
    assert store.delete("k") is False


def test_clear_removes_all(store: MemoryStore) -> None:
    store.store("a", "v")
    store.store("b", "v")
    store.clear()
    assert store.list() == ()


def test_recall_substring_in_value(store: MemoryStore) -> None:
    store.store("a", "the quick brown fox")
    store.store("b", "lazy dog")
    matches = store.recall("brown")
    assert len(matches) == 1
    assert matches[0].key == "a"


def test_recall_substring_in_key(store: MemoryStore) -> None:
    store.store("snake-case-key", "x")
    matches = store.recall("snake")
    assert len(matches) == 1


def test_recall_case_insensitive(store: MemoryStore) -> None:
    store.store("a", "QUICK Brown FOX")
    matches = store.recall("brown")
    assert len(matches) == 1


def test_recall_respects_limit(store: MemoryStore) -> None:
    for i in range(5):
        store.store(f"key{i}", "matchme content")
    matches = store.recall("matchme", limit=2)
    assert len(matches) == 2


def test_recall_empty_query_returns_nothing(store: MemoryStore) -> None:
    store.store("a", "v")
    assert store.recall("") == ()


def test_invalid_keys_rejected(store: MemoryStore) -> None:
    for bad in ["", "../escape", "with/slash", "with space"]:
        with pytest.raises(ValueError, match="memory key"):
            store.store(bad, "v")


def test_file_store_persists_across_instances(tmp_path: Path) -> None:
    root = tmp_path / "persist"
    s1 = FileMemoryStore(root)
    s1.store("k", "v")
    s2 = FileMemoryStore(root)
    fetched = s2.view("k")
    assert fetched is not None
    assert fetched.value == "v"


def test_file_store_ignores_unrelated_files(tmp_path: Path) -> None:
    root = tmp_path / "mixed"
    s = FileMemoryStore(root)
    s.store("a", "v")
    (root / "stray.txt").write_text("noise")
    items = s.list()
    assert len(items) == 1


def test_memory_tools_count_and_names() -> None:
    s = InMemoryMemoryStore()
    tools = memory_tools(s)
    names = {t.spec.name for t in tools}
    assert names == {"memory_view", "memory_store", "memory_recall", "memory_delete"}


def test_memory_tool_view_returns_value() -> None:
    s = InMemoryMemoryStore()
    s.store("k", "v")
    reg = ToolRegistry()
    register_memory_tools(reg, s)
    msg = reg.execute(ToolCall(id="t", name="memory_view", arguments={"key": "k"}))
    assert msg.content == "v"


def test_memory_tool_view_missing_returns_no_entry() -> None:
    reg = ToolRegistry()
    register_memory_tools(reg, InMemoryMemoryStore())
    msg = reg.execute(ToolCall(id="t", name="memory_view", arguments={"key": "missing"}))
    assert msg.content == "no entry"


def test_memory_tool_store_then_view() -> None:
    s = InMemoryMemoryStore()
    reg = ToolRegistry()
    register_memory_tools(reg, s)
    reg.execute(ToolCall(id="t", name="memory_store", arguments={"key": "k", "value": "hello"}))
    msg = reg.execute(ToolCall(id="t", name="memory_view", arguments={"key": "k"}))
    assert msg.content == "hello"


def test_memory_tool_recall_returns_lines() -> None:
    s = InMemoryMemoryStore()
    reg = ToolRegistry()
    register_memory_tools(reg, s)
    s.store("a", "the quick brown fox")
    s.store("b", "lazy dog")
    msg = reg.execute(ToolCall(id="t", name="memory_recall", arguments={"query": "brown"}))
    assert "a: the quick brown fox" in msg.content


def test_memory_tool_recall_no_match_returns_empty() -> None:
    s = InMemoryMemoryStore()
    reg = ToolRegistry()
    register_memory_tools(reg, s)
    msg = reg.execute(ToolCall(id="t", name="memory_recall", arguments={"query": "x"}))
    assert msg.content == ""


def test_memory_tool_delete_existing_and_missing() -> None:
    s = InMemoryMemoryStore()
    reg = ToolRegistry()
    register_memory_tools(reg, s)
    s.store("k", "v")
    msg = reg.execute(ToolCall(id="t", name="memory_delete", arguments={"key": "k"}))
    assert msg.content == "deleted"
    msg2 = reg.execute(ToolCall(id="t", name="memory_delete", arguments={"key": "k"}))
    assert msg2.content == "no entry"


def test_memory_tools_custom_prefix() -> None:
    s = InMemoryMemoryStore()
    tools = memory_tools(s, name_prefix="notes_")
    names = {t.spec.name for t in tools}
    assert names == {"notes_view", "notes_store", "notes_recall", "notes_delete"}


def test_memory_tool_invalid_key_returned_as_error_string() -> None:
    """Tool errors are surfaced as 'ERROR:' strings via ToolRegistry."""
    s = InMemoryMemoryStore()
    reg = ToolRegistry()
    register_memory_tools(reg, s)
    msg = reg.execute(ToolCall(
        id="t", name="memory_store",
        arguments={"key": "../bad", "value": "v"},
    ))
    assert msg.content.startswith("ERROR:")
    assert "memory key" in msg.content
