"""Storage backends for the agent memory tool.

The :class:`MemoryStore` ABC defines the interface every backend must
satisfy. Two concrete backends ship:

* :class:`InMemoryMemoryStore` — dict-backed, ephemeral. Useful for
  tests + short-lived sessions.
* :class:`FileMemoryStore` — flat directory: each ``key`` becomes one
  text file ``<key>.md``. The store keeps a sidecar JSON file per key
  (``.meta``) with a creation timestamp so :meth:`list` can return
  entries sorted by recency without reading the bodies.

Keys are restricted to ``[A-Za-z0-9._-]+`` (no slashes, no dots-only)
so a key can never escape the store's directory. Attempting to use a
key with disallowed characters raises :class:`ValueError`.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

KEY_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_key(key: str) -> None:
    if not key or not KEY_PATTERN.match(key):
        raise ValueError(
            f"memory key must match {KEY_PATTERN.pattern!r}; got {key!r}",
        )
    if key in {".", ".."}:
        raise ValueError(f"memory key must not be {key!r}")


@dataclass(frozen=True)
class MemoryEntry:
    """One memory record.

    Attributes:
        key:        unique identifier (alphanumeric / dot / underscore / hyphen).
        value:      free-form text body the agent stored.
        created_at: unix timestamp (float seconds) of last write.
    """

    key: str
    value: str
    created_at: float


class MemoryStore(ABC):
    """Abstract memory backend."""

    @abstractmethod
    def store(self, key: str, value: str) -> MemoryEntry:
        """Insert or overwrite ``key`` with ``value``; return the entry."""

    @abstractmethod
    def view(self, key: str) -> MemoryEntry | None:
        """Return one entry by key, or ``None`` if missing."""

    @abstractmethod
    def list(self) -> tuple[MemoryEntry, ...]:
        """Return every entry, sorted newest-first by ``created_at``."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove ``key``; return True if it existed, False otherwise."""

    @abstractmethod
    def clear(self) -> None:
        """Remove every entry. Used by tests + agent reset."""

    def recall(self, query: str, *, limit: int = 5) -> tuple[MemoryEntry, ...]:
        """Naive substring search: every entry whose value contains ``query``.

        Backends can override with embedding search; the default
        implementation walks ``list()``. Returns at most ``limit``
        matches, newest first.
        """
        if not query:
            return ()
        q = query.lower()
        matches = [
            e for e in self.list()
            if q in e.key.lower() or q in e.value.lower()
        ]
        return tuple(matches[:limit])


class InMemoryMemoryStore(MemoryStore):
    """Dict-backed memory. Ephemeral; cleared at process exit."""

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}

    def store(self, key: str, value: str) -> MemoryEntry:
        _validate_key(key)
        entry = MemoryEntry(key=key, value=value, created_at=time.time())
        self._entries[key] = entry
        return entry

    def view(self, key: str) -> MemoryEntry | None:
        _validate_key(key)
        return self._entries.get(key)

    def list(self) -> tuple[MemoryEntry, ...]:
        return tuple(
            sorted(self._entries.values(), key=lambda e: e.created_at, reverse=True),
        )

    def delete(self, key: str) -> bool:
        _validate_key(key)
        return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        self._entries.clear()


class FileMemoryStore(MemoryStore):
    """Filesystem-backed memory. One ``.md`` body file + ``.meta`` sidecar per key."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def store(self, key: str, value: str) -> MemoryEntry:
        _validate_key(key)
        body = self._root / f"{key}.md"
        meta = self._root / f"{key}.meta"
        body.write_text(value, encoding="utf-8")
        ts = time.time()
        meta.write_text(json.dumps({"created_at": ts}), encoding="utf-8")
        return MemoryEntry(key=key, value=value, created_at=ts)

    def view(self, key: str) -> MemoryEntry | None:
        _validate_key(key)
        body = self._root / f"{key}.md"
        meta = self._root / f"{key}.meta"
        if not body.is_file():
            return None
        value = body.read_text(encoding="utf-8")
        ts = self._read_ts(meta)
        return MemoryEntry(key=key, value=value, created_at=ts)

    def list(self) -> tuple[MemoryEntry, ...]:
        entries: list[MemoryEntry] = []
        for body in self._root.glob("*.md"):
            key = body.stem
            if not KEY_PATTERN.match(key):
                continue
            entry = self.view(key)
            if entry is not None:
                entries.append(entry)
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return tuple(entries)

    def delete(self, key: str) -> bool:
        _validate_key(key)
        body = self._root / f"{key}.md"
        meta = self._root / f"{key}.meta"
        existed = body.is_file()
        body.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)
        return existed

    def clear(self) -> None:
        for child in self._root.iterdir():
            if child.is_file() and child.suffix in (".md", ".meta"):
                child.unlink(missing_ok=True)

    @staticmethod
    def _read_ts(meta: Path) -> float:
        if not meta.is_file():
            return 0.0
        try:
            return float(json.loads(meta.read_text(encoding="utf-8"))["created_at"])
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            return 0.0


__all__: Iterable[str] = (
    "FileMemoryStore",
    "InMemoryMemoryStore",
    "MemoryEntry",
    "MemoryStore",
)
