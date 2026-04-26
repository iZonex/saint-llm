"""Pipeline + Stage abstractions for D0.0.2 text data pipeline.

A ``Stage`` is anything callable with signature ``(doc) -> doc | None``:
the document either passes through (possibly modified) or is dropped
(``None``). Stages compose into a ``Pipeline`` that streams documents
through them in order.

Document shape (the ``doc`` dict):

* ``"text": str`` — the document text. Required.
* ``"language": str`` — ISO 639-3 code. Required for per-language
  filters (quality classifier, MT-pollution detector).
* ``"slice": str`` — name of the source slice (e.g. ``"hplt-en"``,
  ``"kobza-uk"``). Used by stages that apply only to a subset.
* ``"meta": dict[str, Any]`` — free-form extra metadata. Stages may
  read or augment.

Stages should not mutate inputs. Return a new dict or the original
if no changes (Python's mutability behavior is acceptable for
performance — but documented stages return-or-drop only).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

# Document is a plain dict for flexibility; stages can attach extra fields.
Document = dict[str, Any]


class Stage(Protocol):
    """One pipeline stage: takes a doc, returns a doc or drops it."""

    name: str

    def __call__(self, doc: Document) -> Document | None: ...


@dataclass
class Pipeline:
    """Compose stages into a streaming pipeline.

    ``run(source)`` yields documents that survived every stage in order.
    """

    stages: list[Stage]

    def run(self, source: Iterable[Document]) -> Iterator[Document]:
        """Stream documents from ``source`` through every stage."""
        for doc in source:
            current: Document | None = doc
            for stage in self.stages:
                current = stage(current)
                if current is None:
                    break
            if current is not None:
                yield current

    def stats_run(self, source: Iterable[Document]) -> tuple[list[Document], dict[str, int]]:
        """Run and report per-stage drop counts.

        Slow path — buffers full output. Useful for ablation reporting on
        small samples; not for the production pipeline.

        Returns ``(passing_documents, drops_by_stage)``. The drops dict
        maps each stage's ``name`` to the count of documents it dropped.
        """
        drops: dict[str, int] = {stage.name: 0 for stage in self.stages}
        passing: list[Document] = []
        for doc in source:
            current: Document | None = doc
            for stage in self.stages:
                next_doc = stage(current)
                if next_doc is None:
                    drops[stage.name] += 1
                    current = None
                    break
                current = next_doc
            if current is not None:
                passing.append(current)
        return passing, drops
