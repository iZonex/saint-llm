"""Tokenize stage — encode ``doc["text"]`` via the v0.0 tokenizer.

The result is attached as ``doc["token_ids"]`` (a list[int]) so the
downstream pack step (``saint_llm_data.packing``) can consume it
without re-tokenizing. Documents without text or producing zero tokens
are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

from saint_llm_data.pipeline.stage import Document
from saint_llm_data.tokenizer import Tokenizer


@dataclass
class TokenizeStage:
    """Apply ``tokenizer`` to ``doc["text"]`` and store ``token_ids``."""

    tokenizer: Tokenizer
    name: str = "tokenize"

    def __call__(self, doc: Document) -> Document | None:
        text = doc.get("text", "")
        if not text:
            return None
        token_ids = self.tokenizer.encode(text)
        if not token_ids:
            return None
        # Mutate to avoid copying potentially large dicts; pipeline
        # contract is "stages may mutate or replace".
        doc["token_ids"] = token_ids
        return doc
