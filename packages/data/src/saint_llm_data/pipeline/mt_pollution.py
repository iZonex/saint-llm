"""LANG-05 3-stack MT-pollution detector for Ukrainian (and similar lang slices).

The published recipe (LANG-05) is a 3-stack:

1. **Translationese classifier** — small fastText trained on
   UA-original vs UA-translated paired data. Drops "this looks like
   it was translated by GPT" articles.
2. **KenLM perplexity threshold** — UA-native KenLM rejects text
   whose token distribution is unusual.
3. **URL heuristics** — drop docs from known MT-source domains
   (e.g. content farms that re-publish auto-translated articles).

Real fastText + KenLM models are large and trained externally; this
module exposes Protocols for them and ships URL-heuristic
implementation immediately. The stub implementations let pipeline
composition + integration tests run without HF model downloads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse

from saint_llm_data.pipeline.stage import Document


class TranslationeseDetector(Protocol):
    """Score how "translated" a document looks. Higher = more translation-like."""

    def __call__(self, text: str) -> float: ...


class PerplexityModel(Protocol):
    """KenLM-style perplexity computation. Higher = more unusual text."""

    def perplexity(self, text: str) -> float: ...


# Common content-farm / known-MT-source domain patterns. Conservative;
# extend per-language. Production lists live in a versioned text file
# loaded from disk, not here — these are the v0.0 baseline.
DEFAULT_BLOCKED_DOMAINS: tuple[str, ...] = (
    # Add real entries during D0.0.2 production rollout. The patterns
    # below match domains that historically host machine-translated
    # content; treat as starter heuristic only.
    "translated-articles.example",
    "auto-translate.example",
)


def _domain_of(url: str) -> str:
    """Lower-case host of a URL; '' if unparseable."""
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


@dataclass
class URLBlocklistFilter:
    """Drop documents whose ``meta.url`` matches a blocked-domain pattern.

    ``apply_to_languages`` restricts the filter (LANG-05 applies UA-only
    by default).
    """

    blocked_domains: tuple[str, ...] = DEFAULT_BLOCKED_DOMAINS
    apply_to_languages: tuple[str, ...] = ("uk",)
    name: str = "url_blocklist"

    _patterns: tuple[re.Pattern[str], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Build a case-insensitive regex per blocked domain for substring
        # match against the host. Patterns are simple — no path-level
        # filtering at v0.0.
        object.__setattr__(
            self,
            "_patterns",
            tuple(re.compile(re.escape(d), re.IGNORECASE) for d in self.blocked_domains),
        )

    def __call__(self, doc: Document) -> Document | None:
        if self.apply_to_languages and doc.get("language") not in self.apply_to_languages:
            return doc
        meta = doc.get("meta", {})
        url = meta.get("url", "") if isinstance(meta, dict) else ""
        if not url:
            return doc
        host = _domain_of(url)
        if not host:
            return doc
        for pat in self._patterns:
            if pat.search(host):
                return None
        return doc


@dataclass
class TranslationeseFilter:
    """Drop documents whose ``detector(text) >= threshold``.

    Threshold convention: detector returns higher score for more
    translation-like text; documents at or above threshold are dropped.
    """

    detector: TranslationeseDetector
    threshold: float
    apply_to_languages: tuple[str, ...] = ("uk",)
    name: str = "translationese_filter"

    def __call__(self, doc: Document) -> Document | None:
        if self.apply_to_languages and doc.get("language") not in self.apply_to_languages:
            return doc
        text = doc.get("text", "")
        if not text:
            return doc
        score = self.detector(text)
        if score >= self.threshold:
            return None
        return doc


@dataclass
class PerplexityFilter:
    """Drop documents whose KenLM perplexity exceeds ``max_perplexity``."""

    model: PerplexityModel
    max_perplexity: float
    apply_to_languages: tuple[str, ...] = ("uk",)
    name: str = "perplexity_filter"

    def __call__(self, doc: Document) -> Document | None:
        if self.apply_to_languages and doc.get("language") not in self.apply_to_languages:
            return doc
        text = doc.get("text", "")
        if not text:
            return doc
        ppl = self.model.perplexity(text)
        if ppl > self.max_perplexity:
            return None
        return doc
