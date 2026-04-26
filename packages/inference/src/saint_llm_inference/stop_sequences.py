"""Stop-sequence matching for streaming decoders.

Streaming decoders normally stop on EOS or after ``max_new_tokens``.
For chat-style generation we typically also want to stop when the
model emits a specific *token sequence* — e.g. ``<|/assistant|>``,
``"User:"``, or any caller-defined sentinel. This module provides:

* :class:`StopSequenceMatcher` — keeps a rolling tail of generated
  tokens and answers ``matched()`` if any of N target sequences
  appears at the tail.
* :func:`encode_stop_strings` — tokenize a list of stop strings via
  any :class:`Tokenizer`, returning the token-ID sequences ready to
  hand to the matcher.

The matcher is per-call (callers create a fresh one for each
generation) and per-batch (each row gets its own matcher; the streaming
decoders short-circuit when *every* row matches).
"""

from __future__ import annotations

from collections.abc import Sequence

from saint_llm_data.tokenizer import Tokenizer


class StopSequenceMatcher:
    """Rolling tail-match against N alternative stop sequences.

    Args:
        sequences: list of token-ID sequences. Empty sequences are
            ignored. The matcher returns True when any sequence is a
            *suffix* of the tokens seen so far via :meth:`feed`.

    Example::

        m = StopSequenceMatcher([[10, 11], [20]])
        m.feed(7); m.matched()  # False
        m.feed(10); m.matched()  # False
        m.feed(11); m.matched()  # True (sequence [10, 11] matched)
    """

    def __init__(self, sequences: Sequence[Sequence[int]]) -> None:
        cleaned: list[tuple[int, ...]] = []
        for seq in sequences:
            tup = tuple(int(x) for x in seq)
            if tup:
                cleaned.append(tup)
        self._sequences = tuple(cleaned)
        self._max_len = max((len(s) for s in self._sequences), default=0)
        self._tail: list[int] = []
        self._matched = False
        self._matched_seq: tuple[int, ...] | None = None

    @property
    def sequences(self) -> tuple[tuple[int, ...], ...]:
        return self._sequences

    @property
    def matched_sequence(self) -> tuple[int, ...] | None:
        """The sequence that triggered the match, or ``None``."""
        return self._matched_seq

    def feed(self, token: int) -> bool:
        """Add a token to the rolling tail and return :meth:`matched`."""
        if self._max_len == 0:
            return False
        if self._matched:
            return True
        self._tail.append(int(token))
        # Trim to the longest sequence's length so the tail can never
        # grow without bound.
        if len(self._tail) > self._max_len:
            self._tail = self._tail[-self._max_len :]
        for seq in self._sequences:
            if len(self._tail) >= len(seq) and tuple(self._tail[-len(seq) :]) == seq:
                self._matched = True
                self._matched_seq = seq
                return True
        return False

    def matched(self) -> bool:
        return self._matched

    def reset(self) -> None:
        self._tail = []
        self._matched = False
        self._matched_seq = None


def encode_stop_strings(
    stops: Sequence[str], tokenizer: Tokenizer,
) -> list[list[int]]:
    """Encode caller-supplied stop strings into token-ID sequences.

    Empty strings are dropped silently. Empty post-encoding sequences
    are also dropped — some tokenizers emit nothing for whitespace-only
    input.
    """
    result: list[list[int]] = []
    for s in stops:
        if not s:
            continue
        ids = list(tokenizer.encode(s))
        if ids:
            result.append(ids)
    return result
