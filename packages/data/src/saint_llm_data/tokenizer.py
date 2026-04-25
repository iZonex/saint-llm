"""Tokenizer surface — string ↔ token-ID conversion.

Two implementations:

* ``HFTokenizer`` — thin wrapper around ``tokenizers.Tokenizer``. Loads any
  saved tokenizer.json or HF Hub tokenizer (the underlying library uses HF
  Hub when ``from_pretrained`` is called).
* ``CharTokenizer`` — codepoint-as-id fallback for hermetic tests. No
  download, no vendor library, no unicode normalization. Vocab = ``base_vocab``
  control slots + raw codepoints up to ``unicode_max``.

Both expose the ``Tokenizer`` Protocol so downstream packing / training code
can accept either without branching.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from tokenizers import Tokenizer as _BackendTokenizer


class Tokenizer(Protocol):
    """Minimal contract every tokenizer implements."""

    @property
    def vocab_size(self) -> int: ...

    @property
    def eos_token_id(self) -> int: ...

    @property
    def pad_token_id(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]: ...

    def decode(self, ids: Iterable[int]) -> str: ...


class CharTokenizer:
    """Codepoint-as-id tokenizer. ``id = codepoint + base_vocab``.

    The first ``base_vocab`` ids are reserved control slots:
    * 0 = pad
    * 1 = eos
    * 2 = bos
    * 3..base_vocab-1 = unallocated (mirror AUGMENTATIONS slot pattern)

    Vocab size = ``base_vocab + unicode_max``. Default keeps it reasonable for
    tests (BMP only).
    """

    def __init__(
        self,
        *,
        base_vocab: int = 16,
        unicode_max: int = 0x110000,
    ) -> None:
        self._base_vocab = base_vocab
        self._unicode_max = unicode_max

    @property
    def vocab_size(self) -> int:
        return self._base_vocab + self._unicode_max

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def eos_token_id(self) -> int:
        return 1

    @property
    def bos_token_id(self) -> int:
        return 2

    def encode(self, text: str) -> list[int]:
        return [self._base_vocab + ord(ch) for ch in text]

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, ids: Iterable[int]) -> str:
        out: list[str] = []
        for i in ids:
            if i < self._base_vocab:
                continue  # skip control slots silently
            cp = i - self._base_vocab
            if 0 <= cp <= self._unicode_max:
                out.append(chr(cp))
        return "".join(out)


class HFTokenizer:
    """Thin wrapper around ``tokenizers.Tokenizer`` (the HF tokenizers library).

    Special-token IDs are looked up by name with sensible defaults; pass
    ``eos_token`` / ``pad_token`` to override.
    """

    def __init__(
        self,
        backend: object,
        *,
        eos_token: str = "<|endoftext|>",
        pad_token: str | None = None,
    ) -> None:
        self._backend = backend
        self._eos_id = self._lookup_id(eos_token, fallback=None)
        if self._eos_id is None:
            raise ValueError(
                f"eos token {eos_token!r} not present in this tokenizer's vocab — "
                "pass eos_token=... that the tokenizer knows.",
            )
        self._pad_id = self._lookup_id(pad_token, fallback=self._eos_id) if pad_token else self._eos_id

    def _lookup_id(self, token: str | None, *, fallback: int | None) -> int | None:
        if token is None:
            return fallback
        # tokenizers.Tokenizer.token_to_id returns int | None.
        tid = self._backend.token_to_id(token)  # type: ignore[attr-defined]
        return int(tid) if tid is not None else fallback

    @property
    def vocab_size(self) -> int:
        return int(self._backend.get_vocab_size())  # type: ignore[attr-defined]

    @property
    def eos_token_id(self) -> int:
        return self._eos_id

    @property
    def pad_token_id(self) -> int:
        return self._pad_id

    def encode(self, text: str) -> list[int]:
        return list(self._backend.encode(text).ids)  # type: ignore[attr-defined]

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        encs = self._backend.encode_batch(list(texts))  # type: ignore[attr-defined]
        return [list(e.ids) for e in encs]

    def decode(self, ids: Iterable[int]) -> str:
        return str(self._backend.decode(list(ids)))  # type: ignore[attr-defined]

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: object) -> HFTokenizer:
        backend = _BackendTokenizer.from_file(str(path))
        return cls(backend, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_pretrained(cls, identifier: str, **kwargs: object) -> HFTokenizer:
        """Load via the HF Hub. Requires network access on first call."""
        backend = _BackendTokenizer.from_pretrained(identifier)
        return cls(backend, **kwargs)  # type: ignore[arg-type]
