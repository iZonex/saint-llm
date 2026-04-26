"""Engram conditional memory (MEM-10 / arXiv 2601.07372).

DeepSeek + Peking U, January 2026: a "third sparsity axis" alongside
MoE. Stores static knowledge as N-gram-keyed embeddings in system RAM
(decoupled from HBM), with O(1) lookup at every layer. Reported gains
at 27B: MMLU +3.4, BBH +5.0, ARC-Challenge +3.7, HumanEval +3.0.

The architectural integration is **additive**: a parallel pathway
that fetches a memory embedding from N-gram context, projects it into
the model's hidden space, and adds it to the residual stream. Empty
N-gram cells return zero (model behavior unchanged).

This module ships:

* :class:`EngramConfig` — pydantic config.
* :class:`EngramTable` — the N-gram-keyed embedding table. Lives on
  CPU/system RAM by design (HBM would defeat the point).
* :class:`EngramHook` — ``nn.Module`` that takes recent token IDs +
  hidden state, looks up the matching N-gram embedding, projects to
  hidden space, and returns the additive contribution.

Production deployment swaps the in-memory ``EngramTable`` for a
sharded on-disk store (LMDB / RocksDB) when the table grows past
RAM. For v0.0 framework-completion the in-memory shape is enough.
"""

from __future__ import annotations

import hashlib

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn


class EngramConfig(BaseModel):
    """Engram conditional memory config."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    n_gram: int = 4
    n_buckets: int = 65_536
    embedding_dim: int = 128
    enabled: bool = False  # default off; v0.3 production flips on


def _ngram_hash(ids: list[int], n_buckets: int) -> int:
    """Stable bucket index from a tuple of recent token IDs."""
    blob = ",".join(str(int(i)) for i in ids).encode()
    h = hashlib.sha1(blob).digest()[:8]
    return int.from_bytes(h, "big") % n_buckets


class EngramTable(nn.Module):
    """N-gram-keyed embedding table living on the host (system RAM).

    The table holds ``n_buckets`` rows of ``embedding_dim``. Buckets
    are indexed by SHA-1 hash of the trailing N-gram of token IDs.
    Collisions are silently merged; this is acceptable per the Engram
    paper because the decoder of these embeddings is learned to be
    robust to small noise.

    The table is registered as an ``nn.Module`` so its weights are
    saved with the model checkpoint; ``param.device`` is CPU by
    construction (no .to(cuda)).
    """

    def __init__(self, cfg: EngramConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Buffer (not Parameter) — Engram embeddings are typically
        # learned via a separate optimizer step on CPU. v0.0 uses
        # zero-init so an unenabled Engram is identity-behavior.
        self.register_buffer(
            "embeddings",
            torch.zeros(cfg.n_buckets, cfg.embedding_dim),
            persistent=True,
        )

    def lookup(self, recent_token_ids: list[int]) -> Tensor:
        """Return the embedding for the trailing N-gram (or zeros)."""
        if len(recent_token_ids) < 1:
            return torch.zeros(self.cfg.embedding_dim, device=self.embeddings.device)
        ngram = recent_token_ids[-self.cfg.n_gram:]
        bucket = _ngram_hash(ngram, self.cfg.n_buckets)
        return self.embeddings[bucket]

    def update(self, recent_token_ids: list[int], delta: Tensor) -> None:
        """Add ``delta`` to the bucket pointed at by the N-gram.

        Used during the offline knowledge-injection phase that the
        Engram paper describes (separate from the main gradient
        pass). Out of the autograd graph by design — ``delta`` is
        applied in-place.
        """
        if len(recent_token_ids) < 1:
            return
        ngram = recent_token_ids[-self.cfg.n_gram:]
        bucket = _ngram_hash(ngram, self.cfg.n_buckets)
        with torch.no_grad():
            self.embeddings[bucket] += delta.to(self.embeddings.device)


class EngramHook(nn.Module):
    """Per-layer conditional memory hook.

    Inputs (per :meth:`forward`):
        hidden:           ``(B, T, D)`` hidden state at this layer.
        recent_token_ids: ``(B, T)`` long — the model's input tokens.

    Output: same shape as ``hidden``. The hook produces a residual
    contribution at every position by:

    1. Looking up the N-gram-keyed embedding from :class:`EngramTable`.
    2. Projecting it into ``hidden_dim`` via a small Linear.
    3. Adding to ``hidden`` (zero-init projection so the hook is
       identity at v0.0; v0.3 RL trains the projection).

    The ``EngramTable`` is CPU-resident; this hook moves the per-
    position embeddings to the hidden's device once per forward.
    """

    def __init__(self, hidden_dim: int, table: EngramTable) -> None:
        super().__init__()
        self.table = table
        self.proj = nn.Linear(table.cfg.embedding_dim, hidden_dim, bias=False)
        # Zero-init so an enabled-but-untrained Engram is identity.
        nn.init.zeros_(self.proj.weight)

    def forward(self, hidden: Tensor, recent_token_ids: Tensor) -> Tensor:
        if not self.table.cfg.enabled:
            return hidden
        if hidden.dim() != 3:
            raise ValueError(f"hidden must be (B,T,D); got {tuple(hidden.shape)}")
        b, t, _d = hidden.shape
        # Per-position N-gram lookup. CPU-bound; for production this
        # gets vectorized via a hash kernel, v0.0 uses Python loop.
        with torch.no_grad():
            embeds = torch.zeros(
                b, t, self.table.cfg.embedding_dim, device="cpu",
            )
            ids_cpu = recent_token_ids.detach().cpu().tolist()
            for bi in range(b):
                row = ids_cpu[bi]
                for ti in range(t):
                    embeds[bi, ti] = self.table.lookup(row[: ti + 1])
        embeds = embeds.to(hidden.device).to(hidden.dtype)
        delta = self.proj(embeds)
        return hidden + delta
