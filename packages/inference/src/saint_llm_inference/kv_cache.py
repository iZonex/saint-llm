"""Heterogeneous KV cache primitives.

Each attention variant in saint-llm has a different cache shape:

* ``SWAKVCacheLayer`` — sliding-window MQA: cache (B, max_seq_len, head_dim)
  per layer for K and V (shared across heads). Window mask is re-derived from
  the full cached length at each step.
* ``HCAKVCacheLayer`` — heavy-compressed: a (B, max_blocks, head_dim)
  compressed-block cache (output of the model's ``TokenLevelCompressor``),
  plus an h-buffer accumulating the in-progress block, plus a sliding-window
  branch K/V cache (shape as ``SWAKVCacheLayer``). New compressed blocks are
  emitted on every ``compression_rate``-token boundary by replaying the
  model's compressor on the buffered raw hidden states.
* ``CSAKVCacheLayer`` — compressed-sparse: same compressed-block cache as
  HCA plus the indexer's K^IComp cache + a sliding-window uncompressed cache
  for the SWA branch. *(stage 6 — not implemented yet)*

``KVCacheBundle`` composes per-layer entries by attention type. The model's
forward looks up the bundle entry for each block and passes it to that
block's attention forward; blocks whose attention type doesn't have a cache
implementation yet silently no-op (the bundle entry is None).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


class SWAKVCacheLayer:
    """Append-only KV cache for ``SWAttention``.

    Shared across query heads (MQA): one (B, max_seq_len, head_dim) buffer per
    K and V. ``append`` writes new entries at the running cursor and returns
    a slice of the cache up to the current length so the attention can
    consume it directly.

    The cache stores the *post-RoPE* keys: each new key arrives with RoPE
    already applied for its absolute position, and cached keys retain the
    RoPE that encoded their original positions. No re-rotation needed.
    """

    def __init__(
        self,
        max_seq_len: int,
        head_dim: int,
        *,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive; got {max_seq_len}")
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.k = torch.zeros(batch_size, max_seq_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(batch_size, max_seq_len, head_dim, device=device, dtype=dtype)
        self.length = 0

    def append(self, k_new: Tensor, v_new: Tensor) -> tuple[Tensor, Tensor]:
        """Append new K/V entries; return the full cache slice up to current length.

        ``k_new`` / ``v_new`` shape (B, T_new, head_dim). Raises if the append
        would overflow ``max_seq_len``.
        """
        if k_new.dim() != 3 or v_new.dim() != 3:
            raise ValueError(
                f"k_new/v_new must be 3D (B, T, head_dim); got {tuple(k_new.shape)}, "
                f"{tuple(v_new.shape)}",
            )
        n = k_new.shape[1]
        if self.length + n > self.max_seq_len:
            raise RuntimeError(
                f"SWAKVCacheLayer overflow: tried to append {n} tokens at "
                f"length {self.length} into max_seq_len={self.max_seq_len}",
            )
        self.k[:, self.length:self.length + n] = k_new
        self.v[:, self.length:self.length + n] = v_new
        self.length += n
        return self.k[:, :self.length], self.v[:, :self.length]

    def reset(self) -> None:
        """Drop all cached entries; ready for a fresh prompt."""
        self.length = 0


class HCAKVCacheLayer:
    """Append-only KV cache for ``HCA``.

    Three buffers per layer:

    * ``compressed`` (B, max_blocks, head_dim) — output of the model's
      ``TokenLevelCompressor`` for each already-emitted m-token block.
    * ``h_buffer`` (B, m, hidden_dim) — raw hidden states for the
      in-progress block, accumulating until ``compression_rate`` are
      present, then flushed through the compressor.
    * ``sw_k`` / ``sw_v`` (B, max_seq_len, head_dim) — uncompressed K/V
      for the sliding-window branch. Same contract as ``SWAKVCacheLayer``.

    The cache is module-decoupled: ``append`` accepts a ``compressor``
    callable supplied by the caller (typically ``hca.kv_compressor``).
    """

    def __init__(
        self,
        max_seq_len: int,
        hidden_dim: int,
        head_dim: int,
        compression_rate: int,
        *,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive; got {max_seq_len}")
        if compression_rate <= 0:
            raise ValueError(f"compression_rate must be positive; got {compression_rate}")
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.compression_rate = compression_rate
        self.batch_size = batch_size
        max_blocks = max_seq_len // compression_rate
        self.compressed = torch.zeros(batch_size, max_blocks, head_dim, device=device, dtype=dtype)
        self.h_buffer = torch.zeros(
            batch_size, compression_rate, hidden_dim, device=device, dtype=dtype,
        )
        self.h_buf_count = 0
        self.n_blocks = 0
        self.sw_k = torch.zeros(batch_size, max_seq_len, head_dim, device=device, dtype=dtype)
        self.sw_v = torch.zeros(batch_size, max_seq_len, head_dim, device=device, dtype=dtype)
        self.length = 0  # KVCacheBundle interface — total decoded positions.

    def append(
        self,
        h_new: Tensor,
        sw_k_new: Tensor,
        sw_v_new: Tensor,
        *,
        compressor: Callable[[Tensor], Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Append T_new tokens; return (compressed_view, sw_k_view, sw_v_view).

        Bulk-emits all complete blocks in this call by combining the
        h-buffer with ``h_new`` and feeding the contiguous m-aligned prefix
        to ``compressor`` in one shot. Leftover tokens stay in the buffer.
        Raises on overflow of either the SW cache or the compressed-block
        cache.
        """
        if h_new.dim() != 3 or sw_k_new.dim() != 3 or sw_v_new.dim() != 3:
            raise ValueError(
                "h_new, sw_k_new, sw_v_new must all be 3D (B, T, *); got "
                f"{tuple(h_new.shape)}, {tuple(sw_k_new.shape)}, {tuple(sw_v_new.shape)}",
            )
        n = sw_k_new.shape[1]
        if self.length + n > self.max_seq_len:
            raise RuntimeError(
                f"HCAKVCacheLayer overflow: tried to append {n} tokens at "
                f"length {self.length} into max_seq_len={self.max_seq_len}",
            )

        # 1) Append SW branch K/V.
        self.sw_k[:, self.length:self.length + n] = sw_k_new
        self.sw_v[:, self.length:self.length + n] = sw_v_new

        # 2) Combine buffered h with the new chunk.
        if self.h_buf_count > 0:
            combined = torch.cat(
                [self.h_buffer[:, :self.h_buf_count], h_new], dim=1,
            )
        else:
            combined = h_new
        total = self.h_buf_count + n
        n_complete = total // self.compression_rate
        n_consumed = n_complete * self.compression_rate

        # 3) Bulk-emit any newly-complete blocks.
        if n_complete > 0:
            new_blocks = compressor(combined[:, :n_consumed])  # (B, n_complete, head_dim)
            if new_blocks.shape[1] != n_complete:
                raise RuntimeError(
                    f"compressor returned {new_blocks.shape[1]} blocks, expected {n_complete}",
                )
            self.compressed[:, self.n_blocks:self.n_blocks + n_complete] = new_blocks
            self.n_blocks += n_complete

        # 4) Stash the leftover tail (< compression_rate tokens).
        leftover = total - n_consumed
        if leftover > 0:
            self.h_buffer[:, :leftover] = combined[:, n_consumed:]
        self.h_buf_count = leftover

        self.length += n
        return (
            self.compressed[:, :self.n_blocks],
            self.sw_k[:, :self.length],
            self.sw_v[:, :self.length],
        )

    def reset(self) -> None:
        self.length = 0
        self.n_blocks = 0
        self.h_buf_count = 0


_CacheLayer = SWAKVCacheLayer | HCAKVCacheLayer | None


class KVCacheBundle:
    """Per-layer KV cache container indexed by transformer block index.

    Entries are typed ``SWAKVCacheLayer | HCAKVCacheLayer | None``. ``None``
    means "this block has no cache; fall back to recompute". The ``length``
    property returns the global decoded length from any populated entry —
    all entries advance in lockstep.
    """

    def __init__(self, layers: list[_CacheLayer]) -> None:
        self._layers = list(layers)

    def __len__(self) -> int:
        return len(self._layers)

    def for_layer(self, idx: int) -> _CacheLayer:
        """Return the cache for block ``idx`` or ``None`` if uncached."""
        return self._layers[idx]

    @property
    def length(self) -> int:
        """Number of decoded positions cached in any populated entry.

        All populated entries advance in lockstep, so the first non-None
        entry's length is authoritative. Returns 0 when nothing is cached.
        """
        for layer in self._layers:
            if layer is not None:
                return layer.length
        return 0

    def reset(self) -> None:
        for layer in self._layers:
            if layer is not None:
                layer.reset()

    @classmethod
    def for_model(
        cls,
        model: object,
        *,
        max_seq_len: int,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> KVCacheBundle:
        """Build a bundle by walking ``model.blocks``.

        SWAttention → ``SWAKVCacheLayer``.
        HCA → ``HCAKVCacheLayer`` (uses ``cfg.hca.compression_rate`` and
        ``cfg.hidden_dim`` for the h-buffer).
        CSA → ``None`` (stage 6 — pending).
        """
        # Local imports to keep the runtime graph free of saint_llm_core when
        # this helper isn't used. Avoids a hard dep cycle in tests.
        from saint_llm_core.attention import HCA, SWAttention  # noqa: PLC0415

        cfg = model.cfg  # type: ignore[attr-defined]
        head_dim = cfg.attention.head_dim
        hidden_dim = cfg.hidden_dim
        layers: list[_CacheLayer] = []
        for block in model.blocks:  # type: ignore[attr-defined]
            if isinstance(block.attention, SWAttention):
                layers.append(SWAKVCacheLayer(
                    max_seq_len=max_seq_len,
                    head_dim=head_dim,
                    batch_size=batch_size,
                    device=device,
                    dtype=dtype,
                ))
            elif isinstance(block.attention, HCA):
                layers.append(HCAKVCacheLayer(
                    max_seq_len=max_seq_len,
                    hidden_dim=hidden_dim,
                    head_dim=head_dim,
                    compression_rate=cfg.hca.compression_rate,
                    batch_size=batch_size,
                    device=device,
                    dtype=dtype,
                ))
            else:
                layers.append(None)
        return cls(layers)
