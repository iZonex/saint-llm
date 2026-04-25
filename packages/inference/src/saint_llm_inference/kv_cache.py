"""Heterogeneous KV cache primitives.

Each attention variant in saint-llm has a different cache shape:

* ``SWAKVCacheLayer`` — sliding-window MQA: cache (B, max_seq_len, head_dim)
  per layer for K and V (shared across heads). Window mask is re-derived from
  the full cached length at each step.
* ``HCAKVCacheLayer`` — heavy-compressed: cache (B, n_blocks, head_dim) of
  *compressed* K/V. New tokens flow through the block-level compressor only
  on block boundaries (every ``compression_rate`` steps); in between the
  cached compression is reused. *(stage 2 — not implemented yet)*
* ``CSAKVCacheLayer`` — compressed-sparse: same compressed-block cache as
  HCA plus the indexer's K^IComp cache + a sliding-window uncompressed cache
  for the SWA branch. *(stage 3 — not implemented yet)*

This module ships stage 1: SWA only. Stages 2/3 will land here next; the
``KVCache`` class composes per-layer entries by attention type.
"""

from __future__ import annotations

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
