"""Fused CSA Lightning Indexer kernel.

The Lightning Indexer scores compressed KV blocks per query token::

    head_scores[b,t,h,s] = ReLU(q_indexer[b,t,h,:] · k_compressed[b,s,:])
    scores[b,t,s]        = sum_h head_weights[b,t,h] * head_scores[b,t,h,s]
    + optional visual-block cohesion bias
    + causal mask (block s valid only after its last token is past q at t)
    top_idx[b,t,:]       = argtopk_s scores[b,t,s]

The intermediate ``head_scores`` tensor has shape ``(B, T, n_heads, n_blocks)``.
For T=128K, n_blocks=T/m=8K, n_heads=8 it weighs 32 GiB in fp32 — bigger than
hpomen VRAM. Fusing collapses the head axis on the fly so only the (B, T, n_blocks)
output is materialized.

Two entry points:
* ``lightning_indexer_scores`` — pure score computation, returns the (B, T, n_blocks)
  tensor before causal masking. Useful for auxiliary losses / soft attention.
* ``lightning_indexer_topk`` — full pipeline: scores → mask → top-k.
  This is the indexer's hot-path replacement.

Both have an eager reference and a CUDA-only ``torch.compile`` cached path.
"""

from __future__ import annotations

from functools import cache

import torch
from torch import Tensor


def lightning_indexer_scores_reference(
    q_indexer: Tensor,
    k_compressed: Tensor,
    head_weights: Tensor,
    visual_bias: Tensor | None = None,
) -> Tensor:
    """Eager reference. Returns (B, T, n_blocks) unmasked scores.

    Shapes::
        q_indexer    : (B, T, n_heads, c)
        k_compressed : (B, n_blocks, c)
        head_weights : (B, T, n_heads)
        visual_bias  : (B, n_blocks) or None — added with weight 0.5 (MM-V-06)
    """
    head_scores = torch.einsum("bthc,bnc->bthn", q_indexer, k_compressed).clamp(min=0.0)
    if visual_bias is not None:
        head_scores = head_scores + visual_bias.unsqueeze(1).unsqueeze(2) * 0.5
    return (head_weights.unsqueeze(-1) * head_scores).sum(dim=-2)


def _causal_block_mask(t: int, n_blocks: int, block_size_m: int, device: torch.device) -> Tensor:
    """``valid[q, s]`` = block ``s`` is fully written by position ``q``."""
    q_idx = torch.arange(t, device=device).unsqueeze(-1)
    block_end = torch.arange(n_blocks, device=device).unsqueeze(0) * block_size_m + (block_size_m - 1)
    return block_end < q_idx


def lightning_indexer_topk_reference(
    q_indexer: Tensor,
    k_compressed: Tensor,
    head_weights: Tensor,
    top_k: int,
    block_size_m: int,
    visual_bias: Tensor | None = None,
) -> Tensor:
    """Eager reference. Returns (B, T, k_eff) top-k block indices.

    ``k_eff = min(top_k, n_blocks)``. Picks where the masked score is ``-inf``
    (i.e. no valid block exists at that rank for this query position) are
    returned as ``-1`` so callers can drop them. Returns shape ``(B, T, 0)`` of
    dtype int64 when no compressed blocks exist.
    """
    b, t, _, _ = q_indexer.shape
    n_blocks = k_compressed.shape[1]
    if top_k == 0 or n_blocks == 0:
        return q_indexer.new_full((b, t, 0), -1, dtype=torch.long)

    scores = lightning_indexer_scores_reference(
        q_indexer, k_compressed, head_weights, visual_bias=visual_bias,
    )
    valid = _causal_block_mask(t, n_blocks, block_size_m, q_indexer.device)
    masked = scores.masked_fill(~valid, float("-inf"))
    k_eff = min(top_k, n_blocks)
    top_scores, top_idx = masked.topk(k_eff, dim=-1)
    valid_pick = torch.isfinite(top_scores)
    return torch.where(valid_pick, top_idx, top_idx.new_full((), -1))


@cache
def _get_compiled_scores() -> object:
    return torch.compile(lightning_indexer_scores_reference, dynamic=True)


@cache
def _get_compiled_topk() -> object:
    return torch.compile(lightning_indexer_topk_reference, dynamic=True)


def lightning_indexer_scores(
    q_indexer: Tensor,
    k_compressed: Tensor,
    head_weights: Tensor,
    visual_bias: Tensor | None = None,
) -> Tensor:
    """Dispatch: compiled fused path on CUDA, eager reference elsewhere."""
    if q_indexer.is_cuda:
        return _get_compiled_scores()(q_indexer, k_compressed, head_weights, visual_bias)  # type: ignore[operator,no-any-return]
    return lightning_indexer_scores_reference(q_indexer, k_compressed, head_weights, visual_bias)


def lightning_indexer_topk(
    q_indexer: Tensor,
    k_compressed: Tensor,
    head_weights: Tensor,
    top_k: int,
    block_size_m: int,
    visual_bias: Tensor | None = None,
) -> Tensor:
    """Dispatch: compiled fused path on CUDA, eager reference elsewhere."""
    if q_indexer.is_cuda:
        return _get_compiled_topk()(  # type: ignore[operator,no-any-return]
            q_indexer, k_compressed, head_weights, top_k, block_size_m, visual_bias,
        )
    return lightning_indexer_topk_reference(
        q_indexer, k_compressed, head_weights, top_k, block_size_m, visual_bias,
    )
