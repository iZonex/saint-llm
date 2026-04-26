"""Tree Attention — sequence-parallel attention via online softmax.

For very long contexts the K/V sequence is partitioned across N
chunks (or N ranks in the distributed case). Each chunk computes a
local attention partial; partials are combined via the same
log-sum-exp recurrence Flash Attention uses internally::

    given partials (m_i, l_i, o_i) where
        m_i = local rowmax of QK^T scores
        l_i = sum_j exp(score_j - m_i)  over chunk i
        o_i = sum_j exp(score_j - m_i) * v_j / l_i

    combine two partials (m_a, l_a, o_a) and (m_b, l_b, o_b):
        m   = max(m_a, m_b)
        l   = exp(m_a - m) * l_a + exp(m_b - m) * l_b
        o   = (exp(m_a - m) * l_a * o_a + exp(m_b - m) * l_b * o_b) / l

The combine is associative, so the partials can be reduced via a
tree (or a ring all-reduce in the distributed case). Tree Attention
is the recommended algorithm for long-context attention across
multi-GPU clusters because it requires only one all-reduce per
attention layer rather than the O(N^2) communication of naive
sequence parallelism.

This module ships :func:`tree_attention` — pure-tensor implementation
that walks chunks locally, mathematically equivalent to dense
attention. Production distributed version swaps the chunk loop for
``torch.distributed.all_reduce`` over the per-rank partials.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def tree_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    n_chunks: int = 1,
    mask: Tensor | None = None,
) -> Tensor:
    """Sequence-parallel attention via online softmax over K/V chunks.

    Args:
        q:        ``(B, H, T_q, D)`` query tensor.
        k:        ``(B, H, T_k, D)`` key tensor.
        v:        ``(B, H, T_k, D)`` value tensor.
        n_chunks: number of K/V chunks to split along ``T_k``. Must be
            a positive divisor of ``T_k`` for a clean split; trailing
            tokens that don't fit fall into the last chunk.
        mask:     optional ``(T_q, T_k)`` bool mask (True = visible).
            Causal / sliding-window masks are passed through as-is;
            the function slices the per-chunk view internally.

    Returns:
        ``(B, H, T_q, D)`` attention output. Mathematically equivalent
        to dense scaled-dot-product within FP tolerance.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"q/k/v must be 4-D (B, H, T, D); got shapes "
            f"{tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}",
        )
    if k.shape != v.shape:
        raise ValueError(f"k and v must share shape; got {tuple(k.shape)} vs {tuple(v.shape)}")
    if q.shape[:2] != k.shape[:2]:
        raise ValueError(
            f"q and k must agree on (B, H); got {tuple(q.shape[:2])} vs {tuple(k.shape[:2])}",
        )
    if n_chunks <= 0:
        raise ValueError(f"n_chunks must be positive; got {n_chunks}")

    t_k = k.shape[-2]
    chunk_size = max(1, math.ceil(t_k / n_chunks))
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    # Running partials.
    out_dtype = q.dtype
    m = torch.full(q.shape[:-1], float("-inf"), dtype=torch.float32, device=q.device)
    l = torch.zeros(q.shape[:-1], dtype=torch.float32, device=q.device)  # noqa: E741
    acc = torch.zeros(q.shape, dtype=torch.float32, device=q.device)

    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)
    v_f32 = v.to(torch.float32)

    for start in range(0, t_k, chunk_size):
        end = min(t_k, start + chunk_size)
        k_chunk = k_f32[..., start:end, :]
        v_chunk = v_f32[..., start:end, :]
        # (B, H, T_q, chunk_size)
        scores = torch.einsum("bhqd,bhkd->bhqk", q_f32, k_chunk) * scale
        if mask is not None:
            chunk_mask = mask[:, start:end]
            scores = scores.masked_fill(~chunk_mask, float("-inf"))

        # Per-chunk rowmax and exp-stabilized weights.
        chunk_max = scores.amax(dim=-1)  # (B, H, T_q)
        new_m = torch.maximum(m, chunk_max)
        # exp(m - new_m) for the existing accumulator's correction.
        scale_old = torch.exp(m - new_m)  # (B, H, T_q)
        # exp(scores - new_m) for this chunk.
        weights = torch.exp(scores - new_m.unsqueeze(-1))  # (B, H, T_q, chunk_size)
        chunk_l = weights.sum(dim=-1)  # (B, H, T_q)

        l = scale_old * l + chunk_l  # noqa: E741
        acc = acc * scale_old.unsqueeze(-1) + torch.einsum(
            "bhqk,bhkd->bhqd", weights, v_chunk,
        )
        m = new_m

    # Rows where every chunk score was -inf (e.g. all masked) end up
    # with l == 0; clamp to avoid 0/0 → NaN.
    l_safe = l.clamp(min=1e-30).unsqueeze(-1)
    out = acc / l_safe
    # Zero out rows that had no visible keys at all.
    if mask is not None:
        any_visible = mask.any(dim=-1)  # (T_q,)
        out = torch.where(any_visible.view(1, 1, -1, 1), out, torch.zeros_like(out))
    return out.to(out_dtype)
