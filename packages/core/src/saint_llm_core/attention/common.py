"""Shared attention primitives: RMSNorm, partial RoPE, sliding-window mask, attention sink."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(var + self.eps)


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
    """Standard RoPE precomputation. head_dim must be even."""
    assert head_dim % 2 == 0, "RoPE requires even head_dim"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("t,f->tf", t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_partial_rope(x: Tensor, cos: Tensor, sin: Tensor, rope_dim: int) -> Tensor:
    """Apply RoPE to the last `rope_dim` dimensions of `x`.

    x shape: (..., T, head_dim). cos/sin shape: (T, rope_dim/2).
    """
    head_dim = x.shape[-1]
    assert rope_dim <= head_dim
    if rope_dim == 0:
        return x
    pass_through = x[..., :-rope_dim] if rope_dim < head_dim else None
    rope_part = x[..., -rope_dim:]

    x1 = rope_part[..., 0::2]
    x2 = rope_part[..., 1::2]

    while cos.dim() < x1.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    rot_x1 = x1 * cos - x2 * sin
    rot_x2 = x1 * sin + x2 * cos

    rotated = torch.stack((rot_x1, rot_x2), dim=-1).flatten(-2)
    if pass_through is None:
        return rotated
    return torch.cat([pass_through, rotated], dim=-1)


def causal_mask(t_q: int, t_k: int, device: torch.device | None = None) -> Tensor:
    """Lower-triangular mask: True = visible. Aligned to right (key index <= query index, with offset)."""
    q_idx = torch.arange(t_q, device=device).unsqueeze(-1)
    k_idx = torch.arange(t_k, device=device).unsqueeze(0)
    offset = t_k - t_q
    return k_idx <= (q_idx + offset)


def sliding_window_mask(t_q: int, window: int, device: torch.device | None = None) -> Tensor:
    """Square sliding-window mask: True = visible (within `window` tokens, including self)."""
    q_idx = torch.arange(t_q, device=device).unsqueeze(-1)
    k_idx = torch.arange(t_q, device=device).unsqueeze(0)
    in_window = (q_idx - k_idx) >= 0
    in_window &= (q_idx - k_idx) < window
    return in_window


def softmax_with_sink(scores: Tensor, sink_logit: Tensor | None) -> Tensor:
    """Softmax that includes a learnable sink logit in the denominator (per attention head).

    scores: (..., n_heads, T_q, T_k). sink_logit: (n_heads,) or None.
    """
    if sink_logit is None:
        return torch.softmax(scores, dim=-1)
    # Broadcast sink to (..., n_heads, T_q, 1)
    sink = sink_logit.view(*([1] * (scores.dim() - 3)), -1, 1, 1).expand(*scores.shape[:-1], 1)
    combined = torch.cat([scores, sink], dim=-1)
    probs = torch.softmax(combined, dim=-1)
    return probs[..., :-1]


def scaled_dot_product(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    sink_logit: Tensor | None = None,
) -> Tensor:
    """MQA-friendly attention with optional mask and attention sink.

    q: (B, n_heads, T_q, d). k, v: (B, 1, T_k, d) for shared-KV MQA, or matching n_heads.
    """
    head_dim = q.shape[-1]
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(head_dim)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = softmax_with_sink(scores, sink_logit)
    return torch.einsum("bhqk,bhkd->bhqd", probs, v)
