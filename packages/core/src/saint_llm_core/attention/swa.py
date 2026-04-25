"""Pure sliding-window multi-query attention for the first dense SWA layers.

Reference: DeepSeek V4 §4.2.1 — "For the first two layers, we use pure sliding window attention."
Used when no compressed branch is wanted (early layers focus on local patterns).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from saint_llm_core.attention.common import (
    RMSNorm,
    apply_partial_rope,
    build_rope_cache,
    causal_mask,
    scaled_dot_product,
    sliding_window_mask,
)
from saint_llm_core.attention.csa import GroupedOutputProjection
from saint_llm_core.config import AttentionConfig


class SWAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn: AttentionConfig) -> None:
        super().__init__()
        self.attn_cfg = attn
        self.q_compressor = nn.Linear(hidden_dim, attn.query_compression_dim, bias=False)
        self.q_up = nn.Linear(attn.query_compression_dim, attn.query_heads * attn.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, attn.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, attn.head_dim, bias=False)
        self.q_norm = RMSNorm(attn.head_dim)
        self.k_norm = RMSNorm(attn.head_dim)

        self.sink_logit: nn.Parameter | None
        if attn.use_attention_sink:
            self.sink_logit = nn.Parameter(torch.zeros(attn.query_heads))
        else:
            self.sink_logit = None

        self.output = GroupedOutputProjection(
            hidden_dim, attn.query_heads, attn.head_dim, attn.output_proj_groups, attn.attention_intermediate_dim,
        )

    def forward(self, h: Tensor, is_visual: Tensor | None = None) -> Tensor:  # noqa: ARG002
        b, t, _ = h.shape
        device = h.device
        cfg = self.attn_cfg

        c_q = self.q_compressor(h)
        q = self.q_up(c_q).view(b, t, cfg.query_heads, cfg.head_dim).transpose(1, 2)
        cos, sin = build_rope_cache(t, cfg.rope_dim, cfg.rope_theta, device)
        q = apply_partial_rope(q, cos, sin, cfg.rope_dim)
        q = self.q_norm(q)

        k = apply_partial_rope(self.k_proj(h), cos, sin, cfg.rope_dim)
        v = self.v_proj(h)
        k = self.k_norm(k)

        mask = sliding_window_mask(t, cfg.sliding_window_size, device=device) & causal_mask(t, t, device=device)
        k_h = k.unsqueeze(1).expand(b, cfg.query_heads, t, cfg.head_dim)
        v_h = v.unsqueeze(1).expand(b, cfg.query_heads, t, cfg.head_dim)

        attn_out = scaled_dot_product(q, k_h, v_h, mask=mask, sink_logit=self.sink_logit)
        return self.output(attn_out)
