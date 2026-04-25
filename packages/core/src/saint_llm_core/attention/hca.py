"""Heavily Compressed Attention (HCA).

Reference: DeepSeek V4 paper §2.3.2, Figure 4.

Same compression machinery as CSA but:
    - heavier compression rate m' (typ. 128 vs CSA's 4),
    - no Lightning indexer / no top-k sparse selection (dense attention over compressed KV),
    - sliding-window branch retained for local fine-grained dependencies.
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
from saint_llm_core.attention.csa import GroupedOutputProjection, TokenLevelCompressor
from saint_llm_core.config import AttentionConfig, HCAConfig


class HCA(nn.Module):
    def __init__(self, hidden_dim: int, attn: AttentionConfig, hca: HCAConfig) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_cfg = attn
        self.hca_cfg = hca

        self.q_compressor = nn.Linear(hidden_dim, attn.query_compression_dim, bias=False)
        self.q_up = nn.Linear(attn.query_compression_dim, attn.query_heads * attn.head_dim, bias=False)

        self.kv_compressor = TokenLevelCompressor(hidden_dim, attn.head_dim, hca.compression_rate)
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

    def forward(self, h: Tensor, is_visual: Tensor | None = None) -> Tensor:
        b, t, _ = h.shape
        device = h.device

        c_q = self.q_compressor(h)
        q = self.q_up(c_q).view(b, t, self.attn_cfg.query_heads, self.attn_cfg.head_dim).transpose(1, 2)
        cos, sin = build_rope_cache(t, self.attn_cfg.rope_dim, self.attn_cfg.rope_theta, device)
        q = apply_partial_rope(q, cos, sin, self.attn_cfg.rope_dim)
        q = self.q_norm(q)

        compressed_kv = self.kv_compressor(h)  # (B, n_blocks, head_dim) MQA shared KV
        n_blocks = compressed_kv.shape[1]
        n_heads = self.attn_cfg.query_heads

        compressed_attn_out = q.new_zeros(b, n_heads, t, self.attn_cfg.head_dim)
        if n_blocks > 0:
            compressed_kv_n = self.k_norm(compressed_kv)
            # Causal mask: query at position t may attend to blocks complete up to t.
            m = self.hca_cfg.compression_rate
            q_idx = torch.arange(t, device=device).unsqueeze(-1)
            block_end = torch.arange(n_blocks, device=device).unsqueeze(0) * m + (m - 1)
            valid_2d = block_end < q_idx  # (T, n_blocks)
            valid = valid_2d.unsqueeze(0).unsqueeze(0).expand(b, n_heads, t, n_blocks)
            ck = compressed_kv_n.unsqueeze(1).expand(b, n_heads, n_blocks, self.attn_cfg.head_dim)
            compressed_attn_out = scaled_dot_product(q, ck, ck, mask=valid, sink_logit=self.sink_logit)
            if self.sink_logit is None:
                # All-(-inf) rows would softmax to NaN. Zero them post-hoc; their
                # contribution is 0 (no valid block to attend to).
                row_has_block = valid_2d.any(dim=-1)  # (T,)
                compressed_attn_out = torch.where(
                    row_has_block.view(1, 1, t, 1),
                    compressed_attn_out,
                    torch.zeros_like(compressed_attn_out),
                )

        # Sliding-window branch
        sw_k = apply_partial_rope(self.k_proj(h).unsqueeze(1), cos, sin, self.attn_cfg.rope_dim).squeeze(1)
        sw_v = self.v_proj(h)
        sw_k = self.k_norm(sw_k)
        sw_mask = sliding_window_mask(t, self.attn_cfg.sliding_window_size, device=device) & causal_mask(t, t, device=device)
        sw_k_h = sw_k.unsqueeze(1).expand(b, n_heads, t, self.attn_cfg.head_dim)
        sw_v_h = sw_v.unsqueeze(1).expand(b, n_heads, t, self.attn_cfg.head_dim)
        sw_attn_out = scaled_dot_product(q, sw_k_h, sw_v_h, mask=sw_mask, sink_logit=self.sink_logit)

        return self.output(compressed_attn_out + sw_attn_out)
