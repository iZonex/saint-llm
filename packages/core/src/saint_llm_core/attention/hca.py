"""Heavily Compressed Attention (HCA).

Reference: DeepSeek V4 paper §2.3.2, Figure 4.

Same compression machinery as CSA but:
    - heavier compression rate m' (typ. 128 vs CSA's 4),
    - no Lightning indexer / no top-k sparse selection (dense attention over compressed KV),
    - sliding-window branch retained for local fine-grained dependencies.
"""

from __future__ import annotations

from typing import Any

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
from saint_llm_core.moe import LinearFactory, _default_linear


class HCA(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn: AttentionConfig,
        hca: HCAConfig,
        *,
        linear_factory: LinearFactory = _default_linear,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_cfg = attn
        self.hca_cfg = hca

        self.q_compressor = linear_factory(hidden_dim, attn.query_compression_dim, bias=False)
        self.q_up = linear_factory(attn.query_compression_dim, attn.query_heads * attn.head_dim, bias=False)

        self.kv_compressor = TokenLevelCompressor(
            hidden_dim, attn.head_dim, hca.compression_rate,
            linear_factory=linear_factory,
        )
        self.k_proj = linear_factory(hidden_dim, attn.head_dim, bias=False)
        self.v_proj = linear_factory(hidden_dim, attn.head_dim, bias=False)

        self.q_norm = RMSNorm(attn.head_dim)
        self.k_norm = RMSNorm(attn.head_dim)

        self.sink_logit: nn.Parameter | None
        if attn.use_attention_sink:
            self.sink_logit = nn.Parameter(torch.zeros(attn.query_heads))
        else:
            self.sink_logit = None

        self.output = GroupedOutputProjection(
            hidden_dim, attn.query_heads, attn.head_dim, attn.output_proj_groups, attn.attention_intermediate_dim,
            linear_factory=linear_factory,
        )

        # MuonClip QK-Clip telemetry — see SWAttention for the rationale.
        self._last_max_attn_logit: float = 0.0

    def qk_clip_targets(self) -> list[Tensor]:
        """Q/K projection weights to rescale on QK-Clip trigger."""
        return [self.q_up.weight, self.k_proj.weight]

    def forward(
        self,
        h: Tensor,
        is_visual: Tensor | None = None,
        *,
        kv_cache: Any | None = None,
    ) -> Tensor:
        """Heavily-compressed attention with optional KV cache.

        ``kv_cache`` is duck-typed (``HCAKVCacheLayer``-shaped: ``length: int``,
        ``append(h_new, sw_k_new, sw_v_new, compressor=...) -> (compressed,
        sw_k_full, sw_v_full)``). Without a cache, the math is identical to
        the pre-cache implementation.
        """
        b, t_new, _ = h.shape
        device = h.device
        cfg = self.attn_cfg
        cache_offset = int(kv_cache.length) if kv_cache is not None else 0
        total_len = cache_offset + t_new

        # Q with RoPE at absolute positions.
        c_q = self.q_compressor(h)
        q = self.q_up(c_q).view(b, t_new, cfg.query_heads, cfg.head_dim).transpose(1, 2)
        cos_full, sin_full = build_rope_cache(total_len, cfg.rope_dim, cfg.rope_theta, device)
        cos_new = cos_full[cache_offset:total_len]
        sin_new = sin_full[cache_offset:total_len]
        q = apply_partial_rope(q, cos_new, sin_new, cfg.rope_dim)
        q = self.q_norm(q)

        # SW branch K/V for new tokens.
        sw_k_new = apply_partial_rope(self.k_proj(h), cos_new, sin_new, cfg.rope_dim)
        sw_v_new = self.v_proj(h)
        sw_k_new = self.k_norm(sw_k_new)

        if kv_cache is not None:
            compressed_kv, sw_k, sw_v = kv_cache.append(
                h, sw_k_new, sw_v_new, compressor=self.kv_compressor,
            )
            n_blocks = compressed_kv.shape[1]
            sw_total_len = sw_k.shape[1]
        else:
            compressed_kv = self.kv_compressor(h)
            n_blocks = compressed_kv.shape[1]
            sw_k = sw_k_new
            sw_v = sw_v_new
            sw_total_len = t_new

        n_heads = cfg.query_heads
        m = self.hca_cfg.compression_rate

        compressed_attn_out = q.new_zeros(b, n_heads, t_new, cfg.head_dim)
        if n_blocks > 0:
            compressed_kv_n = self.k_norm(compressed_kv)
            q_pos = torch.arange(cache_offset, total_len, device=device).unsqueeze(-1)
            block_end = torch.arange(n_blocks, device=device).unsqueeze(0) * m + (m - 1)
            valid_2d = block_end < q_pos  # (t_new, n_blocks)
            valid = valid_2d.unsqueeze(0).unsqueeze(0).expand(b, n_heads, t_new, n_blocks)
            ck = compressed_kv_n.unsqueeze(1).expand(b, n_heads, n_blocks, cfg.head_dim)
            compressed_attn_out = scaled_dot_product(q, ck, ck, mask=valid, sink_logit=self.sink_logit)
            if self.sink_logit is None:
                row_has_block = valid_2d.any(dim=-1)  # (t_new,)
                compressed_attn_out = torch.where(
                    row_has_block.view(1, 1, t_new, 1),
                    compressed_attn_out,
                    torch.zeros_like(compressed_attn_out),
                )

        # SW mask: causal + sliding window. Square when no cache; non-square
        # (t_new x sw_total_len) when running incremental.
        if cache_offset == 0:
            sw_mask = (
                sliding_window_mask(sw_total_len, cfg.sliding_window_size, device=device)
                & causal_mask(sw_total_len, sw_total_len, device=device)
            )
        else:
            q_pos = torch.arange(cache_offset, total_len, device=device).unsqueeze(-1)
            k_pos = torch.arange(sw_total_len, device=device).unsqueeze(0)
            sw_mask = (k_pos <= q_pos) & ((q_pos - k_pos) < cfg.sliding_window_size)

        sw_k_h = sw_k.unsqueeze(1).expand(b, n_heads, sw_total_len, cfg.head_dim)
        sw_v_h = sw_v.unsqueeze(1).expand(b, n_heads, sw_total_len, cfg.head_dim)

        def _observe(scores: Tensor) -> None:
            # MuonClip telemetry — track max logit on the SW path; same
            # projection weights as the sparse path, so this is sufficient.
            self._last_max_attn_logit = float(scores.detach().abs().max())

        sw_attn_out = scaled_dot_product(
            q, sw_k_h, sw_v_h, mask=sw_mask, sink_logit=self.sink_logit,
            score_observer=_observe,
            logit_softcap=self.attn_cfg.logit_softcap,
        )

        return self.output(compressed_attn_out + sw_attn_out)
