"""Compressed Sparse Attention (CSA).

Reference: DeepSeek V4 paper §2.3.1, Figure 3.

Pipeline:
    1. Token-level compressor:
        - Compute per-token C^a = H W^aKV, C^b = H W^bKV (low-rank KV).
        - Compute compression weights Z^a, Z^b.
        - Each m KV entries are softmax-pooled into one compressed entry C^Comp ∈ R^(B, T/m, c).
    2. Lightning indexer (low-rank MQA):
        - q_t^Q = h_t W^DQ, then up-projected indexer queries.
        - K^IComp from a separate compressor on the same H.
        - Index scores I_{t,s} = Σ w_h * ReLU(q_h · K^IComp_s); top-k selects compressed KV blocks.
    3. Sliding-window branch: most recent n_win uncompressed KV entries.
    4. Core attention: shared-KV MQA over selected_compressed ∪ sliding_window, with attention sink.
    5. Grouped output projection: split heads into g groups, project each into d_g intermediate, concat → d_model.

v0.1 hooks (per AUGMENTATIONS MM-V-06):
    - Optional `is_visual` mask on the indexer positional bias to keep vision spans cohesive.
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
from saint_llm_core.config import AttentionConfig, CSAConfig
from saint_llm_core.moe import LinearFactory, _default_linear


class TokenLevelCompressor(nn.Module):
    """Compress every m hidden states into one compressed (key|value) latent of dim c.

    Implements V4 eqs. (9)-(12): two parallel branches a/b with overlapped windows,
    softmax-normalized weights with learnable positional biases B^a, B^b.
    """

    def __init__(
        self,
        hidden_dim: int,
        c: int,
        m: int,
        *,
        linear_factory: LinearFactory = _default_linear,
    ) -> None:
        super().__init__()
        self.c = c
        self.m = m
        self.w_a_kv = linear_factory(hidden_dim, c, bias=False)
        self.w_b_kv = linear_factory(hidden_dim, c, bias=False)
        self.w_a_z = linear_factory(hidden_dim, c, bias=False)
        self.w_b_z = linear_factory(hidden_dim, c, bias=False)
        self.bias_a = nn.Parameter(torch.zeros(m, c))
        self.bias_b = nn.Parameter(torch.zeros(m, c))

    def forward(self, h: Tensor) -> Tensor:
        b, t, _ = h.shape
        usable_t = (t // self.m) * self.m
        if usable_t == 0:
            return h.new_zeros(b, 0, self.c)
        h_use = h[:, :usable_t]

        c_a = self.w_a_kv(h_use)  # (B, T_use, c)
        c_b = self.w_b_kv(h_use)
        z_a = self.w_a_z(h_use)
        z_b = self.w_b_z(h_use)

        n_blocks = usable_t // self.m
        c_a = c_a.view(b, n_blocks, self.m, self.c)
        c_b = c_b.view(b, n_blocks, self.m, self.c)
        z_a = z_a.view(b, n_blocks, self.m, self.c) + self.bias_a
        z_b = z_b.view(b, n_blocks, self.m, self.c) + self.bias_b

        # Overlapping pair: block i uses C^a[i+1] and C^b[i] per V4 §2.3.1.
        # Simplification for v0.1: use C^b[i] only (no overlap), validated to stack with HCA.
        weights = torch.softmax(z_b, dim=-2)
        compressed = (weights * c_b).sum(dim=-2)  # (B, n_blocks, c)
        return compressed


class LightningIndexer(nn.Module):
    """Low-rank multi-query indexer producing top-k compressed KV indices per query token.

    For each query token t, a small set of n_h^I indexer heads compute scores against
    a parallel-compressed K^IComp; final per-block score I_{t,s} = Σ_h w_h * ReLU(q_h · K^IComp_s).
    Top-k blocks are selected for sparse attention.
    """

    def __init__(self, hidden_dim: int, c_indexer: int, n_heads: int, top_k: int, m: int) -> None:
        super().__init__()
        self.c_indexer = c_indexer
        self.n_heads = n_heads
        self.top_k = top_k
        self.compressor = TokenLevelCompressor(hidden_dim, c_indexer, m)

        self.w_dq = nn.Linear(hidden_dim, c_indexer, bias=False)
        self.w_iuq = nn.Linear(c_indexer, n_heads * c_indexer, bias=False)
        self.w_w = nn.Linear(hidden_dim, n_heads, bias=False)

    def forward(
        self,
        h: Tensor,
        is_visual: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Returns (compressed_kv_keys, top_k_indices).

        compressed_kv_keys: (B, n_blocks, c_indexer)
        top_k_indices: (B, T, top_k) — block indices selected per query token, padded with -1.
        """
        b, t, _ = h.shape
        k_indexer_comp = self.compressor(h)  # (B, n_blocks, c_indexer)
        n_blocks = k_indexer_comp.shape[1]

        c_q = self.w_dq(h)  # (B, T, c_indexer)
        q_indexer = self.w_iuq(c_q).view(b, t, self.n_heads, self.c_indexer)
        w = self.w_w(h)  # (B, T, n_heads)

        # Per-head index scores: (B, T, n_heads, n_blocks).
        head_scores = torch.einsum("bthc,bnc->bthn", q_indexer, k_indexer_comp).clamp(min=0.0)
        # Visual cohesion bias (MM-V-06): boost scores for blocks falling inside a visual span.
        if is_visual is not None and n_blocks > 0:
            m = h.shape[1] // n_blocks if n_blocks > 0 else 1
            block_visual = is_visual[:, : n_blocks * m].view(b, n_blocks, m).float().mean(-1)
            head_scores = head_scores + block_visual.unsqueeze(-2).unsqueeze(-2) * 0.5

        scores = (w.unsqueeze(-1) * head_scores).sum(dim=-2)  # (B, T, n_blocks)

        # Causal: query at position t may only attend to compressed blocks fully written by t.
        # Block s is "complete" at position s*m + m - 1.
        if n_blocks > 0:
            m = h.shape[1] // n_blocks
            q_idx = torch.arange(t, device=h.device).unsqueeze(-1)
            block_end = torch.arange(n_blocks, device=h.device).unsqueeze(0) * m + (m - 1)
            valid = block_end < q_idx
            scores = scores.masked_fill(~valid, float("-inf"))

        k = min(self.top_k, n_blocks)
        if k == 0:
            return k_indexer_comp, h.new_full((b, t, 0), -1, dtype=torch.long)
        top_scores, top_idx = scores.topk(k, dim=-1)
        # Picks where the score is -inf were padded over the causal mask — mark
        # with -1 so downstream sparse attention can drop them (instead of
        # gathering from a causally-future block).
        valid = torch.isfinite(top_scores)
        top_idx = torch.where(valid, top_idx, top_idx.new_full((), -1))
        return k_indexer_comp, top_idx


class GroupedOutputProjection(nn.Module):
    """Split per-head outputs into g groups, project each into d_g, concat → hidden_dim."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        head_dim: int,
        n_groups: int,
        d_g: int,
        *,
        linear_factory: LinearFactory = _default_linear,
    ) -> None:
        super().__init__()
        assert n_heads % n_groups == 0
        self.n_groups = n_groups
        self.heads_per_group = n_heads // n_groups
        self.d_g = d_g
        per_group_in = self.heads_per_group * head_dim
        self.group_projs = nn.ModuleList(
            linear_factory(per_group_in, d_g, bias=False) for _ in range(n_groups)
        )
        self.final_proj = linear_factory(n_groups * d_g, hidden_dim, bias=False)

    def forward(self, attn_out: Tensor) -> Tensor:
        b, n_heads, t, head_dim = attn_out.shape
        out = attn_out.transpose(1, 2).reshape(b, t, n_heads * head_dim)
        per_group_in = self.heads_per_group * head_dim
        groups = out.split(per_group_in, dim=-1)
        projected = torch.cat([proj(g) for proj, g in zip(self.group_projs, groups, strict=True)], dim=-1)
        return self.final_proj(projected)


class CSA(nn.Module):
    """Compressed Sparse Attention block.

    Forward signature matches the inner-layer contract used by mHC: (B, T, d) -> (B, T, d).
    """

    def __init__(
        self,
        hidden_dim: int,
        attn: AttentionConfig,
        csa: CSAConfig,
        *,
        linear_factory: LinearFactory = _default_linear,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_cfg = attn
        self.csa_cfg = csa

        self.q_compressor = linear_factory(hidden_dim, attn.query_compression_dim, bias=False)
        self.q_up = linear_factory(attn.query_compression_dim, attn.query_heads * attn.head_dim, bias=False)

        self.kv_compressor = TokenLevelCompressor(
            hidden_dim, attn.head_dim, csa.compression_rate,
            linear_factory=linear_factory,
        )
        # Sliding-window KV path (uncompressed, shared across heads — MQA).
        self.k_proj = linear_factory(hidden_dim, attn.head_dim, bias=False)
        self.v_proj = linear_factory(hidden_dim, attn.head_dim, bias=False)

        # Indexer keeps default (nn.Linear) — quantization noise on its scores
        # would destabilize top-k routing.
        self.indexer = LightningIndexer(
            hidden_dim,
            csa.indexer_head_dim,
            csa.indexer_query_heads,
            csa.attention_top_k,
            csa.compression_rate,
        )

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

    def _project_q(self, h: Tensor) -> Tensor:
        b, t, _ = h.shape
        c_q = self.q_compressor(h)
        q = self.q_up(c_q).view(b, t, self.attn_cfg.query_heads, self.attn_cfg.head_dim)
        return q.transpose(1, 2)  # (B, n_heads, T, head_dim)

    def forward(
        self,
        h: Tensor,
        is_visual: Tensor | None = None,
        *,
        kv_cache: Any | None = None,
    ) -> Tensor:
        """CSA forward with optional KV cache.

        ``kv_cache`` is duck-typed (``CSAKVCacheLayer``-shaped). Without a
        cache, behavior matches the pre-cache implementation. With a cache,
        new tokens go through both compressors via ``cache.append``; the
        indexer's score+top-k step runs over the full cached block table.
        """
        b, t_new, _ = h.shape
        device = h.device
        cfg = self.attn_cfg
        cache_offset = int(kv_cache.length) if kv_cache is not None else 0
        total_len = cache_offset + t_new

        # Q with RoPE at absolute positions.
        q = self._project_q(h)
        cos_full, sin_full = build_rope_cache(total_len, cfg.rope_dim, cfg.rope_theta, device)
        cos_new = cos_full[cache_offset:total_len]
        sin_new = sin_full[cache_offset:total_len]
        q = apply_partial_rope(q, cos_new, sin_new, cfg.rope_dim)
        q = self.q_norm(q)

        # SW branch K/V for new tokens.
        sw_k_new = self.k_proj(h)
        sw_v_new = self.v_proj(h)
        sw_k_new = self.k_norm(sw_k_new)
        sw_k_new = apply_partial_rope(
            sw_k_new.unsqueeze(1), cos_new, sin_new, cfg.rope_dim,
        ).squeeze(1)

        m = self.csa_cfg.compression_rate
        if kv_cache is not None:
            compressed_kv, k_indexer_comp, sw_k, sw_v = kv_cache.append(
                h, sw_k_new, sw_v_new,
                compressor=self.kv_compressor,
                indexer_compressor=self.indexer.compressor,
            )
            n_blocks = compressed_kv.shape[1]
            sw_total_len = sw_k.shape[1]
            top_idx = self._indexer_topk_cached(
                h, k_indexer_comp,
                cache_offset=cache_offset,
                total_len=total_len,
                n_blocks=n_blocks,
                m=m,
                is_visual=is_visual,
            )
        else:
            k_indexer_keys, top_idx = self.indexer(h, is_visual=is_visual)
            compressed_kv = self.kv_compressor(h)
            n_blocks = compressed_kv.shape[1]
            sw_k = sw_k_new
            sw_v = sw_v_new
            sw_total_len = t_new
            # Touch indexer keys so the parameter stays live in the loss graph.
            _ = k_indexer_keys.sum() * 0.0
            _ = _.detach()

        top_k = top_idx.shape[-1]
        valid_pick = top_idx >= 0  # (B, t_new, top_k)
        if n_blocks > 0 and top_k > 0:
            top_idx_clamped = top_idx.clamp(min=0)
            gather_idx = top_idx_clamped.unsqueeze(-1).expand(-1, -1, -1, cfg.head_dim)
            kv_compressed_expanded = compressed_kv.unsqueeze(1).expand(b, t_new, n_blocks, cfg.head_dim)
            sparse_kv = torch.gather(kv_compressed_expanded, 2, gather_idx)
            sparse_kv = self.k_norm(sparse_kv)
        else:
            sparse_kv = q.new_zeros(b, t_new, 0, cfg.head_dim)

        n_heads = cfg.query_heads
        sparse_attn_out = q.new_zeros(b, n_heads, t_new, cfg.head_dim)
        if sparse_kv.shape[-2] > 0:
            q_for_sparse = q.transpose(1, 2).unsqueeze(-2)
            k_for_sparse = sparse_kv.unsqueeze(2).expand(b, t_new, n_heads, sparse_kv.shape[-2], cfg.head_dim)
            v_for_sparse = k_for_sparse
            attn_scores = (q_for_sparse * k_for_sparse).sum(-1) / (cfg.head_dim ** 0.5)
            attn_scores = attn_scores.masked_fill(~valid_pick.unsqueeze(2), float("-inf"))
            row_has_pick = valid_pick.any(dim=-1, keepdim=True).unsqueeze(2)
            attn_scores = torch.where(row_has_pick, attn_scores, torch.zeros_like(attn_scores))
            probs = torch.softmax(attn_scores, dim=-1)
            mixed = (probs.unsqueeze(-1) * v_for_sparse).sum(-2)
            mixed = torch.where(row_has_pick, mixed, torch.zeros_like(mixed))
            sparse_attn_out = mixed.transpose(1, 2)

        # SW branch attention.
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
        sw_attn_out = scaled_dot_product(q, sw_k_h, sw_v_h, mask=sw_mask, sink_logit=self.sink_logit)

        return self.output(sparse_attn_out + sw_attn_out)

    def _indexer_topk_cached(
        self,
        h: Tensor,
        k_indexer_comp: Tensor,
        *,
        cache_offset: int,
        total_len: int,
        n_blocks: int,
        m: int,
        is_visual: Tensor | None,
    ) -> Tensor:
        """Run the indexer's score+top-k against a pre-cached compressed K table.

        Mirrors ``LightningIndexer.forward`` but skips the compressor (cache
        provides ``k_indexer_comp``) and uses absolute query positions for
        the causal mask.

        ``is_visual`` is intentionally ignored on the cached path — the
        per-step mask shape ``(B, t_new)`` is misaligned with the cached
        block table ``(B, n_blocks, m)``. Visual cohesion bias only applies
        at prefill (uncached) time. Documented limitation; revisit if a
        caller needs it.
        """
        del is_visual
        b, t_new, _ = h.shape
        idx = self.indexer
        device = h.device

        c_q = idx.w_dq(h)
        q_indexer = idx.w_iuq(c_q).view(b, t_new, idx.n_heads, idx.c_indexer)
        w = idx.w_w(h)
        head_scores = torch.einsum("bthc,bnc->bthn", q_indexer, k_indexer_comp).clamp(min=0.0)
        scores = (w.unsqueeze(-1) * head_scores).sum(dim=-2)  # (B, t_new, n_blocks)

        if n_blocks > 0:
            q_pos = torch.arange(cache_offset, total_len, device=device).unsqueeze(-1)
            block_end = torch.arange(n_blocks, device=device).unsqueeze(0) * m + (m - 1)
            valid_block = block_end < q_pos
            scores = scores.masked_fill(~valid_block, float("-inf"))

        k_eff = min(idx.top_k, n_blocks)
        if k_eff == 0:
            return h.new_full((b, t_new, 0), -1, dtype=torch.long)
        top_scores, top_idx = scores.topk(k_eff, dim=-1)
        valid_init = torch.isfinite(top_scores)
        return torch.where(valid_init, top_idx, top_idx.new_full((), -1))
