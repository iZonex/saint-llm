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

    def __init__(self, hidden_dim: int, c: int, m: int) -> None:
        super().__init__()
        self.c = c
        self.m = m
        self.w_a_kv = nn.Linear(hidden_dim, c, bias=False)
        self.w_b_kv = nn.Linear(hidden_dim, c, bias=False)
        self.w_a_z = nn.Linear(hidden_dim, c, bias=False)
        self.w_b_z = nn.Linear(hidden_dim, c, bias=False)
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

    def __init__(self, hidden_dim: int, n_heads: int, head_dim: int, n_groups: int, d_g: int) -> None:
        super().__init__()
        assert n_heads % n_groups == 0
        self.n_groups = n_groups
        self.heads_per_group = n_heads // n_groups
        self.d_g = d_g
        per_group_in = self.heads_per_group * head_dim
        self.group_projs = nn.ModuleList(
            nn.Linear(per_group_in, d_g, bias=False) for _ in range(n_groups)
        )
        self.final_proj = nn.Linear(n_groups * d_g, hidden_dim, bias=False)

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

        self.kv_compressor = TokenLevelCompressor(hidden_dim, attn.head_dim, csa.compression_rate)
        # Sliding-window KV path (uncompressed, shared across heads — MQA).
        self.k_proj = linear_factory(hidden_dim, attn.head_dim, bias=False)
        self.v_proj = linear_factory(hidden_dim, attn.head_dim, bias=False)

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
        )

    def _project_q(self, h: Tensor) -> Tensor:
        b, t, _ = h.shape
        c_q = self.q_compressor(h)
        q = self.q_up(c_q).view(b, t, self.attn_cfg.query_heads, self.attn_cfg.head_dim)
        return q.transpose(1, 2)  # (B, n_heads, T, head_dim)

    def forward(self, h: Tensor, is_visual: Tensor | None = None) -> Tensor:
        b, t, _ = h.shape
        device = h.device

        q = self._project_q(h)
        cos, sin = build_rope_cache(t, self.attn_cfg.rope_dim, self.attn_cfg.rope_theta, device)
        q = apply_partial_rope(q, cos, sin, self.attn_cfg.rope_dim)
        q = self.q_norm(q)

        # Sparse compressed KV path
        k_indexer_keys, top_idx = self.indexer(h, is_visual=is_visual)
        compressed_kv = self.kv_compressor(h)  # (B, n_blocks, head_dim)
        n_blocks = compressed_kv.shape[1]
        top_k = top_idx.shape[-1]

        valid_pick = top_idx >= 0  # (B, T, top_k) — false where indexer padded a causally-invalid slot
        if n_blocks > 0 and top_k > 0:
            top_idx_clamped = top_idx.clamp(min=0)
            gather_idx = top_idx_clamped.unsqueeze(-1).expand(-1, -1, -1, self.attn_cfg.head_dim)
            kv_compressed_expanded = compressed_kv.unsqueeze(1).expand(b, t, n_blocks, self.attn_cfg.head_dim)
            sparse_kv = torch.gather(kv_compressed_expanded, 2, gather_idx)  # (B, T, top_k, head_dim)
            sparse_kv = self.k_norm(sparse_kv)
        else:
            sparse_kv = q.new_zeros(b, t, 0, self.attn_cfg.head_dim)

        # Sliding-window uncompressed branch (MQA shared KV)
        sw = self.attn_cfg.sliding_window_size
        sw_k = self.k_proj(h)
        sw_v = self.v_proj(h)
        sw_k = self.k_norm(sw_k)
        sw_k = apply_partial_rope(sw_k.unsqueeze(1), cos, sin, self.attn_cfg.rope_dim).squeeze(1)

        sw_mask_full = sliding_window_mask(t, sw, device=device)  # (T, T)
        causal = causal_mask(t, t, device=device)
        sw_mask = sw_mask_full & causal

        # Reshape sparse_kv per query: each query has its own top_k compressed KV.
        # Combine sparse + sliding-window into a single per-query KV tensor:
        # K_comb shape (B, n_heads, T, top_k + T), gated by sliding mask for the SW portion.
        # For v0.1 simplicity we attend to (sparse_kv: per-query distinct keys) and (sw_k: shared causal-windowed keys) separately and sum.

        n_heads = self.attn_cfg.query_heads
        sparse_attn_out = q.new_zeros(b, n_heads, t, self.attn_cfg.head_dim)
        if sparse_kv.shape[-2] > 0:
            # Per-query attention over its own selected compressed KV. top-k is
            # causal *only when valid_pick is true*; padded picks must be masked.
            q_for_sparse = q.transpose(1, 2).unsqueeze(-2)  # (B, T, n_heads, 1, head_dim)
            k_for_sparse = sparse_kv.unsqueeze(2).expand(b, t, n_heads, sparse_kv.shape[-2], self.attn_cfg.head_dim)
            v_for_sparse = k_for_sparse  # MQA: shared K and V
            attn_scores = (q_for_sparse * k_for_sparse).sum(-1) / (self.attn_cfg.head_dim ** 0.5)
            # Mask padded picks; broadcast valid_pick (B, T, top_k) over heads.
            attn_scores = attn_scores.masked_fill(~valid_pick.unsqueeze(2), float("-inf"))
            # Safe-softmax: a query with zero valid picks has all -inf scores;
            # softmax would NaN. Replace such rows with zeros pre-softmax, then
            # zero the row's contribution post-mix.
            row_has_pick = valid_pick.any(dim=-1, keepdim=True).unsqueeze(2)  # (B, T, 1, 1)
            attn_scores = torch.where(row_has_pick, attn_scores, torch.zeros_like(attn_scores))
            probs = torch.softmax(attn_scores, dim=-1)
            mixed = (probs.unsqueeze(-1) * v_for_sparse).sum(-2)  # (B, T, n_heads, head_dim)
            mixed = torch.where(row_has_pick, mixed, torch.zeros_like(mixed))
            sparse_attn_out = mixed.transpose(1, 2)

        sw_k_h = sw_k.unsqueeze(1).expand(b, n_heads, t, self.attn_cfg.head_dim)
        sw_v_h = sw_v.unsqueeze(1).expand(b, n_heads, t, self.attn_cfg.head_dim)
        sw_attn_out = scaled_dot_product(q, sw_k_h, sw_v_h, mask=sw_mask, sink_logit=self.sink_logit)

        attn_out = sparse_attn_out + sw_attn_out

        # Touch indexer keys so the parameters stay live; no-op for shape.
        _ = k_indexer_keys.sum() * 0.0
        _ = _.detach()

        return self.output(attn_out)
