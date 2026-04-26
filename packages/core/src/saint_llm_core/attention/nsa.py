"""NSA — Native Sparse Attention (ARCH-03 / arXiv 2502.11089).

DeepSeek's NSA replaces dense attention with three parallel branches
combined via learned gates:

* **Compressed branch** — global coarse-grained attention. Each
  block of ``compression_rate`` tokens is mean-pooled into one
  compressed key/value; queries attend to the full compressed
  history.
* **Selective branch** — top-k block selection. Block-level scores
  rank past blocks by relevance to the current query; the top-k
  blocks contribute their *uncompressed* tokens to attention.
* **Sliding-window branch** — local high-resolution attention over
  the most recent ``window`` tokens.

Three branches are computed in parallel; their outputs are combined
by a per-head learned gate (sigmoid + normalize). The whole thing is
**natively trainable** (no retrofit), hardware-aligned (block-
structured access), and faster than dense attention end-to-end on
long context per the published evaluation.

This module ships :class:`NSAttention` as a drop-in alternative to
the project's existing ``CSA``/``HCA``/``SWA`` interleave. The
existing modules stay in place — NSA is opt-in via configuration.

Reference:
    arXiv 2502.11089 (DeepSeek NSA)
"""

from __future__ import annotations

import math
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
from saint_llm_core.attention.csa import GroupedOutputProjection
from saint_llm_core.config import AttentionConfig
from saint_llm_core.moe import LinearFactory, _default_linear


def _block_compress(x: Tensor, *, block_size: int) -> Tensor:
    """Mean-pool ``x`` along the time dim into blocks of ``block_size``.

    ``x`` shape ``(..., T, D)``; returns ``(..., T // block_size, D)``.
    Trailing tokens that don't form a full block are dropped (the
    sliding-window branch covers them anyway). When ``T < block_size``
    the result has zero blocks along the time dim.
    """
    n = x.shape[-2]
    n_blocks = n // block_size if block_size > 0 else 0
    if n_blocks == 0:
        return x.new_zeros(*x.shape[:-2], 0, x.shape[-1])
    trimmed = x[..., : n_blocks * block_size, :]
    reshaped = trimmed.reshape(
        *x.shape[:-2], n_blocks, block_size, x.shape[-1],
    )
    return reshaped.mean(-2)


class NSAttention(nn.Module):
    """NSA three-branch attention with learned gating.

    Args:
        hidden_dim:        residual dim.
        attn:              shared :class:`AttentionConfig` (RoPE, head dims,
            sliding-window size, query heads, etc.).
        compression_rate:  block size for the compressed branch.
            Published recipe: 32-64 for long context.
        selection_k:       top-k blocks the selective branch attends
            to per query. 16-32 typical.
        linear_factory:    quant-aware linear factory (per
            ``ModelConfig.linear_quant``).
    """

    def __init__(
        self,
        hidden_dim: int,
        attn: AttentionConfig,
        *,
        compression_rate: int = 32,
        selection_k: int = 16,
        linear_factory: LinearFactory = _default_linear,
    ) -> None:
        super().__init__()
        self.attn_cfg = attn
        self.compression_rate = compression_rate
        self.selection_k = selection_k

        self.q_compressor = linear_factory(
            hidden_dim, attn.query_compression_dim, bias=False,
        )
        self.q_up = linear_factory(
            attn.query_compression_dim, attn.query_heads * attn.head_dim, bias=False,
        )
        self.k_proj = linear_factory(hidden_dim, attn.head_dim, bias=False)
        self.v_proj = linear_factory(hidden_dim, attn.head_dim, bias=False)

        self.q_norm = RMSNorm(attn.head_dim)
        self.k_norm = RMSNorm(attn.head_dim)

        # Per-token-per-head 3-way gate (compressed / selective / sliding).
        # Zero-init weight + bias gives equal-mix start (sigmoid(0) = 0.5
        # per branch, then normalize -> 1/3 each).
        self.branch_gate = nn.Linear(attn.head_dim, 3, bias=True)
        nn.init.zeros_(self.branch_gate.weight)
        nn.init.zeros_(self.branch_gate.bias)

        self.sink_logit: nn.Parameter | None
        if attn.use_attention_sink:
            self.sink_logit = nn.Parameter(torch.zeros(attn.query_heads))
        else:
            self.sink_logit = None

        self.output = GroupedOutputProjection(
            hidden_dim,
            attn.query_heads,
            attn.head_dim,
            attn.output_proj_groups,
            attn.attention_intermediate_dim,
            linear_factory=linear_factory,
        )

        # MuonClip telemetry shared with the rest of the attention modules.
        self._last_max_attn_logit: float = 0.0

    def qk_clip_targets(self) -> list[Tensor]:
        return [self.q_up.weight, self.k_proj.weight]

    def forward(
        self,
        h: Tensor,
        is_visual: Tensor | None = None,
        *,
        kv_cache: Any | None = None,
    ) -> Tensor:
        """Three-branch NSA forward.

        ``kv_cache`` is reserved — NSA's cache layer is a v0.0.1 polish
        item; this forward operates on the full sequence each call. The
        signature matches the project's other attention modules so
        :class:`SaintLLM` can swap NSA in transparently.
        """
        del is_visual  # NSA doesn't apply CSA's is_visual cohesion bias.
        b, t, _ = h.shape
        device = h.device
        cfg = self.attn_cfg

        # --- Q / K / V projections + RoPE -----------------------------
        c_q = self.q_compressor(h)
        q = self.q_up(c_q).view(b, t, cfg.query_heads, cfg.head_dim).transpose(1, 2)
        cos_full, sin_full = build_rope_cache(t, cfg.rope_dim, cfg.rope_theta, device)
        q = apply_partial_rope(q, cos_full, sin_full, cfg.rope_dim)
        q = self.q_norm(q)

        k = apply_partial_rope(self.k_proj(h), cos_full, sin_full, cfg.rope_dim)
        v = self.v_proj(h)
        k = self.k_norm(k)

        n_heads = cfg.query_heads
        k_h = k.unsqueeze(1).expand(b, n_heads, t, cfg.head_dim)
        v_h = v.unsqueeze(1).expand(b, n_heads, t, cfg.head_dim)

        def _track_logit(scores: Tensor) -> None:
            self._last_max_attn_logit = max(
                self._last_max_attn_logit,
                float(scores.detach().abs().max()),
            )

        self._last_max_attn_logit = 0.0

        cr = self.compression_rate
        n_blocks = t // cr if cr > 0 else 0

        # --- Branch 1: compressed (global coarse) ---------------------
        if n_blocks > 0:
            k_compressed = _block_compress(k_h, block_size=cr)
            v_compressed = _block_compress(v_h, block_size=cr)
            # Block i is "visible" to query t iff (i + 1) * cr - 1 <= t.
            q_pos = torch.arange(t, device=device).unsqueeze(-1)
            block_end_pos = (
                torch.arange(n_blocks, device=device).unsqueeze(0) + 1
            ) * cr - 1
            comp_mask = block_end_pos <= q_pos  # (T, n_blocks)
            # Rows where no block is yet visible (early positions) would
            # softmax over an all-masked set -> NaN forward AND backward.
            # Substitute an all-true mask for those rows, run the softmax
            # safely, then zero the output. Sliding-window branch covers
            # those positions.
            any_block = comp_mask.any(dim=-1, keepdim=True)  # (T, 1)
            safe_mask = torch.where(any_block, comp_mask, torch.ones_like(comp_mask))
            comp_out = scaled_dot_product(
                q, k_compressed, v_compressed,
                mask=safe_mask, sink_logit=self.sink_logit,
                score_observer=_track_logit,
                logit_softcap=cfg.logit_softcap,
            )
            comp_out = torch.where(
                any_block.view(1, 1, t, 1), comp_out, torch.zeros_like(comp_out),
            )
        else:
            comp_out = torch.zeros_like(q)

        # --- Branch 2: selective (top-k blocks) -----------------------
        if n_blocks > 0 and self.selection_k > 0:
            # Reuse compressed-branch keys to rank blocks per (B, H, T).
            kc_for_score = _block_compress(k_h, block_size=cr)
            scores = torch.einsum(
                "bhqd,bhnd->bhqn", q, kc_for_score,
            ) / math.sqrt(cfg.head_dim)
            q_pos_b = torch.arange(t, device=device).view(1, 1, t, 1)
            block_end_b = (
                torch.arange(n_blocks, device=device).view(1, 1, 1, n_blocks) + 1
            ) * cr - 1
            valid = block_end_b <= q_pos_b
            scores = scores.masked_fill(~valid, float("-inf"))
            k_eff = min(self.selection_k, n_blocks)
            top_block_idx = scores.topk(k_eff, dim=-1, largest=True).indices
            sel_out = self._gather_selective(
                q, k_h, v_h, top_block_idx, cr, cfg.head_dim, cfg.logit_softcap,
            )
        else:
            sel_out = torch.zeros_like(q)

        # --- Branch 3: sliding-window local ---------------------------
        sw_mask = (
            sliding_window_mask(t, cfg.sliding_window_size, device=device)
            & causal_mask(t, t, device=device)
        )
        sw_out = scaled_dot_product(
            q, k_h, v_h, mask=sw_mask, sink_logit=self.sink_logit,
            score_observer=_track_logit,
            logit_softcap=cfg.logit_softcap,
        )

        # --- Mix branches via learned gate ----------------------------
        # Gate is computed per-token-per-head from the post-norm query.
        gates = torch.sigmoid(self.branch_gate(q))  # (B, H, T, 3)
        gates = gates / gates.sum(dim=-1, keepdim=True).clamp(min=1.0e-6)
        g_comp = gates[..., 0:1]
        g_sel = gates[..., 1:2]
        g_sw = gates[..., 2:3]
        mixed = g_comp * comp_out + g_sel * sel_out + g_sw * sw_out
        return self.output(mixed)

    @staticmethod
    def _gather_selective(
        q: Tensor,
        k_h: Tensor,
        v_h: Tensor,
        top_block_idx: Tensor,
        block_size: int,
        head_dim: int,
        logit_softcap: float | None,
    ) -> Tensor:
        """Per-(B, H, T) attend over the union of selected blocks' tokens.

        Naive python-loop reference impl. Fast enough for unit tests +
        small-scale forward; production NSA cache + fused selective gather
        lands in v0.0.1 alongside the KV cache layer.
        """
        b, n_heads, t, _ = q.shape
        device = q.device
        out = torch.zeros_like(q)
        for bi in range(b):
            for hi in range(n_heads):
                for tq in range(t):
                    blocks = top_block_idx[bi, hi, tq].tolist()
                    token_idxs: list[int] = []
                    for blk in blocks:
                        start = blk * block_size
                        end = min(start + block_size, tq + 1)
                        token_idxs.extend(range(start, end))
                    if not token_idxs:
                        continue
                    idx_t = torch.tensor(token_idxs, device=device, dtype=torch.long)
                    k_sel = k_h[bi, hi, idx_t]
                    v_sel = v_h[bi, hi, idx_t]
                    q_one = q[bi, hi, tq].unsqueeze(0)
                    s = (q_one @ k_sel.transpose(-1, -2)) / math.sqrt(head_dim)
                    if logit_softcap is not None:
                        s = logit_softcap * torch.tanh(s / logit_softcap)
                    probs = torch.softmax(s, dim=-1)
                    out[bi, hi, tq] = (probs @ v_sel).squeeze(0)
        return out
