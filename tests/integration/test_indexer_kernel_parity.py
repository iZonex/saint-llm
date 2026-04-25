"""Cross-package: kernels.lightning_indexer_topk must match core.LightningIndexer.

The kernel is a drop-in replacement for the score+mask+topk slice of
LightningIndexer.forward. Bit-identical scores; topk indices match on the
in-distribution case (positions with ≥ top_k valid blocks). Padded picks
diverge because torch.topk over -inf ties is unspecified — that is a
pre-existing core behavior, not a kernel regression.
"""

from __future__ import annotations

import pytest
import torch
from saint_llm_core.attention.csa import LightningIndexer
from saint_llm_kernels.attention import (
    lightning_indexer_scores_reference,
    lightning_indexer_topk_reference,
)


def _build_indexer(
    hidden_dim: int = 64,
    c_indexer: int = 16,
    n_heads: int = 4,
    top_k: int = 3,
    block_size_m: int = 4,
) -> LightningIndexer:
    torch.manual_seed(0)
    return LightningIndexer(hidden_dim, c_indexer, n_heads, top_k, block_size_m)


@pytest.mark.parametrize("t", [16, 32, 64])
def test_kernel_scores_match_indexer_internals(t: int) -> None:
    """Reference kernel produces the same pre-mask scores as the indexer's einsum chain."""
    block_size_m = 4
    indexer = _build_indexer(block_size_m=block_size_m)
    indexer.eval()

    h = torch.randn(2, t, 64)
    with torch.no_grad():
        k_comp = indexer.compressor(h)
        c_q = indexer.w_dq(h)
        q_ind = indexer.w_iuq(c_q).view(2, t, indexer.n_heads, indexer.c_indexer)
        w = indexer.w_w(h)

        canonical_scores = (
            torch.einsum("bthc,bnc->bthn", q_ind, k_comp).clamp(min=0.0) * w.unsqueeze(-1)
        ).sum(dim=-2)
        kernel_scores = lightning_indexer_scores_reference(q_ind, k_comp, w)

    assert torch.allclose(kernel_scores, canonical_scores, atol=1.0e-6)


def test_kernel_topk_matches_indexer_when_unpadded() -> None:
    """For positions where valid_count ≥ top_k, kernel topk == indexer topk."""
    block_size_m = 4
    top_k = 3
    n_blocks = 16
    t = n_blocks * block_size_m  # 64; positions ≥ top_k*m have ≥ top_k valid blocks
    indexer = _build_indexer(top_k=top_k, block_size_m=block_size_m)
    indexer.eval()

    h = torch.randn(1, t, 64)
    with torch.no_grad():
        k_comp_indexer, idx_indexer = indexer(h)
        c_q = indexer.w_dq(h)
        q_ind = indexer.w_iuq(c_q).view(1, t, indexer.n_heads, indexer.c_indexer)
        w = indexer.w_w(h)
        idx_kernel = lightning_indexer_topk_reference(
            q_ind, k_comp_indexer, w, top_k=top_k, block_size_m=block_size_m,
        )

    assert idx_kernel.shape == idx_indexer.shape
    # Positions with ≥ top_k valid blocks: t_pos > (top_k - 1) * block_size_m + (block_size_m - 1)
    threshold = (top_k - 1) * block_size_m + block_size_m
    for t_pos in range(threshold, t):
        # Compare as sets — topk order may differ on numerical ties; sets are stable.
        assert set(idx_kernel[0, t_pos].tolist()) == set(idx_indexer[0, t_pos].tolist())
