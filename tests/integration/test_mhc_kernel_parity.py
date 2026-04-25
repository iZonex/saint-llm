"""Cross-package: kernels.mhc_carry must match core.MHC.combine bit-for-bit.

This is the contract: any kernel optimization (torch.compile, future Triton) must
keep the canonical math from core.MHC.combine intact. If this drifts, the model
silently changes meaning across kernel-vs-eager paths.
"""

from __future__ import annotations

import pytest
import torch
from saint_llm_core.config import MHCConfig
from saint_llm_core.residual.mhc import MHC
from saint_llm_kernels.mhc import mhc_carry, mhc_carry_reference


@pytest.mark.parametrize(
    "shape",
    [(1, 8, 64), (2, 16, 128), (1, 1, 32)],
)
def test_kernel_matches_core_mhc_combine(shape: tuple[int, int, int]) -> None:
    torch.manual_seed(0)
    b, t, hidden_dim = shape
    cfg = MHCConfig(
        expansion_factor=4,
        sinkhorn_iters=4,
        init_alpha=0.0,
        init_static_bias=0.0,
    )
    block = MHC(hidden_dim=hidden_dim, cfg=cfg)
    block.eval()

    x = torch.randn(b, t, cfg.expansion_factor, hidden_dim)
    inner_out = torch.randn(b, t, hidden_dim)

    with torch.no_grad():
        _, _, b_l, c_l = block.split(x)
        canonical = block.combine(x, inner_out, b_l, c_l)
        fused = mhc_carry(b_l, c_l, x, inner_out)
        ref = mhc_carry_reference(b_l, c_l, x, inner_out)

    assert torch.allclose(fused, canonical, atol=1.0e-6)
    assert torch.allclose(ref, canonical, atol=1.0e-6)
