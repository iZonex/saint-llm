"""Tests for ModelConfig + Manifold-Constrained Hyper-Connections."""

from __future__ import annotations

import torch
from saint_llm_core.config import MHCConfig, ModelConfig
from saint_llm_core.residual.mhc import MHC, sinkhorn_knopp


def test_v4_flash_config_matches_paper() -> None:
    cfg = ModelConfig.v4_flash()
    assert cfg.n_layers == 43
    assert cfg.hidden_dim == 4096
    assert cfg.csa.compression_rate == 4
    assert cfg.hca.compression_rate == 128
    assert cfg.moe.routed_experts == 256
    assert cfg.moe.experts_per_token == 6
    assert cfg.moe.affinity_fn == "sqrt_softplus"
    assert cfg.mhc.expansion_factor == 4
    assert cfg.mhc.sinkhorn_iters == 20


def test_v4_pro_config_matches_paper() -> None:
    cfg = ModelConfig.v4_pro()
    assert cfg.n_layers == 61
    assert cfg.hidden_dim == 7168
    assert cfg.csa.attention_top_k == 1024
    assert cfg.attention.query_heads == 128
    assert cfg.moe.routed_experts == 384


def test_tiny_config_self_consistent() -> None:
    cfg = ModelConfig.tiny()
    assert cfg.attention.query_heads % cfg.attention.output_proj_groups == 0


def test_sinkhorn_knopp_doubly_stochastic() -> None:
    torch.manual_seed(0)
    raw = torch.randn(4, 4)
    m = sinkhorn_knopp(raw, n_iter=20)
    row_sums = m.sum(dim=-1)
    col_sums = m.sum(dim=-2)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1.0e-4)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1.0e-4)


def test_sinkhorn_knopp_spectral_norm_le_one() -> None:
    """Doubly stochastic ⇒ spectral norm ≤ 1, the key stability property mHC depends on."""
    torch.manual_seed(0)
    for _ in range(5):
        raw = torch.randn(8, 8)
        m = sinkhorn_knopp(raw, n_iter=20)
        sigma_max = torch.linalg.svdvals(m).max()
        assert sigma_max.item() <= 1.0 + 1.0e-4


def test_mhc_split_combine_shapes() -> None:
    cfg = MHCConfig(expansion_factor=3, sinkhorn_iters=10)
    mhc = MHC(hidden_dim=16, cfg=cfg)
    x = torch.randn(2, 5, 3, 16)
    inner_in, _a, b_l, c_l = mhc.split(x)
    assert inner_in.shape == (2, 5, 16)
    assert b_l.shape == (2, 5, 3, 3)
    assert c_l.shape == (2, 5, 3)
    inner_out = inner_in * 0.1
    out = mhc.combine(x, inner_out, b_l, c_l)
    assert out.shape == x.shape


def test_mhc_b_is_doubly_stochastic_per_token() -> None:
    cfg = MHCConfig(expansion_factor=4, sinkhorn_iters=20)
    mhc = MHC(hidden_dim=8, cfg=cfg)
    x = torch.randn(1, 3, 4, 8)
    _inner_in, _a, b_l, _c = mhc.split(x)
    rows = b_l.sum(dim=-1)
    cols = b_l.sum(dim=-2)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1.0e-3)
    assert torch.allclose(cols, torch.ones_like(cols), atol=1.0e-3)


def test_mhc_expand_collapse_roundtrip_preserves_mean() -> None:
    x = torch.randn(2, 5, 16)
    expanded = MHC.expand(x, n_hc=4)
    assert expanded.shape == (2, 5, 4, 16)
    collapsed = MHC.collapse(expanded)
    assert torch.allclose(collapsed, x)


def test_mhc_zero_init_inner_zero_keeps_residual_stable() -> None:
    cfg = MHCConfig(expansion_factor=4, sinkhorn_iters=20, init_alpha=0.0, init_static_bias=0.0)
    mhc = MHC(hidden_dim=8, cfg=cfg)

    class Zero(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

    x = torch.randn(1, 3, 4, 8)
    out = mhc(x, Zero())
    # B is uniform 1/n_hc → out[t, i, :] = mean of x[t, :, :] across the n_hc axis.
    expected_mean = x.mean(dim=-2, keepdim=True).expand_as(x)
    assert torch.allclose(out, expected_mean, atol=1.0e-3)
