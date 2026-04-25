"""Cross-package: MoE accepts Fp8/Fp4 linear factories without breaking forward.

Proves the linear_factory plumbing in core.moe.SwiGLU/DeepSeekMoE works with
the kernel-level FP8/FP4 Linear wrappers. Existing core tests keep using the
default nn.Linear path, so they continue to validate the canonical numerics.
"""

from __future__ import annotations

import torch
from saint_llm_core.config import MoEConfig
from saint_llm_core.moe import DeepSeekMoE, SwiGLU
from saint_llm_kernels import Fp4Linear, Fp8Linear
from torch import nn


def _moe_cfg() -> MoEConfig:
    return MoEConfig(
        routed_experts=4,
        shared_experts=1,
        experts_per_token=2,
        hash_routed_layers=0,
        expert_intermediate_dim=64,
    )


def test_swiglu_default_uses_nn_linear() -> None:
    """No factory specified → plain nn.Linear (backwards-compat baseline)."""
    layer = SwiGLU(32, 64, (-10.0, 10.0), 10.0)
    assert isinstance(layer.gate_proj, nn.Linear)
    assert not isinstance(layer.gate_proj, Fp8Linear)


def test_swiglu_fp8_factory_swaps_linears() -> None:
    layer = SwiGLU(
        32, 64, (-10.0, 10.0), 10.0,
        linear_factory=lambda in_f, out_f, *, bias: Fp8Linear(in_f, out_f, bias=bias),
    )
    assert isinstance(layer.gate_proj, Fp8Linear)
    assert isinstance(layer.up_proj, Fp8Linear)
    assert isinstance(layer.down_proj, Fp8Linear)
    x = torch.randn(2, 4, 32)
    y = layer(x)
    assert y.shape == (2, 4, 32)
    assert torch.isfinite(y).all()


def test_swiglu_fp4_factory_swaps_linears() -> None:
    layer = SwiGLU(
        32, 64, (-10.0, 10.0), 10.0,
        linear_factory=lambda in_f, out_f, *, bias: Fp4Linear(in_f, out_f, bias=bias, block_size=32),
    )
    assert isinstance(layer.gate_proj, Fp4Linear)
    x = torch.randn(2, 4, 32)
    y = layer(x)
    assert y.shape == (2, 4, 32)
    assert torch.isfinite(y).all()


def test_deepseek_moe_accepts_fp8_factory() -> None:
    """Build a learned-routing MoE with Fp8 experts; forward must stay finite."""
    cfg = _moe_cfg()
    moe = DeepSeekMoE(
        hidden_dim=32,
        cfg=cfg,
        layer_idx=2,  # past hash-routed layers
        linear_factory=lambda in_f, out_f, *, bias: Fp8Linear(in_f, out_f, bias=bias),
    )
    moe.eval()
    h = torch.randn(2, 8, 32)
    token_ids = torch.zeros(2, 8, dtype=torch.long)
    with torch.no_grad():
        out = moe(h, token_ids=token_ids)
    assert out.shape == h.shape
    assert torch.isfinite(out).all()
    # Sanity: shared + routed both used Fp8.
    assert isinstance(moe.shared_experts[0].gate_proj, Fp8Linear)
    assert isinstance(moe.routed_experts[0].gate_proj, Fp8Linear)


def test_deepseek_moe_accepts_fp4_factory() -> None:
    cfg = _moe_cfg()
    moe = DeepSeekMoE(
        hidden_dim=32,
        cfg=cfg,
        layer_idx=2,
        linear_factory=lambda in_f, out_f, *, bias: Fp4Linear(in_f, out_f, bias=bias, block_size=32),
    )
    moe.eval()
    h = torch.randn(2, 8, 32)
    token_ids = torch.zeros(2, 8, dtype=torch.long)
    with torch.no_grad():
        out = moe(h, token_ids=token_ids)
    assert out.shape == h.shape
    assert torch.isfinite(out).all()


def test_deepseek_moe_default_unchanged() -> None:
    """Default factory keeps nn.Linear — guards against accidental regression."""
    cfg = _moe_cfg()
    moe = DeepSeekMoE(hidden_dim=32, cfg=cfg, layer_idx=2)
    assert isinstance(moe.shared_experts[0].gate_proj, nn.Linear)
    assert not isinstance(moe.shared_experts[0].gate_proj, Fp8Linear)
