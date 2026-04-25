"""Top-level linear-quantization factory.

Reads ``ModelConfig.linear_quant`` and returns a ``LinearFactory`` that
``saint_llm_core.moe.SwiGLU`` / ``DeepSeekMoE`` consume to build their
projections.

Quant scope (v0.1): MoE expert SwiGLU only. Router score_proj, attention
projections, lm_head and the embedding stay full-precision. Expand the scope
by passing the same factory into more modules; the factory itself does not
change.
"""

from __future__ import annotations

from typing import Literal

from saint_llm_kernels import Fp4Linear, Fp8Linear
from torch import nn

from saint_llm_core.moe import LinearFactory

LinearQuantMode = Literal["bf16", "fp8", "fp4"]


def _bf16_factory(in_features: int, out_features: int, *, bias: bool = True) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=bias)


def make_linear_factory(
    mode: LinearQuantMode,
    *,
    fp4_block_size: int = 32,
    fp8_use_real_gemm: bool = False,
) -> LinearFactory:
    """Build a LinearFactory for the configured quant mode."""
    if mode == "bf16":
        return _bf16_factory
    if mode == "fp8":
        use_real_gemm = fp8_use_real_gemm

        def fp8_factory(
            in_features: int,
            out_features: int,
            *,
            bias: bool = True,
        ) -> nn.Module:
            return Fp8Linear(
                in_features, out_features,
                bias=bias,
                use_real_fp8_gemm=use_real_gemm,
            )

        return fp8_factory
    if mode == "fp4":
        block_size = fp4_block_size

        def fp4_factory(
            in_features: int,
            out_features: int,
            *,
            bias: bool = True,
        ) -> nn.Module:
            return Fp4Linear(in_features, out_features, bias=bias, block_size=block_size)

        return fp4_factory
    raise ValueError(f"unknown linear_quant mode: {mode!r}")
