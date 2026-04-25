"""Quantization kernels.

Public API:
    Fp8Format          — supported FP8 formats (E4M3, E5M2)
    cast_to_fp8        — quantize float tensor → (fp8, scale)
    dequant_from_fp8   — reconstruct float tensor from (fp8, scale)
    fake_quant_fp8     — quantize-dequantize with straight-through estimator (QAT)
    cast_to_fp4_mx     — MXFP4-style block-quantized FP4 E2M1 (emulated)
    dequant_from_fp4_mx
    fake_quant_fp4_mx  — FP4 QAT noise injection with straight-through gradient

Backend selection is automatic: Triton kernels on CUDA when available, otherwise the
torch-eager reference path. The reference and Triton paths are bit-identical for the
quantized output and scale; reconstruction differs only by the dequantization arithmetic
order, which is within rounding error.
"""

from saint_llm_kernels.quant.fp4 import (
    cast_to_fp4_mx,
    dequant_from_fp4_mx,
    fake_quant_fp4_mx,
)
from saint_llm_kernels.quant.reference import (
    Fp8Format,
    cast_to_fp8,
    dequant_from_fp8,
    fake_quant_fp8,
)

__all__ = [
    "Fp8Format",
    "cast_to_fp4_mx",
    "cast_to_fp8",
    "dequant_from_fp4_mx",
    "dequant_from_fp8",
    "fake_quant_fp4_mx",
    "fake_quant_fp8",
]
