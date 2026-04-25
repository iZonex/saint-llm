"""Real FP8 GEMM via ``torch._scaled_mm`` with straight-through gradient.

Replaces the ``fake_quant_fp8 + F.linear`` emulation path with a fused fp8 matmul
on CUDA Ada (sm89) and Hopper (sm90+). Falls back through ``is_fp8_gemm_supported``
on devices that lack the kernel.

Scaling layout: rowwise activation (per-row of the 2D-reshaped input, equivalent to
per-token) plus colwise weight (per-output-channel). This is the only supported
combo on Ada that retains per-channel weight precision; cuBLAS's blockwise-128x128
and MX 1x32/1x16 modes are reserved for follow-ups.

Backward: STE — gradients computed in full precision against the saved un-quantized
inputs. Identical to the fake-quant path.
"""

from __future__ import annotations

import torch
from torch import Tensor

from saint_llm_kernels.quant import Fp8Format, cast_to_fp8


def is_fp8_gemm_supported(device: torch.device | None = None) -> bool:
    """True iff ``torch._scaled_mm`` is callable on ``device`` (Ada sm89 or newer)."""
    if device is None:
        if not torch.cuda.is_available():
            return False
        device = torch.device("cuda")
    if device.type != "cuda":
        return False
    capability = torch.cuda.get_device_capability(device)
    return capability >= (8, 9)


class _Fp8GemmFn(torch.autograd.Function):
    """Forward: rowwise/colwise FP8 _scaled_mm. Backward: STE."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: Tensor,
        weight: Tensor,
        weight_fmt: Fp8Format,
        activation_fmt: Fp8Format,
        out_dtype: torch.dtype,
    ) -> Tensor:
        orig_shape = x.shape
        n_out, k = weight.shape
        x_2d = x.reshape(-1, k)

        # Rowwise activation scale: (M, 1).
        x_fp8, x_scale = cast_to_fp8(x_2d, fmt=activation_fmt, axis=0)
        # Per-output-channel weight scale: (N, 1) on (N, K).
        weight_fp8, weight_scale = cast_to_fp8(weight, fmt=weight_fmt, axis=0)

        # _scaled_mm wants b in col-major (K, N). weight_fp8 is (N, K) row-major;
        # weight_fp8.t() returns the (K, N) col-major view at zero cost. The
        # corresponding scale layout is (1, N) — transpose of (N, 1).
        out_2d = torch._scaled_mm(
            x_fp8,
            weight_fp8.t(),
            scale_a=x_scale,
            scale_b=weight_scale.t(),
            out_dtype=out_dtype,
        )
        ctx.save_for_backward(x, weight)
        return out_2d.reshape(*orig_shape[:-1], n_out)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
    ) -> tuple[Tensor, Tensor, None, None, None]:
        x, weight = ctx.saved_tensors  # type: ignore[attr-defined]
        n_out = weight.shape[0]
        # Cast grads to weight dtype for stable matmul (out is bf16 typically).
        grad_2d = grad_output.reshape(-1, n_out).to(weight.dtype)
        x_2d = x.reshape(-1, weight.shape[1])

        grad_x = (grad_2d @ weight).reshape(x.shape).to(x.dtype)
        grad_w = grad_2d.t() @ x_2d
        return grad_x, grad_w, None, None, None


def fp8_gemm(
    x: Tensor,
    weight: Tensor,
    *,
    weight_fmt: Fp8Format = Fp8Format.E4M3,
    activation_fmt: Fp8Format = Fp8Format.E4M3,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Real FP8 GEMM with STE backward.

    ``x`` of any leading shape ending in ``K``; ``weight`` shape ``(N, K)``.
    Returns ``(..., N)`` in ``out_dtype``. Caller must check
    ``is_fp8_gemm_supported(x.device)`` before calling — the kernel raises on
    incompatible hardware.
    """
    return _Fp8GemmFn.apply(x, weight, weight_fmt, activation_fmt, out_dtype)  # type: ignore[no-any-return,no-untyped-call]
