"""FP8-quantized Linear layer with straight-through gradient (QAT recipe).

The weight is stored at full precision (fp32/bf16, owned by the optimizer); each
forward casts both weight and activation to FP8 via ``fake_quant_fp8`` so the
backward pass receives unmodified gradients (STE) while the forward sees FP8
rounding noise. This is the standard FP8 / FP4 QAT recipe used by DeepSeek-V4,
NVFP4 and MXFP4 references.

Default scale layout:
* Weight: per-output-channel (axis=0) — matches cuBLASLt FP8 GEMM expectations
  and avoids the cliff where one heavy row dominates a per-tensor scale.
* Activation: per-tensor — simplest, matches the DeepSeek FP8 recipe. Per-token
  is exposable via ``activation_axis`` for downstream experiments.

The CUDA path goes through eager fake-quant + ``F.linear``; once a fused FP8
GEMM (torch._scaled_mm / DeepGEMM) is wired in, swap that under the same API.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from saint_llm_kernels.quant import Fp8Format, fake_quant_fp8


class Fp8Linear(nn.Module):
    """``nn.Linear``-compatible layer with FP8-quantized forward.

    Forward equivalent to::

        x_q = fake_quant_fp8(x, fmt=activation_fmt, axis=activation_axis)
        w_q = fake_quant_fp8(weight, fmt=weight_fmt, axis=weight_axis)
        y   = F.linear(x_q, w_q, bias)

    Backward: gradient flows through ``fake_quant_fp8`` unchanged (STE).
    """

    weight: Tensor
    bias: Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        weight_fmt: Fp8Format = Fp8Format.E4M3,
        activation_fmt: Fp8Format = Fp8Format.E4M3,
        weight_axis: int | None = 0,
        activation_axis: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_fmt = weight_fmt
        self.activation_fmt = activation_fmt
        self.weight_axis = weight_axis
        self.activation_axis = activation_axis

        factory_kwargs: dict[str, Any] = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs),
        )
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        x_q = fake_quant_fp8(x, fmt=self.activation_fmt, axis=self.activation_axis)
        w_q = fake_quant_fp8(self.weight, fmt=self.weight_fmt, axis=self.weight_axis)
        return F.linear(x_q, w_q, self.bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        weight_fmt: Fp8Format = Fp8Format.E4M3,
        activation_fmt: Fp8Format = Fp8Format.E4M3,
        weight_axis: int | None = 0,
        activation_axis: int | None = None,
    ) -> Fp8Linear:
        """Build an Fp8Linear with weights/bias copied from an existing Linear."""
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            weight_fmt=weight_fmt,
            activation_fmt=activation_fmt,
            weight_axis=weight_axis,
            activation_axis=activation_axis,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None and layer.bias is not None:
                layer.bias.copy_(linear.bias)
        return layer

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, weight_fmt={self.weight_fmt.value}, "
            f"activation_fmt={self.activation_fmt.value}"
        )
