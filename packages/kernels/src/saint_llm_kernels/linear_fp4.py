"""FP4 (MXFP4 E2M1) QAT-quantized Linear layer with straight-through gradient.

Symmetric to ``Fp8Linear`` but block-quantizes both weight and activation along
the last (in_features) dimension via ``fake_quant_fp4_mx``. Each block of
``block_size`` (default 32, MXFP4 spec; NVFP4=16) elements gets its own E8M0
power-of-two scale.

Constraint: ``in_features`` must be divisible by ``block_size``. Validated at
``__init__``. Use Fp8Linear or pad-to-block externally if your dims don't fit.

Storage stays float32 (FP4 emulation reasons — see ``quant.fp4`` docstring); the
forward output is bf16/fp32 per input dtype. Once Blackwell native FP4 GEMM
arrives, swap the impl under the same module API.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from saint_llm_kernels.quant import fake_quant_fp4_mx


class Fp4Linear(nn.Module):
    """``nn.Linear``-compatible layer with MXFP4 QAT forward.

    Forward equivalent to::

        x_q = fake_quant_fp4_mx(x,      block_size=block_size, axis=-1)
        w_q = fake_quant_fp4_mx(weight, block_size=block_size, axis=-1)
        y   = F.linear(x_q, w_q, bias)

    Backward: gradient flows through the fake-quant unchanged (STE).
    """

    weight: Tensor
    bias: Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        block_size: int = 32,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if in_features % block_size != 0:
            raise ValueError(
                f"in_features={in_features} must be a multiple of block_size={block_size}",
            )
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

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
        x_q = fake_quant_fp4_mx(x, block_size=self.block_size, axis=-1)
        w_q = fake_quant_fp4_mx(self.weight, block_size=self.block_size, axis=-1)
        return F.linear(x_q, w_q, self.bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        block_size: int = 32,
    ) -> Fp4Linear:
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            block_size=block_size,
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
            f"bias={self.bias is not None}, block_size={self.block_size}"
        )
