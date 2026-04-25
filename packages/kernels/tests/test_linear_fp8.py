"""Fp8Linear tests: shape, numerics, gradient (STE), conversion."""

from __future__ import annotations

import torch
from saint_llm_kernels import Fp8Format, Fp8Linear
from saint_llm_kernels.quant import fake_quant_fp8
from torch import nn


def test_forward_shape_matches_linear() -> None:
    layer = Fp8Linear(64, 128, bias=True)
    x = torch.randn(2, 16, 64)
    y = layer(x)
    assert y.shape == (2, 16, 128)


def test_no_bias_param_when_disabled() -> None:
    layer = Fp8Linear(8, 16, bias=False)
    assert layer.bias is None
    x = torch.randn(4, 8)
    y = layer(x)
    assert y.shape == (4, 16)


def test_close_to_bf16_reference_within_fp8_budget() -> None:
    """Output should be close to BF16 reference, within FP8 quantization noise."""
    torch.manual_seed(0)
    ref = nn.Linear(64, 128)
    fp8 = Fp8Linear.from_linear(ref)

    x = torch.randn(4, 32, 64)
    y_ref = ref(x)
    y_fp8 = fp8(x)

    rel = (y_fp8 - y_ref).abs() / y_ref.abs().clamp(min=1.0e-3)
    # Per-output-channel weight scale + per-tensor activation scale → median ~10-15%.
    assert rel.median() < 0.20


def test_gradient_flows_to_weights_and_input() -> None:
    layer = Fp8Linear(16, 32)
    x = torch.randn(4, 16, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
    assert layer.bias is not None and layer.bias.grad is not None
    assert torch.isfinite(layer.bias.grad).all()


def test_straight_through_grad_matches_reference_linear() -> None:
    """Backward pass through fake_quant is identity → gradients should match
    those of a Linear with FP8-quantized weight (no quant on backward)."""
    torch.manual_seed(0)
    ref = nn.Linear(32, 64, bias=False)
    fp8 = Fp8Linear.from_linear(ref, weight_axis=0)

    x = torch.randn(8, 32, requires_grad=True)
    y = fp8(x)
    y.sum().backward()
    grad_x_fp8 = x.grad.clone()

    # Hand-rolled equivalent: fake-quant the weight (no axis on x), run F.linear,
    # backprop. Gradients must match because STE is the only thing in play.
    x.grad = None
    fp8.weight.grad = None
    x2 = x.detach().clone().requires_grad_(True)
    w_q = fake_quant_fp8(fp8.weight, fmt=Fp8Format.E4M3, axis=0)
    x_q = fake_quant_fp8(x2, fmt=Fp8Format.E4M3)
    y2 = torch.nn.functional.linear(x_q, w_q)
    y2.sum().backward()

    assert torch.allclose(x2.grad, grad_x_fp8, atol=1.0e-6)


def test_from_linear_copies_weights_and_bias() -> None:
    torch.manual_seed(0)
    ref = nn.Linear(8, 16, bias=True)
    fp8 = Fp8Linear.from_linear(ref)
    assert torch.equal(fp8.weight, ref.weight)
    assert fp8.bias is not None and ref.bias is not None
    assert torch.equal(fp8.bias, ref.bias)


def test_from_linear_no_bias() -> None:
    ref = nn.Linear(8, 16, bias=False)
    fp8 = Fp8Linear.from_linear(ref)
    assert fp8.bias is None


def test_format_choices_propagate() -> None:
    layer = Fp8Linear(
        16, 32,
        weight_fmt=Fp8Format.E5M2,
        activation_fmt=Fp8Format.E4M3,
        weight_axis=None,
        activation_axis=-1,
    )
    assert layer.weight_fmt is Fp8Format.E5M2
    assert layer.activation_fmt is Fp8Format.E4M3
    x = torch.randn(2, 4, 16)
    y = layer(x)
    assert y.shape == (2, 4, 32)
    assert torch.isfinite(y).all()


def test_dtype_preserved_through_forward() -> None:
    layer = Fp8Linear(8, 16, dtype=torch.bfloat16)
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    y = layer(x)
    assert y.dtype == torch.bfloat16


def test_extra_repr_includes_format_info() -> None:
    layer = Fp8Linear(8, 16, weight_fmt=Fp8Format.E4M3, activation_fmt=Fp8Format.E5M2)
    repr_str = layer.extra_repr()
    assert "in_features=8" in repr_str
    assert "out_features=16" in repr_str
    assert "e4m3" in repr_str
    assert "e5m2" in repr_str
