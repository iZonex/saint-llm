"""Fp4Linear tests: shape, numerics, gradient (STE), conversion, validation."""

from __future__ import annotations

import pytest
import torch
from saint_llm_kernels import Fp4Linear
from saint_llm_kernels.quant import fake_quant_fp4_mx
from torch import nn


def test_forward_shape_matches_linear() -> None:
    layer = Fp4Linear(64, 128, bias=True, block_size=32)
    x = torch.randn(2, 16, 64)
    y = layer(x)
    assert y.shape == (2, 16, 128)


def test_no_bias_param_when_disabled() -> None:
    layer = Fp4Linear(32, 16, bias=False, block_size=32)
    assert layer.bias is None
    x = torch.randn(4, 32)
    y = layer(x)
    assert y.shape == (4, 16)


def test_in_features_must_be_multiple_of_block_size() -> None:
    with pytest.raises(ValueError, match="multiple of block_size"):
        Fp4Linear(33, 16, block_size=32)


def test_block_size_16_works() -> None:
    """NVFP4 block size = 16 is a valid alternative to MXFP4 = 32."""
    layer = Fp4Linear(48, 16, block_size=16)
    x = torch.randn(2, 48)
    y = layer(x)
    assert y.shape == (2, 16)


def test_close_to_bf16_reference_within_fp4_budget() -> None:
    """Output should be close to BF16 reference, within FP4 quantization noise."""
    torch.manual_seed(0)
    ref = nn.Linear(64, 128)
    fp4 = Fp4Linear.from_linear(ref, block_size=32)

    x = torch.randn(4, 32, 64)
    y_ref = ref(x)
    y_fp4 = fp4(x)

    rel = (y_fp4 - y_ref).abs() / y_ref.abs().clamp(min=1.0e-3)
    # FP4 4-bit precision + per-block-32 scale → median ~25-35%.
    assert rel.median() < 0.40


def test_gradient_flows_to_weights_and_input() -> None:
    layer = Fp4Linear(32, 64, block_size=32)
    x = torch.randn(4, 32, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    assert layer.bias is not None and layer.bias.grad is not None
    assert torch.isfinite(layer.bias.grad).all()


def test_straight_through_grad_matches_hand_rolled() -> None:
    torch.manual_seed(0)
    ref = nn.Linear(32, 64, bias=False)
    fp4 = Fp4Linear.from_linear(ref, block_size=32)

    x = torch.randn(8, 32, requires_grad=True)
    y = fp4(x)
    y.sum().backward()
    grad_x_module = x.grad.clone()

    x.grad = None
    fp4.weight.grad = None
    x2 = x.detach().clone().requires_grad_(True)
    w_q = fake_quant_fp4_mx(fp4.weight, block_size=32, axis=-1)
    x_q = fake_quant_fp4_mx(x2, block_size=32, axis=-1)
    y2 = torch.nn.functional.linear(x_q, w_q)
    y2.sum().backward()

    assert torch.allclose(x2.grad, grad_x_module, atol=1.0e-6)


def test_from_linear_copies_weights_and_bias() -> None:
    torch.manual_seed(0)
    ref = nn.Linear(32, 16, bias=True)
    fp4 = Fp4Linear.from_linear(ref, block_size=32)
    assert torch.equal(fp4.weight, ref.weight)
    assert fp4.bias is not None and ref.bias is not None
    assert torch.equal(fp4.bias, ref.bias)


def test_from_linear_propagates_block_size() -> None:
    ref = nn.Linear(32, 16)
    fp4 = Fp4Linear.from_linear(ref, block_size=16)
    assert fp4.block_size == 16


def test_from_linear_rejects_incompatible_in_features() -> None:
    ref = nn.Linear(33, 16)
    with pytest.raises(ValueError, match="multiple of block_size"):
        Fp4Linear.from_linear(ref, block_size=32)


def test_dtype_preserved_through_forward() -> None:
    layer = Fp4Linear(32, 16, block_size=32, dtype=torch.bfloat16)
    x = torch.randn(2, 32, dtype=torch.bfloat16)
    y = layer(x)
    assert y.dtype == torch.bfloat16


def test_extra_repr_includes_block_size() -> None:
    layer = Fp4Linear(32, 16, block_size=32)
    repr_str = layer.extra_repr()
    assert "in_features=32" in repr_str
    assert "out_features=16" in repr_str
    assert "block_size=32" in repr_str
