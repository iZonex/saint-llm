"""Real FP8 GEMM tests: capability detection on CPU; numerics on CUDA Ada+."""

from __future__ import annotations

import pytest
import torch
from saint_llm_kernels import Fp8Linear, fp8_gemm, is_fp8_gemm_supported


def test_capability_check_false_on_cpu() -> None:
    assert is_fp8_gemm_supported(torch.device("cpu")) is False


def test_capability_check_no_args_when_no_cuda() -> None:
    if not torch.cuda.is_available():
        assert is_fp8_gemm_supported() is False


def test_fp8_linear_default_does_not_use_real_gemm() -> None:
    """use_real_fp8_gemm defaults to False — no behavior change for existing users."""
    layer = Fp8Linear(32, 64)
    assert layer.use_real_fp8_gemm is False


def test_fp8_linear_falls_back_on_cpu_when_real_gemm_requested() -> None:
    """Even with use_real_fp8_gemm=True, CPU input must fall back to fake_quant path."""
    layer = Fp8Linear(32, 64, use_real_fp8_gemm=True)
    x = torch.randn(2, 16, 32)  # CPU
    y = layer(x)
    assert y.shape == (2, 16, 64)
    assert torch.isfinite(y).all()


@pytest.mark.gpu
def test_fp8_gemm_matches_reference_on_cuda() -> None:
    """Real fp8 GEMM output must be close to a fp32 reference within FP8 noise."""
    if not is_fp8_gemm_supported():
        pytest.skip("FP8 GEMM not supported on this device")
    cuda = torch.device("cuda")
    torch.manual_seed(0)
    M, K, N = 64, 128, 256
    x = torch.randn(M, K, device=cuda)
    weight = torch.randn(N, K, device=cuda)

    out = fp8_gemm(x, weight)
    ref = (x @ weight.t()).to(torch.bfloat16)

    rel = (out - ref).abs() / ref.abs().clamp(min=1.0e-2)
    assert rel.median() < 0.20


@pytest.mark.gpu
def test_fp8_gemm_handles_3d_input_on_cuda() -> None:
    """(B, T, K) input should pass through (reshaped internally to (M, K))."""
    if not is_fp8_gemm_supported():
        pytest.skip("FP8 GEMM not supported on this device")
    cuda = torch.device("cuda")
    x = torch.randn(2, 16, 64, device=cuda)
    weight = torch.randn(128, 64, device=cuda)
    out = fp8_gemm(x, weight)
    assert out.shape == (2, 16, 128)
    assert torch.isfinite(out).all()


@pytest.mark.gpu
def test_fp8_gemm_gradient_flows_in_full_precision() -> None:
    """STE: backward must populate grad on x and weight, both finite."""
    if not is_fp8_gemm_supported():
        pytest.skip("FP8 GEMM not supported on this device")
    cuda = torch.device("cuda")
    x = torch.randn(32, 64, device=cuda, requires_grad=True)
    weight = torch.randn(128, 64, device=cuda, requires_grad=True)
    out = fp8_gemm(x, weight)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


@pytest.mark.gpu
def test_fp8_linear_real_gemm_close_to_fake_quant_on_cuda() -> None:
    """Real GEMM and fake-quant produce close (but not identical) outputs."""
    if not is_fp8_gemm_supported():
        pytest.skip("FP8 GEMM not supported on this device")
    cuda = torch.device("cuda")
    torch.manual_seed(1)
    layer_fake = Fp8Linear(64, 128, bias=False, device=cuda)
    layer_real = Fp8Linear(64, 128, bias=False, use_real_fp8_gemm=True, device=cuda)
    layer_real.load_state_dict(layer_fake.state_dict())

    x = torch.randn(8, 16, 64, device=cuda)
    y_fake = layer_fake(x)
    y_real = layer_real(x)

    # Different scaling layouts → different rounding; check distribution-level closeness.
    rel = (y_real.float() - y_fake.float()).abs() / y_fake.abs().clamp(min=1.0e-2)
    assert rel.median() < 0.30


@pytest.mark.gpu
def test_fp8_linear_real_gemm_qat_step() -> None:
    """End-to-end: real fp8 GEMM Linear participates in a single optimizer step."""
    if not is_fp8_gemm_supported():
        pytest.skip("FP8 GEMM not supported on this device")
    cuda = torch.device("cuda")
    torch.manual_seed(2)
    layer = Fp8Linear(32, 64, use_real_fp8_gemm=True, device=cuda)
    optim = torch.optim.AdamW(layer.parameters(), lr=1.0e-3)

    x = torch.randn(4, 32, device=cuda)
    target = torch.randn(4, 64, device=cuda)

    out = layer(x)
    loss = (out - target).pow(2).mean()
    optim.zero_grad()
    loss.backward()
    optim.step()
    assert torch.isfinite(loss).item()
    assert torch.isfinite(layer.weight).all()
