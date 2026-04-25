"""GroupedSwiGLUExperts tests: parity vs reference loop, shape, gradient."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from saint_llm_kernels import GroupedSwiGLUExperts, grouped_mm
from saint_llm_kernels.moe_grouped import _grouped_mm_reference


def _reference_swiglu(
    flat_h: torch.Tensor,
    flat_idx: torch.Tensor,
    flat_gate: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    clamp_linear: tuple[float, float],
    clamp_gate_max: float,
) -> torch.Tensor:
    """Per-token Python loop matching DeepSeekMoE.forward semantics."""
    n_experts = gate_w.shape[0]
    out = torch.zeros_like(flat_h)
    for e_id in range(n_experts):
        mask = (flat_idx == e_id)
        if not mask.any():
            continue
        token_pos, slot_pos = mask.nonzero(as_tuple=True)
        toks = flat_h[token_pos]
        weights = flat_gate[token_pos, slot_pos].unsqueeze(-1)
        gate = (toks @ gate_w[e_id].t()).clamp(max=clamp_gate_max)
        up = (toks @ up_w[e_id].t()).clamp(min=clamp_linear[0], max=clamp_linear[1])
        expert_out = (F.silu(gate) * up) @ down_w[e_id].t() * weights
        out.index_add_(0, token_pos, expert_out)
    return out


def test_grouped_mm_reference_matches_per_expert_loop() -> None:
    """The reference fallback must equal an explicit per-group matmul."""
    torch.manual_seed(0)
    n_experts, K, N = 4, 32, 64
    a = torch.randn(20, K)
    b = torch.randn(n_experts, K, N)
    offsets = torch.tensor([5, 10, 15, 20], dtype=torch.int32)
    out = _grouped_mm_reference(a, b, offsets)
    expected = torch.cat([a[i * 5:(i + 1) * 5] @ b[i] for i in range(n_experts)], dim=0)
    assert torch.allclose(out, expected)


def test_grouped_mm_reference_handles_empty_groups() -> None:
    """A group with zero rows must contribute nothing to the output."""
    n_experts, K, N = 3, 8, 16
    a = torch.randn(4, K)
    b = torch.randn(n_experts, K, N)
    # group 0 = 2 rows, group 1 = 0 rows, group 2 = 2 rows
    offsets = torch.tensor([2, 2, 4], dtype=torch.int32)
    out = _grouped_mm_reference(a, b, offsets)
    assert out.shape == (4, N)
    expected_top = a[:2] @ b[0]
    expected_bot = a[2:] @ b[2]
    assert torch.allclose(out[:2], expected_top)
    assert torch.allclose(out[2:], expected_bot)


def test_grouped_mm_dispatches_to_reference_on_cpu() -> None:
    """grouped_mm on CPU = _grouped_mm_reference output."""
    a = torch.randn(8, 16)
    b = torch.randn(2, 16, 32)
    offsets = torch.tensor([4, 8], dtype=torch.int32)
    out = grouped_mm(a, b, offsets)
    expected = _grouped_mm_reference(a, b, offsets)
    assert torch.equal(out, expected)


def test_grouped_swiglu_shape_and_finite() -> None:
    torch.manual_seed(1)
    pool = GroupedSwiGLUExperts(hidden_dim=32, intermediate_dim=64, n_experts=4)
    flat_h = torch.randn(8, 32)
    flat_idx = torch.tensor([
        [0, 1], [1, 2], [0, 3], [2, 3],
        [0, 1], [1, 2], [3, 0], [2, 1],
    ], dtype=torch.long)
    flat_gate = torch.full((8, 2), 0.5)
    out = pool(flat_h, flat_idx, flat_gate)
    assert out.shape == (8, 32)
    assert torch.isfinite(out).all()


def test_grouped_swiglu_matches_per_expert_loop() -> None:
    """End-to-end numerics: GroupedSwiGLUExperts == per-token reference."""
    torch.manual_seed(2)
    pool = GroupedSwiGLUExperts(hidden_dim=16, intermediate_dim=32, n_experts=4)
    flat_h = torch.randn(12, 16)
    flat_idx = torch.randint(0, 4, (12, 2))
    flat_gate = torch.softmax(torch.randn(12, 2), dim=-1)

    actual = pool(flat_h, flat_idx, flat_gate)
    expected = _reference_swiglu(
        flat_h, flat_idx, flat_gate,
        pool.gate_weight, pool.up_weight, pool.down_weight,
        pool.clamp_linear, pool.clamp_gate_max,
    )
    assert torch.allclose(actual, expected, atol=1.0e-5)


def test_grouped_swiglu_gradient_flows() -> None:
    pool = GroupedSwiGLUExperts(hidden_dim=8, intermediate_dim=16, n_experts=4)
    flat_h = torch.randn(6, 8, requires_grad=True)
    flat_idx = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3], [0, 3], [1, 2]])
    flat_gate = torch.full((6, 2), 0.5, requires_grad=True)
    out = pool(flat_h, flat_idx, flat_gate)
    out.sum().backward()
    assert flat_h.grad is not None and torch.isfinite(flat_h.grad).all()
    assert flat_gate.grad is not None and torch.isfinite(flat_gate.grad).all()
    for p in (pool.gate_weight, pool.up_weight, pool.down_weight):
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_grouped_swiglu_fp8_qat_changes_output_but_stays_finite() -> None:
    """FP8 fake-quant injects noise; output should differ from bf16 but remain finite."""
    torch.manual_seed(3)
    common_kwargs = dict(hidden_dim=32, intermediate_dim=64, n_experts=4)
    pool_bf16 = GroupedSwiGLUExperts(**common_kwargs, linear_quant="bf16")
    pool_fp8 = GroupedSwiGLUExperts(**common_kwargs, linear_quant="fp8")
    pool_fp8.load_state_dict(pool_bf16.state_dict())

    flat_h = torch.randn(8, 32)
    flat_idx = torch.randint(0, 4, (8, 2))
    flat_gate = torch.softmax(torch.randn(8, 2), dim=-1)
    out_bf16 = pool_bf16(flat_h, flat_idx, flat_gate)
    out_fp8 = pool_fp8(flat_h, flat_idx, flat_gate)
    assert torch.isfinite(out_fp8).all()
    assert not torch.equal(out_bf16, out_fp8)


def test_grouped_swiglu_fp4_qat_changes_output_but_stays_finite() -> None:
    torch.manual_seed(4)
    common_kwargs = dict(hidden_dim=32, intermediate_dim=64, n_experts=4)
    pool_bf16 = GroupedSwiGLUExperts(**common_kwargs, linear_quant="bf16")
    pool_fp4 = GroupedSwiGLUExperts(**common_kwargs, linear_quant="fp4", fp4_block_size=32)
    pool_fp4.load_state_dict(pool_bf16.state_dict())

    flat_h = torch.randn(8, 32)
    flat_idx = torch.randint(0, 4, (8, 2))
    flat_gate = torch.softmax(torch.randn(8, 2), dim=-1)
    out_bf16 = pool_bf16(flat_h, flat_idx, flat_gate)
    out_fp4 = pool_fp4(flat_h, flat_idx, flat_gate)
    assert torch.isfinite(out_fp4).all()
    assert not torch.equal(out_bf16, out_fp4)


@pytest.mark.parametrize("mode", ["fp8", "fp4"])
def test_grouped_swiglu_quant_passes_gradient_through(mode: str) -> None:
    """STE: backward must yield finite grads on weights and inputs even under quant."""
    torch.manual_seed(5)
    pool = GroupedSwiGLUExperts(
        hidden_dim=32, intermediate_dim=64, n_experts=4,
        linear_quant=mode, fp4_block_size=32,
    )
    flat_h = torch.randn(8, 32, requires_grad=True)
    flat_idx = torch.randint(0, 4, (8, 2))
    flat_gate = torch.softmax(torch.randn(8, 2, requires_grad=True), dim=-1)
    out = pool(flat_h, flat_idx, flat_gate)
    out.sum().backward()
    assert flat_h.grad is not None and torch.isfinite(flat_h.grad).all()
    for p in (pool.gate_weight, pool.up_weight, pool.down_weight):
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_grouped_swiglu_handles_unused_expert() -> None:
    """If a routing pass picks no token for some expert, that expert contributes 0."""
    pool = GroupedSwiGLUExperts(hidden_dim=8, intermediate_dim=16, n_experts=4)
    flat_h = torch.randn(4, 8)
    # Only experts 0 and 1 used; experts 2 and 3 idle.
    flat_idx = torch.tensor([[0, 1], [0, 1], [1, 0], [0, 1]])
    flat_gate = torch.full((4, 2), 0.5)
    out = pool(flat_h, flat_idx, flat_gate)
    assert out.shape == (4, 8)
    assert torch.isfinite(out).all()


@pytest.mark.gpu
def test_grouped_mm_cuda_matches_reference() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cuda = torch.device("cuda")
    a = torch.randn(20, 32, device=cuda, dtype=torch.bfloat16)
    b = torch.randn(4, 32, 64, device=cuda, dtype=torch.bfloat16)
    offsets = torch.tensor([5, 10, 15, 20], dtype=torch.int32, device=cuda)
    actual = grouped_mm(a, b, offsets)
    expected = _grouped_mm_reference(a, b, offsets)
    assert torch.allclose(actual, expected, atol=1.0e-2)


@pytest.mark.gpu
def test_grouped_swiglu_cuda_matches_reference() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cuda = torch.device("cuda")
    pool = GroupedSwiGLUExperts(
        hidden_dim=16, intermediate_dim=32, n_experts=4,
        device=cuda, dtype=torch.float32,
    )
    flat_h = torch.randn(12, 16, device=cuda)
    flat_idx = torch.randint(0, 4, (12, 2), device=cuda)
    flat_gate = torch.softmax(torch.randn(12, 2, device=cuda), dim=-1)
    actual = pool(flat_h, flat_idx, flat_gate)
    expected = _reference_swiglu(
        flat_h, flat_idx, flat_gate,
        pool.gate_weight, pool.up_weight, pool.down_weight,
        pool.clamp_linear, pool.clamp_gate_max,
    )
    assert torch.allclose(actual, expected, atol=1.0e-4)
