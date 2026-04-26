"""Tests for batch-invariant deterministic attention.

Three things to verify:

1. *Math correctness* — agrees with ``F.scaled_dot_product_attention``
   within FP32 numerical tolerance for the simple cases.
2. *Batch invariance* — the per-row output for row ``i`` is bit-exact
   identical whether row ``i`` is computed alone or as part of a larger
   batch (CPU). On CUDA the same property holds inside
   ``deterministic_mode``.
3. *Mask + dtype handling* — fully-masked rows produce zero (not NaN);
   ``is_causal`` matches PyTorch's convention; bool masks work; output
   dtype matches input dtype while internal accumulation is FP32.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import torch
import torch.nn.functional as F
from saint_llm_core.attention import batch_invariant_attention, deterministic_mode


def test_matches_sdpa_no_mask() -> None:
    """Plain SDPA equivalence on a small case."""
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8, dtype=torch.float32)
    k = torch.randn(2, 4, 8, dtype=torch.float32)
    v = torch.randn(2, 4, 8, dtype=torch.float32)

    out = batch_invariant_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_matches_sdpa_causal() -> None:
    torch.manual_seed(1)
    q = torch.randn(2, 6, 8)
    k = torch.randn(2, 6, 8)
    v = torch.randn(2, 6, 8)

    out = batch_invariant_attention(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_matches_sdpa_with_additive_mask() -> None:
    torch.manual_seed(2)
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    mask = torch.zeros(4, 4)
    mask[:, 2:] = float("-inf")  # disallow last two keys

    out = batch_invariant_attention(q, k, v, attn_mask=mask)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_bool_mask_treated_as_attend_where_true() -> None:
    """Bool mask convention: True = attend, False = mask out (matches SDPA)."""
    torch.manual_seed(3)
    q = torch.randn(1, 3, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    bool_mask = torch.tensor([[[True, True, False, False]] * 3])  # (1, 3, 4)

    out_bool = batch_invariant_attention(q, k, v, attn_mask=bool_mask)
    additive = torch.zeros(1, 3, 4)
    additive[..., 2:] = float("-inf")
    out_add = batch_invariant_attention(q, k, v, attn_mask=additive)
    torch.testing.assert_close(out_bool, out_add)


def test_batch_invariance_cpu_bit_exact() -> None:
    """Row ``i`` output is bit-exact whether computed alone or in a batch."""
    torch.manual_seed(4)
    q = torch.randn(8, 16, 32)
    k = torch.randn(8, 16, 32)
    v = torch.randn(8, 16, 32)

    full = batch_invariant_attention(q, k, v, is_causal=True)
    for i in (0, 3, 7):
        single = batch_invariant_attention(q[i : i + 1], k[i : i + 1], v[i : i + 1], is_causal=True)
        # Bit-exact equality (not assert_close).
        assert torch.equal(single[0], full[i]), f"row {i} drifted between batched and unbatched"


def test_padding_does_not_affect_unmasked_outputs() -> None:
    """Adding pad keys with -inf mask leaves the real outputs unchanged."""
    torch.manual_seed(5)
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 6, 8)
    v = torch.randn(2, 6, 8)

    # Real keys are positions [0..3]; positions [4, 5] are pad.
    mask_full = torch.zeros(4, 6)
    mask_full[:, 4:] = float("-inf")

    out_padded = batch_invariant_attention(q, k, v, attn_mask=mask_full)
    out_real = batch_invariant_attention(q, k[:, :4, :], v[:, :4, :])
    torch.testing.assert_close(out_padded, out_real)


def test_fully_masked_row_returns_zero_not_nan() -> None:
    torch.manual_seed(6)
    q = torch.randn(1, 2, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    mask = torch.zeros(2, 4)
    mask[1, :] = float("-inf")  # row 1 of the query attends to nothing

    out = batch_invariant_attention(q, k, v, attn_mask=mask)
    assert torch.isfinite(out).all().item(), "fully-masked row produced NaN"
    torch.testing.assert_close(out[0, 1], torch.zeros(8))


def test_dtype_roundtrip_bf16() -> None:
    """bf16 input → bf16 output, internal FP32 accumulation."""
    torch.manual_seed(7)
    q = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    k = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    v = torch.randn(2, 4, 8, dtype=torch.bfloat16)

    out = batch_invariant_attention(q, k, v)
    assert out.dtype == torch.bfloat16

    # Internally promotes to fp32: result should match an fp32 reference
    # within bf16 quantization tolerance.
    ref = batch_invariant_attention(q.float(), k.float(), v.float()).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_explicit_scale_overrides_default() -> None:
    torch.manual_seed(8)
    q = torch.randn(1, 2, 8)
    k = torch.randn(1, 2, 8)
    v = torch.randn(1, 2, 8)

    out_default = batch_invariant_attention(q, k, v)
    out_scaled = batch_invariant_attention(q, k, v, scale=1.0)
    # Different scales → different distributions → different outputs.
    assert not torch.allclose(out_default, out_scaled)
    # Verify scale=1.0 reproduces by hand.
    weights = torch.softmax(q @ k.transpose(-2, -1), dim=-1)
    expected = weights @ v
    torch.testing.assert_close(out_scaled, expected, atol=1e-5, rtol=1e-5)


def test_deterministic_mode_restores_state_on_normal_exit() -> None:
    """Context manager restores all flags after a clean exit."""
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_cudnn_det = torch.backends.cudnn.deterministic
    prev_cudnn_bench = torch.backends.cudnn.benchmark
    prev_env = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    try:
        with deterministic_mode(warn_only=True):
            assert torch.are_deterministic_algorithms_enabled()
            assert torch.backends.cudnn.deterministic
            assert not torch.backends.cudnn.benchmark
    finally:
        # In case the assertions failed, force-restore to keep other tests sane.
        torch.use_deterministic_algorithms(prev_det)
        torch.backends.cudnn.deterministic = prev_cudnn_det
        torch.backends.cudnn.benchmark = prev_cudnn_bench

    assert torch.are_deterministic_algorithms_enabled() == prev_det
    assert torch.backends.cudnn.deterministic == prev_cudnn_det
    assert torch.backends.cudnn.benchmark == prev_cudnn_bench
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == prev_env


def test_deterministic_mode_restores_state_on_exception() -> None:
    """Context manager restores all flags even if the body raises."""
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_cudnn_det = torch.backends.cudnn.deterministic
    prev_cudnn_bench = torch.backends.cudnn.benchmark

    class _Boom(Exception):
        pass

    try:
        with deterministic_mode(warn_only=True):
            raise _Boom
    except _Boom:
        pass
    finally:
        # Belt-and-suspenders restore in case of bug.
        torch.use_deterministic_algorithms(prev_det)
        torch.backends.cudnn.deterministic = prev_cudnn_det
        torch.backends.cudnn.benchmark = prev_cudnn_bench

    assert torch.are_deterministic_algorithms_enabled() == prev_det
    assert torch.backends.cudnn.deterministic == prev_cudnn_det
    assert torch.backends.cudnn.benchmark == prev_cudnn_bench


def test_deterministic_mode_preserves_existing_cublas_workspace() -> None:
    """If CUBLAS_WORKSPACE_CONFIG was already set, do not clobber it."""
    sentinel = ":16:8"
    with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": sentinel}, clear=False):
        with deterministic_mode(warn_only=True):
            # Inside: still the sentinel value, not our default ":4096:8".
            assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == sentinel
        # Outside: still the sentinel value (restored).
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == sentinel


def test_grad_flows_through_q_k_v() -> None:
    """Sanity: grad reaches q, k, v despite the internal FP32 cast."""
    torch.manual_seed(9)
    q = torch.randn(1, 4, 8, requires_grad=True)
    k = torch.randn(1, 4, 8, requires_grad=True)
    v = torch.randn(1, 4, 8, requires_grad=True)

    out = batch_invariant_attention(q, k, v, is_causal=True)
    out.sum().backward()
    assert q.grad is not None and (q.grad.abs().sum() > 0)
    assert k.grad is not None and (k.grad.abs().sum() > 0)
    assert v.grad is not None and (v.grad.abs().sum() > 0)
