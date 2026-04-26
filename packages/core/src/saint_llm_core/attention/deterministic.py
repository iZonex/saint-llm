"""Batch-invariant deterministic attention for verifiable evaluation.

The fast attention paths in this codebase (SWA, HCA, CSA, plus the
``torch.nn.functional.scaled_dot_product_attention`` defaults) use
flash-style tiled reductions and dispatch to different kernels based on
batch shape, head dim, and hardware. Their floating-point output for the
same logical query/key/value can drift across:

* batch size — Q @ K^T may pick a different cuBLAS algorithm at B=1 vs B=64;
* sequence padding — flash splits along L into tiles whose boundaries depend
  on the padded length;
* hardware feature flags — TF32 vs FP32 accumulation, tensor cores vs
  CUDA cores, cuDNN heuristic selection.

For verifiable post-training (GRPO with rule-based rewards, replay-based
auditing, bit-exact pre-training resume) we need a slow, *batch-invariant*
attention path that gives the same output for the same per-row inputs no
matter how those rows were assembled. This module supplies it:

* :func:`batch_invariant_attention` — plain ``softmax(QK^T / √d) V``
  computed in FP32, no tiling, no fused kernels. Bit-exact on CPU
  regardless of batch shape; bit-exact on CUDA inside
  :func:`deterministic_mode` (which pins cuBLAS to a single algorithm).
* :func:`deterministic_mode` — context manager that flips PyTorch into a
  fully-deterministic configuration and restores prior state on exit.
"""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Iterator

import torch
from torch import Tensor


@contextlib.contextmanager
def deterministic_mode(*, warn_only: bool = False) -> Iterator[None]:
    """Pin PyTorch to deterministic kernel selection inside the ``with`` block.

    Sets:

    * ``torch.use_deterministic_algorithms(True, warn_only=...)``
    * ``torch.backends.cudnn.deterministic = True``
    * ``torch.backends.cudnn.benchmark = False``
    * ``CUBLAS_WORKSPACE_CONFIG=":4096:8"`` if not already set
      (cuBLAS requires this to commit to a single algorithm in
      deterministic mode; otherwise it raises at GEMM time).

    All previous values are saved on entry and restored on exit, including
    when the body raises. Note that flipping ``CUBLAS_WORKSPACE_CONFIG``
    after CUDA streams have been created has no effect for the current
    process — set it from the entrypoint when bit-exactness across batch
    shapes is required.
    """
    prev_use_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    prev_cudnn_det = torch.backends.cudnn.deterministic
    prev_cudnn_bench = torch.backends.cudnn.benchmark
    prev_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    if prev_cublas is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev_use_det, warn_only=prev_warn)
        torch.backends.cudnn.deterministic = prev_cudnn_det
        torch.backends.cudnn.benchmark = prev_cudnn_bench
        if prev_cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = prev_cublas


def batch_invariant_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    attn_mask: Tensor | None = None,
    scale: float | None = None,
    is_causal: bool = False,
) -> Tensor:
    """Reference scaled-dot-product attention in FP32 with fixed math.

    Math: ``out = softmax((Q K^T) * scale + mask) @ V``. Done as one
    ``torch.matmul`` for scores, one softmax, one ``torch.matmul`` for
    output — no flash, no tile, no fused kernel. Internal accumulation is
    FP32; the output is cast back to ``q.dtype``.

    Args:
        q: query of shape ``(..., L_q, D)``.
        k: key of shape ``(..., L_k, D)``. Leading dims must broadcast
            with ``q``.
        v: value of shape ``(..., L_k, D_v)``. Leading dims must broadcast
            with ``q``.
        attn_mask: additive mask broadcast-compatible with the score
            tensor's shape ``(..., L_q, L_k)``. Use ``-inf`` to mask out
            and ``0`` to attend; bool masks are converted with
            ``True → 0`` (attend) and ``False → -inf`` (mask).
        scale: explicit attention scale; defaults to ``1/√D``.
        is_causal: if True, additionally applies a top-right causal mask
            aligned so that ``q[..., i, :]`` cannot attend to ``k[..., j, :]``
            with ``j > i + (L_k - L_q)``. The same convention as
            ``F.scaled_dot_product_attention`` so cross-attention against
            a longer key sequence still does the right thing.

    Returns:
        Tensor of shape ``(..., L_q, D_v)`` in ``q.dtype``. Rows that are
        fully masked produce zero output (the would-be NaN from
        ``softmax([-inf, ...])`` is replaced with 0 weights).
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    out_dtype = q.dtype
    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)
    v_f32 = v.to(torch.float32)

    scores = torch.matmul(q_f32, k_f32.transpose(-2, -1)) * scale

    if is_causal:
        l_q = scores.shape[-2]
        l_k = scores.shape[-1]
        causal = torch.triu(
            torch.full((l_q, l_k), float("-inf"), dtype=torch.float32, device=scores.device),
            diagonal=l_k - l_q + 1,
        )
        scores = scores + causal

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            additive = torch.zeros_like(attn_mask, dtype=torch.float32)
            additive.masked_fill_(~attn_mask, float("-inf"))
        else:
            additive = attn_mask.to(torch.float32)
        scores = scores + additive

    weights = torch.softmax(scores, dim=-1)
    # Fully-masked rows produce all-NaN softmax (exp(-inf) / sum(0) = 0/0).
    # Replace with zero weights so the matmul produces zero output rather
    # than NaN — callers that care about "did this row attend to anything?"
    # should inspect the mask, not the output.
    weights = torch.nan_to_num(weights, nan=0.0)
    out = torch.matmul(weights, v_f32)
    return out.to(out_dtype)
