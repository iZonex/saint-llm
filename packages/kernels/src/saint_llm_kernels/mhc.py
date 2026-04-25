"""Fused mHC carry kernel.

The mHC residual update splits into two einsums plus an add::

    carry[b,t,h,d] = sum_k b_l[b,t,h,k] * x[b,t,k,d]   # B @ X per token
    write[b,t,h,d] = c_l[b,t,h]    * inner_out[b,t,d]  # outer product per token
    out                = carry + write

Both ops are memory-bound (n_hc is small, typically 4) and benefit from fusion to
avoid materializing the intermediates. This module exposes one entry point that:

* On CUDA, dispatches through ``torch.compile`` so the inductor backend fuses the
  two einsums + the add into a single kernel.
* Elsewhere (CPU, MPS), runs the eager reference.

Both paths must produce numerically equivalent output up to floating-point
associativity. The reference is the source of truth for tests.

Future work: a hand-written Triton kernel that fuses the load of ``b_l``/``c_l``
with the inner GEMM. Reserved for when ``torch.compile`` ceases to keep up with
target SM-utilization on hpomen.
"""

from __future__ import annotations

from functools import cache

import torch
from torch import Tensor


def mhc_carry_reference(
    b_l: Tensor,
    c_l: Tensor,
    x: Tensor,
    inner_out: Tensor,
) -> Tensor:
    """Eager reference implementation — source of truth for numerics.

    Shapes::
        b_l       : (B, T, n_hc, n_hc)
        c_l       : (B, T, n_hc)
        x         : (B, T, n_hc, d)
        inner_out : (B, T, d)
        return    : (B, T, n_hc, d)
    """
    carry = torch.einsum("bthk,btkd->bthd", b_l, x)
    write = torch.einsum("bth,btd->bthd", c_l, inner_out)
    return carry + write


@cache
def _get_compiled_carry() -> object:
    """Lazily compile the carry kernel; cached for the rest of the process."""
    return torch.compile(mhc_carry_reference, dynamic=True)


def mhc_carry(
    b_l: Tensor,
    c_l: Tensor,
    x: Tensor,
    inner_out: Tensor,
) -> Tensor:
    """Dispatch: compiled fused path on CUDA, eager reference elsewhere.

    The compiled path is lazy — first call on CUDA pays the compile cost, every
    subsequent call uses the cached graph. dynamic=True allows shape changes
    across calls without recompile.
    """
    if x.is_cuda:
        return _get_compiled_carry()(b_l, c_l, x, inner_out)  # type: ignore[operator,no-any-return]
    return mhc_carry_reference(b_l, c_l, x, inner_out)
