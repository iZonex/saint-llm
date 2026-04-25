"""Hybrid Newton-Schulz orthogonalization for the Muon optimizer.

Reference: DeepSeek V4 paper §2.4. For a matrix M with SVD M = UΣV^T, Newton-Schulz
iterations approximate UV^T (orthogonal columns / rows depending on shape).

Hybrid schedule:
    - Stage 1 (n_iter_stage1=8 by default): coeffs (3.4445, -4.7750, 2.0315) drive
      singular values quickly toward 1 from any starting point with σ_max ≤ 1.
    - Stage 2 (n_iter_stage2=2 by default): coeffs (2.0, -1.5, 0.5) stabilize
      singular values precisely at 1.

Per-iteration update: M ← a M + b (M M^T) M + c (M M^T)^2 M.
"""

from __future__ import annotations

from torch import Tensor


def hybrid_newton_schulz(
    matrix: Tensor,
    n_iter_stage1: int = 8,
    n_iter_stage2: int = 2,
    coeffs_stage1: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    coeffs_stage2: tuple[float, float, float] = (2.0, -1.5, 0.5),
    eps: float = 1.0e-7,
) -> Tensor:
    """Approximate UV^T for the matrix `matrix` using hybrid Newton-Schulz.

    Operates on the last two dimensions; leading batch dims are supported.
    """
    orig_dtype = matrix.dtype
    m = matrix.float()

    # Frobenius normalize so σ_max ≤ 1 — required for NS convergence.
    m = m / (m.norm(dim=(-2, -1), keepdim=True) + eps)

    # NS is conventionally derived for tall matrices (n ≥ k); transpose if wide.
    transpose = m.shape[-2] < m.shape[-1]
    if transpose:
        m = m.transpose(-2, -1)

    def _step(x: Tensor, a: float, b: float, c: float) -> Tensor:
        xxt = x @ x.transpose(-2, -1)
        return a * x + b * (xxt @ x) + c * (xxt @ xxt @ x)

    a, b, c = coeffs_stage1
    for _ in range(n_iter_stage1):
        m = _step(m, a, b, c)

    a, b, c = coeffs_stage2
    for _ in range(n_iter_stage2):
        m = _step(m, a, b, c)

    if transpose:
        m = m.transpose(-2, -1)
    return m.to(orig_dtype)
