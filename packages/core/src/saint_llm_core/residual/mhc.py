"""Manifold-Constrained Hyper-Connections (mHC).

Reference: DeepSeek V4 paper §2.2; Xie et al. 2026.

Forward (per layer l):
    X_hat = RMSNorm(vec(X_l))                           ∈ R^(B, T, n_hc*d)
    A_l_raw = α_pre · (X_hat W_pre) + S_pre             ∈ R^(B, T, n_hc)
    B_l_raw = α_res · Mat(X_hat W_res) + S_res          ∈ R^(B, T, n_hc, n_hc)
    C_l_raw = α_post · (X_hat W_post) + S_post          ∈ R^(B, T, n_hc)

Constraints:
    A_l = sigmoid(A_l_raw)                              non-negative, in (0, 1)
    C_l = 2 · sigmoid(C_l_raw)                          non-negative, in (0, 2)
    B_l = SinkhornKnopp(exp(B_l_raw))                   doubly stochastic; ‖B_l‖₂ ≤ 1

Update:
    inner_in  = einsum("bth,bthd->btd", A_l, X_l)       (per token, weighted sum over n_hc rows)
    inner_out = layer(inner_in)                          F_l in R^d
    X_{l+1}   = einsum("bthk,btkd->bthd", B_l, X_l)
                + einsum("bth,btd->bthd", C_l, inner_out)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from saint_llm_core.config import MHCConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(var + self.eps)


def sinkhorn_knopp(logits: Tensor, n_iter: int) -> Tensor:
    """Project a non-negative matrix onto the Birkhoff polytope of doubly stochastic matrices.

    Operates over the last two dimensions; preserves leading batch/sequence dims.
    """
    m = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))
    for _ in range(n_iter):
        m = m / (m.sum(dim=-1, keepdim=True) + 1.0e-12)
        m = m / (m.sum(dim=-2, keepdim=True) + 1.0e-12)
    return m


class MHC(nn.Module):
    """Manifold-Constrained Hyper-Connection wrapping a single inner layer.

    The inner layer is a function (B, T, d) -> (B, T, d). The mHC maintains an
    expanded residual stream of shape (B, T, n_hc, d) across blocks.
    """

    def __init__(
        self,
        hidden_dim: int,
        cfg: MHCConfig,
        rms_norm_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_hc = cfg.expansion_factor
        self.sinkhorn_iters = cfg.sinkhorn_iters

        flat_dim = self.n_hc * hidden_dim
        self.norm = RMSNorm(flat_dim, eps=rms_norm_eps)

        self.w_pre = nn.Linear(flat_dim, self.n_hc, bias=False)
        self.w_res = nn.Linear(flat_dim, self.n_hc * self.n_hc, bias=False)
        self.w_post = nn.Linear(flat_dim, self.n_hc, bias=False)

        self.alpha_pre = nn.Parameter(torch.full((1,), cfg.init_alpha))
        self.alpha_res = nn.Parameter(torch.full((1,), cfg.init_alpha))
        self.alpha_post = nn.Parameter(torch.full((1,), cfg.init_alpha))

        self.s_pre = nn.Parameter(torch.full((self.n_hc,), cfg.init_static_bias))
        self.s_res = nn.Parameter(torch.full((self.n_hc, self.n_hc), cfg.init_static_bias))
        self.s_post = nn.Parameter(torch.full((self.n_hc,), cfg.init_static_bias))

        nn.init.zeros_(self.w_pre.weight)
        nn.init.zeros_(self.w_res.weight)
        nn.init.zeros_(self.w_post.weight)

    @staticmethod
    def expand(x: Tensor, n_hc: int) -> Tensor:
        """Replicate a (B, T, d) input into the (B, T, n_hc, d) expanded residual stream."""
        return x.unsqueeze(-2).expand(*x.shape[:-1], n_hc, x.shape[-1]).contiguous()

    @staticmethod
    def collapse(x: Tensor) -> Tensor:
        """Reduce a (B, T, n_hc, d) expanded residual stream back to (B, T, d).

        Mean across the n_hc axis. Acceptable when downstream is the LM head — see V4 §2.2.
        """
        return x.mean(dim=-2)

    def _gates(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, t, n_hc, d = x.shape
        flat = x.reshape(b, t, n_hc * d)
        normed = self.norm(flat)

        a_raw = self.alpha_pre * self.w_pre(normed) + self.s_pre
        b_raw = self.alpha_res * self.w_res(normed).reshape(b, t, n_hc, n_hc) + self.s_res
        c_raw = self.alpha_post * self.w_post(normed) + self.s_post

        a_l = torch.sigmoid(a_raw)
        c_l = 2.0 * torch.sigmoid(c_raw)
        b_l = sinkhorn_knopp(b_raw, self.sinkhorn_iters)
        return a_l, b_l, c_l

    def split(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute the inner-layer input and the carry components.

        Returns:
            inner_in: (B, T, d) — input to wrapped layer F
            a_l, b_l, c_l: gating tensors used by `combine`
        """
        a_l, b_l, c_l = self._gates(x)
        inner_in = torch.einsum("bth,bthd->btd", a_l, x)
        return inner_in, a_l, b_l, c_l

    def combine(
        self,
        x: Tensor,
        inner_out: Tensor,
        b_l: Tensor,
        c_l: Tensor,
    ) -> Tensor:
        """Apply mHC update given the inner-layer output."""
        carry = torch.einsum("bthk,btkd->bthd", b_l, x)
        write = torch.einsum("bth,btd->bthd", c_l, inner_out)
        return carry + write

    def forward(self, x: Tensor, inner: nn.Module) -> Tensor:
        inner_in, _a, b_l, c_l = self.split(x)
        inner_out = inner(inner_in)
        return self.combine(x, inner_out, b_l, c_l)
