"""Muon optimizer with hybrid Newton-Schulz orthogonalization.

Reference: DeepSeek V4 paper §2.4 (Algorithm 1), Liu et al. 2025 (RMS rescale convention),
Keller Jordan 2024 (original Muon).

Algorithm 1 per V4:
    for each step t:
        for each matrix W ∈ R^(n×m):
            G_t = ∇_W L
            M_t = μ M_{t-1} + G_t                   ▷ momentum buffer
            O_t' = HybridNewtonSchulz(μ M_t + G_t)  ▷ Nesterov look-ahead + NS
            O_t  = O_t' · sqrt(max(n,m)) · γ        ▷ rescale RMS for AdamW LR reuse
            W_t  = W_{t-1} (1 - η λ) - η O_t        ▷ decoupled weight decay + update
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor

from saint_llm_optim.newton_schulz import hybrid_newton_schulz


class Muon(torch.optim.Optimizer):
    """Muon optimizer for matrix-shaped parameters.

    Use `split_for_muon_adamw` to partition a model's parameters; pass the matrix subset here
    and the rest to AdamW. Muon raises on non-2D parameters by design.

    V4-Flash defaults: lr=2.7e-4, momentum=0.95, weight_decay=0.1, rms_rescale=0.18.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 2.7e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        rms_rescale: float = 0.18,
        ns_iter_stage1: int = 8,
        ns_iter_stage2: int = 2,
        ns_coeffs_stage1: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        ns_coeffs_stage2: tuple[float, float, float] = (2.0, -1.5, 0.5),
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid lr {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum {momentum} (must be in [0, 1))")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay {weight_decay}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            rms_rescale=rms_rescale,
            ns_iter_stage1=ns_iter_stage1,
            ns_iter_stage2=ns_iter_stage2,
            ns_coeffs_stage1=ns_coeffs_stage1,
            ns_coeffs_stage2=ns_coeffs_stage2,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss: Tensor | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            mu: float = group["momentum"]
            wd: float = group["weight_decay"]
            gamma: float = group["rms_rescale"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dim() != 2:
                    raise ValueError(
                        f"Muon expects 2D matrix params; got shape {tuple(p.shape)}. "
                        "Use AdamW for 1D params (RMSNorm weights, biases, gating factors).",
                    )

                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                m_t: Tensor = state["momentum_buffer"]
                m_t.mul_(mu).add_(grad)

                # Nesterov look-ahead update: u_t = μ M_t + G_t
                update = mu * m_t + grad

                update_orth = hybrid_newton_schulz(
                    update,
                    n_iter_stage1=group["ns_iter_stage1"],
                    n_iter_stage2=group["ns_iter_stage2"],
                    coeffs_stage1=group["ns_coeffs_stage1"],
                    coeffs_stage2=group["ns_coeffs_stage2"],
                )

                n_dim, m_dim = p.shape
                update_orth.mul_(math.sqrt(max(n_dim, m_dim)) * gamma)

                # Decoupled weight decay + update.
                p.mul_(1.0 - lr * wd).add_(update_orth, alpha=-lr)

        return loss
