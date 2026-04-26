# ruff: noqa: RUF002
"""Muon optimizer with hybrid Newton-Schulz orthogonalization.

Reference: DeepSeek V4 paper §2.4 (Algorithm 1), Liu et al. 2025 (RMS rescale convention),
Keller Jordan 2024 (original Muon), Kimi K2 (arXiv 2507.20534) for QK-Clip.

Algorithm 1 per V4:
    for each step t:
        for each matrix W ∈ R^(n×m):
            G_t = ∇_W L
            M_t = μ M_{t-1} + G_t                   ▷ momentum buffer
            O_t' = HybridNewtonSchulz(μ M_t + G_t)  ▷ Nesterov look-ahead + NS
            O_t  = O_t' · sqrt(max(n,m)) · γ        ▷ rescale RMS for AdamW LR reuse
            W_t  = W_{t-1} (1 - η λ) - η O_t        ▷ decoupled weight decay + update

QK-Clip (MuonClip, ADR-0010 / OPT-01):
    after the standard step, for each registered attention layer:
        if max(|attention_logits|) > tau:
            scale = sqrt(tau / max_logit)
            for W in layer.qk_clip_targets():  # [W_q, W_k]
                W *= scale
            increment self.qk_clip_count[layer_id]
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from saint_llm_optim.newton_schulz import hybrid_newton_schulz


@runtime_checkable
class _QKClippable(Protocol):
    """Module contract for MuonClip QK-Clip post-step rescale.

    Attention modules implement this Protocol by exposing:

    * ``_last_max_attn_logit`` — float updated during forward pass with
      ``max(|scores|)`` from the score_observer hook in
      ``saint_llm_core.attention.common.scaled_dot_product``. Pre-mask,
      pre-softmax, post-scale.
    * ``qk_clip_targets()`` — returns the list of weight tensors
      (typically ``[W_q.weight, W_k.weight]``) to rescale when QK-Clip
      triggers.
    """

    _last_max_attn_logit: float

    def qk_clip_targets(self) -> list[Tensor]: ...


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
        *,
        qk_clip_enabled: bool = True,
        qk_clip_tau: float = 100.0,
        qk_clip_layers: Iterable[nn.Module] | None = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid lr {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum {momentum} (must be in [0, 1))")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay {weight_decay}")
        if qk_clip_tau <= 0.0:
            raise ValueError(f"qk_clip_tau must be positive; got {qk_clip_tau}")
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

        # MuonClip QK-Clip state (ADR-0010).
        self.qk_clip_enabled = qk_clip_enabled
        self.qk_clip_tau = qk_clip_tau
        self._qk_clip_layers: list[nn.Module] = list(qk_clip_layers or [])
        self._qk_clip_count: dict[int, int] = {}

    def register_qk_clip_layer(self, layer: nn.Module) -> None:
        """Add an attention layer to the QK-Clip post-step pass.

        Layer must satisfy the ``_QKClippable`` Protocol (expose
        ``_last_max_attn_logit`` and ``qk_clip_targets()``).
        """
        if not isinstance(layer, _QKClippable):
            raise TypeError(
                f"Layer {type(layer).__name__} does not satisfy _QKClippable "
                "protocol (needs _last_max_attn_logit and qk_clip_targets()).",
            )
        self._qk_clip_layers.append(layer)

    @property
    def qk_clip_count(self) -> dict[int, int]:
        """Per-layer QK-Clip trigger count keyed by ``id(layer)``.

        Stable across training within a process; resets only via
        ``self._qk_clip_count.clear()``.
        """
        return self._qk_clip_count

    def _apply_qk_clip(self) -> None:
        """Rescale W_q / W_k of layers whose recent max logit exceeded tau."""
        for layer in self._qk_clip_layers:
            max_logit = float(getattr(layer, "_last_max_attn_logit", 0.0))
            if max_logit <= self.qk_clip_tau:
                continue
            scale = (self.qk_clip_tau / max_logit) ** 0.5
            for w in layer.qk_clip_targets():
                w.data.mul_(scale)
            layer_id = id(layer)
            self._qk_clip_count[layer_id] = self._qk_clip_count.get(layer_id, 0) + 1

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

        if self.qk_clip_enabled and self._qk_clip_layers:
            self._apply_qk_clip()

        return loss
