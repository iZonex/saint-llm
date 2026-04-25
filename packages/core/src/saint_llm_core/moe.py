"""DeepSeekMoE with hash routing for first N layers, aux-loss-free balancing, modality bias.

Reference: DeepSeek V4 paper §2.1 (Mixture-of-Experts), augmentations MM-V-* (modality bias).

Affinity function: sqrt(softplus(x)) — V4 changed from V3's sigmoid.
Load balancing: aux-loss-free bias adjustment + small sequence-wise balance loss.
Hash routing: first N layers use a deterministic hash of token ID into experts.
SwiGLU clamping per training stability section: linear ∈ [-10,10], gate ≤ 10.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from saint_llm_core.attention.common import RMSNorm
from saint_llm_core.config import MoEConfig


class SwiGLU(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        clamp_linear: tuple[float, float],
        clamp_gate_max: float,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.clamp_linear = clamp_linear
        self.clamp_gate_max = clamp_gate_max

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate_proj(x).clamp(max=self.clamp_gate_max)
        up = self.up_proj(x).clamp(min=self.clamp_linear[0], max=self.clamp_linear[1])
        return self.down_proj(F.silu(gate) * up)


class HashRouter:
    """Stateless deterministic router: token ID → fixed top-k experts via splitmix hashing."""

    @staticmethod
    def route(token_ids: Tensor, n_experts: int, k: int) -> tuple[Tensor, Tensor]:
        """Returns (expert_idx, weights) with weights uniform 1/k.

        token_ids: (B, T)
        expert_idx: (B, T, k) long
        weights: (B, T, k) float
        """
        h = token_ids.long()
        choices = []
        for j in range(k):
            h = (h * 2654435769 + j * 0x9E3779B9) % (2**31)
            choices.append((h % n_experts).unsqueeze(-1))
        idx = torch.cat(choices, dim=-1)
        # Deduplicate within a token by remapping collisions to (idx + j) % n_experts.
        for j in range(1, k):
            collide = (idx[..., j].unsqueeze(-1) == idx[..., :j]).any(-1)
            idx[..., j] = torch.where(collide, (idx[..., j] + 1) % n_experts, idx[..., j])
        weights = idx.new_full(idx.shape, 1.0 / k, dtype=torch.float32)
        return idx, weights


class LearnedRouter(nn.Module):
    """Affinity-based router with aux-loss-free bias + optional modality bias."""

    def __init__(self, hidden_dim: int, n_routed: int, top_k: int, cfg: MoEConfig, enable_modality_bias: bool) -> None:
        super().__init__()
        self.n_routed = n_routed
        self.top_k = top_k
        self.cfg = cfg
        self.score_proj = nn.Linear(hidden_dim, n_routed, bias=False)
        self.register_buffer("aux_free_bias", torch.zeros(n_routed))
        if enable_modality_bias:
            # Two modalities for v0.1: text=0, vision=1. Audio shares text bias until v0.2.
            self.modality_bias = nn.Parameter(torch.zeros(2, n_routed))
        else:
            self.modality_bias = None

    def affinity(self, x: Tensor) -> Tensor:
        logits = self.score_proj(x)
        if self.cfg.affinity_fn == "sqrt_softplus":
            return torch.sqrt(F.softplus(logits) + 1.0e-12)
        return torch.sigmoid(logits)

    def forward(self, x: Tensor, is_visual: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        affinity = self.affinity(x)  # (B, T, n_routed)
        biased = affinity + self.aux_free_bias
        if self.modality_bias is not None and is_visual is not None:
            mod_idx = is_visual.long()  # (B, T)
            biased = biased + self.modality_bias[mod_idx]
        gate, idx = biased.topk(self.top_k, dim=-1)
        # Re-normalize using *unbiased* affinity at selected indices.
        gate_raw = affinity.gather(-1, idx)
        gate = gate_raw / (gate_raw.sum(-1, keepdim=True) + 1.0e-12)
        return idx, gate, affinity


class DeepSeekMoE(nn.Module):
    """One MoE FFN layer.

    Args:
        layer_idx: position in stack — first cfg.hash_routed_layers use HashRouter.
    """

    def __init__(
        self,
        hidden_dim: int,
        cfg: MoEConfig,
        layer_idx: int,
        enable_modality_router_bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cfg = cfg
        self.layer_idx = layer_idx

        self.use_hash_routing = layer_idx < cfg.hash_routed_layers

        self.norm = RMSNorm(hidden_dim)
        self.shared_experts = nn.ModuleList(
            SwiGLU(hidden_dim, cfg.expert_intermediate_dim, cfg.swiglu_clamp_linear, cfg.swiglu_clamp_gate_max)
            for _ in range(cfg.shared_experts)
        )
        self.routed_experts = nn.ModuleList(
            SwiGLU(hidden_dim, cfg.expert_intermediate_dim, cfg.swiglu_clamp_linear, cfg.swiglu_clamp_gate_max)
            for _ in range(cfg.routed_experts)
        )

        if not self.use_hash_routing:
            self.router = LearnedRouter(
                hidden_dim,
                cfg.routed_experts,
                cfg.experts_per_token,
                cfg,
                enable_modality_bias=enable_modality_router_bias,
            )
        else:
            self.router = None

    def forward(
        self,
        x: Tensor,
        token_ids: Tensor | None = None,
        is_visual: Tensor | None = None,
    ) -> Tensor:
        h = self.norm(x)
        b, t, d = h.shape

        if self.use_hash_routing:
            assert token_ids is not None, "Hash-routed layers require token_ids"
            expert_idx, gate = HashRouter.route(token_ids, self.cfg.routed_experts, self.cfg.experts_per_token)
            self._last_aux_loss = h.new_zeros(())
        else:
            assert self.router is not None
            expert_idx, gate, affinity = self.router(h, is_visual=is_visual)
            self._last_aux_loss = self._sequence_balance_loss(affinity, expert_idx)

        routed_out = torch.zeros_like(h)
        flat_h = h.reshape(b * t, d)
        flat_gate = gate.reshape(b * t, self.cfg.experts_per_token)
        flat_idx = expert_idx.reshape(b * t, self.cfg.experts_per_token)

        for e_id in range(self.cfg.routed_experts):
            mask = (flat_idx == e_id)
            if not mask.any():
                continue
            token_pos, slot_pos = mask.nonzero(as_tuple=True)
            tokens = flat_h[token_pos]
            weights = flat_gate[token_pos, slot_pos].unsqueeze(-1)
            expert_out = self.routed_experts[e_id](tokens) * weights
            routed_out.view(b * t, d).index_add_(0, token_pos, expert_out)

        shared_out = sum((expert(h) for expert in self.shared_experts), start=torch.zeros_like(h))
        return shared_out + routed_out

    @staticmethod
    def _sequence_balance_loss(affinity: Tensor, expert_idx: Tensor) -> Tensor:
        """Per-V4 §2.1: small sequence-wise balance loss to prevent extreme imbalance."""
        n_routed = affinity.shape[-1]
        # Frequency of each expert being selected per sequence.
        b, t, k = expert_idx.shape
        one_hot = F.one_hot(expert_idx, n_routed).float()
        freq = one_hot.sum(dim=(1, 2)) / (t * k)  # (B, n_routed)
        # Mean affinity per expert per sequence.
        mean_aff = affinity.mean(dim=1)  # (B, n_routed)
        return (freq * mean_aff).sum(-1).mean() * n_routed
