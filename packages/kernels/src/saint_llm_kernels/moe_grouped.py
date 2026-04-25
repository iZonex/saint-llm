"""Grouped-GEMM MoE expert pool.

Replaces the per-expert Python loop in ``saint_llm_core.moe.DeepSeekMoE`` with one
sort + three grouped matmuls. On CUDA Ada/Hopper the matmuls go through
``torch._grouped_mm`` (a single fused kernel per matmul); elsewhere we fall back
to a per-expert reference loop with identical numerics.

The module owns three stacked weight tensors (gate / up / down) and exposes the
SwiGLU-style forward used by ``DeepSeekMoE``::

    out[m] = sum_{k=0..top_k-1}  gate[m, k] * SwiGLU(weights[expert[m, k]], h[m])

where ``SwiGLU(W, x) = W_down @ (silu(W_gate x) * W_up x)`` with V4 clamping.

This is the local (single-device) flavour of "MegaMoE EP". Multi-device EP
(all-to-all dispatch) is the next layer up; this kernel is the building block.

QAT support: ``linear_quant`` selects fp8 / fp4 fake-quant on inputs and
stacked weights. The grouped matmul itself runs in bf16/fp32 — a real fused
fp8 grouped GEMM via ``torch._scaled_grouped_mm`` requires Hopper sm9.0+
(not Ada), so it stays a follow-up.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from saint_llm_kernels.quant import Fp8Format, fake_quant_fp4_mx, fake_quant_fp8

GroupedQuantMode = Literal["bf16", "fp8", "fp4"]


def _grouped_mm_supported(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    return torch.cuda.get_device_capability(device) >= (8, 9)


def _grouped_mm_with_offsets(
    a: Tensor,
    b: Tensor,
    offsets: Tensor,
) -> Tensor:
    """Variable-group GEMM: ``a`` shape (M_total, K), ``b`` shape (n_experts, K, N).

    ``offsets[i]`` is the cumulative end index of group ``i`` along ``a``'s rows.
    Returns ``(M_total, N)``.
    """
    return torch._grouped_mm(a, b, offsets)


def _grouped_mm_reference(
    a: Tensor,
    b: Tensor,
    offsets: Tensor,
) -> Tensor:
    """Reference loop for CPU/MPS — must produce the same shapes/values as torch._grouped_mm."""
    n_experts = b.shape[0]
    out_chunks = []
    start = 0
    for e in range(n_experts):
        end = int(offsets[e].item())
        if end > start:
            out_chunks.append(a[start:end] @ b[e])
        else:
            out_chunks.append(a.new_zeros(0, b.shape[2]))
        start = end
    return torch.cat(out_chunks, dim=0)


def grouped_mm(a: Tensor, b: Tensor, offsets: Tensor) -> Tensor:
    """Dispatch to torch._grouped_mm on Ada+ CUDA, eager loop elsewhere."""
    if _grouped_mm_supported(a.device):
        return _grouped_mm_with_offsets(a, b, offsets)
    return _grouped_mm_reference(a, b, offsets)


class GroupedSwiGLUExperts(nn.Module):
    """Stacked-weight SwiGLU expert pool with sorted-token grouped GEMM forward.

    Weights live in three nn.Parameter tensors:
        gate_weight : (n_experts, intermediate_dim, hidden_dim)
        up_weight   : (n_experts, intermediate_dim, hidden_dim)
        down_weight : (n_experts, hidden_dim, intermediate_dim)

    Layout matches ``nn.Linear.weight`` per expert: ``(out_features, in_features)``.
    Forward arguments mirror what ``DeepSeekMoE`` already produces.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        n_experts: int,
        *,
        clamp_linear: tuple[float, float] = (-10.0, 10.0),
        clamp_gate_max: float = 10.0,
        linear_quant: GroupedQuantMode = "bf16",
        fp4_block_size: int = 32,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.n_experts = n_experts
        self.clamp_linear = clamp_linear
        self.clamp_gate_max = clamp_gate_max
        self.linear_quant = linear_quant
        self.fp4_block_size = fp4_block_size

        factory_kwargs: dict[str, Any] = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        self.gate_weight = nn.Parameter(
            torch.empty((n_experts, intermediate_dim, hidden_dim), **factory_kwargs),
        )
        self.up_weight = nn.Parameter(
            torch.empty((n_experts, intermediate_dim, hidden_dim), **factory_kwargs),
        )
        self.down_weight = nn.Parameter(
            torch.empty((n_experts, hidden_dim, intermediate_dim), **factory_kwargs),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for w in (self.gate_weight, self.up_weight, self.down_weight):
            for e in range(self.n_experts):
                nn.init.kaiming_uniform_(w[e], a=5.0**0.5)

    def _maybe_quant_act(self, x: Tensor) -> Tensor:
        if self.linear_quant == "fp8":
            return fake_quant_fp8(x, fmt=Fp8Format.E4M3, axis=0)
        if self.linear_quant == "fp4":
            return fake_quant_fp4_mx(x, block_size=self.fp4_block_size, axis=-1)
        return x

    def _maybe_quant_weight(self, w: Tensor) -> Tensor:
        """Per-(expert, output-channel) fake quant on (E, out, in) stacked weight."""
        if self.linear_quant == "bf16":
            return w
        e, out, _ = w.shape
        flat = w.reshape(e * out, -1)
        if self.linear_quant == "fp8":
            quantized = fake_quant_fp8(flat, fmt=Fp8Format.E4M3, axis=0)
        else:  # fp4
            quantized = fake_quant_fp4_mx(flat, block_size=self.fp4_block_size, axis=-1)
        return quantized.reshape(w.shape)

    def forward(
        self,
        flat_h: Tensor,
        flat_idx: Tensor,
        flat_gate: Tensor,
    ) -> Tensor:
        """Compute the routed-expert contribution for every token.

        Args:
            flat_h:    (M, K) hidden states, M = batch * seq.
            flat_idx:  (M, top_k) selected expert IDs per token.
            flat_gate: (M, top_k) gate weights per (token, slot).

        Returns:
            (M, K) — sum over top_k of ``gate * expert_out``.
        """
        m, top_k = flat_idx.shape
        k = self.hidden_dim

        # 1) Expand to (M*top_k, K) and (M*top_k,) for expert id + gate.
        token_expanded = flat_h.repeat_interleave(top_k, dim=0)  # (M*top_k, K)
        expert_id = flat_idx.reshape(-1)
        gate = flat_gate.reshape(-1)
        # Original token index — used to scatter results back.
        token_index = (
            torch.arange(m, device=flat_h.device).unsqueeze(-1).expand(m, top_k).reshape(-1)
        )

        # 2) Sort by expert_id so each expert's slice is contiguous.
        sort_idx = torch.argsort(expert_id, stable=True)
        token_sorted = token_expanded[sort_idx]
        gate_sorted = gate[sort_idx]
        token_index_sorted = token_index[sort_idx]

        # 3) Compute per-expert end-offsets via bincount + cumsum (int32 for kernel).
        bincount = torch.bincount(expert_id, minlength=self.n_experts)
        offsets = torch.cumsum(bincount, dim=0).to(torch.int32)

        # 4) Optional fake-quant (STE on backward) + three grouped matmuls.
        # Stacked weights are (n_experts, out, in); _grouped_mm expects (n_experts, in, out).
        gate_w = self._maybe_quant_weight(self.gate_weight).transpose(1, 2).contiguous()
        up_w = self._maybe_quant_weight(self.up_weight).transpose(1, 2).contiguous()
        down_w = self._maybe_quant_weight(self.down_weight).transpose(1, 2).contiguous()

        token_sorted_q = self._maybe_quant_act(token_sorted)
        gate_out = grouped_mm(token_sorted_q, gate_w, offsets).clamp(max=self.clamp_gate_max)
        up_out = grouped_mm(token_sorted_q, up_w, offsets).clamp(
            min=self.clamp_linear[0], max=self.clamp_linear[1],
        )
        hidden = F.silu(gate_out) * up_out  # (M*top_k, intermediate)
        hidden_q = self._maybe_quant_act(hidden)
        expert_out = grouped_mm(hidden_q, down_w, offsets)  # (M*top_k, K)

        # 5) Multiply by gate weights and scatter-add back to (M, K).
        weighted = expert_out * gate_sorted.unsqueeze(-1)
        out = flat_h.new_zeros(m, k)
        out.index_add_(0, token_index_sorted, weighted)
        return out
