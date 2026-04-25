"""Multi-Token Prediction (MTP) modules.

Reference: DeepSeek V4 §2.1 (inherited from V3); depth-K mode reserved for v0.2 audio-out
(Moshi-style parallel codebook prediction — see AUGMENTATIONS MM-A-04).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from saint_llm_core.attention.common import RMSNorm
from saint_llm_core.config import MTPConfig


class MTPModule(nn.Module):
    """Single MTP head: takes the LM's penultimate hidden + the embedding of the previous next-token."""

    def __init__(self, hidden_dim: int, vocab_size: int, rms_norm_eps: float = 1.0e-6) -> None:
        super().__init__()
        self.norm_h = RMSNorm(hidden_dim, eps=rms_norm_eps)
        self.norm_e = RMSNorm(hidden_dim, eps=rms_norm_eps)
        self.combine = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        # The actual prediction projection is tied to the input embeddings in the parent model;
        # here we only produce a hidden state.

    def forward(self, hidden: Tensor, prev_token_embed: Tensor) -> Tensor:
        return self.combine(torch.cat([self.norm_h(hidden), self.norm_e(prev_token_embed)], dim=-1))


class MTPStack(nn.Module):
    """K MTP modules stacked. K=1 in v0.1; K>1 reserved for v0.2 (audio codec parallel prediction)."""

    def __init__(self, hidden_dim: int, vocab_size: int, cfg: MTPConfig, rms_norm_eps: float = 1.0e-6) -> None:
        super().__init__()
        self.depth = cfg.depth
        self.modules_ = nn.ModuleList(
            MTPModule(hidden_dim, vocab_size, rms_norm_eps=rms_norm_eps) for _ in range(cfg.depth)
        )

    def forward(self, hidden: Tensor, embeddings: Tensor) -> list[Tensor]:
        """Predict the next K tokens after each position.

        hidden: (B, T, d) — final transformer hidden state.
        embeddings: (B, T, d) — embeddings of the input token sequence (already shifted as needed).
        Returns: list of length depth, each (B, T, d) — to be projected by the LM head.
        """
        outs: list[Tensor] = []
        cur_hidden = hidden
        for k, mtp in enumerate(self.modules_):
            # k-th MTP attends to embedding shifted by k+1.
            shift = k + 1
            if embeddings.shape[1] <= shift:
                shifted = torch.zeros_like(embeddings)
            else:
                shifted = torch.cat(
                    [embeddings[:, shift:], embeddings.new_zeros(embeddings.shape[0], shift, embeddings.shape[-1])],
                    dim=1,
                )
            cur_hidden = mtp(cur_hidden, shifted)
            outs.append(cur_hidden)
        return outs
