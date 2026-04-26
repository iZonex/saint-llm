"""Adaptive thinking effort head per ADR-0017 / RL-07.

Replaces the static V4 Non/High/Max think-mode tag (REA-01) and the
single-token ``<budget:adaptive>`` (REA-05, superseded) with a learned
5-tier effort router trained jointly with RL at v0.2:

    effort ∈ {low, medium, high, xhigh, max} → tier_id ∈ {0, 1, 2, 3, 4}

The runtime control flow:

1. User picks a tier (Anthropic Claude 4.7 convention).
2. The corresponding ``<|effort:N|>`` token is prepended to the
   assistant turn (already reserved in SAINT_V0_0_SPECIAL_TOKENS).
3. At sampling time, :class:`EffortRouter` takes the mean-pooled
   query embedding + tier id, predicts (a) CoT length budget in
   tokens, (b) termination probability per generation step.
4. The sampler gates ``eos`` emission against the budget so
   ``low`` produces short answers and ``max`` produces long CoT.

v0.0 ships the **architecture only**: the head exists, parameters are
zero-init so the base model output is identical to a model without
the head until v0.2 RL training learns budget+termination targets.

References:
    Anthropic Claude Opus 4.6 / 4.7 system cards (effort tier API)
    arXiv 2603.07915 (Ares, learned per-step adaptive routing)
    arXiv 2505.11274 (SelfBudgeter, pre-estimating budget per query)
"""

from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn


class EffortConfig(BaseModel):
    """v0.0 / v0.2 adaptive thinking head config."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    n_tiers: int = 5
    hidden: int = 64
    enabled: bool = False  # default off; v0.2 RL flips on


class EffortRouter(nn.Module):
    """Adaptive thinking effort head.

    Inputs (per :meth:`forward`):
        query_pooled  ``(B, hidden_dim)`` — mean-pooled query embeddings
            (typically the model's hidden state at the position right
            before the assistant turn begins).
        effort_tier   ``(B,)`` long — tier IDs in ``[0, cfg.n_tiers)``.

    Outputs (dict):
        ``budget``           ``(B,)`` float — predicted CoT length
            budget in tokens. Constrained ``>= 0`` via ``relu`` on the
            head output. v0.0: zero-init → 0 (sampler ignores).
        ``terminate_logit``  ``(B,)`` float — sigmoid logit for
            "should this step be the last?". v0.0: zero-init → 0.5
            probability (sampler ignores by default).

    All weights zero-init at construction so adding the head to a base
    model is a no-op until RL training fills it.
    """

    def __init__(self, hidden_dim: int, cfg: EffortConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tier_emb = nn.Embedding(cfg.n_tiers, cfg.hidden)
        self.fuse = nn.Linear(hidden_dim + cfg.hidden, cfg.hidden, bias=True)
        self.budget_head = nn.Linear(cfg.hidden, 1, bias=True)
        self.terminate_head = nn.Linear(cfg.hidden, 1, bias=True)
        # Zero-init so the head is identity-behavior until trained.
        nn.init.zeros_(self.tier_emb.weight)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        nn.init.zeros_(self.budget_head.weight)
        nn.init.zeros_(self.budget_head.bias)
        nn.init.zeros_(self.terminate_head.weight)
        nn.init.zeros_(self.terminate_head.bias)

    def forward(
        self,
        query_pooled: Tensor,
        effort_tier: Tensor,
    ) -> dict[str, Tensor]:
        if effort_tier.dim() != 1:
            raise ValueError(
                f"effort_tier must be 1D (B,); got shape {tuple(effort_tier.shape)}",
            )
        if (effort_tier < 0).any() or (effort_tier >= self.cfg.n_tiers).any():
            raise ValueError(
                f"effort_tier values must be in [0, {self.cfg.n_tiers}); "
                f"got {effort_tier.tolist()}",
            )
        tier_vec = self.tier_emb(effort_tier)  # (B, cfg.hidden)
        fused = self.fuse(torch.cat([query_pooled, tier_vec], dim=-1))
        fused = torch.relu(fused)
        budget = torch.relu(self.budget_head(fused).squeeze(-1))
        terminate_logit = self.terminate_head(fused).squeeze(-1)
        return {"budget": budget, "terminate_logit": terminate_logit}


# Effort tier name <-> id mapping. Mirrors Claude Opus 4.7 convention so
# external tooling (API clients) can pass strings without translation.
EFFORT_TIER_NAMES: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")


def effort_tier_to_id(name: str) -> int:
    """Map ``"low"`` / ``"medium"`` / ``"high"`` / ``"xhigh"`` / ``"max"`` to int."""
    name = name.lower()
    if name not in EFFORT_TIER_NAMES:
        raise ValueError(
            f"Unknown effort tier {name!r}; must be one of {EFFORT_TIER_NAMES}",
        )
    return EFFORT_TIER_NAMES.index(name)


def effort_id_to_token(tier_id: int) -> str:
    """Map tier int to the special token name reserved in TOK-04.

    The token ``<|effort:N|>`` exists in ``SAINT_V0_0_SPECIAL_TOKENS``
    for every ``N`` in ``[0, 5)``. The trainer prepends this token to
    the assistant turn so the model can condition on the chosen tier.
    """
    if not 0 <= tier_id < len(EFFORT_TIER_NAMES):
        raise ValueError(
            f"tier_id {tier_id} out of range [0, {len(EFFORT_TIER_NAMES)})",
        )
    return f"<|effort:{tier_id}|>"
