"""Split a Saint LLM model's parameters into Muon-managed and AdamW-managed groups.

Per DeepSeek V4 §2.4 ("Basic Configurations"):

    AdamW for: embedding module, prediction head module, static biases and gating factors of
    mHC modules, and the weights of all RMSNorm modules. All other modules are updated with Muon.

In our codebase that maps to:
    AdamW: embed.weight, lm_head.weight (when not tied), every RMSNorm.weight,
           every mHC s_pre/s_res/s_post, every mHC alpha_pre/alpha_res/alpha_post,
           any 1D parameter (attention sink logits, side-channel alpha, etc.).
    Muon:  every other 2D-or-higher parameter (attention projections, MoE expert FFNs,
           mHC dynamic Ws, modality projectors, MTP combine, generation head when enabled).
"""

from __future__ import annotations

from torch import nn

ADAMW_NAME_PATTERNS: tuple[str, ...] = (
    "embed.weight",
    "lm_head.weight",
    ".s_pre",
    ".s_res",
    ".s_post",
    ".alpha_pre",
    ".alpha_res",
    ".alpha_post",
)


def split_for_muon_adamw(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return `(muon_params, adamw_params)` according to V4 §2.4.

    Frozen (requires_grad=False) parameters are excluded from both groups — they belong to no
    optimizer. The split is by parameter name + dimensionality, not by module type, so it is
    robust to refactoring inside MHC / RMSNorm / etc.
    """
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # 1D params always go to AdamW (RMSNorm weights, attention sink, side-channel alpha, etc.).
        if p.dim() < 2:
            adamw_params.append(p)
            continue

        if any(pat in name for pat in ADAMW_NAME_PATTERNS):
            adamw_params.append(p)
            continue

        muon_params.append(p)

    return muon_params, adamw_params
