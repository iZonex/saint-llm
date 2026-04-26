"""u-μP (unit-scaled μP) initialization for SaintLLM (ADR-0011 / OPT-02).

Applies width-aware initialization variances and per-parameter LR
scaling so hyperparameters tune at one model width transfer cleanly
to other widths. Replaces vanilla μP whose embedding-tying interaction
is fragile.

Per-parameter rules (ADR-0011 §Decision):

* **Embedding** — ``Normal(0, sigma_e=1.0)``. Unit-normalized.
* **Tied lm_head** — shares ``embed.weight`` (no separate init).
* **Standard hidden Linear** — ``Normal(0, 1/sqrt(fan_in))``.
* **Residual-feeder Linear** (output of attention or MoE block,
  feeding the residual sum) — ``Normal(0, 1/sqrt(fan_in * n_layers))``
  so the depth-summed residual stays unit-scale at init.
* **mHC dynamic-parameterization weights** (``w_pre``, ``w_res``,
  ``w_post``) — stay at zero by design (uniform-start gating).
* **Frozen modules** — leave existing init alone.

Per-parameter LR scaling (ADR-0011 §Decision (3)):

* ``embedding`` group — constant LR, AdamW.
* ``hidden`` group — μ-scaled LR ``base_lr * (base_width / actual_width)``,
  Muon.

The ``umup_param_groups`` helper builds ``[ {params: ..., lr: ...}, ... ]``
suitable for direct passing into the optimizer constructors.
"""

from __future__ import annotations

import math

from torch import nn

# Suffix patterns identifying residual-feeder Linear layers — outputs of
# attention or MoE blocks whose result is added directly into the residual
# stream. They get an extra 1/sqrt(n_layers) factor at init.
_RESIDUAL_FEEDER_SUFFIXES = (
    "output.final_proj.weight",  # GroupedOutputProjection final layer
    "down_proj.weight",  # SwiGLU's down projection in MoE experts
)


def _is_residual_feeder(param_name: str) -> bool:
    """Return True if the parameter feeds directly into the residual sum."""
    return any(param_name.endswith(suffix) for suffix in _RESIDUAL_FEEDER_SUFFIXES)


# mHC dynamic-parameterization weights — must remain zero at init by the
# mHC convention (uniform-start gating). The check matches any module /
# parameter whose dotted path component is one of these names — works for
# both nested (e.g. ``blocks.0.attn_mhc.w_pre``) and top-level paths.
_MHC_DYNAMIC_NAMES = frozenset({"w_pre", "w_res", "w_post"})


def _is_mhc_dynamic(name: str) -> bool:
    """Return True for an mHC dynamic-parameterization module / weight."""
    return any(part in _MHC_DYNAMIC_NAMES for part in name.split("."))


def umup_init(model: nn.Module, *, n_layers: int, embedding_sigma: float = 1.0) -> None:
    """Apply u-μP init in place to ``model``.

    Args:
        model:            the SaintLLM (or any nn.Module with similar
                          structure).
        n_layers:         depth of the transformer stack — used to scale
                          residual-feeder init by ``1/sqrt(n_layers)``.
        embedding_sigma:  std for embeddings; default 1.0 per u-μP.

    Mutates parameters in place. Frozen parameters
    (``requires_grad=False``) are skipped.
    """
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive; got {n_layers}")

    # Track tensors we've already initialized so tied parameters
    # (e.g. ``lm_head.weight is embed.weight``) don't get re-initialized
    # under their second name with a different std.
    seen_param_ids: set[int] = set()

    # Pass 1: embeddings first. If lm_head ties to the same tensor it's
    # picked up by id in pass 2 and skipped.
    for _module_name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            if not module.weight.requires_grad:
                continue
            if id(module.weight) in seen_param_ids:
                continue
            nn.init.normal_(module.weight, mean=0.0, std=embedding_sigma)
            seen_param_ids.add(id(module.weight))

    # Pass 2: Linear layers — width-aware std + residual-feeder factor.
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_full_name = f"{module_name}.weight" if module_name else "weight"
            if _is_mhc_dynamic(weight_full_name):
                continue
            if not module.weight.requires_grad:
                continue
            if id(module.weight) in seen_param_ids:
                # Tied to an already-initialized tensor (e.g. lm_head -> embed).
                continue
            fan_in = module.in_features
            std = 1.0 / math.sqrt(fan_in)
            if _is_residual_feeder(weight_full_name):
                std /= math.sqrt(n_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            seen_param_ids.add(id(module.weight))
            if module.bias is not None and module.bias.requires_grad:
                nn.init.zeros_(module.bias)


def umup_param_groups(
    model: nn.Module,
    *,
    base_lr_hidden: float,
    base_lr_embedding: float,
    base_width: int,
    actual_width: int | None = None,
) -> list[dict[str, object]]:
    """Build u-μP-scaled optimizer parameter groups.

    Returns a list of two dicts:

    * Group 0 — ``embedding`` group: ``embed.weight`` (and tied
      ``lm_head.weight`` if tied to the same tensor; tied weights
      appear once across both names but only as one parameter, so
      assignment by id deduplicates). LR = ``base_lr_embedding``.
    * Group 1 — ``hidden`` group: everything else. LR =
      ``base_lr_hidden * (base_width / actual_width)`` (μ-scaled).

    Args:
        model:              the model to scan.
        base_lr_hidden:     reference LR tuned at ``base_width``.
        base_lr_embedding:  constant LR for the embedding group.
        base_width:         hidden width at which ``base_lr_hidden``
                            was tuned.
        actual_width:       current model's hidden width. If ``None``,
                            inferred from ``model.embed.weight.shape[1]``.

    Returns:
        ``[{params: [...], lr: ..., name: "embedding"},
           {params: [...], lr: ..., name: "hidden"}]``
        ready for ``torch.optim.AdamW(group_0)`` and ``Muon(group_1)``
        respectively (or any optimizer that accepts a list of dicts).
    """
    if base_width <= 0:
        raise ValueError(f"base_width must be positive; got {base_width}")
    if actual_width is None:
        embed = getattr(model, "embed", None)
        if embed is None or not isinstance(embed, nn.Embedding):
            raise ValueError(
                "Model has no `embed: nn.Embedding`; pass actual_width explicitly.",
            )
        actual_width = embed.embedding_dim
    if actual_width <= 0:
        raise ValueError(f"actual_width must be positive; got {actual_width}")

    embed_param: nn.Parameter | None = None
    embed = getattr(model, "embed", None)
    if isinstance(embed, nn.Embedding):
        embed_param = embed.weight

    seen_ids: set[int] = set()
    embedding_params: list[nn.Parameter] = []
    hidden_params: list[nn.Parameter] = []

    for _name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        if embed_param is not None and pid == id(embed_param):
            embedding_params.append(param)
        else:
            hidden_params.append(param)

    width_ratio = base_width / actual_width
    return [
        {
            "name": "embedding",
            "params": embedding_params,
            "lr": base_lr_embedding,
        },
        {
            "name": "hidden",
            "params": hidden_params,
            "lr": base_lr_hidden * width_ratio,
        },
    ]


def split_umup_groups(
    model: nn.Module,
    *,
    base_lr_hidden: float,
    base_lr_embedding: float,
    base_width: int,
    actual_width: int | None = None,
) -> tuple[list[nn.Parameter], list[nn.Parameter], dict[str, float]]:
    """Convenience splitter returning ``(embedding_params, hidden_params, lrs)``.

    ``lrs`` is ``{"embedding": embedding_lr, "hidden": hidden_lr}``. Useful
    when the caller wants to instantiate two separate optimizers (AdamW for
    embedding group, Muon for hidden group) rather than a single optimizer
    with multiple parameter groups.
    """
    groups = umup_param_groups(
        model,
        base_lr_hidden=base_lr_hidden,
        base_lr_embedding=base_lr_embedding,
        base_width=base_width,
        actual_width=actual_width,
    )
    by_name = {g["name"]: g for g in groups}
    return (
        list(by_name["embedding"]["params"]),  # type: ignore[arg-type]
        list(by_name["hidden"]["params"]),  # type: ignore[arg-type]
        {
            "embedding": float(by_name["embedding"]["lr"]),  # type: ignore[arg-type]
            "hidden": float(by_name["hidden"]["lr"]),  # type: ignore[arg-type]
        },
    )


__all__ = [
    "split_umup_groups",
    "umup_init",
    "umup_param_groups",
]
