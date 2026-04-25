"""Loss helpers for SaintLLM training.

The trainer's ``loss_fn`` callback ultimately wants a scalar Tensor. These
helpers wrap the shifted-cross-entropy idiom and the V4 multi-token-prediction
weighting so callers don't re-derive the math.

References:
* DeepSeek-V4 §2.4 — main next-token loss + MTP heads with depth K, weight
  ``alpha_main * decay^k`` for the k-th MTP head (k>=0).
* AUGMENTATIONS MM-A-04 — depth-K MTP reserved for audio codec parallel
  prediction in v0.2.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from saint_llm_core.config import MTPConfig
from torch import Tensor, nn


def cross_entropy_main(out: dict[str, Tensor | list[Tensor]], batch: Tensor) -> Tensor:
    """Standard next-token CE on ``out["logits"]``.

    ``batch`` shape (B, T); we shift internally so position t predicts token t+1.
    """
    logits = out["logits"]
    if not isinstance(logits, Tensor):
        raise TypeError("out['logits'] must be a Tensor")
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        batch[:, 1:].reshape(-1),
    )


def cross_entropy_with_mtp(
    out: dict[str, Tensor | list[Tensor]],
    batch: Tensor,
    *,
    cfg: MTPConfig,
) -> Tensor:
    """Total loss = main CE + sum_k alpha_k * CE on MTP head k.

    ``alpha_k = cfg.loss_weight_main * (1 - cfg.loss_weight_decay)^k`` for
    k = 0..depth-1, where the k-th MTP head predicts the (k+1)-th-next token
    (so its target shift is k+2).

    When ``cfg.depth == 0`` or ``out["mtp_logits"]`` is empty, this reduces
    exactly to ``cross_entropy_main``.
    """
    main = cross_entropy_main(out, batch)
    mtp_logits_raw = out.get("mtp_logits") if isinstance(out, dict) else None
    if mtp_logits_raw is None or cfg.depth == 0:
        return main
    if not isinstance(mtp_logits_raw, list):
        raise TypeError("out['mtp_logits'] must be a list of Tensors")

    total = main
    for k, head_logits in enumerate(mtp_logits_raw):
        shift = k + 2  # main is shift=1; head k predicts token at +shift
        if batch.shape[1] <= shift:
            continue
        target = batch[:, shift:].reshape(-1)
        logits_chunk = head_logits[:, :-shift].reshape(-1, head_logits.shape[-1])
        if logits_chunk.shape[0] == 0:
            continue
        weight = cfg.loss_weight_main * ((1.0 - cfg.loss_weight_decay) ** k)
        total = total + weight * F.cross_entropy(logits_chunk, target)
    return total


def make_loss_fn(cfg: MTPConfig | None = None) -> "torch.nn.Module":  # noqa: UP037
    """Convenience factory: returns a module wrapping ``cross_entropy_with_mtp``
    (when ``cfg`` is given) or ``cross_entropy_main``.

    The result is callable as ``loss_fn(model, batch)`` so it plugs straight
    into ``Trainer(loss_fn=...)``.
    """
    class _Loss(nn.Module):
        def forward(self_inner, model: nn.Module, batch: Tensor) -> Tensor:
            out = model(batch)
            if cfg is None:
                return cross_entropy_main(out, batch)
            return cross_entropy_with_mtp(out, batch, cfg=cfg)

    return _Loss()
