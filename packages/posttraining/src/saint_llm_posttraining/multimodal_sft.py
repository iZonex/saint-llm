"""Multimodal SFT — masked next-token CE for (text + image / audio) batches.

The CE math is identical to :func:`sft_cross_entropy`; this module
wires up the model.forward() call so vision / audio features actually
land in the embedding stream. The pipeline:

    MultimodalExample(prompt, response, image_features, audio_features)
        ↓ pack_multimodal_examples
    MultimodalPackedBatch(tokens, loss_mask, segment_ids,
                          vision_features, audio_features)
        ↓ multimodal_sft_step(model, batch)
    scalar loss

The trainer is the same — :class:`saint_llm_training.Trainer` calls
``multimodal_sft_step`` instead of plain SFT step when the dataset
yields :class:`MultimodalPackedBatch` rows.

What's *not* here:

* **Encoding raw images / audio** into feature tensors. The data side
  expects pre-computed features; we don't run SigLIP / Whisper inside
  the training step. That's a deliberate split — the encoders are
  expensive and typically run once offline, then features are cached.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from saint_llm_data.multimodal import MultimodalPackedBatch
from torch import Tensor


@dataclass(frozen=True)
class MultimodalSFTOutput:
    """Bundle returned by :func:`multimodal_sft_step`.

    Attributes:
        loss:           scalar masked next-token CE.
        n_response_tokens: integer count of supervision positions
            actually contributing to the loss (post-mask). Useful for
            logging effective batch size; zero for degenerate batches.
        logits_for_aux: model logits ``(B, T, V)`` exposed for any
            auxiliary loss the trainer wants to add (MTP, sequence
            balance, etc.).
    """

    loss: Tensor
    n_response_tokens: int
    logits_for_aux: Tensor


def multimodal_sft_loss(
    logits: Tensor,
    batch: MultimodalPackedBatch,
) -> tuple[Tensor, int]:
    """Masked next-token CE on response positions only.

    Position ``t`` predicts token ``t+1``. CE at ``t`` counts iff
    ``batch.loss_mask[:, t+1] == 1`` — i.e. the *target* is a response
    token. Returns ``(loss, n_active_positions)``. ``loss`` is zero
    (no NaN) when no positions are active.
    """
    if logits.dim() != 3:
        raise ValueError(
            f"logits must be (B, T, V); got shape {tuple(logits.shape)}",
        )
    pred_logits = logits[:, :-1]
    targets = batch.tokens[:, 1:].to(device=pred_logits.device)
    mask = batch.loss_mask[:, 1:].to(
        device=pred_logits.device, dtype=pred_logits.dtype,
    )

    ce = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).view_as(mask)
    masked = ce * mask
    denom = mask.sum().clamp(min=1.0)
    n_active = int(mask.sum().item())
    return masked.sum() / denom, n_active


def multimodal_sft_step(
    model: torch.nn.Module,
    batch: MultimodalPackedBatch,
) -> MultimodalSFTOutput:
    """Forward + masked CE for one multimodal SFT batch.

    Calls ``model(tokens, vision_features=batch.vision_features,
    audio_features=batch.audio_features)``. The model's
    ``_embed_inputs`` splices the feature rows in at placeholder
    positions; from then on it's standard next-token training.

    Returns the loss, the active-position count, and the raw logits
    so a trainer can stack on MTP / aux-balance losses without a
    second forward pass.
    """
    out = model(
        batch.tokens,
        vision_features=batch.vision_features,
        audio_features=batch.audio_features,
    )
    logits = out["logits"]
    if not isinstance(logits, Tensor):
        raise TypeError("model output 'logits' must be a Tensor")
    loss, n_active = multimodal_sft_loss(logits, batch)
    return MultimodalSFTOutput(
        loss=loss,
        n_response_tokens=n_active,
        logits_for_aux=logits,
    )
