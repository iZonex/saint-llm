"""Causal-LM perplexity over a token-id batch.

Perplexity = exp(mean cross-entropy of next-token prediction). Computed in fp32
even when the model runs in bf16 to keep the exp() stable on long sequences.

Usage:
    ppl = compute_perplexity(model, token_ids, ignore_index=pad_id)

Caller is responsible for putting ``model`` in ``eval()`` mode and providing
appropriate device-placed tensors.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from saint_llm_core.model import SaintLLM
from torch import Tensor


@torch.no_grad()
def compute_perplexity(
    model: SaintLLM,
    token_ids: Tensor,
    *,
    ignore_index: int = -100,
) -> float:
    """Causal-LM perplexity on ``(B, T)`` token IDs.

    Uses next-token shift: predicts ``token_ids[:, 1:]`` from ``token_ids[:, :-1]``.
    Tokens equal to ``ignore_index`` in the labels are excluded from the average.
    Returns infinity rather than NaN when no scoring positions remain.
    """
    if token_ids.dim() != 2:
        raise ValueError(f"token_ids must be 2D (B, T); got {tuple(token_ids.shape)}")
    if token_ids.shape[1] < 2:
        raise ValueError(f"need at least 2 tokens to score; got T={token_ids.shape[1]}")

    out = model(token_ids)
    logits = out["logits"]
    assert isinstance(logits, Tensor)

    flat_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1]).to(torch.float32)
    flat_labels = token_ids[:, 1:].reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index)
    if not torch.isfinite(loss):
        return math.inf
    return math.exp(loss.item())


@torch.no_grad()
def compute_perplexity_streaming(
    model: SaintLLM,
    token_id_batches: list[Tensor],
    *,
    ignore_index: int = -100,
) -> float:
    """Perplexity averaged across multiple ``(B, T)`` batches.

    Token-weighted average — a long batch contributes proportionally more than
    a short one. Useful when an eval set doesn't fit in a single forward.
    """
    total_loss = 0.0
    total_tokens = 0
    for batch in token_id_batches:
        if batch.dim() != 2 or batch.shape[1] < 2:
            continue
        out = model(batch)
        logits = out["logits"]
        assert isinstance(logits, Tensor)
        flat_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1]).to(torch.float32)
        flat_labels = batch[:, 1:].reshape(-1)
        valid = flat_labels != ignore_index
        n = int(valid.sum().item())
        if n == 0:
            continue
        loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index)
        if not torch.isfinite(loss):
            return math.inf
        total_loss += loss.item() * n
        total_tokens += n
    if total_tokens == 0:
        return math.inf
    return math.exp(total_loss / total_tokens)
