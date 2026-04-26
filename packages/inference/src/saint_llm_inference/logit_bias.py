"""Logit-bias utilities for decoders.

A *logit bias* is a per-token-id additive offset applied to the
model's logits at sampling time. Common uses:

* **Forbid a token** — bias ``-inf`` (or just ``-1e9``) on the
  forbidden ID.
* **Force a token** — bias ``+1e9`` on the desired ID, every other
  bias ``-inf``. The next sample is deterministic.
* **Soft preference** — bias ``+2.0`` on a few tokens to nudge the
  distribution toward them without overriding it.

The OpenAI / Anthropic chat APIs expose this knob in the same shape
(``{token_id: bias}``); we mirror it. :func:`apply_logit_bias` is
the pure-tensor primitive used by the streaming + non-streaming
decoders.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor


def apply_logit_bias(
    logits: Tensor,
    bias: Mapping[int, float] | None,
) -> Tensor:
    """Add per-token-id bias to ``logits``; return a new tensor.

    Args:
        logits: ``(B, V)`` or ``(..., V)`` raw logits.
        bias:   mapping ``token_id -> bias_value``. ``None`` is a
            no-op (returns ``logits`` unchanged).

    Returns:
        Logits with bias added at the indexed positions. Token IDs
        outside the vocabulary are silently dropped (defensive — some
        tokenizers expose virtual IDs the model can't actually emit).
    """
    if not bias:
        return logits
    vocab_size = logits.shape[-1]
    bias_tensor = torch.zeros(vocab_size, dtype=logits.dtype, device=logits.device)
    for tok_id, value in bias.items():
        if 0 <= int(tok_id) < vocab_size:
            bias_tensor[int(tok_id)] = float(value)
    return logits + bias_tensor


def force_token_bias(token_id: int, *, magnitude: float = 1e9) -> dict[int, float]:
    """Convenience: bias dict that forces ``token_id`` as the next pick.

    Use as ``decoder(..., logit_bias=force_token_bias(my_id))`` to
    pin one position. The remaining positions sample as usual; the
    bias only applies for one step (re-issue per step if needed).
    """
    return {int(token_id): float(magnitude)}


def forbid_tokens_bias(
    token_ids: list[int], *, magnitude: float = -1e9,
) -> dict[int, float]:
    """Convenience: bias dict that forbids every ID in ``token_ids``."""
    return {int(t): float(magnitude) for t in token_ids}
