"""Multimodal-aware decoding loops.

Vanilla :func:`saint_llm_inference.greedy_decode` ignores
``vision_features`` / ``audio_features``, which means it can only
decode text-only prompts. This module adds:

* :func:`multimodal_greedy_decode` — same contract as
  :func:`greedy_decode` plus optional ``vision_features`` /
  ``audio_features`` that are passed to ``model.forward`` on every
  step. Features are held constant across decode steps: they were
  spliced into the prompt embeddings during the prefill, and any
  newly generated text tokens are pure text (no extra placeholders).
* :func:`build_multimodal_prompt_ids` — assemble a ``(B, T)`` token
  tensor that interleaves text-encoded prompt + the right number of
  image-pad slots + an audio bracket if features are provided. The
  caller's job is to stack the per-image patches into a flat
  ``(N_total, vision_dim)`` tensor that lines up with the placeholder
  count.

KV-cache-aware multimodal generation is a v0.0.1 polish item — the
non-cached variant is enough to ship the captioning / VQA loop and
exercises the model's splice path end-to-end.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from saint_llm_core.model import SaintLLM
from torch import Tensor


@dataclass(frozen=True)
class MultimodalSlots:
    """Special token IDs used by the multimodal prompt builder.

    Mirrors the data-side :class:`saint_llm_data.TokenSlots`. We keep
    it independent so the inference package doesn't depend on
    saint-llm-data — production code passes slot IDs through.
    """

    image_pad: int
    audio_start: int
    audio_end: int


def build_multimodal_prompt_ids(
    text_ids_before: Sequence[int],
    text_ids_after: Sequence[int],
    slots: MultimodalSlots,
    *,
    n_image_pads: int = 0,
    n_audio_tokens: int = 0,
    device: torch.device | None = None,
) -> Tensor:
    """Build a ``(1, T)`` prompt tensor with image / audio placeholders.

    Layout: ``text_ids_before + [image_pad]*n_image_pads + audio_start
    + [audio_start]*n_audio_tokens + audio_end + text_ids_after``.

    The audio bracket follows the data-side encoding convention from
    :func:`saint_llm_data.encode_multimodal`: open-bracket, body
    (filled with ``audio_start`` as a placeholder ID), close-bracket.
    The model attaches the real audio features at every position
    inside the bracket.

    Returns a ``(1, T)`` long tensor; callers can wrap multiple rows
    with ``torch.cat`` if batching is needed.
    """
    ids: list[int] = list(text_ids_before)
    if n_image_pads > 0:
        ids.extend([slots.image_pad] * n_image_pads)
    if n_audio_tokens > 0:
        ids.append(slots.audio_start)
        ids.extend([slots.audio_start] * n_audio_tokens)
        ids.append(slots.audio_end)
    ids.extend(text_ids_after)
    return torch.tensor([ids], dtype=torch.long, device=device)


@torch.no_grad()
def multimodal_greedy_decode(
    model: SaintLLM,
    prompt_ids: Tensor,
    *,
    max_new_tokens: int,
    vision_features: Tensor | None = None,
    audio_features: Tensor | None = None,
    eos_token: int | None = None,
) -> Tensor:
    """Greedy decoding with multimodal features held constant per step.

    Args:
        model:           a SaintLLM in eval mode.
        prompt_ids:      ``(B, T)`` long tensor — already interleaved
            with placeholder slots if features are present.
        max_new_tokens:  number of tokens to append.
        vision_features: ``(N_total_image_tokens, vision_input_dim)``
            float tensor; total rows must equal the count of
            ``image_pad`` tokens in ``prompt_ids``. ``None`` skips the
            vision path.
        audio_features:  ``(N_total_audio_tokens, audio_input_dim)``
            float tensor; rows must equal the audio-bracket span. The
            model's ``_embed_inputs`` defines the slot ID range.
        eos_token:       optional early-stop sentinel.

    Returns:
        ``(B, T + max_new_tokens)`` long tensor (eos may be replicated
        at the tail, matching :func:`greedy_decode`'s contract).
    """
    if prompt_ids.dim() != 2:
        raise ValueError(
            f"prompt_ids must be 2D (B, T); got shape {tuple(prompt_ids.shape)}",
        )
    tokens = prompt_ids
    seen_eos = torch.zeros(
        prompt_ids.shape[0], dtype=torch.bool, device=prompt_ids.device,
    )

    for _ in range(max_new_tokens):
        out = model(
            tokens,
            vision_features=vision_features,
            audio_features=audio_features,
        )
        logits = out["logits"]
        assert isinstance(logits, Tensor)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=-1)
        if eos_token is not None:
            seen_eos = seen_eos | (next_token.squeeze(-1) == eos_token)
            if bool(seen_eos.all().item()):
                pad = max_new_tokens - (tokens.shape[1] - prompt_ids.shape[1])
                if pad > 0:
                    pad_tensor = next_token.new_full(
                        (prompt_ids.shape[0], pad), eos_token,
                    )
                    tokens = torch.cat([tokens, pad_tensor], dim=-1)
                break
    return tokens
