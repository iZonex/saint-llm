"""Streaming token generation — yield one token at a time.

The vanilla decoders return the full ``(B, T + max_new)`` tensor at the
end. For interactive UIs (chat, agent traces, live captions) we want
each new token as it's produced so the caller can render it. This
module provides:

* :func:`stream_greedy_decode` — yields tokens one at a time, stopping
  on EOS or after ``max_new_tokens``.
* :func:`stream_top_p_sample` — same shape, with nucleus + temperature
  sampling.

Both functions are batch-aware: each yielded item is a ``(B, 1)``
tensor with the next token for every row in the prompt batch. The
caller decides when individual rows have stopped (e.g. after seeing
EOS for that row).

The streaming loop runs the same forward pass as the non-streaming
variants — we don't share KV cache here for simplicity. Callers that
want O(T) inference should compose with :func:`greedy_decode_cached`
manually; a streaming KV-cached variant is a v0.0.1 polish item.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import torch
from saint_llm_core.model import SaintLLM
from torch import Tensor

from saint_llm_inference.generate import _filter_top_p
from saint_llm_inference.logit_bias import apply_logit_bias
from saint_llm_inference.stop_sequences import StopSequenceMatcher


def _modality_kwargs(
    vision_features: Tensor | None,
    audio_features: Tensor | None,
) -> dict[str, Tensor]:
    """Build the kwargs dict for model.forward; omits None-valued entries."""
    out: dict[str, Tensor] = {}
    if vision_features is not None:
        out["vision_features"] = vision_features
    if audio_features is not None:
        out["audio_features"] = audio_features
    return out


@torch.no_grad()
def stream_greedy_decode(
    model: SaintLLM,
    prompt_ids: Tensor,
    *,
    max_new_tokens: int,
    eos_token: int | None = None,
    stop_sequences: Sequence[Sequence[int]] | None = None,
    vision_features: Tensor | None = None,
    audio_features: Tensor | None = None,
    logit_bias: Mapping[int, float] | None = None,
) -> Iterator[Tensor]:
    """Yield greedy next-token picks one at a time.

    Each yielded value is a ``(B, 1)`` long tensor — the next argmax
    token for every row in the prompt batch. The iterator stops when
    *every* row has emitted ``eos_token`` at least once, when every
    row has matched a ``stop_sequences`` pattern, or when
    ``max_new_tokens`` have been produced.

    Args:
        stop_sequences:  optional list of token-ID sequences that
            terminate generation when any appears at the tail of a
            row.
        vision_features: optional ``(N_image_tokens, vision_dim)``
            float tensor; passed to ``model.forward`` each step. Held
            constant across steps (placeholders only live in the
            prompt).
        audio_features:  optional ``(N_audio_tokens, audio_dim)``
            float tensor; same constant-per-step semantics.
    """
    if prompt_ids.dim() != 2:
        raise ValueError(
            f"prompt_ids must be 2D (B, T); got shape {tuple(prompt_ids.shape)}",
        )
    if max_new_tokens <= 0:
        raise ValueError(
            f"max_new_tokens must be positive; got {max_new_tokens}",
        )

    tokens = prompt_ids
    b = prompt_ids.shape[0]
    seen_eos = torch.zeros(b, dtype=torch.bool, device=prompt_ids.device)
    matchers = (
        [StopSequenceMatcher(stop_sequences) for _ in range(b)]
        if stop_sequences
        else None
    )
    forward_kwargs = _modality_kwargs(vision_features, audio_features)
    for _ in range(max_new_tokens):
        out = model(tokens, **forward_kwargs)
        logits = out["logits"]
        assert isinstance(logits, Tensor)
        last = apply_logit_bias(logits[:, -1, :], logit_bias)
        next_token = last.argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=-1)
        yield next_token
        if eos_token is not None:
            seen_eos = seen_eos | (next_token.squeeze(-1) == eos_token)
            if bool(seen_eos.all().item()):
                return
        if matchers is not None:
            ids = next_token.squeeze(-1).tolist()
            for i, m in enumerate(matchers):
                m.feed(int(ids[i]))
            if all(m.matched() for m in matchers):
                return


@torch.no_grad()
def stream_top_p_sample(
    model: SaintLLM,
    prompt_ids: Tensor,
    *,
    max_new_tokens: int,
    p: float = 0.9,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token: int | None = None,
    generator: torch.Generator | None = None,
    stop_sequences: Sequence[Sequence[int]] | None = None,
    vision_features: Tensor | None = None,
    audio_features: Tensor | None = None,
    logit_bias: Mapping[int, float] | None = None,
) -> Iterator[Tensor]:
    """Yield nucleus-sampled next tokens one at a time.

    Same shape as :func:`stream_greedy_decode` but with temperature
    + top-p (+ optional top-k cap) sampling. ``temperature == 0``
    falls through to argmax (greedy), so a temperature-zero call is
    deterministic and reproducible.

    Pass a seeded ``torch.Generator`` for reproducibility.
    ``stop_sequences`` (optional) terminates when every row matches
    a sentinel pattern. ``vision_features`` / ``audio_features`` pass
    through to ``model.forward`` each step (held constant — placeholders
    only live in the prompt).
    """
    if prompt_ids.dim() != 2:
        raise ValueError(
            f"prompt_ids must be 2D (B, T); got shape {tuple(prompt_ids.shape)}",
        )
    if max_new_tokens <= 0:
        raise ValueError(
            f"max_new_tokens must be positive; got {max_new_tokens}",
        )
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0; got {temperature}")
    if not (0.0 < p <= 1.0):
        raise ValueError(f"p must be in (0, 1]; got {p}")
    if top_k is not None and top_k <= 0:
        raise ValueError(f"top_k must be positive when set; got {top_k}")

    greedy = temperature < 1.0e-8

    tokens = prompt_ids
    b = prompt_ids.shape[0]
    seen_eos = torch.zeros(b, dtype=torch.bool, device=prompt_ids.device)
    matchers = (
        [StopSequenceMatcher(stop_sequences) for _ in range(b)]
        if stop_sequences
        else None
    )
    forward_kwargs = _modality_kwargs(vision_features, audio_features)
    for _ in range(max_new_tokens):
        out = model(tokens, **forward_kwargs)
        logits = out["logits"]
        assert isinstance(logits, Tensor)
        last = apply_logit_bias(logits[:, -1, :], logit_bias)
        if greedy:
            next_token = last.argmax(dim=-1, keepdim=True)
        else:
            scaled = last / temperature
            if top_k is not None:
                k_eff = min(top_k, scaled.shape[-1])
                cutoff = scaled.topk(k_eff, dim=-1).values[..., -1:]
                scaled = torch.where(
                    scaled < cutoff, scaled.new_full((), float("-inf")), scaled,
                )
            if p < 1.0:
                scaled = _filter_top_p(scaled, p)
            probs = torch.softmax(scaled, dim=-1)
            next_token = torch.multinomial(
                probs, num_samples=1, generator=generator,
            )
        tokens = torch.cat([tokens, next_token], dim=-1)
        yield next_token
        if eos_token is not None:
            seen_eos = seen_eos | (next_token.squeeze(-1) == eos_token)
            if bool(seen_eos.all().item()):
                return
        if matchers is not None:
            ids = next_token.squeeze(-1).tolist()
            for i, m in enumerate(matchers):
                m.feed(int(ids[i]))
            if all(m.matched() for m in matchers):
                return
