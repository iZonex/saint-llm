"""Basic decoding loops — greedy and top-k temperature sampling.

No KV cache yet: every step recomputes the full forward over the running prompt.
That's intentional for v0.1 — gets us a working "talk to the model" surface
without committing to a cache layout. Heterogeneous KV cache (the inference
package's main remit) layers on top with the same generate() entry points.
"""

from __future__ import annotations

import torch
from saint_llm_core.model import SaintLLM
from torch import Tensor


@torch.no_grad()
def greedy_decode(
    model: SaintLLM,
    prompt_ids: Tensor,
    *,
    max_new_tokens: int,
    eos_token: int | None = None,
) -> Tensor:
    """Append ``max_new_tokens`` argmax-of-logits tokens to ``prompt_ids``.

    Args:
        model: a SaintLLM in eval mode (caller responsibility).
        prompt_ids: ``(B, T)`` long tensor.
        max_new_tokens: number of tokens to append.
        eos_token: if set, stop early when *every* sequence in the batch has
            emitted this token at least once. The returned tensor still has the
            full ``T + max_new_tokens`` width (eos is included; subsequent
            tokens past eos are arbitrary).

    Returns:
        ``(B, T + max_new_tokens)`` long tensor.
    """
    if prompt_ids.dim() != 2:
        raise ValueError(f"prompt_ids must be 2D (B, T); got shape {tuple(prompt_ids.shape)}")
    tokens = prompt_ids
    seen_eos = torch.zeros(prompt_ids.shape[0], dtype=torch.bool, device=prompt_ids.device)
    for _ in range(max_new_tokens):
        out = model(tokens)
        logits = out["logits"]
        assert isinstance(logits, Tensor)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=-1)
        if eos_token is not None:
            seen_eos = seen_eos | (next_token.squeeze(-1) == eos_token)
            if bool(seen_eos.all().item()):
                # Pad the tail with the eos token to keep output shape consistent.
                pad = max_new_tokens - (tokens.shape[1] - prompt_ids.shape[1])
                if pad > 0:
                    pad_tensor = next_token.new_full((prompt_ids.shape[0], pad), eos_token)
                    tokens = torch.cat([tokens, pad_tensor], dim=-1)
                break
    return tokens


@torch.no_grad()
def top_k_sample(
    model: SaintLLM,
    prompt_ids: Tensor,
    *,
    max_new_tokens: int,
    k: int = 50,
    temperature: float = 1.0,
    eos_token: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Top-k temperature sampling.

    For each step, restricts the categorical distribution to the ``k`` highest-
    logit tokens and samples one. ``temperature == 0.0`` falls back to greedy.

    Args:
        k: top-k cutoff. Capped to vocab size.
        temperature: sampling temperature; values below ``1e-8`` are treated as
            greedy to avoid division by zero.
        generator: optional torch.Generator for reproducibility.
    """
    if prompt_ids.dim() != 2:
        raise ValueError(f"prompt_ids must be 2D (B, T); got shape {tuple(prompt_ids.shape)}")
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    if temperature < 0.0:
        raise ValueError(f"temperature must be ≥ 0; got {temperature}")

    if temperature < 1.0e-8:
        return greedy_decode(model, prompt_ids, max_new_tokens=max_new_tokens, eos_token=eos_token)

    tokens = prompt_ids
    seen_eos = torch.zeros(prompt_ids.shape[0], dtype=torch.bool, device=prompt_ids.device)
    for _ in range(max_new_tokens):
        out = model(tokens)
        logits = out["logits"]
        assert isinstance(logits, Tensor)
        last = logits[:, -1, :] / temperature  # (B, V)
        k_eff = min(k, last.shape[-1])
        top_vals, top_idx = last.topk(k_eff, dim=-1)
        probs = torch.softmax(top_vals, dim=-1)
        sampled_in_topk = torch.multinomial(probs, num_samples=1, generator=generator)
        next_token = top_idx.gather(-1, sampled_in_topk)
        tokens = torch.cat([tokens, next_token], dim=-1)
        if eos_token is not None:
            seen_eos = seen_eos | (next_token.squeeze(-1) == eos_token)
            if bool(seen_eos.all().item()):
                pad = max_new_tokens - (tokens.shape[1] - prompt_ids.shape[1])
                if pad > 0:
                    pad_tensor = next_token.new_full((prompt_ids.shape[0], pad), eos_token)
                    tokens = torch.cat([tokens, pad_tensor], dim=-1)
                break
    return tokens
