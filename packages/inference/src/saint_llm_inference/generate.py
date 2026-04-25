"""Decoding loops — greedy / top-k / top-p with optional KV cache.

The base loops (``greedy_decode``, ``top_k_sample``, ``top_p_sample``)
recompute the full forward at every step (O(T^2) total). The
``_cached`` variants build a ``KVCacheBundle`` once, prefill the prompt
in a single forward, then decode one token at a time — partial
support for now: only ``SWAttention`` layers honor the cache; CSA/HCA
fall back to recompute (stages 3/4). On cfg.tiny / cfg.small_flash with
``first_dense_swa_layers=1`` the cached path saves the SWA layer's
forward each step (small but real).
"""

from __future__ import annotations

import torch
from saint_llm_core.model import SaintLLM
from torch import Tensor

from saint_llm_inference.kv_cache import KVCacheBundle


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
def greedy_decode_cached(
    model: SaintLLM,
    prompt_ids: Tensor,
    *,
    max_new_tokens: int,
    eos_token: int | None = None,
    bundle: KVCacheBundle | None = None,
) -> Tensor:
    """Greedy decoding with KV cache. Same contract as ``greedy_decode``.

    Builds a ``KVCacheBundle`` for ``model`` if one isn't supplied (capacity
    is ``prompt_len + max_new_tokens``). Prefill runs the full prompt in one
    forward; subsequent steps feed only the most recent token, leaning on
    the cache for layers that support it.
    """
    if prompt_ids.dim() != 2:
        raise ValueError(f"prompt_ids must be 2D (B, T); got shape {tuple(prompt_ids.shape)}")
    b, prompt_t = prompt_ids.shape
    capacity = prompt_t + max_new_tokens
    if bundle is None:
        bundle = KVCacheBundle.for_model(
            model,
            max_seq_len=capacity,
            batch_size=b,
            device=prompt_ids.device,
        )

    # Prefill.
    out = model(prompt_ids, kv_cache_bundle=bundle)
    logits = out["logits"]
    assert isinstance(logits, Tensor)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = torch.cat([prompt_ids, next_token], dim=-1)

    seen_eos = (
        torch.zeros(b, dtype=torch.bool, device=prompt_ids.device)
        if eos_token is None
        else (next_token.squeeze(-1) == eos_token)
    )
    if eos_token is not None and bool(seen_eos.all().item()):
        pad = max_new_tokens - 1
        if pad > 0:
            pad_tensor = next_token.new_full((b, pad), eos_token)
            tokens = torch.cat([tokens, pad_tensor], dim=-1)
        return tokens

    for _ in range(max_new_tokens - 1):
        out = model(tokens[:, -1:], kv_cache_bundle=bundle)
        logits = out["logits"]
        assert isinstance(logits, Tensor)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=-1)
        if eos_token is not None:
            seen_eos = seen_eos | (next_token.squeeze(-1) == eos_token)
            if bool(seen_eos.all().item()):
                pad = capacity - tokens.shape[1]
                if pad > 0:
                    pad_tensor = next_token.new_full((b, pad), eos_token)
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


def _filter_top_p(logits: Tensor, p: float) -> Tensor:
    """Mask out tokens outside the smallest-cumulative-mass-≤-p nucleus.

    Returns a copy of ``logits`` with ``-inf`` on masked positions. The nucleus
    always contains at least the highest-probability token (so an extreme
    p value never blanks out the distribution).
    """
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    # Keep the smallest set whose cumulative mass crosses p.
    sorted_keep = cumulative - sorted_probs <= p  # bool (B, V)
    # Always keep the top token even if p is tiny.
    sorted_keep[..., 0] = True
    keep = torch.zeros_like(sorted_keep)
    keep.scatter_(-1, sorted_idx, sorted_keep)
    return logits.masked_fill(~keep, float("-inf"))


@torch.no_grad()
def top_p_sample(
    model: SaintLLM,
    prompt_ids: Tensor,
    *,
    max_new_tokens: int,
    p: float = 0.9,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Nucleus (top-p) sampling, optionally combined with a top-k cap.

    For each step:
      1. Scale logits by ``1/temperature``.
      2. (Optional) restrict to the top ``top_k`` logits first.
      3. Drop the smallest-probability tail outside the cumulative-≤-p
         nucleus (rescaled).
      4. Sample from the remaining distribution.

    ``temperature == 0`` falls back to greedy. ``p`` outside (0, 1] raises.
    """
    if prompt_ids.dim() != 2:
        raise ValueError(f"prompt_ids must be 2D (B, T); got shape {tuple(prompt_ids.shape)}")
    if temperature < 0.0:
        raise ValueError(f"temperature must be ≥ 0; got {temperature}")
    if not (0.0 < p <= 1.0):
        raise ValueError(f"p must be in (0, 1]; got {p}")
    if top_k is not None and top_k <= 0:
        raise ValueError(f"top_k must be positive when set; got {top_k}")

    if temperature < 1.0e-8:
        return greedy_decode(model, prompt_ids, max_new_tokens=max_new_tokens, eos_token=eos_token)

    tokens = prompt_ids
    seen_eos = torch.zeros(prompt_ids.shape[0], dtype=torch.bool, device=prompt_ids.device)
    for _ in range(max_new_tokens):
        out = model(tokens)
        logits = out["logits"]
        assert isinstance(logits, Tensor)
        last = logits[:, -1, :] / temperature

        if top_k is not None:
            k_eff = min(top_k, last.shape[-1])
            cutoff = last.topk(k_eff, dim=-1).values[..., -1:]
            last = torch.where(last < cutoff, last.new_full((), float("-inf")), last)
        if p < 1.0:
            last = _filter_top_p(last, p)
        probs = torch.softmax(last, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, generator=generator)
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
