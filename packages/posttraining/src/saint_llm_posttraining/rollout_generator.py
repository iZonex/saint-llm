"""GRPO rollout generation — sample G completions per prompt.

The :func:`grpo_train_step` driver consumes a :class:`RolloutBatch`
that has already been shaped to ``(B*G, T)``. The missing piece is
the part that *produces* those rollouts:

* Take a batch of prompts ``(B, P)``.
* For each prompt, sample ``G`` completions (typically with
  temperature > 0 so the group has variance).
* Pad the completions so every row in the resulting batch shares one
  sequence length.
* Build a parallel ``response_mask`` flagging the response positions
  (1) vs prompt + pad (0).
* Return ``(tokens (B*G, T), response_mask (B*G, T))`` ready to plug
  into :func:`build_rollout_batch`.

This module deliberately does **not** call the optimizer or compute
losses — it only handles the sampling + padding + masking math. The
caller decides which sampler to use (greedy, top-k, top-p) and feeds
the result into the existing GRPO trainer.

The default sampler is :func:`top_p_sample` because a temperature > 0
is required for the group to have non-degenerate variance. A greedy
fallback is supported for ablations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from saint_llm_inference.generate import greedy_decode, top_p_sample
from torch import Tensor

SamplerKind = Literal["top_p", "greedy"]


@dataclass(frozen=True)
class RolloutGenConfig:
    """Knobs for :func:`generate_grpo_rollouts`.

    Attributes:
        group_size:     G — completions per prompt.
        max_new_tokens: maximum response length.
        sampler:        ``"top_p"`` (default) or ``"greedy"``. Greedy
            yields G identical completions per prompt; useful only for
            shape ablations.
        temperature:    sampling temperature (top_p only).
        top_p:          nucleus probability (top_p only).
        eos_token:      optional early-stop sentinel.
        pad_token:      pad ID used to right-pad ragged completions.
        seed:           if set, seeds a per-call torch.Generator so
            the rollout pool is reproducible.
    """

    group_size: int = 8
    max_new_tokens: int = 64
    sampler: SamplerKind = "top_p"
    temperature: float = 1.0
    top_p: float = 0.9
    eos_token: int | None = None
    pad_token: int = 0
    seed: int | None = None


@torch.no_grad()
def generate_grpo_rollouts(
    actor: torch.nn.Module,
    prompt_ids: Tensor,
    cfg: RolloutGenConfig,
    *,
    sampler: Callable[..., Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample G completions per prompt and assemble a GRPO-shaped batch.

    Args:
        actor:      a :class:`SaintLLM` (or any module exposing the
            same forward dict). Must be in eval mode (caller's
            responsibility) to avoid dropout perturbing rollouts.
        prompt_ids: ``(B, P)`` long tensor.
        cfg:        :class:`RolloutGenConfig`.
        sampler:    optional override for the sampler callable.
            Default picks ``top_p_sample`` or ``greedy_decode`` per
            ``cfg.sampler``.

    Returns:
        ``(tokens, response_mask)``: both ``(B*G, P + max_new_tokens)``
        long tensors. Layout matches the GRPO trainer's expectation —
        rows ``[b*G : (b+1)*G]`` are the G completions for prompt b.
    """
    if prompt_ids.dim() != 2:
        raise ValueError(
            f"prompt_ids must be 2D (B, P); got shape {tuple(prompt_ids.shape)}",
        )
    if cfg.group_size <= 0:
        raise ValueError(f"group_size must be positive; got {cfg.group_size}")
    if cfg.max_new_tokens <= 0:
        raise ValueError(
            f"max_new_tokens must be positive; got {cfg.max_new_tokens}",
        )

    b, p = prompt_ids.shape
    expanded_prompt = prompt_ids.unsqueeze(1).expand(b, cfg.group_size, p).reshape(
        b * cfg.group_size, p,
    )

    sampler_fn = sampler if sampler is not None else _default_sampler(cfg)
    generator = (
        torch.Generator(device=prompt_ids.device).manual_seed(cfg.seed)
        if cfg.seed is not None
        else None
    )
    tokens = sampler_fn(
        actor=actor,
        prompt_ids=expanded_prompt,
        cfg=cfg,
        generator=generator,
    )

    # ``tokens`` is (B*G, P + max_new_tokens). Build the response mask:
    # 1 on response positions (P .. P + max_new_tokens), 0 elsewhere.
    bg, total = tokens.shape
    response_mask = torch.zeros((bg, total), dtype=torch.long, device=tokens.device)
    response_mask[:, p:] = 1
    # Mask out positions beyond the first eos in each row, if eos is set.
    if cfg.eos_token is not None:
        # Find first eos position in the response region of each row.
        in_response = tokens[:, p:]
        # eos_pos: (bg,) index of first eos within response, or
        # (max_new_tokens) when none found.
        is_eos = in_response == cfg.eos_token
        first_eos = torch.where(
            is_eos.any(dim=-1),
            is_eos.float().argmax(dim=-1),
            torch.full((bg,), cfg.max_new_tokens, device=tokens.device, dtype=torch.long),
        )
        # Positions strictly greater than first_eos within the response
        # become pad (no loss). The eos itself counts as response (the
        # model is supervised on emitting it).
        col = torch.arange(cfg.max_new_tokens, device=tokens.device).unsqueeze(0)
        post_eos = col > first_eos.unsqueeze(-1)
        response_mask[:, p:] = (~post_eos).to(torch.long)

    return tokens, response_mask


def _default_sampler(
    cfg: RolloutGenConfig,
) -> Callable[..., Tensor]:
    """Resolve the default sampler based on ``cfg.sampler``."""
    if cfg.sampler == "top_p":
        def _sample(
            *,
            actor: torch.nn.Module,
            prompt_ids: Tensor,
            cfg: RolloutGenConfig,
            generator: torch.Generator | None,
        ) -> Tensor:
            return top_p_sample(
                actor,
                prompt_ids,
                max_new_tokens=cfg.max_new_tokens,
                p=cfg.top_p,
                temperature=cfg.temperature,
                eos_token=cfg.eos_token,
                generator=generator,
            )
        return _sample
    if cfg.sampler == "greedy":
        def _sample_greedy(
            *,
            actor: torch.nn.Module,
            prompt_ids: Tensor,
            cfg: RolloutGenConfig,
            generator: torch.Generator | None,
        ) -> Tensor:
            return greedy_decode(
                actor,
                prompt_ids,
                max_new_tokens=cfg.max_new_tokens,
                eos_token=cfg.eos_token,
            )
        return _sample_greedy
    raise ValueError(f"unknown sampler {cfg.sampler!r}")


def decode_rollouts_for_reward(
    tokens: Tensor,
    decode: Callable[[list[int]], str],
    *,
    prompt_len: int,
) -> list[str]:
    """Decode the response slice of each rollout for the reward function.

    Args:
        tokens:     ``(B*G, T)`` long output of :func:`generate_grpo_rollouts`.
        decode:     callable mapping a list of token IDs to text
            (typically ``tokenizer.decode``).
        prompt_len: ``P`` — number of prompt tokens to skip when
            extracting the response.

    Returns:
        list of length ``B*G`` of decoded response strings.
    """
    if tokens.dim() != 2:
        raise ValueError(
            f"tokens must be 2D (B*G, T); got shape {tuple(tokens.shape)}",
        )
    if prompt_len < 0 or prompt_len > tokens.shape[1]:
        raise ValueError(
            f"prompt_len {prompt_len} out of range for T={tokens.shape[1]}",
        )
    return [decode(row[prompt_len:].tolist()) for row in tokens]
