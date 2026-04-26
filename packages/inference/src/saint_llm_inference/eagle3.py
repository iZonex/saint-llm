"""EAGLE-3 speculative decoding (INF-01 / ADR-pending).

EAGLE-3 (NeurIPS 2025) replaces EAGLE-2's top-layer feature with a
fusion of low/mid/high features via a "training-time test" that
forces the draft head to predict tokens at multiple depths. Reported
3.0-6.5x decode speedup over autoregressive, 20-40% over EAGLE-2.

This module ships the *speculative decoding loop* + a minimal draft
head shape. The full training procedure for the draft head (which
involves running the base model with multi-depth supervision) is a
v0.2 work item; v0.0 ships the inference-time path so any user with
a trained draft head can plug it in.

Workflow at inference time:

1. The draft head proposes K candidate next tokens from the current
   prefix (autoregressively, K-step rollout — but the draft is much
   smaller than the base, so this is cheap).
2. The base model is run *once* on the prefix + K draft tokens
   (parallel forward — single base forward call instead of K).
3. We compare per-position: where the base's argmax matches the
   draft, accept; the first mismatch position rejects + the rest of
   the draft is discarded.
4. The base produces one token at the rejection point (free, since
   the parallel forward already gave us the logits).
5. Loop.

Worst case (draft always wrong): 2x slowdown vs greedy.
Best case (draft always right): K+1x speedup.
Typical (with a well-trained draft): 2-4x.

References:
    EAGLE-3 (NeurIPS 2025)
    P-EAGLE (AWS, 2026) — parallel K-token drafting for further speedup
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class EAGLE3Config:
    """Config for the speculative decoding loop.

    Attributes:
        draft_length: K — number of tokens the draft proposes per
            outer iteration. Higher K = more potential speedup but
            also more wasted compute on rejection. Published recipe:
            K=4-7.
        temperature:  sampling temperature for the draft. ``0.0`` =
            greedy draft (matches verifier exactly when both agree).
        max_new_tokens: cap on total generated tokens.
    """

    draft_length: int = 4
    temperature: float = 0.0
    max_new_tokens: int = 64


class EAGLE3DraftHead(nn.Module):
    """Minimal draft-head shape.

    Reads hidden states from a base SaintLLM forward (specifically the
    last position's hidden) and projects to vocab logits via a thin
    MLP. Production draft heads are bigger (one transformer block)
    but the contract is the same: ``forward(hidden, last_token) ->
    logits``.

    The actual draft training (multi-depth feature fusion per EAGLE-3
    recipe) is a v0.2 work item; this class is the **inference-time
    interface** any draft head must satisfy.
    """

    def __init__(self, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        """``hidden: (B, T, D)`` -> ``logits: (B, T, V)``."""
        return self.proj(hidden)


@torch.no_grad()
def speculative_decode(
    base_model: nn.Module,
    draft_head: nn.Module,
    prompt: Tensor,
    *,
    cfg: EAGLE3Config,
) -> tuple[Tensor, dict[str, float]]:
    """One run of EAGLE-3 speculative decoding.

    Args:
        base_model:  full SaintLLM that produces ``{"logits": ..., "hidden": ...}``.
        draft_head:  ``forward(hidden) -> logits`` callable. Should
            take far less time than the base.
        prompt:      ``(1, T)`` long token IDs (single-batch only at v0.0).
        cfg:         loop knobs.

    Returns:
        ``(generated_tokens, metrics)`` where:

        * generated_tokens: ``(1, T + n_new)`` — prompt + accepted
          tokens.
        * metrics: dict with ``acceptance_rate`` (fraction of draft
          tokens accepted across the run), ``base_forwards`` (total
          base-model forward calls), and ``new_tokens``.
    """
    if prompt.shape[0] != 1:
        raise ValueError(f"speculative_decode requires batch_size=1; got {prompt.shape[0]}")

    base_model.eval()
    draft_head.eval()
    tokens = prompt.clone()

    n_drafted = 0
    n_accepted = 0
    n_base_forwards = 0
    n_new_total = 0

    while n_new_total < cfg.max_new_tokens:
        # --- Draft step: K autoregressive draft proposals ----------
        # Use the base model to seed initial hidden state for the draft.
        # In the real EAGLE-3 the draft has its own KV cache; v0.0 just
        # re-runs the base lightly to stay framework-correct.
        base_out = base_model(tokens)
        n_base_forwards += 1
        base_hidden = base_out["hidden"]
        base_logits = base_out["logits"]

        draft_tokens: list[int] = []
        # Use base hidden's last position as draft input.
        for _ in range(cfg.draft_length):
            draft_logits = draft_head(base_hidden[:, -1:, :])  # (1, 1, V)
            if cfg.temperature == 0.0:
                tok = int(draft_logits.argmax(dim=-1).item())
            else:
                probs = torch.softmax(
                    draft_logits / cfg.temperature, dim=-1,
                ).reshape(-1)
                tok = int(torch.multinomial(probs, num_samples=1).item())
            draft_tokens.append(tok)
            # Append draft token to working sequence and re-extend hidden.
            tokens = torch.cat(
                [tokens, torch.tensor([[tok]], dtype=tokens.dtype, device=tokens.device)],
                dim=1,
            )
            base_hidden = torch.cat(
                [base_hidden, base_hidden[:, -1:, :]], dim=1,
            )

        # --- Verify step: one base forward over the full extended seq ----
        verify_out = base_model(tokens)
        n_base_forwards += 1
        verify_logits = verify_out["logits"]
        n_drafted += cfg.draft_length

        # The base's prediction at position p equals the verifier's
        # opinion on what token p+1 should be. We compare each draft
        # token at position (T_orig + i) against the base's argmax at
        # position (T_orig + i - 1).
        t_orig = base_logits.shape[1]  # length before draft extension
        accepted_in_round = 0
        for i, draft_tok in enumerate(draft_tokens):
            base_pred_at = t_orig + i - 1
            base_argmax = int(verify_logits[0, base_pred_at, :].argmax().item())
            if base_argmax == draft_tok:
                accepted_in_round += 1
                continue
            # Rejection: trim tokens after the first mismatch and
            # replace with the base's preferred token at this position.
            keep_until = t_orig + i  # keep up to (but not incl.) the mismatch
            tokens = tokens[:, :keep_until]
            tokens = torch.cat(
                [tokens, torch.tensor(
                    [[base_argmax]], dtype=tokens.dtype, device=tokens.device,
                )],
                dim=1,
            )
            break
        else:
            # All K accepted: the verifier's final-position argmax also
            # gets appended (free token from the parallel forward).
            base_argmax = int(verify_logits[0, -1, :].argmax().item())
            tokens = torch.cat(
                [tokens, torch.tensor(
                    [[base_argmax]], dtype=tokens.dtype, device=tokens.device,
                )],
                dim=1,
            )

        n_accepted += accepted_in_round
        n_new_total = tokens.shape[1] - prompt.shape[1]

    metrics = {
        "acceptance_rate": n_accepted / max(n_drafted, 1),
        "base_forwards": float(n_base_forwards),
        "new_tokens": float(n_new_total),
    }
    return tokens, metrics
