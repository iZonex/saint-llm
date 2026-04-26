# ADR-0012: Logit softcap + learnable sink tokens for v0.0 — promote ARCH-02

- **Status**: proposed → accepted-v0.0 (this ADR)
- **Date**: 2026-04-25
- **Deciders**: Dmytro
- **Related**: AUGMENTATIONS row ARCH-02, ADR-0010 (MuonClip),
  `docs/specs/v0.0.md` deliverable D0.0.7
- **Sources**: Gemma 4 model card (logit softcap default in
  Gemma family), [Streaming-LLM (mit-han-lab)](https://github.com/mit-han-lab/streaming-llm)
  (attention sinks), arxiv 2511.21377 (controlling attention
  logits trade-offs), arxiv 2604.01563 (normalization-optimizer
  coupling)

## Context

Two related stability issues affect Transformer attention at long
context and large scale:

1. **Attention logit explosion.** Q · K^T scores can grow without
   bound during training, causing softmax saturation and gradient
   pathologies. ADR-0010 (MuonClip) addresses the *weight* side
   (post-step Q/K rescale). The *forward* side is independent: even
   with bounded weights, the input distribution can produce large
   logits (e.g. visual feature norm ≠ text feature norm).

2. **Attention-sink behavior at long context.** During streaming
   inference and long-document training, the model develops a
   strong dependence on the first 1-4 tokens — they become
   "attention sinks" that absorb otherwise-distracting attention
   mass. Removing them (via sliding-window / context drop)
   destroys generation quality. The Streaming-LLM paper documented
   this; subsequent work formalized it as a feature, not a bug.

Q1 2026 frontier evidence (research review 2026-04-25):

- **Gemma 4** uses logit softcap as the default attention-logit
  control mechanism. Gemma 4 model card explicitly cites it as
  a stability + long-tail-task quality fix.
- **Pre-trained learnable sink tokens** (vs ad-hoc preserving
  the first 4 tokens) is the cleaner production pattern. Anchored
  long context per [SinkTrack (ICLR 25 OpenReview Gg1aPETCL6)](https://openreview.net/forum?id=Gg1aPETCL6).
- **Logit softcap > QK-Norm for long-context tasks.** arxiv
  2511.21377 + 2604.01563 establish that blanket QK-Norm hurts
  long-context regression on long-tail tasks. Logit softcap
  preserves long-context behavior while bounding logit magnitude.

Combined recipe: softcap as forward-time soft bound + MuonClip as
post-step weight rescale + learnable sink tokens for streaming /
long-context attention dispersal control. The three are
orthogonal and complementary.

## Decision

Adopt logit softcap + learnable sink tokens for v0.0. Specifically:

### Logit softcap (Gemma-style)

In each attention layer (`CSA`, `HCA`, `SWAttention`), after
computing `Q · K^T / sqrt(d_head)` and before applying the
attention mask, apply:

```
attn_logits = soft_cap * tanh(attn_logits / soft_cap)
```

with `soft_cap = 50.0` default. This bounds logits to [-soft_cap,
+soft_cap] smoothly (no hard clip discontinuity).

Config knob: `AttentionConfig.logit_softcap: float | None = 50.0`.
`None` disables (escape hatch).

### Learnable sink tokens

Each sequence is prepended with `n_sink_tokens` (default 4)
learnable embedding rows. They have positions 0..n_sink-1; real
input tokens start at position n_sink_tokens.

Implementation:
- New parameter `SaintLLM.sink_embeddings: nn.Parameter` of shape
  `(n_sink_tokens, hidden_dim)`, initialized as
  `Normal(0, 1/sqrt(hidden_dim))`. Always trainable.
- `SaintLLM._embed_inputs` prepends them to the input embedding
  sequence. Real tokens after.
- KV cache layers (SWA / HCA / CSA) account for the sink prefix
  in their position tracking — sinks are at absolute positions
  0..n_sink-1, never evicted from the cache.
- LM head and MTP heads ignore sink positions in the loss
  computation (loss-mask gates positions 0..n_sink-1 to 0).
- Sample generation: sink tokens never produced by the model;
  they only exist on input.

Config knobs:
- `ModelConfig.n_sink_tokens: int = 4` (Streaming-LLM convention).
- `ModelConfig.sink_token_init_scheme: Literal["normal", "zero"] = "normal"`.

### Interaction with other v0.0 picks

- **MuonClip (ADR-0010):** soft_cap reads pre-clip logits.
  `_last_max_attn_logit` is taken **before** softcap applied
  (to preserve the signal Muon needs to detect drift). Softcap
  bounds the value the softmax sees; MuonClip bounds the value
  W_q/W_k can produce.
- **u-μP (ADR-0011):** sink-token init is `Normal(0, 1/sqrt(d))`
  per u-μP convention. No special LR scaling — sink tokens use
  the embedding parameter group.
- **Activation checkpointing:** sink tokens add 4 positions of
  activations per layer per sample. Negligible memory cost.

## Consequences

**Intended:**
- Forward-time logit explosion bounded — stable softmax even with
  noisy or extreme inputs (vision features, long-context
  out-of-distribution sequences).
- Long-context streaming inference stable: dedicated sinks absorb
  attention mass without being arbitrary first-K tokens of the
  input. We get streaming-quality generation by construction, not
  by post-hoc sliding-window heuristics.
- Combined with MuonClip (training-time weight rescale) + softcap
  (forward-time bound) + sink tokens (long-context dispersal),
  the attention path is end-to-end stabilized.

**Unintended / accepted:**
- Sink-token positions consume `n_sink_tokens × n_layers` KV cache
  slots — small, but non-zero. At `n_sink=4` and 8 layers, 32
  cache slots — negligible.
- Loss must mask out sink positions (positions 0..n_sink-1 are
  not predicted; they're fixed parameters). One additional masking
  in the loss helper.
- Generation samplers must skip sink positions when reporting
  generated text. Cosmetic.
- Softcap value (50.0) is empirical; if pretraining converges to
  natural logit distribution where 50 is a hard ceiling vs a soft
  bound, behavior may differ from no-cap baseline. We accept this
  trade for stability.

**Explicit non-effects:**
- Does NOT use QK-Norm. We deliberately avoid it (long-context
  regression per arxiv 2511.21377 + 2604.01563).
- Does NOT change MuonClip behavior (orthogonal, addressed
  separately).
- Does NOT interact with quantization paths beyond the dtype the
  attention logits land in (softcap composes through bf16/fp32).

## Alternatives considered

- **No softcap, only QK-Norm.** Rejected: QK-Norm regresses
  long-context tasks per published Q1 2026 ablations.
- **No softcap, only MuonClip.** Considered. MuonClip alone
  prevents *trained* weights from drifting, but doesn't prevent
  *forward-time* spikes from extreme inputs. Softcap is a cheap
  belt-and-suspenders.
- **Hard logit clip (e.g. `clamp(-50, +50)`) instead of tanh
  softcap.** Discontinuous; gradients zero outside the clip
  region. tanh is smooth, no zero-gradient region.
- **z-loss regularizer added to total loss.** Effective but
  costlier than softcap at training time.
- **No sink tokens** (rely on first-N input tokens being natural
  sinks). Rejected: published Streaming-LLM evidence and Q1 2026
  follow-ups (SinkTrack) show learnable dedicated sinks are
  cleaner and unlock streaming inference quality.
- **More than 4 sink tokens.** Considered. 4 is the published
  Streaming-LLM convention. Going to 16 or 32 is possible but
  adds cache cost for diminishing returns. Defer to v0.2 polish
  ablation if needed.
- **softcap value other than 50.0.** Gemma 4 uses 50 for the
  attention logit cap and 30 for the final logit cap (output of
  lm_head). v0.0 uses Gemma's value as-is. Tuning is a v0.1
  ablation if needed.

## Implementation notes

### Files affected

- `packages/core/src/saint_llm_core/attention/csa.py` — apply
  softcap inside the score computation, before mask + softmax.
- Same for `hca.py`, `swa.py`.
- `packages/core/src/saint_llm_core/config.py`:
  - `AttentionConfig.logit_softcap: float | None = 50.0`
  - `ModelConfig.n_sink_tokens: int = 4`
  - `ModelConfig.sink_token_init_scheme: Literal["normal", "zero"] = "normal"`
  - `ModelConfig.final_logit_softcap: float | None = 30.0`
    (applied at lm_head output)
- `packages/core/src/saint_llm_core/model.py`:
  - `SaintLLM.__init__`: register `sink_embeddings` parameter.
  - `SaintLLM._embed_inputs`: prepend sink rows.
  - `SaintLLM.forward`: pass through sink-aware position offsets
    to attention.
  - LM head: apply `final_logit_softcap` if set.
- `packages/training/src/saint_llm_training/losses.py`:
  - `cross_entropy_main` and `cross_entropy_with_mtp` get a
    `n_sink_tokens` parameter; mask out sink positions from
    targets.
- `packages/inference/src/saint_llm_inference/`:
  - KV cache layers track sink prefix; never evict positions
    0..n_sink-1.
  - Generation samplers (`greedy_decode`, `top_k_sample`,
    `top_p_sample`) start position counting at n_sink_tokens.

### Tests

- `packages/core/tests/test_logit_softcap.py`:
  - With softcap=50: feed deliberate spike (logits=200), assert
    post-softcap values in (-50, 50).
  - Gradient flows through softcap (sanity).
  - softcap=None disables (no-op).
- `packages/core/tests/test_sink_tokens.py`:
  - `n_sink_tokens=4`: forward pass with input shape (B, T) emits
    output shape (B, T+4) of attention activations, but loss only
    sees real T positions.
  - KV cache: sink positions persist across cached forward calls.
  - Generation: sample doesn't emit sink-position tokens.

### Validation in v0.0

- Long-context smoke test: 8K context generation on a tiny model
  doesn't NaN with softcap on.
- Eval comparison at v0.0 close: HellaSwag/ARC/MMLU with softcap
  on vs off (planned ablation if time permits).

### Out of scope for this ADR

- Tuning soft_cap value beyond Gemma's 50/30 defaults — v0.1
  ablation if needed.
- Sink token count beyond 4 — v0.2 polish ablation.
- "Anchored long-context" extensions per SinkTrack — v0.2+
  research track.
- Interaction with NSA-style native sparse attention (ARCH-03,
  v0.3) — addressed when NSA lands.

### Promotion of ARCH-02

Update AUGMENTATIONS.md row ARCH-02 status to `accepted-v0.0`.
