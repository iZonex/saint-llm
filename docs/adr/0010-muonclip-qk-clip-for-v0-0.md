# ADR-0010: MuonClip QK-Clip for v0.0 — promote OPT-01

- **Status**: proposed → accepted-v0.0 (this ADR)
- **Date**: 2026-04-25
- **Deciders**: Dmytro
- **Related**: AUGMENTATIONS row OPT-01, ADR-0003 (V4 spine),
  `docs/specs/v0.0.md` deliverable D0.0.5,
  `packages/optim/src/saint_llm_optim/muon.py` (existing Muon
  implementation)
- **Sources**: arxiv 2507.20534 (Kimi K2 technical report),
  [QK-Clip blog](https://frontier.soket.ai/posts/muon_qk_clip/)

## Context

The project's spine includes Muon as the optimizer for hidden-layer
weight matrices (with AdamW splitter for embeddings/biases/RMSNorm
scales) — accepted at bootstrap per ADR-0003 and the V4 spine plan.
Muon at scale has a known pathology: attention logits can grow
unboundedly across training steps, leading to softmax saturation,
gradient instability, and eventually NaN losses or step-rejection
spikes. Vanilla Muon papers (Jordan 2024, Liu et al. "Muon is
Scalable" 2502.16982) acknowledge this informally; production
deployments have addressed it ad-hoc.

Q1 2026 frontier evidence (Q1 2026 research review 2026-04-25)
crystallized the fix:

**MuonClip** — a Q4 2025 / Q1 2026 stability technique from the Kimi
K2 team. After each Muon step, the maximum attention logit per layer
is monitored. When `max_logit > τ` for some threshold τ (default 100),
the Q and K projection weights of that layer are rescaled by
`(τ / max_logit)^0.5` so subsequent forward passes produce smaller
logits without changing the optimizer's update direction. The fix
operates **outside** the optimizer math — it is a post-step
parameter rescale, orthogonal to whatever update rule produced the
weights.

Empirical evidence:
- Kimi K2 (32B active / 1T total MoE) was pretrained for **15.5T
  tokens with zero loss spikes** using Muon + MuonClip ([2507.20534](https://arxiv.org/pdf/2507.20534)).
- The blog post above documents that the same recipe is being
  copied by other Q1 2026 frontier projects (referenced but not
  yet publicly named in published material).
- Cost: one max-reduce per attention layer per step + occasional
  rescale. Negligible vs the gradient pass.

The project's v0.0 plan targets ~10B tokens at 185M scale. Risk of
spikes is real even at 185M when Muon is the primary optimizer:
small-scale runs spike too if the LR is aggressive. Beyond v0.0 (1B
at 30B tokens for v0.1+), MuonClip is effectively required to avoid
re-discovering the pathology under cloud Blackwell time pressure.

## Decision

Adopt MuonClip QK-Clip as part of v0.0. Specifically:

1. Each attention layer (`CSA`, `HCA`, `SWAttention`) exposes a
   per-step `_last_max_attn_logit: float` updated during the
   forward pass. Updated only when `optimizer is Muon` and
   `MuonConfig.qk_clip_enabled is True` (no overhead when off).

2. The Muon optimizer's `step()` method, after applying the
   Newton-Schulz orthogonalization update, iterates over the
   model's attention layers. For each layer where
   `_last_max_attn_logit > tau`, multiplies the layer's
   `W_q.weight` and `W_k.weight` (or their multi-projection
   equivalents in CSA) by `(tau / _last_max_attn_logit) ** 0.5`.

3. `MuonConfig.qk_clip_enabled: bool = True` (on by default in v0.0).
   `MuonConfig.qk_clip_tau: float = 100.0` (default per published
   recipe; tunable).

4. Per-layer `qk_clip_count` counter is exposed via the trainer's
   metrics callback (`train/qk_clip_count_layer_{i}` in wandb).

5. A unit test asserts: feed attention scores with a deliberate
   spike (logits ≈ 10 * τ); after one Muon step + clip, the next
   forward pass produces logits ≤ τ.

## Consequences

**Intended:**
- v0.0 hpomen run completes 10B+ tokens with zero loss spikes,
  matching the Kimi K2 published guarantee.
- Eliminates a known scale-up risk that would otherwise resurface
  at v0.1 cloud Blackwell training under stricter time pressure.
- Cost is negligible: one reduce per layer per step (~constant time
  per layer), only when clip is needed.
- Pattern documented for v0.1+ where it matters more.

**Unintended / accepted:**
- One additional state per attention layer (`_last_max_attn_logit`).
  Tiny memory overhead.
- One additional config knob in `MuonConfig`. Defaults must be set
  carefully — `tau=100` is published; downstream tuning may be
  needed for non-standard architectures (CSA's lightning indexer
  logits may have different distribution; verify in unit test
  that CSA's max-attention-logit signal feeds the same path).
- Adds minor coupling between optimizer and attention modules
  (attention layers must report `_last_max_attn_logit`). This
  coupling is one-directional: optimizer reads, attention writes.
  Documented in code comments.

**Explicit non-effects:**
- Does NOT change Muon's update rule (still hybrid Newton-Schulz).
- Does NOT change forward-pass attention math (no clipping during
  forward — only post-step Q/K rescale).
- Does NOT interact with FP8 / FP4 quantization paths beyond
  multiplying the rescale factor through the same dtype the layer
  uses. Quantization-aware: the rescale happens on the master
  weights (FP32) where the optimizer step landed.

## Alternatives considered

- **Logit softcapping (Gemma-style) only.** Logit softcap clips
  *during* forward (`logits = c * tanh(logits / c)`). Effective at
  preventing softmax saturation but does NOT address the underlying
  W_q/W_k weight growth — it just hides the symptom. Combining
  softcap (ARCH-02 / D0.0.7) with MuonClip (this ADR) is
  recommended; they are orthogonal. Softcap protects against
  forward-time saturation; MuonClip prevents weights from drifting
  to a regime where they require softcap.
- **QK-Norm.** Apply RMSNorm to Q and K before attention. Effective
  at preventing logit growth at the cost of long-context regression
  on long-tail tasks per arxiv 2604.01563 + 2511.21377. Not adopted.
  We use logit softcap + MuonClip instead, which preserves long-
  context behavior.
- **Lower learning rate.** Shrink Muon's LR enough that weights
  never grow into the spike regime. Trades stability for converge
  rate; published evidence (Kimi K2) shows MuonClip lets us keep
  the original LR.
- **Switch optimizer to AdamW for attention.** Defeats the purpose
  of Muon (which is faster compute-efficient training). Rejected.
- **Add z-loss regularizer.** Adds an explicit logsumexp penalty.
  Effective but cost is higher (extra term in loss). Logit softcap
  is cheaper and equivalent in published ablations.
- **Wait until v0.1 to add.** Kimi K2 published evidence is at
  large scale, but small-scale runs also spike if LR is aggressive.
  No reason to delay; cost is negligible.

## Implementation notes

### Files affected

- `packages/optim/src/saint_llm_optim/muon.py` — add `qk_clip_step`
  method, called from `step()` after Newton-Schulz update.
  Reads each attention layer's `_last_max_attn_logit`, rescales
  if above threshold.
- `packages/optim/src/saint_llm_optim/config.py` (or wherever
  `MuonConfig` lives) — add `qk_clip_enabled`, `qk_clip_tau`,
  `qk_clip_layer_predicate`. Last is a callable filter for which
  modules count as "attention" — defaults to
  `isinstance(m, (CSA, HCA, SWAttention))`.
- `packages/core/src/saint_llm_core/attention/csa.py` — patch
  forward to update `self._last_max_attn_logit` from the post-
  scale-pre-softmax score tensor's max. Same for `hca.py`, `swa.py`.
  Implementation: `self._last_max_attn_logit = float(scores.detach().abs().max())`
  inside the score-computation block, gated by a flag set by the
  optimizer at init (so it's a no-op when MuonClip is off).

### Tests

- `packages/optim/tests/test_muon_qk_clip.py`:
  - Construct a 2-layer model with synthetic CSA layer.
  - Force its W_q to a large norm so first forward produces
    `max_attn_logit ≈ 10 * τ`.
  - One forward + backward + Muon.step() + verify
    `qk_clip_count == 1`.
  - Second forward: verify `max_attn_logit ≤ τ`.
- `packages/optim/tests/test_muon_qk_clip_disabled.py`:
  - Same setup but `qk_clip_enabled=False`. Verify no rescale,
    `_last_max_attn_logit` is not read.

### Validation in v0.0

- v0.0 hpomen run reports `train/qk_clip_count_*` metrics in wandb.
- Exit criterion: zero loss spikes across the run.

### Out of scope for this ADR

- Adapting MuonClip for HCA's compressed-block attention or CSA's
  lightning-indexer scores: the published recipe is for vanilla
  attention. v0.0 implementation uses CSA's main-attention logit
  path; indexer logits are a separate decision (likely no clip
  needed since indexer scores are post-softmax already), tracked
  as v0.1 polish.
- Tuning τ per architecture beyond default 100. v0.0 keeps default;
  empirical ablation is a v0.1 deliverable if zero spikes is not
  enough.
- Combining with z-loss or other regularizers (out of scope; v0.1
  research if MuonClip is insufficient).

### Promotion of OPT-01

This ADR promotes `OPT-01` from `proposed` to `accepted-v0.0` in
`docs/AUGMENTATIONS.md`. Update the OPT-01 row's status column when
the implementation lands, with `[implemented: <commit_sha>]`
appended per existing process.
