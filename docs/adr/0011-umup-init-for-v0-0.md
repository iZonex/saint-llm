# ADR-0011: u-μP init for v0.0 — promote OPT-02

- **Status**: proposed → accepted-v0.0 (this ADR)
- **Date**: 2026-04-25
- **Deciders**: Dmytro
- **Related**: AUGMENTATIONS row OPT-02, TOK-05 (tied embeddings),
  ADR-0003 (V4 spine), `docs/specs/v0.0.md` deliverable D0.0.6
- **Sources**: arxiv 2407.17465 (u-μP "unit-scaled μP"),
  arxiv 2409.19913 (LR-with-horizon scaling),
  arxiv 2305.10429 (μP / Tensor Programs)

## Context

Maximum-update parameterization (μP) is the standard 2024-2026
recipe for transferring learning rates across model widths: tune
hyperparameters at small scale, transfer to large scale without
re-tuning. The original μP (Yang et al.) is mathematically clean
but has two practical pitfalls:

1. **Embedding tying interaction.** When `lm_head.weight` is tied
   to `embed.weight` (TOK-05 in our spine), vanilla μP's
   embedding-LR scaling leaves the head un-scaled — or, worse,
   double-scaled — depending on framework implementation. Reading
   the original paper requires careful interpretation; reading
   most reference implementations exposes the bug.

2. **Width-conversion bookkeeping.** Vanilla μP requires the user
   to define a base width and a target width and apply per-param
   ratios; getting this wrong silently produces wrong LRs that
   look fine but converge to worse losses.

**u-μP** ("unit-scaled μP", arxiv 2407.17465, Microsoft + DeepMind
2024) is the cleaned-up successor:
- Per-param scaling is applied at init time so the per-param LR
  ratios become 1 across widths — the optimizer sees a uniform LR.
- Embedding-tying is handled correctly: tied rows get init
  variance computed from `1 / sqrt(fan_in)` of the embedding,
  and the head reads the same matrix without re-scaling.
- Hyperparameters transfer cleanly across width sweeps: the
  base-LR you tune at 64M-hidden transfers to 768M-hidden with no
  manual ratio.

The Q1 2026 frontier evidence (research review 2026-04-25):
- u-μP is now the default in NVIDIA Megatron's reference μP
  configurations (per [Megatron emerging optimizers blog](https://developer.nvidia.com/blog/advancing-emerging-optimizers-for-accelerated-llm-training-with-nvidia-megatron/)).
- Adopted in 2025-2026 codebases including Smollm-corpus releases
  and several 2026-published ablations.
- Strictly improves over vanilla μP per published ablations
  (matches or beats on held-out PPL at iso-tokens and iso-FLOPs).
- Combined with horizon-corrected LR (arxiv 2409.19913) — longer
  training horizons want smaller peak LRs even after μP transfer
  — gives the most reliable HP transfer recipe in 2026.

The project plans width sweeps from `cfg.tiny` (64M) for testing
through `cfg.small_flash` (185M) for v0.0 hpomen and up to ~1B for
v0.1 cloud Blackwell. Without HP transfer, every width requires a
fresh sweep — wasted compute.

## Decision

Adopt u-μP as the initialization + LR-scaling scheme for v0.0
onward. Specifically:

1. New module `packages/core/src/saint_llm_core/init.py` with
   `umup_init(model, cfg)` function applied during `SaintLLM.__init__`.
   Replaces the current `_init_weights` ad-hoc logic.

2. Per-parameter rules:
   - **Embedding** (`nn.Embedding`): init from `Normal(0, sigma_e)`
     with `sigma_e = 1.0`. (u-μP convention: embeddings are unit-
     normalized.)
   - **Tied lm_head**: shares `embed.weight`; no separate init
     and no separate LR scaling.
   - **Standard `nn.Linear` (hidden layers)**: init from
     `Normal(0, 1 / sqrt(fan_in))`. Optimizer LR scaled by
     `base_lr * (base_width / actual_width)` for the hidden
     parameter group.
   - **Output projections feeding residual** (typically `o_proj`
     in attention, `down_proj` in MLP): init from
     `Normal(0, 1 / sqrt(fan_in * n_layers))` (1 / sqrt(L) per
     layer) so the residual sum stays unit-scale at init.
   - **mHC dynamic-parameterization Ws** (`w_pre`, `w_res`,
     `w_post`): stay at zero init by design (uniform-start
     initial gating, per current code) — u-μP defers to existing
     mHC convention.
   - **Frozen modules** (e.g. `generation_head` while gated off):
     skip — leave their zero init alone.

3. Optimizer config grows:
   - `Trainer.set_param_groups()` builds two groups: "embedding"
     (constant LR, AdamW) and "hidden" (μ-scaled LR, Muon).
   - `MuonConfig.base_lr_hidden`, `MuonConfig.base_width`,
     `AdamWConfig.base_lr_embedding` — explicit knobs documented
     in the config docstring.

4. `ModelConfig.init_scheme: Literal["normal", "umup"] = "umup"`
   for v0.0.

5. Ablation: at v0.0 small-scale validation, run two 1B-token
   small_flash variants — `init_scheme="normal"` (control) and
   `init_scheme="umup"` (treatment). u-μP must match or beat
   normal on held-out PPL.

## Consequences

**Intended:**
- HP transfer from 185M (v0.0 hpomen) to 1B (v0.1 cloud Blackwell)
  works without re-tuning peak LR.
- Tied-embedding pathology (lm_head and embed both want their own
  LR scaling, vanilla μP gets confused) is eliminated.
- Per-param LR ratios are 1 everywhere; the optimizer sees a clean
  uniform LR. Easier to reason about.

**Unintended / accepted:**
- Migration cost: existing `cfg.tiny` and `cfg.small_flash` were
  trained against vanilla normal init. If we resume from existing
  checkpoints, init scheme doesn't matter (weights are already
  trained). For from-scratch v0.0, u-μP applies fresh.
- Init module adds ~80 LOC of code paths to maintain.
- One additional config knob (`init_scheme`) — defaults to "umup"
  but "normal" stays available as escape hatch.

**Explicit non-effects:**
- Does NOT change architecture (CSA + HCA + mHC + DeepSeekMoE
  unchanged).
- Does NOT change optimizer math (Muon + AdamW splitter
  unchanged; only the per-param LR scaling is computed differently).
- Does NOT interact with quantization (init happens in FP32 master
  weights regardless of forward-pass quant policy).
- Does NOT interact with MuonClip (ADR-0010): MuonClip operates on
  trained weights; u-μP operates on init. They are independent.

## Alternatives considered

- **Vanilla μP (Yang 2022).** Strictly inferior per published u-μP
  ablations; embedding-tying interaction is a real pitfall.
- **No HP transfer (re-tune at every width).** Wastes compute at
  every scale-up. Rejected.
- **Tensor Programs Δ-init (full TP framework).** More expressive
  but huge ergonomic cost. u-μP captures the practical wins
  without the framework overhead.
- **DeepNorm or fixup-style depth-aware init alone.** Addresses
  depth scaling but not width scaling. Combined with horizon-
  corrected LR, u-μP covers both.
- **Defer to v0.1.** Considered but: implementation cost is small
  (~80 LOC + ablation), and v0.0's whole point is to validate the
  spine for HP transfer to v0.1+. Better to land at v0.0 so we
  prove the HP transfer story end-to-end.

## Implementation notes

### Files affected

- New: `packages/core/src/saint_llm_core/init.py` — u-μP init
  helpers.
- `packages/core/src/saint_llm_core/model.py` — `SaintLLM._init_weights`
  becomes a thin dispatch on `cfg.init_scheme`. Default delegates
  to `init.umup_init`.
- `packages/core/src/saint_llm_core/config.py` — add
  `ModelConfig.init_scheme: Literal["normal", "umup"] = "umup"`.
- `packages/training/src/saint_llm_training/trainer.py` — accept
  param-group split from model rather than hardcoding optimizer
  setup. Pass-through for u-μP scaling.
- `packages/optim/` — Muon and AdamW configs extended with
  `base_lr` + `base_width` per group.

### Tests

- `packages/core/tests/test_umup_init.py`:
  - Initialize 64M and 256M variants with same `base_lr_hidden=1e-3`
    and `base_width=64`.
  - Verify per-layer LR scaling: 256M's hidden LR is `1e-3 *
    (64/256) = 2.5e-4`.
  - Verify tied embedding has expected variance.
  - Verify residual-feeder layers (o_proj, down_proj) have
    `1/sqrt(L)` extra factor.
- `packages/core/tests/test_init_scheme_dispatch.py`: smoke test
  that `init_scheme="normal"` and `init_scheme="umup"` both
  produce non-NaN initial losses on the tiny config.

### Ablation in v0.0

- Two 1B-token small_flash runs:
  - `runs/v0.0/ablations/init_normal/` — control
  - `runs/v0.0/ablations/init_umup/` — treatment
- Same data, same optimizer (Muon + MuonClip per ADR-0010),
  same schedule, only init differs.
- Report PPL trajectory + final eval (HellaSwag/ARC/MMLU).
- u-μP must match or beat normal on at least 2 of 3 evals; if
  tie/loss, escalate to a per-param diagnostic.

### Out of scope for this ADR

- Tensor-Programs-style scaling beyond u-μP (Δ-init). v0.1+ research
  if u-μP is insufficient.
- Horizon-corrected LR scaling (arxiv 2409.19913) — separate
  decision, captured as future ADR-NNNN.
- MoE-specific init scaling (granularity G as a μP axis per
  arxiv 2509.23678) — research track for v0.1+.

### Promotion of OPT-02

Update AUGMENTATIONS.md row OPT-02 status to `accepted-v0.0`.
Append `[implemented: <commit_sha>]` when ablation lands.
