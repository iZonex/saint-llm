# ADR-0020: RegMix data mixture optimization for v0.0 — promote DATA-05

- **Status**: proposed → accepted-v0.0 (this ADR)
- **Deciders**: Dmytro
- **Date**: 2026-04-25
- **Related**: AUGMENTATIONS row DATA-05, ADR-0005 (13 langs incl
  UK), ADR-0018 (corpus mix), `docs/specs/v0.0.md` D0.0.2 + D0.0.9
- **Sources**: [arXiv 2407.01492 (RegMix)](https://arxiv.org/html/2407.01492v1),
  [Apple Optimal Data Mixtures](https://machinelearning.apple.com/research/optimal-data-mixtures),
  [arXiv 2305.10429 (DoReMi)](https://arxiv.org/abs/2305.10429),
  [arXiv 2503.01506 (SampleMix)](https://arxiv.org/html/2503.01506)

## Context

Once corpus slices are chosen (ADR-0018) and filtered (ADR-0019),
the question is **how to weight them**. Per-language token volume
is wildly imbalanced (EN ~60B vs UK ~12B); naively training
proportional to volume → English-dominant model. LANG-02 in
AUGMENTATIONS specifies a target vocab allocation, but that's the
*tokenizer* allocation — the *training mix* requires its own
weighting decision per language and per slice.

Frontier 2025-2026 mixture-optimization landscape:

- **DoReMi** ([arXiv 2305.10429](https://arxiv.org/abs/2305.10429))
  — domain reweighting via min-max optimization with a small
  proxy model. Industry standard 2024-2025. Costly to run
  (full proxy training per mixture candidate).
- **RegMix** ([arXiv 2407.01492](https://arxiv.org/html/2407.01492v1))
  — regression-based mixture optimization. Train ~64 small models
  with different random mixtures, fit a regression on (mixture,
  validation loss), use regression to predict optimal mixture.
  **~10% of compute, matches/beats DoReMi** per published
  ablation. The Q1 2026 reference recipe.
- **Apple Optimal Data Mixtures** — formalizes mixture as
  scaling-law-predictable; can predict optimal mixture for target
  domain via small proxy runs. Compatible with RegMix.
- **SampleMix** ([arXiv 2503.01506](https://arxiv.org/html/2503.01506))
  — sample-level mixing rather than domain-level. More granular
  but expensive; v0.1+ research.
- **Bivariate Data Mixing Law** — joint scaling of mixture
  proportion × volume. Useful for predicting per-language
  effective epochs.
- **Pure heuristic LANG-02 weights.** What we'd use without
  data-driven optimization; published frontier evidence shows
  data-driven beats heuristic by ~2-5% perplexity on held-out
  multilingual evals.

For v0.0's 13-language + code mix, RegMix is the right balance of
quality and cost.

## Decision

Adopt RegMix-style mixture optimization for v0.0 with the
following workflow:

### Phase 1 — Initial heuristic mix

Start v0.0 pipeline with the LANG-02-derived heuristic mix:

| Slice | Initial weight |
|---|---|
| EN (HPLT 3.0 + UltraFineWeb + FineWeb-Edu) | 35% |
| ZH (HPLT 3.0) | 14% |
| RU (HPLT 3.0) | 8% |
| **UK (Kobza + UberText)** | **3.5%** (with ~9 effective epochs) |
| ES, FR, DE | 4% each = 12% |
| JA, KO | 3% each = 6% |
| AR, HI, PT, IT | 2% each = 8% |
| Code (StarCoder2 filtered) | 9% |
| Reasoning (Nemotron Diverse-QA) | 4.5% |

Sums to 100%.

### Phase 2 — RegMix small-proxy regression

After v0.0 base run completes the first 1B tokens of training
with the heuristic mix:

1. Train ~32 small_flash variants (each ~200M tokens, ~4 hours
   on hpomen) with random perturbations of the Phase 1 mix
   (Dirichlet-perturbed weights).
2. Score each variant on held-out per-language perplexity +
   downstream eval (HellaSwag-en/uk, ARC-en, xnli-multilingual).
3. Fit a regression on (mixture vector, scalar quality metric).
4. Predict optimal mixture; compare to Phase 1 heuristic.

### Phase 3 — RegMix-optimized continuation

If Phase 2 predicts a meaningfully better mix (>1% improvement
on combined held-out), restart v0.0 main run with the optimized
mix. Otherwise stick with the heuristic.

### Compute budget

- Phase 1 + main run: ~10B tokens at ~24-48h on hpomen
- Phase 2: 32 × 200M = 6.4B tokens at ~6h on hpomen total
  (sequential), or 1-2h if hpomen + cloud parallel
- Phase 3: depends on whether mix changed; if it did, restart
  with reused tokenizer

Total budget overhead vs heuristic-only: ~30% additional compute
at v0.0, justified by **better foundation for v0.1+** (mixture
choice carries forward).

## Consequences

**Intended:**
- v0.0 pretrain mix is data-driven, not heuristic. Published
  evidence (RegMix paper) projects ~2-5% perplexity improvement
  on held-out multilingual evals at small scale; benefit scales
  with model size.
- Per-language balance (especially UK) is empirically tuned
  rather than guessed. UK upsample factor (3.5% × ~9 epochs)
  may be revised by RegMix evidence.
- Foundation set: v0.1+ inherits a known-good mix; the
  RegMix-fitted regression is reusable for v0.1 multimodal
  mixture decisions.

**Unintended / accepted:**
- ~30% compute overhead at v0.0 (32 small proxy runs + analysis).
  Acceptable trade for foundation quality.
- RegMix's regression is approximate; the predicted optimal mix
  may not be the true global optimum. We accept the local
  optimum near the heuristic starting point.
- Per-language eval choice (HellaSwag-en/uk + xnli + ARC) drives
  the optimization target. If the eval choice doesn't match
  v0.0 final eval, mix may be over-fitted to wrong metric.
  Mitigated by using a *combined* held-out metric, not single-
  benchmark.

**Explicit non-effects:**
- Does NOT change corpus selection (ADR-0018) or filter+dedup
  (ADR-0019).
- Does NOT change tokenizer (already built before Phase 2).
- Does NOT replace LANG-02 vocab allocation; LANG-02 controls
  *tokenizer* vocab share, RegMix controls *training* mix.

## Alternatives considered

- **Heuristic mix only (LANG-02 derived).** Cheaper. Rejected:
  loses ~2-5% perplexity per RegMix paper. Foundation matters.
- **DoReMi (full proxy training per candidate).** ~10× cost of
  RegMix for similar or worse results. Rejected.
- **SampleMix (sample-level mixing).** v0.1+ research; too
  expensive at v0.0.
- **No optimization, train on all data uniformly.** Rejected:
  English-dominant outcome, defeats the 13-language goal.
- **Single-step RegMix at v0.0 start (no Phase 1 heuristic
  warmup).** Considered. Risk: small-proxy regressions are
  noisier when starting from a uniform-random mix. Two-phase
  is safer.
- **Skip RegMix at v0.0, defer to v0.1.** Rejected: v0.1 mixture
  is multimodal (text + image + audio + video), much harder to
  optimize. Better to validate the mixture-optimization
  pipeline at v0.0 on the simpler text-only case first.

## Implementation notes

### Files affected

- New: `packages/data/src/saint_llm_data/pipeline/regmix.py`
  - `sample_dirichlet_perturbations(base_mix, n_samples, alpha)`
  - `fit_regression(mixtures, metrics) -> RegressionModel`
  - `predict_optimal(model, base_mix) -> dict[str, float]`
- New: `experiments/regmix_v0_0.py` driver:
  - Read Phase 1 mix from `configs/v0.0/pretrain-mix.yaml`
  - Generate 32 perturbed mixes
  - Run small_flash 200M-token trainings sequentially
  - Score each on held-out per-language eval
  - Fit regression, predict optimal
  - Output Phase 3 mix to `configs/v0.0/pretrain-mix-optimized.yaml`
- `experiments/run_small_flash.py` reads either Phase 1 or
  Phase 3 mix file based on flag.

### Tests

- `packages/data/tests/test_regmix_perturbations.py`:
  - Dirichlet samples sum to 1.0 (sanity)
  - Perturbation magnitude controllable via `alpha`
- `packages/data/tests/test_regmix_regression.py`:
  - Synthetic (mixture, metric) pairs with known optimum;
    fit regression; predicted optimum within tolerance.

### Validation in v0.0

- v0.0 exit report (D0.0.9) includes:
  - Phase 1 vs Phase 3 mixture comparison
  - Per-language perplexity at end of main run
  - Decision on whether RegMix recommendation was adopted

### Out of scope for this ADR

- Multi-objective RegMix (optimize for AIME + xnli + HellaSwag
  jointly with explicit Pareto trade) — v0.1+ research.
- SampleMix sample-level mixing (v0.1+ if budget allows).
- Apple Optimal Data Mixtures' explicit scaling-law extrapolation
  to predict v0.1 1B mix from v0.0 185M proxy data — captured
  as v0.1 deliverable, not v0.0.
- Bivariate Data Mixing Law application (predicts per-language
  effective epochs) — v0.1 polish.

### Promotion of DATA-05

Update AUGMENTATIONS.md row DATA-05 status to `accepted-v0.0`.
