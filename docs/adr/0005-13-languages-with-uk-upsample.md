# ADR-0005: 13 languages including Ukrainian with aggressive UA upsample

- **Status**: accepted
- **Date**: 2026-04-25 (backfilled from 2026-04-25 user decision in
  `~/.claude/.../memory/project_v01_arch_decisions.md`)
- **Deciders**: Dmytro
- **Related**: LANG-01..LANG-06 in `docs/AUGMENTATIONS.md`,
  ADR-0004 (own tokenizer)

## Context

A multilingual base model has to choose its language set up-front
(tokenizer is permanent). Frontier models target different mixes:
Llama 4 ~30 langs EN-heavy, Gemma 4 ~140 langs but coverage is
shallow for low-resource, DeepSeek V3 EN/ZH-heavy with limited UA.

Ukrainian is severely under-represented in standard pretraining
mixes. UA-specific corpora (Kobza ~60B, UberText 2.0) exist but are
small relative to EN. UNLP 2025 papers establish that aggressive
UA upsampling combined with rigorous MT-pollution detection
defensible without quality regression.

## Decision

Target 13 languages: **EN, ZH, RU, UK, ES, FR, DE, JA, KO, AR, HI,
PT, IT**, with Ukrainian explicitly added per user decision and
upsampled aggressively (3.5% mix → ~9 effective epochs over Kobza).

Pretraining recipe per language follows the project's
`v01_arch_decisions` memory: temperature-scheduled epoch sampling
(τ 1.0 → 0.5) + 3-stack MT-pollution detector for UK
(translationese classifier + KenLM perplexity + URL heuristics).

## Consequences

**Intended:**
- UK quality reaches a frontier tier alongside EN/ZH/RU.
- Tokenizer (ADR-0004) and pretraining mix are aligned at vocab
  allocation level.
- Targeted distinctiveness: very few frontier models prioritize UK at
  this level.

**Unintended / accepted:**
- Pretraining mix is fixed at v0.0; adding a 14th language post-hoc
  would require retraining tokenizer and weights.
- UA upsample factor consumes EN-equivalent compute that a more
  EN-heavy mix would not need. Cost-of-distinctiveness, accepted.
- Quality of low-resource UA upsample is bounded by corpus-quality
  filter; under-detected MT pollution would degrade rather than
  improve UA.

## Alternatives considered

- **EN/ZH-only** for v0.0, add other languages at v0.1+. Rejected:
  tokenizer is permanent (ADR-0004); adding languages later means
  re-tokenizing.
- **Llama-4-style 30+ languages, no upsample.** Rejected: doesn't
  achieve UK quality target; user explicitly added UK.
- **Gemma-4-style 140+ languages with shallow coverage.** Rejected:
  spreads compute too thin for our target use case.
- **Drop UK requirement.** Rejected by user 2026-04-25.

## Implementation notes

- Mix recipe: see LANG-02 in AUGMENTATIONS.md.
- UA primary corpus: Goader/kobza (LANG-03).
- UA quality boost: UberText 2.0 (LANG-04, lower MT-pollution risk
  than newer corpora).
- MT-pollution detector: 3-stack per LANG-05.
- UNLP 2026 Lviv (May 29-30): consider submitting UA-tokenizer
  results from v0.0 run.
