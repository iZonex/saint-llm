# ADR-0018: v0.0 multilingual corpus mix — promote DATA-01, DATA-02

- **Status**: proposed → accepted-v0.0 (this ADR)
- **Date**: 2026-04-25
- **Deciders**: Dmytro
- **Related**: AUGMENTATIONS rows DATA-01, DATA-02; LANG-01..06,
  ADR-0005 (13 langs incl UK), ADR-0004 (own BBPE 131K),
  `docs/specs/v0.0.md` D0.0.1 + D0.0.2
- **Sources**: [arXiv 2511.01066 (HPLT 3.0)](https://arxiv.org/abs/2511.01066),
  [Nemotron-CC v2 HF](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2),
  [arXiv 2412.02595 (Nemotron-CC)](https://arxiv.org/abs/2412.02595),
  [Goader/kobza](https://huggingface.co/datasets/Goader/kobza),
  [UberText 2.0](https://lang.org.ua/en/corpora/),
  [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
  [bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata)

## Context

v0.0 needs a real text corpus for both tokenizer training (D0.0.1)
and pretraining (D0.0.8). The corpus must cover the 13-language
plan (ADR-0005), include code, and have aggressive UA upsample
with pollution filtering.

Q1 2026 frontier consensus on text-pretraining corpora:

- **HPLT 3.0** ([arXiv 2511.01066](https://arxiv.org/abs/2511.01066))
  is the new gold standard for multilingual base data: **30T tokens
  / ~200 (lang, script) pairs / 16T English / 13T non-English**.
  Plans 3PB augmentation from ArchiveBot in early 2026. Replaces
  mC4/CulturaX-tier corpora.
- **Nemotron-CC v2** ([HF](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2))
  adds 8 new CC snapshots (2024-25), global dedup, synthetic
  rephrasing via Qwen3-30B-A3B, and **synthetic Diverse-QA
  translated into 15 languages** (boosts Global-MMLU +10). 6.3T
  tokens (4.4T original + 1.9T synthetic).
- **FineWeb-Edu** remains the cleanest English-only educational
  slice, useful for tokenizer training proportion.
- **Ukrainian**: Goader/kobza (~60B tokens, primary UA, LANG-03)
  + UberText 2.0 (high-quality, pre-LLM era — lower MT-pollution
  risk per LANG-04).
- **Code**: StarCoder2-data (bigcode/starcoderdata) is the standard
  open code corpus.

Frontier comparisons:
- Llama 4: ~30T+ tokens private mix, English-heavy
- Gemma 4: ~2T-15T mix incl multilingual
- Smollm-corpus 2: open mix using FineWeb + multilingual
- Qwen3: private mix
- DCLM v2: 240T-token open corpus, weak per-language coverage for
  low-resource

HPLT 3.0 + Nemotron-CC v2 + Kobza UK + UberText + StarCoder2 +
FineWeb-Edu is the best-in-class **open** mix for our 13-language +
code goal as of April 2026.

## Decision

Adopt the following corpus slices for both v0.0 tokenizer training
(D0.0.1) and v0.0 pretraining data pipeline (D0.0.2):

| Slice | Source | Role | Approx pretrain tokens |
|---|---|---|---|
| English web | HPLT 3.0 (en) + UltraFineWeb-en (filtered) | Primary EN | 60B |
| English educational | FineWeb-Edu (filtered) | High-quality EN | 5B |
| English reasoning | Nemotron-CC v2 Diverse-QA (en + translated 15 langs) | Reasoning + multilingual reasoning | 5B |
| Multilingual web | HPLT 3.0 by (lang, script) for 11 non-EN/non-UK languages | Base multilingual | 80B |
| Ukrainian primary | Goader/kobza | UA primary | 9B |
| Ukrainian quality | UberText 2.0 | UA pre-LLM-era boost | 3B |
| Code | bigcode/starcoderdata, filtered to top-quality langs | Code | 8B |

**Total target ~170B tokens** for the v0.0 pretraining pool. The
small_flash run consumes ~10B; the rest is headroom for re-runs
and v0.0.1 polish.

Tokenizer training (D0.0.1) samples ~5-10M sentences from this
mix proportional to LANG-02 vocab allocation (Latin 38, CJK 30,
Cyrillic 12, Arabic 6, Devanagari 5, code 9), NOT proportional to
token volume — to avoid English-dominated merges.

## Consequences

**Intended:**
- Q1 2026 frontier-class data without paying for proprietary
  mixes; saint-llm's open-only commitment honored.
- HPLT 3.0 by (lang, script) ISO pair gives clean per-language
  shards we can balance in D0.0.2 mixture step (DATA-05).
- Nemotron-CC v2's Diverse-QA-in-15-languages closes the gap
  between EN reasoning capability and multilingual reasoning.
- UA quality story is the strongest in any open project (Kobza
  primary + UberText quality boost + LANG-05 3-stack MT-pollution
  detector applied during D0.0.2 filter stage).

**Unintended / accepted:**
- Disk: HPLT 3.0 fully downloaded is multi-TB. We pull
  per-language shards and filter aggressively in D0.0.2 to land
  on ~170B tokens at our quality threshold.
- Nemotron-CC v2 includes synthetic data (1.9T of 6.3T). Some
  papers raise concerns about synthetic data drift; v0.0 accepts
  this trade — Nemotron's synthetic Diverse-QA is well-curated
  per their published recipe, and the +10 Global-MMLU gain is
  documented.
- StarCoder2-data filtering: we use only the top languages
  (Python, JS/TS, Go, Rust, C++, Java) to avoid blowing up code
  vocabulary share.
- Some HPLT 3.0 shards may have residual MT pollution for
  low-resource langs (their pipeline is good but not perfect);
  saint-llm's UK-specific 3-stack detector (LANG-05) catches it
  for UK; for other low-resource langs we accept the residual.

**Explicit non-effects:**
- Does NOT commit to specific sub-language proportions. Those are
  set by RegMix (DATA-05, ADR-0020) based on small-proxy regression.
- Does NOT exclude future corpus additions at v0.1+ (e.g. CommonPile,
  HPLT 3.5 if released, language-specific high-quality slices).

## Alternatives considered

- **mC4 / CulturaX (current Llama-3-era stack).** Rejected: HPLT 3.0
  outperforms per FineWeb-2 ablation (arXiv 2506.20920).
- **DCLM v2 240T as primary EN.** Rejected: weaker per-language
  coverage for our non-EN priorities; EN-only.
- **RedPajama-V3.** Rejected: smaller, less well-filtered for
  multilingual.
- **Wait for HPLT 3.5 / FineWeb-3.** Considered. No firm release
  schedule; would slip v0.0 by quarter+. v0.0 ships on HPLT 3.0
  + Nemotron-CC v2; v0.1 can refresh if newer corpora arrive.
- **Add Common Crawl raw + own filtering pipeline.** Rejected for
  v0.0: too much engineering work; HPLT 3.0 + Nemotron-CC v2
  already did the CC-filter work.
- **Drop synthetic data (Nemotron Diverse-QA).** Considered. Pure
  CC + UA + code is "cleaner" but loses the multilingual reasoning
  +10 Global-MMLU. Synthetic is in v0.0; if a regression is
  detected, we ablate at v0.0 close.
- **More UA volume (12B+).** Considered. Kobza + UberText is ~12B
  total available; sampling ratio per LANG-02 (UA 3.5% of mix → ~9
  effective epochs over Kobza). Going beyond requires synthetic
  UA data, deferred to v0.1.

## Implementation notes

### Files affected (all v0.0 D0.0.1 + D0.0.2 deliverables)

- New: `configs/v0.0/tokenizer-mix.yaml` declaring slices + sample
  proportions for tokenizer training
- New: `configs/v0.0/pretrain-mix.yaml` declaring slices + initial
  RegMix-pre weights for pretraining
- `packages/data/src/saint_llm_data/pipeline/v0_0.py` reads these
  configs, dispatches per-slice loaders
- New per-slice loaders:
  - `hplt_loader.py` (HPLT 3.0 by lang+script)
  - `nemotron_cc_loader.py` (Nemotron-CC v2 + Diverse-QA)
  - `kobza_loader.py` (UA primary)
  - `ubertext_loader.py` (UA quality)
  - `starcoder2_loader.py` (code, language-filtered)
  - `fineweb_edu_loader.py` (existing or thin wrapper)

### Validation

- Manifest report at end of pipeline run lists per-slice token
  count, doc count, byte size — committed to
  `runs/v0.0/data/manifest.json`.
- Spot-check 100 random docs per language for quality regressions.
- Per-language fertility cross-check with tokenizer (D0.0.1).

### Out of scope for this ADR

- Quality classifier choice (DATA-03 in ADR-0019)
- Dedup algorithm choice (DATA-04 in ADR-0019)
- Mixture weighting (DATA-05 in ADR-0020)
- MorphBPE evaluation (DATA-10 in ADR-0021)
- v0.1 multimodal corpora (OBELICS-2 etc., separate ADR at v0.1)

### Promotion of DATA-01 + DATA-02

Update AUGMENTATIONS.md rows:
- DATA-01 status → `accepted-v0.0`
- DATA-02 status → `accepted-v0.0`
