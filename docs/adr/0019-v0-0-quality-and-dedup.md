# ADR-0019: v0.0 quality classifier + dedup tooling — promote DATA-03, DATA-04

- **Status**: proposed → accepted-v0.0 (this ADR)
- **Date**: 2026-04-25
- **Deciders**: Dmytro
- **Related**: AUGMENTATIONS rows DATA-03, DATA-04, LANG-05 (UK
  3-stack MT-pollution detector), `docs/specs/v0.0.md` D0.0.2
- **Sources**: [arXiv 2505.05427 (Ultra-FineWeb)](https://arxiv.org/html/2505.05427v1),
  [arXiv 2411.04257 (LSHBloom)](https://arxiv.org/html/2411.04257v3),
  [arXiv 2412.02595 (Nemotron-CC classifier ensemble)](https://arxiv.org/abs/2412.02595)

## Context

Once v0.0 corpus slices are chosen (ADR-0018), two pipeline stages
matter most for output quality at low cost: **quality filtering**
and **deduplication**. Q1 2026 frontier evidence:

### Quality classification

- **Ultra-FineWeb fastText classifier** ([arXiv 2505.05427](https://arxiv.org/html/2505.05427v1))
  — small fastText (256-d vectors, n=3 char ngrams, 3 epochs)
  trained on a 1B-param model's perplexity-annealed labels.
  Orders of magnitude cheaper than LLM-classifier-based filtering
  (Nemotron-CC's classifier ensemble runs Mistral-8x22B + Nemotron-
  340B + DCLM ensembled at ~$1k+ per 1T tokens; fastText is ~$10
  per 1T tokens). Outputs Ultra-FineWeb-en (~1T) + Ultra-FineWeb-zh
  (~120B) at quality matching the costlier classifiers.
- **Nemotron-CC classifier ensemble** — three classifiers
  (Mistral-8x22B-prompted, Nemotron-340B-prompted, DCLM) with
  Snowflake-arctic-embed-m as feature extractor. Higher quality at
  10-100× cost.
- **DCLM-baseline classifier** — older, weaker than Ultra-FineWeb.

For saint-llm's open-only commitment + budget reality, fastText is
the right pick. Ultra-FineWeb is the published reference;
Nemotron-CC v2 already includes Nemotron's classifier output, so
we get the ensemble effect on Nemotron data slices for free.

### Deduplication

- **LSHBloom** ([arXiv 2411.04257](https://arxiv.org/html/2411.04257v3))
  — Bloom-filter approximation of MinHash-LSH. **12× faster than
  MinHash-LSH** at controllable accuracy loss. Internet-scale
  dedup at lower cost. Modern reference.
- **MinHash-LSH** (datatrove default) — slower, well-validated,
  lossless reference.
- **SemHash** ([github.com/MinishLab/semhash](https://github.com/MinishLab/semhash))
  — semantic dedup via embedding similarity. Useful for catching
  paraphrased duplicates that pure-syntactic MinHash misses; more
  expensive.
- **D4** (NeurIPS 2023, arXiv 2308.12284) — canonical dedup +
  diversity reference; 18-20% efficiency gain on validation
  perplexity. Slower than MinHash-LSH; usually combined with it.

### MT pollution (UK-specific)

LANG-05 is a 3-stack detector for UK MT-pollution: translationese
classifier + KenLM perplexity + URL heuristics. This is its own
filter stage applied only to UA shards.

## Decision

Adopt this filter+dedup pipeline for v0.0 (D0.0.2):

1. **Quality classification — Ultra-FineWeb fastText** for EN +
   non-Nemotron-CC sources. Drop bottom-quartile per-language.
   Nemotron-CC v2 data passes through unchanged (already has
   ensemble classifier applied upstream). UA data: skip
   classifier (Kobza/UberText already curated) but apply
   LANG-05 3-stack MT-pollution detector.

2. **Dedup — LSHBloom** for primary near-duplicate dedup across
   the full mix. Configuration: 256-permutation MinHash-equivalent,
   Bloom filter sized for our ~170B-token corpus
   (~10⁹ documents). Per-language pass first, then cross-language
   pass against EN to catch translation duplicates.

3. **MT pollution detector — LANG-05 3-stack on UA only.** Stages:
   - Translationese classifier (small fastText trained on
     UA-original vs UA-translated paired data)
   - KenLM perplexity threshold (UA-native KenLM)
   - URL heuristics (drop docs from known MT-source domains)

4. **Document-level dedup only**, not n-gram-level. N-gram dedup is
   v0.1+ research if v0.0 perplexity shows over-fitting on
   repeated phrases.

5. **No semantic dedup at v0.0.** SemHash is on the wishlist for
   v0.1; v0.0 budget doesn't justify its cost (additional
   embedding pass on 170B tokens).

Pipeline order:

```
[per-slice loader]
  → [quality filter] (Ultra-FineWeb fastText if applicable)
  → [LANG-05 MT-pollution] (UA only)
  → [LSHBloom dedup] (per-language, then cross-language)
  → [tokenize via v0.0 tokenizer]
  → [pack into 4096-token windows]
  → [shard write as parquet, ~200 MB shards]
```

## Consequences

**Intended:**
- Filter+dedup cost stays in budget: fastText is ~$10-50 per 1T
  tokens; LSHBloom 12× cheaper than MinHash-LSH. v0.0 pipeline
  runs on a single workstation in days, not weeks.
- UA quality protected by 3-stack pollution detector — the most
  important output for our 13-language differentiation.
- Nemotron-CC v2 data flows through without redundant
  classification (it's already filtered upstream).

**Unintended / accepted:**
- LSHBloom's "controllable accuracy loss" means a small fraction
  of true duplicates pass through (Bloom false negatives) and a
  small fraction of non-duplicates get flagged (false positives).
  Acceptable trade for 12× speedup on internet-scale data.
- fastText classifier produces a single quality score — no
  per-domain (math/code/encyclopedic) gradation. v0.1 may need
  domain-specific classifiers.
- No semantic dedup means paraphrased duplicates pass through;
  acceptable at v0.0, address at v0.1 if needed.

**Explicit non-effects:**
- Does NOT change corpus selection (ADR-0018).
- Does NOT change tokenizer or pretraining recipe.
- Does NOT touch multimodal data filtering (DATA-09 unified
  multimodal classifier is v0.1; separate ADR).

## Alternatives considered

- **Nemotron-CC ensemble classifier (Mistral-8x22B + Nemotron-340B
  + DCLM).** Strictly higher quality but 10-100× cost. Not
  budget-justified at v0.0; v0.1 may revisit if a quality
  ceiling is hit.
- **MinHash-LSH (datatrove default)** instead of LSHBloom.
  Slower, lossless. Considered as fallback if LSHBloom shows
  unacceptable false-negative rate. v0.0 ships LSHBloom; if
  v0.0 close report shows residual duplicate problem,
  back-fill MinHash-LSH for v0.1.
- **D4 dedup+diversity** as primary. Slower; usually combined
  with MinHash-LSH. v0.0 sticks with LSHBloom only; D4
  is v0.1 polish.
- **SemHash (semantic dedup)** as primary or stacked.
  Embedding-pass over 170B tokens at v0.0 is a budget overrun.
  Defer to v0.1.
- **No dedup.** Rejected — published evidence (D4, MinHash
  studies) consistently shows 18-20%+ efficiency loss without
  dedup.
- **Domain-specific classifiers per language.** Considered.
  Ultra-FineWeb has en + zh classifiers only; for the other
  11 languages we'd need to train our own. v0.0 uses single-
  language classifier where available, skips quality filter for
  others (HPLT 3.0 already does substantial filtering upstream).
  Per-language classifier training is a v0.1 deliverable if
  perplexity gap shows.

## Implementation notes

### Files affected (D0.0.2 pipeline implementation)

- `packages/data/src/saint_llm_data/pipeline/v0_0.py`:
  - `quality_filter_step` — wraps Ultra-FineWeb fastText
  - `mt_pollution_step` — wraps LANG-05 3-stack (UA only)
  - `dedup_step` — wraps LSHBloom
- New helpers under `packages/data/src/saint_llm_data/quality/`:
  - `ultra_fineweb_classifier.py` — load + apply fastText classifier
  - `mt_pollution_uk.py` — translationese + KenLM + URL stages
- New helpers under `packages/data/src/saint_llm_data/dedup/`:
  - `lshbloom.py` — LSHBloom implementation or wrapper
- `pyproject.toml` (data package): add `fasttext` (or fasttext-wheel),
  `kenlm` deps; LSHBloom typically rolls own (small implementation,
  dependency-free).

### Tests

- `packages/data/tests/test_ultra_fineweb_classifier.py`:
  - Score known-quality vs known-spam doc; assert quality > spam.
  - Smoke test runs in <1s on a few sample docs.
- `packages/data/tests/test_lshbloom.py`:
  - Two near-identical docs deduplicated; two unrelated docs
    kept.
  - False-negative rate measured on a 10K-doc benchmark; report
    in test output.
- `packages/data/tests/test_mt_pollution_uk.py`:
  - Known MT-translation UA text scored as polluted.
  - Native UA text scored as clean.

### Configuration knobs

```yaml
# configs/v0.0/data-pipeline.yaml
quality_filter:
  classifier: ultra_fineweb_fasttext
  threshold_quantile: 0.25  # drop bottom 25% per-language
  apply_to_slices: [hplt, fineweb_edu, starcoder2]  # skip nemotron_cc, kobza, ubertext
mt_pollution:
  enabled_for: [uk]
  translationese_threshold: 0.6
  kenlm_perplexity_threshold: 200.0
dedup:
  algorithm: lshbloom
  num_permutations: 256
  bloom_size_estimate: 1.0e9
  cross_language_dedup: true
```

### Out of scope for this ADR

- Mixture weighting (DATA-05, separate ADR-0020).
- MorphBPE evaluation (DATA-10, separate ADR-0021).
- Multimodal data quality filter (DATA-09, v0.1).
- Domain-specific (math/code/encyclopedic) classifiers (v0.1+).
- N-gram-level or semantic dedup (v0.1+ if needed).

### Promotion of DATA-03 + DATA-04

Update AUGMENTATIONS.md rows:
- DATA-03 status → `accepted-v0.0`
- DATA-04 status → `accepted-v0.0`
