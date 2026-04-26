# ADR-0021: MorphBPE evaluation gate for UA tokenization — promote DATA-10

- **Status**: proposed → accepted-v0.0 (this ADR; gate, not an
  immediate implementation commit)
- **Date**: 2026-04-25
- **Deciders**: Dmytro
- **Related**: AUGMENTATIONS row DATA-10, ADR-0004 (own BBPE 131K),
  ADR-0005 (13 langs incl UA), TOK-01..06,
  `docs/specs/v0.0.md` D0.0.1 (validation gate)
- **Sources**: [arXiv 2502.00894 (MorphBPE)](https://arxiv.org/abs/2502.00894),
  [PMC 11622830 (SKMT Slovak morpho BPE)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11622830/),
  [Frontiers 2025 Ukrainian tokenization study](https://www.frontiersin.org/journals/computer-science/articles/...)

## Context

ADR-0004 commits to own BBPE 131K trained on the 13-language mix.
The TOK-01 target is **UK fertility 1.65–1.75** (chars per token);
the V3 BPE baseline is 2.5-3.0, which would tokenize UK text ~2×
more than EN.

Open question: does plain BBPE on a UA-balanced corpus actually hit
that 1.65-1.75 target? Two related publications suggest a stronger
fix may be needed for morphologically-rich Cyrillic languages:

- **MorphBPE** ([arXiv 2502.00894](https://arxiv.org/abs/2502.00894))
  — morpho-aware BPE that respects morpheme boundaries. Tested on
  EN, RU, HU, AR; reduces cross-entropy loss and accelerates
  convergence at 300M and 1B scale. RU result is the closest
  proxy for UA (both East-Slavic, agglutinative-leaning, four-case
  Cyrillic).
- **SKMT Slovak morpho BPE** ([PMC 11622830](https://pmc.ncbi.nlm.nih.gov/articles/PMC11622830/))
  — Slovak-specific morphological BPE. Pattern replicable for UA.

But MorphBPE comes with costs:
- Requires a morphological analyzer or dictionary for the target
  language (UA-specific tooling needs to be vetted).
- Breaks compatibility with own BBPE 131K vocab — using MorphBPE
  for UA means either (a) merging MorphBPE-derived UA merges into
  the BBPE vocab, or (b) a separate UA tokenizer routed by
  language detection.
- Q1 2026 evidence is encouraging but not yet at frontier-model
  scale — MorphBPE's largest validation is 1B.

The **right move is a gate**, not an unconditional commit:
1. Train plain BBPE 131K (D0.0.1) and measure UA fertility on the
   held-out validation set.
2. If UA fertility ≤ 1.75 → done, MorphBPE not needed.
3. If UA fertility > 1.85 → MorphBPE evaluation triggered before
   v0.0 production tokenizer is finalized.
4. If 1.75 < UA fertility ≤ 1.85 → grey zone; user decides whether
   to invest MorphBPE engineering or accept slightly above-target
   fertility.

## Decision

Defer the MorphBPE adoption decision to a measurement gate during
D0.0.1 (tokenizer training). The gate is:

| UA fertility | Action |
|---|---|
| ≤ 1.75 | **Adopt plain BBPE 131K** as v0.0 final tokenizer. MorphBPE remains research-track for v0.2+. |
| 1.75 < f ≤ 1.85 | **User decision**. Default: accept, defer MorphBPE to v0.1. Escalation: if v0.0 multilingual eval (xnli-uk, HellaSwag-uk) regresses materially vs RU, revisit. |
| > 1.85 | **MorphBPE evaluation required before v0.0 production tokenizer is finalized.** Implementation: integrate a UA morphological analyzer, retrain BBPE with morpho-aware merges, re-measure. If MorphBPE fertility ≤ 1.75, adopt it. If MorphBPE doesn't break ≤ 1.75, accept the higher plain-BBPE fertility and add a note to v0.1 plan that UA tokenizer needs deeper rework. |

The decision is recorded in `docs/reports/v0.0-tokenizer-report.md`
when D0.0.1 completes, with the measured per-language fertility
table, the gate outcome, and the path taken.

## Consequences

**Intended:**
- Plain BBPE 131K is the default; MorphBPE only triggers if
  measured fertility actually exceeds the target. Avoids
  premature engineering work.
- UA quality story is empirically gated: we know whether plain
  BBPE meets the target, and we have a documented escalation
  path if it doesn't.
- The TOK-01 1.65-1.75 target stays load-bearing — it's not
  aspirational; it's the gate criterion.

**Unintended / accepted:**
- If the gate triggers (>1.85), v0.0 timeline slips by 1-2 weeks
  for MorphBPE integration. Acceptable: tokenizer is permanent
  per ADR-0004; getting it right is more important than shipping
  fast.
- Grey-zone (1.75-1.85) creates a user-decision moment.
  Documented; not blocking.
- MorphBPE adoption (if triggered) breaks compatibility with the
  default BBPE vocab; we'd be shipping a hybrid tokenizer with
  UA-specific morpho merges layered on top. Maintenance overhead.

**Explicit non-effects:**
- Does NOT commit to MorphBPE up-front.
- Does NOT change UA upsample plan (LANG-02, ADR-0005); upsample
  is independent of merge algorithm.
- Does NOT block v0.0 start; tokenizer training proceeds with
  plain BBPE 131K, gate runs at end of D0.0.1.

## Alternatives considered

- **Adopt MorphBPE unconditionally for UA.** Rejected for v0.0:
  no measurement evidence yet that plain BBPE fails the target.
  Premature complexity.
- **Skip MorphBPE entirely; accept whatever fertility plain BBPE
  produces.** Rejected: TOK-01 target is load-bearing for the
  UA quality story (ADR-0005). If plain BBPE produces 2.0+
  fertility, accepting it silently degrades the project's stated
  multilingual goal.
- **Use a separate UA-only tokenizer and route by language
  detection.** Considered. More complex (two tokenizers, language
  detector) and breaks the unified-vocab-for-13-languages
  story. Defer to v0.1+ if hybrid BBPE+MorphBPE proves
  insufficient.
- **Use BLT (byte-patch tokenizer, no tokenizer at all)
  instead.** Considered (TOK-07). Too disruptive at v0.0;
  defer to v1.0 consideration.
- **Use a per-language fertility threshold, not just UA.** The
  same gate applies in principle for other low-resource langs
  (HI, AR). v0.0 focuses gate on UA because UA is the project's
  flagged distinctiveness; other languages get measured in the
  D0.0.1 fertility table but don't trigger MorphBPE evaluation
  unless egregiously bad (>2.5).

## Implementation notes

### Files affected (during D0.0.1)

- `experiments/train_tokenizer.py` (existing CLI):
  - On completion, computes per-language fertility on held-out
    validation samples.
  - Emits `runs/v0.0/tokenizer-fertility.json` with per-language
    numbers.
- `experiments/check_morphbpe_gate.py` (new):
  - Reads `tokenizer-fertility.json`, applies gate logic, prints
    decision.
- New (only if MorphBPE gate triggered):
  `packages/data/src/saint_llm_data/morphbpe/`:
  - `ua_morpho_analyzer.py` — wraps a UA morphological tool
    (candidates: pymorphy3-uk, UDPipe with UA model)
  - `morpho_aware_bpe.py` — wraps `tokenizers` library with
    morpho-segmented input
  - `train_morphbpe_ua.py` — re-trains UA-merge subset with
    morpho-aware merges, layers onto base BBPE vocab

### Validation gate

D0.0.1 exit checklist (extend `docs/specs/v0.0.md` D0.0.1):

```
[ ] Plain BBPE 131K trained on v0.0 corpus mix
[ ] Per-language fertility measured on held-out 10K-sentence sample
[ ] tokenizer-fertility.json committed with the trained model
[ ] Gate outcome documented in tokenizer-report.md:
    - UA fertility measured: <number>
    - Path taken: [plain BBPE | grey-zone accept | MorphBPE evaluation]
    - If MorphBPE: results of MorphBPE retrain + final fertility
```

### Tests

- `packages/data/tests/test_fertility_measurement.py`:
  - Compute fertility on a known-text sample with a known
    tokenizer; assert deterministic output.
- (Conditional) `packages/data/tests/test_morphbpe_ua.py`:
  - Only added if gate triggers MorphBPE work.

### Out of scope for this ADR

- BLT byte-patch tokenizer (TOK-07, deferred per AUGMENTATIONS).
- Per-language tokenizer routing (v0.1+ if gate path requires).
- Other-language morpho-aware BPE (HI, AR) — same gate logic
  applies in principle but only triggers at egregious fertility
  (>2.5).
- Frontiers 2025 UK study replication beyond fertility
  measurement — research track for UNLP 2026 submission.

### Promotion of DATA-10

Update AUGMENTATIONS.md row DATA-10 status to
`accepted-v0.0 (gate)` — accepted as a *measurement gate*, not
as an implementation commit. Status will be re-evaluated at
D0.0.1 completion.
