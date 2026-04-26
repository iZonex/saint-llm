# ADR-0004: Own BBPE 131K tokenizer

- **Status**: accepted
- **Date**: 2026-04-25 (backfilled from 2026-04-15 project bootstrap)
- **Deciders**: Dmytro
- **Related**: TOK-01..TOK-06 in `docs/AUGMENTATIONS.md`

## Context

A frontier multimodal multi-agent model needs a tokenizer that handles
13 target languages (EN/ZH/RU/UK/ES/FR/DE/JA/KO/AR/HI/PT/IT) plus
reserved control slots for vision/audio/video/memory/reasoning markers.

Two paths existed at bootstrap:
1. Extend the DeepSeek V3 BPE tokenizer (~100K vocab) by adding new
   merges for under-represented languages and modality slots.
2. Train an own BBPE 131K from scratch on the target multilingual mix.

The Frontiers 2025 Ukrainian tokenization study showed V3 BPE has UK
fertility 2.5–3.0 vs target 1.65–1.75 (i.e. UK text tokenizes ~2×
worse than EN). This is a **permanent** weight-baked-in property —
not fixable post-pretrain without retraining.

## Decision

Train an own BBPE 131K on the target language mix with explicit:
- Vocab allocation: Latin 38, CJK 30, Cyrillic 12, Arabic 6,
  Devanagari 5, code/byte 9 (TOK-02)
- Force-include UK chars (ї, є, ґ, апостроф) + four-case Cyrillic
  (TOK-03)
- ~50K nominal control slots for modality markers, ~150 actually
  allocated v0.1 (TOK-04)
- Tied input/output embeddings (TOK-05)
- BBPE dropout=0.1 during training (TOK-06)

## Consequences

**Intended:**
- UK fertility lands in the 1.65–1.75 target range, matching EN cost.
- Reserved slots for v0.2 modalities (Janus VQ image gen, Mimi/Moshi
  audio codec, Cosmos video codec) are present from day one.
- Tied embeddings halve embedding cost and enable lazy slot allocation.

**Unintended / accepted:**
- Cannot warm-start from V3-trained weights (vocab mismatch).
  Combined with ADR-0002 (no Gemma bridge), saint-llm pays full
  pretrain cost from scratch.
- Tokenizer training itself requires real corpus access — a
  prerequisite to v0.0 (see ROADMAP exit criteria).

## Alternatives considered

- **Extend V3 BPE.** Rejected. UK fertility cannot be fixed by adding
  merges — it requires re-training from scratch on a UK-balanced corpus.
- **Reuse Gemma 4 / Llama 4 tokenizer (~128K).** Rejected. Pre-trained
  vocabulary distribution targets EN-heavy mix; UK + agglutinative
  languages still under-represented.
- **BLT byte-patch tokenizer (no tokenizer at all).** Deferred to v1.0
  consideration (TOK-07). Too disruptive at v0.0; track for future.

## Implementation notes

- Trainer: `packages/data/src/saint_llm_data/tokenizer_trainer.py`
  (already shipped; never run on real corpus)
- CLI: `experiments/train_tokenizer.py`
- v0.0 first run target: HuggingFaceFW/fineweb-edu + HPLT 3.0 +
  Kobza UK + Nemotron-CC v2 multilingual slice
- MorphBPE consideration (DATA-10) — evaluate but don't commit;
  may not justify breaking compatibility with own 131K vocab
