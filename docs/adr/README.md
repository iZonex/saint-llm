# Architecture Decision Records

Each ADR captures one significant decision: context, options weighed,
choice, consequences. Follow `0000-template.md` for new ones.

## Index

- [0001 — Versioning philosophy: incremental, not narrowed](0001-versioning-philosophy.md) — **proposed**
- [0002 — No Gemma 4 / HF-weights bridge](0002-no-gemma-bridge.md) — **accepted**
- [0003 — DeepSeek-V4 spine as architectural baseline](0003-deepseek-v4-spine.md) — **accepted**
- [0004 — Own BBPE 131K tokenizer](0004-own-bbpe-131k-tokenizer.md) — **accepted**
- [0005 — 13 languages with UA upsample](0005-13-languages-with-uk-upsample.md) — **accepted**
- [0006 — uv workspace monorepo](0006-uv-monorepo.md) — **accepted**
- [0007 — Native multimodal pretraining](0007-native-multimodal-pretrain.md) — **accepted**
- [0008 — Letta-style memory tool surface](0008-letta-style-memory-tools.md) — **accepted**
- [0009 — GRPO baseline + Q1 2026 revisions (GSPO/DAPO/Critique-GRPO)](0009-grpo-baseline-and-q1-2026-revisions.md) — **accepted (baseline) + proposed (revisions)**
- [0010 — MuonClip QK-Clip for v0.0 (promotes OPT-01)](0010-muonclip-qk-clip-for-v0-0.md) — **accepted-v0.0**
- [0011 — u-μP init for v0.0 (promotes OPT-02)](0011-umup-init-for-v0-0.md) — **accepted-v0.0**
- [0012 — Logit softcap + sink tokens for v0.0 (promotes ARCH-02)](0012-logit-softcap-and-sink-tokens-for-v0-0.md) — **accepted-v0.0**
- [0013 — Code Mode supersedes DSML XML (promotes AGT-01, supersedes ETC-02)](0013-code-mode-supersedes-dsml-xml.md) — **accepted-v0.3**
- [0014 — GSPO replaces token-level GRPO ratio for MoE (promotes RL-01)](0014-gspo-replaces-token-level-grpo-ratio.md) — **accepted-v0.2**
- [0015 — DAPO four patches on top of GRPO (promotes RL-02)](0015-dapo-four-patches.md) — **accepted-v0.2**
- [0016 — Critique-GRPO for plateau breaking (promotes RL-04)](0016-critique-grpo-for-plateau-breaking.md) — **accepted-v0.2**
- [0017 — Adaptive thinking effort head (promotes RL-07, supersedes REA-05)](0017-adaptive-thinking-head.md) — **accepted-v0.2**
- [0018 — v0.0 multilingual corpus mix (promotes DATA-01, DATA-02)](0018-v0-0-corpus-mix.md) — **accepted-v0.0**
- [0019 — v0.0 quality classifier + dedup (promotes DATA-03, DATA-04)](0019-v0-0-quality-and-dedup.md) — **accepted-v0.0**
- [0020 — RegMix mixture optimization (promotes DATA-05)](0020-regmix-mixture-optimization.md) — **accepted-v0.0**
- [0021 — MorphBPE evaluation gate (promotes DATA-10 as gate)](0021-morphbpe-evaluation-gate.md) — **accepted-v0.0 (gate)**

## Status legend

- `proposed` — drafted, awaiting review
- `accepted` — agreed, in effect
- `rejected` — explicitly decided against
- `deferred` — agreed to revisit at a future version
- `superseded` — a later ADR replaces this one (link forward)

## Backfill complete (as of 2026-04-25)

ADR-0004..0009 backfilled from existing AUGMENTATIONS rows. New
decisions land as ADR-0010+.

## ADRs needed for new Q1 2026 proposals

When promoting a `proposed` augmentation to `accepted-vX.Y`, write an ADR.
Highest-priority Q1 2026 picks needing ADRs before implementation:

- OPT-01 MuonClip — proposed for v0.0
- RL-01 GSPO — proposed for v0.2
- AGT-01 Code Mode (replaces DSML XML) — proposed for v0.3
- MEM-10 Engram — proposed for v0.3
- DIST-05 Templar SparseLoCo — proposed for v0.4
- ARCH-01 Hybrid linear+full attention — pending review for v0.0/v0.1
