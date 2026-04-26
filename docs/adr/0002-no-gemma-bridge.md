# ADR-0002: No Gemma 4 / HF-weights bridge into saint-llm

- **Status**: accepted
- **Date**: 2026-04-25
- **Deciders**: Dmytro
- **Related**: `~/.claude/.../memory/project_trading_finetune.md`

## Context

Gemma 4 (Apache-2.0, April 2026 release) is a strong base-model candidate
for finetuning, including a planned trading-finetune in a separate
codebase. The question came up: should saint-llm bridge Gemma 4 weights —
i.e. translate Gemma's standard MHA + dense FFN weights into saint-llm's
CSA + HCA + mHC + DeepSeekMoE architecture as a warm-start?

## Decision

No. saint-llm trains its own weights from scratch on its own architecture.
Gemma 4 finetune work lives entirely in a separate codebase using HF
`transformers` directly with no integration into saint-llm.

## Consequences

**Intended:**
- saint-llm architecture stays clean — no compatibility layer for foreign
  weight shapes.
- Trading-finetune ships fast on Gemma 4 in parallel without blocking on
  saint-llm pretrain.
- Two clean separate codebases, two clean separate problems.

**Unintended / accepted:**
- saint-llm has no warm-start; we pay the full pretraining bill from
  scratch.
- Loss-of-information from any potential weight transfer is not available
  as an experiment.

## Alternatives considered

- **Bridge Gemma weights into saint-llm as a warm-start option** — rejected.
  Architectures don't map cleanly (standard MHA ≠ CSA + HCA + mHC + MoE
  indexer); translation would be lossy and add maintenance debt for no
  meaningful gain. Validated by external evidence: no frontier own-model
  project (DeepSeek, Kimi, Qwen, Mistral) bridges weights from another
  family at architecture-mismatch level.
- **Use Gemma 4 as the base entirely (no own model)** — rejected. saint-llm
  is a from-scratch own-model project. See `ROADMAP.md` Vision section.

## Implementation notes

- AUGMENTATIONS.md does NOT list any Gemma-bridge augmentation.
- saint-llm package code does NOT import `transformers.AutoModel.from_pretrained`
  for any Gemma family weights.
- Trading-finetune codebase is named separately and is out of scope here.
