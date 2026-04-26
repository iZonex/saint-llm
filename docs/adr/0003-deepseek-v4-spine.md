# ADR-0003: DeepSeek-V4 spine as architectural baseline

- **Status**: accepted
- **Date**: 2026-04-25 (backfilled from 2026-04-15 project bootstrap)
- **Deciders**: Dmytro
- **Related**: `docs/AUGMENTATIONS.md`, `docs/DeepSeek_V4.pdf`,
  `~/.claude/.../memory/project_v01_arch_decisions.md`

## Context

The project committed at bootstrap (April 2026) to using the DeepSeek-V4
architectural spine: CSA (Compressed Sparse Attention with Lightning
Indexer) + HCA (Heavy Compressed Attention) + SWAttention (sliding window) +
mHC (manifold-Constrained Hyper-Connections) + DeepSeekMoE (shared + routed
fine-grained experts) + Muon optimizer (hybrid Newton-Schulz) + MTP
(multi-token prediction) + FP4 QAT + 1M context target.

Alternatives considered at the time included Llama-style standard
MHA + dense FFN, Mistral-style sparse mixture-of-experts on top of MHA,
Mamba/RWKV state-space models, and pure linear-attention architectures.

## Decision

Use the DeepSeek-V4 spine as the architectural baseline. Augmentations
(multimodality, tokenizer, multilingual, memory, reasoning, post-training)
are layered on top of the spine and tracked individually in
`docs/AUGMENTATIONS.md`.

## Consequences

**Intended:**
- Strong frontier validation: DeepSeek V3 / V4 are SOTA-class open-weight
  reasoning models. The spine is not speculative — it has been trained at
  236B / 671B parameters with public weights.
- The spine includes integral support for sparse attention and MoE without
  retrofitting.
- 1M-context support is built-in via CSA's compression mechanism.

**Unintended / accepted:**
- Architectural complexity is significant (CSA + HCA + mHC interact in
  non-obvious ways). Higher engineering cost vs a vanilla MHA + dense
  baseline.
- Bridges from foreign weights (Llama, Gemma, Mistral) are not available;
  we pretrain from scratch. See ADR-0002.
- Hybrid linear+full attention research (Q1 2026 — Kimi Linear, Gated
  DeltaNet) post-dates the choice. Whether to revise the spine is open
  (see ARCH-01 in AUGMENTATIONS).

## Alternatives considered

- **Llama-style standard MHA + dense FFN.** Rejected: lacks built-in
  long-context efficiency; would require retrofitting sparse attention.
- **Llama + sparse MoE bolted on.** Rejected: routing + capacity tuning
  not as cleanly integrated as DeepSeekMoE shared+routed split.
- **Pure Mamba / RWKV / SSM-only.** Rejected: validated only ≤7B at the
  time of decision; hostile to MTP head; conflicts with V4 reasoning
  stack.
- **Hybrid Mamba + Transformer (Jamba-like).** Considered; deferred. Now
  resurfacing as ARCH-01 with stronger Q1 2026 evidence (Qwen3-Next, Kimi
  Linear).
- **Chameleon early fusion / Transfusion.** Rejected for multimodal
  augmentation reasons (see MM-V-09, MM-V-10).

## Implementation notes

- Spine implementation: `packages/core/src/saint_llm_core/`.
- AUGMENTATIONS.md tracks every deviation from a pure DeepSeek-V4
  reproduction.
- Open question: ARCH-01 hybrid linear+full attention may revise this ADR
  or supersede individual spine components. New ADR required if accepted.
