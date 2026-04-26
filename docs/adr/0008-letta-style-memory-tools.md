# ADR-0008: Letta-style verb-oriented memory tool surface

- **Status**: accepted
- **Date**: 2026-04-25 (backfilled from 2026-04-15 design)
- **Deciders**: Dmytro
- **Related**: MEM-01..MEM-09 in `docs/AUGMENTATIONS.md`, ADR-0003,
  AGT-01 (Code Mode supersedes DSML XML for tool-call envelope)

## Context

Memory architecture for an agentic LLM has many options:

- Cross-attention to a vector DB (RAG-style).
- Neural memory module (Titans, Engram).
- Tool-mediated memory (`<|memory_recall|>`, `<|memory_save|>` etc.)
  with backend-agnostic verb interface.

Anthropic's memory tool, Letta v1, and Mem0 / Zep all converged on
the **tool-mediated verb-oriented** surface: the model emits memory
operations via tool calls; an external runtime executes them
against whatever backend (vector / graph / FS / DB).

This decouples memory mechanism from model weights — the same model
can use any memory backend without retraining.

## Decision

Adopt verb-oriented memory tools as the v0.1 baseline:
- Special tokens `<|memory_recall|>`, `<|memory_save|>`,
  `<|memory_result|>` reserved in tokenizer (TOK-04, MEM-01)
- CoALA-aligned memory-type tags: episodic / semantic / procedural /
  working / core / archival (MEM-02)
- In-context injection between `<|memory_result|>` markers (MEM-04)
- Three-layer fallback: miss / error / identity-passthrough (MEM-05)
- 8-12% post-training tokens are memory-using trajectories (MEM-06)
- Reserved residual side-channel slot per block for v0.2
  Titans/MIRAS neural memory (MEM-03), gated off / identity-init in
  v0.1
- Engram (DeepSeek + Peking U, Jan 2026) added as MEM-10 proposal
  for v0.3 — orthogonal third sparsity axis on top of MoE

## Consequences

**Intended:**
- Backend-agnostic: vector / graph / FS swap without retrain.
- Composes cleanly with V4 CSA (no cross-attention added).
- v0.1 ships as "1M context carries it" — no memory backend
  required at minimum viable product.
- Future-proof: Engram (MEM-10) and Titans-style neural memory
  (MEM-07) plug into reserved slots without touching v0.1 weights.

**Unintended / accepted:**
- Memory quality bounded by external backend quality.
- Tool-call latency on every memory access; mitigated in v0.3 by
  Code Mode (AGT-01) batching.
- Post-training data must include memory-using trajectories
  (MEM-06) — adds data-pipeline work in v0.2.

## Alternatives considered

- **Cross-attention to vector DB** at every layer. Rejected:
  conflicts with V4 CSA's causal sparse top-k design (MEM-09 logic).
- **Neural memory only (Titans)** as primary. Deferred to v0.2 — the
  benchmark wins to justify checkpoint surgery aren't established
  yet.
- **HippoRAG-2 / Letta v1 / Anthropic memory tool exclusively**
  without reserving Engram / Titans slots. Rejected: closes the
  door on neural-memory experiments in v0.2+.
- **Mamba/SSM/RWKV-style replacement for HCA.** Rejected per MEM-09.

## Implementation notes

- Memory tokens: reserved in TOK-04 control slots
- Side-channel slot: implemented in `core/multimodal/hooks.py`
  `ResidualSideChannel` with `alpha=0` default (identity at v0.1)
- Memory tool runtime: not yet built; v0.3 deliverable per ROADMAP
- Anthropic memory tool pattern (file-based, persistent across
  compaction) is MEM-11 — proposed for v0.3 as the concrete
  implementation choice
