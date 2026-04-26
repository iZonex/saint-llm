# ADR-0001: Versioning philosophy — incremental, not narrowed

- **Status**: proposed
- **Date**: 2026-04-25
- **Deciders**: Dmytro
- **Related**: `docs/ROADMAP.md`, `~/.claude/.../memory/feedback_no_scaffolds_on_scaffolds.md`

## Context

Through 2026-04-15 to 2026-04-25 the project accumulated a large amount of
unit-tested scaffolding (architecture, quantization, KV cache, SFT/GRPO
math, deterministic attention, multi-agent runtime v0.0) without a single
end-to-end run on real data. The earlier `project_milestone1.md` memory
declared "code-side READY" while in reality:

- No `tokenizer.json` was ever produced from a real corpus.
- No real text data pipeline (datatrove dedup/filter/lang-balance) existed.
- Wandb was a dependency but not wired.
- Eval was perplexity on a 4-line synthetic corpus.
- ModalityProjector existed but never saw a real SigLIP/Whisper output.
- Multi-agent / tool-use / reasoning-modes / GRM / OPD / sandbox were
  empty namespaces.
- No realistic loss reading existed.

The user called this out explicitly on 2026-04-25 ("какого хрена ты что-то
обучать собрался?", "у нас кучи всего нету что мы обучаем") and rejected
both narrowing scope to text-only forever AND rebuilding scaffolds without
validating the foundation.

## Decision

Adopt **incremental ordered versioning** (v0.0 → v0.1 → ... → v1.0) where
each version produces an end-to-end validatable artifact on real data, and
no version's scope reduces the project's overall ambition. Specifically:

- **v0.0** — text spine works on real text data, real eval benchmarks
- **v0.1** — multimodal native pretrain (vision + audio + video) end-to-end
- **v0.2** — post-training, reasoning, specialists
- **v0.3** — agent runtime + tool use integrated end-to-end
- **v0.4** — distributed + decentralized training
- **v1.0** — autonomous decentralized organism

Each version has measurable exit criteria. We do not skip ahead.

## Consequences

**Intended:**
- The foundation gets validated before being layered on.
- Every version produces an artifact (weights, eval report) the user can
  inspect.
- Scope-creep within a version is constrained by exit criteria.

**Unintended / accepted debt:**
- Multi-agent runtime (already scaffolded) sits unused until v0.3.
- SFT / GRPO / deterministic-attention math (already shipped) sits unused
  until v0.2.
- Total time to "BTC-style organism" (v1.0) is longer than the parallel-build
  path *would have been if it worked* — but the parallel path empirically
  did not produce working artifacts.
- Multimodal scaffolding (vision/audio modules) stays cold for v0.0.

**Explicit non-narrowing:**
- We are NOT dropping multi-agent / multimodal / decentralized from scope.
- We are NOT saying "text-only forever."
- We ARE saying "validate the spine on real text first, then layer."

## Alternatives considered

- **Narrow to text-only, ship a text LLM, call it done.** Rejected by user.
  Project goal is multimodal multi-agent.
- **Build all parallel tracks simultaneously without ordering.** Rejected
  because that's what produced the over-promised milestone state already.
  No track is validated end-to-end without the others.
- **Switch to Gemma 4 finetune.** Rejected 2026-04-25 by user. saint-llm is
  a from-scratch own-model.
- **Drop decentralized goal.** Rejected — it's the north star.

## Implementation notes

- The roadmap (`docs/ROADMAP.md`) lists exit criteria per version.
- Each `docs/AUGMENTATIONS.md` row gets a target version (`accepted-vX.Y`).
- Memory file `feedback_no_scaffolds_on_scaffolds.md` enforces this for
  future Claude sessions: no auto-piloting through scaffolds.
