# ADR-0016: Critique-GRPO for plateau-breaking on hard reasoning — promote RL-04

- **Status**: proposed → accepted-v0.2 (this ADR)
- **Date**: 2026-04-25
- **Deciders**: Dmytro
- **Related**: AUGMENTATIONS row RL-04, ADR-0014 (GSPO),
  ADR-0015 (DAPO patches), ADR-0009 (GRPO baseline)
- **Sources**: [arXiv 2506.03106](https://arxiv.org/abs/2506.03106)
  (Critique-GRPO, Cambridge / CUHK / SH-AI-Lab; v6 Feb 2026)

## Context

Even with GSPO (ADR-0014) + DAPO patches (ADR-0015), GRPO-family
RL plateaus on hard reasoning problems. The pathology: when most
rollouts in a group fail and the few successes are accidental
(reward-hacking via lucky tokens), the gradient signal points
toward "more lucky tokens" rather than "better reasoning."
Outcome-only rewards can't distinguish.

Critique-GRPO (Cambridge / CUHK / SH-AI-Lab, June 2025, v6 February
2026) introduces a second supervision signal: **natural-language
critique** of failed rollouts, then in-context refinement
*inside* the rollout loop. Mechanism:

1. Generate G rollouts as in standard GRPO.
2. For each failed rollout, run a critique step: a critic model
   (or the actor itself in self-critique mode) explains *why*
   the rollout failed in natural language.
3. Append the critique to the prompt and generate a refined
   rollout. The refined rollout is what gets graded for advantage
   computation.
4. Standard GRPO advantage + clipped surrogate + KL on the
   refined rollouts.

Reported gains on Qwen2.5-Math-7B and Qwen3-8B:
- AIME 2024: +16.7% Pass@1 vs vanilla GRPO
- Math-domain average: +15-22% Pass@1 across 4 reasoning
  benchmarks

The mechanism breaks the plateau because the critique provides
signal about *why* failures happened — not just that they did.
Effectively a cheap form of process-supervision derived from
outcome-supervision via the critic step.

## Decision

Add Critique-GRPO as an opt-in mode in the v0.2 RL stack. Specifically:

1. New module `packages/posttraining/src/saint_llm_posttraining/critique_grpo.py`
   with:
   - `CritiqueGRPOConfig` (extends `GRPOConfig` knobs):
     - `enabled: bool = False` (off by default; opt-in for hard
       domains)
     - `critique_threshold: float = 0.5` (rollouts with reward
       below this trigger critique)
     - `critique_max_attempts: int = 1` (1 = single re-rollout;
       higher = recursive critique, capped to avoid blow-up)
     - `critic_policy: Policy | None = None` (when None, use
       self-critique on the actor; when set, use a separate
       critic model)
   - `critique_grpo_step` function that wraps the standard GRPO
     rollout with the critique loop.

2. Composition with GSPO + DAPO: Critique-GRPO operates on the
   rollout *generation*; advantage / clip / aggregation downstream
   still use GSPO + DAPO (ADRs 0014, 0015). No conflict — the
   refined rollout is just a normal rollout from the loss's
   perspective.

3. Default usage: turn on for **specialist post-training**
   (Math, Code) where the failure mode is hard plateaus; keep off
   for general SFT-grade RL where it adds rollout cost without
   clear benefit.

## Consequences

**Intended:**
- Math + Code specialists break through the GRPO-with-DAPO plateau,
  achieving the published +15-22% Pass@1 gains on hard reasoning
  benchmarks.
- Free process-supervision signal without training a separate
  process reward model (PRM); critique is a cheap LLM call.
- Composes naturally with verbalized PRM (RL-10 / ThinkPRM, future
  ADR) — both produce text-form supervision; they could share
  prompts.

**Unintended / accepted:**
- Rollout cost increases: each failed rollout becomes 2× the LLM
  calls (original + refined). With `critique_max_attempts > 1`,
  worst-case 3-4×. Mitigated by `critique_threshold` filter (only
  failed rollouts trigger critique).
- Self-critique mode uses the actor itself; risks feedback loop
  where the actor critiques its own failures with the same blind
  spots. Separate critic model is safer but doubles model
  inference cost.
- Critique quality bounded by the critic. If the critic is a
  separate model, it's a hyperparameter that needs choosing
  (Qwen3 / DeepSeek V4 / our own v0.1 base).
- `critique_max_attempts > 1` enables recursive critique. Cap
  needed; published recipe uses 1.

**Explicit non-effects:**
- Does NOT change advantage computation, clip, KL, or aggregation
  (those are GRPO + DAPO + GSPO).
- Does NOT replace outcome reward with critique-derived reward.
  Critique only changes *which* rollouts get graded; the reward
  is still outcome-based (verifiable / ground-truth / GRM).
- Does NOT require a process reward model.

## Alternatives considered

- **Add a process reward model (PRM) instead.** Considered (RL-10
  / ThinkPRM, separate ADR). Verbalized PRM is the modern revival
  of PRM after DeepSeek's R1 dismissal. Critique-GRPO is cheaper
  in implementation and often sufficient on its own; a PRM is
  more expensive (extra trained model) but more general.
  saint-llm plans both: Critique-GRPO at v0.2, ThinkPRM-style
  PRM also at v0.2 for math / Lean specialists.
- **Reflexion at inference time only.** Doesn't change training;
  doesn't break the training plateau. Different problem.
- **Best-of-N with verifier filter.** Test-time augmentation;
  compatible with Critique-GRPO at training time. Orthogonal.
- **SPIRAL (self-play on zero-sum games)** as alternative
  plateau-break mechanism. Different approach (REA-12 in
  AUGMENTATIONS, research track for v0.2). Don't conflict;
  evaluate independently.
- **Always-on Critique-GRPO**. Rejected for v0.2: rollout cost
  doubles; not worth it for general post-training. Specialist-
  only is the right scope.

## Implementation notes

### Files affected

- New: `packages/posttraining/src/saint_llm_posttraining/critique_grpo.py`.
- `packages/posttraining/src/saint_llm_posttraining/__init__.py` —
  re-export `CritiqueGRPOConfig`, `critique_grpo_step`.
- `experiments/run_specialist_rl.py` (v0.2 deliverable) — wire
  the critique loop for Math / Code specialists.

### Tests

- `packages/posttraining/tests/test_critique_grpo.py`:
  - Mock policy + mock critic; rollout below threshold triggers
    critique; refined rollout used in advantage computation.
  - `enabled=False` matches plain GRPO behavior bit-for-bit.
  - `critique_max_attempts=2` correctly caps recursion.
  - `critic_policy=None` uses self-critique with the actor.
  - Self-critique vs separate critic produce different
    refinements on the same input (sanity).

### Critic prompt format (initial)

```
The following completion received a low reward. Diagnose the
specific reasoning error and explain what should have been done
differently. Be brief — 2-3 sentences.

Problem: <prompt>
Failed completion: <rollout text>
Reward: <numerical reward>

Critique:
```

The refined rollout prompt:

```
<original prompt>

A previous attempt failed because: <critique>

Try again, addressing the issue:
```

Format will be tuned during v0.2 implementation; this is the
documented starting point.

### Compute budget

For Math + Code specialists, expected rollout cost increase:
- Threshold = 0.5 (triggers on ~30-50% of rollouts in early
  training)
- max_attempts = 1
- ~1.4× rollout cost early; converges to ~1.1× as the model
  improves.

Acceptable for specialist runs (single domain, ~50B tokens
budget per ROADMAP v0.2 estimates).

### Out of scope for this ADR

- Verbalized PRM / ThinkPRM (separate ADR for RL-10).
- Tree-GRPO for agentic RL (separate ADR for v0.3).
- SPIRAL self-play (separate research-track ADR if accepted).
- Critic-model selection (Qwen3 vs own v0.1 base) — v0.2
  implementation decision.

### Promotion of RL-04

Update AUGMENTATIONS.md row RL-04 status to `accepted-v0.2`.
