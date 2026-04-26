# ADR-0017: Adaptive thinking effort head supersedes static V4 Non/High/Max — promote RL-07, supersede REA-05

- **Status**: proposed → accepted-v0.2 (this ADR)
- **Date**: 2026-04-25
- **Deciders**: Dmytro
- **Related**: AUGMENTATIONS rows RL-07 (this ADR promotes),
  REA-01 (V4 Think modes baseline, preserved),
  REA-05 (`<budget:adaptive>` token, this ADR supersedes)
- **Sources**: [Anthropic Claude Opus 4.6 system card](https://www.anthropic.com/claude-opus-4-6-system-card),
  [Claude Opus 4.7 adaptive thinking writeup](https://medium.com/@dalio8/claude-opus-4-7-3e1e14a8a3c3),
  [Anthropic effort parameter docs](https://docs.litellm.ai/docs/providers/anthropic_effort),
  [arXiv 2603.07915 (Ares)](https://arxiv.org/html/2603.07915v1),
  [arXiv 2505.11274 (SelfBudgeter)](https://arxiv.org/html/2505.11274v2)
- **Supersedes:** REA-05 (`<budget:adaptive>` token)

## Context

The V4 spine ships three reasoning effort modes:
- **Non-think** — direct answer, no CoT
- **High** — moderate CoT
- **Max** — extensive CoT with self-verification

The original AUGMENTATIONS plan also included REA-05: a special
`<budget:adaptive>` token the user could prepend to delegate effort
choice to the model itself.

Q1 2026 frontier evidence:

- **Anthropic deprecated explicit budget tokens entirely.** Claude
  Opus 4.7 (April 2026) deprecated `extended_thinking_budget`,
  `temperature`, `top_p`, `top_k`. They now return 400 errors. The
  new API surface is `effort` ∈ {low, medium, high, xhigh, max},
  and the *model* learns to use the right depth conditioned on
  the effort tier.
- **Ares (arXiv 2603.07915, Q1 2026)** showed per-step learned
  routers beat both static-low (which underthinks hard problems)
  and random-selection routers; learned > heuristic.
- **SelfBudgeter** (arXiv 2505.11274) showed pre-estimating budget
  per query then GRPO with budget-guidance produces well-calibrated
  effort.
- The frontier direction: **learned, query-conditioned effort**
  trained jointly with RL — not user-specified or tag-driven.

The static V4 Non/High/Max tag is a coarse user-facing abstraction.
It needs to be backed by a learned head that, conditioned on the
prompt + the effort tier, produces the right reasoning depth. The
`<budget:adaptive>` token in REA-05 was a stab at this — but it's
a single discrete token that doesn't compose with the effort tiers
and doesn't get RL-trained jointly.

## Decision

Replace the `<budget:adaptive>` token (REA-05) with a **learned
adaptive thinking effort head** trained jointly during v0.2 RL.
Specifically:

### Architecture

1. New module `packages/core/src/saint_llm_core/multimodal/reasoning_head.py`
   (lives in `multimodal/` alongside the other reserved auxiliary
   heads; despite the name, the file holds reasoning-effort
   plumbing — rename later if confusing).

2. `EffortRouter` — small MLP head reading the embedding of the
   query (mean-pooled prompt embeddings before the first generated
   token) and the user-specified effort tier (5-way one-hot:
   {low, medium, high, xhigh, max}). Outputs:
   - **Predicted CoT length budget** (continuous, in tokens)
   - **Predicted termination probability per generation step** (a
     sigmoid head used during sampling)

3. The model emits a special `<effort:N>` token at the start of
   the assistant turn, where N ∈ {0..4} encodes the selected tier.
   The router's predicted budget conditions sampling: termination
   probability rises as the cumulative reasoning length approaches
   the budget.

### Training

- **v0.0 + v0.1**: model learns to *recognize* the `<effort:N>`
  token via supervised pretraining data containing CoT traces of
  varying depth tagged with effort levels. (Synthetic; built from
  existing reasoning corpora — REA-06 frontloaded data.)
- **v0.2**: learned effort head trained jointly with RL.
  Reward function adds a budget-respect term: penalize completions
  that exceed the predicted budget by 2×, reward those that
  produce correct answers within or under budget.
- **Length-controlled GRPO** (REA-04, Kimi K1.5) is the
  underlying length-penalty mechanism the budget term builds on.

### API surface

External (matching the Claude 4.7 convention for industry
familiarity):

```python
model.generate(
    prompt=...,
    effort="medium",  # or "low" | "high" | "xhigh" | "max"
)
```

Internal: the `effort` string maps to an integer tier,
prepended as `<effort:N>` to the assistant turn at sampling time.

## Consequences

**Intended:**
- Cost-quality trade is *learned*, not hand-coded:
  - `low`: matches v0.1 base direct-answer latency
  - `medium`/`high`: moderate CoT
  - `xhigh`/`max`: extensive CoT with self-verification
- Aligns with frontier API convention (Anthropic, OpenAI both
  using `effort`-style tiers in 2026).
- Joint RL training avoids the "model learned to ignore the tag"
  pathology: the budget head is in the loop with the policy.
- Composes with adaptive thinking (REA-05 was a single-token
  variant; this is the full implementation).

**Unintended / accepted:**
- Adds a small auxiliary head (~hidden_dim → 64 → 2 outputs).
  Negligible parameter cost (~50K params at 768 hidden).
- Adds RL reward shaping complexity; budget-respect term must be
  weighted carefully (too high → underthinking on hard problems;
  too low → effort head ignored).
- Five-tier API is more complex than three-tier (V4 baseline
  Non/High/Max). Defensible — Claude went from 4 → 5 specifically
  for `xhigh` granularity.
- REA-05 `<budget:adaptive>` token slot becomes dead; no embedding
  cost (lazy slot allocation per TOK-04).

**Explicit non-effects:**
- Does NOT replace the adaptive thinking *concept* — only the
  implementation. The architectural reservation (special tokens,
  budget control) was the right plan; execution is now via a
  learned head.
- Does NOT change V4 Non/High/Max (REA-01) — the three V4 tiers
  map to {low, high, max} of the new five-tier API. {medium,
  xhigh} are saint-llm additions matching Claude 4.7.
- Does NOT change reasoning data plan (REA-06 frontloaded CoT
  injection ~10% during pretraining, unchanged).

## Alternatives considered

- **Keep static V4 Non/High/Max tags only.** Rejected: leaves the
  Claude 4.7 frontier capability on the table.
- **Keep `<budget:adaptive>` as a single token.** Rejected:
  doesn't compose with effort tiers; not RL-trained jointly.
- **Continuous user-controlled budget (Anthropic's old API).**
  Rejected: Anthropic just deprecated this. Industry direction
  is learned-effort.
- **Three-tier {low, high, max} only (drop medium / xhigh).**
  Considered. V4 baseline. Rejected for parity with Claude 4.7
  five-tier. Five tiers cost ~nothing and give finer cost-quality
  control.
- **Train at v0.1 instead of v0.2.** Rejected: requires RL signal,
  which only lands at v0.2. v0.1 ships with the architecture
  ready (token slots reserved, head exists at zero-init); v0.2
  RL trains the head.

## Implementation notes

### Files affected

- New: `packages/core/src/saint_llm_core/multimodal/reasoning_head.py`
  with `EffortRouter` module and `EffortConfig`.
- `packages/core/src/saint_llm_core/model.py`:
  - Add `self.effort_router: EffortRouter` to `SaintLLM`.
  - In `forward`, when `effort` provided, mean-pool the query
    embedding and feed through router for budget prediction.
  - Budget prediction surfaces in the output dict for sampler
    consumption.
- `packages/inference/src/saint_llm_inference/generate.py`:
  - Sampler accepts `effort: int | str` argument.
  - Effort token prepended to assistant turn.
  - Budget-aware termination probability gates `eos_token`
    emission once cumulative length nears budget.
- `packages/posttraining/src/saint_llm_posttraining/grpo.py`:
  - Optional `budget_respect_weight: float = 0.1` reward-shaping
    knob.
  - When set, advantage adjusted by penalty for exceeding
    router-predicted budget.

### Tests

- `packages/core/tests/test_effort_router.py`:
  - Forward pass produces budget + termination prob outputs of
    correct shape.
  - Five-tier input produces five distinct outputs (sanity).
- `packages/inference/tests/test_effort_sampling.py`:
  - `effort="low"` produces shorter average output than
    `effort="max"` (probabilistic; relative tendency).
  - `effort` prepended as `<effort:N>` token in generated
    sequence.
- `packages/posttraining/tests/test_budget_respect.py`:
  - Budget-respect term: completions exceeding 2× predicted
    budget receive negative advantage adjustment.

### Tokenizer slots needed

Five effort tokens: `<effort:0>` .. `<effort:4>`. Reserve in
TOK-04 lazy-allocation range. Five embedding rows allocated when
first used.

### Out of scope for this ADR

- Curriculum-aware budget scheduling (arXiv 2604.19780, REA-13
  related research-track).
- BudgetThinker dual-objective formulation (arXiv 2508.17196).
  Defer; SelfBudgeter recipe is simpler and sufficient.
- Effort tier transfer between specialist and generalist models
  (v0.3 specialist+OPD merge consideration).

### Promotion / supersession

- Promote RL-07 from `proposed` to `accepted-v0.2` in
  AUGMENTATIONS.md.
- Supersede REA-05: change row status to
  `revised (ADR-0017): superseded by RL-07 learned effort head`.
- REA-01 (V4 Think modes baseline) stays `accepted-v0.1` —
  underlying think-mode framing is preserved; this ADR only
  changes the *runtime control mechanism*.
