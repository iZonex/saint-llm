# ADR-0015: DAPO four patches on top of GRPO core — promote RL-02

- **Status**: proposed → accepted-v0.2 (this ADR)
- **Date**: 2026-04-25
- **Deciders**: Dmytro
- **Related**: AUGMENTATIONS row RL-02, ADR-0014 (GSPO),
  ADR-0009 (GRPO baseline), `posttraining/grpo.py`
- **Sources**: [arXiv 2503.14476](https://arxiv.org/abs/2503.14476)
  (DAPO, ByteDance Seed + Tsinghua), [verl DAPO recipe](https://verl.readthedocs.io/en/latest/algo/dapo.html)

## Context

DAPO (Decoupled clip + dynamic sAmpling Policy Optimization) is the
ByteDance/Tsinghua paper that compiled four orthogonal patches on
top of vanilla GRPO. Reported headline: **AIME 2024 score 50 on
Qwen2.5-32B base, beating DeepSeek-R1-Zero-Qwen-32B's 47, with 50%
the steps**. Fully open-source on `verl`.

The four patches:

1. **Clip-Higher.** GRPO uses symmetric clip `(1-ε, 1+ε)` with
   ε=0.2. DAPO decouples upper and lower bounds: lower stays 0.2,
   upper goes to 0.28. Rationale: tightening downward keeps
   stability, but tightening upward stifles exploration of
   high-reward regions and contributes to entropy collapse.

2. **Dynamic Sampling.** Drop rollout groups where all G
   completions get the same reward (all-correct or all-wrong).
   Such groups have zero advantage variance → zero gradient
   contribution → wasted compute. Filter them out, generate
   replacements until the batch is full.

3. **Token-Level Policy Gradient Loss.** GRPO conventionally
   averages the surrogate over tokens then over sequences in a
   batch. DAPO sums tokens within a sequence (so longer responses
   contribute proportionally more), then averages across sequences.
   Fixes long-CoT under-weighting where short responses dominated
   the gradient signal.

4. **Overlong Reward Shaping.** Hard length cliff (reward = 0 at
   max_len) creates discontinuity. DAPO replaces with a soft band:
   gradual penalty starts at e.g. 95% of max_len and reaches max
   penalty at 100%. Smooth gradients near the boundary.

All four are 5-50 LOC additions, no algorithmic redesign, and
compose with both vanilla GRPO and GSPO (ADR-0014).

## Decision

Adopt all four DAPO patches in v0.2 as configurable knobs in
`GRPOConfig`:

```python
@dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 8
    clip_eps: float = 0.2                   # legacy symmetric clip
    clip_eps_upper: float = 0.28            # DAPO Clip-Higher
    clip_eps_lower: float = 0.20            # DAPO Clip-Higher
    use_clip_higher: bool = True            # DAPO patch 1
    dynamic_sampling: bool = True           # DAPO patch 2
    token_level_pg: bool = True             # DAPO patch 3
    overlong_reward_band: tuple[float, float] | None = (0.95, 1.0)
                                            # DAPO patch 4 (None = hard cliff)
    advantage_eps: float = 1e-8
    kl_coef: float = 0.04
    importance_ratio_level: Literal["token", "sequence"] = "sequence"  # GSPO
```

Defaults are DAPO + GSPO on. Set individual flags off for
ablation / regression.

## Consequences

**Intended:**
- AIME / GPQA / MATH benchmark uplift documented at +3pp class on
  Qwen2.5-32B-base from DAPO alone vs vanilla GRPO. Combined with
  GSPO and the rest of the RL stack, target +10pp on AIME 2024 vs
  v0.1 base (per ROADMAP v0.2 exit).
- No-regression on training stability: each patch is documented to
  improve, not destabilize, the loss curve.
- Free 2× compute efficiency from Dynamic Sampling alone in
  curriculum-style RL (hard-easy mix where many groups are
  trivially all-correct or all-wrong).
- Long-CoT scaling: Token-Level PG fixes the dominant under-
  weighting that caused vanilla GRPO to plateau on hard reasoning.

**Unintended / accepted:**
- Hyperparameter surface grows: 4 new knobs. Defaults from
  published recipe; tuning per domain (math vs code vs agent) is
  v0.2 ablation work.
- Dynamic Sampling adds rollout-replacement loop overhead. Net
  positive on compute since all-same-reward groups would have
  contributed zero gradient anyway, but the loop has bookkeeping
  cost.
- Token-Level PG vs sequence-mean changes the *scale* of the loss
  output. Logging consumers must be aware (compare across same
  setting only).

**Explicit non-effects:**
- Does NOT replace KL penalty (still Schulman unbiased).
- Does NOT replace advantage normalization (still group-relative).
- Does NOT interact with GSPO's importance ratio (orthogonal axes).

## Alternatives considered

- **Adopt only Clip-Higher.** Single-patch baseline, well-validated.
  Rejected as final state — leaves wins on the table. Used as
  ablation point.
- **Adopt only Dynamic Sampling.** Pure compute efficiency win.
  Same logic — wins on table.
- **Wait for community consensus on patches.** Rejected: 50% step
  reduction at iso-AIME is the consensus. Already in `verl`
  reference.
- **Token-Level PG with weighted sequences (length-aware
  reweighting).** More complex; DAPO's plain sum is what's
  validated. Defer weighted variant to v0.2 ablation.

## Implementation notes

### Files affected

- `packages/posttraining/src/saint_llm_posttraining/grpo.py`:
  - Extend `GRPOConfig` with the four flags above.
  - In `grpo_loss`:
    - Apply Clip-Higher: `surrogate2 = ratio.clamp(1 - cfg.clip_eps_lower,
      1 + cfg.clip_eps_upper) * adv_tok` when `cfg.use_clip_higher`.
    - Apply Token-Level PG: aggregation
      `(pg_objective * mask).sum(dim=-1) / 1.0` (no per-sequence
      mean) vs current `/ n_tok`.
    - Add `apply_overlong_reward_band(rewards, lengths, max_len, band)`
      helper.
- `packages/posttraining/src/saint_llm_posttraining/grpo.py` (new
  function): `dynamic_sampling_filter(rollout: RolloutBatch) -> Tensor`
  returns mask for which group rows to keep (all-correct/all-wrong
  groups masked off). Caller responsible for replacement.

### Tests

- `packages/posttraining/tests/test_dapo_patches.py`:
  - **Clip-Higher**: synthetic batch with high positive advantage,
    ratios > 1.2; verify upper-clip at 1.28 not 1.20.
  - **Dynamic Sampling**: synthetic batch where groups 0,2 are
    all-correct, group 1 is mixed; verify filter mask = `[F, T, F, …]`
    (group-row-aligned).
  - **Token-Level PG**: two synthetic completions of length 5 and
    50, same per-token loss; verify long completion contributes
    10× to the total under Token-Level PG vs equal under
    sequence-mean.
  - **Overlong Reward Shaping**: feed completions at length 0.95
    × max_len and 1.0 × max_len; verify smooth penalty transition,
    no discontinuity.

### Stack ordering

With ADR-0014 GSPO:

```
loss = pg_term + β * kl_term

pg_term = -(per-token: min(r * A, clip(r, 1-ε_lower, 1+ε_upper) * A))
                .sum(dim=tokens)        # DAPO Token-Level PG
                .mean(dim=sequences)
where:
  r = sequence-level GSPO ratio (broadcast back to per-token)
  A = group-relative advantage with Dynamic Sampling pre-filter
  ε_lower, ε_upper = DAPO Clip-Higher
  reward going into A respects DAPO Overlong band
```

### Out of scope for this ADR

- Critique-GRPO additional layer (separate ADR-0016).
- Tree-GRPO at v0.3 (agentic-RL, separate ADR).
- λ-GRPO learnable token-preference (research track).

### Promotion of RL-02

Update AUGMENTATIONS.md row RL-02 status to `accepted-v0.2`.
