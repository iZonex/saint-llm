# ADR-0014: GSPO replaces token-level GRPO importance ratio for MoE — promote RL-01

- **Status**: proposed → accepted-v0.2 (this ADR)
- **Date**: 2026-04-25
- **Deciders**: Dmytro
- **Related**: AUGMENTATIONS row RL-01, ADR-0009 (GRPO baseline +
  Q1 2026 revisions), `posttraining/grpo.py` (already shipped at
  unit-test scale)
- **Sources**: [arXiv 2507.18071](https://arxiv.org/abs/2507.18071)
  (GSPO, Qwen team), [Qwen3 GSPO blog](https://qwenlm.github.io/blog/gspo/)

## Context

Vanilla GRPO computes a per-token importance ratio
`r_t = π(o_t | prefix) / π_old(o_t | prefix)`, applies the PPO
clipped surrogate per token, then averages or sums across the
response. For MoE models there's a specific pathology Qwen team
documented while training Qwen3:

- Each gradient step changes the routing decisions for ~10% of
  activated experts (typical fine-grained MoE behavior).
- A token routed to a different expert cluster between `π_old` and
  `π` produces a per-token ratio with extreme variance.
- Token-level ratios magnify this expert-flip noise; the clipped
  surrogate becomes effectively a constant on most tokens (clip
  triggered) plus extreme outliers on others.
- Their initial mitigation was "Routing Replay" — replay the old
  routing decisions when computing `π(o_t | prefix)`. This works
  but is operationally complex and breaks the "the new policy is
  what the new policy is" semantics.

GSPO (Group Sequence Policy Optimization) replaces the token-level
ratio with a **length-normalized sequence-level ratio** computed
once per rollout:

```
r_seq = ( prod_t π(o_t | prefix_t) / π_old(o_t | prefix_t) )^(1/L)
      = exp( (1/L) * sum_t (log π(o_t | ...) - log π_old(o_t | ...)) )
```

The clip is applied at sequence level, not per token. Expert-flip
variance averages out across the sequence; ratios stay near 1 even
with substantial routing changes. Qwen team eliminated the Routing
Replay hack with GSPO, ran Qwen3 production training stably.

saint-llm is MoE (DeepSeekMoE per ADR-0003). Token-level GRPO is a
direct path to the same instability. GSPO is the documented fix.

## Decision

Replace token-level importance ratio with GSPO sequence-level
length-normalized ratio in the v0.2 RL stack. Specifically:

1. New code path in `packages/posttraining/src/saint_llm_posttraining/grpo.py`
   (extends existing `grpo_loss`):
   - Compute per-token log-ratios as today.
   - Sum and divide by sequence length to get
     `log_r_seq = (1/L_i) * sum_t (new_logprob_t - old_logprob_t)`.
   - `ratio_seq = exp(log_r_seq)`, broadcast back to per-token for
     the surrogate.
   - PPO clip applied at sequence level (single clip per rollout).

2. New config flag in `GRPOConfig`:
   - `importance_ratio_level: Literal["token", "sequence"] = "sequence"`
     (defaults to GSPO going forward).
   - When `"token"` requested, fall back to current behavior for
     ablation / regression testing.

3. Loss reduction: per-completion advantage × per-token clipped
   surrogate, then mean over response tokens (existing behavior).
   The change is **only** in how the ratio is computed; mask
   handling stays the same.

4. Migration: existing 12 unit tests for token-level GRPO
   (`packages/posttraining/tests/test_grpo.py`) are preserved
   under `importance_ratio_level="token"`. New tests cover the
   sequence-level path.

## Consequences

**Intended:**
- MoE RL stability matches Qwen3 production: no Routing Replay
  hack required.
- Lower variance per gradient step at iso-rollout-budget; smoother
  loss curves.
- Compatible with existing length-controlled GRPO (Kimi K1.5 length
  penalty per REA-04 / ROADMAP) since GSPO operates on the ratio,
  not on the reward.
- Compatible with DAPO four-patch stack (ADR-0015) — DAPO patches
  are on different axes (clip-higher, dynamic sampling, token-level
  PG, overlong reward shaping). GSPO replaces the importance-ratio
  computation; DAPO modifies advantage/sampling/loss aggregation.
  See implementation notes for stack ordering.

**Unintended / accepted:**
- Sequence-level clip is coarser than token-level: a single bad
  token in a 1000-token completion can no longer be clipped in
  isolation (its outlier value is geometrically averaged into
  `r_seq`). Acceptable trade — published Qwen3 evidence shows
  this rarely matters in practice.
- Existing GRPO unit tests must keep working under
  `importance_ratio_level="token"`. Locks the old code path in
  a fallback branch — small maintenance debt.
- The DAPO "Token-Level Policy Gradient Loss" patch (ADR-0015)
  interacts: token-level PG aggregation is *separate* from
  token-level *importance ratio*. Must implement carefully so
  GSPO + DAPO compose, not clash.

**Explicit non-effects:**
- Does NOT change advantage computation
  (`compute_group_advantage` unchanged).
- Does NOT change KL penalty computation (Schulman unbiased
  estimator unchanged).
- Does NOT change rollout / data path.

## Alternatives considered

- **Keep token-level GRPO + Routing Replay.** Rejected: Qwen team
  explicitly moved away from this — it's complex, breaks
  policy-update semantics, doesn't fully solve the variance
  problem.
- **Token-level GRPO without Routing Replay.** Rejected: known to
  destabilize MoE RL per published evidence.
- **REINFORCE++ (no PPO clip at all).** Considered. Strong
  benchmark numbers (24.10 avg vs GRPO 22.58 / PPO 21.85 on
  AIME/HMMT/CMIMC). Lower invasiveness than GSPO but doesn't
  *specifically* address the MoE expert-flip variance source.
  GSPO is more targeted; REINFORCE++ is more general. Defer
  REINFORCE++ ablation to v0.2 work — both can land if GSPO
  underperforms.
- **DAPO without GSPO.** Considered. DAPO's four patches are
  orthogonal and improve GRPO stability; they don't specifically
  address MoE variance though. Stacking GSPO + DAPO is the
  Qwen3+ByteDance combined recipe and what saint-llm should adopt.
- **Defer GSPO to v0.3.** Rejected: v0.2 is when post-training RL
  begins. Without GSPO, v0.2 risks early divergence on MoE.

## Implementation notes

### Files affected

- `packages/posttraining/src/saint_llm_posttraining/grpo.py` —
  extend `GRPOConfig` and `grpo_loss` with `importance_ratio_level`
  branch.

### Tests

- `packages/posttraining/tests/test_gspo.py`:
  - Compare `importance_ratio_level="token"` vs `"sequence"` on
    a synthetic batch where new policy ≈ old policy: both should
    produce ~zero loss.
  - Sequence-level should have lower per-step variance than
    token-level when feed deliberate per-token noise (assert).
- Existing 12 unit tests in `test_grpo.py` keep passing under
  `importance_ratio_level="token"` (regression).

### Stack ordering with DAPO

The composed v0.2 RL loss is:

```
loss = pg_term + β * kl_term
```

where:
- `pg_term` = − E_{i, t ∈ R_i} [ min(ratio × A_i, clip(ratio, …) × A_i) ]
  - `ratio` is per-sequence (GSPO) or per-token (legacy)
  - `clip` is `(1 - ε_lower, 1 + ε_upper)` with DAPO's Clip-Higher
    asymmetry
  - Aggregation across tokens is DAPO's Token-Level PG (sum, not
    mean)
- `A_i` is group-relative advantage, maybe with DAPO's Dynamic
  Sampling pre-filter (drop all-correct/all-wrong groups)
- Reward includes DAPO's Overlong Reward Shaping (soft-penalty
  band)
- `kl_term` is Schulman unbiased KL against `π_ref`

GSPO sets the ratio shape; DAPO sets clip + aggregation + advantage
filter + reward. They compose without conflict.

### Out of scope for this ADR

- DAPO patches (separate ADR-0015).
- REINFORCE++ ablation (v0.2 work, separate decision if GSPO
  insufficient).
- Tree-GRPO / Critique-GRPO (separate ADRs for v0.2 / v0.3).

### Promotion of RL-01

Update AUGMENTATIONS.md row RL-01 status to `accepted-v0.2`.
