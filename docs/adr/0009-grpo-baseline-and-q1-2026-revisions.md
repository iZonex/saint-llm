# ADR-0009: GRPO baseline + Q1 2026 revisions (GSPO + DAPO + Critique-GRPO)

- **Status**: accepted (baseline) + proposed (revisions)
- **Date**: 2026-04-25
- **Deciders**: Dmytro
- **Related**: REA-02, RL-01..RL-15 in `docs/AUGMENTATIONS.md`,
  `packages/posttraining/src/saint_llm_posttraining/grpo.py` (already
  shipped at unit-test scale)

## Context

The baseline plan (REA-02) commits to GRPO + length-controlled GRPO
(Kimi K1.5) + on-policy distillation as the post-training reasoning
core. The math layer (group-normalized advantage + clipped surrogate
+ Schulman unbiased KL) ships in `posttraining/grpo.py` at
unit-test scale.

Q1 2026 research (research review 2026-04-25) shows GRPO has been
**patched, not replaced**, and the patches matter materially:

- **GSPO** (Qwen3, arxiv 2507.18071) replaces token-level importance
  ratio with length-normalized sequence-level ratio. Specifically
  invented to fix MoE RL stability; saint-llm is MoE so this is
  directly relevant.
- **DAPO** (ByteDance Seed, arxiv 2503.14476) adds four orthogonal
  patches: Clip-Higher (decouples upper/lower clip ε to fight
  entropy collapse), Dynamic Sampling (drop all-correct/all-wrong
  groups so gradients aren't zero), Token-Level PG (sum tokens not
  mean responses, fixes long-CoT under-weighting), Overlong Reward
  Shaping (soft-penalty band instead of length cliff).
- **Dr.GRPO** (arxiv 2503.20783, COLM 2025) removes biased
  length-normalization and per-group std normalization. Two-line
  delete. Free.
- **Critique-GRPO** (arxiv 2506.03106) adds natural-language
  critique to numerical advantage. +15-22% Pass@1 on Qwen2.5-Math.
  Breaks GRPO plateau on hard reasoning.

## Decision

Two-part decision:

**(a) Baseline (accepted):** GRPO core stays as the foundation of
post-training RL. The math is correct; what we shipped in
`posttraining/grpo.py` is the right starting point.

**(b) Q1 2026 revisions (proposed for v0.2 implementation):**
- **RL-01 GSPO:** replace token-level importance ratio with
  length-normalized sequence ratio in `posttraining/grpo.py`. Required
  for MoE training stability (saint-llm is MoE).
- **RL-02 DAPO:** add the four patches as configurable knobs in
  `GRPOConfig` (`clip_eps_upper`, `dynamic_sampling`,
  `token_level_pg`, `overlong_reward_band`).
- **RL-03 Dr.GRPO + CE-GPPO:** unbiased loss tweaks; default-on once
  validated.
- **RL-04 Critique-GRPO:** new `critique_grpo.py` module wrapping
  GRPO with natural-language-critique conditioned refinement.

Length-controlled (REA-04, Kimi K1.5) stays accepted as orthogonal.

## Consequences

**Intended:**
- saint-llm RL stability at MoE scale matches Qwen3-class production.
- Reasoning post-training quality gains target +10pp on AIME, +5pp on
  GPQA per ROADMAP v0.2 exit criteria.
- Existing GRPO unit tests stay valid; new patches add new tests.

**Unintended / accepted:**
- `posttraining/grpo.py` API will change at v0.2: callers using
  current `compute_group_advantage` + `grpo_loss` get a flag-set on
  GRPOConfig rather than a function rename. Backward compat at API
  level; behavior change behind config flag.
- Hyperparameter surface grows; defaults need careful per-domain
  tuning (math vs code vs agent).

## Alternatives considered

- **Stick with vanilla GRPO.** Rejected: MoE instability is real per
  Qwen3 published evidence; we'd hit it at scale.
- **Full PPO with critic.** Rejected: critic-free is the 2025-2026
  consensus; PPO+critic is higher memory and slower to converge for
  comparable results.
- **DPO/SimPO/KTO/ORPO as primary RL.** Rejected: production stack
  (Tülu 3, Nemotron 3 Super) uses these as alignment / preference
  stages **before** RLVR, not as replacements. RLVR / GRPO-family
  remains the on-policy core.
- **REINFORCE++ as primary.** Considered. Strong head-to-head numbers
  vs GRPO on math/code (24.10 avg vs 22.58 / 21.85). Lower
  invasiveness than GSPO. **Open question:** does REINFORCE++ or
  GSPO better fit our MoE architecture? Defer to v0.2 ablation.

## Implementation notes

- Existing `posttraining/grpo.py` API:
  `compute_group_advantage`, `gather_token_logprobs`, `grpo_loss`,
  `GRPOConfig`, `RolloutBatch`. Already 12 unit tests passing.
- v0.2 work: extend `GRPOConfig` with new knobs (per RL-01..RL-04);
  add `compute_sequence_advantage` for GSPO; add critique branch
  conditional on `cfg.critique_enabled`.
- ROADMAP v0.2 exit criteria depend on these being shipped; ADR
  must promote `proposed` to `accepted-v0.2` before v0.2
  implementation begins.
