"""Saint LLM post-training pipeline.

Modules:
    sft               — SFT data plumbing: encode (prompt, response) → packed batches
                        with response-only loss masking, plus masked CE loss
    grpo              — Group Relative Policy Optimization: group-normalized
                        advantage, clipped surrogate, KL-to-ref penalty
    specialist        — SFT + GRPO specialist training (per-domain)
    grm               — Generative Reward Model (joint actor + judge)
    opd               — Full-vocabulary On-Policy Distillation (multi-teacher)
    teacher_offload   — ZeRO-like teacher param sharding to centralized storage
    reasoning_modes   — Non-think / High / Max reasoning effort modes
    tools.dsml        — DSML XML tool-call schema
    quick_instruction — Quick Instruction special tokens (action/title/query/...)
"""

from saint_llm_posttraining.caption_rewards import (
    PerExampleCaptionRewardFn,
    composite_caption_reward,
    constant_reward,
    length_target_reward,
    unigram_overlap_reward,
)
from saint_llm_posttraining.critique_grpo import (
    DEFAULT_CRITIQUE_TEMPLATE,
    DEFAULT_REFINEMENT_TEMPLATE,
    CritiqueGRPOConfig,
    CritiqueRequest,
    collect_critique_requests,
    make_critique_prompt,
    make_refinement_prompt,
    needs_critique_mask,
)
from saint_llm_posttraining.grm import (
    GRMConfig,
    GRMRewardFn,
    binary_grm_config,
    judge_completion,
    likert_grm_config,
)
from saint_llm_posttraining.grpo import (
    GRPOConfig,
    RolloutBatch,
    compute_group_advantage,
    dynamic_sampling_mask,
    gather_token_logprobs,
    grpo_loss,
    overlong_reward_shape,
)
from saint_llm_posttraining.grpo_trainer import (
    GRPOTrainerStep,
    RewardFn,
    build_rollout_batch,
    grpo_train_step,
    score_rollouts,
)
from saint_llm_posttraining.gtpo import (
    GTPOConfig,
    entropy_weight,
    gtpo_loss,
    per_token_entropy,
)
from saint_llm_posttraining.sft import (
    SFTExample,
    SFTPackedBatch,
    encode_sft,
    pack_sft_examples,
    pack_sft_into_batch,
    sft_cross_entropy,
)
from saint_llm_posttraining.sft_dataset import JsonlSFTDataset
from saint_llm_posttraining.specialist import (
    RLStepOutput,
    SFTStepOutput,
    SpecialistPipeline,
)
from saint_llm_posttraining.multimodal_grpo_trainer import (
    MultimodalRolloutBatch,
    build_multimodal_rollout_batch,
    multimodal_grpo_train_step,
)
from saint_llm_posttraining.multimodal_sft import (
    MultimodalSFTOutput,
    multimodal_sft_loss,
    multimodal_sft_step,
)
from saint_llm_posttraining.opd import (
    OPDConfig,
    aggregate_teacher_logits,
    opd_kl_loss,
    opd_step,
)
from saint_llm_posttraining.rollout_generator import (
    RolloutGenConfig,
    decode_rollouts_for_reward,
    generate_grpo_rollouts,
)
from saint_llm_posttraining.think_prm import (
    StepRewardHead,
    ThinkPRMConfig,
    compute_step_rewards,
    find_step_boundaries,
    gather_step_logits,
    gather_step_scores,
    prm_filter_mask,
    step_prm_loss,
)
from saint_llm_posttraining.tree_grpo import (
    TreeRolloutBatch,
    compute_tree_advantage,
    tree_grpo_loss,
)
from saint_llm_posttraining.tree_grpo_trainer import (
    TreeGRPOTrainerStep,
    build_tree_rollout_batch,
    tree_dynamic_sampling_mask,
    tree_grpo_train_step,
)

__version__ = "0.0.1"

__all__ = [
    "DEFAULT_CRITIQUE_TEMPLATE",
    "DEFAULT_REFINEMENT_TEMPLATE",
    "CritiqueGRPOConfig",
    "CritiqueRequest",
    "GRMConfig",
    "GRMRewardFn",
    "GRPOConfig",
    "GRPOTrainerStep",
    "GTPOConfig",
    "JsonlSFTDataset",
    "MultimodalRolloutBatch",
    "MultimodalSFTOutput",
    "OPDConfig",
    "PerExampleCaptionRewardFn",
    "RLStepOutput",
    "RewardFn",
    "RolloutBatch",
    "RolloutGenConfig",
    "SFTExample",
    "SFTPackedBatch",
    "SFTStepOutput",
    "SpecialistPipeline",
    "StepRewardHead",
    "ThinkPRMConfig",
    "TreeGRPOTrainerStep",
    "TreeRolloutBatch",
    "aggregate_teacher_logits",
    "binary_grm_config",
    "build_multimodal_rollout_batch",
    "build_rollout_batch",
    "build_tree_rollout_batch",
    "collect_critique_requests",
    "composite_caption_reward",
    "compute_group_advantage",
    "compute_step_rewards",
    "compute_tree_advantage",
    "constant_reward",
    "decode_rollouts_for_reward",
    "dynamic_sampling_mask",
    "encode_sft",
    "entropy_weight",
    "find_step_boundaries",
    "gather_step_logits",
    "gather_step_scores",
    "gather_token_logprobs",
    "generate_grpo_rollouts",
    "grpo_loss",
    "grpo_train_step",
    "gtpo_loss",
    "judge_completion",
    "length_target_reward",
    "likert_grm_config",
    "make_critique_prompt",
    "make_refinement_prompt",
    "multimodal_grpo_train_step",
    "multimodal_sft_loss",
    "multimodal_sft_step",
    "needs_critique_mask",
    "opd_kl_loss",
    "opd_step",
    "overlong_reward_shape",
    "pack_sft_examples",
    "pack_sft_into_batch",
    "per_token_entropy",
    "prm_filter_mask",
    "score_rollouts",
    "sft_cross_entropy",
    "step_prm_loss",
    "tree_dynamic_sampling_mask",
    "tree_grpo_loss",
    "tree_grpo_train_step",
    "unigram_overlap_reward",
]
