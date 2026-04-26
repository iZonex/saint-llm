"""Saint LLM distributed training (Workstream F).

Modules:

* gradient_compression — top-k, top-k+residual, SparseLoCo (DIST-05)
* diloco               — DiLoCo outer-loop synchronizer (DIST-04)
* noloco               — NoLoCo (log(N) ring + AdamW outer)

Planned (not yet implemented):
* ep            — Fine-grained Expert Parallelism with wave scheduling (MegaMoE)
* zero          — Hybrid ZeRO bucket assignment for Muon (knapsack + per-expert)
* cp            — Two-stage Contextual Parallelism for compressed attention
* dualpipe      — DualPipe 1F1B with mHC overlap adjustments

Production deployment of FSDP2 / EP / DeepEP / DualPipeV is layered on
top of these abstractions when the runtime networking is wired up.
"""

from saint_llm_distributed.diloco import (
    DiLoCo,
    DiLoCoConfig,
    PseudoGradientReducer,
)
from saint_llm_distributed.gradient_compression import (
    CompressedGradient,
    SparseLoCoBloom,
    TopKWithResidualCompressor,
    decompress,
    sparse_loco_compress,
    top_k_compression,
)
from saint_llm_distributed.noloco import (
    NoLoCo,
    NoLoCoConfig,
    PeerReducer,
    RingAllReduceSimulator,
    ring_all_reduce,
    ring_log2_rounds,
)
from saint_llm_distributed.decentralized_round import (
    DecentralizedTrainingRound,
    PeerOutcome,
    PeerSubmission,
    RoundResult,
)
from saint_llm_distributed.bittensor import (
    BittensorMinerServer,
    BittensorValidatorClient,
    MinerHandler,
    MinerInfo,
    MinerResponse,
    ValidatorScorer,
    WeightedScore,
    aggregate_cohort_scores,
    cohort_consensus_score,
)
from saint_llm_distributed.expert_parallel import (
    DispatchPlan,
    combine_outputs_by_expert,
    dispatch_tokens_by_expert,
)
from saint_llm_distributed.fsdp import (
    default_block_policy,
    ensure_local_gloo_pg,
    fsdp2_wrap,
    init_default_mesh,
)
from saint_llm_distributed.reputation import (
    PeerReputationConfig,
    PeerReputationTracker,
    reputation_weighted_reduce,
)
from saint_llm_distributed.verification import (
    GradientCheck,
    GradientVerifier,
    VerificationResult,
    random_peer_gradient,
)
from saint_llm_distributed.zero import (
    BucketAssignment,
    adamw_param_cost,
    hybrid_zero_assign,
    muon_param_cost,
    pack_buckets_by_cost,
)

__version__ = "0.0.1"

__all__ = [
    "BittensorMinerServer",
    "BittensorValidatorClient",
    "BucketAssignment",
    "CompressedGradient",
    "DecentralizedTrainingRound",
    "DiLoCo",
    "DiLoCoConfig",
    "DispatchPlan",
    "GradientCheck",
    "GradientVerifier",
    "MinerHandler",
    "MinerInfo",
    "MinerResponse",
    "NoLoCo",
    "NoLoCoConfig",
    "PeerOutcome",
    "PeerReducer",
    "PeerReputationConfig",
    "PeerReputationTracker",
    "PeerSubmission",
    "PseudoGradientReducer",
    "RingAllReduceSimulator",
    "RoundResult",
    "SparseLoCoBloom",
    "TopKWithResidualCompressor",
    "ValidatorScorer",
    "VerificationResult",
    "WeightedScore",
    "adamw_param_cost",
    "aggregate_cohort_scores",
    "cohort_consensus_score",
    "combine_outputs_by_expert",
    "decompress",
    "default_block_policy",
    "dispatch_tokens_by_expert",
    "ensure_local_gloo_pg",
    "fsdp2_wrap",
    "hybrid_zero_assign",
    "init_default_mesh",
    "muon_param_cost",
    "pack_buckets_by_cost",
    "random_peer_gradient",
    "reputation_weighted_reduce",
    "ring_all_reduce",
    "ring_log2_rounds",
    "sparse_loco_compress",
    "top_k_compression",
]
