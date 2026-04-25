"""Pydantic schemas for saint-llm core model architecture.

Mirrors `configs/model/v4_flash.yaml` and `configs/model/v4_pro.yaml`.
Reference: `docs/DeepSeek_V4.pdf` §4.2.1, `docs/AUGMENTATIONS.md`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CSAConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    compression_rate: int = 4
    indexer_query_heads: int = 64
    indexer_head_dim: int = 128
    attention_top_k: int = 512


class HCAConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    compression_rate: int = 128


class AttentionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    query_heads: int = 64
    head_dim: int = 512
    query_compression_dim: int = 1024
    output_proj_groups: int = 8
    attention_intermediate_dim: int = 1024
    sliding_window_size: int = 128
    rope_dim: int = 64
    rope_theta: float = 10000.0
    use_attention_sink: bool = True


class MoEConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    hash_routed_layers: int = 3
    shared_experts: int = 1
    routed_experts: int = 256
    experts_per_token: int = 6
    expert_intermediate_dim: int = 2048
    affinity_fn: Literal["sqrt_softplus", "sigmoid"] = "sqrt_softplus"
    aux_loss_free_bias_speed: float = 0.001
    sequence_balance_weight: float = 0.0001
    swiglu_clamp_linear: tuple[float, float] = (-10.0, 10.0)
    swiglu_clamp_gate_max: float = 10.0


class MHCConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    expansion_factor: int = 4
    sinkhorn_iters: int = 20
    init_alpha: float = 0.02
    init_static_bias: float = 0.0


class MTPConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    depth: int = 1
    loss_weight_main: float = 0.3
    loss_weight_decay: float = 0.1


class MultimodalConfig(BaseModel):
    """Multimodal hooks reserved in v0.1 (per AUGMENTATIONS.md MM-V/MM-A/MM-Vid/MEM-03)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    enable_vision_projector: bool = False
    vision_input_dim: int = 1152
    enable_audio_projector: bool = False
    audio_input_dim: int = 1280
    enable_mrope: bool = False
    enable_residual_side_channel: bool = True
    side_channel_alpha_init: float = 0.0
    enable_generation_head: bool = False
    generation_head_vocab: int = 2368
    enable_modality_router_bias: bool = True


class TokenizerSlots(BaseModel):
    """Reserved control-token offsets — see AUGMENTATIONS.md §Tokenizer."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    image_pad: int = 128514
    audio_start: int = 128518
    audio_end: int = 128519
    video_start: int = 128515
    video_end: int = 128516
    think_start: int = 128648
    think_end: int = 128649
    memory_block_base: int = 128710


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    variant: Literal["flash", "pro"] = "flash"
    n_layers: int = 43
    hidden_dim: int = 4096
    vocab_size: int = 131072

    first_dense_swa_layers: int = 2
    attention_pattern: Literal["csa_hca_interleaved"] = "csa_hca_interleaved"

    csa: CSAConfig = Field(default_factory=CSAConfig)
    hca: HCAConfig = Field(default_factory=HCAConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    moe: MoEConfig = Field(default_factory=MoEConfig)
    mhc: MHCConfig = Field(default_factory=MHCConfig)
    mtp: MTPConfig = Field(default_factory=MTPConfig)
    multimodal: MultimodalConfig = Field(default_factory=MultimodalConfig)
    tokenizer_slots: TokenizerSlots = Field(default_factory=TokenizerSlots)

    rms_norm_eps: float = 1.0e-6
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    # Linear-layer quantization for QAT-style training.
    # bf16: standard nn.Linear (no quant). fp8: E4M3 with STE (Fp8Linear).
    # fp4:  MXFP4 E2M1 block-32 with STE (Fp4Linear).
    linear_quant: Literal["bf16", "fp8", "fp4"] = "bf16"
    fp4_block_size: int = 32
    # When linear_quant=="fp8" and we're on Ada sm89+, swap fake-quant + F.linear
    # for a real torch._scaled_mm call. Falls back transparently on CPU / older GPUs.
    fp8_use_real_gemm: bool = False
    # When True, MoE routed_experts go through saint_llm_kernels.GroupedSwiGLUExperts
    # (one sort + 3 torch._grouped_mm calls) instead of the per-expert Python loop.
    # Limitation: grouped path is bf16/fp32 only — no fp8/fp4 quant on routed experts
    # in this mode. shared_experts still respect linear_quant.
    moe_use_grouped_gemm: bool = False

    @classmethod
    def v4_flash(cls) -> ModelConfig:
        return cls()

    @classmethod
    def v4_pro(cls) -> ModelConfig:
        return cls(
            variant="pro",
            n_layers=61,
            hidden_dim=7168,
            csa=CSAConfig(attention_top_k=1024),
            attention=AttentionConfig(
                query_heads=128,
                head_dim=512,
                query_compression_dim=1536,
                output_proj_groups=16,
            ),
            moe=MoEConfig(routed_experts=384, expert_intermediate_dim=3072),
        )

    @classmethod
    def small_flash(cls) -> ModelConfig:
        """First-real-training config — ~150M params, fits 12 GB hpomen Ada.

        Designed for the v0.1 "does the architecture train at scale?" milestone:
        validate loss curve shape and basic generation quality on a meaningful
        corpus (FineWeb-Edu sample) before committing to a multi-GPU run.

        Sizing rationale:
        * vocab=32768 — matches a tokenizer trainable on a few-GB sample.
        * hidden_dim=768, head_dim=64, query_heads=12 — GPT-2 small footprint.
        * 8 layers, each (CSA/HCA + MoE) — keeps depth modest for hpomen latency.
        * 4 routed experts at 1536 intermediate, top-2 routing — keeps MoE
          parameter count below the embedding budget.
        * All in_features that go through fp4 quant are multiples of 32.
        """
        return cls(
            variant="flash",
            n_layers=8,
            hidden_dim=768,
            vocab_size=32_768,
            first_dense_swa_layers=1,
            csa=CSAConfig(
                compression_rate=4,
                indexer_query_heads=4,
                indexer_head_dim=64,
                attention_top_k=64,
            ),
            hca=HCAConfig(compression_rate=64),
            attention=AttentionConfig(
                query_heads=12,
                head_dim=64,
                query_compression_dim=512,
                output_proj_groups=4,
                attention_intermediate_dim=192,
                sliding_window_size=128,
                rope_dim=32,
            ),
            moe=MoEConfig(
                hash_routed_layers=2,
                shared_experts=1,
                routed_experts=4,
                experts_per_token=2,
                expert_intermediate_dim=1536,
            ),
            mhc=MHCConfig(expansion_factor=2, sinkhorn_iters=8),
            tokenizer_slots=TokenizerSlots(
                image_pad=32500,
                audio_start=32501,
                audio_end=32505,
                video_start=32506,
                video_end=32507,
                think_start=32508,
                think_end=32509,
                memory_block_base=32510,
            ),
        )

    @classmethod
    def tiny(cls) -> ModelConfig:
        """Test-friendly config — fits on CPU/MPS, vocab=512 with slots remapped to fit."""
        return cls(
            variant="flash",
            n_layers=4,
            hidden_dim=128,
            vocab_size=512,
            first_dense_swa_layers=1,
            csa=CSAConfig(
                compression_rate=2,
                indexer_query_heads=2,
                indexer_head_dim=16,
                attention_top_k=8,
            ),
            hca=HCAConfig(compression_rate=8),
            attention=AttentionConfig(
                query_heads=4,
                head_dim=32,
                query_compression_dim=32,
                output_proj_groups=2,
                attention_intermediate_dim=32,
                sliding_window_size=8,
                rope_dim=8,
            ),
            moe=MoEConfig(
                hash_routed_layers=1,
                shared_experts=1,
                routed_experts=8,
                experts_per_token=2,
                expert_intermediate_dim=64,
            ),
            mhc=MHCConfig(expansion_factor=2, sinkhorn_iters=5),
            tokenizer_slots=TokenizerSlots(
                image_pad=500,
                audio_start=501,
                audio_end=505,
                video_start=506,
                video_end=507,
                think_start=508,
                think_end=509,
                memory_block_base=510,
            ),
        )
