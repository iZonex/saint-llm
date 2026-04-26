"""Top-level Saint LLM transformer assembly.

Architecture (per DeepSeek V4 Figure 2 + augmentations from docs/AUGMENTATIONS.md):

    embed(token_ids) ── replace <|image_pad|> with vision_proj(vision)
                    └── replace <|audio_*|> with audio_proj(audio)
        ↓
    mHC.expand → expanded residual stream (B, T, n_hc, d)
        ↓
    [first_dense_swa_layers] × SWA-only block (no compression, sliding-window MQA)
    [remaining layers]      × CSA/HCA interleaved blocks
        ↓
    mHC.collapse → (B, T, d)
        ↓
    final RMSNorm
        ↓
    LM head (tied) ──→ logits over vocab
                  └── MTP head(s) for next-K prediction (K=1 v0.1; reserved K>1 for v0.2 audio)
                  └── GenerationHeadHook → reserved VQ codebook slots (zero-init, frozen v0.1)

Each transformer block:
    attn_mhc.split  → inner_in, gates
    attention(inner_in, is_visual)
    attn_mhc.combine
    moe_mhc.split  → inner_in, gates
    moe(inner_in, token_ids, is_visual)
    moe_mhc.combine
    side_channel(collapsed view) — alpha=0 by default, gated off
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from saint_llm_core.attention import CSA, HCA, RMSNorm, SWAttention
from saint_llm_core.config import ModelConfig
from saint_llm_core.init import umup_init
from saint_llm_core.moe import DeepSeekMoE, LinearFactory
from saint_llm_core.mtp import MTPStack
from saint_llm_core.multimodal import GenerationHeadHook, ModalityProjector, ResidualSideChannel
from saint_llm_core.quant import make_linear_factory
from saint_llm_core.residual import MHC


class _CheckpointedBlock:
    """Closure-style adapter so torch.utils.checkpoint sees a single-tensor input.

    ``token_ids`` and ``is_visual`` are bound at construction time (per-step
    constants from the surrounding forward). ``checkpoint(use_reentrant=False)``
    handles non-tensor saved state without needing them as positional args.
    """

    def __init__(
        self,
        block: nn.Module,
        token_ids: Tensor,
        is_visual: Tensor | None,
    ) -> None:
        self.block = block
        self.token_ids = token_ids
        self.is_visual = is_visual

    def __call__(self, x_expanded: Tensor) -> Tensor:
        return self.block(x_expanded, token_ids=self.token_ids, is_visual=self.is_visual)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        layer_idx: int,
        *,
        linear_factory: LinearFactory,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        attn_module: nn.Module
        if layer_idx < cfg.first_dense_swa_layers:
            attn_module = SWAttention(cfg.hidden_dim, cfg.attention, linear_factory=linear_factory)
        else:
            position_in_csa_hca = layer_idx - cfg.first_dense_swa_layers
            if position_in_csa_hca % 2 == 0:
                attn_module = CSA(cfg.hidden_dim, cfg.attention, cfg.csa, linear_factory=linear_factory)
            else:
                attn_module = HCA(cfg.hidden_dim, cfg.attention, cfg.hca, linear_factory=linear_factory)
        self.attention = attn_module

        self.attn_mhc = MHC(cfg.hidden_dim, cfg.mhc, rms_norm_eps=cfg.rms_norm_eps)
        self.moe_mhc = MHC(cfg.hidden_dim, cfg.mhc, rms_norm_eps=cfg.rms_norm_eps)
        self.moe = DeepSeekMoE(
            cfg.hidden_dim, cfg.moe, layer_idx=layer_idx,
            enable_modality_router_bias=cfg.multimodal.enable_modality_router_bias,
            linear_factory=linear_factory,
            use_grouped_gemm=cfg.moe_use_grouped_gemm,
            grouped_quant=cfg.linear_quant,
            fp4_block_size=cfg.fp4_block_size,
        )
        self.side_channel = ResidualSideChannel(cfg.hidden_dim, cfg.multimodal.side_channel_alpha_init)

    def forward(
        self,
        x_expanded: Tensor,
        token_ids: Tensor,
        is_visual: Tensor | None,
        *,
        kv_cache_layer: object | None = None,
    ) -> Tensor:
        # Attention sub-block. ``kv_cache_layer`` is duck-typed
        # (SWAKVCacheLayer for SWA, HCAKVCacheLayer for HCA, or None). CSA
        # cache support lands in stage 6.
        inner_in, _, b_l, c_l = self.attn_mhc.split(x_expanded)
        if kv_cache_layer is not None and isinstance(self.attention, SWAttention | HCA | CSA):
            attn_out = self.attention(inner_in, is_visual=is_visual, kv_cache=kv_cache_layer)
        else:
            attn_out = self.attention(inner_in, is_visual=is_visual)
        x_expanded = self.attn_mhc.combine(x_expanded, attn_out, b_l, c_l)

        # MoE sub-block
        inner_in, _, b_l, c_l = self.moe_mhc.split(x_expanded)
        moe_out = self.moe(inner_in, token_ids=token_ids, is_visual=is_visual)
        x_expanded = self.moe_mhc.combine(x_expanded, moe_out, b_l, c_l)

        # Reserved memory side-channel (Titans/MIRAS hook). Identity at alpha=0.
        if self.side_channel.alpha.detach().abs().item() > 0.0 and self.side_channel._mem_module is not None:
            collapsed = x_expanded.mean(-2)
            updated = self.side_channel(collapsed)
            delta = (updated - collapsed).unsqueeze(-2) / x_expanded.shape[-2]
            x_expanded = x_expanded + delta

        return x_expanded


class SaintLLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.vision_proj = ModalityProjector(
            cfg.multimodal.vision_input_dim, cfg.hidden_dim, enabled=cfg.multimodal.enable_vision_projector,
        )
        self.audio_proj = ModalityProjector(
            cfg.multimodal.audio_input_dim, cfg.hidden_dim, enabled=cfg.multimodal.enable_audio_projector,
        )

        linear_factory = make_linear_factory(
            cfg.linear_quant,
            fp4_block_size=cfg.fp4_block_size,
            fp8_use_real_gemm=cfg.fp8_use_real_gemm,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(cfg, i, linear_factory=linear_factory) for i in range(cfg.n_layers)
        )
        self.final_norm = RMSNorm(cfg.hidden_dim, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

        self.mtp = MTPStack(cfg.hidden_dim, cfg.vocab_size, cfg.mtp, rms_norm_eps=cfg.rms_norm_eps)
        self.generation_head = GenerationHeadHook(
            cfg.hidden_dim, cfg.multimodal.generation_head_vocab, enabled=cfg.multimodal.enable_generation_head,
        )

        # Learnable sink tokens (ADR-0012 / ARCH-02). When ``n_sink_tokens > 0``,
        # ``forward`` prepends these embedding rows to the hidden state before
        # MHC.expand and slices them off after MHC.collapse — they participate
        # in attention but are not predicted at the LM head.
        self.sink_embeddings: nn.Parameter | None
        if cfg.n_sink_tokens > 0:
            self.sink_embeddings = nn.Parameter(
                torch.randn(cfg.n_sink_tokens, cfg.hidden_dim) / math.sqrt(cfg.hidden_dim),
            )
        else:
            self.sink_embeddings = None

        self._init_weights()

    def _init_weights(self) -> None:
        if self.cfg.init_scheme == "umup":
            # u-μP per ADR-0011: width-aware std + residual-feeder
            # depth scaling + tied-embedding-aware. mHC dynamic Ws and
            # frozen modules (generation_head when disabled) are
            # respected internally by ``umup_init``.
            umup_init(self, n_layers=self.cfg.n_layers)
            return

        # Legacy "normal" init: single std across hidden Linears + embeddings.
        std = self.cfg.initializer_range
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if not m.weight.requires_grad:
                    # Frozen layer (e.g., generation_head when disabled) — leave its zero init alone.
                    continue
                if any(part in name for part in (".w_pre", ".w_res", ".w_post")):
                    # mHC dynamic-parameterization Ws stay at zero by design (uniform-start initial gating).
                    continue
                nn.init.normal_(m.weight, mean=0.0, std=std)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=std)

    def _embed_inputs(
        self,
        token_ids: Tensor,
        vision_features: Tensor | None,
        audio_features: Tensor | None,
    ) -> Tensor:
        h = self.embed(token_ids)
        if vision_features is not None and self.cfg.multimodal.enable_vision_projector:
            mask = token_ids == self.cfg.tokenizer_slots.image_pad
            n_slots = int(mask.sum().item())
            if n_slots > 0:
                # vision_features: (n_image_tokens_total, vision_input_dim) — caller pre-flattens.
                if vision_features.shape[0] != n_slots:
                    raise ValueError(
                        f"vision_features has {vision_features.shape[0]} rows but token sequence has "
                        f"{n_slots} <|image_pad|> slots — must match",
                    )
                projected = self.vision_proj(vision_features)
                h = h.clone()
                h[mask] = projected
        if audio_features is not None and self.cfg.multimodal.enable_audio_projector:
            audio_mask = (token_ids >= self.cfg.tokenizer_slots.audio_start) & (
                token_ids <= self.cfg.tokenizer_slots.audio_end
            )
            n_audio = int(audio_mask.sum().item())
            if n_audio > 0:
                if audio_features.shape[0] != n_audio:
                    raise ValueError(
                        f"audio_features has {audio_features.shape[0]} rows but token sequence has "
                        f"{n_audio} audio slots — must match",
                    )
                projected_a = self.audio_proj(audio_features)
                h = h.clone()
                h[audio_mask] = projected_a
        return h

    def forward(
        self,
        token_ids: Tensor,
        vision_features: Tensor | None = None,
        audio_features: Tensor | None = None,
        is_visual: Tensor | None = None,
        *,
        kv_cache_bundle: object | None = None,
    ) -> dict[str, Tensor | list[Tensor]]:
        h = self._embed_inputs(token_ids, vision_features, audio_features)
        embeddings = h

        # Auto-derive is_visual from token IDs if not provided.
        if is_visual is None:
            is_visual = token_ids == self.cfg.tokenizer_slots.image_pad

        # Sink tokens: prepend learnable rows to every sequence (ADR-0012).
        # The model attends across them in every layer; we slice them off
        # before lm_head so they're not predicted. They participate in
        # attention but are not present in the loss/output positions.
        #
        # Prepend only at prefill (no cache, or cache is empty). On decode
        # calls (cache non-empty), the sinks are already in the cache from
        # the prefill call, so we only feed the new token without re-
        # prepending.
        n_sink = self.cfg.n_sink_tokens
        cache_already_warm = (
            kv_cache_bundle is not None
            and getattr(kv_cache_bundle, "length", 0) > 0
        )
        prepend_sinks = (
            self.sink_embeddings is not None
            and n_sink > 0
            and not cache_already_warm
        )
        if prepend_sinks:
            b = h.shape[0]
            sink = self.sink_embeddings.unsqueeze(0).expand(b, n_sink, -1)
            h = torch.cat([sink, h], dim=1)
            # Pad token_ids / is_visual with leading zeros so blocks that
            # read them downstream (MoE modality bias, is_visual cohesion)
            # see safe values for sink positions.
            sink_pad_ids = torch.zeros(b, n_sink, dtype=token_ids.dtype, device=token_ids.device)
            token_ids_padded = torch.cat([sink_pad_ids, token_ids], dim=1)
            sink_pad_visual = torch.zeros(b, n_sink, dtype=is_visual.dtype, device=is_visual.device)
            is_visual_padded = torch.cat([sink_pad_visual, is_visual], dim=1)
        else:
            token_ids_padded = token_ids
            is_visual_padded = is_visual

        x_expanded = MHC.expand(h, self.cfg.mhc.expansion_factor)
        use_ckpt = self.cfg.activation_checkpointing and self.training and torch.is_grad_enabled()
        for layer_idx, block in enumerate(self.blocks):
            if use_ckpt and kv_cache_bundle is None:
                # Activation checkpointing is incompatible with cached forward
                # (re-running the block during backward would re-append to the
                # cache). Skip cache when checkpointing is active.
                x_expanded = checkpoint(
                    _CheckpointedBlock(block, token_ids_padded, is_visual_padded),
                    x_expanded,
                    use_reentrant=False,
                )
            else:
                cache_layer = (
                    kv_cache_bundle.for_layer(layer_idx)  # type: ignore[attr-defined]
                    if kv_cache_bundle is not None
                    else None
                )
                x_expanded = block(
                    x_expanded,
                    token_ids=token_ids_padded,
                    is_visual=is_visual_padded,
                    kv_cache_layer=cache_layer,
                )
        h = MHC.collapse(x_expanded)
        # Slice off sink positions before final_norm + heads — they're
        # processed through attention but not predicted. Only slice when
        # we actually prepended on this call (prefill); decode calls
        # don't include sinks in their input/output stream.
        if prepend_sinks:
            h = h[:, n_sink:, :]
        h = self.final_norm(h)

        logits = self.lm_head(h)
        # Final logit softcap (Gemma-style, ADR-0012). ``None`` disables.
        if self.cfg.final_logit_softcap is not None:
            c = self.cfg.final_logit_softcap
            logits = c * torch.tanh(logits / c)
        mtp_hidden = self.mtp(h, embeddings)
        mtp_logits = [self.lm_head(mh) for mh in mtp_hidden]
        if self.cfg.final_logit_softcap is not None:
            c = self.cfg.final_logit_softcap
            mtp_logits = [c * torch.tanh(ml / c) for ml in mtp_logits]
        gen_logits = self.generation_head(h)

        # Sum each MoE block's _last_aux_loss (set in DeepSeekMoE.forward when
        # the learned router fires). Hash-routed layers contribute zero.
        aux_loss: Tensor | None = None
        for block in self.blocks:
            block_aux = getattr(block.moe, "_last_aux_loss", None)
            if block_aux is None or not isinstance(block_aux, Tensor):
                continue
            aux_loss = block_aux if aux_loss is None else aux_loss + block_aux

        return {
            "logits": logits,
            "mtp_logits": mtp_logits,
            "generation_logits": gen_logits,
            "hidden": h,
            "aux_loss": aux_loss,
        }
