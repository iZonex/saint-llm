# ADR-0007: Native multimodal pretraining (not adapter-only)

- **Status**: accepted
- **Date**: 2026-04-25 (backfilled from 2026-04-15 design)
- **Deciders**: Dmytro
- **Related**: MM-V-01 in `docs/AUGMENTATIONS.md`, ADR-0003 (V4 spine)

## Context

Two patterns exist for adding multimodality to an LLM:

1. **Adapter-only** (LLaVA-style): freeze pretrained LLM, train a
   projection layer to map vision/audio embeddings into the LLM's
   hidden space. Cheap but caps capability.
2. **Native multimodal pretrain** (Llama 4, Gemma 4, Qwen3-VL,
   Pixtral 2.0, Apollo): train the LLM from scratch (or continue
   pretraining) on interleaved multimodal documents — text +
   image + audio + video together — with shared backbone seeing
   all modalities.

Frontier 2025-2026 evidence (Q1 2026 research review) shows
**only** native multimodal pretrain produces frontier-class
multimodal results at >70B; adapter-only approaches plateau.

## Decision

Native multimodal pretrain. The LLM backbone trains on:
- text-only (majority)
- image-text interleaved (OBELICS-2 style, captions, VQA)
- audio-text (transcribed audio, audio QA)
- video-text (LLaVA-Video-178K, instruction)

All modality features (vision, audio, video) are fed in via
`ModalityProjector` layers and replace placeholder
`<|image_pad|>` / `<|audio_*|>` / `<|video_*|>` token slots in
the input sequence. The model sees them as part of the same
attention stream as text tokens.

## Consequences

**Intended:**
- Capability ceiling matches Llama 4 / Gemma 4 frontier rather than
  adapter-only LLaVA tier.
- Training data pipeline produces interleaved batches end-to-end
  rather than separate text + adapter-finetune phases.
- Reserves architectural room for v0.2 generation modalities (Janus
  VQ image gen, Mimi/Moshi codec, Cosmos video) without retraining
  backbone.

**Unintended / accepted:**
- Significantly more training-data work upfront — multimodal
  data pipeline (OBELICS-2 + LLaVA-Video + audio-text) is part of
  v0.1, not adapter-finetune of v0.0.
- Compute cost is higher than adapter-only because backbone trains
  on all modalities from scratch.
- Validation harder: we can't ship a text-only model and "add vision
  later" via cheap adapter. Multimodal capability stands or falls
  with the v0.1 pretrain.

## Alternatives considered

- **Adapter-only (LLaVA-style)**, deferred multimodal pretrain.
  Rejected per MM-V-01: capability ceiling too low for frontier
  goals. Validated only at <30B; saint-llm targets >>30B-class
  capability per token.
- **Two-stage: text-only v0.0, full multimodal restart v0.1.**
  Adopted in roadmap structure but spine is *designed* for native
  multimodal from layer-0 (placeholder tokens, projector slots,
  reserved vocab). v0.0 simply doesn't *use* the multimodal path
  yet; the architecture supports it from day one.
- **Chameleon-style early fusion / Transfusion** (rejected per
  MM-V-09, MM-V-10).
- **GPT-4o-style fully unified discrete audio vocab** (rejected per
  MM-A-07: requires trillions of audio tokens we don't have).

## Implementation notes

- Backbone integration: `packages/core/src/saint_llm_core/model.py`
  `SaintLLM._embed_inputs` does the placeholder→projection
  replacement (already implemented at unit-test scale).
- Vision: `SigLIP2Wrapper` + `VisionTokenizer` + `DeepStack` already
  present; lazy-load needs to be exercised for v0.1.
- Audio: `WhisperLargeV3Wrapper` + `AudioTokenizer` (Voxtral
  4× MLP downsample to 12.5 Hz) already present; same lazy-load
  status.
- Video: defers to v0.1 — currently `MM-Vid-*` augmentations
  reserve architectural slots.
- Token-budget adaptive 64-1280 visual tokens (MM-V-05) deferred to
  v0.1.1 polish.
