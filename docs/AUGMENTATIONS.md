# Augmentations to DeepSeek-V4 spine

This document tracks every deviation from a pure DeepSeek-V4 reproduction. Each augmentation has a status, a source, and a rationale. Permanent decisions (those baked into tokenizer or pretraining weights) cannot be reverted in v0.x without retrain.

Reference: `docs/DeepSeek_V4.pdf`. Spine: CSA + HCA + mHC + DeepSeekMoE + Muon + MTP + FP4 QAT + 1M context.

Status legend: `accepted-v0.1` (in scope), `deferred-v0.2` (slot reserved), `rejected` (won't add), `proposed` (under review).

---

## Multimodal vision

| ID | Augmentation | Status | Source | Rationale |
|---|---|---|---|---|
| MM-V-01 | Native multimodal pretraining (Gemma 4 / Qwen3-VL / InternVL3 pattern) | `accepted-v0.1` | InternVL3 (arXiv 2504.10479), Qwen3-VL (arXiv 2511.21631), Gemma 4 model card | Only multimodal pattern with frontier validation at >70B. Adapter-only approaches cap capability ceiling. |
| MM-V-02 | SigLIP-2-SO400M ViT encoder | `accepted-v0.1` | arXiv 2502.14786 | Best 2026 open vision encoder. 400M params, 14×14 patches, native 384/512 res. |
| MM-V-03 | RoPE-2D for native aspect ratios | `accepted-v0.1` | Pixtral 12B (arXiv 2410.07073), Qwen3-VL | Avoids center-crop quality loss. |
| MM-V-04 | DeepStack multi-level feature fusion | `accepted-v0.1` | Qwen3-VL technical report | Lower-level ViT features reach LLM, not just final-layer CLS. |
| MM-V-05 | Adaptive token budget 64-1280 per image | `accepted-v0.1` | Gemma 4, Qwen3-VL | Quality/throughput trade per image complexity. |
| MM-V-06 | is_visual mask propagated into CSA Lightning Indexer positional bias | `accepted-v0.1` | Mitigation for V4 risk — vision tokens cluster spatially | Untested: CSA top-k may over-select or starve on dense vision blocks. Force HCA to always include vision blocks via small gating bias. |
| MM-V-07 | Stage-2 ViT unfreeze | `accepted-v0.1` | InternVL3 | Recovers capability ceiling without breaking early stability. |
| MM-V-08 | Janus-Pro-style VQ image-generation head | `deferred-v0.2` | Janus-Pro (CVPR 2025) | Reserved 2,368 vocab slots in 128704-131071 range. Attached without backbone retrain in v0.2. |
| MM-V-09 | Chameleon early fusion | `rejected` | arXiv 2405.09818 | Requires QK-Norm + post-attn dropout + LN reordering — conflicts with Muon stability tricks and SwiGLU clamping. Validated only to 34B. |
| MM-V-10 | Transfusion (diffusion + AR shared backbone) | `rejected` | arXiv 2408.11039 | Bidirectional attention on image patches contradicts CSA causal sparse top-k. Validated only to 7B. |
| MM-V-11 | Dual encoder (SigLIP + InternVideo2) for video specialism | `proposed` (v0.2 candidate) | Apollo (CVPR 2025) | Better video benchmarks vs single SigLIP. Reserved second-encoder slot in v0.1 architecture. |

## Multimodal audio

| ID | Augmentation | Status | Source | Rationale |
|---|---|---|---|---|
| MM-A-01 | Whisper-large-v3 encoder, frozen v0.1 | `accepted-v0.1` | OpenAI Whisper-v3 | Industry-standard, 12.5Hz post-downsample matches Voxtral/Mimi/Qwen-Omni rate. |
| MM-A-02 | Voxtral-style continuous audio embeds (input) | `accepted-v0.1` | Mistral Voxtral 2025 (arXiv 2507.13264) | Continuous preserves paralinguistic; all 2025-2026 audio LLMs converged here. |
| MM-A-03 | 4× MLP downsample to 12.5Hz, project to V4 d_model | `accepted-v0.1` | Voxtral, Qwen3-Omni | Standard rate; 40-min audio = ~30K tokens, fits 1M context. |
| MM-A-04 | MTP head extended to "depth-K mode" | `accepted-v0.1` (config flag, K=1 default) | Moshi/Mimi (Kyutai 2024) | V4 MTP ≈ Moshi depth-transformer. Free architectural win — enables v0.2 parallel codebook prediction. |
| MM-A-05 | Mimi/Moshi-style discrete codec output | `deferred-v0.2` | Kyutai Moshi paper | 32K codec IDs reserved (16 codebooks × 2048 + headroom). Talker-head projection slot live, masked. |
| MM-A-06 | Full-duplex dual-stream training | `deferred-v0.2` | Moshi, MiniCPM-o 4.5 | Dual-stream position-IDs reserved in tokenizer config. |
| MM-A-07 | GPT-4o-style fully-unified discrete audio vocab | `rejected` | OpenAI GPT-4o (architecture leaked) | Requires from-scratch pretrain on trillions of audio tokens we don't have. |

## Multimodal video

| ID | Augmentation | Status | Source | Rationale |
|---|---|---|---|---|
| MM-Vid-01 | Shared image+video encoder (SigLIP-SO400M) | `accepted-v0.1` | Apollo (CVPR 2025), Qwen3-VL | Image instruction data transfers to video; one projector to maintain. |
| MM-Vid-02 | Qwen3-VL interleaved-MRoPE | `accepted-v0.1` | Qwen3-VL | Cleanest fit with V4 1M context; degenerate case = vanilla RoPE on h=w=0. |
| MM-Vid-03 | Dynamic FPS sampling | `accepted-v0.1` | Qwen2.5-VL | Decouples video length from token count; 2hr video at 0.5 FPS ≈ 230K tokens. |
| MM-Vid-04 | Frame-rate metadata token channel | `accepted-v0.1` | Qwen2.5-VL | Model "knows" sampling FPS without retraining. |
| MM-Vid-05 | LongVU spatiotemporal compression slot | `accepted-v0.1` (identity in v0.1) | LongVU (ICML 2025) | Pluggable layer; swap in pruner for v0.2. |
| MM-Vid-06 | V-JEPA-2 embedding-input adapter slot | `deferred-v0.2` | Meta V-JEPA 2 (arXiv 2506.09985) | Zero-init, masked. World-model bridge for trading/agent v0.2 use cases. |
| MM-Vid-07 | Video generation (Cosmos-style discrete codec) | `deferred-v0.3` | NVIDIA Cosmos (arXiv 2501.03575) | 16K codec IDs reserved. Generation is v0.3 territory. |
| MM-Vid-08 | VideoPoet AR generation | `rejected` | Google research blog | Generative-only, irrelevant for v0.1 understanding focus. |

## Tokenizer

| ID | Augmentation | Status | Source | Rationale |
|---|---|---|---|---|
| TOK-01 | Train own BBPE 131K (vs extend V3) | `accepted-v0.1` | Frontiers 2025 Ukrainian tokenization study | V3 BPE: UK fertility 2.5-3.0 vs target 1.65-1.75. Permanent decision. |
| TOK-02 | Vocab allocation: Latin 38, CJK 30, Cyrillic 12, Arabic 6, Devanagari 5, code/byte 9 | `accepted-v0.1` | Polyglot Harmony (arXiv 2509.15556), per-script fertility analysis | Balanced for 13-lang mix with UK morphological merges. |
| TOK-03 | Force-include UK chars (ї, є, ґ, апостроф) + four-case Cyrillic | `accepted-v0.1` | UK linguistics; UberText 2.0 paper | Without this UK tokenizes much worse. |
| TOK-04 | Reserved control slot ranges (vision/audio/video/memory/reasoning) — see per-modality augmentations | `accepted-v0.1` | This document | ~50K nominal slots, ~150 actually allocated v0.1. Lazy embedding row allocation. |
| TOK-05 | Tied input/output embeddings | `accepted-v0.1` | Standard practice | Halves embedding cost; required for lazy slot allocation. |
| TOK-06 | BBPE dropout=0.1 during training | `accepted-v0.1` | Sennrich-style regularization | Robustness on rare merges. |

## Multilingual

| ID | Augmentation | Status | Source | Rationale |
|---|---|---|---|---|
| LANG-01 | 13 languages: EN, ZH, RU, **UK**, ES, FR, DE, JA, KO, AR, HI, PT, IT | `accepted-v0.1` | User decision 2026-04-25 | UK explicitly added on user request. |
| LANG-02 | Pretraining mix per `project_v01_arch_decisions.md` (UK 3.5% / 9 effective epochs) | `accepted-v0.1` | Polyglot Harmony, 2024-2025 low-resource scaling | Aggressive UK upsample defensible with dedup + MT-pollution filter. |
| LANG-03 | Kobza (Goader/kobza) as primary UK corpus (60B tokens) | `accepted-v0.1` | UNLP 2025 paper | Best curated UK corpus. |
| LANG-04 | UberText 2.0 weight boost (pre-LLM era, lower MT pollution risk) | `accepted-v0.1` | UberText 2.0 (UNLP 2023) | High quality, curated, predates GPT-4 translation pollution. |
| LANG-05 | 3-stack MT-pollution detector for UK (translationese classifier + KenLM perplexity + URL heuristics) | `accepted-v0.1` | FineWeb-2 pipeline + custom layer | Critical for UK low-resource integrity. |
| LANG-06 | Temperature-scheduled epoch sampling (τ 1.0 → 0.5) for low-resource langs | `accepted-v0.1` | T5/UL2 multilingual recipe | Avoids over-fitting low-resource data in epoch 1. |

## Memory

| ID | Augmentation | Status | Source | Rationale |
|---|---|---|---|---|
| MEM-01 | Verb-oriented memory tool surface (`<\|memory_recall\|>` etc.) | `accepted-v0.1` | Letta v1, Anthropic memory tool | Backend-agnostic — vector/graph/FS swap without retrain. |
| MEM-02 | CoALA-aligned memory-type tags | `accepted-v0.1` | CoALA (arXiv 2309.02427) | Standard taxonomy: episodic/semantic/procedural/working/core/archival. |
| MEM-03 | Reserved residual side-channel slot per block (Titans/MIRAS escape hatch) | `accepted-v0.1` (gated off, identity-init) | Google Titans-MIRAS 2025 | Allows v0.2 neural-memory module without checkpoint surgery. |
| MEM-04 | In-context memory injection between `<\|memory_result\|>` markers | `accepted-v0.1` | HippoRAG-2, Letta v1, Anthropic memory consensus | No cross-attention in v0.1; composes with V4 CSA. |
| MEM-05 | Three-layer fallback (miss/error/identity-passthrough) | `accepted-v0.1` | Anthropic memory tool patterns | v0.1 shippable with zero memory backend (1M ctx carries it). |
| MEM-06 | 8-12% post-training tokens = memory-using trajectories | `accepted-v0.1` | LongMemEval, AMB benchmarks | Trains tool-use behavior even when in-context is sufficient. |
| MEM-07 | Titans-style neural memory module wired in | `deferred-v0.2` | Google Titans (arXiv 2501.00663) | Side-channel slot reserved; module attaches in v0.2 if benchmark wins. |
| MEM-08 | Engram (V4 conditional-memory) | `proposed` (v0.2 candidate) | github.com/deepseek-ai/Engram (separate Jan 2026 release, NOT in V4 paper) | Static knowledge complement to side-channel dynamic memory. |
| MEM-09 | Mamba/SSM/RWKV-style replacement for HCA | `rejected` | M1, RWKV-7 Goose | Architecturally incompatible with V4 spine. |

## Reasoning

| ID | Augmentation | Status | Source | Rationale |
|---|---|---|---|---|
| REA-01 | V4 Think modes (Non/High/Max) baseline | `accepted-v0.1` | DeepSeek V4 paper §5.1.1 | Spine pattern; preserved as-is. |
| REA-02 | GRPO with verifiable rewards + GRM for soft domains | `accepted-v0.1` | DeepSeek V4 paper §5.1.1 | Spine pattern; preserved. |
| REA-03 | On-Policy Distillation merge of specialists | `accepted-v0.1` | DeepSeek V4 paper §5.1.2 | Spine pattern; full-vocab variant per V4. |
| REA-04 | Length-controlled GRPO (Kimi K1.5 length penalty) | `accepted-v0.1` | Kimi K1.5 (arXiv 2501.12599) | Prevents "longer = better" pathology; makes Non/High/Max meaningful. |
| REA-05 | Adaptive thinking budget control (`<budget:adaptive>` token) | `accepted-v0.1` | Anthropic Claude 4.6+ adaptive thinking | Model-decided budget; future-proofs routing. |
| REA-06 | Pretraining CoT injection ~10% (front-loaded 3% → 15%) | `accepted-v0.1` | NVIDIA Front-Loading Reasoning (Sep 2025) | Highest leverage per dollar — pure data play. |
| REA-07 | Inference-time Best-of-N self-consistency (N=1 default flag) | `accepted-v0.1` | Snell et al (arXiv 2408.03314) | Available, no verifier head; measure verifier-headroom for v0.2. |
| REA-08 | Reflexion / self-correction markers (`<reflect>`/`</reflect>`) | `accepted-v0.1` (tokens reserved, agent-layer impl) | CorrectBench (arXiv 2510.16062) | Built in agent layer not model. Markers reserved. |
| REA-09 | Process Reward Models | `deferred-v0.2` | Math-Shepherd, GenPRM | DeepSeek explicitly removed PRM from R1 (reward hacking, hard to scale). Reconsider only if math specialist plateaus. |
| REA-10 | MCTS at inference (rStar-Math) | `deferred-v0.2` | rStar-Math (arXiv 2501.04519) | Brutal cost; specific to math. Add only for math specialist v0.2. |
| REA-11 | Latent reasoning (Coconut, Quiet-STaR) | `rejected` (slot `<latent>` reserved unused) | Coconut (arXiv 2412.06769); 2025 causal analysis | Worse than CoT on GSM8K; conflicts with interpretability story. |

## Other

| ID | Augmentation | Status | Source | Rationale |
|---|---|---|---|---|
| ETC-01 | EAGLE-3 speculative decoding | `deferred-v0.2` | EAGLE-3 (arXiv 2503.01840) | Pure inference optimization, no weight change. |
| ETC-02 | MCP integration alongside DSML | `accepted-v0.1` (schema only) | Anthropic MCP spec | Industry standard. DSML kept as fallback. |
| ETC-03 | SWE-RL specialist for agentic code | `accepted-v0.1` (post-training data) | SWE-RL (Meta) | Trading agent benefits; same pipeline as math/code specialists. |
| ETC-04 | Lean 4 formal-math agent | `accepted-v0.1` (specialist post-train) | DeepSeek V4 paper §5.3 + DeepSeek-Prover-V2 | V4 already does this; we follow. |

---

## Process

When implementing a component:
1. Open this document
2. Find augmentations marked `accepted-v0.1` for the relevant area (search by ID prefix: MM-V, MM-A, MM-Vid, TOK, LANG, MEM, REA, ETC)
3. Implement them as part of the component
4. After implementation, mark the augmentation with `[implemented: <commit_sha>]` in a new column

When proposing a new augmentation:
1. Add a row with status `proposed`
2. Link the source paper / blog
3. State why it closes a specific weakness in the current spine
4. Request review before promoting to `accepted-v0.1` or `deferred-v0.2`
