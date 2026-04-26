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
| MM-V-03 | RoPE-2D for native aspect ratios | `accepted-v0.1` *(impl deferred to v0.1.1 polish — requires SigLIP-2 internal patch)* | Pixtral 12B (arXiv 2410.07073), Qwen3-VL | Avoids center-crop quality loss. |
| MM-V-04 | DeepStack multi-level feature fusion | `accepted-v0.1` | Qwen3-VL technical report | Lower-level ViT features reach LLM, not just final-layer CLS. |
| MM-V-05 | Adaptive token budget 64-1280 per image | `accepted-v0.1` *(impl deferred to v0.1.1 polish — fixed token count from SigLIP-2 native res for now)* | Gemma 4, Qwen3-VL | Quality/throughput trade per image complexity. |
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

## Architecture / spine

Q1 2026 frontier proposals beyond the V4 spine. ID prefix: `ARCH`.

| ID | Augmentation | Status | Source | Rationale |
|---|---|---|---|---|
| ARCH-01 | Hybrid linear+full attention (Kimi Linear / Gated DeltaNet 3:1) | `proposed` (review for v0.0/v0.1) | arXiv 2510.26692 (Kimi Linear), 2412.06464 (GDN, ICLR 25) | Outperforms full MLA at fair compute, 6× decode at 1M, open weights. Big architectural change vs current CSA+HCA+SWA. Decide via ADR. |
| ARCH-02 | Logit softcapping (Gemma-style) + learnable sink tokens | `accepted-v0.0` (ADR-0012) | Gemma 4, arXiv 2511.21377, mit-han-lab/streaming-llm | Long-context stability without QK-Norm regression on long-tail tasks. Cheap. |
| ARCH-03 | NSA — Native Sparse Attention as kernel choice | `proposed` (v0.3) | arXiv 2502.11089 (DeepSeek) | Hierarchical (compressed coarse + selective fine + sliding-window) AND natively trainable. Aligns with our CSA+HCA+SWA hierarchy idea but trainable. |

## Optimizer

ID prefix: `OPT`. Beyond the existing Muon + AdamW splitter.

| ID | Augmentation | Status | Source | Rationale |
|---|---|---|---|---|
| OPT-01 | MuonClip — QK-Clip post-update rescale of W_q/W_k when max attention logit > τ | `accepted-v0.0` (ADR-0010) | arXiv 2507.20534 (Kimi K2), [QK-Clip blog](https://frontier.soket.ai/posts/muon_qk_clip/) | 15.5T tokens zero loss spikes. Single most copyable Q1 2026 stability trick. ~50 LOC on top of existing Muon. |
| OPT-02 | u-μP — unit-scaled μP, embedding-tying-friendly | `accepted-v0.0` (ADR-0011) | arXiv 2407.17465 | Replaces vanilla μP, fixes embedding LR pathology, cleaner init for tied embeddings (TOK-05). |
| OPT-03 | SOAP+Muon iterative whitening (defer until base run validates) | `deferred-v0.2` | [Vyas SOAP+Muon notes](https://nikhilvyas.github.io/SOAP_Muon.pdf) | Most credible "what's next" optimizer once Muon+MuonClip is solid; not frontier-deployed yet. |
| OPT-04 | APOLLO-Mini for VRAM-bound finetune runs (hpomen 12 GB) | `proposed` (v0.0 utility) | arXiv 2412.05270 (MLSys 25 OPHM) | SGD-like memory, AdamW-quality. Train/finetune on 12 GB. Useful for hpomen validation runs even if not primary optimizer. |

## RL / post-training

ID prefix: `RL`. Beyond GRPO baseline + length-controlled GRPO + OPD.

| ID | Augmentation | Status | Source | Rationale |
|---|---|---|---|---|
| RL-01 | GSPO — sequence-level importance ratio + length normalization | `accepted-v0.2` (ADR-0014) | arXiv 2507.18071 (Qwen3) | Specifically designed for MoE RL stability. Replaces token-level GRPO ratios. Powers Qwen3 family. Highly relevant since saint-llm is MoE. |
| RL-02 | DAPO — Clip-Higher / Dynamic Sampling / Token-Level PG / Overlong Reward Shaping | `accepted-v0.2` (ADR-0015) | arXiv 2503.14476 (ByteDance Seed) | Four orthogonal patches on top of GRPO. AIME 50 vs DeepSeek-R1-Zero-Qwen-32B's 47, 50% steps. Almost free additions. |
| RL-03 | Dr.GRPO + CE-GPPO unbiased loss tweaks | `proposed` (v0.2, cheap) | arXiv 2503.20783, 2509.20712 | Removes length-normalization and per-group std bias. Two-line deletes. |
| RL-04 | Critique-GRPO — numerical advantage + natural-language critiques | `accepted-v0.2` (ADR-0016) | arXiv 2506.03106 | +15-22% Pass@1 over GRPO on Qwen2.5-Math-7B / Qwen3-8B. Breaks GRPO plateau on hard reasoning. |
| RL-05 | Tree-GRPO — node = full agent step; intra+inter-tree advantages | `proposed` (v0.3) | arXiv 2509.21240 (ICLR 26) | Outcome rewards yield process-style supervision via tree structure. 1/4 rollout budget. Highly relevant for agentic RL. |
| RL-06 | GTPO — turn-level reward + return-based advantage for multi-turn tool agents | `proposed` (v0.3) | arXiv 2511.14846 | Direct upgrade to GRPO for agentic tool-use RL specifically (vs math/code). |
| RL-07 | Adaptive thinking head — learned `effort ∈ {low,medium,high,xhigh,max}` router replaces static V4 Non/High/Max tag | `accepted-v0.2` (ADR-0017, supersedes REA-05) | Anthropic Claude Opus 4.6/4.7 system cards | Frontier deprecated explicit budget tokens entirely; learned effort head trained jointly with RL. |
| RL-08 | TIP — Token Importance in OPD | `proposed` (v0.2 OPD enhancement) | arXiv 2604.14084 | Weights tokens by importance during distillation. Cheap accuracy bump for our planned multi-teacher OPD. |
| RL-09 | Structured Agent Distillation — `[REASON]` / `[ACT]` segment-specific OPD losses | `proposed` (v0.2 specialist merge) | arXiv 2505.13820 | Important for agentic specialists distilled back to base. |
| RL-10 | Verbalized PRM (ThinkPRM) — process reward as reasoning model | `proposed` (v0.2, supersedes REA-09 deferred status) | arXiv 2504.16828 | DeepSeek dropped naive step-annotation PRMs. Verbalized CoT-PRM trained on 1% PRM800K labels beats discriminative verifiers. PRM is back, smarter. |
| RL-11 | Self-critic reward bootstrapped from verifiable tasks then transferred to non-verifiable | `proposed` (v0.2) | Kimi K2 tech report | Underused outside Moonshot. |
| RL-12 | Temperature-decay schedule in RL | `proposed` (v0.2, cheap) | Kimi K2 tech report | High temp early for exploration, decay later. |
| RL-13 | Rubrics-as-Rewards (RaR) for subjective domains | `proposed` (v0.2) | arXiv 2507.17746, OpenRubrics arXiv 2510.07743 | Extends RLVR into soft domains; pairs with planned GRM. |
| RL-14 | Imperfect-verifier RLVR | `proposed` (v0.2) | arXiv 2510.00915 | Important once we scale beyond toy math; handles noisy verifiable rewards. |
| RL-15 | Open Reward Standard (ORS) — MCP-shaped environment interface | `proposed` (v0.3) | [Open Reward Standard](https://github.com/openreward) | Emerging standard interface trainer↔environment. Aligns with our multi-agent direction. |

## Data pipeline

ID prefix: `DATA`. Beyond the existing TOK / LANG plan.

| ID | Augmentation | Status | Source | Rationale |
|---|---|---|---|---|
| DATA-01 | HPLT 3.0 — 30T tokens / ~200 (lang,script) pairs / 16T EN, 13T non-EN | `accepted-v0.0` (ADR-0018) | arXiv 2511.01066 | New gold standard for multilingual base. Replaces mC4/CulturaX-tier corpora. |
| DATA-02 | Nemotron-CC v2 — synthetic Diverse-QA translated into 15 languages | `accepted-v0.0` (ADR-0018) | [Nemotron-CC v2 HF](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) | Boosts Global-MMLU +10. 6.3T tokens (4.4T original + 1.9T synthetic). |
| DATA-03 | Ultra-FineWeb fastText quality classifier (256-d, n=3, 3 epochs) | `accepted-v0.0` (ADR-0019) | arXiv 2505.05427 | Orders of magnitude cheaper than LLM-classifiers. Real cost win at our budget. |
| DATA-04 | LSHBloom dedup — Bloom-filter approximation of MinHash-LSH (12× faster) | `accepted-v0.0` (ADR-0019) | arXiv 2411.04257 | Internet-scale dedup at lower cost. |
| DATA-05 | RegMix data mixture optimization (regression on small-proxy) | `accepted-v0.0` (ADR-0020) | arXiv 2407.01492 | Beats DoReMi at ~10% compute. Big win for 13-language pretrain weighting. |
| DATA-06 | OBELICS-2 + CoMM coherent interleaved image-text | `proposed` (v0.1) | [OBELICS-2 HF](https://github.com/huggingface/OBELICS), CVPR 25 CoMM | Multimodal interleaved web docs, drop-in replacement for OBELICS-1. |
| DATA-07 | LLaVA-Video-178K + LLaVA-OneVision-1.5 | `proposed` (v0.1) | [LLaVA-Video](https://llava-vl.github.io/blog/2024-09-30-llava-video/), [LLaVA-OneVision-1.5](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5) | Synthetic video instruction at scale. |
| DATA-08 | CapRL — RL-trained dense image captioner for synthetic captions | `proposed` (v0.1) | [CapRL repo](https://github.com/InternLM/CapRL) (ICLR 26) | Visibly denser/more accurate captions than SFT-only Qwen-VL captioners. |
| DATA-09 | Multimodal data quality classifier | `proposed` (v0.1) | arXiv 2510.15162 | Unified synthetic-data quality classifier for multimodal corpora. |
| DATA-10 | MorphBPE consideration for UA Cyrillic / agglutinative languages | `accepted-v0.0 (gate)` (ADR-0021) | arXiv 2502.00894 | Measurement gate during D0.0.1: triggers MorphBPE evaluation only if plain BBPE UA fertility > 1.85. Default path is plain BBPE 131K. |

## Tokenizer (additions)

| ID | Augmentation | Status | Source | Rationale |
|---|---|---|---|---|
| TOK-07 | BLT (Byte Latent Transformer) — entropy-based byte patches, no tokenizer | `deferred` (architectural rewrite, revisit at v1.0) | arXiv 2412.09871 | Eliminates tokenizer decision entirely at 8B+. Matches Llama 3 8B with -50% inference FLOPs. Too disruptive for v0.0; track for future. |

## Inference

ID prefix: `INF`. Beyond the existing KV-cache + samplers.

| ID | Augmentation | Status | Source | Rationale |
|---|---|---|---|---|
| INF-01 | EAGLE-3 + P-EAGLE speculative decoding | `proposed` (v0.3) | EAGLE-3 NeurIPS 25, P-EAGLE 2026 (AWS Blackwell) | 3-6.5× decode speedup, plug-in draft head. vLLM has it. |
| INF-02 | Disaggregated prefill/decode (SGLang `--disaggregation-mode`) | `proposed` (v0.3) | [SGLang docs](https://sgl-project.github.io/) | For long-prompt high-concurrency serving. |
| INF-03 | Tree Attention for multi-GPU decode | `proposed` (v0.3) | arXiv 2408.04093 (Zyphra) | 8× faster cross-device, 2× less peak mem vs Ring. |
| INF-04 | DuoServe-MoE — dual-phase expert prefetch + cache scheduling | `proposed` (v0.3) | arXiv 2509.07379 | Pipelines CPU→GPU expert fetch behind compute for MoE inference. |

## Memory

| ID | Augmentation | Status | Source | Rationale |
|---|---|---|---|---|
| MEM-10 | Engram conditional memory — N-gram embeddings in system RAM, O(1) lookup, U-shaped sparsity scaling | `proposed` (v0.3) | arXiv 2601.07372 (DeepSeek+Peking U), [GitHub](https://github.com/deepseek-ai/Engram) | New sparsity axis alongside MoE. MMLU +3.4, BBH +5.0, ARC-Challenge +3.7, HumanEval +3.0 at 27B. Open code. Pairs with MEM-03 side-channel slot. **The most distinctive proposal in this batch.** |
| MEM-11 | Anthropic memory tool + context-editing + server-side compaction pattern | `proposed` (v0.3, supersedes MEM-04 baseline) | [Anthropic memory docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool) | File-based store in our infra; persists across compaction boundaries. |

## Agents / runtime / tool use

ID prefix: `AGT`.

| ID | Augmentation | Status | Source | Rationale |
|---|---|---|---|---|
| AGT-01 | Code Mode — convert MCP tools into TypeScript API; LLM writes code that calls them in sandbox | `accepted-v0.3` (ADR-0013) | [Cloudflare blog](https://blog.cloudflare.com/code-mode-mcp/), [Anthropic engineering](https://www.anthropic.com/engineering/code-execution-with-mcp) | 98.7% input-token reduction, 30-40% latency drop. **Replaces the DSML XML idea entirely** — nobody outside our docs uses XML tool-call schemas. |
| AGT-02 | Anthropic Skills — folder-based expertise with SKILL.md + scripts + progressive disclosure | `proposed` (v0.3) | [Anthropic Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) | Trivial to copy, dramatic context reduction for long-running agents. |
| AGT-03 | A2A protocol compatibility (Linux Foundation Agentic AI Foundation, 150+ orgs) | `proposed` (v0.3) | [A2A milestone PR](https://www.prnewswire.com/news-releases/a2a-protocol-surpasses-150-organizations...) | Cross-vendor multi-agent communication. Signed agent cards. Production-grade. |
| AGT-04 | OASF — Open Agentic Schema Framework (capability/identity/messaging discovery) | `proposed` (v0.3) | [agntcy/oasf](https://github.com/agntcy/oasf) | Fills "what does this agent do" gap MCP doesn't address. |
| AGT-05 | Magentic orchestrator pattern (Microsoft) — Orchestrator + WebSurfer + FileSurfer + Coder + Terminal | `proposed` (v0.3 reference) | [MS Learn Magentic](https://learn.microsoft.com/en-us/agent-framework/user-guide/workflows/orchestrations/magentic) | Pattern reference for our Runtime evolution beyond round-robin. |
| AGT-06 | Sandbox tier: K8s Agent Sandbox (gVisor + Kata pluggable) | `proposed` (v0.3) | [k8s blog Mar 26](https://kubernetes.io/blog/2026/03/20/running-agents-on-kubernetes-with-agent-sandbox/) | Standardization vector for our planned function_call/container/microvm/fullvm. Watch closely. |
| AGT-07 | E2B / Daytona / Modal sandbox-as-service alternative to building libdsec | `proposed` (v0.3 evaluation) | [Northflank E2B/Daytona](https://northflank.com/blog/daytona-vs-e2b-ai-code-execution-sandboxes) | Decide whether to build libdsec from scratch or buy. |
| AGT-08 | Live-SWE-agent — agents that self-evolve at deployment | `proposed` (v0.3 research track) | arXiv 2511.13646 | Frontier agentic-RL pattern for production self-improvement. |

## ETC update

| ID | Augmentation | Status | Source | Rationale |
|---|---|---|---|---|
| ETC-02 | MCP integration alongside DSML XML fallback | `revised` (ADR-0013): DSML XML rejected; MCP-only | (this document, 2026-04-25) | DSML XML schema not used by any frontier project. Code Mode (AGT-01) is the actual frontier pattern for tool calls. Drop DSML, keep MCP. |

## Distributed / decentralized

ID prefix: `DIST`.

| ID | Augmentation | Status | Source | Rationale |
|---|---|---|---|---|
| DIST-01 | FSDP2 + EP via DeepEP all-to-all + SGLang Single-Batch Overlap | `proposed` (v0.4) | [DeepEP](https://github.com/deepseek-ai/DeepEP), [SGLang EP docs](https://docs.sglang.io/advanced_features/expert_parallelism.html) | Open-source frontier MoE training stack. |
| DIST-02 | DualPipeV / ZBV (Sea AI) — bubble-reduced pipeline, ZBV when EP not needed | `proposed` (v0.4) | [Sea AI ZBV blog](https://sail.sea.com/blog/articles/63), [DualPipe GitHub](https://github.com/deepseek-ai/DualPipe) | DualPipe successor for non-EP runs. |
| DIST-03 | USP unified context parallel + Zig-Zag Ring | `proposed` (v0.4) | [feifeibear/long-context-attention](https://github.com/feifeibear/long-context-attention) | Hybridizes Ulysses+Ring; Zig-Zag fixes load imbalance. |
| DIST-04 | Decoupled DiLoCo + NoLoCo (Gensyn) | `proposed` (v0.4) | arXiv 2604.21428, [gensyn-ai/noloco](https://github.com/gensyn-ai/noloco) | Lock-step-free DiLoCo. NoLoCo: 4% faster convergence, log(N) all-reduce, 10× over standard at 1024 GPUs. |
| DIST-05 | Templar SparseLoCo — 97% gradient compression, Covenant-72B on 70 nodes | `proposed` (v0.4 — **the credible BTC-organism foundation**) | [Templar Mar 2026](https://x.com/bittingthembits/status/2034904649476583593), Bittensor SN3 | First real evidence decentralized training works at non-trivial scale. Lowest-bandwidth viable algorithm. |
| DIST-06 | Pluralis Node0 patterns — permissionless model-parallel pretraining | `proposed` (v0.4) | [Pluralis blog](https://blog.pluralis.ai/p/beyond-top-k-pipeline-parallelism) | First Internet-scale pipeline-parallel run. ≥99% compression on output projections via shared low-rank subspace. |
| DIST-07 | Bittensor SN integration — pick one of SN3 Templar / SN9 Macrocosmos | `proposed` (v0.4 research) | (per-SN docs) | Decide which decentralized network we plug into. |

## Quantization (additions)

ID prefix: `QNT` (was implicit in `quant.py` etc., now explicit).

| ID | Augmentation | Status | Source | Rationale |
|---|---|---|---|---|
| QNT-01 | NVFP4 native pretraining (block-16, E4M3 scale) on Blackwell | `proposed` (v0.1, on Blackwell only) | arXiv 2509.25149, 2505.14669 (Quartet), 2505.19115 | FP8-equivalent loss curves for FP4 training. Better numerics than MXFP4 in published ablations. Currently we have MXFP4 emulated; switch when Blackwell available. |
| QNT-02 | Four Over Six — adaptive block scaling for NVFP4 | `proposed` (v0.1, follows QNT-01) | arXiv 2512.02010 | Better downstream than vanilla NVFP4 on Llama/Qwen. |

## Reasoning (additions / revisions)

| ID | Augmentation | Status | Source | Rationale |
|---|---|---|---|---|
| REA-05 | Adaptive thinking budget control (`<budget:adaptive>` token) | `revised` (ADR-0017): superseded by RL-07 learned effort head | (this document, 2026-04-25) | Anthropic deprecated explicit budget tokens entirely. Replace with learned effort head (RL-07). |
| REA-09 | Process Reward Models | `revised` from `deferred-v0.2` to `proposed` (v0.2) | arXiv 2504.16828 (ThinkPRM) | Verbalized PRMs sidestep DeepSeek's R1 rejection of naive step-annotation. See RL-10. |
| REA-12 | SPIRAL self-play on zero-sum games for general reasoning | `proposed` (v0.2 research track) | arXiv 2506.24119 | +8.6% math, +8.4% general reasoning on Qwen3-4B-Base, beats SFT on 25k expert traces using only games. |
| REA-13 | Meta-CoT — model the meta-reasoning that produces a CoT (search/backtrack/verify) | `proposed` (v0.2) | arXiv 2501.04682 | Evidence o1/o3 implicitly do this. Process-supervised SFT + search produces Meta-CoT traces. |
| REA-14 | HRM (Hierarchical Reasoning Model) — orthogonal recurrent reasoning module | `proposed` (research track) | arXiv 2506.21734, analysis 2601.10679 | 27M params, 1000-sample training, near-perfect Sudoku/maze without CoT. Track but don't commit. |

---

## Process

When implementing a component:
1. Open this document
2. Find augmentations marked `accepted-vX.Y` for the relevant area (search by ID prefix: ARCH, OPT, MM-V, MM-A, MM-Vid, TOK, LANG, MEM, REA, RL, DATA, INF, AGT, DIST, QNT, ETC)
3. Implement them as part of the component
4. After implementation, mark the augmentation with `[implemented: <commit_sha>]` in a new column

When proposing a new augmentation:
1. Add a row with status `proposed`
2. Link the source paper / blog
3. State why it closes a specific weakness in the current spine
4. Request review before promoting to `accepted-vX.Y` or `deferred-vY.Z`

When superseding / revising an existing entry:
1. Change its status to `revised` or `deferred-vY.Z` or `rejected`
2. Reference the superseding row's ID + this document's date
3. Link a new ADR explaining the change
