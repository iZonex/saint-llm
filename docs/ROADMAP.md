# saint-llm Roadmap

Status: **draft v0** (2026-04-25). Reviewed by: pending. This is a proposal — every section is open to revision.

## Vision

Long-term: a self-sustaining decentralized multimodal multi-agent foundation
model. Not a Gemma finetune. Not a single-model wrapper. A real own-LLM with
own data pipeline, own training pipeline, own agent runtime, own decentralized
training network.

Near-term (next 12-18 months): demonstrate that the V4-derived spine
(CSA+HCA+mHC+DeepSeekMoE+Muon+MTP+FP4 QAT) actually works on real text data,
then incrementally layer multimodality, post-training, agent runtime,
decentralization on top — each layer validated end-to-end before the next is
built on top of it.

## Versioning principle

We do **not** narrow scope to "text-only forever". We **do** order work so
each version produces something validatable end-to-end on real data and
real benchmarks. Skipping ahead to multi-agent before the spine converges on
real text is what produced the over-promised baseline that this roadmap
replaces. See `docs/adr/0001-versioning-philosophy.md` (TBD) for rationale.

Each version has:
- **Scope** — what's added vs prior version
- **Deliverables** — concrete artifacts (code, weights, datasets, reports)
- **Exit criteria** — measurable conditions to declare the version done
- **Q1 2026 techniques landing here** — frontier wins from `AUGMENTATIONS.md`

## Compute reality

- **hpomen** (RTX 4080 Laptop, Ada sm89, 12 GB, FP8 native) — primary local
  validation target through v0.2. Real FP4 GEMM needs Blackwell, not here.
  See `~/.claude/.../memory/reference_hpomen_gpu.md`.
- **Cloud Blackwell** (8×B200 or similar) — needed from v0.1 to actually
  pretrain at sub-1B scale, and for FP4 training.
- **Decentralized fleet** — v0.4 onward.

## Out of scope (explicit)

- Gemma 4 / HF-weights bridge into saint-llm. Decided 2026-04-25. Trading
  finetune lives in a separate codebase.
- Pure Mamba / RWKV-7 spine replacement. Hybrid linear+full attention
  (Kimi Linear-style) is **proposed for review**, not committed.
- GPT-4o-style fully-unified discrete audio vocab. Cannot pretrain on
  trillions of audio tokens we don't have.
- Chameleon early fusion / Transfusion. Frontier converged on
  early-fusion-with-continuous-features (Llama 4, Gemma 4).
- Latent reasoning (Coconut, Quiet-STaR vanilla). Fast Quiet-STaR closed the
  inference-cost gap but is still research-grade; revisit at v0.3+.

---

# v0.0 — Spine works on real text

**Goal**: prove CSA+HCA+SWA+mHC+MoE+MTP architecture converges on real text
corpora, with real tokenizer, on real eval benchmarks. Single-modality, single
agent (no agent loop). This is what should have been v0.0 from day one.

## Scope additions

- Real tokenizer training on real corpus
- Real data pipeline (filter + dedup + lang balance) for 13 languages
- Wandb metrics + structured logging
- Real eval harness — at minimum HellaSwag / MMLU / ARC / MMLU-Pro /
  GSM8K via lm-eval-harness wrapper
- MuonClip QK-Clip (Q1 2026 stability fix; non-optional past 1B)
- u-μP scaling (replaces vanilla μP for embedding-tying-friendly init)
- Logit softcapping + learnable sink tokens (long-context stability)
- Frontloaded reasoning data injection ~10% during pretrain (REA-06 from
  current AUGMENTATIONS)

## Deliverables

| ID | Artifact | Where |
|---|---|---|
| D0.0.1 | `tokenizer.json` BBPE 131K trained on UltraFineWeb-en + HPLT3 + Kobza UK + Nemotron-CC v2 multilingual | `experiments/tokenizers/v0.0/` |
| D0.0.2 | datatrove pipeline config: dedup (LSHBloom or MinHash) + Ultra-FineWeb fastText quality classifier + lang balance per LANG-02 | `packages/data/src/saint_llm_data/pipeline/` |
| D0.0.3 | Wandb integration in `Trainer.metrics_callback` + structured logger | `packages/training/src/saint_llm_training/wandb_logger.py` |
| D0.0.4 | `lm-eval-harness` wrapper with SaintLLM HF-style adapter | `packages/eval/src/saint_llm_eval/harness/` |
| D0.0.5 | MuonClip QK-Clip integrated into Muon optimizer | `packages/optim/` |
| D0.0.6 | u-μP init scheme + per-param LR scaling | `packages/core/src/saint_llm_core/init.py` |
| D0.0.7 | Logit softcap + sink tokens in attention forward | `packages/core/src/saint_llm_core/attention/` |
| D0.0.8 | First real pretrain run: small_flash 185M on hpomen, 24-72h | `runs/v0.0/small_flash/` |
| D0.0.9 | Run report: loss curves, eval results, comparison to peers (TinyLlama-1.1B-3T, SmolLM2-135M, etc.) | `docs/reports/v0.0-report.md` |

## Exit criteria

To declare v0.0 done, all of:

1. small_flash 185M trained for ≥10B tokens (multiple epochs of small slice
   acceptable) on real text corpus
2. **Loss curve smooth, no spikes** (MuonClip working)
3. Held-out perplexity within 2× of TinyLlama-1.1B-3T at iso-tokens
4. **HellaSwag ≥40, ARC-Easy ≥45, MMLU ≥30** (random is 25). Lower bounds —
   we're not chasing SOTA at 185M.
5. Generated text qualitatively coherent (>3 sentences make sense)
6. Documented eval pipeline reproducible by running one command
7. All decisions recorded as ADRs

## Q1 2026 techniques landing in v0.0

- MuonClip ([2507.20534](https://arxiv.org/pdf/2507.20534))
- u-μP ([2407.17465](https://arxiv.org/abs/2407.17465))
- Logit softcap + sink tokens ([Streaming-LLM](https://github.com/mit-han-lab/streaming-llm))
- HPLT 3.0 + Nemotron-CC v2 + Ultra-FineWeb data ([2511.01066](https://arxiv.org/abs/2511.01066), [2505.05427](https://arxiv.org/abs/2505.05427))
- LSHBloom for dedup ([2411.04257](https://arxiv.org/abs/2411.04257))
- MorphBPE consideration for UA Cyrillic ([2502.00894](https://arxiv.org/abs/2502.00894))
- RegMix data mixture optimization ([2407.01492](https://arxiv.org/abs/2407.01492))
- Frontloaded reasoning CoT (NVIDIA, already in baseline)

## Out of v0.0

- Multimodality (vision/audio/video) — v0.1
- Post-training / RL — v0.2
- Agent runtime / tool use — v0.3
- Distributed / decentralized — v0.4
- NVFP4 native pretrain — needs Blackwell, deferred
- Hybrid linear+full attention — architectural decision pending review

---

# v0.1 — Multimodal native pretrain works end-to-end

**Goal**: vision + audio + video integrated into the spine, native multimodal
pretrain (not adapter-only) on real multimodal data, validated on real
multimodal benchmarks.

## Scope additions

- Real SigLIP-2-SO400M load + DeepStack fusion + projector on real images
- Real Whisper-large-v3 load + Voxtral 12.5 Hz on real audio
- Multimodal data pipeline (OBELICS-2 interleaved + LLaVA-Video-178K +
  audio-text pairs)
- Native multimodal pretrain recipe (Llama 4 / Gemma 4 early-fusion style;
  PLE-style placeholder-then-replace)
- Multimodal eval (MMMU, MMVet, AudioBench, VideoMME)
- FastViTHD as alternative encoder option (Apple FastVLM)
- Cosmos-Tokenizer for video input (deferred decision)
- CapRL-style synthetic dense captions

## Deliverables

| ID | Artifact |
|---|---|
| D0.1.1 | Real SigLIP-2 + Whisper integration test (lazy-loads HF weights, runs forward, projects into LLM) |
| D0.1.2 | Multimodal data loader (image+audio+text batches with placeholder-token alignment) |
| D0.1.3 | OBELICS-2 / LLaVA-Video / audio-text dataset adapters in `saint_llm_data` |
| D0.1.4 | `experiments/run_multimodal.py` end-to-end training script |
| D0.1.5 | Multimodal eval harness (MMMU, MMVet, AudioBench) |
| D0.1.6 | First multimodal pretrain run on cloud Blackwell (~1B model, ~30B tokens) |
| D0.1.7 | Run report: multimodal benchmark numbers vs Pixtral 12B / Gemma 4 4B |

## Exit criteria

1. Model produces coherent answers to MMMU questions (above-random)
2. Audio QA on AudioBench above-random
3. Multimodal pretrain run completes without divergence
4. Vision token budget adaptive (64-1280) actually working

## Q1 2026 techniques landing in v0.1

- FastViTHD vision encoder option ([2412.13303](https://arxiv.org/abs/2412.13303))
- Cosmos-Tokenizer for video input ([Cosmos-Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer))
- Mimi codec slot widened to dual-codec (XY/OmniCodec/DualCodec class) ([2506.23325](https://arxiv.org/abs/2506.23325))
- CapRL RL-trained captioner for synthetic dense captions ([CapRL](https://github.com/InternLM/CapRL))
- Llama 4 / Gemma 4 PLE-style placeholder-then-replace
- NVFP4 (block-16, E4M3) native pretrain on Blackwell ([2509.25149](https://arxiv.org/abs/2509.25149))

---

# v0.2 — Post-training, reasoning, specialists

**Goal**: instruction-following + reasoning quality on benchmarks; specialist
training pipeline; learned adaptive thinking effort.

## Scope additions

- SFT pipeline runs on real data (we have the math; need driver + data)
- GSPO replaces token-level GRPO importance ratio (MoE stability)
- DAPO four patches on top of GRPO core
- Critique-GRPO for plateau-breaking
- Tree-GRPO for agentic-trajectory RL
- Adaptive thinking head — learned `effort ∈ {low, medium, high, xhigh, max}`
  router (Claude 4.7-style), not static V4 Non/High/Max tag
- OPD multi-teacher specialist merge with TIP token-importance + Structured
  Agent Distillation segments
- GRM for soft-domain rewards + Rubrics-as-Rewards
- Verbalized PRM (ThinkPRM-style) for math/code chain verification
- Lean 4 + SWE-RL specialists (REA-already-in-baseline + DeepSWE patterns)
- Reasoning eval (AIME, GPQA, MATH, LiveCodeBench, HumanEval+)

## Deliverables

| ID | Artifact |
|---|---|
| D0.2.1 | `experiments/run_sft.py` driver using existing SFT data utilities |
| D0.2.2 | GSPO + DAPO patches in `posttraining/grpo.py` |
| D0.2.3 | `posttraining/critique_grpo.py` |
| D0.2.4 | `posttraining/tree_grpo.py` |
| D0.2.5 | Adaptive thinking head module + training recipe |
| D0.2.6 | OPD multi-teacher merge driver |
| D0.2.7 | GRM training utility + Rubrics-as-Rewards data format |
| D0.2.8 | ThinkPRM-style verbalized PRM specialist |
| D0.2.9 | Lean 4 / SWE-RL specialist post-training scripts |
| D0.2.10 | Reasoning eval harness (AIME, GPQA, MATH, LCB) |

## Exit criteria

1. Post-trained variant beats v0.1 base by ≥10pp on AIME 2024, ≥5pp on GPQA
2. SFT trained model passes IFEval > 60%
3. Specialist merge produces single model with ≤5% regression on per-domain
   eval vs individual specialists

## Q1 2026 techniques landing in v0.2

- GSPO ([2507.18071](https://arxiv.org/abs/2507.18071))
- DAPO ([2503.14476](https://arxiv.org/abs/2503.14476))
- Dr.GRPO + CE-GPPO ([2503.20783](https://arxiv.org/abs/2503.20783), [2509.20712](https://arxiv.org/abs/2509.20712))
- Critique-GRPO ([2506.03106](https://arxiv.org/abs/2506.03106))
- Tree-GRPO ([2509.21240](https://arxiv.org/abs/2509.21240))
- Adaptive thinking head (Claude 4.7-style) ([Anthropic Opus 4.6/4.7 system cards](https://www.anthropic.com/claude-opus-4-6-system-card))
- TIP token-importance OPD ([2604.14084](https://arxiv.org/abs/2604.14084))
- Structured Agent Distillation ([2505.13820](https://arxiv.org/abs/2505.13820))
- ThinkPRM verbalized PRM ([2504.16828](https://arxiv.org/abs/2504.16828))
- Self-critic reward bootstrapped from verifiable tasks (Kimi K2)
- Temperature-decay schedule in RL (Kimi K2)

---

# v0.3 — Agent runtime + tool use, integrated end-to-end

**Goal**: agent runtime is real (not just the basic Agent/Policy/Tool/Runtime
we already scaffolded), integrated with our model + tools + sandbox, validated
on agent benchmarks.

## Scope additions

- Code Mode tool execution (Cloudflare/Anthropic pattern) — replaces DSML XML
  entirely; LLM writes TypeScript that calls MCP tools in sandbox
- MCP server adapter for our model's tool-call schema
- Anthropic Skills folder pattern (SKILL.md + scripts + progressive
  disclosure)
- Real sandbox (E2B-style or own Firecracker pool with libdsec)
- A2A protocol compatibility for cross-vendor multi-agent
- Memory tool (Anthropic-style, file-based, persists across compaction)
- Engram conditional memory as third sparsity axis (after MoE + sparse
  attention)
- EAGLE-3 + P-EAGLE speculative decoding for agent inference
- NSA (native sparse attention) — replaces some of CSA+HCA+SWA hierarchy or
  sits alongside as kernel choice
- Multi-turn agentic RL (GTPO-style turn-level reward)
- Agent benchmark suite (SWE-Bench Pro, OSWorld-Verified, τ²-bench,
  Terminal-Bench, BrowseComp-Plus, GAIA)

## Deliverables

| ID | Artifact |
|---|---|
| D0.3.1 | Code Mode runtime: tool→TS-API generator + sandbox executor |
| D0.3.2 | MCP server adapter for SaintLLMPolicy |
| D0.3.3 | Anthropic Skills loader (folder + SKILL.md + progressive disclosure) |
| D0.3.4 | Sandbox tier implementations: function_call, container, microvm |
| D0.3.5 | A2A protocol adapter for cross-runtime agent communication |
| D0.3.6 | Memory tool implementation + persistent file backend |
| D0.3.7 | Engram conditional memory module + integration |
| D0.3.8 | EAGLE-3 + P-EAGLE speculative decoding in inference package |
| D0.3.9 | NSA-style native sparse attention kernel (kernels package) |
| D0.3.10 | GTPO multi-turn agentic RL trainer |
| D0.3.11 | Agent benchmark harness (SWE-Bench Pro etc.) |
| D0.3.12 | Run report: agent benchmark numbers vs Claude Sonnet 4.6 / Opus 4.7 |

## Exit criteria

1. SWE-Bench Pro ≥30% (frontier ~64-78%; we target competence not parity)
2. OSWorld-Verified ≥25%
3. Tool-using agent runs 10-step task without runtime error
4. Code Mode reduces input-token cost ≥80% vs naive tool-call schema
5. Engram integration produces measurable benchmark gain (target: MMLU +1pp,
   BBH +2pp, paper claims +3-5pp)

## Q1 2026 techniques landing in v0.3

- Code Mode ([Cloudflare](https://blog.cloudflare.com/code-mode-mcp/), [Anthropic](https://www.anthropic.com/engineering/code-execution-with-mcp))
- Anthropic Skills ([Anthropic engineering](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills))
- A2A protocol ([Linux Foundation](https://www.linuxfoundation.org/press/...))
- Engram ([2601.07372](https://arxiv.org/abs/2601.07372), [GitHub](https://github.com/deepseek-ai/Engram))
- EAGLE-3 + P-EAGLE ([Red Hat](https://developers.redhat.com/articles/2025/07/01/...), [AWS](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/))
- NSA ([2502.11089](https://arxiv.org/abs/2502.11089))
- GTPO ([2511.14846](https://arxiv.org/abs/2511.14846))
- Live-SWE-agent self-evolution patterns ([2511.13646](https://arxiv.org/abs/2511.13646))
- ORS environment standard ([Open Reward Standard](https://github.com/...))

---

# v0.4 — Distributed + decentralized training

**Goal**: training scales beyond hpomen to multi-node, then to permissionless
decentralized network. The "BTC-style organism" foundation.

## Scope additions

- FSDP2 + EP for in-house multi-GPU
- DeepEP all-to-all + SGLang Single-Batch Overlap for MoE
- USP (Ulysses + Ring hybrid) for context parallel; Zig-Zag Ring for load
  balance
- ZBV schedule when EP not needed (faster than DualPipe)
- DiLoCo-family async distributed training
- NoLoCo (Gensyn) — no global blocking comm, log(N) all-reduce speedup
- Templar SparseLoCo (97% gradient compression) for low-bandwidth
- Pluralis Node0 patterns for permissionless model-parallel
- Bittensor SN integration (research; pick one of SN3 Templar / SN9 Macrocosmos)
- Block-store + reproducible-pipeline infra so external nodes can join

## Deliverables

| ID | Artifact |
|---|---|
| D0.4.1 | FSDP2 + EP wrappers in `saint_llm_distributed` |
| D0.4.2 | DeepEP integration |
| D0.4.3 | USP + Zig-Zag Ring CP |
| D0.4.4 | DiLoCo + NoLoCo trainers |
| D0.4.5 | Templar SparseLoCo gradient compressor |
| D0.4.6 | Pluralis-style permissionless join protocol |
| D0.4.7 | Bittensor SN-compatible miner reference impl |
| D0.4.8 | First multi-node decentralized run completion (≥4 nodes, ≥1B model) |

## Exit criteria

1. Multi-node FSDP run on cloud completes
2. ≥4-node decentralized run completes with no convergence loss vs centralized
3. SparseLoCo bandwidth measured 97%-class compression in practice
4. External contributor successfully joins a training run

## Q1 2026 techniques landing in v0.4

- DualPipeV / ZBV ([Sea AI](https://sail.sea.com/blog/articles/63))
- DeepEP ([deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP))
- SGLang Single-Batch Overlap
- USP + Zig-Zag Ring ([feifeibear/long-context-attention](https://github.com/feifeibear/long-context-attention))
- NoLoCo ([gensyn-ai/noloco](https://github.com/gensyn-ai/noloco))
- Templar SparseLoCo ([Bittensor SN3](https://x.com/bittingthembits/status/2034904649476583593))
- Pluralis Node0 ([Pluralis blog](https://blog.pluralis.ai/p/beyond-top-k-pipeline-parallelism))
- Decoupled DiLoCo ([2604.21428](https://arxiv.org/abs/2604.21428))

---

# v1.0 — Autonomous decentralized organism

**Goal**: self-sustaining decentralized training + inference network with
economic incentive layer. The original vision.

## Scope additions

- Economic incentive layer (token / reputation / contribution accounting)
- Permissionless inference serving
- On-chain or off-chain reward distribution
- Continuous learning from real usage
- Multi-specialist self-improvement loop
- Production agent platform on top

## Deliverables, exit criteria

To be detailed at v0.4 close. Too far out to specify now.

---

# Cross-version concerns

## Testing discipline

- Every PR keeps `./scripts/sync-and-test-gpu.sh` green on hpomen
- CUDA-only / slow tests: gated with `pytest.mark.gpu` / `pytest.mark.slow`
- Eval results: every version exit produces a `docs/reports/v0.X-report.md`
  with reproducible commands

## Documentation discipline

- Every architectural decision: ADR in `docs/adr/`
- Every new augmentation / Q1-2026 pick: row in `docs/AUGMENTATIONS.md` with
  status (`proposed` → `accepted-vX.Y` after review)
- Every subsystem: spec in `docs/specs/` before implementation begins
- Every version exit: report in `docs/reports/`

## Decision process

1. Propose: PR adds row to AUGMENTATIONS with status `proposed` + ADR draft
2. Review: discussion in PR, validate against current spine and roadmap
3. Accept / reject / defer: status changes to `accepted-vX.Y` / `rejected` /
   `deferred-vY.Z` with rationale recorded
4. Implement: against the accepted ADR; tag commit hash next to the
   AUGMENTATIONS row when shipped

## Things that get rolled back to "proposed" if review pushes back

These are in the roadmap above but should be re-validated by the user before
implementation:

- **Hybrid linear+full attention (Kimi Linear / Gated DeltaNet)** — open
  question: do we keep CSA+HCA+SWA as is, or replace 75% with KDA-style
  channel-wise gated linear attention? Big architectural change. Currently
  not in any version above; proposed for v0.0/v0.1 review.
- **NVFP4 vs MXFP4** — depends on Blackwell access timeline. Roadmap has
  NVFP4 in v0.1; could stay MXFP4 until v0.4.
- **Replace V4 Non/High/Max with learned effort head** — Anthropic-style.
  Roadmap has it in v0.2 as additive; could be deferred to v0.3 or rejected.
- **Penguin-Encoder (LLM-init vision encoder)** — research bet; not in
  roadmap above, would be a v0.1 alternative to SigLIP-2.
- **BLT byte-patch tokenizer** — eliminates tokenizer entirely; deferred,
  too disruptive for v0.0.

## Out-of-roadmap research tracking

- HRM (Hierarchical Reasoning Model) — track for orthogonal reasoning module
- VL-JEPA continuous embedding prediction — track for v0.2+ multimodal
- V-JEPA-2 as motion encoder — track for v0.2+ video
- SPIRAL self-play — track for reasoning post-training experiment

---

# Open questions for review

Before this roadmap goes from `draft v0` to `accepted`:

1. Confirm version split: 0.0 = text spine, 0.1 = multimodal, 0.2 =
   post-training, 0.3 = agents, 0.4 = decentralized, 1.0 = organism. OK?
2. Hybrid linear+full attention: in or out of roadmap?
3. Decentralized track: which Bittensor SN to target (or none)?
4. Which compute do we have / can we get for v0.1+ (cloud Blackwell)?
5. UNLP 2026 (May 29-30 Lviv) — submit anything UA-tokenizer-related?
