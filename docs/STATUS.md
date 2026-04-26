# Project status

Last updated: 2026-04-25.

## Honest current state

**Version**: pre-v0.0. Scaffolding phase. No real training has happened.

**What works (unit-tested, never run end-to-end on real data):**
- DeepSeek-V4-derived architecture (CSA + HCA + SWA + mHC + DeepSeekMoE +
  MTP) compiles and forward/backward run on synthetic input
- Quantization chain (FP8 + MXFP4 + grouped GEMM) compiles
- Trainer + LR schedule + checkpoint rotator + grad accum + mixed precision
- KV cache stages 1-6 (heterogeneous SWA / HCA / CSA)
- SFT data plumbing + GRPO math + deterministic attention math
- Multi-agent runtime v0.0 (Agent / Policy / Tool / Runtime)
- Real `SigLIP2Wrapper` and `WhisperLargeV3Wrapper` exist but the lazy-load
  has never been triggered with real weights
- 506/506 tests passing on hpomen as of 2026-04-25

**What does NOT exist (despite being in package descriptions or planned):**
- Real `tokenizer.json` from a real corpus
- datatrove text data pipeline (datatrove is a dependency, never used)
- Real eval harness (we have perplexity on a 4-line synthetic corpus only)
- Wandb integration (in deps, not wired)
- Real multimodal data loader
- Multi-agent runtime beyond round-robin (no A2A, no Code Mode, no skills)
- Tool execution / sandbox (function_call / container / microvm / fullvm
  packages all empty)
- Distributed / FSDP / EP (`saint_llm_distributed` package empty)
- Decentralized training (no DiLoCo, no SparseLoCo, no Bittensor SN code)
- GRM / OPD / teacher offload / reasoning modes / Quick Instruction —
  empty namespaces in `posttraining/__init__.py` docstring
- Anticipatory routing for MoE (in `training/pyproject.toml`, not built)

## What's been planned (Q1 2026 research review)

See `docs/ROADMAP.md` for the full plan and `docs/AUGMENTATIONS.md` for
proposals at row-level. New Q1 2026 picks (all status `proposed`, awaiting
review):

| Domain | Pick | Target |
|---|---|---|
| Optimizer | MuonClip QK-Clip | v0.0 |
| Init | u-μP | v0.0 |
| Stability | Logit softcap + sink tokens | v0.0 |
| Data | HPLT 3.0 + Nemotron-CC v2 + Ultra-FineWeb | v0.0 |
| Data | LSHBloom dedup, RegMix mixture | v0.0 |
| Data multimodal | OBELICS-2, LLaVA-Video-178K, CapRL | v0.1 |
| Quant | NVFP4 native pretrain (Blackwell) | v0.1 |
| Vision option | FastVLM / FastViTHD | v0.1 |
| Audio codec | Mimi → dual-codec slot | v0.1 |
| Video tokenizer | Cosmos-Tokenizer | v0.1 |
| RL | GSPO + DAPO + Dr.GRPO + Critique-GRPO | v0.2 |
| RL | Tree-GRPO, GTPO | v0.3 |
| Reasoning | Adaptive thinking head (Claude 4.7-style) | v0.2 |
| Reasoning | Verbalized PRM (ThinkPRM) | v0.2 |
| Memory | Engram conditional memory | v0.3 |
| Memory | Anthropic memory tool pattern | v0.3 |
| Agents | Code Mode (replaces DSML XML) | v0.3 |
| Agents | Anthropic Skills, A2A protocol | v0.3 |
| Inference | EAGLE-3 + P-EAGLE speculative | v0.3 |
| Distributed | Templar SparseLoCo, NoLoCo, Pluralis | v0.4 |
| Architecture | Hybrid linear+full attn (Kimi Linear) | review |

## Known revisions

- DSML XML (ETC-02 in baseline) **superseded** by Code Mode (AGT-01).
  Frontier converged on Code Mode pattern; DSML XML schema not used
  anywhere outside our own docs.
- REA-05 `<budget:adaptive>` token superseded by RL-07 learned effort
  head (Claude 4.7 deprecated explicit budget tokens).
- REA-09 PRM (deferred) revised to `proposed` in form of verbalized PRM
  (RL-10) — DeepSeek's R1 rejection was about naive step-annotation, not
  the concept.

## Testing

- Local Mac: `uv run pytest packages/ tests/ -m "not slow"` — 506 tests,
  ~15 deselected/skipped (CUDA-only)
- Remote hpomen: `./scripts/sync-and-test-gpu.sh` — 506/506 passing
- CI: not set up yet (item for v0.0)
