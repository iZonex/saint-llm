# saint-llm

DeepSeek-V4-style training and inference framework. Greenfield monorepo.

Reference: `DeepSeek_V4.pdf` in repo root — architecture, training framework, post-training pipeline, infrastructure.

## Layout

```
packages/
  core/           Model architecture (CSA, HCA, mHC, MoE, MTP)
  kernels/        TileLang kernels, MegaMoE, batch-invariant deterministic ops
  optim/          Muon (hybrid Newton-Schulz) + AdamW wiring
  distributed/    Fine-grained EP, hybrid ZeRO, two-stage CP, DualPipe
  training/       Training loop, FP4 QAT, anticipatory routing, activation checkpointing
  posttraining/   Specialist training, GRPO, full-vocabulary OPD
  sandbox/        DSec sandbox (Rust services + Python SDK)
  inference/      Heterogeneous KV cache, on-disk SWA caching
  data/           Pretraining + post-training data pipelines, tokenizer
  eval/           Benchmarks (knowledge, reasoning, code, agent, long-context)

configs/          Hydra configs (model/training/posttraining/inference/data/eval)
experiments/      Experiment definitions
infra/            Docker, k8s, deployment scripts
scripts/          Utility scripts
tests/            Cross-package integration tests
```

## Setup

```bash
uv sync --all-packages
uv run pytest
```

## Status

Bootstrap. See `TaskList` for active work items.
