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

See [`docs/STATUS.md`](docs/STATUS.md) for honest current state. Pre-v0.0,
scaffolding phase, no real training run yet.

## Documentation map

Read in this order to understand the project:

1. [`docs/STATUS.md`](docs/STATUS.md) — honest current state, what works,
   what doesn't, what's planned.
2. [`docs/ROADMAP.md`](docs/ROADMAP.md) — versioning plan v0.0 → v1.0,
   per-version scope and exit criteria, where each Q1 2026 frontier
   technique lands.
3. [`docs/AUGMENTATIONS.md`](docs/AUGMENTATIONS.md) — every deviation
   from a pure DeepSeek-V4 reproduction, with status (`proposed` /
   `accepted-vX.Y` / `deferred-vY.Z` / `rejected` / `revised`) and source
   citation. Includes Q1 2026 frontier picks across architecture,
   optimizer, RL, data, inference, agents, distributed.
4. [`docs/adr/`](docs/adr/) — Architecture Decision Records, one per
   significant decision. Index in [`docs/adr/README.md`](docs/adr/README.md).
5. [`docs/specs/`](docs/specs/) — subsystem specifications (per-version,
   populated as work begins).
6. [`docs/reports/`](docs/reports/) — version exit reports
   (loss curves, eval results) — populated as versions complete.
7. [`docs/DeepSeek_V4.pdf`](docs/DeepSeek_V4.pdf) — architectural reference.
