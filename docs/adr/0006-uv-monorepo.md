# ADR-0006: uv workspace monorepo with N packages

- **Status**: accepted
- **Date**: 2026-04-25 (backfilled from 2026-04-15 project bootstrap)
- **Deciders**: Dmytro
- **Related**: `pyproject.toml` (root + per-package)

## Context

A multi-subsystem project (architecture, kernels, training, inference,
post-training, agents, sandbox, distributed, data, eval, optim) needs
either:
1. One package with sub-modules, single `pyproject.toml`.
2. Many packages with separate `pyproject.toml` each, coordinated by a
   workspace tool.
3. Multiple repos.

## Decision

Use **uv workspace monorepo** with N packages under `packages/`,
each its own `pyproject.toml`, coordinated by the root
`[tool.uv.workspace] members = ["packages/*"]` declaration.

Currently 11 packages: core, kernels, optim, distributed, training,
posttraining, inference, data, eval, sandbox, agents.

## Consequences

**Intended:**
- Each subsystem has its own dependency surface (sandbox doesn't pull
  torch, agents doesn't pull tokenizers).
- Inter-package deps explicit via `[project.dependencies]` +
  `[tool.uv.sources] X = { workspace = true }`.
- Single `uv sync` resolves everything; tests run from root.
- Sub-team / sub-task ownership is clean (one package per
  contributor's focus area).

**Unintended / accepted:**
- Adding a new package requires touching root `pyproject.toml`
  (deps + sources) AND `tests/test_smoke.py` (import check). Documented
  in `feedback_no_scaffolds_on_scaffolds.md`-class memory: a
  `uv sync` after pyproject changes will uninstall packages not
  declared in the root deps.
- Single-version coupling: all packages currently at `0.0.1`. When we
  ship versions, we'll need an independent versioning policy.

## Alternatives considered

- **Single package with sub-modules.** Rejected: forces unified
  dependency surface (every install pulls torch + transformers + etc.,
  even for `agents` which has no ML deps).
- **Multiple repos.** Rejected: too much overhead for a single-author
  project at this stage; cross-cutting refactors become painful.
- **Poetry / pdm / pip-tools.** Rejected: uv is the fastest, has
  first-class workspace support, and produces deterministic lockfiles
  via `uv.lock`. The 2025-2026 Python tooling consensus has converged
  on uv.

## Implementation notes

- New package checklist:
  1. `packages/<name>/pyproject.toml` with `[project]` + workspace
     deps + `[tool.uv.sources]` for any inter-package deps
  2. Add `<name>` to root `pyproject.toml` `dependencies` AND
     `[tool.uv.sources]`
  3. Add `saint_llm_<name>` to `tests/test_smoke.py` PACKAGES list
  4. `uv sync` to install
- Test-discovery: `[tool.pytest.ini_options] testpaths =
  ["tests", "packages/*/tests"]` in root pyproject.
