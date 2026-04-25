#!/usr/bin/env bash
# Sync local repo to hpomen GPU box and run tests there.
# See ~/.claude/projects/.../memory/reference_hpomen_gpu.md for box details.

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-dlchistyakov@192.168.1.64}"
REMOTE_DIR="${REMOTE_DIR:-~/saint-llm}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

PYTEST_ARGS="${*:-packages/ tests/ -m 'not slow' --tb=short}"

echo "→ rsync $LOCAL_DIR/ to $REMOTE_HOST:$REMOTE_DIR/"
rsync -az --delete \
  --exclude='.venv' \
  --exclude='.pytest_cache' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  --exclude='docs/*.pdf' \
  --exclude='uv.lock' \
  "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

echo "→ uv sync + pytest on $REMOTE_HOST"
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && source \$HOME/.local/bin/env && uv sync --all-packages -q && uv run pytest $PYTEST_ARGS"
