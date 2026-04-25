"""Training checkpoint save/load.

Stores model state, optimizer state, current step, and an arbitrary
``extra`` payload (e.g., LR-scheduler state, RNG seeds, dataset cursor) in a
single ``torch.save``-format file. ``load_checkpoint`` mutates the model
(and optionally the optimizer) in-place and returns the stored metadata.

This is intentionally minimal — sharded / distributed checkpointing,
safetensors support, and resume-aware data cursors live on top.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a checkpoint to ``path``.

    Layout:
        {"step": int, "model": state_dict, "optimizer": state_dict | None,
         "extra": dict | None, "format_version": int}
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "extra": dict(extra) if extra is not None else None,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint from ``path`` into ``model`` (and optionally optimizer).

    Args:
        strict: forwarded to ``model.load_state_dict``. False allows missing
            or extra keys for partial-architecture loading.
        map_location: forwarded to ``torch.load``.

    Returns:
        ``{"step": int, "extra": dict | None, "format_version": int}`` —
        metadata for the caller to use (e.g., resume the loop at this step).
    """
    payload: dict[str, Any] = torch.load(
        path, map_location=map_location, weights_only=False,
    )
    if "model" not in payload:
        raise ValueError(f"checkpoint at {path} missing 'model' key")

    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None:
        opt_state = payload.get("optimizer")
        if opt_state is None:
            raise ValueError(
                f"checkpoint at {path} has no optimizer state but caller passed one",
            )
        optimizer.load_state_dict(opt_state)

    return {
        "step": int(payload.get("step", 0)),
        "extra": payload.get("extra"),
        "format_version": int(payload.get("format_version", 1)),
    }
