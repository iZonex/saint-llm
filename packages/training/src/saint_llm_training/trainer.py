"""Minimal Trainer — orchestration shell around model + optimizer + step counter.

Deliberately small. Owns the four things a training loop actually needs:
    * model, optimizer, optional LR scheduler
    * step counter
    * loss callback (so loss math stays out of the trainer)
    * save / load that survives the LR scheduler

Anything that grows beyond this — gradient accumulation, mixed precision,
distributed sync, eval cadence policies, early stopping, wandb hooks — lives
in caller code or a more featureful subclass. We refuse to grow this into a
HuggingFace-Trainer-style godclass.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from saint_llm_training.checkpoint import load_checkpoint, save_checkpoint


class _LRSchedulerLike(Protocol):
    """Subset of torch.optim.lr_scheduler API the Trainer touches."""

    def step(self) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


LossFn = Callable[[nn.Module, Tensor], Tensor]


class Trainer:
    """Wraps model + optimizer + optional LR scheduler with a step counter.

    Args:
        model: the model to train.
        optimizer: any ``torch.optim.Optimizer``.
        loss_fn: ``(model, batch) -> scalar Tensor``. Caller writes the
            forward + cross-entropy or whatever they want; trainer just
            backprops and optimizer-steps.
        lr_scheduler: optional. ``.step()`` called after each optimizer step.
        device: where the trainer lives. ``train_step`` will move incoming
            batches with ``.to(device)`` so callers can keep data on CPU.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        loss_fn: LossFn,
        lr_scheduler: _LRSchedulerLike | None = None,
        grad_clip_norm: float | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be > 0 when set; got {grad_clip_norm}")
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.grad_clip_norm = grad_clip_norm
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.step = 0

    def train_step(self, batch: Tensor) -> float:
        """One forward + backward + optimizer step. Returns the scalar loss value."""
        self.model.train()
        batch = batch.to(self.device)
        loss = self.loss_fn(self.model, batch)
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.step += 1
        return float(loss.detach().item())

    def save(self, path: str | Path, *, extra: dict[str, Any] | None = None) -> None:
        """Persist model + optimizer + step + LR-scheduler state to ``path``."""
        merged: dict[str, Any] = dict(extra) if extra is not None else {}
        if self.lr_scheduler is not None:
            merged["lr_scheduler"] = self.lr_scheduler.state_dict()
        save_checkpoint(path, self.model, self.optimizer, step=self.step, extra=merged)

    def load(self, path: str | Path, *, strict: bool = True) -> dict[str, Any]:
        """Restore model + optimizer + step + LR scheduler from ``path``.

        Returns the ``extra`` payload (minus the LR-scheduler subkey, which
        the trainer consumes itself) so callers can recover their own metadata.
        """
        meta = load_checkpoint(path, self.model, self.optimizer, strict=strict)
        self.step = int(meta.get("step", 0))
        extra = dict(meta.get("extra") or {})
        if self.lr_scheduler is not None and "lr_scheduler" in extra:
            self.lr_scheduler.load_state_dict(extra.pop("lr_scheduler"))
        return extra
