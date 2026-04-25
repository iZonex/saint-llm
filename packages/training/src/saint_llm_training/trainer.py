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

import math
from collections import deque
from collections.abc import Callable, Iterable
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
MetricsCallback = Callable[[int, dict[str, float]], None]


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
        metrics_callback: MetricsCallback | None = None,
        skip_nonfinite_loss: bool = True,
        loss_spike_factor: float | None = None,
        loss_spike_window: int = 32,
        device: torch.device | str | None = None,
    ) -> None:
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be > 0 when set; got {grad_clip_norm}")
        if loss_spike_factor is not None and loss_spike_factor <= 1.0:
            raise ValueError(
                f"loss_spike_factor must be > 1.0 when set; got {loss_spike_factor}",
            )
        if loss_spike_window <= 0:
            raise ValueError(f"loss_spike_window must be > 0; got {loss_spike_window}")
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.grad_clip_norm = grad_clip_norm
        self.metrics_callback = metrics_callback
        self.skip_nonfinite_loss = skip_nonfinite_loss
        self.loss_spike_factor = loss_spike_factor
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.step = 0
        self.skipped_steps = 0
        self._recent_losses: deque[float] = deque(maxlen=loss_spike_window)

    def train_step(self, batch: Tensor) -> float:
        """One forward + backward + optimizer step. Returns the scalar loss value.

        Skip semantics:
        * non-finite loss (``skip_nonfinite_loss=True``, default) skips the
          backward + optimizer step, zeros gradients, increments
          ``skipped_steps``, and does not advance ``step`` or the LR scheduler.
        * loss > ``median(recent) * loss_spike_factor`` (when configured) is
          treated the same way, with reason ``"spike"``.

        ``metrics_callback`` (if set) fires for every call — successful or
        skipped — with at minimum ``loss`` and ``lr``. Skipped calls also
        carry ``skipped: 1.0`` and ``skip_reason`` is encoded as ``nan`` (1.0)
        or ``spike`` (1.0) so the caller can route to wandb/tags directly.
        """
        self.model.train()
        batch = batch.to(self.device)
        loss = self.loss_fn(self.model, batch)
        loss_value = float(loss.detach().item())

        skip_reason: str | None = None
        if self.skip_nonfinite_loss and not math.isfinite(loss_value):
            skip_reason = "nan"
        elif self.loss_spike_factor is not None and self._is_loss_spike(loss_value):
            skip_reason = "spike"

        if skip_reason is not None:
            # Throw away any partial graph + grads accumulated so far.
            self.optimizer.zero_grad(set_to_none=True)
            self.skipped_steps += 1
            self._fire_callback(loss_value, grad_norm=None, skip_reason=skip_reason)
            return loss_value

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm: float | None = None
        if self.grad_clip_norm is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm,
            )
            grad_norm = float(total_norm.detach().item()) if isinstance(total_norm, Tensor) else float(total_norm)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.step += 1
        self._recent_losses.append(loss_value)
        self._fire_callback(loss_value, grad_norm=grad_norm, skip_reason=None)
        return loss_value

    def _is_loss_spike(self, loss_value: float) -> bool:
        """True if ``loss_value`` exceeds ``median(recent) * loss_spike_factor``.

        Returns False before enough history has accumulated (need at least 4
        prior losses to estimate a stable median).
        """
        if self.loss_spike_factor is None or len(self._recent_losses) < 4:
            return False
        sorted_losses = sorted(self._recent_losses)
        n = len(sorted_losses)
        median = sorted_losses[n // 2] if n % 2 == 1 else 0.5 * (sorted_losses[n // 2 - 1] + sorted_losses[n // 2])
        return loss_value > median * self.loss_spike_factor

    def _fire_callback(
        self,
        loss_value: float,
        *,
        grad_norm: float | None,
        skip_reason: str | None,
    ) -> None:
        if self.metrics_callback is None:
            return
        metrics: dict[str, float] = {
            "loss": loss_value,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }
        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm
        if skip_reason is not None:
            metrics["skipped"] = 1.0
            metrics[f"skip_reason_{skip_reason}"] = 1.0
        self.metrics_callback(self.step, metrics)

    @torch.no_grad()
    def evaluate(self, eval_iter: Iterable[Tensor]) -> float:
        """Run ``loss_fn`` under ``torch.no_grad`` over an iterable of batches.

        Returns the unweighted mean of per-batch loss values. Toggles the model
        into eval mode for the duration; the caller's prior train/eval state is
        restored on exit.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            total = 0.0
            count = 0
            for raw_batch in eval_iter:
                batch = raw_batch.to(self.device)
                loss = self.loss_fn(self.model, batch)
                total += float(loss.detach().item())
                count += 1
            if count == 0:
                raise ValueError("evaluate() received an empty iterator")
            return total / count
        finally:
            if was_training:
                self.model.train()

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
