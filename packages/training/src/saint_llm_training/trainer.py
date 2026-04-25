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
        gradient_accumulation_steps: int = 1,
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
        if gradient_accumulation_steps <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be > 0; got {gradient_accumulation_steps}",
            )
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.grad_clip_norm = grad_clip_norm
        self.metrics_callback = metrics_callback
        self.skip_nonfinite_loss = skip_nonfinite_loss
        self.loss_spike_factor = loss_spike_factor
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.step = 0
        self.skipped_steps = 0
        self._recent_losses: deque[float] = deque(maxlen=loss_spike_window)
        self._micro_step = 0  # 0..gradient_accumulation_steps-1; resets after optimizer.step

    def train_step(self, batch: Tensor) -> float:
        """One micro-step. Returns the (un-divided) scalar loss value.

        With ``gradient_accumulation_steps > 1`` the loss is divided by K
        before backward so the accumulated gradient equals the average over K
        microbatches; the optimizer step (and scheduler step, callback,
        ``self.step`` increment) only fires on the K-th call.

        Skip semantics:
        * non-finite loss (``skip_nonfinite_loss=True``, default) — backward
          and optimizer step skipped, gradients zeroed, ``skipped_steps``
          incremented. Counts as an optimizer step boundary: micro-step
          counter resets so the next call starts a fresh K-window.
        * loss > ``median(recent) * loss_spike_factor`` — same path, reason
          ``"spike"``.

        ``metrics_callback`` (if set) fires on optimizer-step boundaries only
        (successful or skipped). Returned loss value is the *un-scaled*
        per-microbatch loss; useful for live progress logging.
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
            self.optimizer.zero_grad(set_to_none=True)
            self.skipped_steps += 1
            self._micro_step = 0
            self._fire_callback(loss_value, grad_norm=None, skip_reason=skip_reason)
            return loss_value

        # Scale and accumulate. zero_grad happens after the macro step.
        if self.gradient_accumulation_steps > 1:
            (loss / self.gradient_accumulation_steps).backward()
        else:
            self.optimizer.zero_grad()
            loss.backward()

        self._micro_step += 1
        if self._micro_step < self.gradient_accumulation_steps:
            # Mid-accumulation: no optimizer step, no scheduler tick, no
            # counter increment. Caller still gets the loss back for live
            # logging if they want to track per-microbatch progress
            # (callback is intentionally not fired here — keeps "step" in
            # metrics aligned with ``self.step``).
            return loss_value

        # K-th micro-step → optimizer + scheduler boundary.
        grad_norm: float | None = None
        if self.grad_clip_norm is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm,
            )
            grad_norm = float(total_norm.detach().item()) if isinstance(total_norm, Tensor) else float(total_norm)
        self.optimizer.step()
        if self.gradient_accumulation_steps > 1:
            self.optimizer.zero_grad()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.step += 1
        self._micro_step = 0
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
