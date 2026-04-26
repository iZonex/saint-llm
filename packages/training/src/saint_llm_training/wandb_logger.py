"""Wandb logger adapter for the Trainer's ``metrics_callback`` hook.

Implements the ``MetricsCallback`` contract
(``Callable[[int, dict[str, float]], None]``) so it can be plugged
straight into ``Trainer(metrics_callback=...)``. Wandb is imported
lazily so importing this module never forces a wandb-init at import
time; the actual ``wandb.init`` call happens in the constructor.

Metric naming convention:
    Metrics whose key contains ``/`` are passed through unchanged
    (caller controls the namespace, e.g. ``eval/ppl``,
    ``system/gpu_util``). Metrics without ``/`` are auto-prefixed with
    ``train/`` so the standard Trainer keys (``loss``, ``lr``,
    ``grad_norm``, ``skipped``) land under one panel in the dashboard.

Modes:
    - ``"online"`` (default) — live sync to wandb cloud
    - ``"offline"`` — local-only; sync later with ``wandb sync``
    - ``"disabled"`` — wandb calls become no-ops; useful for tests /
      headless dev runs without losing the call sites
"""

from __future__ import annotations

from typing import Any, Literal


class WandbLogger:
    """Saint-llm Trainer ``metrics_callback`` adapter for Weights & Biases.

    Args:
        project: wandb project name (creates if not exists).
        run_name: human-readable run identifier; ``None`` lets wandb
            auto-generate.
        config: dict serialized to ``wandb.config`` for run metadata
            (model hyperparameters, dataset, etc.).
        mode: ``"online"`` / ``"offline"`` / ``"disabled"``.
        tags: short tag strings filtered/grouped in the dashboard.
        wandb_module: optional override for the wandb module — primarily
            for tests, where a ``unittest.mock.MagicMock`` is injected.
            Production callers leave this ``None`` and the real
            ``wandb`` package is imported lazily.
    """

    def __init__(
        self,
        *,
        project: str,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        mode: Literal["online", "offline", "disabled"] = "online",
        tags: tuple[str, ...] = (),
        wandb_module: Any | None = None,
    ) -> None:
        if wandb_module is None:
            import wandb as _wandb_module  # noqa: PLC0415 — lazy import by design

            self._wandb = _wandb_module
        else:
            self._wandb = wandb_module

        self._run = self._wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            mode=mode,
            tags=list(tags),
            reinit=True,
        )
        self._finished = False

    def __call__(self, step: int, metrics: dict[str, float]) -> None:
        if self._finished:
            return
        prefixed = {
            (k if "/" in k else f"train/{k}"): v for k, v in metrics.items()
        }
        self._wandb.log(prefixed, step=step)

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Alias for ``__call__`` so callers can use ``logger.log(...)``
        when calling explicitly outside the Trainer's hook.
        """
        self(step, metrics)

    def finish(self) -> None:
        """Close the wandb run. Safe to call multiple times."""
        if self._finished:
            return
        self._wandb.finish()
        self._finished = True

    def __enter__(self) -> WandbLogger:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.finish()
