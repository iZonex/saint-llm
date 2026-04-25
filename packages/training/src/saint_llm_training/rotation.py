"""Periodic checkpoint rotation for long training runs.

The training loop calls ``CheckpointRotator.save(trainer, step)`` every K
steps; the rotator persists ``{base_path.stem}.step{step}{base_path.suffix}``
and evicts the oldest tracked file once more than ``keep_last_n`` exist.

This is intentionally separate from ``Trainer.save`` — the trainer doesn't
own the schedule (caller owns the cadence and the policy of "what to keep").
"""

from __future__ import annotations

import contextlib
from collections import deque
from pathlib import Path
from typing import Any

from saint_llm_training.trainer import Trainer


class CheckpointRotator:
    """Rotating checkpoint writer.

    Args:
        base_path: target path; ``step{N}`` is inserted before the suffix so
            ``run/ckpt.pt`` becomes ``run/ckpt.step100.pt``, ``run/ckpt.step200.pt``,
            etc.
        keep_last_n: cap on simultaneously-resident checkpoint files. Older
            files are deleted when this is exceeded. Use ``None`` to keep all.
    """

    def __init__(self, base_path: str | Path, *, keep_last_n: int | None = 3) -> None:
        if keep_last_n is not None and keep_last_n <= 0:
            raise ValueError(f"keep_last_n must be > 0 when set; got {keep_last_n}")
        self.base_path = Path(base_path)
        self.keep_last_n = keep_last_n
        self._saved: deque[Path] = deque()

    def _path_for_step(self, step: int) -> Path:
        return self.base_path.with_name(
            f"{self.base_path.stem}.step{step}{self.base_path.suffix}",
        )

    def save(
        self,
        trainer: Trainer,
        *,
        step: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Persist ``trainer`` state at ``step`` (default: ``trainer.step``).

        Returns the path written. Evicts the oldest tracked path when the
        ``keep_last_n`` cap is exceeded.
        """
        actual_step = trainer.step if step is None else step
        path = self._path_for_step(actual_step)
        trainer.save(path, extra=extra)
        self._saved.append(path)
        self._evict_old()
        return path

    def _evict_old(self) -> None:
        if self.keep_last_n is None:
            return
        while len(self._saved) > self.keep_last_n:
            old = self._saved.popleft()
            with contextlib.suppress(FileNotFoundError):
                old.unlink()

    @property
    def saved_paths(self) -> tuple[Path, ...]:
        """Currently-resident checkpoint files, oldest first."""
        return tuple(self._saved)
