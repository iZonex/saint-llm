"""CheckpointRotator: rotating save with N-cap eviction."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM
from saint_llm_training import CheckpointRotator, Trainer
from torch import nn


def _ce(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    out = model(batch)
    return F.cross_entropy(
        out["logits"][:, :-1].reshape(-1, out["logits"].shape[-1]),
        batch[:, 1:].reshape(-1),
    )


def _trainer() -> Trainer:
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    return Trainer(model, opt, loss_fn=_ce)


def test_rotator_writes_step_suffix(tmp_path: Path) -> None:
    rot = CheckpointRotator(tmp_path / "ckpt.pt", keep_last_n=5)
    trainer = _trainer()
    trainer.step = 100
    path = rot.save(trainer)
    assert path.name == "ckpt.step100.pt"
    assert path.exists()


def test_rotator_evicts_beyond_cap(tmp_path: Path) -> None:
    rot = CheckpointRotator(tmp_path / "ckpt.pt", keep_last_n=2)
    trainer = _trainer()
    saved_paths = []
    for step in (10, 20, 30, 40):
        trainer.step = step
        saved_paths.append(rot.save(trainer))

    # Only the last 2 should survive on disk.
    assert not saved_paths[0].exists()
    assert not saved_paths[1].exists()
    assert saved_paths[2].exists()
    assert saved_paths[3].exists()
    assert rot.saved_paths == (saved_paths[2], saved_paths[3])


def test_rotator_keeps_all_when_cap_none(tmp_path: Path) -> None:
    rot = CheckpointRotator(tmp_path / "ckpt.pt", keep_last_n=None)
    trainer = _trainer()
    paths = []
    for step in (10, 20, 30, 40):
        trainer.step = step
        paths.append(rot.save(trainer))
    for p in paths:
        assert p.exists()


def test_rotator_uses_trainer_step_when_step_omitted(tmp_path: Path) -> None:
    rot = CheckpointRotator(tmp_path / "ckpt.pt", keep_last_n=3)
    trainer = _trainer()
    trainer.step = 7
    path = rot.save(trainer)
    assert "step7" in path.name


def test_rotator_explicit_step_overrides_trainer_step(tmp_path: Path) -> None:
    rot = CheckpointRotator(tmp_path / "ckpt.pt", keep_last_n=3)
    trainer = _trainer()
    trainer.step = 7
    path = rot.save(trainer, step=42)
    assert "step42" in path.name


def test_rotator_invalid_keep_last_n() -> None:
    with pytest.raises(ValueError, match="keep_last_n"):
        CheckpointRotator("/tmp/ckpt.pt", keep_last_n=0)
    with pytest.raises(ValueError, match="keep_last_n"):
        CheckpointRotator("/tmp/ckpt.pt", keep_last_n=-1)


def test_rotator_handles_already_deleted_file(tmp_path: Path) -> None:
    """If the user manually deletes one of the tracked files, eviction shouldn't crash."""
    rot = CheckpointRotator(tmp_path / "ckpt.pt", keep_last_n=2)
    trainer = _trainer()
    trainer.step = 10
    p1 = rot.save(trainer)
    p1.unlink()  # manual cleanup
    trainer.step = 20
    rot.save(trainer)
    trainer.step = 30
    rot.save(trainer)  # this triggers eviction of p1, which is already gone
    # Should not have raised.


def test_rotator_extra_payload_passed_through(tmp_path: Path) -> None:
    rot = CheckpointRotator(tmp_path / "ckpt.pt", keep_last_n=3)
    trainer = _trainer()
    trainer.step = 5
    path = rot.save(trainer, extra={"data_cursor": 12345})
    payload = torch.load(path, weights_only=False)
    assert payload["extra"] == {"data_cursor": 12345}
