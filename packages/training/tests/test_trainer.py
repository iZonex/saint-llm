"""Trainer: step counter, loss decrease, save/load with LR scheduler, device move."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM
from saint_llm_training import Trainer
from torch import nn


def _ce_loss_fn(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Causal-LM cross-entropy. ``batch`` shape (B, T)."""
    out = model(batch)
    logits = out["logits"]
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        batch[:, 1:].reshape(-1),
    )


@pytest.fixture
def trainer() -> Trainer:
    torch.manual_seed(0)
    model = SaintLLM(ModelConfig.tiny())
    optim = torch.optim.AdamW(model.parameters(), lr=3.0e-3)
    return Trainer(model, optim, loss_fn=_ce_loss_fn)


def test_step_counter_increments(trainer: Trainer) -> None:
    batch = torch.zeros(1, 16, dtype=torch.long)
    assert trainer.step == 0
    trainer.train_step(batch)
    assert trainer.step == 1
    trainer.train_step(batch)
    assert trainer.step == 2


def test_train_step_decreases_loss(trainer: Trainer) -> None:
    """Memorize a fixed batch; loss must drop monotonically over a few steps."""
    cfg = ModelConfig.tiny()
    batch = torch.randint(0, cfg.vocab_size, (1, 16))
    losses = [trainer.train_step(batch) for _ in range(4)]
    assert all(loss > 0 for loss in losses)
    assert losses[-1] < losses[0]


def test_train_step_returns_finite_float(trainer: Trainer) -> None:
    batch = torch.zeros(1, 16, dtype=torch.long)
    loss = trainer.train_step(batch)
    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))


def test_save_load_round_trip(tmp_path: Path) -> None:
    cfg = ModelConfig.tiny()
    torch.manual_seed(0)
    model_a = SaintLLM(cfg)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1.0e-3)
    trainer_a = Trainer(model_a, opt_a, loss_fn=_ce_loss_fn)

    batch = torch.randint(0, cfg.vocab_size, (1, 16))
    trainer_a.train_step(batch)
    trainer_a.train_step(batch)

    ckpt = tmp_path / "trainer.pt"
    trainer_a.save(ckpt)

    torch.manual_seed(99)  # ensure fresh init differs
    model_b = SaintLLM(cfg)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1.0e-3)
    trainer_b = Trainer(model_b, opt_b, loss_fn=_ce_loss_fn)

    trainer_b.load(ckpt)
    assert trainer_b.step == trainer_a.step

    # Subsequent step in B should see the same loss A would (deterministic resume).
    batch2 = torch.randint(0, cfg.vocab_size, (1, 16))
    loss_a = trainer_a.train_step(batch2)
    loss_b = trainer_b.train_step(batch2)
    assert abs(loss_a - loss_b) < 1.0e-3


def test_save_load_with_lr_scheduler(tmp_path: Path) -> None:
    cfg = ModelConfig.tiny()
    torch.manual_seed(0)
    model_a = SaintLLM(cfg)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1.0e-3)
    sched_a = torch.optim.lr_scheduler.StepLR(opt_a, step_size=1, gamma=0.5)
    trainer_a = Trainer(model_a, opt_a, loss_fn=_ce_loss_fn, lr_scheduler=sched_a)

    batch = torch.randint(0, cfg.vocab_size, (1, 16))
    trainer_a.train_step(batch)
    trainer_a.train_step(batch)
    lr_a_after = opt_a.param_groups[0]["lr"]

    ckpt = tmp_path / "trainer.pt"
    trainer_a.save(ckpt)

    torch.manual_seed(99)
    model_b = SaintLLM(cfg)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1.0e-3)  # different starting lr
    sched_b = torch.optim.lr_scheduler.StepLR(opt_b, step_size=1, gamma=0.5)
    trainer_b = Trainer(model_b, opt_b, loss_fn=_ce_loss_fn, lr_scheduler=sched_b)
    trainer_b.load(ckpt)

    assert opt_b.param_groups[0]["lr"] == lr_a_after


def test_load_returns_remaining_extra(tmp_path: Path) -> None:
    """save(extra=...) round-trips through load() minus the lr_scheduler subkey."""
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    trainer = Trainer(model, opt, loss_fn=_ce_loss_fn)
    ckpt = tmp_path / "with_extra.pt"
    trainer.save(ckpt, extra={"data_cursor": 42, "rng_seed": 7})

    fresh = Trainer(SaintLLM(cfg), torch.optim.AdamW(model.parameters(), lr=1.0e-3),
                    loss_fn=_ce_loss_fn)
    extra = fresh.load(ckpt)
    assert extra == {"data_cursor": 42, "rng_seed": 7}


def test_device_inferred_from_model_when_unspecified() -> None:
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    trainer = Trainer(model, opt, loss_fn=_ce_loss_fn)
    assert trainer.device == next(model.parameters()).device


def test_train_step_moves_batch_to_device() -> None:
    """A CPU batch passed to a CPU trainer should run without device errors."""
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    trainer = Trainer(model, opt, loss_fn=_ce_loss_fn, device="cpu")
    batch_cpu = torch.zeros(1, 16, dtype=torch.long)
    loss = trainer.train_step(batch_cpu)
    assert torch.isfinite(torch.tensor(loss))
