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


def test_grad_clip_norm_caps_gradient_norm() -> None:
    """With a tiny clip threshold, the global grad norm post-clip must be bounded."""
    cfg = ModelConfig.tiny()
    torch.manual_seed(0)
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    clip = 0.01
    trainer = Trainer(model, opt, loss_fn=_ce_loss_fn, grad_clip_norm=clip)

    # Bypass train_step so we can inspect grads after clip but before step zeros them.
    batch = torch.randint(0, cfg.vocab_size, (1, 16))
    model.train()
    loss = _ce_loss_fn(model, batch)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    total_sq = sum(p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None)
    norm_after = total_sq.sqrt().item()
    assert norm_after <= clip + 1.0e-5
    # Clipped trainer's next train_step should produce a finite loss.
    loss2 = trainer.train_step(batch)
    assert torch.isfinite(torch.tensor(loss2))


def test_grad_clip_norm_rejects_non_positive() -> None:
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    with pytest.raises(ValueError, match="must be > 0"):
        Trainer(model, opt, loss_fn=_ce_loss_fn, grad_clip_norm=0.0)
    with pytest.raises(ValueError, match="must be > 0"):
        Trainer(model, opt, loss_fn=_ce_loss_fn, grad_clip_norm=-1.0)


def test_no_grad_clip_by_default() -> None:
    """Default Trainer leaves gradients unclipped."""
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    trainer = Trainer(model, opt, loss_fn=_ce_loss_fn)
    assert trainer.grad_clip_norm is None


def test_evaluate_returns_mean_loss(trainer: Trainer) -> None:
    cfg = ModelConfig.tiny()
    batches = [torch.randint(0, cfg.vocab_size, (1, 16)) for _ in range(3)]
    mean = trainer.evaluate(iter(batches))
    assert isinstance(mean, float)
    assert torch.isfinite(torch.tensor(mean))


def test_evaluate_does_not_update_weights() -> None:
    cfg = ModelConfig.tiny()
    torch.manual_seed(0)
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    trainer = Trainer(model, opt, loss_fn=_ce_loss_fn)

    snap = {k: v.detach().clone() for k, v in model.state_dict().items()}
    trainer.evaluate(iter([torch.randint(0, cfg.vocab_size, (1, 16))]))
    for k, v in model.state_dict().items():
        assert torch.equal(v, snap[k]), f"evaluate() mutated {k}"


def test_evaluate_restores_train_mode_when_called_from_training() -> None:
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    trainer = Trainer(model, opt, loss_fn=_ce_loss_fn)
    model.train()
    trainer.evaluate(iter([torch.randint(0, cfg.vocab_size, (1, 16))]))
    assert model.training is True


def test_evaluate_leaves_eval_mode_when_called_from_eval() -> None:
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    trainer = Trainer(model, opt, loss_fn=_ce_loss_fn)
    model.eval()
    trainer.evaluate(iter([torch.randint(0, cfg.vocab_size, (1, 16))]))
    assert model.training is False


def test_evaluate_empty_iterator_raises(trainer: Trainer) -> None:
    with pytest.raises(ValueError, match="empty iterator"):
        trainer.evaluate(iter([]))


def test_metrics_callback_fires_per_step() -> None:
    """metrics_callback receives (step, dict) on every train_step."""
    cfg = ModelConfig.tiny()
    torch.manual_seed(0)
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    seen: list[tuple[int, dict[str, float]]] = []
    trainer = Trainer(
        model, opt, loss_fn=_ce_loss_fn,
        metrics_callback=lambda step, m: seen.append((step, dict(m))),
    )
    batch = torch.zeros(1, 16, dtype=torch.long)
    trainer.train_step(batch)
    trainer.train_step(batch)
    assert len(seen) == 2
    # Steps recorded post-increment.
    assert seen[0][0] == 1
    assert seen[1][0] == 2
    assert "loss" in seen[0][1]
    assert "lr" in seen[0][1]
    assert "grad_norm" not in seen[0][1]  # no clip → no grad_norm


def test_metrics_callback_includes_grad_norm_when_clipping() -> None:
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    seen: list[dict[str, float]] = []
    trainer = Trainer(
        model, opt, loss_fn=_ce_loss_fn,
        grad_clip_norm=1.0,
        metrics_callback=lambda _step, m: seen.append(dict(m)),
    )
    trainer.train_step(torch.zeros(1, 16, dtype=torch.long))
    assert "grad_norm" in seen[0]
    assert torch.isfinite(torch.tensor(seen[0]["grad_norm"]))


def test_metrics_callback_lr_tracks_scheduler() -> None:
    """When an LR scheduler is set, the reported lr matches the scheduled value."""
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)
    seen: list[float] = []
    trainer = Trainer(
        model, opt, loss_fn=_ce_loss_fn,
        lr_scheduler=sched,
        metrics_callback=lambda _step, m: seen.append(m["lr"]),
    )
    trainer.train_step(torch.zeros(1, 16, dtype=torch.long))
    trainer.train_step(torch.zeros(1, 16, dtype=torch.long))
    # After step 1, lr decayed to 1e-4; step 2, decayed to 1e-5.
    assert seen[0] == pytest.approx(1.0e-4, rel=1.0e-3)
    assert seen[1] == pytest.approx(1.0e-5, rel=1.0e-3)


def test_no_metrics_callback_default(trainer: Trainer) -> None:
    """Default Trainer has no callback — train_step doesn't blow up trying to fire one."""
    assert trainer.metrics_callback is None
    trainer.train_step(torch.zeros(1, 16, dtype=torch.long))
