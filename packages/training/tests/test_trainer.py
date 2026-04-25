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


def _nan_loss_fn(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Loss function that returns NaN regardless of input (to test skip path)."""
    out = model(batch)
    return out["logits"].mean() * float("nan")


def test_nonfinite_loss_skipped_by_default() -> None:
    """Default Trainer skips a non-finite loss step instead of corrupting weights."""
    cfg = ModelConfig.tiny()
    torch.manual_seed(0)
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    trainer = Trainer(model, opt, loss_fn=_nan_loss_fn)

    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    trainer.train_step(torch.zeros(1, 16, dtype=torch.long))
    assert trainer.skipped_steps == 1
    assert trainer.step == 0
    # Weights must be unchanged after the skip.
    for k, v in model.state_dict().items():
        assert torch.equal(v, snapshot[k]), f"weights changed under NaN loss: {k}"


def test_skip_nonfinite_loss_can_be_disabled() -> None:
    """skip_nonfinite_loss=False lets the NaN through (default off → backward fires)."""
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    trainer = Trainer(model, opt, loss_fn=_nan_loss_fn, skip_nonfinite_loss=False)
    trainer.train_step(torch.zeros(1, 16, dtype=torch.long))
    # step counter incremented (no skip)
    assert trainer.step == 1
    assert trainer.skipped_steps == 0


def test_loss_spike_skips_step() -> None:
    """A loss far above the recent median must be skipped when factor is set."""
    cfg = ModelConfig.tiny()
    torch.manual_seed(0)
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    spike_loss = [None]

    def variable_loss(m: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        out = m(batch)
        base = out["logits"].mean() * 0.0  # zero gradient contribution
        if spike_loss[0] is None:
            return base + 1.0  # baseline ~1.0
        return base + spike_loss[0]

    trainer = Trainer(
        model, opt, loss_fn=variable_loss,
        loss_spike_factor=10.0,
    )
    # Build history.
    for _ in range(6):
        trainer.train_step(torch.zeros(1, 16, dtype=torch.long))
    assert trainer.step == 6

    # Now inject a loss 100x above median — must be skipped.
    spike_loss[0] = 100.0
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    trainer.train_step(torch.zeros(1, 16, dtype=torch.long))
    assert trainer.skipped_steps == 1
    assert trainer.step == 6  # no advance
    # Weights unchanged because the spike triggered skip.
    for k, v in model.state_dict().items():
        assert torch.equal(v, snapshot[k]), f"weights changed under spike: {k}"


def test_loss_spike_factor_validated() -> None:
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    with pytest.raises(ValueError, match="loss_spike_factor"):
        Trainer(model, opt, loss_fn=_ce_loss_fn, loss_spike_factor=1.0)
    with pytest.raises(ValueError, match="loss_spike_factor"):
        Trainer(model, opt, loss_fn=_ce_loss_fn, loss_spike_factor=0.5)


def test_metrics_callback_marks_skipped() -> None:
    """Callback gets ``skipped`` + ``skip_reason_nan`` when NaN happens."""
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    seen: list[dict[str, float]] = []
    trainer = Trainer(
        model, opt, loss_fn=_nan_loss_fn,
        metrics_callback=lambda _step, m: seen.append(dict(m)),
    )
    trainer.train_step(torch.zeros(1, 16, dtype=torch.long))
    assert "skipped" in seen[0]
    assert "skip_reason_nan" in seen[0]


def test_grad_accum_holds_optimizer_until_kth_microstep() -> None:
    """Weights unchanged on micro-steps 1..K-1; updated on the K-th call."""
    cfg = ModelConfig.tiny()
    torch.manual_seed(0)
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    trainer = Trainer(
        model, opt, loss_fn=_ce_loss_fn,
        gradient_accumulation_steps=3,
    )
    batch = torch.randint(0, cfg.vocab_size, (1, 16))

    snap = {k: v.detach().clone() for k, v in model.state_dict().items()}
    trainer.train_step(batch)
    assert trainer.step == 0
    for k, v in model.state_dict().items():
        assert torch.equal(v, snap[k]), f"weights moved on micro-step 1: {k}"

    trainer.train_step(batch)
    assert trainer.step == 0  # still mid-accumulation
    for k, v in model.state_dict().items():
        assert torch.equal(v, snap[k]), f"weights moved on micro-step 2: {k}"

    trainer.train_step(batch)
    assert trainer.step == 1  # macro step 1 fired
    changed = any(not torch.equal(v, snap[k]) for k, v in model.state_dict().items())
    assert changed, "weights did not move on the 3rd micro-step"


def test_grad_accum_metrics_callback_only_on_macro_step() -> None:
    """Callback fires K times less than train_step calls under K-step accumulation."""
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    seen: list[int] = []
    trainer = Trainer(
        model, opt, loss_fn=_ce_loss_fn,
        gradient_accumulation_steps=4,
        metrics_callback=lambda step, _m: seen.append(step),
    )
    for _ in range(8):
        trainer.train_step(torch.zeros(1, 16, dtype=torch.long))
    # 8 micro-steps / 4 accumulation = 2 macro steps fired.
    assert seen == [1, 2]


def test_grad_accum_lr_scheduler_ticks_on_macro_step_only() -> None:
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)
    trainer = Trainer(
        model, opt, loss_fn=_ce_loss_fn,
        lr_scheduler=sched,
        gradient_accumulation_steps=3,
    )
    batch = torch.zeros(1, 16, dtype=torch.long)
    # Two micro-steps: scheduler shouldn't have ticked.
    trainer.train_step(batch)
    trainer.train_step(batch)
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0e-3)
    # Third → macro step → scheduler ticks.
    trainer.train_step(batch)
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0e-4)


def test_grad_accum_invalid_value() -> None:
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        Trainer(model, opt, loss_fn=_ce_loss_fn, gradient_accumulation_steps=0)


def test_grad_accum_default_is_one() -> None:
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    trainer = Trainer(model, opt, loss_fn=_ce_loss_fn)
    assert trainer.gradient_accumulation_steps == 1
    # And step counter should advance per call.
    trainer.train_step(torch.zeros(1, 16, dtype=torch.long))
    assert trainer.step == 1


def test_grad_accum_skip_resets_micro_step_counter() -> None:
    """A NaN skip mid-accumulation discards the partial accumulation and resets."""
    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

    nan_now = [False]

    def maybe_nan_loss(m: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        out = m(batch)
        v = out["logits"].mean() * 0.0
        if nan_now[0]:
            return v + float("nan")
        return v + 1.0

    trainer = Trainer(model, opt, loss_fn=maybe_nan_loss, gradient_accumulation_steps=3)
    trainer.train_step(torch.zeros(1, 16, dtype=torch.long))  # micro 1
    nan_now[0] = True
    trainer.train_step(torch.zeros(1, 16, dtype=torch.long))  # NaN skip, resets
    assert trainer._micro_step == 0
    assert trainer.skipped_steps == 1
    assert trainer.step == 0
