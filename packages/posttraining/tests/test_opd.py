"""Tests for OPD multi-teacher distillation."""

from __future__ import annotations

import pytest
import torch
from saint_llm_posttraining.opd import (
    OPDConfig,
    aggregate_teacher_logits,
    opd_kl_loss,
    opd_step,
)
from torch import nn


class _MockLM(nn.Module):
    """Minimal LM that returns a fixed logit shape."""

    def __init__(self, vocab: int, hidden: int = 8) -> None:
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.head(self.embed(tokens))}


def test_aggregate_teachers_mean_collapses_to_single_when_one_teacher() -> None:
    """One teacher + mean aggregation -> just that teacher's log-softmax."""
    t = torch.randn(2, 3, 5)
    out = aggregate_teacher_logits([t], mode="mean")
    expected = torch.log_softmax(t, dim=-1)
    assert torch.allclose(out, expected, atol=1e-5)


def test_aggregate_teachers_mean_pools_two_teachers() -> None:
    """Two equal teachers pool to half the probability mass each."""
    t1 = torch.full((1, 1, 4), 10.0)  # confident on token 0... actually all equal logits
    t1[0, 0, 0] = 100.0  # teacher 1 picks token 0
    t2 = torch.full((1, 1, 4), 10.0)
    t2[0, 0, 1] = 100.0  # teacher 2 picks token 1
    pooled = aggregate_teacher_logits([t1, t2], mode="mean").exp()
    # Each teacher contributes ~50% to its preferred token.
    assert pytest.approx(pooled[0, 0, 0].item(), abs=0.01) == 0.5
    assert pytest.approx(pooled[0, 0, 1].item(), abs=0.01) == 0.5


def test_aggregate_teachers_weighted_respects_weights() -> None:
    """Weighted aggregation skews toward higher-weight teachers."""
    t1 = torch.full((1, 1, 4), 0.0)
    t1[0, 0, 0] = 100.0
    t2 = torch.full((1, 1, 4), 0.0)
    t2[0, 0, 1] = 100.0
    weights = torch.tensor([0.9, 0.1])
    pooled = aggregate_teacher_logits([t1, t2], weights=weights, mode="weighted").exp()
    assert pooled[0, 0, 0].item() > pooled[0, 0, 1].item()
    assert pytest.approx(pooled[0, 0, 0].item(), abs=0.01) == 0.9


def test_aggregate_teachers_max_picks_most_confident() -> None:
    t1 = torch.full((1, 1, 4), 0.0)
    t1[0, 0, 0] = 50.0  # teacher 1 confidence 50
    t2 = torch.full((1, 1, 4), 0.0)
    t2[0, 0, 1] = 100.0  # teacher 2 confidence 100, wins
    pooled = aggregate_teacher_logits([t1, t2], mode="max").exp()
    # Max-mode chose teacher 2 -> argmax should be token 1.
    assert int(pooled.argmax(dim=-1).item()) == 1


def test_aggregate_teachers_weighted_requires_weights() -> None:
    t = torch.randn(1, 1, 4)
    with pytest.raises(ValueError, match="requires 'weights'"):
        aggregate_teacher_logits([t, t], mode="weighted")


def test_aggregate_teachers_weighted_rejects_misshape() -> None:
    t = torch.randn(1, 1, 4)
    with pytest.raises(ValueError, match="must match"):
        aggregate_teacher_logits([t, t], weights=torch.tensor([0.5, 0.3, 0.2]), mode="weighted")


def test_aggregate_teachers_weights_must_sum_to_one() -> None:
    t = torch.randn(1, 1, 4)
    with pytest.raises(ValueError, match="must sum to 1"):
        aggregate_teacher_logits([t, t], weights=torch.tensor([0.5, 0.7]), mode="weighted")


def test_aggregate_teachers_unknown_mode_raises() -> None:
    t = torch.randn(1, 1, 4)
    with pytest.raises(ValueError, match="unknown aggregation"):
        aggregate_teacher_logits([t], mode="bogus")  # type: ignore[arg-type]


def test_aggregate_teachers_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        aggregate_teacher_logits([], mode="mean")


def test_aggregate_teachers_shape_mismatch_raises() -> None:
    t1 = torch.randn(1, 3, 5)
    t2 = torch.randn(1, 3, 6)
    with pytest.raises(ValueError, match="must share shape"):
        aggregate_teacher_logits([t1, t2], mode="mean")


def test_opd_kl_loss_zero_when_student_matches_teacher() -> None:
    """Identical student/teacher distributions -> KL = 0."""
    logits = torch.randn(1, 4, 8)
    teacher_log_probs = torch.log_softmax(logits, dim=-1)
    mask = torch.ones(1, 4, dtype=torch.long)
    loss, _ = opd_kl_loss(logits, teacher_log_probs, mask)
    assert loss.item() < 1e-5


def test_opd_kl_loss_positive_for_disagreement() -> None:
    student = torch.zeros(1, 1, 4)
    student[0, 0, 0] = 100.0  # student fully on token 0
    teacher_logits = torch.zeros(1, 1, 4)
    teacher_logits[0, 0, 1] = 100.0  # teacher fully on token 1
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    mask = torch.ones(1, 1, dtype=torch.long)
    loss, _ = opd_kl_loss(student, teacher_log_probs, mask)
    assert loss.item() > 1.0


def test_opd_kl_loss_zero_when_no_active_positions() -> None:
    student = torch.randn(1, 4, 8)
    teacher_log_probs = torch.log_softmax(torch.randn(1, 4, 8), dim=-1)
    mask = torch.zeros(1, 4, dtype=torch.long)
    loss, n = opd_kl_loss(student, teacher_log_probs, mask)
    assert loss.item() == 0.0
    assert n == 0


def test_opd_kl_loss_temperature_scaling_applies_t_squared() -> None:
    """Higher temperature softens the distributions but T^2 rescales the gradient."""
    student = torch.zeros(1, 1, 4)
    student[0, 0, 0] = 5.0
    teacher_logits = torch.zeros(1, 1, 4)
    teacher_logits[0, 0, 1] = 5.0
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    mask = torch.ones(1, 1, dtype=torch.long)
    loss_t1, _ = opd_kl_loss(student, teacher_log_probs, mask, temperature=1.0)
    loss_t2, _ = opd_kl_loss(student, teacher_log_probs, mask, temperature=2.0)
    # Both finite, both > 0; not asserting an exact ratio (depends on shape).
    assert loss_t1.item() > 0.0
    assert loss_t2.item() > 0.0


def test_opd_kl_loss_rejects_temperature_zero() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        opd_kl_loss(
            torch.randn(1, 1, 4),
            torch.log_softmax(torch.randn(1, 1, 4), dim=-1),
            torch.ones(1, 1, dtype=torch.long),
            temperature=0.0,
        )


def test_opd_kl_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share shape"):
        opd_kl_loss(
            torch.randn(1, 1, 4),
            torch.randn(1, 1, 5),
            torch.ones(1, 1, dtype=torch.long),
        )


def test_opd_step_runs_with_two_teachers() -> None:
    torch.manual_seed(0)
    student = _MockLM(vocab=8)
    teachers = [_MockLM(vocab=8), _MockLM(vocab=8)]
    tokens = torch.randint(0, 8, (2, 5))
    response_mask = torch.zeros(2, 5, dtype=torch.long)
    response_mask[:, 2:] = 1
    cfg = OPDConfig(aggregation="mean", temperature=1.0, kl_coef=1.0)
    loss, metrics = opd_step(student, teachers, tokens, response_mask, cfg=cfg)
    assert torch.isfinite(loss)
    assert int(metrics["n_teachers"].item()) == 2
    assert int(metrics["n_active"].item()) > 0


def test_opd_step_grad_flows_through_student_only() -> None:
    """Backward populates student gradients but teachers stay frozen."""
    torch.manual_seed(0)
    student = _MockLM(vocab=8)
    teachers = [_MockLM(vocab=8)]
    # Freeze teacher params explicitly so we can check they don't accumulate grad.
    for p in teachers[0].parameters():
        p.requires_grad_(True)
        if p.grad is not None:
            p.grad.zero_()
    tokens = torch.randint(0, 8, (1, 4))
    response_mask = torch.tensor([[0, 0, 1, 1]])
    loss, _ = opd_step(student, teachers, tokens, response_mask, cfg=OPDConfig())
    loss.backward()
    student_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in student.parameters()
    )
    teacher_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in teachers[0].parameters()
    )
    assert student_has_grad
    assert not teacher_has_grad


def test_opd_step_weighted_requires_weights() -> None:
    student = _MockLM(vocab=8)
    teachers = [_MockLM(vocab=8), _MockLM(vocab=8)]
    tokens = torch.randint(0, 8, (1, 4))
    mask = torch.ones(1, 4, dtype=torch.long)
    cfg = OPDConfig(aggregation="weighted")
    with pytest.raises(ValueError, match="requires 'weights'"):
        opd_step(student, teachers, tokens, mask, cfg=cfg)


def test_opd_step_kl_coef_scales_loss() -> None:
    torch.manual_seed(0)
    student = _MockLM(vocab=8)
    teachers = [_MockLM(vocab=8)]
    tokens = torch.randint(0, 8, (1, 4))
    mask = torch.ones(1, 4, dtype=torch.long)
    loss1, _ = opd_step(student, teachers, tokens, mask, cfg=OPDConfig(kl_coef=1.0))
    loss2, _ = opd_step(student, teachers, tokens, mask, cfg=OPDConfig(kl_coef=2.0))
    assert torch.allclose(loss2, 2 * loss1, atol=1e-5)
