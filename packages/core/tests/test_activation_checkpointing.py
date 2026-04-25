"""Activation checkpointing: equivalence on/off, gradient flow, default off."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM


def _ce(model: SaintLLM, batch: torch.Tensor) -> torch.Tensor:
    out = model(batch)
    return F.cross_entropy(
        out["logits"][:, :-1].reshape(-1, model.cfg.vocab_size),
        batch[:, 1:].reshape(-1),
    )


def test_default_activation_checkpointing_is_off() -> None:
    cfg = ModelConfig.tiny()
    assert cfg.activation_checkpointing is False


def test_eval_mode_skips_checkpointing() -> None:
    """Inference (model.eval()) must not invoke checkpoint regardless of flag."""
    cfg = ModelConfig.tiny().model_copy(update={"activation_checkpointing": True})
    torch.manual_seed(0)
    model = SaintLLM(cfg).eval()
    token_ids = torch.zeros(1, 8, dtype=torch.long)
    with torch.no_grad():
        out = model(token_ids)
    assert torch.isfinite(out["logits"]).all()


def test_forward_output_close_with_and_without_checkpointing() -> None:
    """Forward in train mode should produce equal logits with checkpointing on/off."""
    torch.manual_seed(0)
    cfg_off = ModelConfig.tiny()
    cfg_on = cfg_off.model_copy(update={"activation_checkpointing": True})

    torch.manual_seed(0)
    model_off = SaintLLM(cfg_off).train()
    torch.manual_seed(0)
    model_on = SaintLLM(cfg_on).train()
    model_on.load_state_dict(model_off.state_dict())

    token_ids = torch.randint(0, cfg_off.vocab_size, (1, 8))
    out_off = model_off(token_ids)["logits"]
    out_on = model_on(token_ids)["logits"]
    assert torch.allclose(out_off, out_on, atol=1.0e-5)


def test_gradients_match_with_and_without_checkpointing() -> None:
    """Backward gradients on a representative parameter must match."""
    torch.manual_seed(0)
    cfg_off = ModelConfig.tiny()
    cfg_on = cfg_off.model_copy(update={"activation_checkpointing": True})

    torch.manual_seed(0)
    model_off = SaintLLM(cfg_off).train()
    torch.manual_seed(0)
    model_on = SaintLLM(cfg_on).train()
    model_on.load_state_dict(model_off.state_dict())

    token_ids = torch.randint(0, cfg_off.vocab_size, (1, 8))
    _ce(model_off, token_ids).backward()
    _ce(model_on, token_ids).backward()

    # Compare gradients on every parameter — they must be very close.
    for (n_off, p_off), (n_on, p_on) in zip(
        model_off.named_parameters(), model_on.named_parameters(), strict=True,
    ):
        assert n_off == n_on
        if p_off.grad is None and p_on.grad is None:
            continue
        assert p_off.grad is not None and p_on.grad is not None, n_off
        assert torch.allclose(p_off.grad, p_on.grad, atol=1.0e-4), (
            f"grad mismatch on {n_off}: max diff {(p_off.grad - p_on.grad).abs().max().item()}"
        )


def test_one_train_step_with_checkpointing_on() -> None:
    """End-to-end AdamW step with checkpointing on must produce a finite loss."""
    cfg = ModelConfig.tiny().model_copy(update={"activation_checkpointing": True})
    torch.manual_seed(0)
    model = SaintLLM(cfg).train()
    optim = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

    token_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    losses = []
    for _ in range(3):
        loss = _ce(model, token_ids)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses), losses
    assert losses[-1] < losses[0]


@pytest.mark.parametrize("mode", ["bf16", "fp8", "fp4"])
def test_checkpointing_works_under_each_quant_mode(mode: str) -> None:
    """Checkpointing combined with quant — common combo; both must coexist."""
    cfg = ModelConfig.tiny().model_copy(update={
        "activation_checkpointing": True,
        "linear_quant": mode,
    })
    torch.manual_seed(0)
    model = SaintLLM(cfg).train()
    optim = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    token_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    loss = _ce(model, token_ids)
    optim.zero_grad()
    loss.backward()
    optim.step()
    assert torch.isfinite(loss).item()
