"""End-to-end QAT smoke test: SaintLLM trains under fp8 / fp4 quant modes.

Validates the whole quant chain — kernels → Fp8/Fp4Linear → ModelConfig.linear_quant
→ SaintLLM forward + backward + AdamW step. Asserts loss strictly decreases over a
few steps on a degenerate task (predict next token of a fixed sequence). Catches:
* STE not actually identity (gradients zeroed by quant op);
* Quant ops yielding non-finite gradients;
* Plumbing not reaching the experts at all.

Tiny scope (cfg.tiny + 16 tokens + 4 steps) so it runs in seconds on Mac/MPS.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM


def _fixed_token_batch(cfg: ModelConfig, *, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (1, 16), generator=g)


def _train_step_loss(model: SaintLLM, token_ids: torch.Tensor, optim: torch.optim.Optimizer) -> float:
    out = model(token_ids)
    logits = out["logits"]
    assert isinstance(logits, torch.Tensor)
    flat_logits = logits[:, :-1].reshape(-1, model.cfg.vocab_size)
    flat_labels = token_ids[:, 1:].reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_labels)
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()


@pytest.mark.parametrize("mode", ["bf16", "fp8", "fp4"])
def test_loss_decreases_over_a_few_steps(mode: str) -> None:
    """Memorize a 16-token sequence; loss must strictly drop over 4 steps."""
    torch.manual_seed(0)
    cfg = ModelConfig.tiny().model_copy(update={"linear_quant": mode})
    model = SaintLLM(cfg)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=3.0e-3)
    token_ids = _fixed_token_batch(cfg)

    losses = [_train_step_loss(model, token_ids, optim) for _ in range(4)]

    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses), losses
    assert losses[-1] < losses[0], f"loss did not decrease in {mode}: {losses}"


@pytest.mark.parametrize("mode", ["fp8", "fp4"])
def test_quant_modes_yield_finite_gradients(mode: str) -> None:
    """STE must produce finite gradients on every parameter that has one."""
    torch.manual_seed(0)
    cfg = ModelConfig.tiny().model_copy(update={"linear_quant": mode})
    model = SaintLLM(cfg)
    model.train()
    token_ids = _fixed_token_batch(cfg)

    out = model(token_ids)
    logits = out["logits"]
    assert isinstance(logits, torch.Tensor)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size),
        token_ids[:, 1:].reshape(-1),
    )
    loss.backward()

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
