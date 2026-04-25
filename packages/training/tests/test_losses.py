"""Loss helpers: equivalence to plain CE without MTP, MTP weighting math."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from saint_llm_core.config import ModelConfig, MTPConfig
from saint_llm_core.model import SaintLLM
from saint_llm_training import (
    cross_entropy_main,
    cross_entropy_with_mtp,
    make_loss_fn,
)


@pytest.fixture(scope="module")
def model() -> SaintLLM:
    torch.manual_seed(0)
    return SaintLLM(ModelConfig.tiny())


def test_cross_entropy_main_matches_manual(model: SaintLLM) -> None:
    cfg = ModelConfig.tiny()
    batch = torch.zeros(1, 8, dtype=torch.long)
    with torch.no_grad():
        out = model(batch)
    actual = cross_entropy_main(out, batch)
    expected = F.cross_entropy(
        out["logits"][:, :-1].reshape(-1, cfg.vocab_size),
        batch[:, 1:].reshape(-1),
    )
    assert torch.allclose(actual, expected, atol=1.0e-6)


def test_cross_entropy_with_mtp_includes_main_term(model: SaintLLM) -> None:
    cfg = ModelConfig.tiny()
    batch = torch.zeros(1, 16, dtype=torch.long)
    with torch.no_grad():
        out = model(batch)
    main_only = cross_entropy_main(out, batch)
    with_mtp = cross_entropy_with_mtp(out, batch, cfg=cfg.mtp)
    # Total should be >= main alone (MTP heads add non-negative weighted CE).
    assert with_mtp >= main_only


def test_cross_entropy_with_mtp_zero_depth_falls_back(model: SaintLLM) -> None:
    """depth=0 short-circuits to plain main loss."""
    batch = torch.zeros(1, 8, dtype=torch.long)
    with torch.no_grad():
        out = model(batch)
    cfg_zero = MTPConfig(depth=0)
    main_only = cross_entropy_main(out, batch)
    with_mtp = cross_entropy_with_mtp(out, batch, cfg=cfg_zero)
    assert torch.equal(with_mtp, main_only)


def test_cross_entropy_with_mtp_decay_weights() -> None:
    """Hand-computed weights: alpha_k = main * (1-decay)^k for k=0..depth-1."""
    main = 1.0
    decay = 0.1
    cfg = MTPConfig(depth=3, loss_weight_main=main, loss_weight_decay=decay)
    expected = [main * ((1.0 - decay) ** k) for k in range(cfg.depth)]
    assert expected == [pytest.approx(1.0), pytest.approx(0.9), pytest.approx(0.81)]


def test_cross_entropy_with_mtp_skips_too_short_sequences(model: SaintLLM) -> None:
    """Sequences shorter than the MTP head's shift skip that head silently."""
    cfg = MTPConfig(depth=4, loss_weight_main=0.3, loss_weight_decay=0.1)
    # batch length 3 — only main (shift=1) and MTP head 0 (shift=2) contribute.
    batch = torch.zeros(1, 3, dtype=torch.long)
    with torch.no_grad():
        out = model(batch)
    loss = cross_entropy_with_mtp(out, batch, cfg=cfg)
    assert torch.isfinite(loss).item()


def test_make_loss_fn_no_mtp(model: SaintLLM) -> None:
    """make_loss_fn(None) plugs into Trainer's loss_fn signature and matches plain CE."""
    loss_fn = make_loss_fn(None)
    batch = torch.zeros(1, 8, dtype=torch.long)
    out_loss = loss_fn(model, batch)
    with torch.no_grad():
        expected = cross_entropy_main(model(batch), batch)
    assert torch.allclose(out_loss, expected, atol=1.0e-6)


def test_make_loss_fn_with_mtp(model: SaintLLM) -> None:
    cfg = ModelConfig.tiny()
    loss_fn = make_loss_fn(cfg.mtp)
    batch = torch.zeros(1, 16, dtype=torch.long)
    out_loss = loss_fn(model, batch)
    assert torch.isfinite(out_loss).item()


def test_cross_entropy_main_rejects_non_tensor_logits() -> None:
    bad_out: dict = {"logits": [1, 2, 3]}
    batch = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(TypeError, match="must be a Tensor"):
        cross_entropy_main(bad_out, batch)


def test_cross_entropy_with_mtp_rejects_non_list_mtp() -> None:
    cfg = MTPConfig(depth=1)
    bad_out: dict = {"logits": torch.zeros(1, 4, 8), "mtp_logits": "oops"}
    batch = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(TypeError, match="list of Tensors"):
        cross_entropy_with_mtp(bad_out, batch, cfg=cfg)
