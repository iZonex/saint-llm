"""Checkpoint round-trip: weights, optimizer state, step, extra payload."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM
from saint_llm_training import load_checkpoint, save_checkpoint
from torch import nn


def _all_close(a: nn.Module, b: nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    if set(sa.keys()) != set(sb.keys()):
        return False
    return all(torch.equal(sa[k], sb[k]) for k in sa)


def _opt_states_equal(a: torch.optim.Optimizer, b: torch.optim.Optimizer) -> bool:
    sa = a.state_dict()
    sb = b.state_dict()
    return str(sa) == str(sb)


def test_round_trip_model_weights(tmp_path: Path) -> None:
    cfg = ModelConfig.tiny()
    torch.manual_seed(0)
    src = SaintLLM(cfg)
    save_checkpoint(tmp_path / "ckpt.pt", src, step=42)

    torch.manual_seed(1)  # different init
    dst = SaintLLM(cfg)
    assert not _all_close(src, dst)

    meta = load_checkpoint(tmp_path / "ckpt.pt", dst)
    assert _all_close(src, dst)
    assert meta["step"] == 42


def test_round_trip_with_optimizer(tmp_path: Path) -> None:
    cfg = ModelConfig.tiny()
    torch.manual_seed(0)
    src = SaintLLM(cfg)
    src_opt = torch.optim.AdamW(src.parameters(), lr=1.0e-3)
    # Take one step so optimizer accumulates moments.
    token_ids = torch.zeros(1, 8, dtype=torch.long)
    out = src(token_ids)
    loss = out["logits"].mean()
    loss.backward()
    src_opt.step()

    save_checkpoint(tmp_path / "ckpt.pt", src, src_opt, step=1)

    torch.manual_seed(1)
    dst = SaintLLM(cfg)
    dst_opt = torch.optim.AdamW(dst.parameters(), lr=1.0e-3)
    load_checkpoint(tmp_path / "ckpt.pt", dst, dst_opt)

    assert _all_close(src, dst)
    assert _opt_states_equal(src_opt, dst_opt)


def test_extra_payload_preserved(tmp_path: Path) -> None:
    cfg = ModelConfig.tiny()
    src = SaintLLM(cfg)
    save_checkpoint(
        tmp_path / "ckpt.pt", src, step=5,
        extra={"data_cursor": 1234, "rng_seed": 99},
    )
    dst = SaintLLM(cfg)
    meta = load_checkpoint(tmp_path / "ckpt.pt", dst)
    assert meta["extra"] == {"data_cursor": 1234, "rng_seed": 99}


def test_step_zero_default(tmp_path: Path) -> None:
    cfg = ModelConfig.tiny()
    src = SaintLLM(cfg)
    save_checkpoint(tmp_path / "ckpt.pt", src, step=0)
    dst = SaintLLM(cfg)
    meta = load_checkpoint(tmp_path / "ckpt.pt", dst)
    assert meta["step"] == 0


def test_load_without_optimizer_when_saved_with(tmp_path: Path) -> None:
    """Saving with optimizer + loading without should silently skip it."""
    cfg = ModelConfig.tiny()
    src = SaintLLM(cfg)
    src_opt = torch.optim.AdamW(src.parameters(), lr=1.0e-3)
    save_checkpoint(tmp_path / "ckpt.pt", src, src_opt, step=0)
    dst = SaintLLM(cfg)
    load_checkpoint(tmp_path / "ckpt.pt", dst)  # no optimizer arg
    # No exception, weights loaded.
    assert _all_close(src, dst)


def test_load_with_optimizer_but_none_saved_raises(tmp_path: Path) -> None:
    cfg = ModelConfig.tiny()
    src = SaintLLM(cfg)
    save_checkpoint(tmp_path / "ckpt.pt", src, step=0)
    dst = SaintLLM(cfg)
    dst_opt = torch.optim.AdamW(dst.parameters(), lr=1.0e-3)
    with pytest.raises(ValueError, match="no optimizer state"):
        load_checkpoint(tmp_path / "ckpt.pt", dst, dst_opt)


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    cfg = ModelConfig.tiny()
    src = SaintLLM(cfg)
    nested = tmp_path / "deeply" / "nested" / "dir" / "ckpt.pt"
    save_checkpoint(nested, src, step=0)
    assert nested.exists()


def test_load_missing_model_key_raises(tmp_path: Path) -> None:
    cfg = ModelConfig.tiny()
    dst = SaintLLM(cfg)
    bad = tmp_path / "bad.pt"
    torch.save({"step": 0, "extra": {}}, bad)
    with pytest.raises(ValueError, match="missing 'model'"):
        load_checkpoint(bad, dst)


def test_load_strict_false_allows_partial_state(tmp_path: Path) -> None:
    """strict=False permits state-dict shape mismatches (extra/missing keys)."""
    cfg = ModelConfig.tiny()
    src = SaintLLM(cfg)
    save_checkpoint(tmp_path / "ckpt.pt", src, step=0)

    # Tamper: drop one key from saved state to simulate a missing key.
    payload = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    state = payload["model"]
    dropped_key = next(iter(state.keys()))
    del state[dropped_key]
    payload["model"] = state
    torch.save(payload, tmp_path / "partial.pt")

    dst = SaintLLM(cfg)
    # strict=True would raise; strict=False loads what it can.
    load_checkpoint(tmp_path / "partial.pt", dst, strict=False)
