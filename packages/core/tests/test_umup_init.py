"""Tests for u-μP init scheme (ADR-0011 / OPT-02)."""

from __future__ import annotations

import math

import pytest
import torch
from saint_llm_core.config import ModelConfig
from saint_llm_core.init import (
    split_umup_groups,
    umup_init,
    umup_param_groups,
)
from saint_llm_core.model import SaintLLM
from torch import nn


def _tiny_cfg(**overrides: object) -> ModelConfig:
    cfg = ModelConfig.tiny()  # type: ignore[attr-defined]
    if overrides:
        cfg = cfg.model_copy(update=overrides)
    return cfg


def _empirical_std(t: torch.Tensor) -> float:
    return float(t.detach().flatten().std(unbiased=False))


def test_umup_init_n_layers_must_be_positive() -> None:
    model = nn.Linear(8, 8)
    with pytest.raises(ValueError, match="n_layers must be positive"):
        umup_init(model, n_layers=0)


def test_umup_init_embedding_unit_std() -> None:
    embed = nn.Embedding(1024, 256)
    umup_init(embed, n_layers=4)
    # Empirical std with embedding_sigma=1.0 should land near 1.0.
    assert _empirical_std(embed.weight) == pytest.approx(1.0, abs=0.05)


def test_umup_init_embedding_custom_sigma() -> None:
    embed = nn.Embedding(1024, 256)
    umup_init(embed, n_layers=4, embedding_sigma=0.5)
    assert _empirical_std(embed.weight) == pytest.approx(0.5, abs=0.05)


def test_umup_init_hidden_linear_uses_inv_sqrt_fan_in() -> None:
    fan_in = 1024
    layer = nn.Linear(fan_in, 512, bias=False)
    umup_init(layer, n_layers=4)
    expected = 1.0 / math.sqrt(fan_in)
    assert _empirical_std(layer.weight) == pytest.approx(expected, rel=0.1)


def test_umup_init_residual_feeder_scaled_by_inv_sqrt_n_layers() -> None:
    """A Linear named ``output.final_proj.weight`` gets the depth factor."""
    container = nn.Module()
    inner = nn.Module()
    proj = nn.Linear(512, 256, bias=False)
    inner.add_module("final_proj", proj)
    container.add_module("output", inner)
    n_layers = 16
    umup_init(container, n_layers=n_layers)
    expected = (1.0 / math.sqrt(512)) / math.sqrt(n_layers)
    assert _empirical_std(proj.weight) == pytest.approx(expected, rel=0.15)


def test_umup_init_down_proj_is_residual_feeder() -> None:
    """Linear named ``down_proj.weight`` (SwiGLU output) gets depth factor."""
    container = nn.Module()
    proj = nn.Linear(2048, 768, bias=False)
    container.add_module("down_proj", proj)
    n_layers = 9
    umup_init(container, n_layers=n_layers)
    expected = (1.0 / math.sqrt(2048)) / math.sqrt(n_layers)
    assert _empirical_std(proj.weight) == pytest.approx(expected, rel=0.15)


def test_umup_init_skips_mhc_dynamic_weights() -> None:
    """w_pre / w_res / w_post stay at zero (uniform gating)."""
    container = nn.Module()
    w_pre = nn.Linear(8, 8, bias=False)
    nn.init.zeros_(w_pre.weight)  # explicit pre-state
    w_post = nn.Linear(8, 8, bias=False)
    nn.init.zeros_(w_post.weight)
    container.add_module("w_pre", w_pre)
    container.add_module("w_post", w_post)

    umup_init(container, n_layers=4)
    assert torch.all(w_pre.weight == 0.0)
    assert torch.all(w_post.weight == 0.0)


def test_umup_init_skips_frozen_params() -> None:
    layer = nn.Linear(8, 8, bias=False)
    nn.init.zeros_(layer.weight)
    layer.weight.requires_grad_(False)
    umup_init(layer, n_layers=4)
    # Frozen, kept at zero.
    assert torch.all(layer.weight == 0.0)


def test_umup_init_zeroes_bias() -> None:
    layer = nn.Linear(16, 16, bias=True)
    nn.init.constant_(layer.bias, 1.5)
    umup_init(layer, n_layers=4)
    assert torch.all(layer.bias == 0.0)


def test_init_scheme_normal_default() -> None:
    cfg = _tiny_cfg()
    assert cfg.init_scheme == "normal"


def test_init_scheme_umup_dispatches_in_saintllm() -> None:
    """A SaintLLM with init_scheme='umup' applies u-μP at construction."""
    cfg = _tiny_cfg(init_scheme="umup")
    model = SaintLLM(cfg)
    # Embedding should be ~ Normal(0, 1.0).
    embed_std = _empirical_std(model.embed.weight)
    assert embed_std == pytest.approx(1.0, abs=0.1)


def test_umup_param_groups_basic_shape() -> None:
    cfg = _tiny_cfg()
    model = SaintLLM(cfg)
    groups = umup_param_groups(
        model,
        base_lr_hidden=1e-3,
        base_lr_embedding=1e-2,
        base_width=cfg.hidden_dim,
    )
    assert len(groups) == 2
    names = {g["name"] for g in groups}
    assert names == {"embedding", "hidden"}
    by_name = {g["name"]: g for g in groups}
    # Embedding LR is the base.
    assert by_name["embedding"]["lr"] == 1e-2
    # Hidden LR scaled by base_width / actual_width = 1.
    assert by_name["hidden"]["lr"] == 1e-3


def test_umup_param_groups_width_scales_hidden_lr() -> None:
    cfg = _tiny_cfg()
    model = SaintLLM(cfg)
    base_width = cfg.hidden_dim * 2  # double the production tuned width
    groups = umup_param_groups(
        model,
        base_lr_hidden=1e-3,
        base_lr_embedding=1e-2,
        base_width=base_width,
    )
    by_name = {g["name"]: g for g in groups}
    # Smaller actual_width than base → hidden LR scales UP.
    assert by_name["hidden"]["lr"] == pytest.approx(1e-3 * 2.0)


def test_umup_param_groups_tied_embedding_dedup() -> None:
    """Tied lm_head + embed appears once in the embedding group."""
    cfg = _tiny_cfg(tie_word_embeddings=True)
    model = SaintLLM(cfg)
    groups = umup_param_groups(
        model,
        base_lr_hidden=1e-3,
        base_lr_embedding=1e-2,
        base_width=cfg.hidden_dim,
    )
    embed_group = next(g for g in groups if g["name"] == "embedding")
    # The tied parameter appears exactly once.
    embed_ids = [id(p) for p in embed_group["params"]]  # type: ignore[union-attr]
    assert len(embed_ids) == len(set(embed_ids))


def test_umup_param_groups_skips_frozen() -> None:
    cfg = _tiny_cfg()
    model = SaintLLM(cfg)
    # Freeze a parameter; it shouldn't appear in any group.
    model.embed.weight.requires_grad_(False)
    groups = umup_param_groups(
        model,
        base_lr_hidden=1e-3,
        base_lr_embedding=1e-2,
        base_width=cfg.hidden_dim,
    )
    embed_group = next(g for g in groups if g["name"] == "embedding")
    assert len(embed_group["params"]) == 0  # type: ignore[arg-type]


def test_split_umup_groups_returns_tuple() -> None:
    cfg = _tiny_cfg()
    model = SaintLLM(cfg)
    _embed_p, hidden_p, lrs = split_umup_groups(
        model,
        base_lr_hidden=1e-3,
        base_lr_embedding=1e-2,
        base_width=cfg.hidden_dim,
    )
    assert lrs == {"embedding": 1e-2, "hidden": 1e-3}
    # Hidden has at least the LM head and block weights.
    assert len(hidden_p) > 0
