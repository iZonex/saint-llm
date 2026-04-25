"""Tests for split_for_muon_adamw on a real Saint LLM model."""

from __future__ import annotations

import torch
from saint_llm_optim import split_for_muon_adamw
from torch import nn


def test_basic_split_disjoint_and_complete() -> None:
    """Every trainable param ends up in exactly one group; no overlaps."""
    from saint_llm_core import ModelConfig, SaintLLM

    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    muon_params, adamw_params = split_for_muon_adamw(model)

    muon_ids = {id(p) for p in muon_params}
    adamw_ids = {id(p) for p in adamw_params}
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}

    assert muon_ids.isdisjoint(adamw_ids)
    assert muon_ids | adamw_ids == trainable_ids


def test_embed_in_adamw() -> None:
    from saint_llm_core import ModelConfig, SaintLLM

    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    _muon, adamw = split_for_muon_adamw(model)
    assert any(p is model.embed.weight for p in adamw)


def test_rmsnorm_weights_in_adamw() -> None:
    from saint_llm_core import ModelConfig, SaintLLM
    from saint_llm_core.attention.common import RMSNorm

    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    _muon, adamw = split_for_muon_adamw(model)
    adamw_ids = {id(p) for p in adamw}

    rmsnorm_weights = [m.weight for m in model.modules() if isinstance(m, RMSNorm)]
    assert len(rmsnorm_weights) > 0
    for w in rmsnorm_weights:
        assert id(w) in adamw_ids


def test_mhc_static_biases_and_alphas_in_adamw() -> None:
    from saint_llm_core import ModelConfig, SaintLLM
    from saint_llm_core.residual import MHC

    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    _muon, adamw = split_for_muon_adamw(model)
    adamw_ids = {id(p) for p in adamw}

    mhc_modules = [m for m in model.modules() if isinstance(m, MHC)]
    assert len(mhc_modules) > 0
    for mhc in mhc_modules:
        for p in (mhc.s_pre, mhc.s_res, mhc.s_post, mhc.alpha_pre, mhc.alpha_res, mhc.alpha_post):
            assert id(p) in adamw_ids


def test_mhc_dynamic_w_in_muon() -> None:
    """mHC W_pre/W_res/W_post are 2D matrix Linear weights — Muon territory per V4."""
    from saint_llm_core import ModelConfig, SaintLLM
    from saint_llm_core.residual import MHC

    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    muon, _adamw = split_for_muon_adamw(model)
    muon_ids = {id(p) for p in muon}

    for mhc in (m for m in model.modules() if isinstance(m, MHC)):
        for w in (mhc.w_pre.weight, mhc.w_res.weight, mhc.w_post.weight):
            assert id(w) in muon_ids


def test_attention_projections_in_muon() -> None:
    from saint_llm_core import ModelConfig, SaintLLM
    from saint_llm_core.attention import CSA, HCA, SWAttention

    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    muon, _adamw = split_for_muon_adamw(model)
    muon_ids = {id(p) for p in muon}

    for block in model.blocks:
        attn = block.attention
        assert isinstance(attn, (CSA, HCA, SWAttention))
        assert id(attn.q_compressor.weight) in muon_ids
        assert id(attn.q_up.weight) in muon_ids


def test_no_frozen_params_in_either_group() -> None:
    """Frozen GenerationHeadHook (when disabled) must not reach either optimizer."""
    from saint_llm_core import ModelConfig, SaintLLM

    cfg = ModelConfig.tiny()
    model = SaintLLM(cfg)
    muon, adamw = split_for_muon_adamw(model)
    all_assigned = {id(p) for p in muon} | {id(p) for p in adamw}
    frozen = [p for p in model.parameters() if not p.requires_grad]
    for p in frozen:
        assert id(p) not in all_assigned


def test_works_on_arbitrary_module() -> None:
    """`split_for_muon_adamw` should work on any nn.Module, not just SaintLLM."""
    m = nn.Sequential(
        nn.Linear(8, 16),
        nn.LayerNorm(16),
        nn.Linear(16, 4),
    )
    muon, adamw = split_for_muon_adamw(m)
    # Two Linear weights → Muon; Linear biases + LayerNorm weight/bias → AdamW.
    assert len(muon) == 2
    assert len(adamw) == 4  # 2 Linear biases + LN weight + LN bias

    # Sanity: optimizer accepts the split.
    from saint_llm_optim import Muon as MuonOpt

    MuonOpt(muon)
    torch.optim.AdamW(adamw)
