"""SaintLLM with linear_quant config flag: build + forward in each mode."""

from __future__ import annotations

import pytest
import torch
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM
from saint_llm_core.moe import SwiGLU
from saint_llm_core.quant import make_linear_factory
from saint_llm_kernels import Fp4Linear, Fp8Linear
from torch import nn


def _tiny() -> ModelConfig:
    return ModelConfig.tiny()


def test_default_linear_quant_is_bf16() -> None:
    cfg = _tiny()
    assert cfg.linear_quant == "bf16"


def test_bf16_mode_keeps_nn_linear_in_experts() -> None:
    cfg = _tiny()
    model = SaintLLM(cfg)
    # Find the first MoE block past hash routing and check expert linear types.
    for block in model.blocks:
        first_expert = block.moe.routed_experts[0]
        assert isinstance(first_expert, SwiGLU)
        assert isinstance(first_expert.gate_proj, nn.Linear)
        assert type(first_expert.gate_proj) is nn.Linear  # exact, not a subclass


@pytest.mark.parametrize("mode", ["fp8", "fp4"])
def test_quant_mode_swaps_expert_linears(mode: str) -> None:
    cfg = _tiny().model_copy(update={"linear_quant": mode})
    model = SaintLLM(cfg)
    first_expert = model.blocks[0].moe.routed_experts[0]
    assert type(first_expert.gate_proj) is not nn.Linear


@pytest.mark.parametrize("mode", ["fp8", "fp4"])
def test_quant_mode_swaps_attention_projections(mode: str) -> None:
    """CSA/HCA/SWAttention top-level projections (q_compressor, q_up, k_proj, v_proj)
    must swap to Fp8Linear/Fp4Linear under fp8/fp4 modes."""
    cfg = _tiny().model_copy(update={"linear_quant": mode})
    model = SaintLLM(cfg)
    saw_csa_or_hca = False
    for block in model.blocks:
        attn = block.attention
        for attr in ("q_compressor", "q_up", "k_proj", "v_proj"):
            layer = getattr(attn, attr, None)
            assert layer is not None, f"{type(attn).__name__} missing {attr}"
            assert type(layer) is not nn.Linear, (
                f"{type(attn).__name__}.{attr} not swapped under {mode}"
            )
        if type(attn).__name__ in ("CSA", "HCA"):
            saw_csa_or_hca = True
    assert saw_csa_or_hca, "tiny config should have at least one CSA/HCA block"


@pytest.mark.parametrize("mode", ["fp8", "fp4"])
def test_quant_mode_swaps_attention_nested_helpers(mode: str) -> None:
    """kv_compressor (TokenLevelCompressor) + output (GroupedOutputProjection)
    in CSA/HCA/SWAttention must use the quantized linear_factory under fp8/fp4."""
    cfg = _tiny().model_copy(update={"linear_quant": mode})
    model = SaintLLM(cfg)
    for block in model.blocks:
        attn = block.attention
        # GroupedOutputProjection is shared by all attention types.
        out = attn.output
        assert type(out.final_proj) is not nn.Linear
        for proj in out.group_projs:
            assert type(proj) is not nn.Linear
        # kv_compressor only exists on CSA/HCA, not SWAttention.
        kv = getattr(attn, "kv_compressor", None)
        if kv is not None:
            for attr in ("w_a_kv", "w_b_kv", "w_a_z", "w_b_z"):
                layer = getattr(kv, attr)
                assert type(layer) is not nn.Linear, f"kv_compressor.{attr} not swapped"


def test_indexer_nested_compressor_stays_unquantized() -> None:
    """LightningIndexer's nested TokenLevelCompressor must stay nn.Linear —
    quant noise on indexer scores would destabilize routing."""
    cfg = _tiny().model_copy(update={"linear_quant": "fp8"})
    model = SaintLLM(cfg)
    csa_block = next(b for b in model.blocks if type(b.attention).__name__ == "CSA")
    indexer = csa_block.attention.indexer
    for attr in ("w_a_kv", "w_b_kv", "w_a_z", "w_b_z"):
        layer = getattr(indexer.compressor, attr)
        assert type(layer) is nn.Linear, f"indexer.compressor.{attr} should stay nn.Linear"
    for attr in ("w_dq", "w_iuq", "w_w"):
        assert type(getattr(indexer, attr)) is nn.Linear


@pytest.mark.parametrize("mode", ["bf16", "fp8", "fp4"])
def test_forward_finite_in_each_quant_mode(mode: str) -> None:
    """End-to-end: SaintLLM forward in each quant mode produces a finite logits tensor."""
    cfg = _tiny().model_copy(update={"linear_quant": mode})
    model = SaintLLM(cfg)
    model.eval()
    token_ids = torch.zeros(1, 16, dtype=torch.long)
    with torch.no_grad():
        out = model(token_ids)
    logits = out["logits"] if isinstance(out, dict) else out
    assert torch.isfinite(logits).all()


def test_router_score_proj_stays_unquantized() -> None:
    """Routing decisions must not be quantized — score_proj kept as nn.Linear."""
    cfg = _tiny().model_copy(update={"linear_quant": "fp8"})
    model = SaintLLM(cfg)
    # Layers past hash_routed_layers have learned routers.
    for block in model.blocks:
        if block.moe.router is not None:
            assert type(block.moe.router.score_proj) is nn.Linear


def test_lm_head_stays_unquantized() -> None:
    cfg = _tiny().model_copy(update={"linear_quant": "fp8"})
    model = SaintLLM(cfg)
    assert type(model.lm_head) is nn.Linear


def test_invalid_mode_rejected_at_factory() -> None:
    with pytest.raises(ValueError, match="unknown linear_quant mode"):
        make_linear_factory("int8")  # type: ignore[arg-type]


def test_fp4_block_size_propagates_to_factory() -> None:
    cfg = _tiny().model_copy(update={"linear_quant": "fp4", "fp4_block_size": 16})
    model = SaintLLM(cfg)
    first_expert = model.blocks[0].moe.routed_experts[0]
    assert isinstance(first_expert.gate_proj, Fp4Linear)
    assert first_expert.gate_proj.block_size == 16


def test_fp8_use_real_gemm_default_false() -> None:
    """Default config does not opt into _scaled_mm — preserves CPU/Mac behavior."""
    cfg = _tiny().model_copy(update={"linear_quant": "fp8"})
    model = SaintLLM(cfg)
    first_expert = model.blocks[0].moe.routed_experts[0]
    assert isinstance(first_expert.gate_proj, Fp8Linear)
    assert first_expert.gate_proj.use_real_fp8_gemm is False


def test_fp8_use_real_gemm_propagates_to_layers() -> None:
    cfg = _tiny().model_copy(update={"linear_quant": "fp8", "fp8_use_real_gemm": True})
    model = SaintLLM(cfg)
    first_expert = model.blocks[0].moe.routed_experts[0]
    assert isinstance(first_expert.gate_proj, Fp8Linear)
    assert first_expert.gate_proj.use_real_fp8_gemm is True
    # Attention top-level Linears should also flip the flag.
    for block in model.blocks:
        attn = block.attention
        for attr in ("q_compressor", "q_up", "k_proj", "v_proj"):
            layer = getattr(attn, attr)
            assert isinstance(layer, Fp8Linear)
            assert layer.use_real_fp8_gemm is True
