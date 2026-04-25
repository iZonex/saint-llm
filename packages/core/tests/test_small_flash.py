"""ModelConfig.small_flash: builds, forwards finite, parameter count in target range."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM


@pytest.fixture(scope="module")
def small_cfg() -> ModelConfig:
    return ModelConfig.small_flash()


def test_small_flash_dimensions_quant_compatible(small_cfg: ModelConfig) -> None:
    """Every in_features that goes through fp4 quant must divide block_size=32."""
    assert small_cfg.hidden_dim % 32 == 0
    assert small_cfg.attention.query_compression_dim % 32 == 0
    assert (small_cfg.attention.query_heads * small_cfg.attention.head_dim) % 32 == 0
    assert small_cfg.moe.expert_intermediate_dim % 32 == 0


def test_small_flash_builds(small_cfg: ModelConfig) -> None:
    """SaintLLM(cfg.small_flash) constructs without OOM on CPU."""
    torch.manual_seed(0)
    model = SaintLLM(small_cfg)
    assert sum(1 for _ in model.parameters()) > 0


def test_small_flash_param_count_in_target_range(small_cfg: ModelConfig) -> None:
    """Total param count should land in the ~100-250M band for the v0.1 milestone."""
    model = SaintLLM(small_cfg)
    total = sum(p.numel() for p in model.parameters())
    assert 80_000_000 <= total <= 300_000_000, (
        f"small_flash param count {total:,} outside target band [80M, 300M]"
    )


@pytest.mark.slow
def test_small_flash_forward_finite(small_cfg: ModelConfig) -> None:
    """Forward on a tiny batch must produce finite logits."""
    torch.manual_seed(0)
    model = SaintLLM(small_cfg)
    model.eval()
    token_ids = torch.zeros(1, 64, dtype=torch.long)
    with torch.no_grad():
        out = model(token_ids)
    assert torch.isfinite(out["logits"]).all()


@pytest.mark.slow
def test_small_flash_one_train_step(small_cfg: ModelConfig) -> None:
    """End-to-end: one AdamW step on small_flash without OOM, finite loss."""
    torch.manual_seed(0)
    model = SaintLLM(small_cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    token_ids = torch.randint(0, small_cfg.vocab_size, (1, 64))
    out = model(token_ids)
    loss = F.cross_entropy(
        out["logits"][:, :-1].reshape(-1, small_cfg.vocab_size),
        token_ids[:, 1:].reshape(-1),
    )
    optim.zero_grad()
    loss.backward()
    optim.step()
    assert torch.isfinite(loss).item()
    assert torch.isfinite(model.lm_head.weight).all()
