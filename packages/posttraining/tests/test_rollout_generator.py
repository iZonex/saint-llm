"""Tests for GRPO rollout generation."""

from __future__ import annotations

import pytest
import torch
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM
from saint_llm_posttraining import (
    GRPOConfig,
    build_rollout_batch,
    grpo_loss,
)
from saint_llm_posttraining.rollout_generator import (
    RolloutGenConfig,
    decode_rollouts_for_reward,
    generate_grpo_rollouts,
)
from torch import nn


class _FixedNextTokenLM(nn.Module):
    """Tiny LM whose argmax always picks ``next_token`` regardless of input."""

    def __init__(self, vocab: int, next_token: int) -> None:
        super().__init__()
        self.vocab = vocab
        self.next_token = next_token
        self.embed = nn.Embedding(vocab, 4)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t = tokens.shape
        logits = torch.full((b, t, self.vocab), -1e9)
        logits[:, :, self.next_token] = 0.0
        return {"logits": logits}


@pytest.fixture(scope="module")
def real_model() -> SaintLLM:
    torch.manual_seed(0)
    m = SaintLLM(ModelConfig.tiny())
    m.eval()
    return m


def test_rollouts_have_correct_shape(real_model: SaintLLM) -> None:
    cfg = RolloutGenConfig(group_size=4, max_new_tokens=3, sampler="greedy")
    prompts = torch.randint(0, ModelConfig.tiny().vocab_size, (2, 5))
    tokens, mask = generate_grpo_rollouts(real_model, prompts, cfg)
    assert tokens.shape == (2 * 4, 5 + 3)
    assert mask.shape == (2 * 4, 5 + 3)


def test_rollouts_layout_groups_by_prompt(real_model: SaintLLM) -> None:
    """Rows [b*G : (b+1)*G] should share the same prompt prefix."""
    cfg = RolloutGenConfig(group_size=3, max_new_tokens=2, sampler="greedy")
    prompts = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tokens, _ = generate_grpo_rollouts(real_model, prompts, cfg)
    # First 3 rows share prompt [1, 2, 3]; next 3 rows share [4, 5, 6].
    for i in range(3):
        assert tokens[i, :3].tolist() == [1, 2, 3]
    for i in range(3, 6):
        assert tokens[i, :3].tolist() == [4, 5, 6]


def test_response_mask_is_zero_on_prompt_positions(real_model: SaintLLM) -> None:
    cfg = RolloutGenConfig(group_size=2, max_new_tokens=4, sampler="greedy")
    prompts = torch.zeros(1, 3, dtype=torch.long)
    _, mask = generate_grpo_rollouts(real_model, prompts, cfg)
    # First 3 columns are prompt -> mask 0.
    assert mask[:, :3].sum().item() == 0
    # Remaining 4 columns are response -> mask 1 (no eos in this case).
    assert mask[:, 3:].sum().item() == 2 * 4


def test_response_mask_zeroes_after_first_eos() -> None:
    """When EOS appears, positions strictly after the EOS are masked out."""
    eos = 7
    model = _FixedNextTokenLM(vocab=8, next_token=eos)
    cfg = RolloutGenConfig(
        group_size=2, max_new_tokens=5, sampler="greedy", eos_token=eos,
    )
    prompts = torch.zeros(1, 2, dtype=torch.long)
    _tokens, mask = generate_grpo_rollouts(model, prompts, cfg)
    # Response begins at column 2. First emitted token is eos (since
    # _FixedNextTokenLM always picks eos). The mask should be 1 only at
    # the eos position itself, then 0 thereafter.
    response_mask = mask[:, 2:]
    # Each row: first response position is 1 (eos itself), rest are 0.
    for row in response_mask.tolist():
        assert row[0] == 1
        assert all(v == 0 for v in row[1:])


def test_response_mask_full_when_no_eos_set(real_model: SaintLLM) -> None:
    cfg = RolloutGenConfig(group_size=2, max_new_tokens=3, sampler="greedy", eos_token=None)
    prompts = torch.zeros(1, 2, dtype=torch.long)
    _, mask = generate_grpo_rollouts(real_model, prompts, cfg)
    # All response positions are 1 when eos isn't set.
    assert mask[:, 2:].sum().item() == 2 * 3


def test_top_p_sampling_with_seed_is_reproducible(real_model: SaintLLM) -> None:
    cfg = RolloutGenConfig(
        group_size=4, max_new_tokens=3,
        sampler="top_p", temperature=1.0, top_p=0.9, seed=42,
    )
    prompts = torch.zeros(1, 2, dtype=torch.long)
    out1, _ = generate_grpo_rollouts(real_model, prompts, cfg)
    out2, _ = generate_grpo_rollouts(real_model, prompts, cfg)
    assert torch.equal(out1, out2)


def test_top_p_sampling_creates_group_variance(real_model: SaintLLM) -> None:
    """Different group members should not all be identical (with temp > 0)."""
    cfg = RolloutGenConfig(
        group_size=8, max_new_tokens=4,
        sampler="top_p", temperature=1.0, top_p=0.95, seed=0,
    )
    prompts = torch.zeros(1, 2, dtype=torch.long)
    tokens, _ = generate_grpo_rollouts(real_model, prompts, cfg)
    # The 8 sampled completions in this single-prompt batch should not
    # all be identical token-for-token (probability ~ 0 for tiny models).
    response = tokens[:, 2:]
    n_unique = len({tuple(row.tolist()) for row in response})
    assert n_unique > 1


def test_rollouts_reject_1d_prompts(real_model: SaintLLM) -> None:
    cfg = RolloutGenConfig(group_size=2, max_new_tokens=3)
    with pytest.raises(ValueError, match=r"\(B, P\)"):
        generate_grpo_rollouts(real_model, torch.tensor([1, 2, 3]), cfg)


def test_rollouts_reject_zero_group_size(real_model: SaintLLM) -> None:
    cfg = RolloutGenConfig(group_size=0, max_new_tokens=3)
    with pytest.raises(ValueError, match="group_size"):
        generate_grpo_rollouts(real_model, torch.zeros(1, 3, dtype=torch.long), cfg)


def test_rollouts_reject_zero_max_new_tokens(real_model: SaintLLM) -> None:
    cfg = RolloutGenConfig(group_size=2, max_new_tokens=0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate_grpo_rollouts(real_model, torch.zeros(1, 3, dtype=torch.long), cfg)


def test_rollouts_unknown_sampler_raises(real_model: SaintLLM) -> None:
    cfg = RolloutGenConfig(group_size=2, max_new_tokens=3, sampler="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown sampler"):
        generate_grpo_rollouts(real_model, torch.zeros(1, 3, dtype=torch.long), cfg)


def test_decode_rollouts_for_reward_extracts_response_only() -> None:
    tokens = torch.tensor([[1, 2, 3, 9, 8, 7], [4, 5, 6, 0, 1, 2]])
    decode = lambda ids: ",".join(str(i) for i in ids)  # noqa: E731
    out = decode_rollouts_for_reward(tokens, decode, prompt_len=3)
    assert out == ["9,8,7", "0,1,2"]


def test_decode_rollouts_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match=r"\(B\*G, T\)"):
        decode_rollouts_for_reward(
            torch.tensor([1, 2, 3]), lambda ids: "", prompt_len=1,
        )


def test_decode_rollouts_rejects_bad_prompt_len() -> None:
    with pytest.raises(ValueError, match="prompt_len"):
        decode_rollouts_for_reward(
            torch.zeros(1, 3, dtype=torch.long), lambda ids: "", prompt_len=10,
        )


def test_rollouts_compose_with_build_rollout_batch(real_model: SaintLLM) -> None:
    """Rollouts feed into build_rollout_batch -> RolloutBatch -> grpo_loss."""
    cfg = RolloutGenConfig(group_size=4, max_new_tokens=3, sampler="greedy")
    prompts = torch.randint(0, ModelConfig.tiny().vocab_size, (2, 5))
    tokens, mask = generate_grpo_rollouts(real_model, prompts, cfg)
    rewards = torch.randn(8)
    batch = build_rollout_batch(
        actor=real_model, ref=None, tokens=tokens,
        response_mask=mask, rewards=rewards,
    )
    out = real_model(tokens)
    loss, metrics = grpo_loss(out["logits"], batch, cfg=GRPOConfig(group_size=4))
    assert torch.isfinite(loss)
    assert "pg_loss" in metrics
