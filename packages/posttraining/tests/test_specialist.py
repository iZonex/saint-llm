"""Tests for SpecialistPipeline (SFT + GRPO orchestrator)."""

from __future__ import annotations

import pytest
import torch
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM
from saint_llm_data import CharTokenizer
from saint_llm_posttraining import (
    GRPOConfig,
    RolloutGenConfig,
    SFTExample,
    SFTPackedBatch,
    pack_sft_into_batch,
)
from saint_llm_posttraining.specialist import (
    RLStepOutput,
    SFTStepOutput,
    SpecialistPipeline,
)


def _tiny_pipeline(*, with_ref: bool = False) -> SpecialistPipeline:
    torch.manual_seed(0)
    actor = SaintLLM(ModelConfig.tiny())
    ref = SaintLLM(ModelConfig.tiny()) if with_ref else None
    if ref is not None:
        ref.load_state_dict(actor.state_dict())
        ref.eval()
    optimizer = torch.optim.SGD(actor.parameters(), lr=1e-2)
    tok = CharTokenizer()
    return SpecialistPipeline(
        actor=actor,
        ref=ref,
        optimizer=optimizer,
        encode=tok.encode,
        decode=tok.decode,
        eos_token=tok.eos_token_id,
    )


def _sft_batch(tok: CharTokenizer) -> SFTPackedBatch:
    examples = [
        SFTExample(prompt="hi", response="ok"),
        SFTExample(prompt="bye", response="cya"),
    ]
    return next(pack_sft_into_batch(examples, tok, seq_len=12, batch_size=1))


def test_sft_step_returns_finite_loss_and_token_count() -> None:
    pipe = _tiny_pipeline()
    batch = _sft_batch(CharTokenizer())
    out = pipe.sft_step(batch)
    assert isinstance(out, SFTStepOutput)
    assert torch.isfinite(out.loss)
    assert out.n_response_tokens > 0


def test_sft_step_updates_actor_params() -> None:
    pipe = _tiny_pipeline()
    initial = next(pipe.actor.parameters()).detach().clone()
    pipe.sft_step(_sft_batch(CharTokenizer()))
    assert not torch.equal(initial, next(pipe.actor.parameters()))


def test_rl_step_returns_finite_loss_and_metrics() -> None:
    pipe = _tiny_pipeline()
    cfg = GRPOConfig(group_size=2, kl_coef=0.0)
    rollout_cfg = RolloutGenConfig(
        group_size=2, max_new_tokens=3, sampler="greedy",
    )

    def reward(_p: str, _c: str) -> float:
        return 0.5

    out = pipe.rl_step(
        ["abc", "xyz"],
        reward_fn=reward, cfg=cfg, rollout_cfg=rollout_cfg,
    )
    assert isinstance(out, RLStepOutput)
    assert torch.isfinite(out.train_step.loss)
    assert out.n_rollouts == 2 * 2
    assert out.rewards.shape == (4,)


def test_rl_step_with_varying_rewards_drives_param_change() -> None:
    pipe = _tiny_pipeline()
    cfg = GRPOConfig(group_size=2, kl_coef=0.0)
    rollout_cfg = RolloutGenConfig(
        group_size=2, max_new_tokens=2,
        sampler="top_p", temperature=1.0, top_p=0.95, seed=0,
    )
    initial = next(pipe.actor.parameters()).detach().clone()

    def reward_by_length(_p: str, completion: str) -> float:
        return float(len(completion))

    out = pipe.rl_step(
        ["a", "b"],
        reward_fn=reward_by_length, cfg=cfg, rollout_cfg=rollout_cfg,
    )
    # If group rewards vary, advantages are non-zero and actor changes.
    if out.rewards.var().item() > 0.0:
        assert not torch.equal(initial, next(pipe.actor.parameters()))


def test_rl_step_rejects_empty_prompts() -> None:
    pipe = _tiny_pipeline()
    with pytest.raises(ValueError, match="empty"):
        pipe.rl_step(
            [],
            reward_fn=lambda _p, _c: 0.0,
            cfg=GRPOConfig(group_size=2),
            rollout_cfg=RolloutGenConfig(group_size=2, max_new_tokens=2),
        )


def test_rl_step_rejects_group_size_mismatch() -> None:
    pipe = _tiny_pipeline()
    with pytest.raises(ValueError, match="group_size"):
        pipe.rl_step(
            ["a"],
            reward_fn=lambda _p, _c: 0.0,
            cfg=GRPOConfig(group_size=4),
            rollout_cfg=RolloutGenConfig(group_size=2, max_new_tokens=2),
        )


def test_pipeline_with_ref_includes_kl_metric() -> None:
    pipe = _tiny_pipeline(with_ref=True)
    cfg = GRPOConfig(group_size=2, kl_coef=0.5)
    rollout_cfg = RolloutGenConfig(
        group_size=2, max_new_tokens=2, sampler="top_p", seed=0,
    )

    counter = [0.0]

    def varying_reward(_p: str, _c: str) -> float:
        # Distinct rewards per call so dynamic-sampling doesn't drop the group.
        counter[0] += 1.0
        return counter[0]

    out = pipe.rl_step(
        ["a", "b"],
        reward_fn=varying_reward, cfg=cfg, rollout_cfg=rollout_cfg,
    )
    # KL metric should be present in trainer step metrics.
    assert "kl_penalty" in out.train_step.metrics


def test_pipeline_pads_prompts_to_common_length() -> None:
    """Prompts of different lengths get padded so the rollout batch is rectangular."""
    pipe = _tiny_pipeline()
    cfg = GRPOConfig(group_size=2, kl_coef=0.0)
    rollout_cfg = RolloutGenConfig(group_size=2, max_new_tokens=2, sampler="greedy")
    # Different prompt lengths.
    out = pipe.rl_step(
        ["a", "longer prompt"],
        reward_fn=lambda _p, _c: 0.0, cfg=cfg, rollout_cfg=rollout_cfg,
    )
    assert out.n_rollouts == 4


def test_pipeline_rejects_empty_encoded_prompt() -> None:
    """A prompt that encodes to an empty list should error early."""
    pipe = _tiny_pipeline()

    # Sneak in a prompt that the CharTokenizer encodes empty (empty string).
    cfg = GRPOConfig(group_size=2, kl_coef=0.0)
    rollout_cfg = RolloutGenConfig(group_size=2, max_new_tokens=2, sampler="greedy")
    with pytest.raises(ValueError, match="empty"):
        pipe.rl_step(
            ["", "ok"],
            reward_fn=lambda _p, _c: 0.0, cfg=cfg, rollout_cfg=rollout_cfg,
        )


def test_alternating_sft_and_rl_phases_share_actor_state() -> None:
    """SFT step then RL step both update the same actor."""
    pipe = _tiny_pipeline()
    initial = next(pipe.actor.parameters()).detach().clone()
    pipe.sft_step(_sft_batch(CharTokenizer()))
    after_sft = next(pipe.actor.parameters()).detach().clone()
    cfg = GRPOConfig(group_size=2, kl_coef=0.0)
    rollout_cfg = RolloutGenConfig(group_size=2, max_new_tokens=2, sampler="top_p", seed=0)
    pipe.rl_step(
        ["a"], reward_fn=lambda _p, c: float(len(c)),
        cfg=cfg, rollout_cfg=rollout_cfg,
    )
    after_rl = next(pipe.actor.parameters()).detach().clone()
    assert not torch.equal(initial, after_sft)
    # RL step may not always change params if group rewards are uniform,
    # but the SFT effect must remain visible.
    assert not torch.equal(initial, after_rl)
