"""Tests for Critique-GRPO orchestration helpers (RL-04 / ADR-0016)."""

from __future__ import annotations

import pytest
import torch
from saint_llm_posttraining import (
    DEFAULT_CRITIQUE_TEMPLATE,
    DEFAULT_REFINEMENT_TEMPLATE,
    CritiqueGRPOConfig,
    CritiqueRequest,
    GRPOConfig,
    collect_critique_requests,
    make_critique_prompt,
    make_refinement_prompt,
    needs_critique_mask,
)


def test_critique_grpo_config_defaults_to_disabled() -> None:
    cfg = CritiqueGRPOConfig(grpo=GRPOConfig())
    assert cfg.enabled is False
    assert cfg.critique_threshold == 0.5
    assert cfg.critique_max_attempts == 1
    assert cfg.critic_self_play is True


def test_critique_grpo_config_inherits_grpo_settings() -> None:
    inner = GRPOConfig(group_size=16, kl_coef=0.02)
    cfg = CritiqueGRPOConfig(grpo=inner, enabled=True)
    assert cfg.grpo.group_size == 16
    assert cfg.grpo.kl_coef == 0.02


def test_needs_critique_mask_below_threshold() -> None:
    rewards = torch.tensor([0.0, 0.4, 0.6, 1.0])
    mask = needs_critique_mask(rewards, threshold=0.5)
    expected = torch.tensor([True, True, False, False])
    assert torch.equal(mask, expected)


def test_needs_critique_mask_strict_inequality() -> None:
    """At-threshold rewards are NOT flagged (strict <)."""
    rewards = torch.tensor([0.5, 0.5])
    mask = needs_critique_mask(rewards, threshold=0.5)
    assert not mask.any().item()


def test_make_critique_prompt_default_template() -> None:
    text = make_critique_prompt(
        prompt="Compute 2+2",
        failed_completion="The answer is 5",
        reward=0.0,
    )
    assert "Compute 2+2" in text
    assert "The answer is 5" in text
    assert "0" in text  # reward
    assert text.endswith("Critique:")


def test_make_critique_prompt_custom_template() -> None:
    custom = "Q: {prompt}\nFail: {completion}\nReward: {reward}\n>>>"
    text = make_critique_prompt(
        prompt="x?", failed_completion="y", reward=0.3, template=custom,
    )
    assert text == "Q: x?\nFail: y\nReward: 0.3\n>>>"


def test_make_refinement_prompt_default_template() -> None:
    text = make_refinement_prompt(
        prompt="Compute 2+2",
        critique="You added wrong",
    )
    assert "Compute 2+2" in text
    assert "You added wrong" in text
    assert text.endswith("Try again, addressing the issue:")


def test_default_templates_have_expected_placeholders() -> None:
    """Sanity: the documented placeholders work in the default templates."""
    crit = DEFAULT_CRITIQUE_TEMPLATE.format(
        prompt="P", completion="C", reward=0.0,
    )
    assert "P" in crit and "C" in crit
    refn = DEFAULT_REFINEMENT_TEMPLATE.format(prompt="P", critique="C")
    assert "P" in refn and "C" in refn


def test_collect_critique_requests_picks_low_reward() -> None:
    rewards = torch.tensor([0.0, 0.8, 0.3])
    prompts = ["p0", "p1", "p2"]
    completions = ["c0", "c1", "c2"]
    out = collect_critique_requests(
        rewards=rewards, prompts=prompts, completions=completions, threshold=0.5,
    )
    indices = [r.rollout_idx for r in out]
    assert indices == [0, 2]
    assert isinstance(out[0], CritiqueRequest)
    assert out[0].original_prompt == "p0"
    assert out[0].failed_completion == "c0"
    assert out[0].reward == 0.0


def test_collect_critique_requests_validates_lengths() -> None:
    rewards = torch.tensor([0.0, 0.5])
    with pytest.raises(ValueError, match="prompts length"):
        collect_critique_requests(
            rewards=rewards, prompts=["p"], completions=["c0", "c1"], threshold=0.5,
        )
    with pytest.raises(ValueError, match="completions length"):
        collect_critique_requests(
            rewards=rewards, prompts=["p0", "p1"], completions=["c"], threshold=0.5,
        )


def test_collect_critique_requests_empty_when_all_above() -> None:
    rewards = torch.tensor([0.9, 1.0])
    out = collect_critique_requests(
        rewards=rewards, prompts=["a", "b"], completions=["x", "y"], threshold=0.5,
    )
    assert out == []
