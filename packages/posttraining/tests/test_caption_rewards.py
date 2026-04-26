"""Tests for caption reward functions."""

from __future__ import annotations

import pytest
from saint_llm_posttraining.caption_rewards import (
    PerExampleCaptionRewardFn,
    composite_caption_reward,
    constant_reward,
    length_target_reward,
    unigram_overlap_reward,
)


def test_unigram_overlap_perfect_match_returns_one() -> None:
    r = unigram_overlap_reward("", "the quick brown fox", target="the quick brown fox")
    assert r == 1.0


def test_unigram_overlap_zero_when_disjoint() -> None:
    r = unigram_overlap_reward("", "hello world", target="goodbye galaxy")
    assert r == 0.0


def test_unigram_overlap_partial_f1() -> None:
    """3-token completion vs 4-token target with 2-token overlap."""
    r = unigram_overlap_reward("", "a cat sat", target="a cat on mat")
    # pred_set = {a, cat, sat}; ref_set = {a, cat, on, mat}; overlap = {a, cat}
    # precision = 2/3, recall = 2/4 = 0.5, f1 = 2*2/3*0.5/(2/3+0.5) = (2/3)/(7/6) = 4/7
    assert pytest.approx(r, abs=1e-5) == 4 / 7


def test_unigram_overlap_precision_mode() -> None:
    r = unigram_overlap_reward(
        "", "a cat sat", target="a cat on mat", mode="precision",
    )
    assert pytest.approx(r, abs=1e-5) == 2 / 3


def test_unigram_overlap_recall_mode() -> None:
    r = unigram_overlap_reward(
        "", "a cat sat", target="a cat on mat", mode="recall",
    )
    assert pytest.approx(r, abs=1e-5) == 0.5


def test_unigram_overlap_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        unigram_overlap_reward("", "a", target="a", mode="bogus")


def test_unigram_overlap_empty_completion_returns_zero() -> None:
    assert unigram_overlap_reward("", "", target="hi") == 0.0


def test_unigram_overlap_empty_target_returns_zero() -> None:
    assert unigram_overlap_reward("", "hi", target="") == 0.0


def test_unigram_overlap_case_insensitive() -> None:
    r1 = unigram_overlap_reward("", "Hello World", target="hello world")
    r2 = unigram_overlap_reward("", "hello world", target="HELLO WORLD")
    assert r1 == r2 == 1.0


def test_length_target_in_band_returns_one() -> None:
    r = length_target_reward(
        "", "one two three four five", target_min=4, target_max=6,
    )
    assert r == 1.0


def test_length_target_below_band_decays_linearly() -> None:
    """Length 2, band [5, 7], decay 5 -> distance=3, reward=1-3/5=0.4."""
    r = length_target_reward(
        "", "one two", target_min=5, target_max=7, decay_tokens=5,
    )
    assert pytest.approx(r, abs=1e-5) == 0.4


def test_length_target_above_band_decays_linearly() -> None:
    """Length 9, band [3, 5], decay 4 -> distance=4, reward=1-4/4=0."""
    r = length_target_reward(
        "", "a b c d e f g h i", target_min=3, target_max=5, decay_tokens=4,
    )
    assert r == 0.0


def test_length_target_far_outside_clamps_at_zero() -> None:
    r = length_target_reward(
        "", "a", target_min=20, target_max=30, decay_tokens=2,
    )
    assert r == 0.0


def test_length_target_zero_decay_is_hard_rect() -> None:
    """decay_tokens=0 -> 1.0 in band, 0 outside."""
    r_in = length_target_reward("", "a b c", target_min=2, target_max=4, decay_tokens=0)
    r_out = length_target_reward("", "a b c d e f", target_min=2, target_max=4, decay_tokens=0)
    assert r_in == 1.0
    assert r_out == 0.0


def test_length_target_rejects_inverted_band() -> None:
    with pytest.raises(ValueError, match="target_max"):
        length_target_reward("", "x", target_min=10, target_max=5)


def test_length_target_rejects_negative_decay() -> None:
    with pytest.raises(ValueError, match="decay_tokens"):
        length_target_reward(
            "", "x", target_min=1, target_max=2, decay_tokens=-1,
        )


def test_constant_reward_returns_value() -> None:
    assert constant_reward("any", "input", value=0.7) == 0.7


def test_composite_combines_components() -> None:
    """0.5*1.0 + 0.3*0.5 = 0.65."""
    r = composite_caption_reward([
        (0.5, lambda _p, _c: 1.0),
        (0.3, lambda _p, _c: 0.5),
    ])
    assert pytest.approx(r("p", "c"), abs=1e-5) == 0.65


def test_composite_empty_returns_zero() -> None:
    r = composite_caption_reward([])
    assert r("p", "c") == 0.0


def test_per_example_max_takes_best_target() -> None:
    rfn = PerExampleCaptionRewardFn(
        prompt_to_targets={"k": ["a cat", "a happy cat"]},
        scorer=lambda c, t: unigram_overlap_reward("", c, target=t),
        multi_target_reduce="max",
    )
    # Completion "a happy cat" exactly matches second target -> reward 1.0.
    assert rfn("k", "a happy cat") == 1.0


def test_per_example_mean_averages_targets() -> None:
    rfn = PerExampleCaptionRewardFn(
        prompt_to_targets={"k": ["a cat", "a happy cat"]},
        scorer=lambda c, t: unigram_overlap_reward("", c, target=t),
        multi_target_reduce="mean",
    )
    r_max = PerExampleCaptionRewardFn(
        prompt_to_targets={"k": ["a cat", "a happy cat"]},
        scorer=lambda c, t: unigram_overlap_reward("", c, target=t),
        multi_target_reduce="max",
    )("k", "a happy cat")
    r_mean = rfn("k", "a happy cat")
    # Mean is below max because first target only partially overlaps.
    assert r_mean < r_max


def test_per_example_missing_prompt_returns_default() -> None:
    rfn = PerExampleCaptionRewardFn(
        prompt_to_targets={"known": ["x"]},
        scorer=lambda c, t: 1.0,
        missing_prompt_value=-1.0,
    )
    assert rfn("unknown", "anything") == -1.0


def test_per_example_rejects_unknown_reduce() -> None:
    with pytest.raises(ValueError, match="multi_target_reduce"):
        PerExampleCaptionRewardFn(
            prompt_to_targets={},
            scorer=lambda c, t: 0.0,
            multi_target_reduce="median",
        )


def test_per_example_satisfies_reward_protocol() -> None:
    """Returns Python float, takes (prompt, completion)."""
    rfn = PerExampleCaptionRewardFn(
        prompt_to_targets={"p": ["target text"]},
        scorer=lambda c, t: unigram_overlap_reward("", c, target=t),
    )
    val = rfn("p", "target text")
    assert isinstance(val, float)
    assert val == 1.0
