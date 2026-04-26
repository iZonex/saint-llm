"""Tests for GRM (Generative Reward Model)."""

from __future__ import annotations

import pytest
import torch
from saint_llm_posttraining.grm import (
    GRMConfig,
    GRMRewardFn,
    binary_grm_config,
    judge_completion,
    likert_grm_config,
)
from torch import nn


class _MockJudge(nn.Module):
    """Judge that returns programmable last-position logits.

    Specifying ``last_logits`` of shape ``(V,)`` makes every batch row
    return the same last-position logits — enough for unit tests.
    """

    def __init__(self, vocab: int, last_logits: torch.Tensor | None = None) -> None:
        super().__init__()
        self.vocab = vocab
        self._last = last_logits if last_logits is not None else torch.zeros(vocab)
        self.embed = nn.Embedding(vocab, 4)  # gives the module .parameters()

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t = tokens.shape
        logits = torch.zeros(b, t, self.vocab)
        logits[:, -1, :] = self._last
        return {"logits": logits}


def test_grm_config_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        GRMConfig(
            prompt_template="x",
            score_token_ids=(1, 2, 3),
            score_values=(1.0, 0.0),
        )


def test_grm_config_rejects_empty_score_tokens() -> None:
    with pytest.raises(ValueError, match="at least one"):
        GRMConfig(
            prompt_template="x", score_token_ids=(), score_values=(),
        )


def test_grm_config_rejects_zero_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        GRMConfig(
            prompt_template="x", score_token_ids=(1,),
            score_values=(1.0,), temperature=0.0,
        )


def test_judge_completion_binary_high_reward_when_good_dominates() -> None:
    """Logit on 'good' >> logit on 'bad' -> reward ~ 1.0."""
    last = torch.full((8,), -1e6)
    last[3] = 100.0  # good token
    last[5] = 0.0    # bad token
    judge = _MockJudge(vocab=8, last_logits=last)
    cfg = binary_grm_config(good_token_id=3, bad_token_id=5)
    tokens = torch.tensor([[1, 2, 3]])
    reward = judge_completion(judge, tokens, cfg)
    assert reward.shape == (1,)
    assert reward.item() > 0.99


def test_judge_completion_binary_low_reward_when_bad_dominates() -> None:
    last = torch.full((8,), -1e6)
    last[3] = 0.0
    last[5] = 100.0
    judge = _MockJudge(vocab=8, last_logits=last)
    cfg = binary_grm_config(good_token_id=3, bad_token_id=5)
    tokens = torch.tensor([[1, 2, 3]])
    reward = judge_completion(judge, tokens, cfg)
    assert reward.item() < 0.01


def test_judge_completion_binary_uniform_gives_half() -> None:
    """Equal logits on good/bad -> reward = 0.5 (expected of {1.0, 0.0})."""
    last = torch.zeros(8)
    judge = _MockJudge(vocab=8, last_logits=last)
    cfg = binary_grm_config(good_token_id=3, bad_token_id=5)
    tokens = torch.tensor([[1, 2, 3]])
    reward = judge_completion(judge, tokens, cfg)
    assert pytest.approx(reward.item(), abs=1e-5) == 0.5


def test_judge_completion_likert_expected_value() -> None:
    """Likert score = expected value over the score distribution."""
    last = torch.full((8,), -1e6)
    last[1] = 100.0  # token "5" of the Likert scale -> reward 1.0
    judge = _MockJudge(vocab=8, last_logits=last)
    cfg = likert_grm_config(
        score_token_ids=(2, 3, 4, 0, 1),  # last entry 1 -> max score
        prompt_template="x",
    )
    tokens = torch.tensor([[7, 6, 5]])
    reward = judge_completion(judge, tokens, cfg)
    # All probability mass on the last score token -> reward 1.0.
    assert pytest.approx(reward.item(), abs=1e-3) == 1.0


def test_judge_completion_temperature_softens_distribution() -> None:
    """Higher temperature pulls reward toward 0.5 (expected of {1, 0})."""
    last = torch.full((8,), -1e6)
    last[3] = 5.0
    last[5] = 0.0
    judge = _MockJudge(vocab=8, last_logits=last)
    cold = binary_grm_config(good_token_id=3, bad_token_id=5, temperature=0.5)
    hot = binary_grm_config(good_token_id=3, bad_token_id=5, temperature=10.0)
    tokens = torch.tensor([[1, 2, 3]])
    r_cold = judge_completion(judge, tokens, cold).item()
    r_hot = judge_completion(judge, tokens, hot).item()
    # Cold is closer to 1.0; hot is closer to 0.5.
    assert r_cold > r_hot
    assert abs(r_hot - 0.5) < abs(r_cold - 0.5)


def test_judge_completion_handles_batch() -> None:
    last = torch.full((8,), -1e6)
    last[3] = 50.0
    last[5] = 0.0
    judge = _MockJudge(vocab=8, last_logits=last)
    cfg = binary_grm_config(good_token_id=3, bad_token_id=5)
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 0, 1]])
    rewards = judge_completion(judge, tokens, cfg)
    assert rewards.shape == (3,)
    # All rows see the same last-position logits -> equal rewards.
    assert torch.allclose(rewards, rewards[0].repeat(3), atol=1e-5)


def test_judge_completion_rejects_non_2d_tokens() -> None:
    judge = _MockJudge(vocab=8)
    cfg = binary_grm_config(good_token_id=3, bad_token_id=5)
    with pytest.raises(ValueError, match=r"\(B, T\)"):
        judge_completion(judge, torch.tensor([1, 2, 3]), cfg)


def test_grm_reward_fn_satisfies_reward_protocol() -> None:
    """GRMRewardFn(prompt, completion) returns a Python float."""
    last = torch.full((16,), -1e6)
    last[3] = 100.0
    last[5] = 0.0
    judge = _MockJudge(vocab=16, last_logits=last)
    cfg = binary_grm_config(good_token_id=3, bad_token_id=5)

    def encode(text: str) -> list[int]:
        # Tokenize as raw byte values mod vocab — deterministic stub.
        return [b % 16 for b in text.encode("utf-8")]

    fn = GRMRewardFn(judge, encode, cfg)
    reward = fn("what is 2+2?", "4")
    assert isinstance(reward, float)
    assert reward > 0.99


def test_grm_reward_fn_restores_judge_train_mode() -> None:
    """After calling the reward fn, judge.training must match what it was."""
    judge = _MockJudge(vocab=16)
    judge.train()
    cfg = binary_grm_config(good_token_id=3, bad_token_id=5)
    fn = GRMRewardFn(judge, lambda s: [b % 16 for b in s.encode("utf-8")], cfg)
    fn("a", "b")
    assert judge.training is True


def test_grm_reward_fn_empty_encoding_raises() -> None:
    judge = _MockJudge(vocab=16)
    cfg = binary_grm_config(good_token_id=3, bad_token_id=5)
    fn = GRMRewardFn(judge, lambda _: [], cfg)
    with pytest.raises(ValueError, match="empty"):
        fn("p", "c")


def test_likert_helper_evenly_spaces_score_values() -> None:
    cfg = likert_grm_config(
        score_token_ids=(1, 2, 3, 4, 5),
        prompt_template="x",
    )
    assert cfg.score_values == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_likert_helper_rejects_singleton() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        likert_grm_config(score_token_ids=(1,), prompt_template="x")
