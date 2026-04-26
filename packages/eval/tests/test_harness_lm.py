"""Tests for SaintLLMHarnessLM (lm-eval-compatible adapter)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM
from saint_llm_data import CharTokenizer
from saint_llm_eval import SaintLLMHarnessLM


@dataclass
class _StubReq:
    """Duck-typed lm-eval Instance: has only ``args`` tuple."""

    args: tuple[object, ...]


def _tiny_model() -> SaintLLM:
    cfg = ModelConfig.tiny()  # type: ignore[attr-defined]
    return SaintLLM(cfg)


def _harness(model: SaintLLM | None = None, max_length: int = 64) -> SaintLLMHarnessLM:
    model = model if model is not None else _tiny_model()
    tok = CharTokenizer(base_vocab=16, unicode_max=cfg_vocab_max(model))
    return SaintLLMHarnessLM(model=model, tokenizer=tok, max_length=max_length)


def cfg_vocab_max(model: SaintLLM) -> int:
    """Pick unicode_max so CharTokenizer ids stay within model vocab."""
    return min(model.cfg.vocab_size - 16, 0x4000)


def test_loglikelihood_returns_per_request_tuple() -> None:
    h = _harness()
    requests = [
        _StubReq(args=("hello", " world")),
        _StubReq(args=("foo", " bar")),
    ]
    out = h.loglikelihood(requests)
    assert len(out) == 2
    for logp, is_greedy in out:
        assert isinstance(logp, float)
        assert isinstance(is_greedy, bool)
        # log-prob of a continuation must be ≤ 0 (sum of log-probs).
        assert logp <= 0.0
        assert math.isfinite(logp)


def test_loglikelihood_empty_continuation_returns_zero_greedy_true() -> None:
    h = _harness()
    out = h.loglikelihood([_StubReq(args=("hello", ""))])
    logp, is_greedy = out[0]
    assert logp == 0.0
    assert is_greedy is True


def test_loglikelihood_rolling_returns_total_logp_per_text() -> None:
    h = _harness()
    out = h.loglikelihood_rolling(
        [_StubReq(args=("a longer sentence to score",)), _StubReq(args=("xy",))],
    )
    assert len(out) == 2
    for logp in out:
        assert isinstance(logp, float)
        assert logp <= 0.0
        assert math.isfinite(logp)


def test_loglikelihood_rolling_short_text_returns_zero() -> None:
    """Single-token text has no conditioning context to score."""
    h = _harness()
    out = h.loglikelihood_rolling([_StubReq(args=("a",))])
    assert out[0] == 0.0


def test_generate_until_produces_string() -> None:
    h = _harness()
    out = h.generate_until(
        [_StubReq(args=("prompt: ", {"max_gen_toks": 4, "until": []}))],
    )
    assert len(out) == 1
    assert isinstance(out[0], str)


def test_generate_until_truncates_at_stop_string() -> None:
    """If a stop string appears in the generated text, output is truncated."""
    h = _harness()
    # Force the generator to produce a known string via mock: tokenizer
    # is char-level, so generate produces character-level decisions. We
    # can't deterministically force an "until" hit on a random model,
    # but we can check the truncation logic works when ``until`` triggers.
    raw = h.generate_until(
        [_StubReq(args=("hello", {"max_gen_toks": 32, "until": []}))],
    )[0]
    # Now verify that a manually-set stop string is honored.
    # We do this by finding any character that exists in raw and
    # asserting the truncated result stops at its first occurrence.
    if not raw:
        pytest.skip("Generator produced no content for the truncation check")
    stop_char = raw[len(raw) // 2]  # middle character
    truncated = h.generate_until(
        [_StubReq(args=("hello", {"max_gen_toks": 32, "until": [stop_char]}))],
    )[0]
    assert stop_char not in truncated or raw.find(stop_char) >= len(truncated)


def test_generate_until_invalid_kwargs_treated_as_empty() -> None:
    h = _harness()
    out = h.generate_until(
        [_StubReq(args=("ctx", "not a dict"))],  # gen_kwargs not dict
    )
    assert isinstance(out[0], str)


def test_eot_token_id_returns_tokenizer_eos() -> None:
    h = _harness()
    assert h.eot_token_id == h.tokenizer.eos_token_id


def test_loglikelihood_restores_train_mode_after_call() -> None:
    h = _harness()
    h.model.train()
    h.loglikelihood([_StubReq(args=("a", "b"))])
    assert h.model.training is True


def test_loglikelihood_restores_eval_mode_after_call() -> None:
    h = _harness()
    h.model.eval()
    h.loglikelihood([_StubReq(args=("a", "b"))])
    assert h.model.training is False


def test_max_length_truncates_long_context() -> None:
    """Right-truncate so most-recent context wins; no error on overflow."""
    h = _harness(max_length=16)
    long_ctx = "a" * 100
    out = h.loglikelihood([_StubReq(args=(long_ctx, "b"))])
    assert len(out) == 1
    logp, _ = out[0]
    assert math.isfinite(logp)


def test_loglikelihood_is_greedy_is_consistent() -> None:
    """If we feed back the model's own greedy continuation, is_greedy=True."""
    h = _harness()
    # Generate a continuation greedily, then score it back.
    gen = h.generate_until(
        [_StubReq(args=("seed text", {"max_gen_toks": 5, "until": []}))],
    )[0]
    if not gen:
        pytest.skip("Generator produced no continuation")
    out = h.loglikelihood([_StubReq(args=("seed text", gen))])
    logp, is_greedy = out[0]
    assert math.isfinite(logp)
    # Greedy continuation of greedy decode should be greedy when scored.
    assert is_greedy
