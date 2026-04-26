"""Tests for v0.0 data pipeline (D0.0.2)."""

from __future__ import annotations

from saint_llm_data import CharTokenizer
from saint_llm_data.pipeline import (
    FingerprintDedup,
    LengthQualityFilter,
    MinHashDedup,
    Pipeline,
    QualityFilter,
    TokenizeStage,
    TranslationeseFilter,
    URLBlocklistFilter,
    minhash_signature,
)


def _doc(text: str, language: str = "en", slice_name: str = "hplt-en", **meta: object) -> dict:
    return {"text": text, "language": language, "slice": slice_name, "meta": dict(meta)}


# ----- Pipeline / Stage composition ------------------------------------------


def test_pipeline_passes_through_when_no_drops() -> None:
    p = Pipeline(stages=[LengthQualityFilter(min_chars=1)])
    docs = [_doc("hello"), _doc("world")]
    out = list(p.run(docs))
    assert len(out) == 2


def test_pipeline_stops_at_first_drop() -> None:
    """A document dropped by stage[0] should not reach stage[1]."""
    seen: list[str] = []

    def _trace_stage(doc: dict) -> dict | None:
        seen.append(doc["text"])
        return doc

    _trace_stage.name = "trace"  # type: ignore[attr-defined]

    p = Pipeline(
        stages=[
            LengthQualityFilter(min_chars=10),  # drops "short"
            _trace_stage,  # type: ignore[list-item]
        ],
    )
    list(p.run([_doc("short"), _doc("a long enough document")]))
    # Only the surviving doc reaches the trace stage.
    assert seen == ["a long enough document"]


def test_pipeline_stats_run_reports_drops() -> None:
    p = Pipeline(stages=[LengthQualityFilter(min_chars=10)])
    passing, drops = p.stats_run([_doc("hi"), _doc("a long enough doc")])
    assert len(passing) == 1
    assert drops["length_quality_filter"] == 1


# ----- LengthQualityFilter ----------------------------------------------------


def test_length_filter_drops_short() -> None:
    f = LengthQualityFilter(min_chars=20)
    assert f(_doc("short")) is None
    assert f(_doc("definitely longer than twenty chars")) is not None


def test_length_filter_per_language_threshold() -> None:
    f = LengthQualityFilter(min_chars=100, min_chars_per_language={"uk": 5})
    # Default 100 chars too high for "hello world" — but UK threshold is 5.
    assert f(_doc("привіт", language="uk")) is not None
    assert f(_doc("привіт", language="en")) is None


# ----- QualityFilter (custom classifier) -------------------------------------


def test_quality_filter_calls_classifier_and_drops_below_threshold() -> None:
    def fake_classifier(text: str, *, language: str | None = None) -> float:
        # Score = 1 if text contains "good", else 0.
        return 1.0 if "good" in text else 0.0

    f = QualityFilter(classifier=fake_classifier, threshold=0.5)
    assert f(_doc("this is a good text")) is not None
    assert f(_doc("this is a poor text")) is None


def test_quality_filter_apply_to_slices_skips_others() -> None:
    def always_drop(text: str, *, language: str | None = None) -> float:
        return 0.0

    f = QualityFilter(
        classifier=always_drop,
        threshold=0.5,
        apply_to_slices=("hplt-en",),
    )
    # Doc in another slice passes through (filter not applied).
    assert f(_doc("anything", slice_name="kobza-uk")) is not None
    # Doc in target slice is dropped.
    assert f(_doc("anything", slice_name="hplt-en")) is None


# ----- FingerprintDedup -------------------------------------------------------


def test_fingerprint_dedup_drops_exact_duplicates() -> None:
    d = FingerprintDedup()
    assert d(_doc("abc")) is not None
    assert d(_doc("abc")) is None  # second time -> drop
    assert d(_doc("abcd")) is not None  # different doc passes


def test_fingerprint_dedup_len_tracks_seen_count() -> None:
    d = FingerprintDedup()
    d(_doc("a"))
    d(_doc("b"))
    d(_doc("a"))  # duplicate
    assert len(d) == 2


def test_fingerprint_dedup_reset_clears_state() -> None:
    d = FingerprintDedup()
    d(_doc("a"))
    d.reset()
    assert d(_doc("a")) is not None


# ----- MinHashDedup -----------------------------------------------------------


def test_minhash_dedup_signature_match() -> None:
    """Identical text produces identical signature → drop second copy."""
    d = MinHashDedup(n_perm=32)
    assert d(_doc("the quick brown fox jumps")) is not None
    assert d(_doc("the quick brown fox jumps")) is None


def test_minhash_dedup_different_text_different_sig() -> None:
    d = MinHashDedup(n_perm=32)
    assert d(_doc("the quick brown fox jumps over the lazy dog")) is not None
    assert d(_doc("a totally different sentence with no shared shingles")) is not None


def test_minhash_signature_is_deterministic_across_calls() -> None:
    sig1 = minhash_signature("hello world", n_perm=8)
    sig2 = minhash_signature("hello world", n_perm=8)
    assert sig1 == sig2


# ----- URLBlocklistFilter -----------------------------------------------------


def test_url_blocklist_drops_blocked_domain() -> None:
    f = URLBlocklistFilter(
        blocked_domains=("badtranslate.com",),
        apply_to_languages=("uk",),
    )
    assert f(_doc("text", language="uk", url="https://badtranslate.com/foo")) is None


def test_url_blocklist_passes_clean_domain() -> None:
    f = URLBlocklistFilter(
        blocked_domains=("badtranslate.com",),
        apply_to_languages=("uk",),
    )
    assert f(_doc("text", language="uk", url="https://kobza.com.ua/foo")) is not None


def test_url_blocklist_only_for_target_languages() -> None:
    f = URLBlocklistFilter(
        blocked_domains=("badtranslate.com",),
        apply_to_languages=("uk",),
    )
    # English doc with blocked URL passes (language not in scope).
    assert f(_doc("text", language="en", url="https://badtranslate.com/foo")) is not None


def test_url_blocklist_no_url_meta_passes() -> None:
    f = URLBlocklistFilter(blocked_domains=("badtranslate.com",), apply_to_languages=("uk",))
    assert f(_doc("text", language="uk")) is not None


# ----- TranslationeseFilter ---------------------------------------------------


def test_translationese_filter_drops_above_threshold() -> None:
    def fake_detector(text: str) -> float:
        return 0.9 if "translated" in text else 0.1

    f = TranslationeseFilter(
        detector=fake_detector,
        threshold=0.5,
        apply_to_languages=("uk",),
    )
    assert f(_doc("this looks translated", language="uk")) is None
    assert f(_doc("looks native", language="uk")) is not None


def test_translationese_filter_skips_non_target_language() -> None:
    def fake_detector(text: str) -> float:
        return 1.0  # always drop if applied

    f = TranslationeseFilter(
        detector=fake_detector,
        threshold=0.5,
        apply_to_languages=("uk",),
    )
    # English doc bypasses the filter even with always-drop detector.
    assert f(_doc("anything", language="en")) is not None


# ----- TokenizeStage ----------------------------------------------------------


def test_tokenize_stage_attaches_token_ids() -> None:
    tok = CharTokenizer(base_vocab=16, unicode_max=0x4000)
    stage = TokenizeStage(tokenizer=tok)
    doc = stage(_doc("hello"))
    assert doc is not None
    assert "token_ids" in doc
    assert isinstance(doc["token_ids"], list)
    assert len(doc["token_ids"]) == 5  # 5 chars


def test_tokenize_stage_drops_empty() -> None:
    tok = CharTokenizer(base_vocab=16, unicode_max=0x4000)
    stage = TokenizeStage(tokenizer=tok)
    assert stage(_doc("")) is None


# ----- Composed pipeline integration -----------------------------------------


def test_full_pipeline_filter_dedup_tokenize() -> None:
    """End-to-end smoke: filter + dedup + tokenize stages compose."""
    tok = CharTokenizer(base_vocab=16, unicode_max=0x4000)
    p = Pipeline(
        stages=[
            LengthQualityFilter(min_chars=5),
            FingerprintDedup(),
            TokenizeStage(tokenizer=tok),
        ],
    )
    docs = [
        _doc("hi"),  # too short -> drop at length filter
        _doc("hello world"),  # passes
        _doc("hello world"),  # duplicate -> drop at dedup
        _doc("foo bar baz"),  # passes
    ]
    out = list(p.run(docs))
    assert len(out) == 2
    assert all("token_ids" in d for d in out)
    texts = [d["text"] for d in out]
    assert texts == ["hello world", "foo bar baz"]
