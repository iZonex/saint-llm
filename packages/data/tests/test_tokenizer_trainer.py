"""train_bbpe smoke tests: trains, saves, reloads, encodes round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
from saint_llm_data import HFTokenizer, train_bbpe
from saint_llm_data.tokenizer_trainer import DEFAULT_SPECIAL_TOKENS


def _tiny_corpus() -> list[str]:
    base = [
        "hello world",
        "the quick brown fox jumps over the lazy dog",
        "saint llm test corpus for tokenizer training",
        "lorem ipsum dolor sit amet consectetur adipiscing elit",
        "Привіт, як справи? Це тест українською.",
    ]
    # Repeat so the BPE merger has enough frequency signal.
    return base * 30


def test_train_bbpe_writes_tokenizer_json(tmp_path: Path) -> None:
    out = tmp_path / "tokenizer.json"
    tok = train_bbpe(_tiny_corpus(), out, vocab_size=200, show_progress=False)
    assert out.exists()
    assert isinstance(tok, HFTokenizer)


def test_train_bbpe_vocab_size_grows_with_target(tmp_path: Path) -> None:
    """Larger vocab_size should yield a larger learned vocab (modulo
    ByteLevel's 256 initial-alphabet floor and special-token overhead)."""
    small = train_bbpe(_tiny_corpus(), tmp_path / "small.json", vocab_size=300, show_progress=False)
    big = train_bbpe(_tiny_corpus(), tmp_path / "big.json", vocab_size=600, show_progress=False)
    assert big.vocab_size > small.vocab_size


def test_train_bbpe_special_tokens_present(tmp_path: Path) -> None:
    out = tmp_path / "tokenizer.json"
    train_bbpe(_tiny_corpus(), out, vocab_size=300, show_progress=False)
    reloaded = HFTokenizer.from_file(out, eos_token="<|endoftext|>", pad_token="<pad>")
    assert reloaded.eos_token_id != reloaded.pad_token_id


def test_train_bbpe_roundtrip_preserves_ascii(tmp_path: Path) -> None:
    out = tmp_path / "tokenizer.json"
    tok = train_bbpe(_tiny_corpus(), out, vocab_size=300, show_progress=False)
    text = "hello world"
    decoded = tok.decode(tok.encode(text))
    assert "hello" in decoded
    assert "world" in decoded


def test_train_bbpe_handles_unicode(tmp_path: Path) -> None:
    out = tmp_path / "tokenizer.json"
    tok = train_bbpe(_tiny_corpus(), out, vocab_size=300, show_progress=False)
    text = "Привіт"
    ids = tok.encode(text)
    assert len(ids) > 0
    # ByteLevel BBPE roundtrips bytes losslessly.
    assert tok.decode(ids).strip() == text


def test_train_bbpe_default_special_tokens_complete() -> None:
    """Sanity: AUGMENTATIONS-mandated special slots are in the default tuple."""
    expected = {
        "<pad>", "<|endoftext|>", "<|bos|>", "<unk>",
        "<|image_pad|>", "<|audio_start|>", "<|audio_end|>",
        "<|video_start|>", "<|video_end|>", "<|think_start|>", "<|think_end|>",
    }
    assert expected.issubset(set(DEFAULT_SPECIAL_TOKENS))


def test_train_bbpe_dropout_disabled_after_save(tmp_path: Path) -> None:
    """Saved tokenizer must encode deterministically (no BPE dropout at inference)."""
    out = tmp_path / "tokenizer.json"
    tok = train_bbpe(_tiny_corpus(), out, vocab_size=300, dropout=0.5, show_progress=False)
    text = "the quick brown fox"
    a = tok.encode(text)
    b = tok.encode(text)
    assert a == b


def test_train_bbpe_creates_parent_directories(tmp_path: Path) -> None:
    out = tmp_path / "deeply" / "nested" / "tokenizer.json"
    train_bbpe(_tiny_corpus(), out, vocab_size=200, show_progress=False)
    assert out.exists()


def test_train_bbpe_custom_eos_pad(tmp_path: Path) -> None:
    out = tmp_path / "tokenizer.json"
    tok = train_bbpe(
        _tiny_corpus(), out, vocab_size=300,
        special_tokens=["<MyPad>", "<MyEos>", "<unk>"],
        eos_token="<MyEos>", pad_token="<MyPad>",
        show_progress=False,
    )
    assert tok.eos_token_id != tok.pad_token_id


def test_train_bbpe_rejects_unknown_eos(tmp_path: Path) -> None:
    out = tmp_path / "tokenizer.json"
    with pytest.raises(ValueError, match="not present"):
        train_bbpe(
            _tiny_corpus(), out, vocab_size=300,
            special_tokens=["<pad>", "<unk>"],
            eos_token="<not-in-corpus>",
            show_progress=False,
        )
