"""Tokenizer wrappers: roundtrip, special tokens, batch, integration with packing."""

from __future__ import annotations

import pytest
from saint_llm_data import CharTokenizer, HFTokenizer, pack_sequences
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# ---------- CharTokenizer ----------


def test_char_tokenizer_roundtrip_ascii() -> None:
    tok = CharTokenizer()
    text = "hello world"
    ids = tok.encode(text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == text


def test_char_tokenizer_roundtrip_unicode() -> None:
    tok = CharTokenizer()
    text = "Привіт, 世界 🚀"
    assert tok.decode(tok.encode(text)) == text


def test_char_tokenizer_special_tokens_distinct() -> None:
    tok = CharTokenizer()
    assert tok.pad_token_id == 0
    assert tok.eos_token_id == 1
    assert tok.bos_token_id == 2
    # And distinct from any encoded character.
    encoded = tok.encode("a")
    assert encoded[0] >= 16


def test_char_tokenizer_vocab_size_consistent() -> None:
    tok = CharTokenizer(base_vocab=8, unicode_max=128)
    assert tok.vocab_size == 8 + 128


def test_char_tokenizer_decode_skips_control_slots() -> None:
    """IDs below base_vocab are control slots and silently skipped on decode."""
    tok = CharTokenizer()
    text = "ab"
    ids = tok.encode(text)
    with_eos = [*ids, tok.eos_token_id]
    assert tok.decode(with_eos) == text


def test_char_tokenizer_encode_batch_roundtrips() -> None:
    tok = CharTokenizer()
    texts = ["hi", "world", ""]
    batches = tok.encode_batch(texts)
    assert len(batches) == 3
    assert tok.decode(batches[0]) == "hi"
    assert tok.decode(batches[1]) == "world"
    assert tok.decode(batches[2]) == ""


def test_char_tokenizer_integrates_with_pack_sequences() -> None:
    """End-to-end: encode strings → pack into windows."""
    tok = CharTokenizer()
    docs = [tok.encode(t) for t in ["hello", "world", "foo"]]
    windows = list(pack_sequences(
        docs, seq_len=8, eos_token_id=tok.eos_token_id, drop_last=False,
    ))
    assert len(windows) >= 1
    flat = []
    for w in windows:
        flat.extend(w.tokens[0].tolist())
    # The eos token should appear three times (one per doc).
    assert flat.count(tok.eos_token_id) == 3


# ---------- HFTokenizer ----------


@pytest.fixture(scope="module")
def hf_tokenizer_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Build a tiny BBPE on the fly and save it — no network required."""
    backend = Tokenizer(BPE(unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=200,
        special_tokens=["<pad>", "<|endoftext|>", "<unk>"],
    )
    backend.train_from_iterator(
        ["hello world", "the quick brown fox", "saint llm test corpus"] * 30,
        trainer=trainer,
    )
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    backend.save(str(path))
    return str(path)


def test_hf_tokenizer_roundtrip(hf_tokenizer_path: str) -> None:
    tok = HFTokenizer.from_file(hf_tokenizer_path, eos_token="<|endoftext|>", pad_token="<pad>")
    text = "hello world"
    ids = tok.encode(text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    decoded = tok.decode(ids)
    # BBPE may add/strip whitespace differently; require token-level equivalence.
    assert "hello" in decoded
    assert "world" in decoded


def test_hf_tokenizer_special_token_ids(hf_tokenizer_path: str) -> None:
    tok = HFTokenizer.from_file(hf_tokenizer_path, eos_token="<|endoftext|>", pad_token="<pad>")
    assert tok.eos_token_id != tok.pad_token_id
    assert tok.vocab_size > 0


def test_hf_tokenizer_pad_defaults_to_eos_when_unset(hf_tokenizer_path: str) -> None:
    tok = HFTokenizer.from_file(hf_tokenizer_path, eos_token="<|endoftext|>")
    assert tok.pad_token_id == tok.eos_token_id


def test_hf_tokenizer_rejects_unknown_eos(hf_tokenizer_path: str) -> None:
    with pytest.raises(ValueError, match="not present"):
        HFTokenizer.from_file(hf_tokenizer_path, eos_token="<<absolutely-not-in-vocab>>")


def test_hf_tokenizer_encode_batch(hf_tokenizer_path: str) -> None:
    tok = HFTokenizer.from_file(hf_tokenizer_path, eos_token="<|endoftext|>")
    batches = tok.encode_batch(["hello", "world"])
    assert len(batches) == 2
    assert all(isinstance(b, list) for b in batches)


def test_hf_tokenizer_integrates_with_pack_sequences(hf_tokenizer_path: str) -> None:
    tok = HFTokenizer.from_file(hf_tokenizer_path, eos_token="<|endoftext|>")
    docs = tok.encode_batch(["hello world", "the quick brown fox"])
    windows = list(pack_sequences(
        docs, seq_len=16, eos_token_id=tok.eos_token_id, drop_last=False,
    ))
    assert len(windows) >= 1
