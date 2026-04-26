"""Tests for Engram conditional memory (MEM-10)."""

from __future__ import annotations

import pytest
import torch
from saint_llm_inference import EngramConfig, EngramHook, EngramTable


def test_engram_config_defaults() -> None:
    cfg = EngramConfig()
    assert cfg.n_gram == 4
    assert cfg.n_buckets == 65_536
    assert cfg.embedding_dim == 128
    assert cfg.enabled is False


def test_engram_table_zero_init() -> None:
    cfg = EngramConfig(n_buckets=128, embedding_dim=8)
    table = EngramTable(cfg)
    out = table.lookup([1, 2, 3, 4])
    torch.testing.assert_close(out, torch.zeros(8))


def test_engram_table_update_then_lookup_roundtrip() -> None:
    cfg = EngramConfig(n_buckets=128, embedding_dim=4)
    table = EngramTable(cfg)
    delta = torch.tensor([1.0, 2.0, 3.0, 4.0])
    table.update([1, 2, 3, 4], delta)
    looked = table.lookup([1, 2, 3, 4])
    torch.testing.assert_close(looked, delta)


def test_engram_table_different_ngrams_hit_different_buckets() -> None:
    """Smoke: most distinct N-grams produce different bucket IDs."""
    cfg = EngramConfig(n_buckets=4096, embedding_dim=4)
    table = EngramTable(cfg)
    # Set bucket of [1,2,3,4] to all-ones.
    table.update([1, 2, 3, 4], torch.ones(4))
    # Different N-gram should still be zero (no collision in a 4096-bucket table).
    looked = table.lookup([5, 6, 7, 8])
    torch.testing.assert_close(looked, torch.zeros(4))


def test_engram_table_uses_only_trailing_ngram() -> None:
    """Long input is truncated to the trailing N-gram window."""
    cfg = EngramConfig(n_gram=2, n_buckets=128, embedding_dim=4)
    table = EngramTable(cfg)
    table.update([99, 1, 2], torch.ones(4))  # hashes [1, 2]
    # Looking up with the same trailing 2-gram should hit the same bucket.
    looked = table.lookup([42, 43, 1, 2])
    torch.testing.assert_close(looked, torch.ones(4))


def test_engram_hook_disabled_is_identity() -> None:
    cfg = EngramConfig(enabled=False)
    table = EngramTable(cfg)
    hook = EngramHook(hidden_dim=16, table=table)
    hidden = torch.randn(2, 4, 16)
    ids = torch.randint(0, 100, (2, 4))
    out = hook(hidden, ids)
    torch.testing.assert_close(out, hidden)


def test_engram_hook_zero_init_proj_is_identity_when_enabled() -> None:
    """Even with enabled=True, zero-init projection -> hook is identity."""
    cfg = EngramConfig(enabled=True, n_buckets=64, embedding_dim=4)
    table = EngramTable(cfg)
    # Inject something into the table so the embedding lookup is non-zero.
    table.update([1, 2, 3, 4], torch.ones(4))
    hook = EngramHook(hidden_dim=8, table=table)
    hidden = torch.randn(1, 4, 8)
    ids = torch.tensor([[1, 2, 3, 4]])
    out = hook(hidden, ids)
    # Zero-init proj -> delta is zero -> output equals input.
    torch.testing.assert_close(out, hidden)


def test_engram_hook_after_proj_perturbation_changes_hidden() -> None:
    cfg = EngramConfig(enabled=True, n_buckets=64, embedding_dim=4)
    table = EngramTable(cfg)
    table.update([1, 2, 3, 4], torch.ones(4))
    hook = EngramHook(hidden_dim=8, table=table)
    with torch.no_grad():
        hook.proj.weight.fill_(0.5)  # non-zero projection
    hidden = torch.zeros(1, 4, 8)
    ids = torch.tensor([[1, 2, 3, 4]])
    out = hook(hidden, ids)
    assert out.abs().sum().item() > 0


def test_engram_hook_validates_hidden_shape() -> None:
    cfg = EngramConfig(enabled=True)
    table = EngramTable(cfg)
    hook = EngramHook(hidden_dim=8, table=table)
    with pytest.raises(ValueError, match="hidden must be"):
        hook(torch.zeros(2, 4), torch.zeros(2, 4, dtype=torch.long))
