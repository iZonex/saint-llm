"""CSA Lightning Indexer kernel tests."""

from __future__ import annotations

import pytest
import torch
from saint_llm_kernels.attention import (
    _causal_block_mask,
    lightning_indexer_scores,
    lightning_indexer_scores_reference,
    lightning_indexer_topk,
    lightning_indexer_topk_reference,
)


def _random_inputs(
    b: int = 2,
    t: int = 32,
    n_heads: int = 4,
    c: int = 16,
    n_blocks: int = 4,
    *,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, t, n_heads, c, generator=g)
    k = torch.randn(b, n_blocks, c, generator=g)
    w = torch.randn(b, t, n_heads, generator=g)
    return q, k, w


def test_scores_shape() -> None:
    q, k, w = _random_inputs()
    out = lightning_indexer_scores_reference(q, k, w)
    assert out.shape == (q.shape[0], q.shape[1], k.shape[1])


def test_scores_non_negative_pre_weighting() -> None:
    """ReLU is applied before weighting; final scores can be negative due to negative head weights."""
    q, k, _ = _random_inputs()
    w = torch.ones(q.shape[0], q.shape[1], q.shape[2])
    out = lightning_indexer_scores_reference(q, k, w)
    # With all-positive head weights, weighted sum of non-negative head scores is non-negative.
    assert (out >= 0.0).all()


def test_scores_visual_bias_shifts_output_uniformly() -> None:
    """visual_bias adds 0.5 * bias[s] to every (b, t) for that block s."""
    q, k, w = _random_inputs()
    vb = torch.zeros(q.shape[0], k.shape[1])
    vb[:, 1] = 1.0  # bias only block 1
    base = lightning_indexer_scores_reference(q, k, w)
    biased = lightning_indexer_scores_reference(q, k, w, visual_bias=vb)
    delta = biased - base
    # Delta should be 0.5 * w.sum(dim=-1) at block 1, 0 elsewhere.
    expected_block1 = 0.5 * w.sum(dim=-1)
    assert torch.allclose(delta[..., 1], expected_block1, atol=1.0e-6)
    assert torch.allclose(delta[..., 0], torch.zeros_like(delta[..., 0]), atol=1.0e-6)


def test_dispatch_scores_matches_reference_on_cpu() -> None:
    q, k, w = _random_inputs()
    expected = lightning_indexer_scores_reference(q, k, w)
    actual = lightning_indexer_scores(q, k, w)
    assert torch.equal(actual, expected)


def test_causal_mask_geometry() -> None:
    """Block s spans positions [s*m, s*m+m-1]; valid only when q is past the last."""
    mask = _causal_block_mask(t=8, n_blocks=4, block_size_m=2, device=torch.device("cpu"))
    assert mask.shape == (8, 4)
    # Position 0 sees no completed blocks.
    assert not mask[0].any()
    # Position 2: block 0 covers [0,1], complete by 2 → valid.
    assert mask[2, 0].item() and not mask[2, 1].item()
    # Position 7: blocks 0,1,2 ([0..5]) all complete; block 3 covers [6,7] not yet complete at 7.
    assert mask[7, 0].item() and mask[7, 1].item() and mask[7, 2].item()
    assert not mask[7, 3].item()


def test_topk_shape_and_dtype() -> None:
    q, k, w = _random_inputs(n_blocks=8)
    top_k = 3
    out = lightning_indexer_topk_reference(q, k, w, top_k=top_k, block_size_m=4)
    assert out.shape == (q.shape[0], q.shape[1], top_k)
    assert out.dtype == torch.long


def test_topk_clamps_to_n_blocks() -> None:
    """If top_k > n_blocks, returns n_blocks indices, not top_k."""
    q, k, w = _random_inputs(n_blocks=2)
    out = lightning_indexer_topk_reference(q, k, w, top_k=5, block_size_m=4)
    assert out.shape[-1] == 2


def test_topk_respects_causal_mask_when_enough_valid_blocks() -> None:
    """When valid_count ≥ top_k, every pick must be causally valid.

    Caveat: when valid_count < top_k, ``torch.topk`` over an all-(-inf)-tail
    returns arbitrary indices among the masked entries. That is a pre-existing
    behavior of the LightningIndexer (the kernel reproduces it bit-for-bit);
    the fix belongs in core (mask -1 on padded picks). Tracked for follow-up.
    """
    n_blocks = 8
    block_size_m = 4
    t = 64  # plenty of valid blocks for most positions
    top_k = 3
    q, k, w = _random_inputs(t=t, n_blocks=n_blocks)
    out = lightning_indexer_topk_reference(q, k, w, top_k=top_k, block_size_m=block_size_m)
    for t_pos in range(t):
        valid_count = sum(
            1 for s in range(n_blocks) if s * block_size_m + (block_size_m - 1) < t_pos
        )
        if valid_count < top_k:
            continue
        for s_idx in out[0, t_pos].tolist():
            block_end = s_idx * block_size_m + (block_size_m - 1)
            assert block_end < t_pos, (
                f"position {t_pos} picked block {s_idx} (end={block_end}); causal violation"
            )


def test_topk_zero_n_blocks_returns_empty() -> None:
    """No compressed blocks → empty (B, T, 0) result."""
    q = torch.randn(2, 8, 4, 16)
    k = torch.empty(2, 0, 16)
    w = torch.randn(2, 8, 4)
    out = lightning_indexer_topk_reference(q, k, w, top_k=4, block_size_m=4)
    assert out.shape == (2, 8, 0)


def test_topk_zero_top_k_returns_empty() -> None:
    q, k, w = _random_inputs()
    out = lightning_indexer_topk_reference(q, k, w, top_k=0, block_size_m=4)
    assert out.shape[-1] == 0


def test_dispatch_topk_matches_reference_on_cpu() -> None:
    q, k, w = _random_inputs(n_blocks=8)
    expected = lightning_indexer_topk_reference(q, k, w, top_k=3, block_size_m=4)
    actual = lightning_indexer_topk(q, k, w, top_k=3, block_size_m=4)
    assert torch.equal(actual, expected)


def test_scores_gradient_flows() -> None:
    q, k, w = _random_inputs()
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    w = w.requires_grad_(True)
    out = lightning_indexer_scores(q, k, w)
    out.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert w.grad is not None and torch.isfinite(w.grad).all()


def test_relu_zero_when_negative_dot_product() -> None:
    """If q · k is everywhere ≤ 0, head_scores = 0 → final score = 0."""
    q = torch.ones(1, 4, 2, 4)
    k = -torch.ones(1, 3, 4)  # produces q·k = -4 everywhere
    w = torch.ones(1, 4, 2)
    out = lightning_indexer_scores_reference(q, k, w)
    assert torch.equal(out, torch.zeros_like(out))


@pytest.mark.gpu
def test_dispatch_uses_compiled_path_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    q, k, w = _random_inputs(n_blocks=8)
    cuda = torch.device("cuda")
    q, k, w = q.to(cuda), k.to(cuda), w.to(cuda)
    expected_scores = lightning_indexer_scores_reference(q, k, w)
    actual_scores = lightning_indexer_scores(q, k, w)
    assert torch.allclose(actual_scores, expected_scores, atol=1.0e-5)

    expected_topk = lightning_indexer_topk_reference(q, k, w, top_k=3, block_size_m=4)
    actual_topk = lightning_indexer_topk(q, k, w, top_k=3, block_size_m=4)
    # topk with potential ties may differ in tie-broken indices; compare scores at picked indices instead.
    picked_expected = torch.gather(expected_scores, -1, expected_topk)
    picked_actual = torch.gather(expected_scores, -1, actual_topk)
    assert torch.allclose(picked_expected, picked_actual, atol=1.0e-4)
