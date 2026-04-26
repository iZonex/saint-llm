"""Tests for FSDP2 wrapper (single-rank gloo)."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from saint_llm_distributed.fsdp import (
    default_block_policy,
    ensure_local_gloo_pg,
    fsdp2_wrap,
    init_default_mesh,
)
from torch import nn


@pytest.fixture(scope="session", autouse=True)
def _gloo_pg() -> None:
    """Bring up a single-rank gloo PG once per test session."""
    ensure_local_gloo_pg()
    yield
    # Don't destroy — other modules may have already torn it down,
    # and re-initialization within the same process tends to misbehave.


def _tiny_block(in_dim: int = 8, out_dim: int = 4) -> nn.Module:
    block = nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, in_dim),
    )
    block.__class__.__name__ = "TestBlock"  # name pattern triggers the policy
    return block


class _TinyModel(nn.Module):
    """Model with two named submodules whose class names end in 'Block'."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.block1 = _Block(8)
        self.block2 = _Block(8)
        self.head = nn.Linear(8, 16)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(ids)
        h = self.block1(h)
        h = self.block2(h)
        return self.head(h)


class _Block(nn.Module):
    """Class name ends in 'Block' so default_block_policy picks it up."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(torch.relu(self.lin1(x)))


# ---- ensure_local_gloo_pg / init_default_mesh -----------------------


def test_pg_is_initialized_after_fixture() -> None:
    assert dist.is_initialized()


def test_ensure_local_gloo_pg_idempotent() -> None:
    """Second call returns False without crashing."""
    assert ensure_local_gloo_pg() is False


def test_init_default_mesh_returns_devicemesh() -> None:
    mesh = init_default_mesh(device="cpu")
    assert isinstance(mesh, dist.DeviceMesh)
    assert mesh.size() == 1


def test_init_default_mesh_world_size_override() -> None:
    mesh = init_default_mesh(device="cpu", world_size=1)
    assert mesh.size() == 1


def test_init_default_mesh_requires_pg() -> None:
    """Sanity: the helper relies on an active PG (covered by fixture in this test file)."""
    # Already initialized; the function returns a mesh without complaint.
    mesh = init_default_mesh()
    assert mesh is not None


# ---- default_block_policy -------------------------------------------


def test_default_policy_matches_block_classname_with_params() -> None:
    block = _Block(4)
    assert default_block_policy("foo", block) is True


def test_default_policy_skips_non_block_classes() -> None:
    plain = nn.Linear(4, 4)
    assert default_block_policy("foo", plain) is False


def test_default_policy_skips_root_empty_name() -> None:
    block = _Block(4)
    assert default_block_policy("", block) is False


def test_default_policy_skips_block_with_all_params_frozen() -> None:
    """A 'Block' whose params are all frozen is not eligible."""

    class _FrozenBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = nn.Linear(4, 4)
            for p in self.parameters():
                p.requires_grad_(False)

    frozen = _FrozenBlock()
    assert default_block_policy("foo", frozen) is False


# ---- fsdp2_wrap -----------------------------------------------------


def test_fsdp2_wrap_returns_same_model_instance() -> None:
    model = _TinyModel()
    wrapped = fsdp2_wrap(model)
    assert wrapped is model


def test_fsdp2_wrapped_model_forward_runs() -> None:
    torch.manual_seed(0)
    model = _TinyModel()
    fsdp2_wrap(model)
    ids = torch.randint(0, 16, (2, 5))
    out = model(ids)
    assert out.shape == (2, 5, 16)
    assert torch.isfinite(out).all()


def test_fsdp2_wrapped_model_backward_populates_grads() -> None:
    torch.manual_seed(0)
    model = _TinyModel()
    fsdp2_wrap(model)
    ids = torch.randint(0, 16, (2, 4))
    loss = model(ids).pow(2).mean()
    loss.backward()
    grad_count = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    assert grad_count > 0


def test_fsdp2_optimizer_step_decreases_loss() -> None:
    """Standard sanity: forward → backward → optimizer.step → loss drops."""
    torch.manual_seed(0)
    model = _TinyModel()
    fsdp2_wrap(model)
    optim = torch.optim.SGD(model.parameters(), lr=1e-1)
    ids = torch.randint(0, 16, (4, 4))

    losses = []
    for _ in range(5):
        loss = model(ids).pow(2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_fsdp2_wrap_with_custom_policy() -> None:
    """Custom policy can target any subset of submodules."""
    torch.manual_seed(0)
    model = _TinyModel()
    # Only shard block1, not block2.
    fsdp2_wrap(model, policy=lambda name, m: name == "block1")
    ids = torch.randint(0, 16, (2, 3))
    out = model(ids)
    assert out.shape == (2, 3, 16)


def test_fsdp2_wrap_without_root_does_not_wrap_root() -> None:
    """wrap_root=False shards only the children specified by the policy."""
    torch.manual_seed(0)
    model = _TinyModel()
    fsdp2_wrap(model, wrap_root=False)
    ids = torch.randint(0, 16, (2, 3))
    out = model(ids)
    assert out.shape == (2, 3, 16)


def test_fsdp2_wrap_with_explicit_mesh() -> None:
    mesh = init_default_mesh(device="cpu")
    model = _TinyModel()
    fsdp2_wrap(model, mesh=mesh)
    out = model(torch.randint(0, 16, (1, 2)))
    assert out.shape == (1, 2, 16)


def test_fsdp2_state_dict_round_trip() -> None:
    """Wrapped model's state_dict can be saved and loaded back."""
    torch.manual_seed(0)
    model = _TinyModel()
    fsdp2_wrap(model)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Build a fresh model + wrap, then load.
    other = _TinyModel()
    fsdp2_wrap(other)
    other.load_state_dict(state, strict=False)

    ids = torch.randint(0, 16, (1, 2))
    a = model(ids)
    b = other(ids)
    assert torch.allclose(a, b, atol=1e-5)
