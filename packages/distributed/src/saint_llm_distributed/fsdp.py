"""FSDP2 wrapper — shard a model via ``torch.distributed.fsdp.fully_shard``.

PyTorch's FSDP2 (the v2 ``fully_shard`` API, stable in torch 2.4+)
shards parameters across ranks while keeping the same Python module
tree. It's the modern replacement for the older ``FullyShardedData
Parallel`` wrapper class.

This module ships:

* :func:`ensure_local_gloo_pg` — init a single-rank gloo process
  group on localhost when no PG is initialized yet. Useful for local
  dev / smoke tests; production deploys init the PG via torchrun
  and skip this helper.
* :func:`init_default_mesh` — thin wrapper over
  ``torch.distributed.device_mesh.init_device_mesh`` defaulting to a
  1-D mesh of the current world size on the requested device.
* :func:`fsdp2_wrap` — walks ``model.named_modules()`` and applies
  ``fully_shard`` to every submodule satisfying the supplied policy.
  Default policy targets ``TransformerBlock``-style children — the
  natural sharding boundary in our SaintLLM stack.

Single-rank ``world_size=1`` runs use FSDP2 as a no-op shard (every
parameter lives on one rank). The wrapping API still runs and
forward/backward work, so single-rank tests verify the integration
shape without needing real distributed hardware.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch.distributed as dist
from torch import nn

ShardPolicy = Callable[[str, nn.Module], bool]


def ensure_local_gloo_pg(
    *,
    master_addr: str = "127.0.0.1",
    master_port: str = "29500",
    rank: int = 0,
    world_size: int = 1,
) -> bool:
    """Initialize a single-rank gloo process group when none is active.

    Returns ``True`` if this call initialized the PG, ``False`` if one
    was already running (idempotent — callers can invoke this from
    every test fixture without conditional guards).
    """
    if dist.is_initialized():
        return False
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(rank))
    dist.init_process_group(
        backend="gloo", world_size=world_size, rank=rank,
    )
    return True


def init_default_mesh(
    *,
    device: str = "cpu",
    world_size: int | None = None,
) -> dist.DeviceMesh:
    """Build a 1-D :class:`DeviceMesh` of the current world size.

    Args:
        device:     ``"cpu"`` or ``"cuda"``. Production typically uses
            ``"cuda"``; tests use ``"cpu"`` so they can run anywhere.
        world_size: override world size. Defaults to the current PG's
            world size (``dist.get_world_size()``).
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "process group not initialized; call "
            "ensure_local_gloo_pg() or torch.distributed.init_process_group "
            "first",
        )
    n = world_size if world_size is not None else dist.get_world_size()
    return dist.device_mesh.init_device_mesh(device, (n,))


def default_block_policy(name: str, module: nn.Module) -> bool:
    """Built-in policy: shard transformer-block-like submodules.

    Matches submodules whose class name ends with ``"Block"`` (the
    saint-llm convention for ``TransformerBlock``,
    ``MoEBlock``, etc.) AND that have at least one learnable
    parameter anywhere in their subtree. Excludes the root module
    itself (empty name) — the caller wraps that separately if they
    want via ``wrap_root=True``.
    """
    if not name:
        return False
    cls = type(module).__name__
    if not cls.endswith("Block"):
        return False
    return any(p.requires_grad for p in module.parameters())


def fsdp2_wrap(
    model: nn.Module,
    *,
    mesh: dist.DeviceMesh | None = None,
    policy: ShardPolicy | None = None,
    wrap_root: bool = True,
) -> nn.Module:
    """Apply ``fully_shard`` to selected submodules of ``model``.

    Walks ``model.named_modules()`` and shards each submodule for
    which ``policy(name, module)`` returns True. The same ``model``
    instance is returned (FSDP2 is composable / in-place).

    Args:
        model:     the module to shard. Must already live on the same
            device as ``mesh`` (caller's responsibility).
        mesh:      :class:`DeviceMesh` to shard along. If ``None``,
            calls :func:`init_default_mesh` with the current PG.
        policy:    ``(name, module) -> bool`` deciding which
            submodules get sharded. Defaults to
            :func:`default_block_policy`.
        wrap_root: also apply ``fully_shard`` to ``model`` itself.
            Default True — matches the recommended FSDP2 setup where
            the root call shards the leftover parameters that no
            child wrap covered (embeddings, output head, etc.).

    Returns:
        The same model instance, with FSDP2 wraps applied.
    """
    # Lazy import — fully_shard's stable API path is
    # ``torch.distributed.fsdp.fully_shard`` from torch 2.4+. We import
    # at call time so this module imports even on older torch builds.
    from torch.distributed.fsdp import fully_shard  # noqa: PLC0415

    eff_policy = policy if policy is not None else default_block_policy
    eff_mesh = mesh if mesh is not None else init_default_mesh()

    # Wrap children first (FSDP2 docs: shard leaves before the root).
    for name, sub in model.named_modules():
        if name == "" or not eff_policy(name, sub):
            continue
        fully_shard(sub, mesh=eff_mesh)

    if wrap_root:
        fully_shard(model, mesh=eff_mesh)
    return model
