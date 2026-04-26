"""Tests for WandbLogger metrics_callback adapter.

Wandb is mocked via ``unittest.mock.MagicMock`` injected through the
``wandb_module`` constructor argument so the tests run offline without
network or filesystem side effects.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from saint_llm_training import WandbLogger


def _mock_wandb() -> MagicMock:
    mod = MagicMock()
    mod.init.return_value = MagicMock(name="run")
    return mod


def test_constructor_invokes_wandb_init_with_args() -> None:
    mod = _mock_wandb()
    logger = WandbLogger(
        project="saint-llm",
        run_name="v0.0-test",
        config={"vocab_size": 131072},
        mode="offline",
        tags=("ci", "smoke"),
        wandb_module=mod,
    )
    mod.init.assert_called_once()
    kwargs = mod.init.call_args.kwargs
    assert kwargs["project"] == "saint-llm"
    assert kwargs["name"] == "v0.0-test"
    assert kwargs["config"] == {"vocab_size": 131072}
    assert kwargs["mode"] == "offline"
    assert kwargs["tags"] == ["ci", "smoke"]
    assert kwargs["reinit"] is True
    logger.finish()


def test_call_logs_metrics_with_train_prefix() -> None:
    mod = _mock_wandb()
    logger = WandbLogger(project="x", wandb_module=mod, mode="disabled")
    logger(42, {"loss": 3.14, "lr": 1e-4})
    mod.log.assert_called_once_with(
        {"train/loss": 3.14, "train/lr": 1e-4},
        step=42,
    )


def test_call_preserves_user_namespacing() -> None:
    """Metrics with explicit namespace pass through unchanged."""
    mod = _mock_wandb()
    logger = WandbLogger(project="x", wandb_module=mod, mode="disabled")
    logger(7, {"eval/ppl": 12.3, "system/gpu_util": 0.85, "loss": 2.7})
    mod.log.assert_called_once_with(
        {"eval/ppl": 12.3, "system/gpu_util": 0.85, "train/loss": 2.7},
        step=7,
    )


def test_log_alias_calls_underlying() -> None:
    mod = _mock_wandb()
    logger = WandbLogger(project="x", wandb_module=mod, mode="disabled")
    logger.log(1, {"loss": 1.0})
    mod.log.assert_called_once_with({"train/loss": 1.0}, step=1)


def test_finish_calls_wandb_finish() -> None:
    mod = _mock_wandb()
    logger = WandbLogger(project="x", wandb_module=mod, mode="disabled")
    logger.finish()
    mod.finish.assert_called_once()


def test_finish_is_idempotent() -> None:
    mod = _mock_wandb()
    logger = WandbLogger(project="x", wandb_module=mod, mode="disabled")
    logger.finish()
    logger.finish()
    mod.finish.assert_called_once()


def test_call_after_finish_is_noop() -> None:
    mod = _mock_wandb()
    logger = WandbLogger(project="x", wandb_module=mod, mode="disabled")
    logger.finish()
    logger(1, {"loss": 1.0})
    mod.log.assert_not_called()


def test_context_manager_finishes_on_exit() -> None:
    mod = _mock_wandb()
    with WandbLogger(project="x", wandb_module=mod, mode="disabled") as logger:
        logger(0, {"loss": 0.5})
    mod.finish.assert_called_once()


def test_context_manager_finishes_on_exception() -> None:
    mod = _mock_wandb()

    class _Boom(Exception):
        pass

    try:
        with WandbLogger(project="x", wandb_module=mod, mode="disabled") as logger:
            logger(0, {"loss": 0.5})
            raise _Boom
    except _Boom:
        pass
    mod.finish.assert_called_once()


def test_default_config_is_empty_dict() -> None:
    mod = _mock_wandb()
    WandbLogger(project="x", wandb_module=mod, mode="disabled")
    assert mod.init.call_args.kwargs["config"] == {}


def test_no_tags_default() -> None:
    mod = _mock_wandb()
    WandbLogger(project="x", wandb_module=mod, mode="disabled")
    assert mod.init.call_args.kwargs["tags"] == []
