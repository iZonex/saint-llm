"""Smoke test: every workspace package imports cleanly."""

from __future__ import annotations

import importlib

PACKAGES = [
    "saint_llm_core",
    "saint_llm_kernels",
    "saint_llm_optim",
    "saint_llm_distributed",
    "saint_llm_training",
    "saint_llm_posttraining",
    "saint_llm_sandbox",
    "saint_llm_inference",
    "saint_llm_data",
    "saint_llm_eval",
]


def test_all_packages_importable() -> None:
    for pkg in PACKAGES:
        mod = importlib.import_module(pkg)
        assert hasattr(mod, "__version__"), f"{pkg} missing __version__"
        assert mod.__version__ == "0.0.1", f"{pkg} version mismatch"
