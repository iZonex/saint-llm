"""Throughput benchmark across quant-mode combinations.

Builds SaintLLM (cfg.tiny) under each of the configured quant modes, runs warmup
plus N timed forward+backward steps with random tokens, prints tokens/sec.

Usage:
    uv run python experiments/bench_quant_modes.py
    uv run python experiments/bench_quant_modes.py --steps 30 --seq-len 64

The "real fp8 GEMM" combo no-ops on devices that don't support it (Mac, older
GPUs); the script reports it as "n/a" rather than raising.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM
from saint_llm_kernels import is_fp8_gemm_supported


@dataclass(frozen=True)
class BenchCase:
    name: str
    linear_quant: str
    moe_use_grouped_gemm: bool
    fp8_use_real_gemm: bool

    def cfg(self, base: ModelConfig) -> ModelConfig:
        return base.model_copy(update={
            "linear_quant": self.linear_quant,
            "moe_use_grouped_gemm": self.moe_use_grouped_gemm,
            "fp8_use_real_gemm": self.fp8_use_real_gemm,
        })


def _all_cases() -> list[BenchCase]:
    cases = []
    for quant in ("bf16", "fp8", "fp4"):
        for grouped in (False, True):
            cases.append(BenchCase(
                name=f"{quant:<4}/{'grouped' if grouped else 'per-exp'}",
                linear_quant=quant,
                moe_use_grouped_gemm=grouped,
                fp8_use_real_gemm=False,
            ))
    # FP8 with real GEMM (separate from emulated fp8).
    cases.append(BenchCase(
        name="fp8 +real-gemm/per-exp",
        linear_quant="fp8",
        moe_use_grouped_gemm=False,
        fp8_use_real_gemm=True,
    ))
    cases.append(BenchCase(
        name="fp8 +real-gemm/grouped",
        linear_quant="fp8",
        moe_use_grouped_gemm=True,
        fp8_use_real_gemm=True,
    ))
    return cases


def _measure(
    case: BenchCase,
    *,
    base_cfg: ModelConfig,
    device: torch.device,
    seq_len: int,
    warmup: int,
    steps: int,
) -> tuple[float, float] | None:
    """Returns (tokens/sec, ms/step) or None if the case can't run on this device."""
    if case.fp8_use_real_gemm and not is_fp8_gemm_supported(device):
        return None

    cfg = case.cfg(base_cfg)
    torch.manual_seed(0)
    model = SaintLLM(cfg).to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

    token_ids = torch.randint(0, cfg.vocab_size, (1, seq_len), device=device)

    def step() -> None:
        out = model(token_ids)
        logits = out["logits"]
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, cfg.vocab_size),
            token_ids[:, 1:].reshape(-1),
        )
        optim.zero_grad()
        loss.backward()
        optim.step()

    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens_per_step = seq_len
    tokens_per_sec = (tokens_per_step * steps) / elapsed
    ms_per_step = (elapsed / steps) * 1000.0

    del model, optim
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return tokens_per_sec, ms_per_step


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    base_cfg = ModelConfig.tiny()
    cases = _all_cases()

    print(f"Device: {device} | tiny config | seq_len={args.seq_len} | "
          f"warmup={args.warmup} steps + {args.steps} timed steps")
    print()
    print(f"{'mode':<24}  {'tokens/sec':>12}  {'ms/step':>10}  {'rel':>6}")
    print("-" * 60)

    baseline: float | None = None
    for case in cases:
        result = _measure(
            case,
            base_cfg=base_cfg,
            device=device,
            seq_len=args.seq_len,
            warmup=args.warmup,
            steps=args.steps,
        )
        if result is None:
            print(f"{case.name:<24}  {'n/a':>12}  {'n/a':>10}  {'n/a':>6}")
            continue
        tps, ms = result
        if baseline is None:
            baseline = tps
            rel = 1.0
        else:
            rel = tps / baseline
        print(f"{case.name:<24}  {tps:>12.1f}  {ms:>10.2f}  {rel:>5.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
