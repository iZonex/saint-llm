"""Production-shape training script for cfg.small_flash on hpomen-class GPU.

Wires every v0.1 surface together:
* ModelConfig.small_flash (~185M params)
* SaintLLM with activation_checkpointing
* HuggingFaceTextDataset (or local --text-file) + DataLoader (num_workers)
* Trainer with bf16 mixed precision, grad_clip, grad_accum, warmup_cosine LR,
  NaN/spike skipping, metrics callback (stdout by default; wire wandb in
  the callback if desired)
* CheckpointRotator with N-rotation
* MTP-aware cross-entropy loss
* periodic perplexity eval on a held-out batch
* greedy + top-k sample at the end so we can eyeball generation quality

This is the script the v0.1 milestone run launches. Real corpus path:

    1) train a BBPE tokenizer with ``experiments/train_tokenizer.py``
    2) point this script at the resulting ``tokenizer.json`` and at a HF
       dataset (FineWeb-Edu sample, RedPajama, etc.)

Quick smoke run (synthetic random tokens, no tokenizer needed):

    uv run python experiments/run_small_flash.py --steps 8 --eval-every 4
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import torch
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM
from saint_llm_data import (
    CharTokenizer,
    HFTokenizer,
    HuggingFaceTextDataset,
    TextFileDataset,
)
from saint_llm_eval import compute_perplexity
from saint_llm_inference import greedy_decode, top_k_sample
from saint_llm_training import (
    CheckpointRotator,
    Trainer,
    make_loss_fn,
    warmup_cosine_schedule,
)
from torch.utils.data import DataLoader


def _build_cfg(args: argparse.Namespace) -> ModelConfig:
    update: dict[str, object] = {
        "linear_quant": args.quant,
        "moe_use_grouped_gemm": args.grouped_moe,
        "fp8_use_real_gemm": args.fp8_use_real_gemm,
        "activation_checkpointing": args.activation_checkpointing,
    }
    return ModelConfig.small_flash().model_copy(update=update)


def _build_tokenizer(args: argparse.Namespace, cfg: ModelConfig) -> object:
    if args.tokenizer_file is not None:
        return HFTokenizer.from_file(args.tokenizer_file)
    return CharTokenizer(unicode_max=cfg.vocab_size - 16)


def _build_data_iter(
    args: argparse.Namespace,
    tokenizer: object,
    device: torch.device,
) -> Iterator[torch.Tensor]:
    """Cycle through a dataset forever, yielding (1, seq_len) token batches."""
    if args.hf_dataset is not None:
        ds = HuggingFaceTextDataset(
            args.hf_dataset,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            batch_size=1,
            split=args.hf_split,
            text_field=args.hf_text_field,
            streaming=True,
            drop_last=True,
        )
        label = args.hf_dataset
    elif args.text_file is not None:
        ds = TextFileDataset(
            args.text_file,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            batch_size=1,
            jsonl=args.jsonl,
            drop_last=True,
        )
        label = args.text_file
    else:
        # Synthetic fallback so the script runs as a smoke test even without data.
        cfg_vocab = tokenizer.vocab_size  # type: ignore[attr-defined]
        while True:
            yield torch.randint(0, cfg_vocab, (1, args.seq_len), device=device)

    loader = DataLoader(ds, batch_size=None, num_workers=args.num_workers)
    while True:
        emitted = False
        for batch in loader:
            emitted = True
            yield batch.tokens.to(device)
        if not emitted:
            raise RuntimeError(f"Corpus {label!r} yielded no batches")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Run cadence
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=256)
    # Optimization
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="effective batch = 1 (per-step) * grad_accum")
    parser.add_argument("--mixed-precision", choices=["none", "bf16", "fp16"],
                        default="bf16")
    parser.add_argument("--loss-spike-factor", type=float, default=8.0)
    # Quant flags
    parser.add_argument("--quant", choices=["bf16", "fp8", "fp4"], default="bf16")
    parser.add_argument("--grouped-moe", action="store_true")
    parser.add_argument("--fp8-use-real-gemm", action="store_true")
    parser.add_argument("--no-activation-checkpointing", dest="activation_checkpointing",
                        action="store_false", default=True)
    # Data
    parser.add_argument("--hf-dataset", default=None)
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-text-field", default="text")
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--jsonl", action="store_true")
    parser.add_argument("--tokenizer-file", default=None,
                        help="path to a tokenizer.json from train_tokenizer.py")
    parser.add_argument("--num-workers", type=int, default=2)
    # IO
    parser.add_argument("--output-dir", type=Path, default=Path("./run"))
    parser.add_argument("--keep-last-n", type=int, default=3)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    cfg = _build_cfg(args)
    device = torch.device(args.device)
    tokenizer = _build_tokenizer(args, cfg)
    print(f"Device: {device} | tokenizer: {type(tokenizer).__name__} (vocab={tokenizer.vocab_size})")  # type: ignore[attr-defined]
    print(
        f"cfg.small_flash | linear_quant={cfg.linear_quant} grouped={cfg.moe_use_grouped_gemm} "
        f"fp8_real_gemm={cfg.fp8_use_real_gemm} act_ckpt={cfg.activation_checkpointing}",
    )
    print(
        f"steps={args.steps} grad_accum={args.grad_accum} (eff bs={args.grad_accum}) "
        f"seq_len={args.seq_len} mp={args.mixed_precision} lr={args.lr} "
        f"warmup={args.warmup_steps} clip={args.grad_clip}",
    )

    model = SaintLLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.1f}M")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = warmup_cosine_schedule(
        optim,
        warmup_steps=min(args.warmup_steps, max(args.steps - 1, 1)),
        total_steps=max(args.steps, 2),
        min_lr_ratio=args.min_lr_ratio,
    )

    losses_log: list[tuple[int, float]] = []

    def _on_metrics(step: int, m: dict[str, float]) -> None:
        # Default callback: stdout. Replace with wandb.log(m, step=step) etc.
        skipped = m.get("skipped", 0.0) > 0
        msg = (
            f"[{step:>5}/{args.steps}] loss={m['loss']:.4f} lr={m['lr']:.2e}"
        )
        if "grad_norm" in m:
            msg += f" gnorm={m['grad_norm']:.3f}"
        if skipped:
            reason = "nan" if "skip_reason_nan" in m else "spike"
            msg += f" SKIPPED({reason})"
        print(msg)
        losses_log.append((step, m["loss"]))

    trainer = Trainer(
        model, optim,
        loss_fn=make_loss_fn(cfg.mtp),
        lr_scheduler=sched,
        grad_clip_norm=args.grad_clip,
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision,
        loss_spike_factor=args.loss_spike_factor,
        metrics_callback=_on_metrics,
        device=device,
    )
    rotator = CheckpointRotator(
        args.output_dir / "ckpt.pt", keep_last_n=args.keep_last_n,
    )

    data_iter = _build_data_iter(args, tokenizer, device)
    eval_batch = next(data_iter)
    prompt = eval_batch[:, :8].clone()

    print()
    t0 = time.perf_counter()
    next_save = args.save_every
    while trainer.step < args.steps:
        batch = next(data_iter)
        trainer.train_step(batch)

        if trainer.step >= next_save and trainer.step > 0:
            path = rotator.save(trainer)
            print(f"  ↳ checkpoint saved: {path.name}")
            next_save = trainer.step + args.save_every

        if trainer.step % args.eval_every == 0 and trainer.step > 0:
            ppl = compute_perplexity(trainer.model, eval_batch)
            elapsed = time.perf_counter() - t0
            tok_per_s = (trainer.step * args.grad_accum * args.seq_len) / max(elapsed, 1.0e-3)
            print(f"  ↳ eval_ppl={ppl:.2f}  throughput={tok_per_s:.0f} tok/s  elapsed={elapsed:.1f}s")

    print()
    final_path = rotator.save(trainer)
    print(f"Final checkpoint → {final_path}")

    trainer.model.eval()
    greedy_out = greedy_decode(trainer.model, prompt, max_new_tokens=16)
    g = torch.Generator(device=device).manual_seed(args.seed) if device.type != "mps" else None
    sample_out = top_k_sample(
        trainer.model, prompt, max_new_tokens=16, k=20, temperature=0.8, generator=g,
    )
    print()
    print("Prompt        :", prompt.tolist())
    print("Greedy decode :", greedy_out.tolist())
    print("Top-k sample  :", sample_out.tolist())
    if hasattr(tokenizer, "decode"):
        try:
            print("Greedy text   :", tokenizer.decode(greedy_out[0].tolist())[:200])  # type: ignore[attr-defined]
        except Exception as e:
            print(f"  (decode failed: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
