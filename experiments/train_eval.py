"""End-to-end demo: train SaintLLM on a corpus, measure PPL, generate samples.

Stitches every v0.1 surface into one runnable script:
* Trainer (train_step + evaluate + grad_clip + save/load)
* warmup_cosine_schedule for LR
* TextFileDataset / HuggingFaceTextDataset for streaming text → packed batches
* compute_perplexity for eval
* greedy_decode + top_k_sample for generation

Defaults: tiny config, synthetic random tokens. Pass --text-file for a local
corpus (plain or JSONL) or --hf-dataset path:split for an HF Hub dataset.

Usage:
    uv run python experiments/train_eval.py
    uv run python experiments/train_eval.py --steps 100 --eval-every 10
    uv run python experiments/train_eval.py --quant fp8 --grouped-moe --device cuda
    uv run python experiments/train_eval.py --text-file path/to/corpus.txt
    uv run python experiments/train_eval.py --hf-dataset roneneldan/TinyStories --hf-split train[:1%]
    uv run python experiments/train_eval.py --save-checkpoint /tmp/ckpt.pt
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM
from saint_llm_data import (
    CharTokenizer,
    HuggingFaceTextDataset,
    TextFileDataset,
)
from saint_llm_eval import compute_perplexity
from saint_llm_inference import greedy_decode, top_k_sample
from saint_llm_training import Trainer, warmup_cosine_schedule


def _build_cfg(args: argparse.Namespace) -> ModelConfig:
    update: dict[str, object] = {"linear_quant": args.quant}
    if args.grouped_moe:
        update["moe_use_grouped_gemm"] = True
    return ModelConfig.tiny().model_copy(update=update)


def _ce_loss(model: SaintLLM, batch: torch.Tensor) -> torch.Tensor:
    out = model(batch)
    logits = out["logits"]
    assert isinstance(logits, torch.Tensor)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        batch[:, 1:].reshape(-1),
    )


def _cycle_dataset(ds: object, *, device: torch.device, label: str) -> Iterator[torch.Tensor]:
    """Iterate forever over a single-shot dataset, raising if it's empty."""
    while True:
        emitted = False
        for batch in ds:  # type: ignore[attr-defined]
            emitted = True
            yield batch.tokens.to(device)
        if not emitted:
            raise RuntimeError(f"Corpus {label!r} produced no batches")


def _data_streams(
    args: argparse.Namespace,
    cfg: ModelConfig,
    device: torch.device,
) -> tuple[Iterator[torch.Tensor] | None, torch.Tensor, torch.Tensor]:
    """Return (train_stream | None, train_seq, eval_seq).

    Synthetic mode → train_stream is None and the trainer keeps reusing
    the same fixed random sequence (memorize task).
    """
    if args.text_file is not None:
        tok = CharTokenizer(unicode_max=cfg.vocab_size - 16)
        ds = TextFileDataset(
            args.text_file, tokenizer=tok,
            seq_len=args.seq_len, batch_size=1,
            jsonl=args.jsonl, drop_last=True,
        )
        stream = _cycle_dataset(ds, device=device, label=args.text_file)
        return stream, next(stream), next(stream)

    if args.hf_dataset is not None:
        tok = CharTokenizer(unicode_max=cfg.vocab_size - 16)
        ds = HuggingFaceTextDataset(
            args.hf_dataset,
            tokenizer=tok,
            seq_len=args.seq_len,
            batch_size=1,
            split=args.hf_split,
            text_field=args.hf_text_field,
            streaming=True,
            drop_last=True,
        )
        stream = _cycle_dataset(ds, device=device, label=args.hf_dataset)
        return stream, next(stream), next(stream)

    train = torch.randint(0, cfg.vocab_size, (1, args.seq_len), device=device)
    eval_ = torch.randint(0, cfg.vocab_size, (1, args.seq_len), device=device)
    return None, train, eval_


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--quant",
        choices=["bf16", "fp8", "fp4"],
        default="bf16",
    )
    parser.add_argument("--grouped-moe", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--jsonl", action="store_true")
    parser.add_argument("--hf-dataset", default=None,
                        help="HF Hub dataset path, e.g. roneneldan/TinyStories.")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-text-field", default="text")
    parser.add_argument("--save-checkpoint", default=None,
                        help="optional path to save the final checkpoint.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = _build_cfg(args)
    device = torch.device(args.device)

    train_stream, train_seq, eval_seq = _data_streams(args, cfg, device)
    prompt = train_seq[:, :4].clone()

    model = SaintLLM(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = warmup_cosine_schedule(
        optim,
        warmup_steps=min(args.warmup_steps, max(args.steps - 1, 1)),
        total_steps=max(args.steps, 2),
        min_lr_ratio=args.min_lr_ratio,
    )
    trainer = Trainer(
        model, optim,
        loss_fn=_ce_loss,
        lr_scheduler=sched,
        grad_clip_norm=args.grad_clip,
        device=device,
    )

    cfg_summary = (
        f"cfg.tiny | linear_quant={cfg.linear_quant} grouped_moe={cfg.moe_use_grouped_gemm}"
    )
    print(f"Device: {device} | {cfg_summary}")
    print(
        f"Steps: {args.steps} | warmup={args.warmup_steps} | "
        f"eval every {args.eval_every} | peak lr={args.lr} | grad_clip={args.grad_clip}",
    )
    print()
    print(f"{'step':>5}  {'lr':>9}  {'train_loss':>10}  {'train_ppl':>10}  {'eval_ppl':>10}")
    print("-" * 56)

    last_loss = float("nan")
    for step in range(args.steps + 1):
        if step > 0:
            batch = next(train_stream) if train_stream is not None else train_seq
            last_loss = trainer.train_step(batch)

        if step % args.eval_every == 0:
            train_ppl = compute_perplexity(trainer.model, train_seq)
            eval_ppl = compute_perplexity(trainer.model, eval_seq)
            current_lr = optim.param_groups[0]["lr"]
            print(
                f"{step:>5}  {current_lr:>9.6f}  {last_loss:>10.4f}  "
                f"{train_ppl:>10.2f}  {eval_ppl:>10.2f}",
            )

    # One greedy + one sampled generation so the user sees output (gibberish at this scale).
    trainer.model.eval()
    greedy_out = greedy_decode(trainer.model, prompt, max_new_tokens=8)
    g = torch.Generator(device=device).manual_seed(args.seed) if device.type != "mps" else None
    sample_out = top_k_sample(
        trainer.model, prompt, max_new_tokens=8, k=20, temperature=0.8, generator=g,
    )
    print()
    print("Prompt          :", prompt.tolist())
    print("Greedy decode   :", greedy_out.tolist())
    print("Top-k sample    :", sample_out.tolist())

    if args.save_checkpoint is not None:
        trainer.save(args.save_checkpoint, extra={"cfg": cfg.model_dump()})
        print(f"\nSaved checkpoint → {args.save_checkpoint} (step {trainer.step})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
