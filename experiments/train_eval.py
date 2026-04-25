"""End-to-end demo: train SaintLLM on a fixed sequence, measure PPL, generate.

Stitches together everything we've built so far — model build, optimizer step,
perplexity eval, greedy + top-k sampling, and (with --text-file) tokenized
text streaming via TextFileDataset — into one runnable script. Smoke test for
the whole pipeline and a starting template for a real run.

Usage:
    uv run python experiments/train_eval.py
    uv run python experiments/train_eval.py --steps 100 --eval-every 10
    uv run python experiments/train_eval.py --quant fp8 --grouped-moe --device cuda
    uv run python experiments/train_eval.py --text-file path/to/corpus.txt

Tiny config so it finishes in seconds on Mac/CPU. Real trainings live in
training/ and consume datasets via the data pipeline.
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F
from saint_llm_core.config import ModelConfig
from saint_llm_core.model import SaintLLM
from saint_llm_data import CharTokenizer, TextFileDataset
from saint_llm_eval import compute_perplexity
from saint_llm_inference import greedy_decode, top_k_sample


def _build_cfg(args: argparse.Namespace) -> ModelConfig:
    update: dict[str, object] = {"linear_quant": args.quant}
    if args.grouped_moe:
        update["moe_use_grouped_gemm"] = True
    return ModelConfig.tiny().model_copy(update=update)


def _train_step(
    model: SaintLLM,
    token_ids: torch.Tensor,
    optim: torch.optim.Optimizer,
    cfg: ModelConfig,
) -> float:
    out = model(token_ids)
    logits = out["logits"]
    assert isinstance(logits, torch.Tensor)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size),
        token_ids[:, 1:].reshape(-1),
    )
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3.0e-3)
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
    parser.add_argument(
        "--text-file",
        default=None,
        help="optional path to a plain-text or JSONL corpus; if set, "
             "training cycles through tokenized batches instead of using a "
             "fixed random sequence.",
    )
    parser.add_argument("--jsonl", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = _build_cfg(args)
    device = torch.device(args.device)

    if args.text_file is not None:
        # CharTokenizer keeps the demo dependency-light. Swap to HFTokenizer
        # for production runs (vocab needs to match cfg.vocab_size).
        tok = CharTokenizer(unicode_max=cfg.vocab_size - 16)
        ds = TextFileDataset(
            args.text_file, tokenizer=tok,
            seq_len=args.seq_len, batch_size=1,
            jsonl=args.jsonl, drop_last=True,
        )

        def _train_batches() -> torch.Tensor:
            while True:
                emitted = False
                for batch in ds:
                    emitted = True
                    yield batch.tokens.to(device)
                if not emitted:
                    raise RuntimeError(f"Corpus {args.text_file} produced no batches")

        text_stream = _train_batches()
        train_seq = next(text_stream)
        eval_seq = next(text_stream)
        prompt = train_seq[:, :4].clone()
    else:
        text_stream = None
        train_seq = torch.randint(0, cfg.vocab_size, (1, args.seq_len), device=device)
        eval_seq = torch.randint(0, cfg.vocab_size, (1, args.seq_len), device=device)
        prompt = train_seq[:, :4].clone()

    model = SaintLLM(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    cfg_summary = (
        f"cfg.tiny | linear_quant={cfg.linear_quant} grouped_moe={cfg.moe_use_grouped_gemm}"
    )
    print(f"Device: {device} | {cfg_summary}")
    print(f"Steps: {args.steps} | eval every {args.eval_every} | lr={args.lr}")
    print()
    print(f"{'step':>5}  {'train_loss':>10}  {'train_ppl':>10}  {'eval_ppl':>10}")
    print("-" * 44)

    for step in range(args.steps + 1):
        if step > 0:
            model.train()
            if text_stream is not None:
                train_seq = next(text_stream)
            train_loss = _train_step(model, train_seq, optim, cfg)
        else:
            train_loss = float("nan")

        if step % args.eval_every == 0:
            model.eval()
            train_ppl = compute_perplexity(model, train_seq)
            eval_ppl = compute_perplexity(model, eval_seq)
            print(
                f"{step:>5}  {train_loss:>10.4f}  {train_ppl:>10.2f}  {eval_ppl:>10.2f}",
            )

    # One greedy generation and one sampled generation at the end so the user
    # sees the model produces *something* — they'll be gibberish at this scale.
    model.eval()
    greedy_out = greedy_decode(model, prompt, max_new_tokens=8)
    g = torch.Generator(device=device).manual_seed(args.seed) if device.type != "mps" else None
    sample_out = top_k_sample(model, prompt, max_new_tokens=8, k=20, temperature=0.8, generator=g)
    print()
    print("Prompt          :", prompt.tolist())
    print("Greedy decode   :", greedy_out.tolist())
    print("Top-k sample    :", sample_out.tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
