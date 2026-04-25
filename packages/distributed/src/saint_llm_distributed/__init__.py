"""Saint LLM distributed training infrastructure.

Modules:
    ep            — Fine-grained Expert Parallelism with wave scheduling (MegaMoE)
    zero          — Hybrid ZeRO bucket assignment for Muon (knapsack + per-expert)
    cp            — Two-stage Contextual Parallelism for compressed attention
    dualpipe      — DualPipe 1F1B with mHC overlap adjustments
    grad_compress — BF16 stochastic-rounding gradient compression for MoE all-reduce
"""

__version__ = "0.0.1"
