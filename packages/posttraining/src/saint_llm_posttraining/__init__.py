"""Saint LLM post-training pipeline.

Modules:
    specialist        — SFT + GRPO specialist training (per-domain)
    grpo              — Group Relative Policy Optimization
    grm               — Generative Reward Model (joint actor + judge)
    opd               — Full-vocabulary On-Policy Distillation (multi-teacher)
    teacher_offload   — ZeRO-like teacher param sharding to centralized storage
    reasoning_modes   — Non-think / High / Max reasoning effort modes
    tools.dsml        — DSML XML tool-call schema
    quick_instruction — Quick Instruction special tokens (action/title/query/...)
"""

__version__ = "0.0.1"
