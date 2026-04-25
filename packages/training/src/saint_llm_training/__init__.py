"""Saint LLM training framework.

Modules:
    checkpoint            — save/load model + optimizer + step state

Planned:
    loop                  — Top-level training loop
    checkpointing.fx      — Tensor-level activation checkpointing via TorchFX
    quant.fp4_qat         — FP4 (MXFP4) quantization-aware training
    routing.anticipatory  — Anticipatory Routing with loss-spike detection
    stability.swiglu      — SwiGLU clamping
    schedule              — Batch size + sequence length + LR schedules
    losses.mtp            — Multi-Token Prediction loss
    losses.balance        — Aux-loss-free balancing + sequence-wise balance loss
"""

from saint_llm_training.checkpoint import load_checkpoint, save_checkpoint
from saint_llm_training.schedule import warmup_cosine_schedule
from saint_llm_training.trainer import Trainer

__version__ = "0.0.1"

__all__ = ["Trainer", "load_checkpoint", "save_checkpoint", "warmup_cosine_schedule"]
