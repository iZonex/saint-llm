"""Saint LLM training framework.

Modules:
    loop                  — Top-level training loop
    checkpointing.fx      — Tensor-level activation checkpointing via TorchFX
    quant.fp4_qat         — FP4 (MXFP4) quantization-aware training
    routing.anticipatory  — Anticipatory Routing with loss-spike detection
    stability.swiglu      — SwiGLU clamping
    schedule              — Batch size + sequence length + LR schedules
    losses.mtp            — Multi-Token Prediction loss
    losses.balance        — Aux-loss-free balancing + sequence-wise balance loss
"""

__version__ = "0.0.1"
