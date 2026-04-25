"""Saint LLM optimizers.

Modules:
    muon          — Muon optimizer (hybrid Newton-Schulz, RMS rescaling, decoupled WD)
    newton_schulz — Two-stage hybrid Newton-Schulz orthogonalization
    param_groups  — Helper to split params (AdamW for embed/head/RMSNorm/mHC, Muon for rest)
"""

from saint_llm_optim.muon import Muon
from saint_llm_optim.newton_schulz import hybrid_newton_schulz
from saint_llm_optim.param_groups import split_for_muon_adamw

__version__ = "0.0.1"
__all__ = ["Muon", "hybrid_newton_schulz", "split_for_muon_adamw"]
