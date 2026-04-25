"""Saint LLM optimizers.

Modules:
    muon          — Muon optimizer (hybrid Newton-Schulz, RMS rescaling, decoupled WD)
    newton_schulz — Two-stage hybrid Newton-Schulz orthogonalization
    param_groups  — Helper to split params (AdamW for embed/head/RMSNorm/mHC, Muon for rest)
"""

__version__ = "0.0.1"
