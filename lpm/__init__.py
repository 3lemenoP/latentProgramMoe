"""Latent-program MoE: SO(3) conjugation hypernetwork over a frozen transformer.

See docs/latent-program-moe-spec.md. Kernel conventions live in lpm.quaternion;
everything else imports from there — no re-implementations.
"""
from .config import LPMConfig, OptimConfig
from .field import FieldSpec, ProgramField, AbelianProgramField, AxisBank, SITE_NAMES
from .compose import compose, increment, invert_field, slerp_field, mean_field
from .model_wrap import LatentProgramModel

__all__ = [
    "LPMConfig", "OptimConfig",
    "FieldSpec", "ProgramField", "AbelianProgramField", "AxisBank", "SITE_NAMES",
    "compose", "increment", "invert_field", "slerp_field", "mean_field",
    "LatentProgramModel",
]
