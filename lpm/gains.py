"""lpm/gains.py — abelian scalar-per-block gains (spec §4, behind enable_gains).

g = exp(rho), one scalar per 3-block, applied IMMEDIATELY AFTER the
corresponding forward rotation as x <- x * repeat_interleave(g, 3). Remainder
dims pass through untouched, mirroring apply_rot. Gains live in the trivial
irrep: they commute with all rotations, compose multiplicatively (rho adds),
interpolate linearly in rho, and preserve the identity program at rho=0 (T10).

rho storage/caching lives on ProgramField (lpm/field.py); this module is the
application kernel only.
"""
import torch

from .quaternion import Partition


def apply_gain(x: torch.Tensor, g: torch.Tensor, part: Partition) -> torch.Tensor:
    """x: (..., dim); g: (n3,) fp32, cast to x.dtype here."""
    ge = torch.repeat_interleave(g.to(dtype=x.dtype), 3, dim=-1)
    xb = x[..., :part.cut] * ge
    if part.rem == 0:
        return xb
    return torch.cat([xb, x[..., part.cut:]], dim=-1)


def apply_gain_head(x: torch.Tensor, g: torch.Tensor, part: Partition) -> torch.Tensor:
    """Per-head variant. x: (B, H, T, d_head); g: (H, n3)."""
    ge = torch.repeat_interleave(g.to(dtype=x.dtype), 3, dim=-1)  # (H, cut)
    xb = x[..., :part.cut] * ge[:, None, :]                       # broadcast over B, T
    if part.rem == 0:
        return xb
    return torch.cat([xb, x[..., part.cut:]], dim=-1)
