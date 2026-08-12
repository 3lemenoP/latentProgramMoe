"""lpm/quaternion.py — reference kernels (latent-program-moe-spec.md §8).

Conventions, fixed project-wide:
- q = (w, x, y, z), scalar first. Unit norm enforced by normalize-in-forward.
- Sign invariance: q and -q are the same rotation. Any loss on quaternions MUST
  use d2_chord (or go through q_to_R). Never raw distances on q.
- Composition "apply a first, then b" = hamilton(b, a); R(b ⊗ a) = R(b) R(a).
- All quaternion math in fp32 (or fp64 in tests); rotation matrices are cast to
  the activation dtype inside apply_rot / apply_rot_head.

Formulas certified against the numpy twin (verify_math.py): 17/17 property
checks pass at ~1e-15 (orthogonality, homomorphism, sign invariance, slerp
endpoints/antipodal stability, block-diag equivalence, spectrum preservation,
automorphism, dead-frame cancellation).

Everything else in the repo imports from this module; no re-implementations.
"""
from dataclasses import dataclass

import torch

__all__ = [
    "q_normalize", "q_to_R", "hamilton", "d2_chord", "slerp",
    "Partition", "make_partition", "apply_rot", "apply_rot_head",
]


def q_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project raw (..., 4) params onto S^3. Safe at the origin."""
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def q_to_R(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternions (..., 4) -> rotation matrices (..., 3, 3). R(-q) = R(q)."""
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


def hamilton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a ⊗ b, shapes (..., 4). R(a ⊗ b) = R(a) R(b).

    Composition convention: skill a first, then skill b  =>  hamilton(b, a).
    """
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    w = aw * bw - (av * bv).sum(-1, keepdim=True)
    v = aw * bv + bw * av + torch.cross(av, bv, dim=-1)
    return torch.cat([w, v], dim=-1)


def d2_chord(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Sign-invariant squared distance in [0, 1]: 1 - <a, b>^2. Unit inputs."""
    return 1.0 - (a * b).sum(-1) ** 2


def slerp(a: torch.Tensor, b: torch.Tensor, t, eps: float = 1e-6) -> torch.Tensor:
    """Geodesic interpolation on S^3 with antipodal sign alignment.

    t: python float or broadcastable tensor. slerp(a, b, 0) = a; slerp(a, b, 1) = ±b.
    After sign alignment dot >= 0, so omega <= pi/2 and the only degenerate case
    is omega -> 0, handled by the normalized-lerp branch. sin(omega) is clamped
    before division so the untaken branch cannot poison gradients with NaN
    (torch.where backward gotcha).
    """
    dot = (a * b).sum(-1, keepdim=True)
    b = torch.where(dot < 0, -b, b)
    dot = dot.abs().clamp(max=1 - 1e-7)
    omega = torch.acos(dot)
    so = torch.sin(omega).clamp_min(1e-12)
    out = (torch.sin((1 - t) * omega) * a + torch.sin(t * omega) * b) / so
    lerp = q_normalize((1 - t) * a + t * b)
    return torch.where(omega < eps, lerp, out)


@dataclass(frozen=True)
class Partition:
    """Block partition of a feature dimension: n3 SO(3) blocks + rem identity dims."""
    dim: int
    n3: int
    rem: int

    @property
    def cut(self) -> int:
        return 3 * self.n3


def make_partition(dim: int) -> Partition:
    n3 = dim // 3
    return Partition(dim=dim, n3=n3, rem=dim - 3 * n3)


def apply_rot(x: torch.Tensor, R: torch.Tensor, part: Partition,
              inverse: bool = False) -> torch.Tensor:
    """Apply a block-diagonal rotation field to activations.

    x: (..., dim); R: (n3, 3, 3) built in fp32, cast here to x.dtype.
    inverse=True applies R^T (exact inverse, orthogonality). Remainder dims
    pass through untouched (spec §1.2). Broadcast-ready for v2 per-token fields:
    R with leading batch dims works with the same einsum pattern extended.
    """
    R = R.to(dtype=x.dtype)
    xb = x[..., :part.cut].reshape(*x.shape[:-1], part.n3, 3)
    Rm = R.transpose(-1, -2) if inverse else R
    yb = torch.einsum('...ni,nji->...nj', xb, Rm).reshape(*x.shape[:-1], part.cut)
    if part.rem == 0:
        return yb
    return torch.cat([yb, x[..., part.cut:]], dim=-1)


def apply_rot_head(x: torch.Tensor, R: torch.Tensor, part: Partition,
                   inverse: bool = False) -> torch.Tensor:
    """Per-head variant. x: (B, H, T, d_head); R: (H, n3, 3, 3)."""
    R = R.to(dtype=x.dtype)
    xb = x[..., :part.cut].reshape(*x.shape[:-1], part.n3, 3)
    Rm = R.transpose(-1, -2) if inverse else R
    yb = torch.einsum('bhtni,hnji->bhtnj', xb, Rm).reshape(*x.shape[:-1], part.cut)
    if part.rem == 0:
        return yb
    return torch.cat([yb, x[..., part.cut:]], dim=-1)
