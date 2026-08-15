"""lpm/quaternion.py — reference kernels (latent-program-moe-spec.md §8).

Conventions, fixed project-wide:
- q = (w, x, y, z), scalar first. Unit norm enforced by normalize-in-forward.
- Sign invariance: q and -q are the same rotation. Any loss on quaternions MUST
  use d2_chord (or go through q_to_R). Never raw distances on q.
- Composition "apply a first, then b" = hamilton(b, a); R(b ⊗ a) = R(b) R(a).
- All quaternion math in fp32 (or fp64 in tests); rotation matrices are cast to
  the activation dtype inside apply_rot / apply_rot_head.

Formulas certified against the numpy twin (verify_math.py): 20/20 property
checks pass at ~1e-15 (T1–T3/T6/T8/T9 + T11 conjugate/commutator, plus
block-diag, automorphism, dead-frame).

Everything else in the repo imports from this module; no re-implementations.
"""
from dataclasses import dataclass

import torch

__all__ = [
    "q_normalize", "q_to_R", "hamilton", "d2_chord", "slerp",
    "q_conjugate", "q_angle2", "q_commutator", "d_geo", "q_pow",
    "q_log", "q_exp",
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


def q_conjugate(q: torch.Tensor) -> torch.Tensor:
    """(w, x, y, z) -> (w, -x, -y, -z). Inverse of a unit quaternion."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def q_angle2(q: torch.Tensor) -> torch.Tensor:
    """Sign-invariant activity s = 1 - w^2 = sin^2(θ/2) ∈ [0, 1]. Unit inputs
    after normalize-in-forward (raw fields are normalized here)."""
    w = q_normalize(q)[..., 0]
    return 1.0 - w * w


def q_commutator(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Group commutator a ⊗ b ⊗ a* ⊗ b* on unit quaternions."""
    a = q_normalize(a)
    b = q_normalize(b)
    return hamilton(hamilton(hamilton(a, b), q_conjugate(a)), q_conjugate(b))


def q_pow(q: torch.Tensor, lam, eps: float = 1e-8) -> torch.Tensor:
    """Rotation power q^lam on unit quaternions (..., 4). q_pow(q, 2) = ±q⊗q.

    Canonicalize FIRST: q and -q are the same rotation, but powering must take
    the shortest representative or results diverge (spec handoff §3.1).
    """
    q = torch.where(q[..., :1] < 0, -q, q)
    w, v = q[..., 0], q[..., 1:]
    vn = v.norm(dim=-1)
    theta = 2.0 * torch.atan2(vn, w)                    # in [0, pi]
    nhat = v / vn.clamp_min(eps).unsqueeze(-1)
    half = 0.5 * lam * theta
    out = torch.cat([torch.cos(half).unsqueeze(-1),
                     torch.sin(half).unsqueeze(-1) * nhat], dim=-1)
    ident = torch.zeros_like(q)
    ident[..., 0] = 1.0
    return torch.where((vn < eps).unsqueeze(-1), ident, out)


def q_log(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rotation-vector log map (..., 4) -> (..., 3): v = θ·n̂ with
    θ = 2·atan2(‖vec‖, w) AFTER w ≥ 0 canonicalization (sign-invariance:
    log(−q) = log(q); log(identity) = 0). Phase-4 spec §1, test T19."""
    q = torch.where(q[..., :1] < 0, -q, q)
    w, v = q[..., 0], q[..., 1:]
    vn = v.norm(dim=-1)
    theta = 2.0 * torch.atan2(vn, w)                    # in [0, pi]
    scale = theta / vn.clamp_min(eps)
    out = scale.unsqueeze(-1) * v
    return torch.where((vn < eps).unsqueeze(-1), torch.zeros_like(v), out)


def q_exp(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rotation-vector exp map (..., 3) -> unit quaternion (..., 4):
    q = (cos(‖v‖/2), sin(‖v‖/2)·v̂); identity at v = 0."""
    theta = v.norm(dim=-1)
    half = 0.5 * theta
    nhat = v / theta.clamp_min(eps).unsqueeze(-1)
    out = torch.cat([torch.cos(half).unsqueeze(-1),
                     torch.sin(half).unsqueeze(-1) * nhat], dim=-1)
    ident = torch.zeros(*v.shape[:-1], 4, dtype=v.dtype, device=v.device)
    ident[..., 0] = 1.0
    return torch.where((theta < eps).unsqueeze(-1), ident, out)


def d_geo(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reporting metric arccos(|⟨a, b⟩|) ∈ [0, π/2]. Losses stay chordal."""
    a = q_normalize(a)
    b = q_normalize(b)
    c = (a * b).sum(-1).abs().clamp(0.0, 1.0)
    return torch.acos(c)


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
