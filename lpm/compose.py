"""lpm/compose.py — composition and merging on the program manifold (spec §6).

Composition convention, fixed project-wide (spec §1.3): applying skill a FIRST,
then skill b, composes per site as q = hamilton(q_b, q_a), i.e. R = R_b @ R_a.
Non-abelian: compose(b, a) != compose(a, b) in general (T9). Gains (trivial
irrep) compose additively in rho.

Merging is geodesic: slerp_field uses the sign-aligned slerp of spec §1.1
sitewise; mean_field is the chordal mean (sign-align to fields[0], weighted
sum, normalize). Everything here is sign-invariant.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import torch

from .field import FieldSpec, ProgramField, SITE_NAMES, SiteKey
from .quaternion import hamilton, q_conjugate, q_normalize, q_pow, slerp

AlphaLike = Union[float, torch.Tensor, Dict]


def _check_specs(*fields: ProgramField) -> FieldSpec:
    spec = fields[0].spec
    for f in fields[1:]:
        if f.spec != spec:
            raise ValueError(f"FieldSpec mismatch: {f.spec} != {spec}")
    return spec


def _merged_rhos(fields: Sequence[ProgramField], combine) -> Optional[Dict[SiteKey, torch.Tensor]]:
    """If any input has gains, treat missing gains as rho=0 and combine.
    Zeros are created on the gains-carrying field's device (a bare
    torch.zeros would land on CPU and crash the combine on CUDA fields)."""
    if not any(f.has_gains for f in fields):
        return None
    spec = fields[0].spec
    ref = next(f for f in fields if f.has_gains)
    dev = ref.rho(*spec.site_keys()[0]).device
    out = {}
    for k in spec.site_keys():
        rhos = [f.rho(*k).float() if f.has_gains else
                torch.zeros(spec.site_shape(k[1]), device=dev) for f in fields]
        out[k] = combine(rhos)
    return out


def invert_field(field: ProgramField) -> ProgramField:
    """Sitewise unit inverse: q* = (w, -v). Gains flip sign in log-space (ρ → −ρ)."""
    spec = field.spec
    quats = {k: q_conjugate(q_normalize(field.q(*k).float())) for k in spec.site_keys()}
    rho = None
    if field.has_gains:
        rho = {k: -field.rho(*k).float() for k in spec.site_keys()}
    return ProgramField(spec, quats, rho, trainable=False)


def pow_field(field: ProgramField, lam) -> ProgramField:
    """Sitewise rotation power z^lam via q_pow (P0 exponentiation probe / D5
    upward strength scan). Gains scale linearly in log-space (ρ → lam·ρ)."""
    spec = field.spec
    quats = {k: q_pow(q_normalize(field.q(*k).float()), lam) for k in spec.site_keys()}
    rho = None
    if field.has_gains:
        rho = {k: lam * field.rho(*k).float() for k in spec.site_keys()}
    return ProgramField(spec, quats, rho, trainable=False)


def increment(z_yx: ProgramField, z_x: ProgramField) -> ProgramField:
    """Relative increment Δ_{y|x} = z_yx ⊗ z_x*  (sanity: Δ ⊗ z_x ≡ z_yx)."""
    return compose(z_yx, invert_field(z_x))


def compose(field_b: ProgramField, field_a: ProgramField) -> ProgramField:
    """Apply a FIRST, then b: per site q = hamilton(q_b, q_a) => R = R_b @ R_a.

    Argument order matches the spec signature compose(field_b, field_a).
    Gains: rho = rho_a + rho_b (abelian).
    """
    spec = _check_specs(field_b, field_a)
    quats = {k: hamilton(q_normalize(field_b.q(*k).float()),
                         q_normalize(field_a.q(*k).float()))
             for k in spec.site_keys()}
    rho = _merged_rhos([field_b, field_a], lambda rs: rs[0] + rs[1])
    return ProgramField(spec, quats, rho, trainable=False)


def _resolve_alpha(alpha: AlphaLike, key: SiteKey, n_layers: int):
    """alpha: scalar | per-layer tensor [L] | dict by (layer, name)."""
    layer, name = key
    if isinstance(alpha, dict):
        return alpha[key]
    if isinstance(alpha, torch.Tensor) and alpha.ndim > 0:
        if alpha.shape[0] != n_layers:
            raise ValueError(f"per-layer alpha must have shape [{n_layers}], got {tuple(alpha.shape)}")
        return alpha[layer]
    return alpha


def slerp_field(f0: ProgramField, f1: ProgramField, alpha: AlphaLike) -> ProgramField:
    """Geodesic interpolation sitewise; alpha=0 -> f0, alpha=1 -> ±f1.

    alpha: scalar, per-layer tensor of shape [L] (the point of per-layer
    fields: depth-wise slerp schedules), or dict keyed by (layer, name).
    Gains interpolate linearly in rho.
    """
    spec = _check_specs(f0, f1)
    quats = {}
    for k in spec.site_keys():
        a = _resolve_alpha(alpha, k, spec.n_layers)
        a = float(a) if not isinstance(a, torch.Tensor) else a.float()
        quats[k] = slerp(q_normalize(f0.q(*k).float()),
                         q_normalize(f1.q(*k).float()), a)
    rho = None
    if f0.has_gains or f1.has_gains:
        ref = f0 if f0.has_gains else f1
        dev = ref.rho(*spec.site_keys()[0]).device
        rho = {}
        for k in spec.site_keys():
            a = _resolve_alpha(alpha, k, spec.n_layers)
            a = float(a) if not isinstance(a, torch.Tensor) else a.float()
            r0 = f0.rho(*k).float() if f0.has_gains else torch.zeros(spec.site_shape(k[1]), device=dev)
            r1 = f1.rho(*k).float() if f1.has_gains else torch.zeros(spec.site_shape(k[1]), device=dev)
            rho[k] = (1 - a) * r0 + a * r1
    return ProgramField(spec, quats, rho, trainable=False)


def mean_field(fields: Sequence[ProgramField],
               weights: Optional[Sequence[float]] = None) -> ProgramField:
    """Chordal mean for k>=2 fields: sign-align all to fields[0] sitewise
    (per-quaternion), weighted sum, normalize. Gains: weighted mean of rho."""
    if len(fields) < 2:
        raise ValueError("mean_field needs at least 2 fields")
    spec = _check_specs(*fields)
    if weights is None:
        weights = [1.0 / len(fields)] * len(fields)
    if len(weights) != len(fields):
        raise ValueError("weights must match fields")
    wsum = float(sum(weights))
    weights = [w / wsum for w in weights]

    quats = {}
    for k in spec.site_keys():
        q0 = q_normalize(fields[0].q(*k).float())
        acc = weights[0] * q0
        for f, w in zip(fields[1:], weights[1:]):
            qi = q_normalize(f.q(*k).float())
            dot = (qi * q0).sum(-1, keepdim=True)
            qi = torch.where(dot < 0, -qi, qi)  # antipodal alignment to fields[0]
            acc = acc + w * qi
        quats[k] = q_normalize(acc)
    rho = _merged_rhos(fields, lambda rs: sum(w * r for w, r in zip(weights, rs)))
    return ProgramField(spec, quats, rho, trainable=False)
