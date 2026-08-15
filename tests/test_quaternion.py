"""tests/test_quaternion.py — kernel property tests (spec §9: T1, T2, T3, T6-math, T8, T9).

Model-level tests T4/T5/T7/T10 live in tests/test_sandwich.py once model_wrap
exists; the dead-frame *math* (the mechanism behind T5) is checked here.
Tests run in float64 for tight tolerances; production code runs fp32.
"""
import torch
import pytest

try:
    from lpm.quaternion import (
        q_normalize, q_to_R, hamilton, d2_chord, slerp,
        q_conjugate, q_angle2, q_commutator, d_geo, q_pow,
        make_partition, apply_rot, apply_rot_head,
    )
except ImportError:
    from quaternion import (
        q_normalize, q_to_R, hamilton, d2_chord, slerp,
        q_conjugate, q_angle2, q_commutator, d_geo, q_pow,
        make_partition, apply_rot, apply_rot_head,
    )

torch.manual_seed(0)
DT = torch.float64


def rand_q(*shape):
    return q_normalize(torch.randn(*shape, 4, dtype=DT))


def blockdiag(R, rem):
    n3 = R.shape[0]
    d = 3 * n3 + rem
    B = torch.eye(d, dtype=DT)
    for i in range(n3):
        B[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R[i]
    return B


def test_t1_orthogonality_and_det():
    R = q_to_R(rand_q(1000))
    eye = torch.eye(3, dtype=DT)
    assert (R.transpose(-1, -2) @ R - eye).abs().max() < 1e-12
    assert (torch.linalg.det(R) - 1).abs().max() < 1e-12


def test_t2_homomorphism():
    a, b = rand_q(1000), rand_q(1000)
    err = (q_to_R(hamilton(a, b)) - q_to_R(a) @ q_to_R(b)).abs().max()
    assert err < 1e-12


def test_t3_sign_invariance():
    q = rand_q(1000)
    assert (q_to_R(-q) - q_to_R(q)).abs().max() == 0
    assert d2_chord(q, -q).abs().max() < 1e-12


def test_composition_convention_a_then_b():
    a, b = rand_q(500), rand_q(500)
    v = torch.randn(500, 3, dtype=DT)
    seq = torch.einsum('nij,nj->ni', q_to_R(b), torch.einsum('nij,nj->ni', q_to_R(a), v))
    one = torch.einsum('nij,nj->ni', q_to_R(hamilton(b, a)), v)
    assert (seq - one).abs().max() < 1e-12


def test_t8_slerp():
    a, b = rand_q(1000), rand_q(1000)
    assert (slerp(a, b, 0.0) - a).abs().max() < 1e-9
    s1 = slerp(a, b, 1.0)
    end = torch.minimum((s1 - b).abs().amax(-1), (s1 + b).abs().amax(-1))
    assert end.max() < 1e-9
    smid = slerp(a, b, 0.37)
    assert (smid.norm(dim=-1) - 1).abs().max() < 1e-9
    b_anti = q_normalize(-a + 1e-9 * torch.randn_like(a))
    s_anti = slerp(a, b_anti, 0.5)
    assert torch.isfinite(s_anti).all()
    assert (s_anti.norm(dim=-1) - 1).abs().max() < 1e-6


def test_slerp_gradients_finite():
    a, b = rand_q(64), rand_q(64)
    raw = torch.randn(64, 4, dtype=DT, requires_grad=True)
    out = slerp(a, q_normalize(raw), 0.5)
    out.sum().backward()
    assert torch.isfinite(raw.grad).all()


def test_t9_noncommutative_generic():
    a, b = rand_q(1000), rand_q(1000)
    assert d2_chord(hamilton(b, a), hamilton(a, b)).mean() > 0.01


def test_apply_rot_matches_blockdiag_and_properties():
    part = make_partition(10)
    assert (part.n3, part.rem) == (3, 1)
    R = q_to_R(rand_q(part.n3))
    x = torch.randn(7, 5, 10, dtype=DT)
    y = apply_rot(x, R, part)
    B = blockdiag(R, part.rem)
    assert (y - x @ B.T).abs().max() < 1e-12
    assert (y.norm(dim=-1) - x.norm(dim=-1)).abs().max() < 1e-12
    assert (y[..., -1] - x[..., -1]).abs().max() == 0
    assert (apply_rot(y, R, part, inverse=True) - x).abs().max() < 1e-12


def test_apply_rot_head_roundtrip():
    part = make_partition(64)
    assert (part.n3, part.rem) == (21, 1)
    H = 4
    R = q_to_R(rand_q(H, part.n3))
    x = torch.randn(2, H, 9, 64, dtype=DT)
    y = apply_rot_head(x, R, part)
    assert y.shape == x.shape
    assert (apply_rot_head(y, R, part, inverse=True) - x).abs().max() < 1e-12
    assert (y[..., -1] - x[..., -1]).abs().max() == 0


def test_t6_spectrum_preserved():
    part = make_partition(12)
    B = blockdiag(q_to_R(rand_q(part.n3)), part.rem)
    W = torch.randn(12, 12, dtype=DT)
    s0 = torch.linalg.svdvals(W)
    s1 = torch.linalg.svdvals(B @ W @ B.T)
    assert (torch.sort(s0).values - torch.sort(s1).values).abs().max() < 1e-10


def test_automorphism():
    part = make_partition(12)
    B = blockdiag(q_to_R(rand_q(part.n3)), part.rem)
    A = torch.randn(12, 12, dtype=DT)
    C = torch.randn(12, 12, dtype=DT)
    lhs = (B @ A @ B.T) @ (B @ C @ B.T)
    rhs = B @ (A @ C) @ B.T
    assert (lhs - rhs).abs().max() < 1e-10


def test_t11_conjugate_commutator_geo():
    """E3′ T11: conjugate is inverse; coaxial commutator is identity."""
    q = rand_q(1000)
    ident = torch.zeros(1000, 4, dtype=DT)
    ident[:, 0] = 1.0
    qc = q_conjugate(q)
    assert (q_to_R(qc) - q_to_R(q).transpose(-1, -2)).abs().max() < 1e-12
    prod = hamilton(q, qc)
    end = torch.minimum((prod - ident).abs().amax(-1), (prod + ident).abs().amax(-1))
    assert end.max() < 1e-12
    n = torch.tensor([1.0, 0.0, 0.0], dtype=DT)
    th = torch.rand(500, dtype=DT) * 3.0 + 0.1
    ph = torch.rand(500, dtype=DT) * 3.0 + 0.1
    half = torch.stack([th, ph], dim=-1) / 2
    axis = n.expand(500, 3)
    qa = torch.cat([torch.cos(half[:, :1]), torch.sin(half[:, :1]) * axis], dim=-1)
    qb = torch.cat([torch.cos(half[:, 1:]), torch.sin(half[:, 1:]) * axis], dim=-1)
    comm = q_commutator(qa, qb)
    cend = torch.minimum((comm - ident[:500]).abs().amax(-1),
                         (comm + ident[:500]).abs().amax(-1))
    assert cend.max() < 1e-10
    assert (q_angle2(q) - (1 - q[..., 0] ** 2)).abs().max() < 1e-12
    assert (q_angle2(q) - q_angle2(-q)).abs().max() < 1e-12
    assert d_geo(q, q).abs().max() < 1e-6
    assert d_geo(q, -q).abs().max() < 1e-6


def test_t12_q_pow():
    """E3′ T12: rotation power. q^1 = canonical(q); q^0 = id; q^2 = ±q⊗q;
    sign invariance q_pow(-q,λ) = q_pow(q,λ); R(q^λ) continuous in λ."""
    q = rand_q(1000)
    canon = torch.where(q[:, :1] < 0, -q, q)
    ident = torch.zeros(1000, 4, dtype=DT)
    ident[:, 0] = 1.0

    assert (q_pow(q, 1.0) - canon).abs().max() < 1e-12
    assert (q_pow(q, 0.0) - ident).abs().max() < 1e-12

    sq = q_pow(q, 2.0)
    hh = hamilton(q, q)
    end = torch.minimum((sq - hh).abs().amax(-1), (sq + hh).abs().amax(-1))
    assert end.max() < 1e-12

    assert (q_pow(-q, 0.37) - q_pow(q, 0.37)).abs().max() < 1e-12

    # unit norm at fractional powers
    assert (q_pow(q, 1.5).norm(dim=-1) - 1).abs().max() < 1e-12

    # identity input stays identity (eps branch)
    assert (q_pow(ident, 2.5) - ident).abs().max() == 0

    # R(q^lam) continuous in lam: small dlam -> small rotation change
    lams = torch.linspace(0.0, 2.0, 41, dtype=DT)
    Rs = torch.stack([q_to_R(q_pow(q[:64], float(l))) for l in lams])
    steps = (Rs[1:] - Rs[:-1]).abs().amax(dim=(-1, -2))
    assert steps.max() < 0.2  # max angle pi * dlam/2 bounds the matrix delta

    # group consistency: q^a ⊗ q^b = ±q^(a+b)
    lhs = hamilton(q_pow(q, 0.7), q_pow(q, 0.6))
    rhs = q_pow(q, 1.3)
    end = torch.minimum((lhs - rhs).abs().amax(-1), (lhs + rhs).abs().amax(-1))
    assert end.max() < 1e-12


def test_dead_value_frame_math():
    """Mechanism behind T5: per-sequence value-frame conjugation cancels through
    linear attention mixing. If this fails, apply_rot broke; if the model-level
    T5 fails while this passes, the sandwich wiring broke."""
    part = make_partition(10)
    R = q_to_R(rand_q(part.n3))
    attn = torch.rand(5, 8, 8, dtype=DT)
    attn = attn / attn.sum(-1, keepdim=True)
    v = torch.randn(5, 8, 10, dtype=DT)
    plain = torch.einsum('bij,bjd->bid', attn, v)
    conj = apply_rot(torch.einsum('bij,bjd->bid', attn, apply_rot(v, R, part)),
                     R, part, inverse=True)
    assert (conj - plain).abs().max() < 1e-12


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
