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
        make_partition, apply_rot, apply_rot_head,
    )
except ImportError:
    from quaternion import (
        q_normalize, q_to_R, hamilton, d2_chord, slerp,
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
