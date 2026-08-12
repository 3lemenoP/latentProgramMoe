"""Composition/merging tests (spec §6, §9): T9 order sensitivity + convention,
T8 at the field level, chordal mean, gain algebra."""
import pytest
import torch

from lpm.compose import compose, mean_field, slerp_field
from lpm.field import FieldSpec, ProgramField
from lpm.quaternion import d2_chord, q_normalize

SPEC = FieldSpec(n_layers=3, d_model=48, n_heads=4, d_head=12, d_ff=192)


def rand_field(sigma=0.3, seed=0, gains=False):
    g = torch.Generator().manual_seed(seed)
    return ProgramField.randn_near_identity(SPEC, sigma=sigma,
                                            enable_gains=gains, generator=g)


def field_dist(f, g):
    return max(d2_chord(q_normalize(f.q(*k).float()),
                        q_normalize(g.q(*k).float())).max().item()
               for k in f.sites())


# ---------------------------------------------------------------------------
# T9 â€” composition order
# ---------------------------------------------------------------------------
def test_t9_composition_noncommutative():
    a, b = rand_field(seed=1), rand_field(seed=2)
    ab = compose(b, a)   # a first, then b
    ba = compose(a, b)
    assert field_dist(ab, ba) > 0.01


def test_t9_composition_convention_matches_Rb_Ra():
    a, b = rand_field(seed=3), rand_field(seed=4)
    c = compose(b, a)
    Ra, Rb, Rc = a.rotations(), b.rotations(), c.rotations()
    for k in a.sites():
        assert (Rc[k] - Rb[k] @ Ra[k]).abs().max() < 1e-5


def test_composition_order_matters_behaviorally(tiny_gpt2, tiny_ids):
    from test_sandwich import logits, rand_field as model_rand_field
    a = model_rand_field(tiny_gpt2, sigma=0.3, seed=21)
    b = model_rand_field(tiny_gpt2, sigma=0.3, seed=22)
    ab = logits(tiny_gpt2, tiny_ids, compose(b, a))
    ba = logits(tiny_gpt2, tiny_ids, compose(a, b))
    assert (ab - ba).abs().mean() > 1e-4


def test_identity_is_neutral_element():
    a = rand_field(seed=5)
    e = ProgramField.identity(SPEC)
    assert field_dist(compose(a, e), a) < 1e-6
    assert field_dist(compose(e, a), a) < 1e-6


# ---------------------------------------------------------------------------
# T8 (field level) â€” slerp_field endpoints / schedules
# ---------------------------------------------------------------------------
def test_slerp_field_endpoints():
    f0, f1 = rand_field(seed=6), rand_field(seed=7)
    assert field_dist(slerp_field(f0, f1, 0.0), f0) < 1e-6
    assert field_dist(slerp_field(f0, f1, 1.0), f1) < 1e-6  # d2_chord is sign-blind
    mid = slerp_field(f0, f1, 0.5)
    for k in mid.sites():
        assert (mid.q(*k).norm(dim=-1) - 1).abs().max() < 1e-5


def test_slerp_field_per_layer_alpha():
    f0, f1 = rand_field(seed=8), rand_field(seed=9)
    alpha = torch.tensor([0.0, 0.5, 1.0])
    m = slerp_field(f0, f1, alpha)
    for name in ("attn_io", "qk_rel", "mlp_io", "ffn_hidden"):
        assert d2_chord(q_normalize(m.q(0, name)), q_normalize(f0.q(0, name))).max() < 1e-6
        assert d2_chord(q_normalize(m.q(2, name)), q_normalize(f1.q(2, name))).max() < 1e-6


def test_slerp_field_dict_alpha():
    f0, f1 = rand_field(seed=10), rand_field(seed=11)
    alpha = {k: (0.0 if k[1] == "attn_io" else 1.0) for k in f0.sites()}
    m = slerp_field(f0, f1, alpha)
    assert d2_chord(q_normalize(m.q(1, "attn_io")), q_normalize(f0.q(1, "attn_io"))).max() < 1e-6
    assert d2_chord(q_normalize(m.q(1, "mlp_io")), q_normalize(f1.q(1, "mlp_io"))).max() < 1e-6


def test_slerp_field_bad_alpha_shape_raises():
    f0, f1 = rand_field(seed=12), rand_field(seed=13)
    with pytest.raises(ValueError):
        slerp_field(f0, f1, torch.tensor([0.5, 0.5]))  # L=3 expected


# ---------------------------------------------------------------------------
# mean_field â€” chordal mean, sign alignment
# ---------------------------------------------------------------------------
def test_mean_field_sign_alignment():
    f = rand_field(seed=14)
    m = mean_field([f, f.flipped_sign()])
    assert field_dist(m, f) < 1e-6


def test_mean_field_weights_degenerate_to_input():
    f0, f1 = rand_field(seed=15), rand_field(seed=16)
    m = mean_field([f0, f1], weights=[1.0, 0.0])
    assert field_dist(m, f0) < 1e-6


def test_mean_field_unit_norm():
    fs = [rand_field(seed=s) for s in (17, 18, 19)]
    m = mean_field(fs)
    for k in m.sites():
        assert (m.q(*k).norm(dim=-1) - 1).abs().max() < 1e-6


# ---------------------------------------------------------------------------
# gains algebra through compose / slerp / mean
# ---------------------------------------------------------------------------
def _with_rho(f, std, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for k in f.sites():
            f.rho(*k).copy_(std * torch.randn(f.rho(*k).shape, generator=g))
    return f


def test_gains_slerp_linear_in_rho():
    f0 = _with_rho(rand_field(seed=20, gains=True), 0.3, 1)
    f1 = _with_rho(rand_field(seed=21, gains=True), 0.3, 2)
    m = slerp_field(f0, f1, 0.25)
    for k in m.sites():
        want = 0.75 * f0.rho(*k) + 0.25 * f1.rho(*k)
        assert torch.allclose(m.rho(*k), want, atol=1e-6)


def test_gains_mean_weighted():
    f0 = _with_rho(rand_field(seed=22, gains=True), 0.3, 3)
    f1 = _with_rho(rand_field(seed=23, gains=True), 0.3, 4)
    m = mean_field([f0, f1], weights=[0.25, 0.75])
    for k in m.sites():
        want = 0.25 * f0.rho(*k) + 0.75 * f1.rho(*k)
        assert torch.allclose(m.rho(*k), want, atol=1e-6)


def test_gains_missing_treated_as_zero():
    f0 = _with_rho(rand_field(seed=24, gains=True), 0.3, 5)
    f1 = rand_field(seed=25, gains=False)
    c = compose(f1, f0)
    assert c.has_gains
    for k in c.sites():
        assert torch.allclose(c.rho(*k), f0.rho(*k), atol=1e-6)


def test_spec_mismatch_raises():
    other = FieldSpec(n_layers=2, d_model=48, n_heads=4, d_head=12, d_ff=192)
    with pytest.raises(ValueError):
        compose(ProgramField.identity(other), rand_field(seed=26))

