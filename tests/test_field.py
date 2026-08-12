"""ProgramField unit tests: identity/init/save-load/cache/sign-invariance."""
import torch
import pytest

from lpm.field import FieldSpec, ProgramField, AxisBank, AbelianProgramField, SITE_NAMES
from lpm.quaternion import d2_chord, q_normalize

SPEC = FieldSpec(n_layers=2, d_model=48, n_heads=4, d_head=12, d_ff=192)


def test_identity_rotations_are_exact_identity():
    f = ProgramField.identity(SPEC)
    eye = torch.eye(3)
    for (l, n), R in f.rotations().items():
        assert (R - eye).abs().max() == 0, (l, n)
    assert f.gains() is None


def test_identity_with_gains_is_ones():
    f = ProgramField.identity(SPEC, enable_gains=True)
    for g in f.gains().values():
        assert (g - 1.0).abs().max() == 0


def test_randn_near_identity_scale():
    g = torch.Generator().manual_seed(0)
    f = ProgramField.randn_near_identity(SPEC, sigma=1e-3, generator=g)
    for (l, n) in f.sites():
        q = f.q(l, n)
        assert (q[..., 0] - 1.0).abs().max() == 0
        assert q[..., 1:].abs().max() < 1e-2


def test_site_shapes_and_partitions():
    assert SPEC.site_shape("attn_io") == (16,)
    assert SPEC.site_shape("mlp_io") == (16,)
    assert SPEC.site_shape("ffn_hidden") == (64,)
    assert SPEC.site_shape("qk_rel") == (4, 4)
    assert SPEC.partition("qk_rel").rem == 0
    spec_tied = FieldSpec(n_layers=2, d_model=48, n_heads=4, d_head=12, d_ff=192,
                          tie_qk_across_heads=True)
    assert spec_tied.site_shape("qk_rel") == (1, 4)
    f = ProgramField.randn_near_identity(spec_tied, sigma=0.1)
    assert f.rotations()[(0, "qk_rel")].shape == (4, 4, 3, 3)  # expanded to H


def test_save_load_roundtrip(tmp_path):
    f = ProgramField.randn_near_identity(SPEC, sigma=0.2, enable_gains=True)
    with torch.no_grad():
        for k in f.sites():
            f.rho(*k).normal_(std=0.1)
    p = str(tmp_path / "field.pt")
    f.save(p)
    g = ProgramField.load(p, trainable=True)
    assert g.spec == f.spec
    assert g.trainable
    for k in f.sites():
        assert torch.equal(f.q(*k), g.q(*k))
        assert torch.equal(f.rho(*k), g.rho(*k))


def test_rotation_cache_hit_and_invalidation_on_param_update():
    f = ProgramField.randn_near_identity(SPEC, sigma=0.1, trainable=True)
    with torch.no_grad():
        r1 = f.rotations()
        r2 = f.rotations()
        assert r1 is r2  # cache hit
        # in-place update (what optimizer.step does) must invalidate
        f.q(0, "attn_io").add_(0.05)
        r3 = f.rotations()
    assert r3 is not r1
    assert (r3[(0, "attn_io")] - r1[(0, "attn_io")]).abs().max() > 0


def test_rotations_carry_graph_when_training():
    f = ProgramField.randn_near_identity(SPEC, sigma=0.1, trainable=True)
    R = f.rotations()[(0, "attn_io")]
    assert R.requires_grad
    R.sum().backward()
    assert f.q(0, "attn_io").grad is not None


def test_dtype_cast_refused():
    f = ProgramField.identity(SPEC)
    with pytest.raises(ValueError):
        f.to(torch.float16)


def test_sign_flip_field_same_rotations():
    f = ProgramField.randn_near_identity(SPEC, sigma=0.3)
    g = f.flipped_sign()
    for k in f.sites():
        assert (f.rotations()[k] - g.rotations()[k]).abs().max() < 1e-6


def test_chordal_reg():
    assert ProgramField.identity(SPEC).chordal_reg_to_identity().item() == 0
    f = ProgramField.randn_near_identity(SPEC, sigma=0.3)
    assert f.chordal_reg_to_identity().item() > 0


def test_abelian_field_commutes():
    """Same-axis rotations commute: the E3 baseline is provably order-blind."""
    from lpm.compose import compose
    from lpm.quaternion import hamilton
    torch.manual_seed(1)
    bank = AxisBank(SPEC)
    fa, fb = AbelianProgramField(bank), AbelianProgramField(bank)
    with torch.no_grad():
        for (l, n) in SPEC.site_keys():
            key = f"L{l}_{n}"
            fa.theta[key].normal_(std=0.7)
            fb.theta[key].normal_(std=0.7)
    ab = compose(fb.to_program_field(), fa.to_program_field())
    ba = compose(fa.to_program_field(), fb.to_program_field())
    for k in SPEC.site_keys():
        assert d2_chord(ab.q(*k), ba.q(*k)).abs().max() < 1e-6

