"""tests/test_neox_axis.py — B1 wiring gates for the NeoX/Pythia axis field
(steering-workstreams-a-b.md B0/B1).

T13: identity program ≡ base, exactly (all fields incl. rope_ax).
T14: relative-position preservation — logits invariant to a constant shift of
     position_ids under a random program (axis conjugation preserves
     q(m)ᵀk(n) = content·A·RoPE(n−m)·Aᵀ·content).
T15: frequency spectrum of the relative rotation unchanged — conjugation
     reorients coupling planes, never frequencies (eigenvalue test).

Model gates run on EleutherAI/pythia-70m by default (same architecture class
as pythia-410m: partial rotary 0.25, parallel residual); override with
LPM_NEOX_MODEL. Skipped cleanly when weights are unavailable (offline CI).
"""
import os

import pytest
import torch

from lpm import LatentProgramModel, ProgramField
from lpm.quaternion import q_normalize, q_to_R

MODEL_ID = os.environ.get("LPM_NEOX_MODEL", "EleutherAI/pythia-70m")

torch.manual_seed(0)


@pytest.fixture(scope="module")
def model():
    try:
        m = LatentProgramModel.from_pretrained(MODEL_ID)
    except Exception as e:  # pragma: no cover - offline environments
        pytest.skip(f"cannot load {MODEL_ID}: {e}")
    m.eval()
    return m


@pytest.fixture(scope="module")
def ids(model):
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, model.config.vocab_size, (2, 48), generator=g)


def test_spec_has_rope_ax(model):
    spec = model.spec
    assert spec.rotary_ndims > 0
    assert "rope_ax" in spec.site_names()
    part = spec.partition("rope_ax")
    assert 3 * part.n3 + part.rem == spec.rotary_ndims
    assert spec.site_shape("rope_ax") == (spec.n_heads, part.n3)


def test_t13_identity_program_is_base(model, ids):
    with torch.no_grad():
        base_logits = model(input_ids=ids).logits
        with model.program(model.identity_field()):
            prog_logits = model(input_ids=ids).logits
    assert (prog_logits - base_logits).abs().max().item() < 1e-5


def _identity_qk_rel(field: ProgramField) -> ProgramField:
    """Zero out qk_rel (set to identity quaternion), keep all other sites.

    A STATIC qk_rel applied post-RoPE is not shift-invariant on a rotary base:
    score = q̃ᵀ·RoPE₋ₘ·M·RoPEₙ·k̃ ≠ f(n−m) unless M commutes with the rotary
    planes. That is exactly why it is the axis field's degenerate control
    (steering doc B0); T14 therefore gates every site EXCEPT qk_rel.
    """
    quats = {}
    for k in field.spec.site_keys():
        q = field.q(*k).detach().clone()
        if k[1] == "qk_rel":
            q.zero_()
            q[..., 0] = 1.0
        quats[k] = q
    return ProgramField(field.spec, quats, trainable=False)


def test_t14_relative_position_preservation(model, ids):
    field = _identity_qk_rel(
        ProgramField.randn_near_identity(model.spec, sigma=0.05))
    T = ids.shape[1]
    pos = torch.arange(T).unsqueeze(0).expand(ids.shape[0], -1)
    with torch.no_grad():
        # sanity: the base itself is relative (fp32 noise scales with the
        # logit magnitude — pythia logits run O(1e3), so tolerance is relative)
        b0 = model(input_ids=ids, position_ids=pos).logits
        b7 = model(input_ids=ids, position_ids=pos + 7).logits
        scale = b0.abs().max().item()
        base_drift = (b0 - b7).abs().max().item()
        assert base_drift < 2e-4 * scale
        with model.program(field):
            l0 = model(input_ids=ids, position_ids=pos).logits
        with model.program(field):
            l7 = model(input_ids=ids, position_ids=pos + 7).logits
    # self-calibrated: the program's shift drift must stay at the base's own
    # fp-accumulation scale (a relativity break would be orders larger — the
    # qk_rel control below lands at >1e-2·scale)
    assert (l0 - l7).abs().max().item() < 3 * base_drift + 1e-5 * scale
    # and the program actually does something (not vacuous)
    assert (l0 - b0).abs().max().item() > 1e-2 * scale


def test_qk_rel_is_the_degenerate_control(model, ids):
    """Documented expectation: a static post-RoPE qk_rel field is NOT
    shift-invariant on a rotary base (the axis field is)."""
    spec = model.spec
    ident = ProgramField.identity(spec)
    quats = {k: ident.q(*k).clone() for k in spec.site_keys()}
    torch.manual_seed(3)
    for k in spec.site_keys():
        if k[1] == "qk_rel":
            q = torch.zeros(*spec.site_shape(k[1]), 4)
            q[..., 0] = 1.0
            q[..., 1:] = 0.05 * torch.randn(*spec.site_shape(k[1]), 3)
            quats[k] = q
    field = ProgramField(spec, quats, trainable=False)
    T = ids.shape[1]
    pos = torch.arange(T).unsqueeze(0).expand(ids.shape[0], -1)
    with torch.no_grad():
        with model.program(field):
            l0 = model(input_ids=ids, position_ids=pos).logits
        with model.program(field):
            l7 = model(input_ids=ids, position_ids=pos + 7).logits
    scale = l0.abs().max().item()
    assert (l0 - l7).abs().max().item() > 1e-4 * scale


def _neox_rope_matrix(nd: int, delta: float) -> torch.Tensor:
    """Relative rotation RoPE(Δ) on nd dims, NeoX pairing (i, i+nd/2)."""
    half = nd // 2
    inv_freq = 10000.0 ** (-torch.arange(0, half, dtype=torch.float64) / half)
    R = torch.zeros(nd, nd, dtype=torch.float64)
    for i in range(half):
        c = torch.cos(inv_freq[i] * delta)
        s = torch.sin(inv_freq[i] * delta)
        R[i, i] = c
        R[i, i + half] = -s
        R[i + half, i] = s
        R[i + half, i + half] = c
    return R


def test_t15_conjugation_preserves_frequency_spectrum():
    nd = 16
    part_n3 = nd // 3          # 5 blocks, 1 pass-through dim
    q = q_normalize(torch.randn(part_n3, 4, dtype=torch.float64))
    blocks = q_to_R(q)
    A = torch.eye(nd, dtype=torch.float64)
    for i in range(part_n3):
        A[3 * i:3 * i + 3, 3 * i:3 * i + 3] = blocks[i]
    rope = _neox_rope_matrix(nd, delta=3.7)
    conj = A @ rope @ A.T
    ev0 = torch.linalg.eigvals(rope)
    ev1 = torch.linalg.eigvals(conj)
    a0 = torch.sort(torch.angle(ev0).abs()).values
    a1 = torch.sort(torch.angle(ev1).abs()).values
    assert (a0 - a1).abs().max().item() < 1e-10


def test_t14_math_axis_sandwich_is_relative():
    """Kernel-level T14: (A RoPE_m Aᵀ q)·(A RoPE_n Aᵀ k) depends only on n−m."""
    nd = 16
    q = torch.randn(nd, dtype=torch.float64)
    k = torch.randn(nd, dtype=torch.float64)
    qq = q_normalize(torch.randn(nd // 3, 4, dtype=torch.float64))
    blocks = q_to_R(qq)
    A = torch.eye(nd, dtype=torch.float64)
    for i in range(nd // 3):
        A[3 * i:3 * i + 3, 3 * i:3 * i + 3] = blocks[i]

    def score(m, n):
        qm = A @ _neox_rope_matrix(nd, m) @ A.T @ q
        kn = A @ _neox_rope_matrix(nd, n) @ A.T @ k
        return float(qm @ kn)

    assert abs(score(3, 10) - score(103, 110)) < 1e-10


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
