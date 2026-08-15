"""tests/test_belief.py — phase-4 belief machinery tests T16–T21 (spec §6).

T18 (the whitening gate) runs inside scripts/calibrate_whitening.py against a
real base; here T17 and T20 are the kernel-level halves of their spec items
(the model-in-the-loop halves live in W0 logging and Experiment D).
"""
import math

import pytest
import torch

from lpm.belief import WhitenedGaussianBelief
from lpm.field import FieldSpec
from lpm.quaternion import q_exp, q_log, q_normalize, q_angle2
from lpm.whitening import Whitening, group_sizes

torch.manual_seed(0)

SPEC = FieldSpec(n_layers=2, d_model=12, n_heads=2, d_head=6, d_ff=24,
                 rotary_ndims=6)
LEV = {"attn_io": 10.0, "qk_rel": 50.0, "mlp_io": 10.0,
       "ffn_hidden": 30.0, "rope_ax": 2.0}


@pytest.fixture()
def wh():
    return Whitening(SPEC, LEV)


def test_t19_exp_log_roundtrip():
    q = q_normalize(torch.randn(2000, 4, dtype=torch.float64))
    canon = torch.where(q[:, :1] < 0, -q, q)
    back = q_exp(q_log(q))
    assert (back - canon).abs().max() < 1e-9
    # sign invariance and identity
    assert (q_log(-q) - q_log(q)).abs().max() < 1e-12
    ident = torch.zeros(5, 4, dtype=torch.float64)
    ident[:, 0] = 1.0
    assert q_log(ident).abs().max() == 0
    v = torch.randn(2000, 3, dtype=torch.float64) * 0.5
    assert (q_log(q_exp(v)) - v).abs().max() < 1e-9


def test_whitening_roundtrip_and_layout(wh):
    sizes = group_sizes(SPEC)
    assert wh.n_quats == sum(sizes.values())
    # c ordering follows leverage/size ratio
    assert wh.c["qk_rel"] > wh.c["attn_io"]
    from lpm.field import ProgramField
    f = ProgramField.randn_near_identity(SPEC, sigma=0.05)
    v = wh.field_to_whitened(f)
    f2 = wh.whitened_to_field(v)
    for k in SPEC.site_keys():
        a = q_normalize(f.q(*k).reshape(-1, 4))
        a = torch.where(a[:, :1] < 0, -a, a)
        b = f2.q(*k).reshape(-1, 4)
        assert (a - b).abs().max() < 1e-5


def test_t16_fusion_additive_order_invariant(wh):
    torch.manual_seed(1)
    ms = [torch.randn(wh.n_quats, 3) for _ in range(4)]
    rs = [0.5, 1.0, 2.0, 0.7]
    b1 = WhitenedGaussianBelief(wh, sigma0=1.0)
    for m, r in zip(ms, rs):
        b1.fuse(m, r)
    b2 = WhitenedGaussianBelief(wh, sigma0=1.0)
    for m, r in reversed(list(zip(ms, rs))):
        b2.fuse(m, r)
    assert (b1.lam - b2.lam).abs().max() < 1e-6
    assert (b1.eta - b2.eta).abs().max() < 1e-5
    # additivity in natural params
    b3 = WhitenedGaussianBelief(wh, sigma0=1.0)
    b3.fuse(ms[0], rs[0])
    assert torch.allclose(b3.lam, b1.lam - sum(1 / r ** 2 for r in rs[1:]),
                          atol=1e-5)


def test_t17_sample_activity_monotone_bounded(wh):
    """Kernel half of T17: sampled-field mean activity is monotone in prior
    width and bounded; the KL envelope itself is measured in W0/D logging."""
    acts = []
    for s0 in (0.01, 0.05, 0.1):
        b = WhitenedGaussianBelief(wh, sigma0=s0)
        f = b.sample_field(torch.Generator().manual_seed(3))
        xs = torch.cat([q_angle2(f.q(*k).reshape(-1, 4)) for k in SPEC.site_keys()])
        acts.append(float(xs.mean()))
    assert acts[0] < acts[1] < acts[2]
    assert acts[2] < 1.0


def test_t20_stationary_convergence(wh):
    """Synthetic stationary task: noisy observations of a fixed target —
    posterior concentrates (potency monotone down), MAP approaches target."""
    torch.manual_seed(2)
    target = torch.randn(wh.n_quats, 3) * 0.1
    b = WhitenedGaussianBelief(wh, sigma0=1.0)
    g = torch.Generator().manual_seed(9)
    pots, errs = [], []
    for step in range(250):
        b.predict(q_drift=0.0)
        obs = target + 0.2 * torch.randn(*target.shape, generator=g)
        b.fuse(obs, r_obs=0.2)
        pots.append(b.potency()["total"])
        errs.append(float((b.mu - target).norm()))
    assert all(p2 <= p1 + 1e-9 for p1, p2 in zip(pots, pots[1:]))
    assert errs[-1] < errs[0] * 0.1


def test_t21_bingham_gaussian_map_agreement():
    """In-regime (small angles), fusing axial-Bingham measurements
    (A = Σ λ_k q_k q_kᵀ, MAP = top eigenvector) agrees with tangent-Gaussian
    fusion (μ = Σλv/Σλ, MAP = exp(μ)) to < 1° per site."""
    torch.manual_seed(4)
    g = torch.Generator().manual_seed(5)
    for _ in range(50):
        vs = 0.05 * torch.randn(3, 3, dtype=torch.float64, generator=g)
        lams = torch.rand(3, dtype=torch.float64, generator=g) * 5 + 1
        qs = q_exp(vs)
        A = sum(l * torch.outer(q, q) for l, q in zip(lams, qs))
        evals, evecs = torch.linalg.eigh(A)
        q_bing = evecs[:, -1]
        q_bing = torch.where(q_bing[0] < 0, -q_bing, q_bing)
        mu = (lams.unsqueeze(-1) * vs).sum(0) / lams.sum()
        q_gauss = q_exp(mu)
        dot = float((q_bing * q_gauss).sum().abs().clamp(max=1.0))
        angle_deg = math.degrees(2 * math.acos(dot))
        assert angle_deg < 1.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
