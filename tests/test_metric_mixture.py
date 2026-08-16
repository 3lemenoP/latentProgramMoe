"""tests/test_metric_mixture.py — phase-4 v2 tests T22–T27 (spec §4).

T22/T23 exercise the M-metric on synthetic libraries with a known planted
live subspace; T24–T27 exercise the IMM belief against synthetic loss
landscapes (the class is model-agnostic by design, so the tests need no
transformer)."""
import torch

from lpm.metric import MetricM, loo_stress, spearman
from lpm.mixture_belief import IMMBelief

torch.manual_seed(0)

N_SITES = 40
D = N_SITES * 3
C_VEC = torch.rand(N_SITES) * 0.1 + 0.02


def synthetic_library(n=20, r_live=3, noise=0.02, seed=0):
    """Fields = live-subspace coords × planted basis + dead-sea noise;
    behavioral distance depends ONLY on live coords (anisotropically)."""
    g = torch.Generator().manual_seed(seed)
    B = torch.linalg.qr(torch.randn(D, r_live, generator=g)).Q
    Zt = torch.randn(n, r_live, generator=g)
    w_true = torch.tensor([4.0, 1.0, 0.25])[:r_live]
    V = Zt @ B.T + noise * torch.randn(n, D, generator=g)
    K = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            d2 = float((w_true * (Zt[i] - Zt[j]) ** 2).sum())
            K[i, j] = K[j, i] = d2
    return V, K, B, Zt


def test_t22_loo_gate_beats_baselines():
    V, K, _, _ = synthetic_library()
    res = loo_stress(V, K, C_VEC, var_keep=0.95, ridge=1e-4)
    assert res["G1_pass"], res
    assert res["spearman_M"] > 0.9


def test_t23_hybrid_consistency():
    V, K, B, _ = synthetic_library(noise=0.0)
    m = MetricM.fit(V, K, C_VEC, var_keep=0.999, ridge=1e-4)
    # live-subspace round-trip lossless for library fields (noise-free case)
    for i in range(V.shape[0]):
        _, resid = m.project(V[i])
        assert float(resid.norm()) < 1e-4 * max(float(V[i].norm()), 1.0)
    # dead-sea direction prices at W0 weights exactly
    g = torch.Generator().manual_seed(3)
    v = torch.randn(D, generator=g)
    z = (v - 0.0) @ m.basis
    v_dead = v - m.basis @ ((v) @ m.basis)      # orthogonal to live subspace
    d2 = m.dist2(m.mean + v_dead, m.mean)
    expect = ((C_VEC ** 2) * v_dead.reshape(-1, 3).pow(2).sum(-1)).sum()
    assert abs(float(d2 - expect)) < 1e-3 * max(float(expect), 1e-6)


def _mk_imm(n_sites=N_SITES, k=3, seed=1, **kw):
    g = torch.Generator().manual_seed(seed)
    means = [0.1 * torch.randn(n_sites, 3, generator=g) for _ in range(k)]
    names = [f"m{i}" for i in range(k)]
    gid = torch.zeros(n_sites, dtype=torch.long)
    return IMMBelief(means, names, gid, ["only"], **kw), means


def test_t24_stationary_responsibility_and_isolation():
    b, means = _mk_imm()
    N = means[0].shape[0]
    others_before = [c.mu.clone() for c in b.components[1:]]
    r_obs = torch.full((N,), 0.02)
    g = torch.Generator().manual_seed(5)
    last = None
    for t in range(30):
        losses = [0.1, 2.0, 2.0, 2.0]           # comp0 always best (+novelty)
        m = means[0] + 0.01 * torch.randn(N, 3, generator=g)
        last = b.step(losses, m, r_obs, t=t)
    assert last["winner"] == 0
    assert last["resp"][0] >= 0.9
    for c, mu0 in zip(b.components[1:], others_before):
        assert torch.equal(c.mu, mu0)           # T27 embedded: untouched


def test_t27_fusion_isolation_hard():
    b, means = _mk_imm()
    N = means[0].shape[0]
    snap = [(c.mu.clone(), c.lam.clone()) for c in b.all_components()]
    r_obs = torch.full((N,), 0.02)
    info = b.step([0.05, 3.0, 3.0, 3.0], means[0] + 0.05, r_obs, t=0)
    assert info["winner"] == 0
    for k, c in enumerate(b.all_components()):
        mu0, lam0 = snap[k]
        if k == 0:
            assert not torch.equal(c.mu, mu0)
        else:
            assert torch.equal(c.mu, mu0) and torch.equal(c.lam, lam0)


def test_t26_single_birth_per_novel_regime():
    b, means = _mk_imm(t_birth=5)
    N = means[0].shape[0]
    target = 0.2 * torch.ones(N, 3)
    r_obs = torch.full((N,), 0.02)
    for t in range(20):
        # all named components bad, novelty best with falling loss
        loss_novel = 1.0 / (t + 1)
        losses = [3.0, 3.0, 3.0, loss_novel]
        if len(b.components) == 4:              # birth happened
            losses = [3.0, 3.0, 3.0, 0.05, 2.0]
        b.step(losses, target, r_obs, t=t)
    assert len(b.births) == 1


def test_t25_scripted_aba_revisit_faster():
    """A→B→A with a synthetic loss landscape: loss_k = ‖μ_k − target‖;
    refine moves a fraction toward the target. Revisit must recover faster."""
    b, means = _mk_imm(k=2, t_birth=4, seed=7)
    N = means[0].shape[0]
    tA, tB = means[0].clone(), 0.25 * torch.ones(N, 3)
    r_obs = torch.full((N,), 0.01)              # informative observations

    def run_regime(target, episodes):
        recs = None
        for t in range(episodes):
            losses = [float((c.mu - target).norm()) for c in b.all_components()]
            winner = int(b.responsibilities(losses).argmax())
            cur = b.all_components()[winner].mu
            if recs is None and float((cur - target).norm()) < 1.0:
                recs = t
            m = cur + 0.8 * (target - cur)      # refine toward target
            b.step(losses, m, r_obs, t=t)
        return episodes if recs is None else recs

    run_regime(tA, 10)
    first_B = run_regime(tB, 30)                # novel: must learn/birth
    run_regime(tA, 10)
    revisit_B = run_regime(tB, 30)
    assert revisit_B < first_B


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
