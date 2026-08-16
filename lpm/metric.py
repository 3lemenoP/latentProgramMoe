"""lpm/metric.py — the learned behavioral metric M (phase-4 v2 spec §1).

Constraint 0: no decision-bearing quantity may be manifold-native. This
module learns the distance that matters from behavior: PCA the fitted
library's tangent vectors to the live subspace, fit nonnegative diagonal
weights in PCA coordinates so weighted squared distances match measured
symmetrized KL, and price out-of-subspace (dead-sea) directions with the
operating-strength random-calibrated W0 weights — random calibration is
wrong for skills precisely because skills live in the subspace; for the
dead sea it is the correct price.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch


def fit_diag_weights(Z: torch.Tensor, K: torch.Tensor, ridge: float = 1e-3,
                     iters: int = 2000, lr: float = 0.05,
                     pair_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Nonnegative diagonal weights w in PCA coords: minimize
    Σ_ij (Σ_c w_c ΔZ² − K_ij)² + ridge·‖w‖². Projected gradient (w ≥ 0)."""
    n, r = Z.shape
    iu = torch.triu_indices(n, n, offset=1)
    F = (Z[iu[0]] - Z[iu[1]]) ** 2            # (P, r)
    y = K[iu[0], iu[1]]                       # (P,)
    if pair_mask is not None:
        keep = pair_mask[iu[0]] & pair_mask[iu[1]]
        F, y = F[keep], y[keep]
    # scale-init: uniform w matching mean magnitudes
    w = torch.full((r,), float((y.mean() / F.sum(dim=1).mean()).clamp_min(1e-8)))
    w = w.clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(iters):
        pred = F @ w.clamp_min(0.0)
        loss = ((pred - y) ** 2).mean() + ridge * (w ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach().clamp_min(0.0)


class MetricM:
    """d_M(a, b)² = Σ_c w_c·(z_a − z_b)_c² + Σ_i c_{g(i)}²·‖r_i‖²  where z are
    PCA coordinates of the flattened tangent vectors and r the out-of-subspace
    residual (priced at W0 whitening weights c, per-site)."""

    def __init__(self, mean: torch.Tensor, basis: torch.Tensor,
                 w: torch.Tensor, c_vec: torch.Tensor, meta: Dict):
        self.mean = mean          # (D,)
        self.basis = basis        # (D, r) orthonormal columns
        self.w = w                # (r,)
        self.c_vec = c_vec        # (N_sites,) — D = 3·N_sites
        self.meta = dict(meta)

    # -- construction --------------------------------------------------------
    @classmethod
    def fit(cls, V: torch.Tensor, symKL: torch.Tensor, c_vec: torch.Tensor,
            var_keep: float = 0.95, ridge: float = 1e-3,
            meta: Optional[Dict] = None) -> "MetricM":
        """V: (n_fields, D) flattened RAW tangent vectors (q_log, site order).
        symKL: (n, n). c_vec: per-site W0 whitening scales (dead-sea price)."""
        mean = V.mean(dim=0)
        Xc = V - mean
        U, S, _ = torch.linalg.svd(Xc.T @ Xc)
        var = S / S.sum().clamp_min(1e-12)
        r = int((var.cumsum(0) < var_keep).sum().item()) + 1
        r = min(r, V.shape[0] - 1)
        basis = U[:, :r]                       # (D, r)
        Z = Xc @ basis
        w = fit_diag_weights(Z, symKL, ridge=ridge)
        return cls(mean, basis, w, c_vec,
                   {"var_keep": var_keep, "rank": r, "ridge": ridge,
                    "n_fields": int(V.shape[0]), **(meta or {})})

    # -- distances -----------------------------------------------------------
    def project(self, v: torch.Tensor):
        """v: (D,) → (pca coords (r,), residual (D,))."""
        x = v - self.mean
        z = x @ self.basis
        resid = x - self.basis @ z
        return z, resid

    def _resid_price(self, resid: torch.Tensor) -> torch.Tensor:
        r3 = resid.reshape(-1, 3)
        return ((self.c_vec ** 2) * (r3 ** 2).sum(-1)).sum()

    def dist2(self, va: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
        za, ra = self.project(va)
        zb, rb = self.project(vb)
        live = (self.w * (za - zb) ** 2).sum()
        dead = self._resid_price(ra - rb)
        return live + dead

    def pdist2(self, V: torch.Tensor) -> torch.Tensor:
        """Pairwise squared distances for (n, D)."""
        n = V.shape[0]
        out = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                d = self.dist2(V[i], V[j])
                out[i, j] = out[j, i] = d
        return out

    # -- persistence ---------------------------------------------------------
    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps({
            "mean": self.mean.tolist(), "basis": self.basis.tolist(),
            "w": self.w.tolist(), "c_vec": self.c_vec.tolist(),
            "meta": self.meta}, indent=None))

    @classmethod
    def load(cls, path: str) -> "MetricM":
        b = json.loads(Path(path).read_text())
        return cls(torch.tensor(b["mean"]), torch.tensor(b["basis"]),
                   torch.tensor(b["w"]), torch.tensor(b["c_vec"]), b["meta"])


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Rank correlation (no scipy dependency)."""
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm()).clamp_min(1e-12))


def loo_stress(V: torch.Tensor, symKL: torch.Tensor, c_vec: torch.Tensor,
               var_keep: float = 0.95, ridge: float = 1e-3) -> Dict:
    """T22/G1: leave-one-out — refit weights without field i (shared basis),
    Spearman-align d²_M(i, ·) with symKL(i, ·); compare against raw-L2 and
    random-whitened baselines on the same held-out rows."""
    n = V.shape[0]
    full = MetricM.fit(V, symKL, c_vec, var_keep=var_keep, ridge=ridge)
    Z = (V - full.mean) @ full.basis
    rho_m, rho_raw, rho_white = [], [], []
    white = V.reshape(n, -1, 3) * c_vec.unsqueeze(0).unsqueeze(-1)
    white = white.reshape(n, -1)
    for i in range(n):
        mask = torch.ones(n, dtype=torch.bool)
        mask[i] = False
        w_i = fit_diag_weights(Z, symKL, ridge=ridge, pair_mask=mask)
        m_i = MetricM(full.mean, full.basis, w_i, c_vec, full.meta)
        others = [j for j in range(n) if j != i]
        dm = torch.tensor([float(m_i.dist2(V[i], V[j])) for j in others])
        dr = torch.tensor([float(((V[i] - V[j]) ** 2).sum()) for j in others])
        dw = torch.tensor([float(((white[i] - white[j]) ** 2).sum())
                           for j in others])
        y = symKL[i][others]
        rho_m.append(spearman(dm, y))
        rho_raw.append(spearman(dr, y))
        rho_white.append(spearman(dw, y))
    mm = sum(rho_m) / n
    mr = sum(rho_raw) / n
    mw = sum(rho_white) / n
    return {"spearman_M": mm, "spearman_raw": mr, "spearman_whitened": mw,
            "G1_pass": mm > mr and mm > mw}
