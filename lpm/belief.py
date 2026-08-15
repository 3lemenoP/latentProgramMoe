"""lpm/belief.py — whitened tangent Gaussian belief over a program field
(phase-4 spec §2).

Per site i (one quaternion), belief over the whitened rotation vector
ṽ_i ~ N(μ_i, σ_i²·I₃), stored in NATURAL parameters (λ_i = 1/σ_i²,
η_i = λ_i·μ_i) so that fusion is exactly additive and order-invariant (T16).
Isotropy per site is preserved by every specified operation (isotropic
prior, isotropic drift Q, isotropic observation covariance R = r²·I).

The deployed program is the MAP field exp(unwhiten(μ)). Potency = entropy;
the differentiation observable (spec §2.1).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from .field import ProgramField
from .quaternion import q_angle2, q_normalize
from .whitening import Whitening


class WhitenedGaussianBelief:
    def __init__(self, whitening: Whitening, sigma0: float):
        self.w = whitening
        N = whitening.n_quats
        self.lam = torch.full((N,), 1.0 / (sigma0 ** 2))   # precision
        self.eta = torch.zeros(N, 3)                       # λ·μ
        self.sigma0 = sigma0

    # -- moments -------------------------------------------------------------
    @property
    def mu(self) -> torch.Tensor:
        return self.eta / self.lam.unsqueeze(-1)

    @property
    def var(self) -> torch.Tensor:
        return 1.0 / self.lam

    # -- ops (spec §2.1) -----------------------------------------------------
    def predict(self, q_drift: float = 0.0, lam_forget: Optional[float] = None):
        """Σ ← Σ + Q (default) or precision forgetting Λ ← λΛ (flag)."""
        if lam_forget is not None:
            self.eta = self.eta * lam_forget
            self.lam = self.lam * lam_forget
        elif q_drift > 0.0:
            mu = self.mu
            self.lam = 1.0 / (1.0 / self.lam + q_drift ** 2)
            self.eta = mu * self.lam.unsqueeze(-1)
        return self

    def fuse(self, m: torch.Tensor, r_obs: float):
        """Evidence (mean m whitened (N,3), isotropic precision r_obs⁻²):
        Λ ← Λ + R, Λμ ← Λμ + Rm. Additive and commutative (T16)."""
        R = 1.0 / (r_obs ** 2)
        self.lam = self.lam + R
        self.eta = self.eta + R * m
        return self

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        std = self.var.sqrt().unsqueeze(-1)
        return self.mu + std * torch.randn(*self.eta.shape, generator=generator)

    def sample_field(self, generator: Optional[torch.Generator] = None) -> ProgramField:
        return self.w.whitened_to_field(self.sample(generator))

    def map_field(self) -> ProgramField:
        return self.w.whitened_to_field(self.mu)

    # -- observables ---------------------------------------------------------
    def potency(self) -> Dict[str, float]:
        """Per-group and total entropy: H_i = (3/2)·log(2πe·σ_i²)."""
        h = 1.5 * (math.log(2 * math.pi * math.e) - self.lam.log())
        out = {}
        for gi, name in enumerate(self.w.group_names):
            mask = self.w.group_ids == gi
            if mask.any():
                out[name] = float(h[mask].sum())
        out["total"] = float(h.sum())
        return out

    def commitment_map(self) -> Dict[str, float]:
        """Per-group mean activity s = sin²(θ/2) of the MAP field — comparable
        to a direct-fit skill's activity profile (spec D-3)."""
        f = self.map_field()
        out = {}
        for name in self.w.group_names:
            xs = []
            for (l, n) in self.w.spec.site_keys():
                if n == name:
                    xs.append(q_angle2(q_normalize(f.q(l, n).float())).reshape(-1))
            out[name] = float(torch.cat(xs).mean())
        return out

    # -- persistence ---------------------------------------------------------
    def state_dict(self) -> Dict:
        return {"lam": self.lam.clone(), "eta": self.eta.clone(),
                "sigma0": self.sigma0}

    def load_state_dict(self, sd: Dict):
        self.lam = sd["lam"].clone()
        self.eta = sd["eta"].clone()
        self.sigma0 = sd["sigma0"]
        return self
