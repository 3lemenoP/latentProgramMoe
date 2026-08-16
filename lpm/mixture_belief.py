"""lpm/mixture_belief.py — IMM mixture belief, the codebook as memory
(phase-4 v2 spec §3).

K named components + 1 novelty component, each a per-site Gaussian in RAW
rotation-vector coordinates (v1 engineering lessons binding: raw geometry,
per-site angle caps). The discrete regime decision is BEHAVIORAL by
Constraint 0: responsibilities come from measured per-component probe
losses through a sticky softmax — never from geometric likelihood. Fusion
goes into the winner only (T27); the novelty component births into a named
component after a sustained take-over with falling loss (T26).

The class is model-agnostic: callers measure losses and produce refined
observations; the belief handles responsibilities, fusion, drift, birth,
and the per-component potency/commitment observables.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch


class GaussianComponent:
    def __init__(self, name: str, mu: torch.Tensor, sigma: torch.Tensor,
                 born_at: int = 0):
        self.name = name
        self.mu = mu.clone()                       # (N, 3) raw rotation vectors
        self.lam = 1.0 / (sigma.clone() ** 2)      # (N,) isotropic precision
        self.born_at = born_at

    def fuse(self, m: torch.Tensor, r_obs: torch.Tensor, weight: float = 1.0):
        R = weight / (r_obs ** 2)
        eta = self.lam.unsqueeze(-1) * self.mu + R.unsqueeze(-1) * m
        self.lam = self.lam + R
        self.mu = eta / self.lam.unsqueeze(-1)

    def predict(self, q_drift: torch.Tensor):
        self.lam = 1.0 / (1.0 / self.lam + q_drift ** 2)

    def entropy(self) -> torch.Tensor:
        return 1.5 * (math.log(2 * math.pi * math.e) - self.lam.log())


class IMMBelief:
    def __init__(self, means: List[torch.Tensor], names: List[str],
                 group_ids: torch.Tensor, group_names: List[str],
                 sigma_init: float = 0.05, sigma_novelty: float = 0.3,
                 beta: float = 5.0, p_stay: float = 0.9, tau: float = 0.7,
                 t_birth: int = 10):
        """means: K raw (N,3) tensors (codebook medoids). Per-site sigmas are
        RAW angles / √3 per dim, capped at 0.3 rad."""
        N = means[0].shape[0]
        s_init = torch.full((N,), min(sigma_init, 0.3) / (3 ** 0.5))
        self.s_novel = torch.full((N,), min(sigma_novelty, 0.3) / (3 ** 0.5))
        self.components = [GaussianComponent(n, m, s_init)
                           for n, m in zip(names, means)]
        self.novelty = GaussianComponent("novelty", torch.zeros(N, 3),
                                         self.s_novel)
        self.group_ids = group_ids
        self.group_names = list(group_names)
        self.beta = beta
        self.p_stay = p_stay
        self.tau = tau
        self.t_birth = t_birth
        self.prev_winner: Optional[int] = None
        self._novel_streak = 0
        self._novel_loss_start: Optional[float] = None
        self._novel_loss_last: Optional[float] = None
        self.births: List[Dict] = []

    # -- accessors -----------------------------------------------------------
    def all_components(self) -> List[GaussianComponent]:
        return self.components + [self.novelty]

    @property
    def k(self) -> int:
        return len(self.components) + 1

    def maps(self) -> List[torch.Tensor]:
        return [c.mu for c in self.all_components()]

    # -- IMM step ------------------------------------------------------------
    def responsibilities(self, losses: List[float]) -> torch.Tensor:
        """Behavioral responsibilities: softmax(−β·loss) × sticky prior."""
        losses = torch.tensor(losses, dtype=torch.float64)
        logp = -self.beta * losses
        prior = torch.full((self.k,), (1.0 - self.p_stay) / max(self.k - 1, 1),
                           dtype=torch.float64)
        if self.prev_winner is not None:
            prior[self.prev_winner] = self.p_stay
        else:
            prior[:] = 1.0 / self.k
        logit = logp + prior.log()
        r = (logit - logit.logsumexp(0)).exp()
        return r.float()

    def step(self, losses: List[float], m: torch.Tensor,
             r_obs: torch.Tensor, q_drift: Optional[torch.Tensor] = None,
             t: int = 0) -> Dict:
        """One IMM update: responsibilities from measured losses; fuse the
        refined observation m into the winner (hard if top resp > τ, else
        responsibility-weighted across components — never component-into-
        component); drift; run the birth rule. Returns step info."""
        r = self.responsibilities(losses)
        winner = int(r.argmax())
        comps = self.all_components()
        if q_drift is not None:
            for c in comps:
                c.predict(q_drift)
        if float(r[winner]) > self.tau:
            comps[winner].fuse(m, r_obs)
        else:
            for k, c in enumerate(comps):
                if float(r[k]) > 1e-3:
                    c.fuse(m, r_obs, weight=float(r[k]))
        birth = self._birth_rule(winner, float(losses[winner]), t)
        self.prev_winner = winner if not birth else len(self.components) - 1
        return {"resp": r.tolist(), "winner": winner,
                "winner_name": comps[winner].name, "birth": birth}

    def _birth_rule(self, winner: int, winner_loss: float, t: int) -> bool:
        novel_idx = self.k - 1
        if winner != novel_idx:
            self._novel_streak = 0
            self._novel_loss_start = None
            return False
        if self._novel_streak == 0:
            self._novel_loss_start = winner_loss
        self._novel_streak += 1
        self._novel_loss_last = winner_loss
        if (self._novel_streak >= self.t_birth
                and winner_loss < self._novel_loss_start):
            born = self.novelty
            born.name = f"born_t{t}"
            born.born_at = t
            self.components.append(born)
            N = born.mu.shape[0]
            self.novelty = GaussianComponent("novelty", torch.zeros(N, 3),
                                             self.s_novel)
            self.births.append({"t": t, "name": born.name})
            self._novel_streak = 0
            self._novel_loss_start = None
            return True
        return False

    # -- observables ---------------------------------------------------------
    def potency(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for c in self.all_components():
            h = c.entropy()
            row = {"total": float(h.sum())}
            for gi, name in enumerate(self.group_names):
                mask = self.group_ids == gi
                if mask.any():
                    row[name] = float(h[mask].sum())
            out[c.name] = row
        return out

    def commitment_raw_activity(self) -> Dict[str, Dict[str, float]]:
        """Per-component per-group mean activity s = sin²(θ/2) of the MAP."""
        out = {}
        for c in self.all_components():
            theta = c.mu.norm(dim=-1)
            s = torch.sin(theta / 2.0) ** 2
            row = {}
            for gi, name in enumerate(self.group_names):
                mask = self.group_ids == gi
                if mask.any():
                    row[name] = float(s[mask].mean())
            out[c.name] = row
        return out
