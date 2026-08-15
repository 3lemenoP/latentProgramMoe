"""lpm/whitening.py — whitened tangent coordinates (phase-4 spec §2.3/§3).

Geometric proximity is not behavioral proximity (report §5.3, four
sightings): every distance, prior, covariance, or cluster downstream of this
module lives in whitened coordinates ṽ_i = c_{g(i)}·v_i, with per-group
c_g² = L_g / (4·n_g) so that Σ_i‖ṽ_i‖² ≈ KL-to-base under linear-response
additivity. L_g comes from a W0 calibration run (scripts/calibrate_whitening.py)
stored as whitening.json; n_g is the total quaternion count of the group.
Only relative scale across groups carries meaning.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch

from .field import FieldSpec, ProgramField
from .quaternion import q_exp, q_log, q_normalize


def group_sizes(spec: FieldSpec) -> Dict[str, int]:
    """Total quaternion count per site group across layers."""
    out: Dict[str, int] = {}
    for (l, n) in spec.site_keys():
        cnt = 1
        for s in spec.site_shape(n):
            cnt *= s
        out[n] = out.get(n, 0) + cnt
    return out


class Whitening:
    """Per-group scale factors c_g plus flatten/unflatten between ProgramField
    and the (N, 3) whitened tangent representation (site order = site_keys)."""

    def __init__(self, spec: FieldSpec, leverage: Dict[str, float],
                 meta: Dict | None = None):
        self.spec = spec
        self.leverage = dict(leverage)
        self.meta = dict(meta or {})
        sizes = group_sizes(spec)
        self.c: Dict[str, float] = {
            g: (leverage[g] / (4.0 * sizes[g])) ** 0.5 for g in sizes}
        # flat layout
        self._slices = []
        i = 0
        for (l, n) in spec.site_keys():
            cnt = 1
            for s in spec.site_shape(n):
                cnt *= s
            self._slices.append(((l, n), slice(i, i + cnt)))
            i += cnt
        self.n_quats = i
        cvec = torch.empty(self.n_quats)
        gid = []
        names = list(spec.site_names())
        for (l, n), sl in self._slices:
            cvec[sl] = self.c[n]
            gid.extend([names.index(n)] * (sl.stop - sl.start))
        self.c_vec = cvec                        # (N,)
        self.group_ids = torch.tensor(gid)       # (N,)
        self.group_names = names

    # -- field <-> whitened tangent -----------------------------------------
    def field_to_whitened(self, field: ProgramField) -> torch.Tensor:
        vs = []
        for (l, n), _sl in self._slices:
            q = q_normalize(field.q(l, n).detach().float()).reshape(-1, 4)
            vs.append(q_log(q) * self.c[n])
        return torch.cat(vs)                     # (N, 3)

    def whitened_to_field(self, v: torch.Tensor) -> ProgramField:
        quats = {}
        for (l, n), sl in self._slices:
            vraw = v[sl] / self.c[n]
            quats[(l, n)] = q_exp(vraw).reshape(*self.spec.site_shape(n), 4)
        return ProgramField(self.spec, quats, trainable=False)

    def whitened_to_quats_graph(self, v: torch.Tensor) -> Dict:
        """Graph-connected quats dict (for whitened-parameterization refine)."""
        quats = {}
        cdev = self.c_vec.to(v.device)
        for (l, n), sl in self._slices:
            vraw = v[sl] / cdev[sl].unsqueeze(-1)
            quats[(l, n)] = q_exp(vraw).reshape(*self.spec.site_shape(n), 4)
        return quats

    # -- persistence ---------------------------------------------------------
    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps({
            "leverage": self.leverage,
            "group_sizes": group_sizes(self.spec),
            "c": self.c, "meta": self.meta}, indent=2))

    @classmethod
    def load(cls, path: str, spec: FieldSpec) -> "Whitening":
        blob = json.loads(Path(path).read_text())
        return cls(spec, blob["leverage"], blob.get("meta"))
