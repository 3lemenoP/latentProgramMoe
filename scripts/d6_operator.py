#!/usr/bin/env python3
"""D6 — learned composition operator: semantics vs coordinates
(e-suite-analysis-e3prime.md §4.6; handoff §4 P5).

Library: 5 atoms a..e (prepend, reverse, append, rotl, swap2). Oracle-fit all
20 ordered pipelines "<x>_then_<y>" (skip-if-exists), then train two operators
on the triples (z_x, z_y, z_oracle) and evaluate on held-out pipelines:

- Constrained (gauged product): C(y, x) = g1 ⊗ y ⊗ g2 ⊗ x ⊗ g3, three learned
  gauge fields; reduces to plain Hamilton at g = id.
- Unconstrained: shared MLP on concatenated site quaternions → raw 4-vector,
  normalized.

Loss is sign-invariant chordal d2_chord to the oracle. Evaluation is both
geometric (d_geo to oracle) and behavioral (exact match on the pipeline,
which is the number that actually decides the tree branch).

    python scripts/d6_operator.py --dir runs/e3strong --fit-steps 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lpm import LatentProgramModel, ProgramField  # noqa: E402
from lpm.quaternion import d2_chord, d_geo, hamilton, q_normalize  # noqa: E402
from lpm.tasks import E3Vocab  # noqa: E402
from lpm.utils import get_device, set_seed  # noqa: E402
from e3_common import fit_field, load_base, ordered_accuracy  # noqa: E402

ATOM_BEHAVIOR = {"a": "prepend", "b": "reverse", "c": "append",
                 "d": "rotl", "e": "swap2"}
ATOMS = "abcde"
PIPES = [f"{x}_then_{y}" for x in ATOMS for y in ATOMS if x != y]
# held out so every atom appears in training triples in both slots
HELDOUT = ["d_then_b", "c_then_a", "e_then_d", "a_then_e"]


def field_path(d: Path, name: str) -> Path:
    if "_then_" in name:
        x, y = name.split("_then_")
        return d / f"z_{x}{y}.pt"
    return d / f"z_{name}.pt"


def get_or_fit(model, name, behavior, vocab, device, d: Path, steps, seed):
    p = field_path(d, name)
    if p.exists():
        f = ProgramField.load(str(p), trainable=False).to(device)
        return f, None
    f = ProgramField.randn_near_identity(model.spec, sigma=1e-3,
                                         trainable=True).to(device)
    f = fit_field(model, f, behavior, vocab, device, steps=steps,
                  seed=seed, tag=name)
    f.save(str(p))
    return f, "fitted"


def flatten_q(f: ProgramField) -> torch.Tensor:
    qs = [q_normalize(f.q(*k).detach().float()).reshape(-1, 4)
          for k in f.spec.site_keys()]
    return torch.cat(qs)


def unflatten_to_field(q: torch.Tensor, spec) -> ProgramField:
    quats, i = {}, 0
    for k in spec.site_keys():
        shape = (*spec.site_shape(k[1]), 4)
        n = 1
        for s in spec.site_shape(k[1]):
            n *= s
        quats[k] = q[i:i + n].reshape(shape)
        i += n
    return ProgramField(spec, quats, trainable=False)


def canon(q: torch.Tensor) -> torch.Tensor:
    return torch.where(q[..., :1] < 0, -q, q)


class GaugedProduct(nn.Module):
    """C(y, x) = g1 ⊗ y ⊗ g2 ⊗ x ⊗ g3 with per-site gauge quaternions."""

    def __init__(self, n_sites: int):
        super().__init__()
        def ident():
            t = torch.zeros(n_sites, 4)
            t[:, 0] = 1.0
            return nn.Parameter(t + 1e-3 * torch.randn(n_sites, 4))
        self.g1, self.g2, self.g3 = ident(), ident(), ident()

    def forward(self, qx, qy):
        g1 = q_normalize(self.g1)
        g2 = q_normalize(self.g2)
        g3 = q_normalize(self.g3)
        return hamilton(g1, hamilton(qy, hamilton(g2, hamilton(qx, g3))))


class MLPOperator(nn.Module):
    """PRIMARY unconstrained model (A3.2): per-site weight-shared MLP on
    concat(q_x, q_y) [+ site-group embedding] → raw 4-vector, normalized.
    Site-locality is the right inductive bias and turns 16 pipelines × ~900
    sites into a real dataset."""

    def __init__(self, hidden: int = 128, n_groups: int = 4, group_dim: int = 8):
        super().__init__()
        self.group_emb = nn.Embedding(n_groups, group_dim)
        self.net = nn.Sequential(
            nn.Linear(8 + group_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 4))

    def forward(self, qx, qy, group_ids):
        emb = self.group_emb(group_ids)
        out = self.net(torch.cat([canon(qx), canon(qy), emb], dim=-1))
        return q_normalize(out)


class FlattenedMLP(nn.Module):
    """Overfitting CONTROL (A3.2, demoted from verdict-bearing): one MLP over
    the whole flattened field pair — no site-locality bias, 16 samples."""

    def __init__(self, n_sites: int, hidden: int = 512):
        super().__init__()
        self.n_sites = n_sites
        self.net = nn.Sequential(
            nn.Linear(8 * n_sites, hidden), nn.SiLU(),
            nn.Linear(hidden, 4 * n_sites))

    def forward(self, qx, qy):
        flat = torch.cat([canon(qx), canon(qy)], dim=-1).reshape(-1)
        out = self.net(flat).reshape(self.n_sites, 4)
        return q_normalize(out)


def train_operator(op, call, triples, epochs, lr, tag):
    """`call(op, qx, qy)` abstracts the per-op signature."""
    opt = torch.optim.Adam(op.parameters(), lr=lr)
    for ep in range(epochs):
        loss = 0.0
        opt.zero_grad()
        for qx, qy, qt in triples:
            pred = call(op, qx, qy)
            loss = loss + d2_chord(pred, qt).mean()
        loss = loss / len(triples)
        loss.backward()
        opt.step()
        if ep % 500 == 0:
            print(f"[{tag}] epoch {ep} chordal loss {loss.item():.6f}")
    print(f"[{tag}] final chordal loss {loss.item():.6f}")
    return op


def site_group_ids(spec) -> torch.Tensor:
    """Group index per flattened quaternion, matching flatten_q's order."""
    from lpm.field import SITE_NAMES
    gid = {n: i for i, n in enumerate(SITE_NAMES)}
    out = []
    for (l, n) in spec.site_keys():
        cnt = 1
        for s in spec.site_shape(n):
            cnt *= s
        out.extend([gid[n]] * cnt)
    return torch.tensor(out, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/e3strong")
    ap.add_argument("--fit-steps", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    d = Path(args.dir)
    vocab = E3Vocab()
    base = load_base(str(d / "base.pt"), device)
    model = LatentProgramModel(base)
    spec = model.spec

    # ---- phase 1: atoms + pipeline oracles (skip-if-exists) ---------------
    atoms, oracle, oracle_exact = {}, {}, {}
    for i, letter in enumerate(ATOMS):
        f, fitted = get_or_fit(model, letter, ATOM_BEHAVIOR[letter], vocab,
                               device, d, args.fit_steps, args.seed + 50 + i)
        atoms[letter] = f
        e = ordered_accuracy(model, f, ATOM_BEHAVIOR[letter], vocab, device)
        print(f"atom {letter} ({ATOM_BEHAVIOR[letter]}): exact {e[0]:.3f}"
              + (" [fitted now]" if fitted else " [loaded]"))
    for i, pipe in enumerate(PIPES):
        f, fitted = get_or_fit(model, pipe, pipe, vocab, device, d,
                               args.fit_steps, args.seed + 100 + i)
        oracle[pipe] = f
        e = ordered_accuracy(model, f, pipe, vocab, device)
        oracle_exact[pipe] = e[0]
        print(f"oracle {pipe}: exact {e[0]:.3f}"
              + (" [fitted now]" if fitted else " [loaded]"))

    # ---- phase 2: operator training on triples ----------------------------
    flat_atoms = {k: flatten_q(f) for k, f in atoms.items()}
    train_pipes = [p for p in PIPES if p not in HELDOUT]
    def triple(pipe):
        x, y = pipe.split("_then_")
        return flat_atoms[x], flat_atoms[y], flatten_q(oracle[pipe])
    train_triples = [triple(p) for p in train_pipes]

    n_sites = next(iter(flat_atoms.values())).shape[0]
    gids = site_group_ids(spec).to(device)
    call_gauged = lambda op, qx, qy: op(qx, qy)          # noqa: E731
    call_persite = lambda op, qx, qy: op(qx, qy, gids)   # noqa: E731
    call_flat = lambda op, qx, qy: op(qx, qy)            # noqa: E731

    gauged = train_operator(GaugedProduct(n_sites).to(device), call_gauged,
                            train_triples, args.epochs, 1e-2, "gauged")
    # A3.4 learning curve for the PRIMARY per-site operator: 8, 12, 16 pipes
    persite_by_n = {}
    for n_train in (8, 12, 16):
        op = train_operator(MLPOperator().to(device), call_persite,
                            train_triples[:n_train], args.epochs, 1e-3,
                            f"persite@{n_train}")
        persite_by_n[n_train] = op
    persite = persite_by_n[16]
    flat_ctrl = train_operator(FlattenedMLP(n_sites).to(device), call_flat,
                               train_triples, args.epochs, 1e-3, "flat-ctrl")

    # ---- phase 3: held-out evaluation (A3.1: task exact is PRIMARY) --------
    report = {"oracle_exact": oracle_exact, "heldout": {}}
    for pipe in HELDOUT:
        x, y = pipe.split("_then_")
        qx, qy = flat_atoms[x], flat_atoms[y]
        qt = flatten_q(oracle[pipe])
        with torch.no_grad():
            cands = {
                "hamilton": hamilton(qy, qx),
                "gauged": call_gauged(gauged, qx, qy),
                "persite_mlp": call_persite(persite, qx, qy),
                "flat_mlp_control": call_flat(flat_ctrl, qx, qy),
            }
        weak = oracle_exact[pipe] < 0.9
        row = {"oracle_exact": oracle_exact[pipe], "weak_oracle": weak}
        for name, q in cands.items():
            f = unflatten_to_field(q.detach(), spec).to(device)
            e = ordered_accuracy(model, f, pipe, vocab, device)
            row[name] = {"exact": e[0], "token": e[1],
                         "d_geo_to_oracle": float(d_geo(q, qt).mean()),
                         "d_geo_flag": "weak-oracle" if weak else "ok"}
        report["heldout"][pipe] = row
        print(pipe, json.dumps(row))

    # learning curve on held-out (task exact, per training-set size)
    curve = {}
    for n_train, op in persite_by_n.items():
        accs = []
        for pipe in HELDOUT:
            x, y = pipe.split("_then_")
            with torch.no_grad():
                q = call_persite(op, flat_atoms[x], flat_atoms[y])
            f = unflatten_to_field(q.detach(), spec).to(device)
            accs.append(ordered_accuracy(model, f, pipe, vocab, device)[0])
        curve[n_train] = sum(accs) / len(accs)
    report["persite_learning_curve"] = curve
    print(f"per-site learning curve (heldout mean exact): {curve}")

    # verdict per companion doc §4.6 tree + A3 controls
    def mean_exact(name):
        return sum(r[name]["exact"] for r in report["heldout"].values()) / len(HELDOUT)
    g, m = mean_exact("gauged"), mean_exact("persite_mlp")
    h, fl = mean_exact("hamilton"), mean_exact("flat_mlp_control")
    report["mean_heldout_exact"] = {"hamilton": h, "gauged": g,
                                    "persite_mlp": m, "flat_mlp_control": fl}
    flat_curve = max(curve.values()) - min(curve.values()) < 0.05
    if g >= 0.5:
        verdict = "gauged product works — ship compose_learned; algebra survives gauged"
    elif m >= 0.5:
        verdict = "only MLP works — group = coordinates, not semantics"
    elif flat_curve:
        verdict = ("both fail with FLAT per-site learning curve — meaningful "
                   "both-fail; retire composition, substrate claims stand")
    else:
        verdict = ("both fail but per-site curve still rising — sample-size "
                   "limited, extend pipeline library before concluding")
    report["verdict"] = verdict
    (d / "report_d6.json").write_text(json.dumps(report, indent=2))
    print(f"\nD6 mean held-out exact: hamilton {h:.3f}  gauged {g:.3f}  "
          f"persite {m:.3f}  flat-ctrl {fl:.3f}")
    print(f"D6 verdict: {verdict}")


if __name__ == "__main__":
    main()
