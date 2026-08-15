#!/usr/bin/env python3
"""E3′ D2–D5 runner (e-suite-analysis-e3prime.md §4).

    python scripts/e3prime.py --stage d2 --dir runs/e3
    python scripts/e3prime.py --stage d3 --dir runs/e3
    python scripts/e3prime.py --stage d4 --dir runs/e3
    python scripts/e3prime.py --stage d5 --dir runs/e3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lpm import (AbelianProgramField, AxisBank, LatentProgramModel,  # noqa: E402
                 ProgramField, compose, increment, invert_field, slerp_field)
from lpm.quaternion import d_geo, q_normalize  # noqa: E402
from lpm.tasks import E3Vocab  # noqa: E402
from lpm.utils import get_device, set_seed  # noqa: E402
from e3_common import (  # noqa: E402
    fit_field, fit_fields_joint, load_base, ordered_accuracy,
)


def _load_wrap(d: Path, device):
    base = load_base(str(d / "base.pt"), device)
    return LatentProgramModel(base)


def _mean_dgeo(fa: ProgramField, fb: ProgramField) -> float:
    xs = []
    for k in fa.spec.site_keys():
        xs.append(d_geo(q_normalize(fa.q(*k).float()),
                        q_normalize(fb.q(*k).float())).mean())
    return float(torch.stack(xs).mean())


def stage_d2(args, model, vocab, device, d: Path):
    za = ProgramField.load(str(d / "z_a.pt"), trainable=False).to(device)
    zb = ProgramField.load(str(d / "z_b.pt"), trainable=False).to(device)
    print("=== D2 oracle z_ab / z_ba ===")
    zab = ProgramField.randn_near_identity(model.spec, sigma=1e-3, trainable=True).to(device)
    zba = ProgramField.randn_near_identity(model.spec, sigma=1e-3, trainable=True).to(device)
    zab = fit_field(model, zab, "a_then_b", vocab, device, steps=args.fit_steps,
                    seed=args.seed + 10, tag="z_ab")
    zba = fit_field(model, zba, "b_then_a", vocab, device, steps=args.fit_steps,
                    seed=args.seed + 11, tag="z_ba")
    zab.save(str(d / "z_ab.pt"))
    zba.save(str(d / "z_ba.pt"))
    ident = ProgramField.identity(model.spec)
    ham = compose(zb, za)
    e_ab = ordered_accuracy(model, zab, "a_then_b", vocab, device)
    e_ba = ordered_accuracy(model, zba, "b_then_a", vocab, device)
    report = {
        "z_ab_exact": e_ab[0], "z_ab_token": e_ab[1],
        "z_ba_exact": e_ba[0], "z_ba_token": e_ba[1],
        "d_geo(hamilton(b,a), z_ab)": _mean_dgeo(ham, zab),
        "d_geo(z_a, z_ab)": _mean_dgeo(za, zab),
        "d_geo(z_b, z_ab)": _mean_dgeo(zb, zab),
        "d_geo(id, z_ab)": _mean_dgeo(ident, zab),
        "oracle_pass": e_ab[0] >= 0.9 and e_ba[0] >= 0.9,
    }
    (d / "report_d2.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["oracle_pass"]:
        print("D2 FAIL: single program cannot represent the pipeline — pause composition.")
    elif report["d_geo(hamilton(b,a), z_ab)"] < min(report["d_geo(z_a, z_ab)"],
                                                    report["d_geo(z_b, z_ab)"]):
        print("D2: Hamilton is nearer z_ab than either atom — algebra is in the ballpark.")
    else:
        print("D2: Hamilton is far from z_ab — proceed to D3/D6 (gauge story).")


def stage_d3(args, model, vocab, device, d: Path):
    print("=== D3 relative increments / transport ===")
    # include append in the frozen base? the saved base was trained without
    # append. Fit z_c / z_cb on the same base anyway (capacity control).
    za = ProgramField.load(str(d / "z_a.pt")).to(device)
    zab_path = d / "z_ab.pt"
    if not zab_path.exists():
        raise SystemExit("run --stage d2 first (need z_ab.pt)")
    zab = ProgramField.load(str(zab_path)).to(device)
    zc = ProgramField.randn_near_identity(model.spec, sigma=1e-3, trainable=True).to(device)
    zcb = ProgramField.randn_near_identity(model.spec, sigma=1e-3, trainable=True).to(device)
    zc = fit_field(model, zc, "append", vocab, device, steps=args.fit_steps,
                   seed=args.seed + 20, tag="z_c")
    zcb = fit_field(model, zcb, "c_then_b", vocab, device, steps=args.fit_steps,
                    seed=args.seed + 21, tag="z_cb")
    zc.save(str(d / "z_c.pt"))
    zcb.save(str(d / "z_cb.pt"))
    delta = increment(zab, za)  # b in the context of a
    left = compose(delta, zc)
    g = compose(zc, invert_field(za))
    adj = compose(compose(compose(g, delta), invert_field(g)), zc)
    e_c = ordered_accuracy(model, zc, "append", vocab, device)
    e_cb = ordered_accuracy(model, zcb, "c_then_b", vocab, device)
    e_left = ordered_accuracy(model, left, "c_then_b", vocab, device)
    e_adj = ordered_accuracy(model, adj, "c_then_b", vocab, device)
    report = {
        "z_c_append_exact": e_c[0],
        "z_cb_oracle_exact": e_cb[0],
        "left_transport_exact": e_left[0],
        "adjoint_transport_exact": e_adj[0],
        "d_geo(left, z_cb)": _mean_dgeo(left, zcb),
        "d_geo(adjoint, z_cb)": _mean_dgeo(adj, zcb),
        "transport_pass": max(e_left[0], e_adj[0]) >= 0.5,
    }
    (d / "report_d3.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def stage_d4(args, model, vocab, device, d: Path):
    print("=== D4 shared-support joint refit ===")
    za = ProgramField.randn_near_identity(model.spec, sigma=1e-3, trainable=True).to(device)
    zb = ProgramField.randn_near_identity(model.spec, sigma=1e-3, trainable=True).to(device)
    za, zb = fit_fields_joint(
        model, [(za, "prepend"), (zb, "reverse")], vocab, device,
        steps=args.fit_steps, seed=args.seed + 30,
        tag="joint overlap", overlap_coef=args.overlap, lambda_reg=0.0)
    za.save(str(d / "z_a_joint.pt"))
    zb.save(str(d / "z_b_joint.pt"))
    bank = AxisBank(model.spec).to(device)
    ta = AbelianProgramField(bank).to(device)
    tb = AbelianProgramField(bank).to(device)
    ta, tb = fit_fields_joint(
        model, [(ta, "prepend"), (tb, "reverse")], vocab, device,
        steps=args.fit_steps, seed=args.seed + 31,
        extra_params=bank.parameters(), tag="abelian joint", lambda_reg=0.0)
    ab = compose(zb, za)
    ba = compose(za, zb)
    tab = compose(tb.to_program_field(), ta.to_program_field())
    tba = compose(ta.to_program_field(), tb.to_program_field())
    rows = {
        "z_a prepend": ordered_accuracy(model, za, "prepend", vocab, device),
        "z_b reverse": ordered_accuracy(model, zb, "reverse", vocab, device),
        "compose(b,a) a_then_b": ordered_accuracy(model, ab, "a_then_b", vocab, device),
        "compose(a,b) b_then_a": ordered_accuracy(model, ba, "b_then_a", vocab, device),
        "compose(b,a) swapped": ordered_accuracy(model, ab, "b_then_a", vocab, device),
        "compose(a,b) swapped": ordered_accuracy(model, ba, "a_then_b", vocab, device),
        "abelian compose(b,a)": ordered_accuracy(model, tab, "a_then_b", vocab, device),
        "abelian compose(a,b)": ordered_accuracy(model, tba, "b_then_a", vocab, device),
    }
    nab = 0.5 * (rows["compose(b,a) a_then_b"][0] + rows["compose(a,b) b_then_a"][0])
    abel = 0.5 * (rows["abelian compose(b,a)"][0] + rows["abelian compose(a,b)"][0])
    margin = 100 * (nab - abel)
    report = {k: {"exact": v[0], "token": v[1]} for k, v in rows.items()}
    report["margin_points"] = margin
    report["d4_pass"] = margin > 20
    (d / "report_d4.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"D4 margin {margin:.1f} points  pass={margin > 20}")


def stage_d5(args, model, vocab, device, d: Path):
    print("=== D5 strength scan ===")
    za = ProgramField.load(str(d / "z_a.pt")).to(device)
    zb = ProgramField.load(str(d / "z_b.pt")).to(device)
    ident = ProgramField.identity(model.spec)
    rows = []
    for lam in (0.25, 0.5, 0.75, 1.0):
        za_l = slerp_field(ident, za, lam)
        zb_l = slerp_field(ident, zb, lam)
        ab = compose(zb_l, za_l)
        ba = compose(za_l, zb_l)
        e_ab = ordered_accuracy(model, ab, "a_then_b", vocab, device)
        e_ba = ordered_accuracy(model, ba, "b_then_a", vocab, device)
        e_sw = ordered_accuracy(model, ab, "b_then_a", vocab, device)
        rows.append({"lambda": lam, "exact_ab": e_ab[0], "exact_ba": e_ba[0],
                     "token_ab": e_ab[1], "token_ba": e_ba[1],
                     "swapped_ab_on_ba": e_sw[0]})
    (d / "report_d5.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["d2", "d3", "d4", "d5"])
    ap.add_argument("--dir", default="runs/e3")
    ap.add_argument("--fit-steps", type=int, default=1500)
    ap.add_argument("--overlap", type=float, default=0.05,
                    help="D4 overlap bonus coefficient")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    d = Path(args.dir)
    if not (d / "base.pt").exists():
        raise SystemExit(f"missing {d}/base.pt — run fit_e3_atoms.py first")
    vocab = E3Vocab()
    model = _load_wrap(d, device)
    {"d2": stage_d2, "d3": stage_d3, "d4": stage_d4, "d5": stage_d5}[args.stage](
        args, model, vocab, device, d)


if __name__ == "__main__":
    main()
