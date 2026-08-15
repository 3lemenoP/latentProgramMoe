#!/usr/bin/env python3
"""B3 (Q2) — does the axis field crack the offset ceiling? (causal test)
(steering-workstreams-a-b.md B3)

Retrain the 4-layer toy WITH full rotary (rotary_pct=1.0), same tasks. Run
the A2 offset ladder twice per rung: programs = conjugation fields only
(rope_ax frozen at identity), then conjugation + axis field. Prediction: the
axis field lifts the k ≥ 1 ceilings toward the k = 0 level (≥ 5 points).

    python scripts/b3_rotary_ladder.py --dir runs/b3
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

from lpm import LatentProgramModel, ProgramField  # noqa: E402
from lpm.tasks import E3Vocab  # noqa: E402
from lpm.utils import get_device, set_seed  # noqa: E402
from e3_common import (  # noqa: E402
    fit_field, load_base, make_base_rotary, ordered_accuracy, save_base,
    train_base,
)
from e3prime import _decode_failures  # noqa: E402

RUNGS = [
    ("k0", "reverse", 0),
    ("k1_padA", "b_then_a", 1),
    ("k1_padB_control", "c_then_b", 1),
    ("k2_padA", "rev_pad2", 2),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/b3")
    ap.add_argument("--base-steps", type=int, default=3000)
    ap.add_argument("--fit-steps", type=int, default=3000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    vocab = E3Vocab()

    base_path = d / "base.pt"
    if base_path.exists():
        base = load_base(str(base_path), device)
    else:
        print("=== rotary toy base (copy/prepend/reverse/append) ===")
        base = make_base_rotary(vocab, device)
        train_base(base, vocab, device, steps=args.base_steps, seed=args.seed,
                   behaviors=("copy", "prepend", "reverse", "append"))
        save_base(base, str(base_path))
    model = LatentProgramModel(base)
    print(f"spec: {model.spec}")
    assert model.spec.rotary_ndims > 0, "base must be rotary for B3"

    base_acc = {b: ordered_accuracy(model, None, b, vocab, device)[0]
                for b in ("copy", "prepend", "reverse", "append")}
    print(f"base task exact: {base_acc}")

    report = {"base_tasks": base_acc, "rungs": {}}
    for name, behavior, k in RUNGS:
        row = {"k": k, "behavior": behavior}
        for cond, freeze in (("conj_only", ("rope_ax",)), ("conj_plus_axis", ())):
            p = d / f"z_{name}_{cond}.pt"
            if p.exists():
                f = ProgramField.load(str(p), trainable=False).to(device)
            else:
                f = ProgramField.randn_near_identity(
                    model.spec, sigma=1e-3, trainable=True).to(device)
                f = fit_field(model, f, behavior, vocab, device,
                              steps=args.fit_steps, seed=args.seed + 80 + k,
                              tag=f"{name}/{cond}", freeze_sites=freeze)
                f.save(str(p))
            e = ordered_accuracy(model, f, behavior, vocab, device)
            counts, _ = _decode_failures(model, f, behavior, vocab, device, n=100)
            # rope_ax usage: mean activity on the axis sites
            from lpm.quaternion import q_angle2, q_normalize
            ax = torch.cat([q_angle2(q_normalize(f.q(l, "rope_ax").float())).reshape(-1)
                            for l in range(model.spec.n_layers)])
            row[cond] = {"exact": e[0], "token": e[1],
                         "failure_counts": counts,
                         "rope_ax_mean_activity": float(ax.mean())}
            print(name, cond, json.dumps(row[cond]))
        row["lift_points"] = 100 * (row["conj_plus_axis"]["exact"]
                                    - row["conj_only"]["exact"])
        report["rungs"][name] = row

    lifts = {n: r["lift_points"] for n, r in report["rungs"].items() if r["k"] >= 1}
    report["b3_pass"] = all(v >= 5 for v in lifts.values())
    (d / "report_b3.json").write_text(json.dumps(report, indent=2))
    print(f"\nB3 lifts (points, k>=1): {lifts}  pass(all >=5)={report['b3_pass']}")


if __name__ == "__main__":
    main()
