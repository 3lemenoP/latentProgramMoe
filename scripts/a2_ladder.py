#!/usr/bin/env python3
"""A2 — offset-reversal ladder: turn the ~0.82 ceiling into a law
(steering-workstreams-a-b.md A2).

rev_k: output = PAD×k + reverse(x). Rungs:
  k=0            = reverse            (existing z_b)
  k=1, pad=⟨A⟩   = b_then_a           (existing z_ba)
  k=1, pad=⟨B⟩   = c_then_b           (existing z_cb — marker-identity control)
  k=2, pad=⟨A⟩   = rev_pad2           (fitted here)

Deliverable: exact-vs-k curve + decode-error classes per rung.

    python scripts/a2_ladder.py --dir runs/e3strong
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lpm import LatentProgramModel, ProgramField  # noqa: E402
from lpm.tasks import E3Vocab  # noqa: E402
from lpm.utils import get_device, set_seed  # noqa: E402
from e3_common import fit_field, load_base, ordered_accuracy  # noqa: E402
from e3prime import _decode_failures  # noqa: E402

RUNGS = [
    ("k0", "reverse", "z_b.pt", 0),
    ("k1_padA", "b_then_a", "z_ba.pt", 1),
    ("k1_padB_control", "c_then_b", "z_cb.pt", 1),
    ("k2_padA", "rev_pad2", "z_rev2.pt", 2),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/e3strong")
    ap.add_argument("--fit-steps", type=int, default=3000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    d = Path(args.dir)
    vocab = E3Vocab()
    model = LatentProgramModel(load_base(str(d / "base.pt"), device))

    report = {}
    for name, behavior, fname, k in RUNGS:
        p = d / fname
        if p.exists():
            f = ProgramField.load(str(p), trainable=False).to(device)
            src = "loaded"
        else:
            f = ProgramField.randn_near_identity(model.spec, sigma=1e-3,
                                                 trainable=True).to(device)
            f = fit_field(model, f, behavior, vocab, device,
                          steps=args.fit_steps, seed=args.seed + 70 + k,
                          tag=name)
            f.save(str(p))
            src = "fitted"
        e = ordered_accuracy(model, f, behavior, vocab, device)
        counts, dumps = _decode_failures(model, f, behavior, vocab, device)
        report[name] = {"k": k, "behavior": behavior, "src": src,
                        "exact": e[0], "token": e[1],
                        "failure_counts": counts,
                        "failure_dumps": dumps[:8]}
        print(name, json.dumps({kk: vv for kk, vv in report[name].items()
                                if kk != "failure_dumps"}))

    curve = {r[0]: report[r[0]]["exact"] for r in RUNGS}
    mono = (curve["k0"] > curve["k1_padA"] > curve["k2_padA"])
    ctrl = abs(curve["k1_padA"] - curve["k1_padB_control"]) < 0.05
    report["exact_vs_k"] = curve
    report["monotone_in_k"] = mono
    report["pad_identity_control_null"] = ctrl
    (d / "report_a2.json").write_text(json.dumps(report, indent=2))
    print(f"\nA2 curve: {curve}")
    print(f"A2: monotone-in-k={mono}  pad-identity-control-null={ctrl}")


if __name__ == "__main__":
    main()
