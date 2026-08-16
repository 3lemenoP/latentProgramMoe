#!/usr/bin/env python3
"""G1-B supplement — the G1-A finding generalized: budget-matched (2000-step)
LoRA teachers overfit-invert on every natural-text corpus at this size
(formal −0.58, hedge −0.26, medical −0.94; only synthetic jsonish +1.50).
Recovery needs a non-degenerate denominator: train shorter teachers per
inverted skill (800, fallback 400), pick the best by eval gap, and build an
`experts-best/` layout for g1b_analyze.

    python scripts/g1b_best_teachers.py --skills formal medical hedge
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def gap_of(out_dir: Path, skill: str) -> float:
    rep = json.loads((out_dir / skill / "report.json").read_text())
    return rep["ce_base"] - rep["ce_expert"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="EleutherAI/pythia-410m")
    ap.add_argument("--skills", nargs="+", default=["formal", "medical", "hedge"])
    ap.add_argument("--good", nargs="+", default=["jsonish"],
                    help="skills whose 2000-step teacher already has a "
                         "positive gap")
    ap.add_argument("--budgets", type=int, nargs="+", default=[800, 400])
    args = ap.parse_args()

    best = {s: ("experts-2000", gap_of(Path("experts-2000"), s))
            for s in args.good}
    for skill in args.skills:
        chosen = ("experts-2000", gap_of(Path("experts-2000"), skill))
        for steps in args.budgets:
            out = f"experts-{steps}"
            rep = Path(out) / skill / "report.json"
            if not rep.exists():
                print(f"training {skill} teacher @{steps} steps...")
                subprocess.run(
                    [PYTHON, "scripts/make_experts.py", "--base", args.base,
                     "--out", out, "--tasks", skill, "--steps", str(steps)],
                    check=True)
            g = gap_of(Path(out), skill)
            print(f"{skill}@{steps}: gap {g:+.4f}")
            if g > chosen[1]:
                chosen = (out, g)
            if g > 0.05:
                break
        best[skill] = chosen

    dest = Path("experts-best")
    dest.mkdir(exist_ok=True)
    for skill, (src, g) in best.items():
        d = dest / skill
        if d.exists() or d.is_symlink():
            if d.is_symlink():
                d.unlink()
            else:
                shutil.rmtree(d)
        d.symlink_to((Path(src) / skill).resolve())
        print(f"experts-best/{skill} -> {src} (gap {g:+.4f})")
    (dest / "provenance.json").write_text(json.dumps(
        {s: {"source": src, "gap": g} for s, (src, g) in best.items()},
        indent=2))


if __name__ == "__main__":
    main()
