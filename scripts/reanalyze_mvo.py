#!/usr/bin/env python3
"""Phase 4 v2 — R1/R2/R3 reanalyses from existing artifacts (spec §2).
No training. Re-scores standing conclusions from data already on disk.

    python scripts/reanalyze_mvo.py \
        --codebook runs/codebook/report_codebook.json \
        --mvo runs/mvo/report_mvo.json \
        --mvo-dir runs/mvo --whitening runs/w0_rotary/whitening.json \
        --base runs/b3/base.pt --out runs/reanalysis
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
from lpm.quaternion import q_angle2, q_normalize  # noqa: E402
from lpm.whitening import Whitening  # noqa: E402
from e3_common import load_base  # noqa: E402
from codebook import kmedoids  # noqa: E402


# ---------------------------------------------------------------------------
# R1 — behavioral rate–distortion (covering radius in symKL vs k)
# ---------------------------------------------------------------------------
def r1(cb):
    D = torch.tensor(cb["D_beh"])
    curve = {}
    for k in range(2, 7):
        _, med = kmedoids(D, k, seed=0)
        curve[k] = float(D[:, med].min(dim=1).values.max())
    ks = sorted(curve)
    drops = {k: (curve[ks[i - 1]] - curve[k]) / max(curve[ks[i - 1]], 1e-9)
             for i, k in enumerate(ks) if i > 0}
    elbow = max(drops, key=drops.get)
    return {"covering_radius_symkl": curve, "relative_drops": drops,
            "largest_drop_at_k": elbow,
            "R1_elbow_at_4": elbow == 4}


# ---------------------------------------------------------------------------
# R2 — revisit-split recovery
# ---------------------------------------------------------------------------
def recovery_after(accs, oaccs, sp, tol=0.05, window=5, cap=None):
    cap = cap if cap is not None else len(accs)
    for i in range(sp, len(accs)):
        lo = max(i - window + 1, sp)
        a = sum(accs[lo:i + 1]) / (i + 1 - lo)
        o = sum(oaccs[lo:i + 1]) / (i + 1 - lo)
        if a >= o - tol:
            return i - sp
    return cap - sp


def r2(mvo):
    switch_points = mvo["switch_points"]
    oracle = mvo["logs"]["oracle"]
    oaccs = [r["acc"] for r in oracle]
    tasks_at = [r["task"] for r in oracle]
    out = {}
    for method, log in mvo["logs"].items():
        if method == "oracle":
            continue
        accs = [r["acc"] for r in log]
        first, revisit = [], []
        seen = set()
        for j, sp in enumerate(switch_points):
            task = tasks_at[sp]
            rec = recovery_after(accs, oaccs, sp)
            if sp == 0:
                seen.add(task)
                continue
            (revisit if task in seen else first).append(rec)
            seen.add(task)
        out[method] = {
            "first_visit_mean": sum(first) / max(len(first), 1),
            "revisit_mean": sum(revisit) / max(len(revisit), 1),
            "n_first": len(first), "n_revisit": len(revisit),
            "revisit_advantage": (sum(first) / max(len(first), 1))
                                 - (sum(revisit) / max(len(revisit), 1)),
        }
    agent = out.get("agent", {})
    adv = agent.get("revisit_advantage", 0.0)
    fv = agent.get("first_visit_mean", 1.0)
    out["R2_agent_no_revisit_advantage"] = bool(abs(adv) < 0.2 * max(fv, 1e-9))
    return out


# ---------------------------------------------------------------------------
# R3 — D-3 scoring in behavioral work shares (leverage × activity)
# ---------------------------------------------------------------------------
def work_shares(profile_activity, leverage):
    w = {g: profile_activity.get(g, 0.0) * leverage.get(g, 0.0)
         for g in profile_activity}
    tot = sum(w.values())
    return {g: v / max(tot, 1e-12) for g, v in w.items()}


def field_group_activity(field, spec):
    out = {}
    for gname in spec.site_names():
        xs = [q_angle2(q_normalize(field.q(l, s).detach().float())).mean()
              for (l, s) in spec.site_keys() if s == gname]
        out[gname] = float(torch.stack(xs).mean())
    return out


def r3(mvo, mvo_dir, wh, model, device):
    leverage = wh.leverage
    logs = mvo["logs"]["agent"]
    tasks = mvo["config"]["tasks"]
    # converged commitment map per task: last logged commitment while in a
    # regime of that task
    last_commit = {}
    for r in logs:
        if r.get("commitment"):
            last_commit[r["task"]] = r["commitment"]
    rows, cosines = {}, []
    for t in tasks:
        oracle_path = Path(mvo_dir) / f"oracle_{t}.pt"
        if t not in last_commit or not oracle_path.exists():
            continue
        f = ProgramField.load(str(oracle_path), trainable=False).to(device)
        prof = work_shares(field_group_activity(f, model.spec), leverage)
        agent = work_shares(last_commit[t], leverage)
        gs = sorted(set(prof) | set(agent))
        a = torch.tensor([agent.get(g, 0.0) for g in gs])
        b = torch.tensor([prof.get(g, 0.0) for g in gs])
        cos = float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))
        rows[t] = {"agent_work_shares": agent, "oracle_work_shares": prof,
                   "cosine": cos}
        cosines.append(cos)
    mean_cos = sum(cosines) / max(len(cosines), 1)
    return {"per_task": rows, "mean_cosine": mean_cos,
            "R3_D3_pass_cos_gt_0.8": all(c > 0.8 for c in cosines) and bool(cosines)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebook", default="runs/codebook/report_codebook.json")
    ap.add_argument("--mvo", default="runs/mvo/report_mvo.json")
    ap.add_argument("--mvo-dir", default="runs/mvo")
    ap.add_argument("--whitening", default="runs/w0_rotary/whitening.json")
    ap.add_argument("--base", default="runs/b3/base.pt")
    ap.add_argument("--out", default="runs/reanalysis")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-r3", action="store_true",
                    help="R3 needs oracle field checkpoints; R1/R2 are JSON-only")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cb = json.loads(Path(args.codebook).read_text())
    mvo = json.loads(Path(args.mvo).read_text())
    report = {"R1": r1(cb), "R2": r2(mvo)}
    if not args.skip_r3:
        model = LatentProgramModel(load_base(args.base, args.device))
        wh = Whitening.load(args.whitening, model.spec)
        report["R3"] = r3(mvo, args.mvo_dir, wh, model, args.device)
    else:
        report["R3"] = {"skipped": True, "mean_cosine": None,
                        "R3_D3_pass_cos_gt_0.8": None}
    (out / "report_reanalysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"R1": {k: v for k, v in report["R1"].items()
                             if k != "relative_drops"},
                      "R2": {k: v for k, v in report["R2"].items()},
                      "R3": {"mean_cosine": report["R3"]["mean_cosine"],
                             "pass": report["R3"]["R3_D3_pass_cos_gt_0.8"]}},
                     indent=2))
    print(f"wrote {out / 'report_reanalysis.json'}")


if __name__ == "__main__":
    main()
