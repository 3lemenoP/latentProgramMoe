#!/usr/bin/env python3
"""G1-A — the B4 controls: interrogate the 120% (gate1-steering §1).

Consumes teachers trained by make_experts (800- and 2000-step) and the
program fit by fit_expert. Computes: recovery vs both teachers with paired
bootstrap CIs; teacher/program train-vs-heldout generalization gaps; the
Mirsky audit on both LoRA-merged teachers; executes the decision rule.

    python scripts/g1a_analyze.py --base EleutherAI/pythia-410m \
        --teacher800 experts-800/french/merged \
        --teacher2000 experts-2000/french/merged \
        --program runs/g1a/z_french_conj_only.pt --out runs/g1a
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
from lpm.report_card import (ce_per_block, paired_bootstrap,  # noqa: E402
                             recovery_bootstrap)
from lpm.tasks import TASKS  # noqa: E402
from lpm.utils import get_device, pack_texts  # noqa: E402
from audit_mirsky import audit_expert, load_2d  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="EleutherAI/pythia-410m")
    ap.add_argument("--task", default="french")
    ap.add_argument("--teacher800", required=True)
    ap.add_argument("--teacher2000", required=True)
    ap.add_argument("--program", required=True)
    ap.add_argument("--out", default="runs/g1a")
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = get_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = LatentProgramModel.from_pretrained(args.base).to(device)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    field = ProgramField.load(args.program, trainable=False).to(device)
    t800 = AutoModelForCausalLM.from_pretrained(
        args.teacher800, dtype=torch.float32).to(device).eval()
    t2000 = AutoModelForCausalLM.from_pretrained(
        args.teacher2000, dtype=torch.float32).to(device).eval()

    task = TASKS[args.task]
    eval_blocks = pack_texts(task.texts("eval", 400), tok, seq_len=128,
                             max_blocks=args.n_eval)
    train_blocks = pack_texts(task.texts("train", 400), tok, seq_len=128,
                              max_blocks=args.n_eval)

    ce = {}
    for split, blocks in (("eval", eval_blocks), ("train", train_blocks)):
        ce[split] = {
            "base": ce_per_block(model, blocks, device),
            "program": ce_per_block(model, blocks, device, field=field),
            "teacher800": ce_per_block(t800, blocks, device),
            "teacher2000": ce_per_block(t2000, blocks, device),
        }

    e = ce["eval"]
    report = {
        "task": args.task, "n_eval_blocks": int(e["base"].shape[0]),
        "eval_ce": {k: float(v.mean()) for k, v in e.items()},
        "train_ce": {k: float(v.mean()) for k, v in ce["train"].items()},
        # generalization gaps (positive = overfit / "noise to filter")
        "generalization_gap": {
            k: float(e[k].mean() - ce["train"][k].mean())
            for k in ("teacher800", "teacher2000", "program", "base")},
        "recovery_vs_800": recovery_bootstrap(e["base"], e["program"],
                                              e["teacher800"]),
        "recovery_vs_2000": recovery_bootstrap(e["base"], e["program"],
                                               e["teacher2000"]),
        "margin_vs_800": paired_bootstrap(e["teacher800"], e["program"]),
        "margin_vs_2000": paired_bootstrap(e["teacher2000"], e["program"]),
    }

    # Mirsky audit on both merged teachers
    print("mirsky audit (cpu, fp32 svd)...")
    base_sd = load_2d(args.base)
    for name, path in (("teacher800", args.teacher800),
                       ("teacher2000", args.teacher2000)):
        rows = audit_expert(base_sd, load_2d(path))
        ratios = sorted(r["ratio"] for r in rows)
        report[f"mirsky_{name}"] = {
            "median_ratio": ratios[len(ratios) // 2] if ratios else None,
            "max_ratio": max(ratios) if ratios else None,
            "n_matrices": len(rows)}

    # ---- decision rule (gate1-steering §1) ---------------------------------
    r2000 = report["recovery_vs_2000"]
    lo = r2000["ci95"][0]
    if r2000["recovery"] >= 1.0 and lo > 0.95:
        headline = "programs beat their teachers"
    elif 0.90 <= r2000["recovery"] < 1.0 or (r2000["recovery"] >= 1.0
                                             and lo <= 0.95):
        headline = "parity at ~30x smaller, with guarantees"
    else:
        headline = "improving with scale (scale-trend claim reverted)"
    report["decision"] = {"headline": headline,
                          "recovery_vs_2000": r2000["recovery"],
                          "ci95": r2000["ci95"]}
    report["registered"] = {
        "controls_preserve_>=100% (prior 0.55)": r2000["recovery"] >= 1.0,
        "teacher_positive_generalization_gap (prior 0.60)":
            report["generalization_gap"]["teacher2000"] > 0,
        "mirsky_median_<=0.03 (prior 0.60)":
            (report["mirsky_teacher2000"]["median_ratio"] or 1) <= 0.03,
    }
    (out / "report_g1a.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ()}, indent=2))
    print(f"wrote {out / 'report_g1a.json'}")


if __name__ == "__main__":
    main()
