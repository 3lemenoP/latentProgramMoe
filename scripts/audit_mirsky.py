#!/usr/bin/env python3
"""scripts/audit_mirsky.py — E0, the phase-0 gate (spec §10).

Conjugation preserves singular spectra, so for every weight matrix
    min_R || R W0 R^T - W_k ||_F  >=  || sigma(W0) - sigma(W_k) ||_2   (Mirsky).
This script measures, per matrix, what fraction of each fine-tune's displacement
is provably unreachable by rotation alone:
    ratio = || sigma(W0) - sigma(W_k) ||_2 / || W_k - W0 ||_F   in [0, 1].
Decision rule (spec): median ratio > 0.3  ->  enable abelian gains (§4).

Usage:
    python audit_mirsky.py --base gpt2 --experts ./ft-a ./ft-b --out report_e0.md
Experts must be full checkpoints loadable by AutoModelForCausalLM (merge LoRA
adapters first, e.g. PeftModel.merge_and_unload()). Runs on CPU in fp32.
"""
import argparse
import re
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM

SKIP = re.compile(r"(wte|wpe|embed|emb\.|rotary|ln|norm|bias|lm_head)", re.I)


def load_2d(name_or_path):
    model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=torch.float32)
    sd = {k: v.detach().clone() for k, v in model.named_parameters()
          if v.ndim == 2 and not SKIP.search(k)}
    del model
    return sd


def layer_of(name):
    m = re.search(r"\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def mat_type(name):
    parts = name.split(".")
    return parts[-2] if parts[-1] == "weight" else parts[-1]


def audit_expert(base_sd, expert_sd):
    rows = []
    for k, W0 in base_sd.items():
        Wk = expert_sd.get(k)
        if Wk is None or Wk.shape != W0.shape:
            continue
        disp = torch.linalg.norm(Wk - W0).item()
        if disp < 1e-12:
            continue
        s0 = torch.linalg.svdvals(W0)
        sk = torch.linalg.svdvals(Wk)
        lower = torch.linalg.norm(s0 - sk).item()
        rows.append(dict(name=k, layer=layer_of(k), mtype=mat_type(k),
                         disp=disp, lower=lower, ratio=lower / disp))
    return rows


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--experts", nargs="+", required=True)
    ap.add_argument("--out", default="report_e0.md")
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()

    base_sd = load_2d(args.base)
    lines = [f"# E0 — Mirsky audit\n", f"Base: `{args.base}`\n"]
    all_ratios = []

    for exp in args.experts:
        rows = audit_expert(base_sd, load_2d(exp))
        if not rows:
            lines.append(f"\n## `{exp}` — no changed 2D matrices found\n")
            continue
        ratios = [r["ratio"] for r in rows]
        all_ratios += ratios
        lines.append(f"\n## `{exp}`\n")
        lines.append(f"- matrices changed: {len(rows)}")
        lines.append(f"- median ratio: **{median(ratios):.3f}**   mean: {sum(ratios)/len(ratios):.3f}\n")

        by_type = defaultdict(list)
        for r in rows:
            by_type[r["mtype"]].append(r["ratio"])
        lines.append("| matrix type | count | median ratio |")
        lines.append("|---|---|---|")
        for t, rs in sorted(by_type.items()):
            lines.append(f"| {t} | {len(rs)} | {median(rs):.3f} |")

        worst = sorted(rows, key=lambda r: -r["ratio"])[:10]
        lines.append("\nworst 10 (most rotation-unreachable):\n")
        lines.append("| name | layer | ‖ΔW‖_F | Mirsky lower | ratio |")
        lines.append("|---|---|---|---|---|")
        for r in worst:
            lines.append(f"| {r['name']} | {r['layer']} | {r['disp']:.4f} "
                         f"| {r['lower']:.4f} | {r['ratio']:.3f} |")

    med = median(all_ratios)
    decision = "ENABLE gains (§4)" if med > args.threshold else "gains OFF (pure conjugation suffices)"
    lines.append(f"\n---\n\n**Overall median ratio: {med:.3f}**  (threshold {args.threshold})")
    lines.append(f"\n**Decision: {decision}** — record `enable_gains` in the config default.\n")

    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[-4:]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
