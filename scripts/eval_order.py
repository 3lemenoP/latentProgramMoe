#!/usr/bin/env python3
"""scripts/eval_order.py — E3: order sensitivity, the falsifier (spec §10).

Superseded as a *gate* by E3′ (e-suite-analysis-e3prime.md). Still the
original protocol. --save-dir writes z_a.pt / z_b.pt / base.pt for D1.

Usage:
    python scripts/eval_order.py --save-dir runs/e3
"""
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lpm import (AbelianProgramField, AxisBank, LatentProgramModel,  # noqa: E402
                 LPMConfig, ProgramField, compose)
from lpm.tasks import E3Vocab  # noqa: E402
from lpm.utils import get_device, set_seed  # noqa: E402
from e3_common import (  # noqa: E402
    fit_field, fit_fields_joint, make_base, ordered_accuracy, save_base, train_base,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e3_order.yaml")
    ap.add_argument("--out", default="report_e3.md")
    ap.add_argument("--save-dir", default="runs/e3")
    ap.add_argument("--base-steps", type=int, default=3000)
    ap.add_argument("--fit-steps", type=int, default=1500)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = LPMConfig.from_yaml(args.config) if Path(args.config).exists() else LPMConfig()
    set_seed(args.seed)
    device = get_device(args.device)
    vocab = E3Vocab()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1. base model (mixture of copy/prepend/reverse) ===")
    base = make_base(vocab, device)
    train_base(base, vocab, device, steps=args.base_steps, seed=args.seed)
    save_base(base, str(save_dir / "base.pt"))
    model = LatentProgramModel(base)

    print("=== 2. direct-fit z_a (prepend), z_b (reverse) ===")
    za = ProgramField.randn_near_identity(model.spec, sigma=cfg.field_init_sigma,
                                          trainable=True).to(device)
    za = fit_field(model, za, "prepend", vocab, device, steps=args.fit_steps,
                   seed=args.seed + 1, tag="z_a")
    zb = ProgramField.randn_near_identity(model.spec, sigma=cfg.field_init_sigma,
                                          trainable=True).to(device)
    zb = fit_field(model, zb, "reverse", vocab, device, steps=args.fit_steps,
                   seed=args.seed + 2, tag="z_b")
    za.save(str(save_dir / "z_a.pt"))
    zb.save(str(save_dir / "z_b.pt"))

    print("=== 3. abelian baseline (shared axes, theta-only) ===")
    bank = AxisBank(model.spec).to(device)
    ta = AbelianProgramField(bank).to(device)
    tb = AbelianProgramField(bank).to(device)
    ta, tb = fit_fields_joint(model, [(ta, "prepend"), (tb, "reverse")], vocab,
                              device, steps=args.fit_steps, seed=args.seed + 3,
                              extra_params=bank.parameters(), tag="abelian joint")

    print("=== 4. zero-shot ordered evaluation ===")
    rows = {}
    rows["z_a on prepend"] = ordered_accuracy(model, za, "prepend", vocab, device)
    rows["z_b on reverse"] = ordered_accuracy(model, zb, "reverse", vocab, device)
    ab = compose(zb, za)
    ba = compose(za, zb)
    rows["compose(b,a) on a_then_b"] = ordered_accuracy(model, ab, "a_then_b", vocab, device)
    rows["compose(a,b) on b_then_a"] = ordered_accuracy(model, ba, "b_then_a", vocab, device)
    rows["compose(b,a) on b_then_a (swapped)"] = ordered_accuracy(model, ab, "b_then_a", vocab, device)
    rows["compose(a,b) on a_then_b (swapped)"] = ordered_accuracy(model, ba, "a_then_b", vocab, device)

    tab_ = compose(tb.to_program_field(), ta.to_program_field())
    tba_ = compose(ta.to_program_field(), tb.to_program_field())
    rows["abelian ta on prepend"] = ordered_accuracy(model, ta, "prepend", vocab, device)
    rows["abelian tb on reverse"] = ordered_accuracy(model, tb, "reverse", vocab, device)
    rows["abelian compose(b,a) on a_then_b"] = ordered_accuracy(model, tab_, "a_then_b", vocab, device)
    rows["abelian compose(a,b) on b_then_a"] = ordered_accuracy(model, tba_, "b_then_a", vocab, device)

    ordered_nonabelian = 0.5 * (rows["compose(b,a) on a_then_b"][0] +
                                rows["compose(a,b) on b_then_a"][0])
    ordered_abelian = 0.5 * (rows["abelian compose(b,a) on a_then_b"][0] +
                             rows["abelian compose(a,b) on b_then_a"][0])
    swapped = 0.5 * (rows["compose(b,a) on b_then_a (swapped)"][0] +
                     rows["compose(a,b) on a_then_b (swapped)"][0])
    margin = 100 * (ordered_nonabelian - ordered_abelian)
    right_direction = ordered_nonabelian > swapped
    verdict = "PASS" if (margin > 20 and right_direction) else "FAIL"

    lines = ["# E3 — order sensitivity (falsifier)\n",
             "| condition | exact match | token acc |", "|---|---|---|"]
    for k, (em, ta_) in rows.items():
        lines.append(f"| {k} | {em:.3f} | {ta_:.3f} |")
    lines += ["",
              f"- ordered accuracy, non-abelian: **{100*ordered_nonabelian:.1f}%**",
              f"- ordered accuracy, abelian baseline: **{100*ordered_abelian:.1f}%**",
              f"- margin: **{margin:.1f} points** (pass needs > 20)",
              f"- swapped-order accuracy (should be lower): {100*swapped:.1f}% "
              f"-> right direction: {right_direction}",
              f"\n**{verdict}** — phase 3 is "
              f"{'unlocked' if verdict == 'PASS' else 'blocked'}; "
              f"E3′ (D1–D6) is the composition gate."]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    (Path(args.out).with_suffix(".json")).write_text(
        json.dumps({k: {"exact": v[0], "token": v[1]} for k, v in rows.items()},
                   indent=2))
    print(f"saved fields under {save_dir}")


if __name__ == "__main__":
    main()
