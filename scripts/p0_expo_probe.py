#!/usr/bin/env python3
"""P0 — exponentiation probe (handoff §4 P0; no training).

For λ ∈ {1, 1.25, 1.5, 2}: amplify z_a, z_b via q_pow. Gate 1: own-task
retention ≥ 0.9 of the λ=1 exact score. If retained: composition table, κ
stats, and order-KL vs λ.

Predictions under handoff §2.1: commutator angle ~λ² (κ(λ=2) ≈ 0.05);
order-KL leaves zero roughly quadratically.

    python scripts/p0_expo_probe.py --dir runs/e3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lpm import LatentProgramModel, ProgramField, compose, pow_field  # noqa: E402
from lpm.quaternion import q_angle2, q_commutator  # noqa: E402
from lpm.tasks import E3Vocab, e3_examples  # noqa: E402
from lpm.utils import get_device, set_seed  # noqa: E402
from e3_common import load_base, ordered_accuracy  # noqa: E402


def kappa_stats(fa: ProgramField, fb: ProgramField):
    kaps = [q_angle2(q_commutator(fa.q(*k).float(), fb.q(*k).float())).reshape(-1)
            for k in fa.spec.site_keys()]
    kap = torch.cat(kaps)
    return float(kap.mean()), float(kap.max())


@torch.no_grad()
def order_kl(model, ab, ba, vocab, device, n=64, seed=7):
    ids, _, attn = e3_examples("copy", n, vocab, seed=seed)
    ids, attn = ids.to(device), attn.to(device)
    with model.program(ab):
        la = model(input_ids=ids, attention_mask=attn).logits
    with model.program(ba):
        lb = model(input_ids=ids, attention_mask=attn).logits
    lp = F.log_softmax(la.float(), dim=-1)
    lq = F.log_softmax(lb.float(), dim=-1)
    kl_tok = (lp.exp() * (lp - lq)).sum(-1)
    return float(kl_tok[attn.bool()].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/e3")
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    d = Path(args.dir)

    base = load_base(str(d / "base.pt"), device)
    model = LatentProgramModel(base)
    vocab = E3Vocab()
    za = ProgramField.load(str(d / "z_a.pt"), trainable=False).to(device)
    zb = ProgramField.load(str(d / "z_b.pt"), trainable=False).to(device)

    base_a = ordered_accuracy(model, za, "prepend", vocab, device)[0]
    base_b = ordered_accuracy(model, zb, "reverse", vocab, device)[0]
    print(f"λ=1 own-task exact: z_a prepend {base_a:.3f}  z_b reverse {base_b:.3f}")

    rows = []
    for lam in args.lams:
        za_l = pow_field(za, lam)
        zb_l = pow_field(zb, lam)
        r_a = ordered_accuracy(model, za_l, "prepend", vocab, device)[0]
        r_b = ordered_accuracy(model, zb_l, "reverse", vocab, device)[0]
        retained = (r_a >= 0.9 * base_a) and (r_b >= 0.9 * base_b)
        row = {"lambda": lam, "exact_a_prepend": r_a, "exact_b_reverse": r_b,
               "retained": retained}
        km, kx = kappa_stats(za_l, zb_l)
        row["kappa_mean"], row["kappa_max"] = km, kx
        ab = compose(zb_l, za_l)
        ba = compose(za_l, zb_l)
        row["order_kl"] = order_kl(model, ab, ba, vocab, device)
        e_ab = ordered_accuracy(model, ab, "a_then_b", vocab, device)
        e_ba = ordered_accuracy(model, ba, "b_then_a", vocab, device)
        e_sw = ordered_accuracy(model, ab, "b_then_a", vocab, device)
        row.update({"compose_ab_exact": e_ab[0], "compose_ab_token": e_ab[1],
                    "compose_ba_exact": e_ba[0], "compose_ba_token": e_ba[1],
                    "swapped_ab_on_ba_exact": e_sw[0]})
        rows.append(row)
        print(json.dumps(row))

    out = {"base_exact_a": base_a, "base_exact_b": base_b, "rows": rows}
    (d / "report_p0.json").write_text(json.dumps(out, indent=2))

    # branch reading (handoff §4 P0 outcomes)
    lam_max = max(r["lambda"] for r in rows)
    r_end = next(r for r in rows if r["lambda"] == lam_max)
    r_one = next(r for r in rows if r["lambda"] == 1.0)
    if not all(r["retained"] for r in rows):
        broke = [r["lambda"] for r in rows if not r["retained"]]
        print(f"P0: skills break at λ ∈ {broke} — probe inconclusive, P1 is the path.")
    elif r_end["order_kl"] > 10 * max(r_one["order_kl"], 1e-6):
        print("P0: κ grows AND order-KL leaves zero — strength story confirmed; "
              "proceed to P1 with confidence.")
    elif r_end["kappa_mean"] > 5 * max(r_one["kappa_mean"], 1e-9):
        print("P0: κ grows on schedule but order-KL stays pinned — commutator "
              "may be stabilizer-bound; weight shifts to D3/D6 (still run P1).")
    else:
        print("P0: κ did not grow as predicted — check q_pow application.")
    print(f"wrote {d / 'report_p0.json'}")


if __name__ == "__main__":
    main()
