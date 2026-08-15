#!/usr/bin/env python3
"""scripts/diagnose_e3.py — E3′ D1: no-training diagnostic on two fields.

Usage:
    python scripts/diagnose_e3.py --fields runs/e3/z_a.pt runs/e3/z_b.pt
    python scripts/diagnose_e3.py --fields runs/e3/z_a.pt runs/e3/z_b.pt \
        --base runs/e3/base.pt --out report_d1.md
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lpm import LatentProgramModel, ProgramField, compose  # noqa: E402
from lpm.quaternion import q_angle2, q_commutator, q_normalize  # noqa: E402
from lpm.tasks import E3Vocab, e3_examples  # noqa: E402
from lpm.utils import get_device  # noqa: E402
from e3_common import load_base  # noqa: E402


def _flat_activity(field: ProgramField) -> torch.Tensor:
    parts = [q_angle2(q_normalize(field.q(*k).float())).reshape(-1)
             for k in field.spec.site_keys()]
    return torch.cat(parts)


def _site_activity(field: ProgramField):
    return {k: q_angle2(q_normalize(field.q(*k).float()))
            for k in field.spec.site_keys()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", nargs=2, required=True, metavar=("Z_A", "Z_B"))
    ap.add_argument("--base", default=None, help="optional saved E3 base for behavioral KL")
    ap.add_argument("--out", default="report_d1.md")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = get_device(args.device)

    za = ProgramField.load(args.fields[0]).to(device)
    zb = ProgramField.load(args.fields[1]).to(device)
    if za.spec != zb.spec:
        raise SystemExit("field specs differ")

    sa_map, sb_map = _site_activity(za), _site_activity(zb)
    sa, sb = _flat_activity(za), _flat_activity(zb)
    cos = (sa @ sb / (sa.norm() * sb.norm()).clamp_min(1e-12)).item()
    ovl = (torch.minimum(sa, sb).sum() / torch.minimum(sa.sum(), sb.sum()).clamp_min(1e-12)).item()

    kappas = []
    kappa_layer = defaultdict(list)
    for k in za.spec.site_keys():
        kap = q_angle2(q_commutator(za.q(*k).float(), zb.q(*k).float())).reshape(-1)
        kappas.append(kap)
        kappa_layer[k[0]].extend(kap.tolist())
    kappa = torch.cat(kappas)
    kappa_mean, kappa_max = kappa.mean().item(), kappa.max().item()

    by_group = defaultdict(lambda: [0.0, 0.0, 0])
    for (l, n), act in sa_map.items():
        by_group[n][0] += act.mean().item()
        by_group[n][1] += sb_map[(l, n)].mean().item()
        by_group[n][2] += 1

    lines = [
        "# E3′ D1 — diagnostic battery\n",
        f"Fields: `{args.fields[0]}`  `{args.fields[1]}`\n",
        "## Support overlap\n",
        f"- activity cosine ⟨s_A, s_B⟩: **{cos:.4f}**",
        f"- OVL = Σ min(s_A,s_B) / min(Σs): **{ovl:.4f}**",
        f"- mean activity A / B: {sa.mean().item():.4f} / {sb.mean().item():.4f}\n",
        "## Geometric commutator κ = angle2(a ⊗ b ⊗ a* ⊗ b*)\n",
        f"- κ mean: **{kappa_mean:.6f}**   max: **{kappa_max:.6f}**",
        "",
        "| layer | κ mean | κ max |",
        "|---|---:|---:|",
    ]
    for l in range(za.spec.n_layers):
        xs = kappa_layer[l]
        lines.append(f"| {l} | {sum(xs)/len(xs):.6f} | {max(xs):.6f} |")
    lines += ["", "| site group | mean s_A | mean s_B |", "|---|---:|---:|"]
    for n, (a, b, c) in sorted(by_group.items()):
        lines.append(f"| {n} | {a/c:.5f} | {b/c:.5f} |")

    kl = None
    if args.base:
        vocab = E3Vocab()
        base = load_base(args.base, device)
        model = LatentProgramModel(base)
        ab = compose(zb, za)
        ba = compose(za, zb)
        ids, _, attn = e3_examples("copy", 64, vocab, seed=7)
        ids, attn = ids.to(device), attn.to(device)
        with torch.no_grad():
            with model.program(ab):
                la = model(input_ids=ids, attention_mask=attn).logits
            with model.program(ba):
                lb = model(input_ids=ids, attention_mask=attn).logits
        pa = F.log_softmax(la, dim=-1)
        pb = F.softmax(lb, dim=-1)
        kl = F.kl_div(pa.flatten(0, 1), pb.flatten(0, 1), reduction="batchmean").item()
        lines += ["", "## Behavioral commutator",
                  f"- mean KL compose(b,a) ‖ compose(a,b) on probe: **{kl:.6f}**"]

    pred = "disjoint-support artifact (§3.2)" if (kappa_mean < 1e-3 and cos < 0.2) \
        else "overlapping but silent" if kappa_mean < 1e-3 \
        else "live geometric commutator — check behavioral KL"
    lines += ["", f"**D1 reading:** {pred}",
              f"(low cosine + κ≈0 → independently fitted skills sit on disjoint sites.)"]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
