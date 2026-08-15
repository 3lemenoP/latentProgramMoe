#!/usr/bin/env python3
"""W0 — whitening calibration gate (phase-4 spec §3).

Per site group: matched-activity random fields at s̄ ∈ {0.01, 0.03}, KL to
base on probe sequences; L_g = mean(KL/s̄). c_g² = L_g/(4·n_g). Gate T18:
random fields at equal WHITENED norm across groups must produce KL within
×2 of each other; on failure, reduce ρ once and re-gate (per-layer
refinement is the next escalation, not automated here).

    python scripts/calibrate_whitening.py --base runs/b3/base.pt --out runs/w0
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

from lpm import LatentProgramModel, ProgramField  # noqa: E402
from lpm.tasks import E3Vocab, e3_examples  # noqa: E402
from lpm.utils import get_device, set_seed  # noqa: E402
from lpm.whitening import Whitening, group_sizes  # noqa: E402
from e3_common import load_base  # noqa: E402


def single_type_field(spec, ftype, sbar, seed):
    g = torch.Generator().manual_seed(seed)
    quats = {}
    for (l, n) in spec.site_keys():
        t = torch.zeros(*spec.site_shape(n), 4)
        t[..., 0] = 1.0
        if n == ftype:
            axis = torch.randn(*spec.site_shape(n), 3, generator=g)
            axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            t[..., 0] = (1.0 - sbar) ** 0.5
            t[..., 1:] = (sbar ** 0.5) * axis
        quats[(l, n)] = t
    return ProgramField(spec, quats, trainable=False)


@torch.no_grad()
def kl_to_base(model, field, batches):
    tot, cnt = 0.0, 0
    for ids, attn in batches:
        base = model(input_ids=ids, attention_mask=attn).logits
        with model.program(field):
            prog = model(input_ids=ids, attention_mask=attn).logits
        lp = F.log_softmax(prog.float(), dim=-1)
        lb = F.log_softmax(base.float(), dim=-1)
        kl = (lp.exp() * (lp - lb)).sum(-1)
        m = attn.bool()
        tot += float(kl[m].sum())
        cnt += int(m.sum())
    return tot / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="runs/w0")
    ap.add_argument("--sbars", type=float, nargs="+", default=[0.01, 0.03])
    ap.add_argument("--n-probe", type=int, default=128)
    ap.add_argument("--rho", type=float, default=0.1,
                    help="whitened norm per site for the T18 gate fields")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = LatentProgramModel(load_base(args.base, device))
    spec = model.spec
    vocab = E3Vocab()
    # probe: mixture of the base's trained behaviors
    batches = []
    for i, beh in enumerate(("copy", "prepend", "reverse", "append")):
        ids, _, attn = e3_examples(beh, args.n_probe // 4, vocab, seed=7 + i)
        batches.append((ids.to(device), attn.to(device)))

    sizes = group_sizes(spec)
    leverage, table = {}, []
    for ftype in spec.site_names():
        Ls = []
        for sbar in args.sbars:
            f = single_type_field(spec, ftype, sbar,
                                  seed=args.seed + hash((ftype, sbar)) % 9973).to(device)
            kl = kl_to_base(model, f, batches)
            Ls.append(kl / sbar)
            table.append({"group": ftype, "sbar": sbar, "kl": kl, "L": kl / sbar})
            print(f"{ftype:11s} s̄={sbar:<5} KL={kl:.5f}  L={kl / sbar:.3f}")
        leverage[ftype] = sum(Ls) / len(Ls)

    wh = Whitening(spec, leverage,
                   meta={"base": args.base, "sbars": args.sbars,
                         "n_probe": args.n_probe, "seed": args.seed})

    # ---- T18 gate: equal whitened norm ⇒ KL within ×2 across groups --------
    def gate(rho):
        kls = {}
        for ftype in spec.site_names():
            # whitened per-site norm rho ⇒ raw angle θ = rho / c_g,
            # s̄ = sin²(θ/2)
            theta = rho / wh.c[ftype]
            sbar = float(torch.sin(torch.tensor(theta / 2.0)) ** 2)
            f = single_type_field(spec, ftype, sbar,
                                  seed=args.seed + 31 + hash(ftype) % 997).to(device)
            kls[ftype] = kl_to_base(model, f, batches)
        vals = list(kls.values())
        ratio = max(vals) / max(min(vals), 1e-12)
        return kls, ratio

    rho = args.rho
    kls, ratio = gate(rho)
    passed = ratio <= 2.0
    if not passed:
        rho = rho / 2.0
        kls2, ratio2 = gate(rho)
        if ratio2 <= 2.0:
            kls, ratio, passed = kls2, ratio2, True
    print(f"T18 gate @rho={rho}: KLs {json.dumps({k: round(v, 5) for k, v in kls.items()})} "
          f"max/min={ratio:.2f} pass={passed}")

    wh.meta.update({"t18_rho": rho, "t18_ratio": ratio, "t18_pass": passed,
                    "t18_kls": kls})
    wh.save(str(out / "whitening.json"))
    (out / "w0_table.json").write_text(json.dumps(
        {"table": table, "leverage": leverage, "group_sizes": sizes,
         "c": wh.c, "t18": {"rho": rho, "ratio": ratio, "pass": passed,
                            "kls": kls}}, indent=2))
    print(f"wrote {out / 'whitening.json'}  (T18 {'PASS' if passed else 'FAIL'})")
    if not passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
