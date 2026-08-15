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
    table = []

    def measure_L(ftype, sbar, salt=0):
        f = single_type_field(spec, ftype, sbar,
                              seed=args.seed + salt + hash((ftype, round(sbar, 6))) % 9973
                              ).to(device)
        kl = kl_to_base(model, f, batches)
        table.append({"group": ftype, "sbar": sbar, "kl": kl, "L": kl / sbar})
        print(f"{ftype:11s} s̄={sbar:<8.5f} KL={kl:.6f}  L={kl / sbar:.3f}")
        return kl / sbar

    # pass 1: seed leverage at the requested s̄ grid
    leverage = {ftype: sum(measure_L(ftype, s) for s in args.sbars) / len(args.sbars)
                for ftype in spec.site_names()}
    wh = Whitening(spec, leverage,
                   meta={"base": args.base, "sbars": args.sbars,
                         "n_probe": args.n_probe, "seed": args.seed})

    def sbar_at(rho, c):
        theta = min(rho / c, 3.0)
        return max(float(torch.sin(torch.tensor(theta / 2.0)) ** 2), 1e-5)

    def gate(rho, w):
        kls = {}
        for ftype in spec.site_names():
            sbar = sbar_at(rho, w.c[ftype])
            f = single_type_field(spec, ftype, sbar,
                                  seed=args.seed + 31 + hash(ftype) % 997).to(device)
            kls[ftype] = kl_to_base(model, f, batches)
        vals = list(kls.values())
        return kls, max(vals) / max(min(vals), 1e-12)

    # KL response is superlinear for some groups on some bases (measured:
    # toy ffn_hidden/qk_rel) — a single small-s̄ L_g misprices other
    # strengths. Anchor the calibration AT the operating strength: re-measure
    # L_g at the s̄ each group hits at gate ρ, rebuild c, re-gate; halve ρ on
    # failure (spec §3 step 2 escalation).
    rho, passed, kls, ratio = args.rho, False, {}, float("inf")
    for attempt in range(4):
        for _ in range(2):  # self-consistency iterations at this rho
            lev2 = {ftype: measure_L(ftype, sbar_at(rho, wh.c[ftype]), salt=53)
                    for ftype in spec.site_names()}
            wh = Whitening(spec, lev2, meta=wh.meta)
        kls, ratio = gate(rho, wh)
        passed = ratio <= 2.0
        print(f"T18 gate @rho={rho}: "
              f"{json.dumps({k: round(v, 6) for k, v in kls.items()})} "
              f"max/min={ratio:.2f} pass={passed}")
        if passed:
            break
        rho = rho / 2.0
    leverage = wh.leverage

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
