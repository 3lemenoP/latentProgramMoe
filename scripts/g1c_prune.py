#!/usr/bin/env python3
"""G1-C — pruning report: the artifact-size story (gate1-steering §1).

Prune sites below activity τ to identity; re-eval task CE and neutral ppl
per τ; report the size-vs-recovery curve with fp16 and 8-bit-sparse bytes.

    python scripts/g1c_prune.py --base EleutherAI/pythia-410m \
        --program runs/g1a/z_french_conj_only.pt --task french \
        --teacher experts-2000/french/merged --out runs/g1c
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lpm import LatentProgramModel, ProgramField  # noqa: E402
from lpm.quaternion import q_angle2, q_normalize  # noqa: E402
from lpm.report_card import HIST_BINS, ce_per_block  # noqa: E402
from lpm.tasks import TASKS, _wikitext  # noqa: E402
from lpm.utils import get_device, pack_texts  # noqa: E402


def pruned(field: ProgramField, tau: float) -> ProgramField:
    spec = field.spec
    quats, kept, total = {}, 0, 0
    for k in spec.site_keys():
        q = q_normalize(field.q(*k).detach().float())
        s = q_angle2(q)
        mask = (s > tau).unsqueeze(-1)
        ident = torch.zeros_like(q)
        ident[..., 0] = 1.0
        quats[k] = torch.where(mask, q, ident)
        kept += int(mask.sum())
        total += mask.numel()
    return ProgramField(spec, quats, trainable=False), kept, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--program", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.0, 1e-4, 3e-4, 1e-3])
    ap.add_argument("--out", default="runs/g1c")
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = get_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    model = LatentProgramModel.from_pretrained(args.base).to(device)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    field = ProgramField.load(args.program, trainable=False).to(device)
    task_blocks = pack_texts(TASKS[args.task].texts("eval", 400), tok,
                             seq_len=128, max_blocks=args.n_eval)
    neutral_blocks = pack_texts(_wikitext("eval", 400), tok, seq_len=128,
                                max_blocks=args.n_eval)

    base_ce = float(ce_per_block(model, task_blocks, device).mean())
    base_ppl = math.exp(float(ce_per_block(model, neutral_blocks, device).mean()))

    # activity histogram
    s_all = torch.cat([q_angle2(q_normalize(field.q(*k).detach().float())
                                ).reshape(-1) for k in field.spec.site_keys()])
    hist = torch.histogram(s_all.cpu(), bins=torch.tensor(HIST_BINS)).hist.tolist()

    rows = []
    for tau in args.taus:
        pf, kept, total = pruned(field, tau)
        pf = pf.to(device)
        ce = float(ce_per_block(model, task_blocks, device, field=pf).mean())
        ppl = math.exp(float(ce_per_block(model, neutral_blocks, device,
                                          field=pf).mean()))
        rows.append({"tau": tau, "kept_sites": kept, "total_sites": total,
                     "kept_frac": kept / total,
                     "task_ce": ce, "neutral_ppl": ppl,
                     "bytes_fp16_dense": total * 8,
                     "bytes_8bit_sparse": kept * 7 + 16})
        print(json.dumps(rows[-1]))

    ce0 = rows[0]["task_ce"]
    verdicts = {}
    for r in rows[1:]:
        if r["kept_frac"] <= 0.10 and (r["task_ce"] - ce0) <= 0.01:
            verdicts["G1C_90pct_prunable_at_0.01nat"] = True
            break
    verdicts.setdefault("G1C_90pct_prunable_at_0.01nat", False)
    verdicts["G1C_under_100KB_8bit"] = any(
        r["bytes_8bit_sparse"] < 100 * 1024 and (r["task_ce"] - ce0) <= 0.01
        for r in rows)
    report = {"base_task_ce": base_ce, "base_neutral_ppl": base_ppl,
              "histogram_bins": HIST_BINS, "histogram": hist,
              "curve": rows, "verdicts": verdicts}
    (out / "report_g1c.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(verdicts, indent=2))
    print(f"wrote {out / 'report_g1c.json'}")


if __name__ == "__main__":
    main()
