#!/usr/bin/env python3
"""B2 (Q1) — leverage ratio of the axis field vs conjugation fields
(steering-workstreams-a-b.md B2).

Random fields at matched mean activity s̄ (uniform random axis per site,
deterministic angle with sin²(θ/2) = s̄), one field type at a time. Measures
KL(program ‖ base) on Pile-validation sequences:

1. leverage L = KL/s̄ per field type (full curve over s̄)
2. distance-resolved KL: per-position KL vs position — THE plot: prediction
   is flat-in-position for conjugation fields, growing for the axis field
3. attention-displacement probe: mean |Δattn| vs relative distance,
   axis vs qk_rel, at s̄ = 0.01

    python scripts/b2_leverage.py --model EleutherAI/pythia-410m --out runs/b2
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

from lpm import LatentProgramModel, ProgramField  # noqa: E402

FIELD_TYPES = ("rope_ax", "qk_rel", "ffn_hidden", "attn_io")
SBARS = (0.005, 0.01, 0.03, 0.06)


def make_single_type_field(spec, ftype: str, sbar: float, seed: int) -> ProgramField:
    """Identity everywhere except `ftype`: uniform random axis, deterministic
    angle with sin²(θ/2) = s̄  ⇒  q = (√(1−s̄), √s̄ · n̂)."""
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


def load_pile_batches(model_name: str, n_seqs: int, seq_len: int, batch: int,
                      device):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    ids_stream, seqs = [], []
    for row in ds:
        ids_stream.extend(tok(row["text"])["input_ids"])
        while len(ids_stream) >= seq_len:
            seqs.append(ids_stream[:seq_len])
            ids_stream = ids_stream[seq_len:]
            if len(seqs) >= n_seqs:
                break
        if len(seqs) >= n_seqs:
            break
    x = torch.tensor(seqs, dtype=torch.long)
    return [x[i:i + batch].to(device) for i in range(0, len(x), batch)]


@torch.no_grad()
def kl_curve_for_field(model, field, batches, device):
    """Per-position mean KL(program ‖ base), full vocab."""
    total = None
    count = 0
    for ids in batches:
        base_logits = model(input_ids=ids).logits
        with model.program(field):
            prog_logits = model(input_ids=ids).logits
        lp = F.log_softmax(prog_logits.float(), dim=-1)
        lb = F.log_softmax(base_logits.float(), dim=-1)
        kl = (lp.exp() * (lp - lb)).sum(-1)          # (B, T)
        s = kl.sum(0)
        total = s if total is None else total + s
        count += ids.shape[0]
    return (total / count).cpu()                      # (T,)


@torch.no_grad()
def attention_displacement(model, field, batches, device, n_seq=2):
    """Mean |Δattention| binned by relative distance, program vs base.
    Captures stream to CPU fp16 immediately (24 layers of 512×512 maps OOM a
    T4 if kept on-device). Also returns the base attention per bin so the
    displacement can be normalized (raw attention decays with distance)."""
    layers = model.base.gpt_neox.layers
    captured = {}

    def hook(_m, _i, out):
        captured.setdefault("attn", []).append(
            out[1].detach().to("cpu", torch.float16))

    ids = batches[0][:n_seq]
    T = ids.shape[1]
    bins = torch.zeros(T)
    base_bins = torch.zeros(T)
    norm = torch.zeros(T)

    def run_pass(program):
        captured.clear()
        handles = [l.attention.register_forward_hook(hook) for l in layers]
        try:
            if program is None:
                model(input_ids=ids, output_attentions=True)
            else:
                with model.program(program):
                    model(input_ids=ids, output_attentions=True)
        finally:
            for h in handles:
                h.remove()
        return list(captured["attn"])

    base_attn = run_pass(None)
    prog_attn = run_pass(field)
    for ab, ap in zip(base_attn, prog_attn):
        d = (ap.float() - ab.float()).abs()   # (B, H, T, T) on CPU
        b = ab.float()
        for delta in range(T):
            diag = torch.diagonal(d, offset=-delta, dim1=-2, dim2=-1)
            bins[delta] += diag.sum()
            base_bins[delta] += torch.diagonal(
                b, offset=-delta, dim1=-2, dim2=-1).sum()
            norm[delta] += diag.numel()
    n = norm.clamp_min(1)
    return (bins / n).tolist(), (base_bins / n).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--n-seqs", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="runs/b2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = LatentProgramModel.from_pretrained(args.model).to(device)
    print(f"spec: {model.spec}")
    batches = load_pile_batches(args.model, args.n_seqs, args.seq_len,
                                args.batch, device)
    print(f"{sum(b.shape[0] for b in batches)} sequences x {args.seq_len}")

    report = {"model": args.model, "sbars": list(SBARS), "curves": {},
              "mean_kl": {}, "leverage": {}}
    for ftype in FIELD_TYPES:
        if ftype == "rope_ax" and model.spec.rotary_ndims == 0:
            continue
        for sbar in SBARS:
            field = make_single_type_field(model.spec, ftype, sbar,
                                           seed=args.seed + hash((ftype, sbar)) % 10000)
            field = field.to(device)
            curve = kl_curve_for_field(model, field, batches, device)
            key = f"{ftype}@{sbar}"
            report["curves"][key] = [round(v, 6) for v in curve.tolist()]
            mean_kl = float(curve.mean())
            report["mean_kl"][key] = mean_kl
            report["leverage"][key] = mean_kl / sbar
            early = float(curve[:64].mean())
            late = float(curve[-64:].mean())
            print(f"{ftype:11s} s̄={sbar:<6} KL={mean_kl:.5f}  L={mean_kl/sbar:8.3f}  "
                  f"early64={early:.5f} late64={late:.5f} late/early={late/max(early,1e-9):.2f}")

    # KL table is the headline — persist before the (memory-hungrier) probe
    (out / "report_b2.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out / 'report_b2.json'} (KL table)")

    print("\nattention-displacement probe at s̄=0.01 (rope_ax vs qk_rel)")
    for ftype in ("rope_ax", "qk_rel"):
        field = make_single_type_field(model.spec, ftype, 0.01,
                                       seed=args.seed + hash((ftype, 0.01)) % 10000).to(device)
        disp, base_disp = attention_displacement(model, field, batches, device)
        report[f"attn_disp_{ftype}"] = [round(v, 8) for v in disp]
        report[f"attn_base_{ftype}"] = [round(v, 8) for v in base_disp]
        d = torch.tensor(disp)
        b = torch.tensor(base_disp)
        rel = d / b.clamp_min(1e-8)          # displacement relative to base attn
        print(f"{ftype:11s} |Δattn| near(1-16)={d[1:17].mean():.6f} "
              f"far(128-511)={d[128:].mean():.6f}  "
              f"RELATIVE near={rel[1:17].mean():.4f} far={rel[128:].mean():.4f} "
              f"rel-ratio={rel[128:].mean()/max(rel[1:17].mean(),1e-9):.2f}")

    (out / "report_b2.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out / 'report_b2.json'}")


if __name__ == "__main__":
    main()
