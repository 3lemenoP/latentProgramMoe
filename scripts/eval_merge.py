#!/usr/bin/env python3
"""scripts/eval_merge.py — E2: geodesic merging vs weight-space baselines
(spec §10). For expert pair (A, B):

  program slerp:   slerp_field(z_A, z_B, alpha)         (exact geodesic, §6)
  weight linear:   W = (1-alpha) W_A + alpha W_B         (merged checkpoints)
  weight slerp:    W = W_0 + slerp of the deltas (spherical interpolation of
                   (W_A - W_0, W_B - W_0) treated as one flat vector each)
  per-layer sched: early layers pinned to A (alpha_l = 0 for l < L/2),
                   late layers interpolated (program slerp only)

Metrics at alpha in {0, .25, .5, .75, 1}: CE on task-A text, CE on task-B
text, and perplexity spike on neutral wikitext. Output: markdown report.

Usage:
    python scripts/eval_merge.py --config configs/e2_merge.yaml \
        --task-a french --task-b caps --experts-dir experts --z-dir runs/e1 \
        --out report_e2_french_caps.md
"""
import argparse
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lpm import LatentProgramModel, LPMConfig, ProgramField, slerp_field  # noqa: E402
from lpm.tasks import TASKS, _wikitext  # noqa: E402
from lpm.utils import get_device, lm_cross_entropy, pack_texts, set_seed  # noqa: E402

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def weight_interp(base_sd, sd_a, sd_b, alpha: float, mode: str):
    """Interpolated full state dict. mode: linear | slerp (on deltas)."""
    if mode == "linear":
        return {k: (1 - alpha) * sd_a[k] + alpha * sd_b[k] for k in sd_a}
    da = torch.cat([(sd_a[k] - base_sd[k]).flatten() for k in sorted(sd_a)])
    db = torch.cat([(sd_b[k] - base_sd[k]).flatten() for k in sorted(sd_a)])
    na, nb = da.norm(), db.norm()
    cos = (da @ db / (na * nb)).clamp(-1 + 1e-7, 1 - 1e-7)
    omega = torch.acos(cos)
    if omega.abs() < 1e-6:
        mix = (1 - alpha) * da + alpha * db
    else:
        mix = (torch.sin((1 - alpha) * omega) * da + torch.sin(alpha * omega) * db) / torch.sin(omega)
    out, off = {}, 0
    for k in sorted(sd_a):
        n = sd_a[k].numel()
        out[k] = base_sd[k] + mix[off:off + n].view_as(sd_a[k])
        off += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e2_merge.yaml")
    ap.add_argument("--task-a", required=True, choices=sorted(TASKS))
    ap.add_argument("--task-b", required=True, choices=sorted(TASKS))
    ap.add_argument("--experts-dir", default="experts")
    ap.add_argument("--z-dir", default="runs/e1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = LPMConfig.from_yaml(args.config)
    set_seed(args.seed)
    device = get_device(args.device)
    out_path = args.out or f"report_e2_{args.task_a}_{args.task_b}.md"

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = LatentProgramModel.from_pretrained(
        cfg.base_model, tie_qk_across_heads=cfg.tie_qk_across_heads)
    model.to(device)
    za = ProgramField.load(str(Path(args.z_dir) / f"z_{args.task_a}.pt")).to(device)
    zb = ProgramField.load(str(Path(args.z_dir) / f"z_{args.task_b}.pt")).to(device)

    blocks_a = pack_texts(TASKS[args.task_a].texts("eval", 400), tok, 128, max_blocks=48)
    blocks_b = pack_texts(TASKS[args.task_b].texts("eval", 400), tok, 128, max_blocks=48)
    blocks_n = pack_texts(_wikitext("eval", 400), tok, 128, max_blocks=48)

    def metrics_program(field):
        with model.program(field):
            return (lm_cross_entropy(model, blocks_a, device=device),
                    lm_cross_entropy(model, blocks_b, device=device),
                    math.exp(lm_cross_entropy(model, blocks_n, device=device)))

    lines = [f"# E2 — geodesic merging: `{args.task_a}` + `{args.task_b}`\n",
             f"Base `{cfg.base_model}`; program slerp vs weight-space baselines.\n",
             "| method | alpha | CE A | CE B | neutral ppl |", "|---|---|---|---|---|"]

    L = model.spec.n_layers
    sched = torch.tensor([0.0] * (L // 2) + [1.0] * (L - L // 2))
    for alpha in ALPHAS:
        ca, cb, ppl = metrics_program(slerp_field(za, zb, alpha))
        lines.append(f"| program slerp | {alpha} | {ca:.4f} | {cb:.4f} | {ppl:.2f} |")
    for alpha in ALPHAS:
        ca, cb, ppl = metrics_program(slerp_field(za, zb, alpha * sched))
        lines.append(f"| program slerp (early→A) | {alpha} | {ca:.4f} | {cb:.4f} | {ppl:.2f} |")

    # weight-space baselines on the merged checkpoints
    base_full = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=torch.float32)
    sd0 = {k: v.clone() for k, v in base_full.state_dict().items()}
    sd_a = AutoModelForCausalLM.from_pretrained(
        str(Path(args.experts_dir) / args.task_a / "merged"), dtype=torch.float32).state_dict()
    sd_b = AutoModelForCausalLM.from_pretrained(
        str(Path(args.experts_dir) / args.task_b / "merged"), dtype=torch.float32).state_dict()

    for mode in ("linear", "slerp"):
        for alpha in ALPHAS:
            base_full.load_state_dict(weight_interp(sd0, sd_a, sd_b, alpha, mode))
            base_full.to(device).eval()
            ca = lm_cross_entropy(base_full, blocks_a, device=device)
            cb = lm_cross_entropy(base_full, blocks_b, device=device)
            ppl = math.exp(lm_cross_entropy(base_full, blocks_n, device=device))
            lines.append(f"| weight {mode} | {alpha} | {ca:.4f} | {cb:.4f} | {ppl:.2f} |")
            base_full.to("cpu")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
