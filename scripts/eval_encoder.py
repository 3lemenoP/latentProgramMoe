#!/usr/bin/env python3
"""scripts/eval_encoder.py — E4: encoder generalization (spec §10).

Conditions, reported vs K in {4, 16, 64} demos on a held-out task (train the
encoder with --tasks excluding it, e.g. leave-one-out):

    encoder        q_hat(demos), T=0 refinement (eval parity default)
    encoder+refine q_hat(demos) then refine T=50 on the demos
    refine-only    refine T=50 from the identity program (no encoder)
    oracle         the task's phase-1 direct fit z_k (upper bound for programs)
    lora           the task's LoRA expert itself (upper baseline)
    base           frozen base, no program

Metric: CE on held-out task text (lower is better).

Usage:
    python scripts/eval_encoder.py --config configs/e4_encoder.yaml \
        --encoder runs/encoder/encoder.pt --task sentiment \
        --experts-dir experts --z-dir runs/e1 --out report_e4_sentiment.md
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lpm import LatentProgramModel, LPMConfig, ProgramField  # noqa: E402
from lpm.encoder import LPNEncoder, build_demo_lm_batch, format_demos, refine  # noqa: E402
from lpm.tasks import TASKS  # noqa: E402
from lpm.utils import get_device, lm_cross_entropy, pack_texts, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e4_encoder.yaml")
    ap.add_argument("--encoder", default="runs/encoder/encoder.pt")
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--experts-dir", default="experts")
    ap.add_argument("--z-dir", default="runs/e1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--k-choices", nargs="+", type=int, default=[4, 16, 64])
    ap.add_argument("--refine-steps", type=int, default=50)
    ap.add_argument("--episodes", type=int, default=5,
                    help="demo resamples per K (mean CE reported)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = LPMConfig.from_yaml(args.config)
    set_seed(args.seed)
    device = get_device(args.device)
    rng = random.Random(args.seed)
    out_path = args.out or f"report_e4_{args.task}.md"

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = LatentProgramModel.from_pretrained(
        cfg.base_model, tie_qk_across_heads=cfg.tie_qk_across_heads)
    model.to(device)

    ckpt = torch.load(args.encoder, map_location="cpu", weights_only=False)
    enc = LPNEncoder(model.spec, vocab_size=model.config.vocab_size,
                     pad_token_id=tok.pad_token_id,
                     enable_gains=cfg.enable_gains).to(device)
    enc.load_state_dict(ckpt["encoder"])
    enc.eval()
    if args.task in ckpt.get("tasks", []):
        print(f"WARNING: task {args.task!r} was in the encoder's training set "
              f"(not held out) — E4 wants leave-one-out")

    demos_all = [json.loads(l) for l in
                 (Path(args.experts_dir) / args.task / "demos.jsonl")
                 .read_text(encoding="utf-8").splitlines()]
    demo_pool = [(r["input"], r["output"]) for r in demos_all if r["split"] == "train"]
    eval_blocks = pack_texts(TASKS[args.task].texts("eval", 400), tok, 128, max_blocks=48)

    def ce(field):
        with model.program(field):
            return lm_cross_entropy(model, eval_blocks, device=device)

    ce_base = lm_cross_entropy(model, eval_blocks, device=device)
    oracle = ProgramField.load(str(Path(args.z_dir) / f"z_{args.task}.pt")).to(device)
    ce_oracle = ce(oracle)
    lora = AutoModelForCausalLM.from_pretrained(
        str(Path(args.experts_dir) / args.task / "merged"), dtype=torch.float32)
    lora.to(device).eval()
    ce_lora = lm_cross_entropy(lora, eval_blocks, device=device)
    del lora

    lines = [f"# E4 — encoder generalization: held-out `{args.task}`\n",
             f"base CE {ce_base:.4f} | oracle (phase-1 z) {ce_oracle:.4f} | "
             f"LoRA expert {ce_lora:.4f}\n",
             "| K | encoder (T=0) | encoder+refine (T=50) | refine-only (T=50) |",
             "|---|---|---|---|"]

    for K in args.k_choices:
        ces = {"enc": [], "enc_ref": [], "ref": []}
        for _ in range(args.episodes):
            demos = rng.sample(demo_pool, min(K, len(demo_pool)))
            ids, mask = format_demos(demos, tok, max_len=1024)
            with torch.no_grad():
                f_hat = enc(ids.to(device), mask.to(device))[0].detached()
            f_hat.to(device)
            ces["enc"].append(ce(f_hat))

            d_ids, d_lab, d_attn = build_demo_lm_batch(demos, tok, device=device)
            f_ref = refine(model, f_hat, d_ids, d_lab, attention_mask=d_attn,
                           steps=args.refine_steps, lr=cfg.refine_lr)
            ces["enc_ref"].append(ce(f_ref))

            f_id = ProgramField.identity(model.spec).to(device)
            f_ref0 = refine(model, f_id, d_ids, d_lab, attention_mask=d_attn,
                            steps=args.refine_steps, lr=cfg.refine_lr)
            ces["ref"].append(ce(f_ref0))

        def mean(xs):
            return sum(xs) / len(xs)
        lines.append(f"| {K} | {mean(ces['enc']):.4f} | {mean(ces['enc_ref']):.4f} "
                     f"| {mean(ces['ref']):.4f} |")
        print(lines[-1])

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
