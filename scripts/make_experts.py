#!/usr/bin/env python3
"""scripts/make_experts.py — LoRA fine-tunes of the base on contrastive tasks
(spec §7 phase 1 prerequisites, §10 E1). GPU recommended (LightningAI studio).

For each task this saves under <out>/<task>/:
    adapter/       the PEFT LoRA adapter
    merged/        merge_and_unload()-ed full checkpoint (audit_mirsky teacher,
                   fit_expert distillation teacher, E2 weight-space baselines)
    demos.jsonl    (input, output) demo pairs for LPN episodes
    report.json    base vs expert CE on held-out task text (E1 gap denominator)

Usage:
    python scripts/make_experts.py --base gpt2 --out experts \
        --tasks french caps jsonish sentiment --steps 800
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lpm.tasks import TASKS  # noqa: E402
from lpm.utils import (batched, cosine_lr, get_device, lm_cross_entropy,  # noqa: E402
                       pack_texts, set_lr, set_seed)


def train_expert(args, task, device):
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    base.to(device)

    train_blocks = pack_texts(task.texts("train", args.n_texts), tok,
                              seq_len=args.seq_len, max_blocks=args.max_blocks)
    eval_blocks = pack_texts(task.texts("eval", 400), tok,
                             seq_len=args.seq_len, max_blocks=64)

    base.eval()
    ce_base = lm_cross_entropy(base, eval_blocks, device=device)

    lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r,
                      lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                      target_modules=["c_attn", "c_proj", "c_fc"])
    model = get_peft_model(base, lora)
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr, weight_decay=0.0)

    step = 0
    while step < args.steps:
        perm = torch.randperm(train_blocks.shape[0])
        for idx in batched(perm.tolist(), args.batch_size):
            if step >= args.steps:
                break
            b = train_blocks[idx].to(device)
            loss = model(input_ids=b, labels=b).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            set_lr(opt, cosine_lr(step, args.steps, args.lr, warmup=20))
            opt.step()
            if step % 50 == 0:
                print(f"[{task.name}] step {step}/{args.steps} loss {loss.item():.4f}")
            step += 1

    out = Path(args.out) / task.name
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out / "adapter"))

    model.eval()
    merged = model.merge_and_unload()
    ce_expert = lm_cross_entropy(merged, eval_blocks, device=device)
    merged.save_pretrained(str(out / "merged"))
    tok.save_pretrained(str(out / "merged"))

    # label by actual membership — demo_pairs returns UP TO n pairs, so an
    # index-based "i < 512" label would mislabel eval demos as train whenever
    # the train pool comes up short
    with open(out / "demos.jsonl", "w", encoding="utf-8") as f:
        for split_name, n_pairs in (("train", 512), ("eval", 128)):
            for x, y in task.demo_pairs(split_name, n_pairs):
                f.write(json.dumps({"input": x, "output": y,
                                    "split": split_name}) + "\n")

    report = {"task": task.name, "base": args.base, "steps": args.steps,
              "lora_r": args.lora_r, "ce_base": ce_base, "ce_expert": ce_expert,
              "gap": ce_base - ce_expert}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[{task.name}] CE base {ce_base:.4f} -> expert {ce_expert:.4f} "
          f"(gap {ce_base - ce_expert:+.4f}); saved to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="gpt2")
    ap.add_argument("--out", default="experts")
    ap.add_argument("--tasks", nargs="+", default=["french", "caps", "jsonish", "sentiment"],
                    choices=sorted(TASKS))
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--n-texts", type=int, default=4000)
    ap.add_argument("--max-blocks", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"device: {device}")
    for name in args.tasks:
        train_expert(args, TASKS[name], device)


if __name__ == "__main__":
    main()
