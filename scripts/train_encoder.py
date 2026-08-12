#!/usr/bin/env python3
"""scripts/train_encoder.py — phase 2: encoder amortization (spec §7), plus the
optional phase-3 compositional curriculum (--compositional, gated on E3).

Per episode: sample a task k and K in {4,16,64} demos; encode q_hat; loss

    L = lambda_geo  * sum_sites d2_chord(q_hat, q_k)      (phase-1 target z_k)
      + lambda_task * CE(task batch | program = q_hat)    (through frozen base)

Gradients flow through apply_rot into the encoder only. Phase 3 episodes are
ordered task pairs (a then b), supervised toward compose(z_b, z_a), plus the
end-task CE when ground-truth composed text exists (b has a text transform);
order is sampled both ways.

Usage:
    python scripts/train_encoder.py --config configs/e4_encoder.yaml \
        --experts-dir experts --z-dir runs/e1 --out runs/encoder \
        --tasks french caps jsonish [--compositional]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lpm import LatentProgramModel, LPMConfig, ProgramField, compose  # noqa: E402
from lpm.encoder import LPNEncoder, format_demos  # noqa: E402
from lpm.quaternion import d2_chord, q_normalize  # noqa: E402
from lpm.tasks import TASKS, composed_texts  # noqa: E402
from lpm.utils import cosine_lr, get_device, pack_texts, set_lr, set_seed  # noqa: E402


def load_demos(experts_dir: Path, task: str):
    rows = [json.loads(l) for l in
            (experts_dir / task / "demos.jsonl").read_text(encoding="utf-8").splitlines()]
    return ([(r["input"], r["output"]) for r in rows if r["split"] == "train"],
            [(r["input"], r["output"]) for r in rows if r["split"] == "eval"])


def geo_loss(field_hat: ProgramField, target: ProgramField) -> torch.Tensor:
    total = None
    for k in field_hat.sites():
        term = d2_chord(q_normalize(field_hat.q(*k)),
                        q_normalize(target.q(*k).to(field_hat.q(*k).device))).sum()
        total = term if total is None else total + term
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e4_encoder.yaml")
    ap.add_argument("--experts-dir", default="experts")
    ap.add_argument("--z-dir", default="runs/e1")
    ap.add_argument("--out", default="runs/encoder")
    ap.add_argument("--tasks", nargs="+", default=["french", "caps", "jsonish", "sentiment"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--k-choices", nargs="+", type=int, default=[4, 16, 64])
    ap.add_argument("--task-batch", type=int, default=4)
    ap.add_argument("--compositional", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = LPMConfig.from_yaml(args.config)
    set_seed(args.seed)
    device = get_device(args.device)
    rng = random.Random(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    experts_dir = Path(args.experts_dir)

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = LatentProgramModel.from_pretrained(
        cfg.base_model, tie_qk_across_heads=cfg.tie_qk_across_heads)
    model.to(device)

    targets, demo_train, task_blocks = {}, {}, {}
    for t in args.tasks:
        targets[t] = ProgramField.load(str(Path(args.z_dir) / f"z_{t}.pt")).to(device)
        demo_train[t], _ = load_demos(experts_dir, t)
        task_blocks[t] = pack_texts(TASKS[t].texts("train", 1500), tok,
                                    seq_len=128, max_blocks=500)

    pair_targets, pair_blocks = {}, {}
    if args.compositional:
        for a in args.tasks:
            for b in args.tasks:
                if a == b:
                    continue
                pair_targets[(a, b)] = compose(targets[b], targets[a])  # a first, then b
                txt = composed_texts(TASKS[a], TASKS[b], "train", 800)
                if txt is not None:
                    pair_blocks[(a, b)] = pack_texts(txt, tok, seq_len=128, max_blocks=300)

    enc = LPNEncoder(model.spec, vocab_size=model.config.vocab_size,
                     pad_token_id=tok.pad_token_id,
                     enable_gains=cfg.enable_gains).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=0.01)

    for step in range(args.steps):
        use_pair = args.compositional and pair_targets and rng.random() < 0.5
        if use_pair:
            a, b = rng.choice(list(pair_targets))
            K = rng.choice(args.k_choices)
            demos = (rng.sample(demo_train[a], min(K // 2 + 1, len(demo_train[a]))) +
                     rng.sample(demo_train[b], min(K // 2 + 1, len(demo_train[b]))))
            target = pair_targets[(a, b)]
            blocks = pair_blocks.get((a, b))
        else:
            t = rng.choice(args.tasks)
            K = rng.choice(args.k_choices)
            demos = rng.sample(demo_train[t], min(K, len(demo_train[t])))
            target = targets[t]
            blocks = task_blocks[t]

        ids, mask = format_demos(demos, tok, max_len=1024)
        field_hat = enc(ids.to(device), mask.to(device))[0]

        loss = cfg.lambda_geo * geo_loss(field_hat, target)
        if blocks is not None and cfg.lambda_task > 0:
            idx = torch.randint(0, blocks.shape[0], (args.task_batch,))
            batch = blocks[idx].to(device)
            with model.program(field_hat):
                task_ce = model(input_ids=batch, labels=batch).loss
            loss = loss + cfg.lambda_task * task_ce

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), cfg.optimizer.grad_clip)
        set_lr(opt, cosine_lr(step, args.steps, args.lr, warmup=100))
        opt.step()
        if step % 50 == 0:
            kind = "pair" if use_pair else "task"
            print(f"step {step}/{args.steps} [{kind}] K={K} loss {loss.item():.4f}")

    torch.save({"encoder": enc.state_dict(),
                "spec": model.spec.__dict__,
                "config": cfg.to_dict(),
                "tasks": args.tasks,
                "compositional": args.compositional},
               out_dir / "encoder.pt")
    print(f"saved {out_dir / 'encoder.pt'}")


if __name__ == "__main__":
    main()
