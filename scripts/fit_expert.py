#!/usr/bin/env python3
"""scripts/fit_expert.py — phase 1: per-expert direct fit by distillation
(spec §7). Learns a program z_k that reprograms the frozen base to imitate an
expert fine-tune:

    L = KL( p_expert || p_program )                       on expert-domain data
      + lambda_h  * sum_l MSE(hidden_l^expert, hidden_l^program)
      + lambda_reg * sum_sites d2_chord(q, identity)

Usage:
    python scripts/fit_expert.py --config configs/e1_experts.yaml \
        --task french --experts-dir experts --out runs/e1

Writes <out>/z_<task>.pt (ProgramField) and <out>/report_<task>.json with the
E1 metric: fraction of the expert-vs-base CE gap recovered, plus the
random-orthogonal control (should recover ~0%).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lpm import LatentProgramModel, LPMConfig, ProgramField  # noqa: E402
from lpm.tasks import TASKS  # noqa: E402
from lpm.utils import (batched, cosine_lr, get_device, lm_cross_entropy,  # noqa: E402
                       pack_texts, set_lr, set_seed)


def distill_loss(program_out, teacher_out, cfg: LPMConfig, field: ProgramField):
    logp = F.log_softmax(program_out.logits, dim=-1)
    p_t = F.softmax(teacher_out.logits, dim=-1)
    kl = F.kl_div(logp.flatten(0, 1), p_t.flatten(0, 1), reduction="batchmean")

    # spec §7: lambda_h * SUM_l MSE(hidden_l) — summed like the reg term, not
    # averaged (a mean would silently weaken the anchor by n_layers).
    h_mse = 0.0
    for ht, hp in zip(teacher_out.hidden_states[1:], program_out.hidden_states[1:]):
        h_mse = h_mse + F.mse_loss(hp, ht)

    reg = field.chordal_reg_to_identity()
    return kl + cfg.lambda_h * h_mse + cfg.lambda_reg * reg, kl, h_mse, reg


@torch.no_grad()
def program_ce(model: LatentProgramModel, field, blocks, device):
    with model.program(field):
        return lm_cross_entropy(model, blocks, device=device)


def fit(model: LatentProgramModel, teacher, blocks, cfg: LPMConfig, device,
        field: ProgramField, tag: str):
    opt = torch.optim.AdamW(field.parameters(), lr=cfg.optimizer.lr,
                            betas=cfg.optimizer.betas,
                            weight_decay=cfg.optimizer.weight_decay)
    steps = cfg.optimizer.steps
    step = 0
    while step < steps:
        perm = torch.randperm(blocks.shape[0])
        for idx in batched(perm.tolist(), cfg.optimizer.batch_size):
            if step >= steps:
                break
            b = blocks[idx].to(device)
            with torch.no_grad():
                t_out = teacher(input_ids=b, output_hidden_states=True)
            with model.program(field):
                p_out = model(input_ids=b, output_hidden_states=True)
            loss, kl, h_mse, reg = distill_loss(p_out, t_out, cfg, field)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(field.parameters(), cfg.optimizer.grad_clip)
            set_lr(opt, cosine_lr(step, steps, cfg.optimizer.lr,
                                  warmup=cfg.optimizer.warmup_steps))
            opt.step()
            if step % 50 == 0:
                print(f"[{tag}] step {step}/{steps} loss {loss.item():.4f} "
                      f"kl {kl.item():.4f} h {float(h_mse):.4f} reg {float(reg):.2e}")
            step += 1
    field.invalidate()
    return field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e1_experts.yaml")
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--experts-dir", default="experts")
    ap.add_argument("--out", default="runs/e1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control", action="store_true",
                    help="also fit from a random-orthogonal init (E1 control)")
    args = ap.parse_args()

    cfg = LPMConfig.from_yaml(args.config)
    set_seed(args.seed)
    device = get_device(args.device)
    task = TASKS[args.task]
    expert_dir = Path(args.experts_dir) / args.task / "merged"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = LatentProgramModel.from_pretrained(
        cfg.base_model, tie_qk_across_heads=cfg.tie_qk_across_heads)
    model.to(device)
    teacher = AutoModelForCausalLM.from_pretrained(str(expert_dir), dtype=torch.float32)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    train_blocks = pack_texts(task.texts("train", 4000), tok, seq_len=128, max_blocks=2000)
    eval_blocks = pack_texts(task.texts("eval", 400), tok, seq_len=128, max_blocks=64)

    ce_base = lm_cross_entropy(model, eval_blocks, device=device)
    ce_expert = lm_cross_entropy(teacher, eval_blocks, device=device)

    field = ProgramField.randn_near_identity(
        model.spec, sigma=cfg.field_init_sigma,
        enable_gains=cfg.enable_gains, trainable=True).to(device)
    field = fit(model, teacher, train_blocks, cfg, device, field, args.task)
    field.save(str(out_dir / f"z_{args.task}.pt"))

    ce_prog = program_ce(model, field, eval_blocks, device)
    gap = ce_base - ce_expert
    report = {
        "task": args.task, "base_model": cfg.base_model,
        "enable_gains": cfg.enable_gains,
        "ce_base": ce_base, "ce_expert": ce_expert, "ce_program": ce_prog,
        "gap_recovered": (ce_base - ce_prog) / gap if abs(gap) > 1e-9 else float("nan"),
    }

    if args.control:
        # random-orthogonal field, held fixed: should recover ~0% of the gap
        g = torch.Generator().manual_seed(args.seed + 1)
        rand_q = {k: torch.randn(*model.spec.site_shape(k[1]), 4, generator=g)
                  for k in model.spec.site_keys()}
        control = ProgramField(model.spec, rand_q).to(device)
        ce_ctrl = program_ce(model, control, eval_blocks, device)
        report["ce_random_orthogonal"] = ce_ctrl
        report["control_gap_recovered"] = ((ce_base - ce_ctrl) / gap
                                           if abs(gap) > 1e-9 else float("nan"))

    (out_dir / f"report_{args.task}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
