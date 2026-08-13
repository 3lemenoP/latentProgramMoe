#!/usr/bin/env python3
"""scripts/eval_order.py — E3: order sensitivity, the falsifier (spec §10).

Pipeline (self-contained; CPU-feasible, GPU faster):
 1. Train a small GPT-2 from scratch on a synthetic seq2seq MIXTURE of
    behaviors {copy, prepend-<A>, reverse} (format [BOS] x [SEP] y [EOS]).
    The base is deliberately ambiguous about which behavior to produce.
 2. Direct-fit programs z_a (prepend) and z_b (reverse) on behavior-pure data
    with the base frozen (CE on the y tokens; spec §7-style field training).
 3. Zero-shot evaluate compose(z_b, z_a) ("a then b" => y = reverse(x)+[A])
    and compose(z_a, z_b) ("b then a" => y = [A]+reverse(x)) against the
    ordered ground truths. Neither composed behavior was ever trained.
 4. Abelian baseline: every site constrained to a fixed learned axis
    (q = (cos t/2, sin t/2 * a_site)), axes SHARED between the two skills,
    theta the only per-skill freedom — all programs commute, composition is
    provably order-blind.

Pass (spec): non-abelian ordered accuracy - abelian ordered accuracy > 20
points, and swapped-order predictions differ in the right direction.

Usage:
    python scripts/eval_order.py --config configs/e3_order.yaml --out report_e3.md
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lpm import (AbelianProgramField, AxisBank, LatentProgramModel,  # noqa: E402
                 LPMConfig, ProgramField, compose)
from lpm.tasks import E3Vocab, e3_examples, e3_output  # noqa: E402
from lpm.utils import cosine_lr, get_device, set_lr, set_seed  # noqa: E402


def make_base(vocab: E3Vocab, device) -> GPT2LMHeadModel:
    cfg = GPT2Config(vocab_size=vocab.size, n_positions=64, n_embd=96,
                     n_layer=4, n_head=4, n_inner=384,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
                     bos_token_id=vocab.BOS, eos_token_id=vocab.EOS,
                     pad_token_id=vocab.PAD)
    cfg._attn_implementation = "eager"
    return GPT2LMHeadModel(cfg).to(device)


def train_base(model, vocab, device, steps=3000, batch=64, lr=3e-4, seed=0):
    ids_all, lab_all, attn_all = [], [], []
    for i, beh in enumerate(("copy", "prepend", "reverse")):
        ids, lab, attn = e3_examples(beh, 4000, vocab, seed=seed + i)
        ids_all.append(ids); lab_all.append(lab); attn_all.append(attn)
    T = max(x.shape[1] for x in ids_all)

    def padT(x, fill):
        return torch.nn.functional.pad(x, (0, T - x.shape[1]), value=fill)
    ids = torch.cat([padT(x, vocab.PAD) for x in ids_all])
    lab = torch.cat([padT(x, -100) for x in lab_all])
    attn = torch.cat([padT(x, 0) for x in attn_all])

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, ids.shape[0], (batch,), generator=g)
        out = model(input_ids=ids[idx].to(device),
                    attention_mask=attn[idx].to(device),
                    labels=lab[idx].to(device))
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        set_lr(opt, cosine_lr(step, steps, lr, warmup=100))
        opt.step()
        if step % 200 == 0:
            print(f"[base] step {step}/{steps} loss {out.loss.item():.4f}")
    model.eval()


def fit_field(model: LatentProgramModel, field, behavior, vocab, device,
              steps=1500, batch=64, lr=1e-3, seed=1, extra_params=(), tag=""):
    ids, lab, attn = e3_examples(behavior, 4000, vocab, seed=seed)
    # dedupe by identity: AbelianProgramField already registers the shared
    # AxisBank as a submodule, so extra_params may overlap field.parameters()
    params = list(dict.fromkeys(list(field.parameters()) + list(extra_params)))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, ids.shape[0], (batch,), generator=g)
        with model.program(field):
            out = model(input_ids=ids[idx].to(device),
                        attention_mask=attn[idx].to(device),
                        labels=lab[idx].to(device))
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        set_lr(opt, cosine_lr(step, steps, lr, warmup=50))
        opt.step()
        if step % 200 == 0:
            print(f"[{tag}] step {step}/{steps} loss {out.loss.item():.4f}")
    field.invalidate()
    return field


def fit_fields_joint(model: LatentProgramModel, pairs, vocab, device,
                     steps=1500, batch=64, lr=1e-3, seed=1, extra_params=(),
                     tag=""):
    """Fit several (field, behavior) pairs JOINTLY: one optimizer over all
    field params plus extra_params (the shared AxisBank), one batch per
    behavior per step, losses summed before the step.

    This is the fair protocol for the abelian baseline: a sequential fit
    (ta first, then tb) lets tb's fit move the SHARED axes out from under
    ta's already-frozen thetas, so any ta degradation would be an artifact
    of the protocol rather than of the abelian constraint the baseline is
    meant to isolate. (A scaled-down probe measured only ~1.5 deg mean
    axis drift at 200-step fits, but drift grows with fit length and the
    falsifier must not depend on it being benign.) Joint fitting keeps
    every theta consistent with the final shared axes, so the baseline's
    ordered-composition score reflects commutativity alone."""
    data = []
    for i, (field, behavior) in enumerate(pairs):
        ids, lab, attn = e3_examples(behavior, 4000, vocab, seed=seed + i)
        data.append((field, ids, lab, attn))
    params = [p for field, *_ in data for p in field.parameters()]
    params = list(dict.fromkeys(params + list(extra_params)))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        opt.zero_grad()
        total = 0.0
        for field, ids, lab, attn in data:
            idx = torch.randint(0, ids.shape[0], (batch,), generator=g)
            with model.program(field):
                out = model(input_ids=ids[idx].to(device),
                            attention_mask=attn[idx].to(device),
                            labels=lab[idx].to(device))
            out.loss.backward()
            total += out.loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        set_lr(opt, cosine_lr(step, steps, lr, warmup=50))
        opt.step()
        if step % 200 == 0:
            print(f"[{tag}] step {step}/{steps} mean loss {total / len(data):.4f}")
    for field, *_ in data:
        field.invalidate()
    return [field for field, *_ in data]


@torch.no_grad()
def ordered_accuracy(model: LatentProgramModel, field, behavior, vocab, device,
                     n=200, seed=99):
    """Greedy-decode y from [BOS] x [SEP]; exact match + token accuracy."""
    import random
    rng = random.Random(seed)
    exact, tok_hits, tok_total = 0, 0, 0
    ctx = model.program(field) if field is not None else model.program(None)
    with ctx:
        for _ in range(n):
            x = vocab.payload(rng)
            want = e3_output(x, behavior, vocab) + [vocab.EOS]
            ids = torch.tensor([[vocab.BOS] + x + [vocab.SEP]], device=device)
            for _step in range(len(want) + 2):
                logits = model(input_ids=ids).logits[0, -1]
                nxt = int(logits.argmax())
                ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
                if nxt == vocab.EOS:
                    break
            got = ids[0, len(x) + 2:].tolist()
            if got == want:
                exact += 1
            L = min(len(got), len(want))
            tok_hits += sum(int(a == b) for a, b in zip(got[:L], want[:L]))
            tok_total += len(want)
    return exact / n, tok_hits / max(tok_total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e3_order.yaml")
    ap.add_argument("--out", default="report_e3.md")
    ap.add_argument("--base-steps", type=int, default=3000)
    ap.add_argument("--fit-steps", type=int, default=1500)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = LPMConfig.from_yaml(args.config) if Path(args.config).exists() else LPMConfig()
    set_seed(args.seed)
    device = get_device(args.device)
    vocab = E3Vocab()

    print("=== 1. base model (mixture of copy/prepend/reverse) ===")
    base = make_base(vocab, device)
    train_base(base, vocab, device, steps=args.base_steps, seed=args.seed)
    model = LatentProgramModel(base)

    print("=== 2. direct-fit z_a (prepend), z_b (reverse) ===")
    za = ProgramField.randn_near_identity(model.spec, sigma=cfg.field_init_sigma,
                                          trainable=True).to(device)
    za = fit_field(model, za, "prepend", vocab, device, steps=args.fit_steps,
                   seed=args.seed + 1, tag="z_a")
    zb = ProgramField.randn_near_identity(model.spec, sigma=cfg.field_init_sigma,
                                          trainable=True).to(device)
    zb = fit_field(model, zb, "reverse", vocab, device, steps=args.fit_steps,
                   seed=args.seed + 2, tag="z_b")

    print("=== 3. abelian baseline (shared axes, theta-only) ===")
    bank = AxisBank(model.spec).to(device)
    ta = AbelianProgramField(bank).to(device)
    tb = AbelianProgramField(bank).to(device)
    # JOINT fit (one optimizer, both tasks every step): a sequential fit would
    # let tb's updates move the shared axes out from under ta's thetas —
    # see fit_fields_joint docstring.
    ta, tb = fit_fields_joint(model, [(ta, "prepend"), (tb, "reverse")], vocab,
                              device, steps=args.fit_steps, seed=args.seed + 3,
                              extra_params=bank.parameters(), tag="abelian joint")

    print("=== 4. zero-shot ordered evaluation ===")
    rows = {}
    rows["z_a on prepend"] = ordered_accuracy(model, za, "prepend", vocab, device)
    rows["z_b on reverse"] = ordered_accuracy(model, zb, "reverse", vocab, device)
    ab = compose(zb, za)   # a first, then b -> reverse(x) + [A]
    ba = compose(za, zb)   # b first, then a -> [A] + reverse(x)
    rows["compose(b,a) on a_then_b"] = ordered_accuracy(model, ab, "a_then_b", vocab, device)
    rows["compose(a,b) on b_then_a"] = ordered_accuracy(model, ba, "b_then_a", vocab, device)
    rows["compose(b,a) on b_then_a (swapped)"] = ordered_accuracy(model, ab, "b_then_a", vocab, device)
    rows["compose(a,b) on a_then_b (swapped)"] = ordered_accuracy(model, ba, "a_then_b", vocab, device)

    tab_ = compose(tb.to_program_field(), ta.to_program_field())
    tba_ = compose(ta.to_program_field(), tb.to_program_field())
    rows["abelian ta on prepend"] = ordered_accuracy(model, ta, "prepend", vocab, device)
    rows["abelian tb on reverse"] = ordered_accuracy(model, tb, "reverse", vocab, device)
    rows["abelian compose(b,a) on a_then_b"] = ordered_accuracy(model, tab_, "a_then_b", vocab, device)
    rows["abelian compose(a,b) on b_then_a"] = ordered_accuracy(model, tba_, "b_then_a", vocab, device)

    ordered_nonabelian = 0.5 * (rows["compose(b,a) on a_then_b"][0] +
                                rows["compose(a,b) on b_then_a"][0])
    ordered_abelian = 0.5 * (rows["abelian compose(b,a) on a_then_b"][0] +
                             rows["abelian compose(a,b) on b_then_a"][0])
    swapped = 0.5 * (rows["compose(b,a) on b_then_a (swapped)"][0] +
                     rows["compose(a,b) on a_then_b (swapped)"][0])
    margin = 100 * (ordered_nonabelian - ordered_abelian)
    right_direction = ordered_nonabelian > swapped
    verdict = "PASS" if (margin > 20 and right_direction) else "FAIL"

    lines = ["# E3 — order sensitivity (falsifier)\n",
             "| condition | exact match | token acc |", "|---|---|---|"]
    for k, (em, ta_) in rows.items():
        lines.append(f"| {k} | {em:.3f} | {ta_:.3f} |")
    lines += ["",
              f"- ordered accuracy, non-abelian: **{100*ordered_nonabelian:.1f}%**",
              f"- ordered accuracy, abelian baseline: **{100*ordered_abelian:.1f}%**",
              f"- margin: **{margin:.1f} points** (pass needs > 20)",
              f"- swapped-order accuracy (should be lower): {100*swapped:.1f}% "
              f"-> right direction: {right_direction}",
              f"\n**{verdict}** — phase 3 (compositional curriculum) is "
              f"{'unlocked' if verdict == 'PASS' else 'blocked'} (spec §7)."]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    (Path(args.out).with_suffix(".json")).write_text(
        json.dumps({k: {"exact": v[0], "token": v[1]} for k, v in rows.items()},
                   indent=2))


if __name__ == "__main__":
    main()
