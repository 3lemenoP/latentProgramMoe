#!/usr/bin/env python3
"""E3′ D2–D5 runner (e-suite-analysis-e3prime.md §4).

    python scripts/e3prime.py --stage d2 --dir runs/e3
    python scripts/e3prime.py --stage d3 --dir runs/e3
    python scripts/e3prime.py --stage d4 --dir runs/e3
    python scripts/e3prime.py --stage d5 --dir runs/e3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lpm import (AbelianProgramField, AxisBank, LatentProgramModel,  # noqa: E402
                 ProgramField, compose, increment, invert_field, pow_field)
from lpm.quaternion import d_geo, q_normalize  # noqa: E402
from lpm.tasks import E3Vocab  # noqa: E402
from lpm.utils import get_device, set_seed  # noqa: E402
from e3_common import (  # noqa: E402
    fit_field, fit_fields_joint, load_base, ordered_accuracy,
)


def _load_wrap(d: Path, device):
    base = load_base(str(d / "base.pt"), device)
    return LatentProgramModel(base)


def _mean_dgeo(fa: ProgramField, fb: ProgramField) -> float:
    xs = []
    for k in fa.spec.site_keys():
        xs.append(d_geo(q_normalize(fa.q(*k).detach().float()),
                        q_normalize(fb.q(*k).detach().float())).mean())
    return float(torch.stack(xs).mean())


@torch.no_grad()
def _mean_kl_vs(model, cand: ProgramField, oracle: ProgramField, vocab, device,
                behavior="copy", n=64, seed=7):
    """Behavioral distance: mean KL(candidate-model ‖ oracle-model) on a probe
    set (handoff §3.2d — separates geometric from behavioral distance)."""
    import torch.nn.functional as F
    from lpm.tasks import e3_examples
    ids, _, attn = e3_examples(behavior, n, vocab, seed=seed)
    ids, attn = ids.to(device), attn.to(device)
    with model.program(cand):
        lc = model(input_ids=ids, attention_mask=attn).logits
    with model.program(oracle):
        lo = model(input_ids=ids, attention_mask=attn).logits
    log_pc = torch.log_softmax(lc.float(), dim=-1)
    log_po = torch.log_softmax(lo.float(), dim=-1)
    # KL(cand ‖ oracle) = sum_c pc * (log pc - log po), masked to real tokens
    kl_tok = (log_pc.exp() * (log_pc - log_po)).sum(-1)
    m = attn.bool()
    return float(kl_tok[m].mean())


@torch.no_grad()
def _decode_failures(model, field: ProgramField, behavior, vocab, device,
                     n_dump=20, n=200, seed=99):
    """Greedy-decode failures classified per handoff §3.2e:
    first-token error / reversal error / length error / other."""
    import random
    from lpm.tasks import e3_output
    rng = random.Random(seed)
    dumps = []
    counts = {"first_token": 0, "reversal": 0, "length": 0, "other": 0}
    with model.program(field):
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
                continue
            if len(got) != len(want):
                kind = "length"
            elif got[0] != want[0]:
                kind = "first_token"
            elif sorted(got) == sorted(want):
                kind = "reversal"  # right multiset, wrong order
            else:
                kind = "other"
            counts[kind] += 1
            if len(dumps) < n_dump:
                dumps.append({"input": x, "want": want, "got": got, "class": kind})
    return counts, dumps


def stage_d2(args, model, vocab, device, d: Path):
    za = ProgramField.load(str(d / "z_a.pt"), trainable=False).to(device)
    zb = ProgramField.load(str(d / "z_b.pt"), trainable=False).to(device)
    print("=== D2 oracle z_ab / z_ba ===")
    zab = ProgramField.randn_near_identity(model.spec, sigma=1e-3, trainable=True).to(device)
    zba = ProgramField.randn_near_identity(model.spec, sigma=1e-3, trainable=True).to(device)
    zab = fit_field(model, zab, "a_then_b", vocab, device, steps=args.fit_steps,
                    seed=args.seed + 10, tag="z_ab")
    zba = fit_field(model, zba, "b_then_a", vocab, device, steps=args.fit_steps,
                    seed=args.seed + 11, tag="z_ba")
    zab.save(str(d / "z_ab.pt"))
    zba.save(str(d / "z_ba.pt"))
    ident = ProgramField.identity(model.spec).to(device)
    ham = compose(zb, za)
    e_ab = ordered_accuracy(model, zab, "a_then_b", vocab, device)
    e_ba = ordered_accuracy(model, zba, "b_then_a", vocab, device)
    report = {
        "z_ab_exact": e_ab[0], "z_ab_token": e_ab[1],
        "z_ba_exact": e_ba[0], "z_ba_token": e_ba[1],
        # geometric distances to the z_ab oracle
        "d_geo(hamilton(b,a), z_ab)": _mean_dgeo(ham, zab),
        "d_geo(z_a, z_ab)": _mean_dgeo(za, zab),
        "d_geo(z_b, z_ab)": _mean_dgeo(zb, zab),
        "d_geo(id, z_ab)": _mean_dgeo(ident, zab),
        # anchor distances (handoff §3.2c): scale for the table above
        "d_geo(id, z_a)": _mean_dgeo(ident, za),
        "d_geo(id, z_b)": _mean_dgeo(ident, zb),
        "d_geo(id, hamilton(b,a))": _mean_dgeo(ident, ham),
        # behavioral column (handoff §3.2d): KL(candidate ‖ z_ab oracle) on probe
        "kl(hamilton(b,a) || z_ab)": _mean_kl_vs(model, ham, zab, vocab, device),
        "kl(z_a || z_ab)": _mean_kl_vs(model, za, zab, vocab, device),
        "kl(z_b || z_ab)": _mean_kl_vs(model, zb, zab, vocab, device),
        "kl(id || z_ab)": _mean_kl_vs(model, ident, zab, vocab, device),
        # per-pipeline gates (handoff §3.2b): never a blanket verdict
        "oracle_pass_ab": e_ab[0] >= 0.9,
        "oracle_pass_ba": e_ba[0] >= 0.9,
    }
    counts, dumps = _decode_failures(model, zba, "b_then_a", vocab, device)
    report["z_ba_failure_counts"] = counts
    report["z_ba_failure_dumps"] = dumps
    (d / "report_d2.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    for name, ok in (("z_ab (a_then_b)", report["oracle_pass_ab"]),
                     ("z_ba (b_then_a)", report["oracle_pass_ba"])):
        if not ok:
            print(f"D2: oracle miss on {name} only — capacity vs optimization "
                  "unresolved for that pipeline (see handoff §2.2); other "
                  "pipeline's verdict is independent.")
    if report["oracle_pass_ab"]:
        if report["d_geo(hamilton(b,a), z_ab)"] < min(report["d_geo(z_a, z_ab)"],
                                                      report["d_geo(z_b, z_ab)"]):
            print("D2: Hamilton is nearer z_ab than either atom — algebra in the ballpark.")
        else:
            print("D2: Hamilton is far from z_ab — gauge story (D3/D6); check the "
                  "behavioral KL column before trusting geometric distance.")


def stage_decode(args, model, vocab, device, d: Path):
    """Failure decode on an already-fitted z_ba (no refit; handoff §3.2e)."""
    zba = ProgramField.load(str(d / "z_ba.pt"), trainable=False).to(device)
    counts, dumps = _decode_failures(model, zba, "b_then_a", vocab, device)
    out = {"z_ba_failure_counts": counts, "z_ba_failure_dumps": dumps}
    (d / "report_zba_decode.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def _fit_tracked(model, field, behavior, vocab, device, steps, seed, tag,
                 lr=1e-3, eval_every=250, eval_n=50, target=0.9):
    """fit_field with periodic exact-accuracy evals; returns (field, history,
    steps_to_target). Converts P2's capacity question into a fine-tune-distance
    metric (handoff §4 P2)."""
    from lpm.tasks import e3_examples
    from lpm.utils import cosine_lr, set_lr
    ids, lab, attn = e3_examples(behavior, 4000, vocab, seed=seed)
    params = list(field.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    history, hit = [], None
    for step in range(steps):
        if step % eval_every == 0:
            field.invalidate()
            e = ordered_accuracy(model, field, behavior, vocab, device, n=eval_n)[0]
            history.append({"step": step, "exact": e})
            print(f"[{tag}] step {step} exact {e:.3f}")
            if hit is None and e >= target:
                hit = step
        idx = torch.randint(0, ids.shape[0], (64,), generator=g)
        with model.program(field):
            out = model(input_ids=ids[idx].to(device),
                        attention_mask=attn[idx].to(device),
                        labels=lab[idx].to(device))
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        set_lr(opt, cosine_lr(step, steps, lr, warmup=50))
        opt.step()
    field.invalidate()
    e = ordered_accuracy(model, field, behavior, vocab, device, n=eval_n)[0]
    history.append({"step": steps, "exact": e})
    if hit is None and e >= target:
        hit = steps
    print(f"[{tag}] final exact {e:.3f}  steps_to_{target} = {hit}")
    return field, history, hit


def _warm_start_from(f: ProgramField, spec, device) -> ProgramField:
    return ProgramField(spec, {k: f.q(*k).detach().float().clone()
                               for k in spec.site_keys()},
                        trainable=True).to(device)


def stage_p2(args, model, vocab, device, d: Path):
    """P2 — z_ba disambiguation (handoff §4 P2): warm starts {id, hamilton,
    z_b} with steps-to-0.9-exact, the z_ab-from-hamilton falsifier leg, and
    the overfit-32 capacity check."""
    za = ProgramField.load(str(d / "z_a.pt"), trainable=False).to(device)
    zb = ProgramField.load(str(d / "z_b.pt"), trainable=False).to(device)
    spec = model.spec
    report = {}

    # z_ba: apply b first then a => algebraic guess hamilton(z_a, z_b) = compose(za, zb)
    inits = {
        "identity": ProgramField.randn_near_identity(spec, sigma=1e-3, trainable=True).to(device),
        "hamilton(a,b)": _warm_start_from(compose(za, zb), spec, device),
        "z_b": _warm_start_from(zb, spec, device),
    }
    for name, f in inits.items():
        _, hist, hit = _fit_tracked(model, f, "b_then_a", vocab, device,
                                    steps=args.fit_steps, seed=args.seed + 40,
                                    tag=f"z_ba<-{name}")
        report[f"z_ba from {name}"] = {"steps_to_0.9": hit, "history": hist}

    # falsifier leg (handoff §4 P1): does the hamilton(b,a) init help z_ab?
    for name, f in (("identity", ProgramField.randn_near_identity(
                        spec, sigma=1e-3, trainable=True).to(device)),
                    ("hamilton(b,a)", _warm_start_from(compose(zb, za), spec, device))):
        _, hist, hit = _fit_tracked(model, f, "a_then_b", vocab, device,
                                    steps=args.fit_steps, seed=args.seed + 41,
                                    tag=f"z_ab<-{name}")
        report[f"z_ab from {name}"] = {"steps_to_0.9": hit, "history": hist}

    # overfit-32 capacity check: memorize 32 fixed b_then_a examples
    from lpm.tasks import e3_examples
    from lpm.utils import cosine_lr, set_lr
    ids, lab, attn = e3_examples("b_then_a", 32, vocab, seed=123)
    ids, lab, attn = ids.to(device), lab.to(device), attn.to(device)
    f32 = ProgramField.randn_near_identity(spec, sigma=1e-3, trainable=True).to(device)
    params = list(f32.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0)
    for step in range(args.fit_steps):
        with model.program(f32):
            out = model(input_ids=ids, attention_mask=attn, labels=lab)
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        set_lr(opt, cosine_lr(step, args.fit_steps, 1e-3, warmup=50))
        opt.step()
        if step % 500 == 0:
            print(f"[overfit32] step {step} loss {out.loss.item():.4f}")
    f32.invalidate()
    with torch.no_grad(), model.program(f32):
        logits = model(input_ids=ids, attention_mask=attn).logits
    mask = lab != -100
    correct = ((logits.argmax(-1)[..., :-1] == lab[..., 1:]) | ~mask[..., 1:])
    seq_exact = correct.all(dim=-1).float().mean().item()
    report["overfit_32"] = {"final_loss": out.loss.item(),
                            "teacher_forced_seq_exact": seq_exact}
    print(f"[overfit32] final loss {out.loss.item():.4f} "
          f"tf-seq-exact {seq_exact:.3f}")

    (d / "report_p2.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: (v if k == "overfit_32" else
                          {"steps_to_0.9": v["steps_to_0.9"]})
                      for k, v in report.items()}, indent=2))


def stage_d3(args, model, vocab, device, d: Path):
    print("=== D3 relative increments / transport ===")
    # include append in the frozen base? the saved base was trained without
    # append. Fit z_c / z_cb on the same base anyway (capacity control).
    za = ProgramField.load(str(d / "z_a.pt")).to(device)
    zab_path = d / "z_ab.pt"
    if not zab_path.exists():
        raise SystemExit("run --stage d2 first (need z_ab.pt)")
    zab = ProgramField.load(str(zab_path)).to(device)
    zc = ProgramField.randn_near_identity(model.spec, sigma=1e-3, trainable=True).to(device)
    zcb = ProgramField.randn_near_identity(model.spec, sigma=1e-3, trainable=True).to(device)
    zc = fit_field(model, zc, "append", vocab, device, steps=args.fit_steps,
                   seed=args.seed + 20, tag="z_c")
    zcb = fit_field(model, zcb, "c_then_b", vocab, device, steps=args.fit_steps,
                    seed=args.seed + 21, tag="z_cb")
    zc.save(str(d / "z_c.pt"))
    zcb.save(str(d / "z_cb.pt"))
    delta = increment(zab, za)  # b in the context of a
    left = compose(delta, zc)
    g = compose(zc, invert_field(za))
    adj = compose(compose(compose(g, delta), invert_field(g)), zc)
    e_c = ordered_accuracy(model, zc, "append", vocab, device)
    e_cb = ordered_accuracy(model, zcb, "c_then_b", vocab, device)
    e_left = ordered_accuracy(model, left, "c_then_b", vocab, device)
    e_adj = ordered_accuracy(model, adj, "c_then_b", vocab, device)
    report = {
        "z_c_append_exact": e_c[0],
        "z_cb_oracle_exact": e_cb[0],
        "left_transport_exact": e_left[0],
        "adjoint_transport_exact": e_adj[0],
        "d_geo(left, z_cb)": _mean_dgeo(left, zcb),
        "d_geo(adjoint, z_cb)": _mean_dgeo(adj, zcb),
        "transport_pass": max(e_left[0], e_adj[0]) >= 0.5,
    }
    (d / "report_d3.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def stage_d4(args, model, vocab, device, d: Path):
    print("=== D4 shared-support joint refit ===")
    za = ProgramField.randn_near_identity(model.spec, sigma=1e-3, trainable=True).to(device)
    zb = ProgramField.randn_near_identity(model.spec, sigma=1e-3, trainable=True).to(device)
    za, zb = fit_fields_joint(
        model, [(za, "prepend"), (zb, "reverse")], vocab, device,
        steps=args.fit_steps, seed=args.seed + 30,
        tag="joint overlap", overlap_coef=args.overlap, lambda_reg=0.0)
    za.save(str(d / "z_a_joint.pt"))
    zb.save(str(d / "z_b_joint.pt"))
    bank = AxisBank(model.spec).to(device)
    ta = AbelianProgramField(bank).to(device)
    tb = AbelianProgramField(bank).to(device)
    ta, tb = fit_fields_joint(
        model, [(ta, "prepend"), (tb, "reverse")], vocab, device,
        steps=args.fit_steps, seed=args.seed + 31,
        extra_params=bank.parameters(), tag="abelian joint", lambda_reg=0.0)
    ab = compose(zb, za)
    ba = compose(za, zb)
    tab = compose(tb.to_program_field(), ta.to_program_field())
    tba = compose(ta.to_program_field(), tb.to_program_field())
    rows = {
        "z_a prepend": ordered_accuracy(model, za, "prepend", vocab, device),
        "z_b reverse": ordered_accuracy(model, zb, "reverse", vocab, device),
        "compose(b,a) a_then_b": ordered_accuracy(model, ab, "a_then_b", vocab, device),
        "compose(a,b) b_then_a": ordered_accuracy(model, ba, "b_then_a", vocab, device),
        "compose(b,a) swapped": ordered_accuracy(model, ab, "b_then_a", vocab, device),
        "compose(a,b) swapped": ordered_accuracy(model, ba, "a_then_b", vocab, device),
        "abelian compose(b,a)": ordered_accuracy(model, tab, "a_then_b", vocab, device),
        "abelian compose(a,b)": ordered_accuracy(model, tba, "b_then_a", vocab, device),
    }
    nab = 0.5 * (rows["compose(b,a) a_then_b"][0] + rows["compose(a,b) b_then_a"][0])
    abel = 0.5 * (rows["abelian compose(b,a)"][0] + rows["abelian compose(a,b)"][0])
    margin = 100 * (nab - abel)
    report = {k: {"exact": v[0], "token": v[1]} for k, v in rows.items()}
    report["margin_points"] = margin
    report["d4_pass"] = margin > 20
    (d / "report_d4.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"D4 margin {margin:.1f} points  pass={margin > 20}")


def stage_d5(args, model, vocab, device, d: Path):
    """Strength scan UPWARD via q_pow (handoff P4 — the old downward slerp scan
    probed the already-abelianized regime and is retired)."""
    print("=== D5 strength scan (upward, q_pow) ===")
    za = ProgramField.load(str(d / "z_a.pt")).to(device)
    zb = ProgramField.load(str(d / "z_b.pt")).to(device)
    rows = []
    for lam in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5):
        za_l = pow_field(za, lam)
        zb_l = pow_field(zb, lam)
        e_a = ordered_accuracy(model, za_l, "prepend", vocab, device)
        e_b = ordered_accuracy(model, zb_l, "reverse", vocab, device)
        ab = compose(zb_l, za_l)
        ba = compose(za_l, zb_l)
        e_ab = ordered_accuracy(model, ab, "a_then_b", vocab, device)
        e_ba = ordered_accuracy(model, ba, "b_then_a", vocab, device)
        e_sw = ordered_accuracy(model, ab, "b_then_a", vocab, device)
        rows.append({"lambda": lam,
                     "retention_a_prepend": e_a[0], "retention_b_reverse": e_b[0],
                     "exact_ab": e_ab[0], "exact_ba": e_ba[0],
                     "token_ab": e_ab[1], "token_ba": e_ba[1],
                     "swapped_ab_on_ba": e_sw[0]})
        print(json.dumps(rows[-1]))
    (d / "report_d5.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["d2", "d3", "d4", "d5", "decode", "p2"])
    ap.add_argument("--dir", default="runs/e3")
    ap.add_argument("--fit-steps", type=int, default=1500)
    ap.add_argument("--overlap", type=float, default=0.05,
                    help="D4 overlap bonus coefficient")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    d = Path(args.dir)
    if not (d / "base.pt").exists():
        raise SystemExit(f"missing {d}/base.pt — run fit_e3_atoms.py first")
    vocab = E3Vocab()
    model = _load_wrap(d, device)
    {"d2": stage_d2, "d3": stage_d3, "d4": stage_d4, "d5": stage_d5,
     "decode": stage_decode, "p2": stage_p2}[args.stage](args, model, vocab, device, d)


if __name__ == "__main__":
    main()
