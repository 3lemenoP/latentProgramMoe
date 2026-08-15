#!/usr/bin/env python3
"""Experiment D — the minimal viable organism (phase-4 spec §5).

ADF agent (predict → Thompson-act → observe → whitened refine → fuse) vs
four equal-compute baselines on a seeded regime-switching stream over the
rotary-toy pipeline family. Logs eval exact, potency, commitment maps,
activity; scores D-1..D-5.

    python scripts/mvo.py --base runs/b3/base.pt \
        --whitening runs/w0_rotary/whitening.json --out runs/mvo
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

from lpm import LatentProgramModel, ProgramField  # noqa: E402
from lpm.belief import WhitenedGaussianBelief  # noqa: E402
from lpm.tasks import E3Vocab, e3_examples  # noqa: E402
from lpm.utils import get_device, set_seed  # noqa: E402
from lpm.whitening import Whitening  # noqa: E402
from envs.task_stream import DEFAULT_TASKS, TaskStream  # noqa: E402
from e3_common import fit_field, load_base  # noqa: E402


# ---------------------------------------------------------------------------
# shared primitives
# ---------------------------------------------------------------------------
@torch.no_grad()
def tf_eval(model, field, batch, device):
    """Teacher-forced (seq_exact, mean_loss) on one batch."""
    ids, lab, attn = (t.to(device) for t in batch)
    ctx = model.program(field) if field is not None else model.program(None)
    with ctx:
        out = model(input_ids=ids, attention_mask=attn, labels=lab)
        logits = out.logits
    mask = lab != -100
    correct = ((logits.argmax(-1)[..., :-1] == lab[..., 1:]) | ~mask[..., 1:])
    return float(correct.all(dim=-1).float().mean()), float(out.loss)


def whitened_refine(model, wh, v0, batch, device, steps, lr=1e-3):
    """T gradient steps from whitened point v0, optimizing in RAW
    rotation-vector geometry (Adam steps then mean uniform raw-angle motion
    at the proven fit_field lr scale; a whitened-parameterized Adam maps a
    uniform lr to ~lr/c_g raw steps — radians for low-leverage groups).
    Interface stays whitened: input and returned point are whitened."""
    if steps == 0:
        return v0.clone()
    ids, lab, attn = (t.to(device) for t in batch)
    cvec = wh.c_vec.to(device).unsqueeze(-1)
    v_raw = (v0.to(device) / cvec).clone().requires_grad_(True)
    opt = torch.optim.Adam([v_raw], lr=lr)
    for _ in range(steps):
        quats = wh.whitened_to_quats_graph(v_raw * cvec)
        field = ProgramField.from_tensors(model.spec, quats)
        with model.program(field):
            loss = model(input_ids=ids, attention_mask=attn, labels=lab).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (v_raw.detach() * cvec).cpu()


def field_from_v(wh, v):
    return wh.whitened_to_field(v)


# ---------------------------------------------------------------------------
# agent + baselines (equal per-episode compute: `steps` refine steps each)
# ---------------------------------------------------------------------------
def make_sigma0(wh, sigma0, theta_cap=0.3):
    """Per-site prior std: whitened target sigma0 capped so the implied RAW
    per-site angle stays <= theta_cap (whitened-isotropic priors put
    radian-scale rotations on low-leverage groups — KL-cheap but far from
    where task solutions live)."""
    cap = wh.c_vec * (theta_cap / (3 ** 0.5))
    return torch.minimum(torch.full_like(wh.c_vec, sigma0), cap)


def run_agent(model, wh, stream, device, steps, r_obs, q_drift, sigma0, seed):
    b = WhitenedGaussianBelief(wh, sigma0=make_sigma0(wh, sigma0))
    # drift and observation noise as FRACTIONS of the per-site prior width —
    # absolute whitened values would be incommensurate with capped groups
    qvec = q_drift * b.sigma0
    rvec = r_obs * b.sigma0
    g = torch.Generator().manual_seed(seed)
    log = []
    for ep in stream:
        b.predict(q_drift=qvec)
        v_s = b.sample(g)
        acc, loss = tf_eval(model, field_from_v(wh, v_s.to(device)), ep.eval, device)
        m = whitened_refine(model, wh, v_s, ep.train, device, steps)
        b.fuse(m, rvec)
        pot = b.potency()
        log.append({"i": ep.idx, "task": ep.task, "acc": acc, "loss": loss,
                    "potency_total": pot["total"],
                    "potency": {k: v for k, v in pot.items() if k != "total"},
                    "commitment": b.commitment_map() if ep.idx % 25 == 0 else None})
    return log, b


def run_sgd_tracker(model, wh, stream, device, steps, seed, reset_on_spike=False):
    torch.manual_seed(seed)

    def near_identity():
        # tiny RAW-angle init (fit_field's sigma scale), expressed whitened
        return 2e-3 * wh.c_vec.unsqueeze(-1) * torch.randn(wh.n_quats, 3)

    v = near_identity()
    log, losses = [], []
    for ep in stream:
        acc, loss = tf_eval(model, field_from_v(wh, v.to(device)), ep.eval, device)
        if reset_on_spike and len(losses) >= 20:
            recent = torch.tensor(losses[-20:])
            if loss > float(recent.mean() + 3 * recent.std().clamp_min(1e-6)):
                v = near_identity()
        losses.append(loss)
        v = whitened_refine(model, wh, v, ep.train, device, steps)
        log.append({"i": ep.idx, "task": ep.task, "acc": acc, "loss": loss})
    return log


def run_fixed(model, stream, device, field_for_task):
    log = []
    for ep in stream:
        f = field_for_task(ep.task)
        acc, loss = tf_eval(model, f, ep.eval, device)
        log.append({"i": ep.idx, "task": ep.task, "acc": acc, "loss": loss})
    return log


# ---------------------------------------------------------------------------
# prefits (oracle per-task, static mixture)
# ---------------------------------------------------------------------------
def prefit_oracles(model, tasks, vocab, device, d, steps=3000, seed=0):
    out = {}
    for i, t in enumerate(tasks):
        p = d / f"oracle_{t}.pt"
        if p.exists():
            out[t] = ProgramField.load(str(p), trainable=False).to(device)
            continue
        f = ProgramField.randn_near_identity(model.spec, sigma=1e-3,
                                             trainable=True).to(device)
        f = fit_field(model, f, t, vocab, device, steps=steps,
                      seed=seed + 200 + i, tag=f"oracle_{t}")
        f.save(str(p))
        out[t] = f
    return out


def prefit_static(model, tasks, vocab, device, d, steps=3000, seed=0):
    p = d / "static_mixture.pt"
    if p.exists():
        return ProgramField.load(str(p), trainable=False).to(device)
    from lpm.utils import cosine_lr, set_lr
    data = {t: e3_examples(t, 2000, vocab, seed=seed + 300 + i)
            for i, t in enumerate(tasks)}
    f = ProgramField.randn_near_identity(model.spec, sigma=1e-3,
                                         trainable=True).to(device)
    params = list(f.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed + 400)
    import random as _r
    rng = _r.Random(seed + 401)
    for step in range(steps):
        t = rng.choice(tasks)
        ids, lab, attn = data[t]
        idx = torch.randint(0, ids.shape[0], (32,), generator=g)
        with model.program(f):
            loss = model(input_ids=ids[idx].to(device),
                         attention_mask=attn[idx].to(device),
                         labels=lab[idx].to(device)).loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        set_lr(opt, cosine_lr(step, steps, 1e-3, warmup=50))
        opt.step()
        if step % 500 == 0:
            print(f"[static] step {step}/{steps} loss {loss.item():.4f}")
    f.invalidate()
    f.save(str(p))
    return f


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def recovery_times(log, oracle_log, switch_points, tol=0.05, window=5):
    accs = [r["acc"] for r in log]
    oaccs = [r["acc"] for r in oracle_log]
    times = []
    for sp in switch_points:
        if sp == 0:
            continue
        t = None
        for i in range(sp, len(accs)):
            lo = max(i - window + 1, sp)
            a = sum(accs[lo:i + 1]) / (i + 1 - lo)
            o = sum(oaccs[lo:i + 1]) / (i + 1 - lo)
            if a >= o - tol:
                t = i - sp
                break
        times.append(t if t is not None else len(accs) - sp)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--whitening", required=True)
    ap.add_argument("--out", default="runs/mvo")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--refine-steps", type=int, default=25)
    ap.add_argument("--sigma0", type=float, default=0.3)
    ap.add_argument("--grid", action="store_true",
                    help="run the (r_obs, q_drift) validation grid first")
    ap.add_argument("--r-obs", type=float, default=0.2)
    ap.add_argument("--q-drift", type=float, default=0.05)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = LatentProgramModel(load_base(args.base, device))
    wh = Whitening.load(args.whitening, model.spec)
    vocab = E3Vocab()
    tasks = DEFAULT_TASKS
    holdout = tasks[-1]
    fit_tasks = tasks[:-1]  # holdout stays never-direct-fit for the de novo rung

    print("=== prefits (oracle per task incl. holdout-for-ceiling, static mixture) ===")
    oracles = prefit_oracles(model, tasks, vocab, device, out, seed=args.seed)
    static = prefit_static(model, fit_tasks, vocab, device, out, seed=args.seed)

    r_obs, q_drift = args.r_obs, args.q_drift
    if args.grid:
        print("=== (r_obs, q_drift) grid on validation stream ===")
        best = (None, -1.0)
        for r in (0.1, 0.3):
            for q in (0.02, 0.1):
                vs = TaskStream(tasks, seed=args.seed + 999, n_episodes=150)
                lg, _ = run_agent(model, wh, vs, device, args.refine_steps,
                                  r, q, args.sigma0, seed=args.seed + 5)
                acc = sum(x["acc"] for x in lg) / len(lg)
                print(f"r={r} q={q}: mean acc {acc:.3f}")
                if acc > best[1]:
                    best = ((r, q), acc)
        (r_obs, q_drift) = best[0]
        print(f"grid winner: r_obs={r_obs} q_drift={q_drift}")

    print("=== main streams ===")
    def fresh():
        return TaskStream(tasks, seed=args.seed + 1, n_episodes=args.episodes)
    stream = fresh()
    switch_points = [sp for sp in stream.switch_points if sp < args.episodes]

    agent_log, belief = run_agent(model, wh, fresh(), device, args.refine_steps,
                                  r_obs, q_drift, args.sigma0, seed=args.seed + 11)
    print("agent done")
    sgd_log = run_sgd_tracker(model, wh, fresh(), device, args.refine_steps,
                              seed=args.seed + 12)
    print("sgd tracker done")
    reset_log = run_sgd_tracker(model, wh, fresh(), device, args.refine_steps,
                                seed=args.seed + 13, reset_on_spike=True)
    print("reset-on-spike done")
    oracle_log = run_fixed(model, fresh(), device, lambda t: oracles[t])
    static_log = run_fixed(model, fresh(), device, lambda t: static)
    print("fixed baselines done")

    # ---- metrics -----------------------------------------------------------
    def mean_acc(lg):
        return sum(x["acc"] for x in lg) / len(lg)
    methods = {"agent": agent_log, "sgd_tracker": sgd_log,
               "reset_on_spike": reset_log, "oracle": oracle_log,
               "static": static_log}
    summary = {m: {"mean_acc": mean_acc(lg),
                   "regret_vs_oracle": mean_acc(oracle_log) - mean_acc(lg)}
               for m, lg in methods.items()}
    for m in ("agent", "sgd_tracker", "reset_on_spike", "static"):
        rt = recovery_times(methods[m], oracle_log, switch_points)
        summary[m]["mean_recovery"] = sum(rt) / max(len(rt), 1)
        summary[m]["recovery_times"] = rt
    print(json.dumps({m: {k: v for k, v in s.items() if k != "recovery_times"}
                      for m, s in summary.items()}, indent=2))

    # ---- de novo rung (D-4): holdout task, fresh vs committed prior --------
    print("=== de novo rung ===")
    def episodes_to_threshold(prior_belief, thr=0.5, max_ep=120):
        ts = TaskStream([holdout], seed=args.seed + 2, n_episodes=max_ep)
        b = prior_belief
        qv = q_drift * b.sigma0
        rv = r_obs * b.sigma0
        g = torch.Generator().manual_seed(args.seed + 21)
        for ep in ts:
            b.predict(q_drift=qv)
            v_s = b.sample(g)
            acc, _ = tf_eval(model, field_from_v(wh, v_s.to(device)), ep.eval, device)
            if acc >= thr:
                return ep.idx
            m = whitened_refine(model, wh, v_s, ep.train, device, args.refine_steps)
            b.fuse(m, rv)
        return max_ep
    fresh_prior = WhitenedGaussianBelief(wh, sigma0=make_sigma0(wh, args.sigma0))
    # committed prior: converge on a different task first
    committed = WhitenedGaussianBelief(wh, sigma0=make_sigma0(wh, args.sigma0))
    ct = TaskStream(["b_then_a"], seed=args.seed + 3, n_episodes=60)
    g2 = torch.Generator().manual_seed(args.seed + 22)
    rv_c = r_obs * committed.sigma0
    for ep in ct:
        committed.predict(q_drift=0.0)
        v_s = committed.sample(g2)
        m = whitened_refine(model, wh, v_s, ep.train, device, args.refine_steps)
        committed.fuse(m, rv_c)
    de_novo = {"fresh_prior_eps": episodes_to_threshold(fresh_prior),
               "committed_prior_eps": episodes_to_threshold(committed)}
    print(json.dumps(de_novo))

    # ---- predictions -------------------------------------------------------
    pots = [x["potency_total"] for x in agent_log]
    within, at_switch = [], []
    for i in range(1, len(pots)):
        (at_switch if i in switch_points else within).append(pots[i] - pots[i - 1])
    verdicts = {
        "D-1 agent beats sgd on recovery": summary["agent"]["mean_recovery"]
            < summary["sgd_tracker"]["mean_recovery"],
        "D-2 potency falls in-regime, rises at switches":
            (sum(within) / max(len(within), 1) < 0)
            and (sum(at_switch) / max(len(at_switch), 1)
                 > sum(within) / max(len(within), 1)),
        "D-4 de novo slower from committed prior":
            de_novo["committed_prior_eps"] > de_novo["fresh_prior_eps"],
    }
    report = {"config": {"r_obs": r_obs, "q_drift": q_drift,
                         "sigma0": args.sigma0, "episodes": args.episodes,
                         "refine_steps": args.refine_steps, "tasks": tasks,
                         "holdout": holdout},
              "summary": summary, "verdicts": verdicts, "de_novo": de_novo,
              "switch_points": switch_points,
              "logs": {m: lg for m, lg in methods.items()}}
    (out / "report_mvo.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(verdicts, indent=2))
    print(f"wrote {out / 'report_mvo.json'}")


if __name__ == "__main__":
    main()
