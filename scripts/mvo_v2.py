#!/usr/bin/env python3
"""Experiment D v2 — the IMM agent, codebook as memory (phase-4 v2 spec §3).

Six methods + one ablation at matched per-episode compute on a recur3
stream over the e3-base pipeline family (codebook and environment share the
base). Regime decisions are behavioral (per-component probe losses through
a sticky softmax); fusion is winner-only; novelty births named components.

    python scripts/mvo_v2.py --dir runs/e3strong \
        --whitening runs/w0_e3/whitening.json \
        --codebook runs/codebook/report_codebook.json --out runs/mvo_v2
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
from lpm.mixture_belief import IMMBelief  # noqa: E402
from lpm.quaternion import q_log, q_normalize  # noqa: E402
from lpm.tasks import E3Vocab  # noqa: E402
from lpm.utils import get_device, set_seed  # noqa: E402
from lpm.whitening import Whitening  # noqa: E402
from envs.task_stream import TaskStream  # noqa: E402
from e3_common import load_base  # noqa: E402
from mvo import (  # noqa: E402
    field_from_v, prefit_oracles, prefit_static, recovery_times, run_agent,
    run_fixed, run_sgd_tracker, tf_eval, whitened_refine,
)

TASKS = ["prepend", "reverse", "a_then_b", "b_then_a", "c_then_b", "rev_pad2"]
HOLDOUT = "rev_pad2"          # never fitted in the regenerated library


def tangent_raw(field, spec) -> torch.Tensor:
    vs = []
    for k in spec.site_keys():
        q = q_normalize(field.q(*k).detach().float().cpu()).reshape(-1, 4)
        vs.append(q_log(q))
    return torch.cat(vs)                      # (N, 3) raw rotation vectors


def behavioral_medoids(codebook_path, k=4):
    cb = json.loads(Path(codebook_path).read_text())
    D = torch.tensor(cb["D_beh"])
    labels = cb["per_k"][k - 2]["labels_behavioral"]
    names = cb["names"]
    meds = []
    for c in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == c]
        sub = D[idx][:, idx]
        meds.append(names[idx[int(sub.sum(dim=1).argmin())]])
    return meds


@torch.no_grad()
def probe_loss(model, field, batch, device, n=8):
    ids, lab, attn = (t[:n].to(device) for t in batch)
    with model.program(field):
        return float(model(input_ids=ids, attention_mask=attn, labels=lab).loss)


def run_imm(model, wh, stream, device, steps, means, names, seed,
            r_obs_raw=0.02, q_drift_raw=1e-3, tag="imm", novelty_gate=None):
    spec = model.spec
    N = wh.n_quats
    belief = IMMBelief(means, names, wh.group_ids, wh.group_names,
                       novelty_gate=novelty_gate)
    r_vec = torch.full((N,), r_obs_raw)
    q_vec = torch.full((N,), q_drift_raw)
    cvec = wh.c_vec.unsqueeze(-1)
    log = []
    for ep in stream:
        comps = belief.all_components()
        losses = [probe_loss(model, field_from_v(wh, (c.mu * cvec).to(device)),
                             ep.train, device) for c in comps]
        decision = belief.decide(losses)
        winner = decision[1]
        acc, loss = tf_eval(model, field_from_v(wh, (comps[winner].mu * cvec).to(device)),
                            ep.eval, device)
        m_white = whitened_refine(model, wh, comps[winner].mu * cvec,
                                  ep.train, device, steps)
        m_raw = m_white / cvec
        info = belief.step(losses, m_raw, r_vec, q_drift=q_vec, t=ep.idx,
                           decision=decision)
        log.append({"i": ep.idx, "task": ep.task, "acc": acc, "loss": loss,
                    "winner": info["winner"], "winner_name": info["winner_name"],
                    "birth": info["birth"],
                    "n_components": belief.k,
                    "potency": belief.potency() if ep.idx % 25 == 0 else None})
        if info["birth"]:
            print(f"[{tag}] episode {ep.idx}: birth -> {belief.components[-1].name} "
                  f"(task {ep.task})")
    return log, belief


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/e3strong")
    ap.add_argument("--whitening", default="runs/w0_e3/whitening.json")
    ap.add_argument("--codebook", default="runs/codebook/report_codebook.json")
    ap.add_argument("--out", default="runs/mvo_v2")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--refine-steps", type=int, default=25)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--novelty-gate", type=float, default=None)
    ap.add_argument("--imm-only", action="store_true",
                    help="run only the (gated) IMM stream and score it "
                         "against the logs of an existing report_mvo_v2.json")
    args = ap.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = Path(args.dir)

    model = LatentProgramModel(load_base(str(d / "base.pt"), device))
    wh = Whitening.load(args.whitening, model.spec)
    vocab = E3Vocab()

    print("=== prefits ===")
    oracles = prefit_oracles(model, TASKS, vocab, device, out, seed=args.seed)
    static = prefit_static(model, [t for t in TASKS if t != HOLDOUT],
                           vocab, device, out, seed=args.seed)

    med_names = behavioral_medoids(args.codebook, k=4)
    print(f"codebook medoids (k=4 behavioral): {med_names}")
    spec = model.spec
    med_means = [tangent_raw(
        ProgramField.load(str(d / f"z_{n}.pt"), trainable=False), spec)
        for n in med_names]
    g = torch.Generator().manual_seed(args.seed + 500)
    rand_means = [torch.randn(m.shape, generator=g) * m.norm(dim=-1).mean() / (3 ** 0.5)
                  for m in med_means]

    def fresh():
        return TaskStream(TASKS, seed=args.seed + 1, n_episodes=args.episodes,
                          schedule_mode="recur3")
    stream = fresh()
    switch_points = [sp for sp in stream.switch_points if sp < args.episodes]

    if args.imm_only:
        prev = json.loads((out / "report_mvo_v2.json").read_text())
        imm_log, imm_belief = run_imm(model, wh, fresh(), device,
                                      args.refine_steps, med_means, med_names,
                                      seed=args.seed + 11, tag="imm-gated",
                                      novelty_gate=args.novelty_gate)
        oracle_log = prev["logs"]["oracle"]
        oaccs = [x["acc"] for x in oracle_log]
        tasks_at = [x["task"] for x in oracle_log]
        accs = [x["acc"] for x in imm_log]
        first, revisit, seen = [], [], set()
        for sp in switch_points:
            task = tasks_at[sp]
            if sp == 0:
                seen.add(task)
                continue
            rec = None
            for i in range(sp, len(accs)):
                lo = max(i - 4, sp)
                if (sum(accs[lo:i + 1]) / (i + 1 - lo)
                        >= sum(oaccs[lo:i + 1]) / (i + 1 - lo) - 0.05):
                    rec = i - sp
                    break
            rec = rec if rec is not None else len(accs) - sp
            (revisit if task in seen else first).append(rec)
            seen.add(task)
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score([x["task"] for x in imm_log],
                                  [x["winner"] for x in imm_log])
        res = {"novelty_gate": args.novelty_gate,
               "mean_acc": sum(accs) / len(accs),
               "first_visit_recovery": sum(first) / max(len(first), 1),
               "revisit_recovery": sum(revisit) / max(len(revisit), 1),
               "ari": ari, "births": imm_belief.births,
               "n_components_end": imm_log[-1]["n_components"],
               "logs": imm_log}
        (out / "report_imm_gated.json").write_text(json.dumps(res, indent=2))
        print(json.dumps({k: v for k, v in res.items() if k != "logs"},
                         indent=2))
        return

    print("=== streams ===")
    imm_log, imm_belief = run_imm(model, wh, fresh(), device,
                                  args.refine_steps, med_means, med_names,
                                  seed=args.seed + 11, tag="imm",
                                  novelty_gate=args.novelty_gate)
    print("imm done")
    rimm_log, _ = run_imm(model, wh, fresh(), device, args.refine_steps,
                          rand_means, [f"rand{i}" for i in range(len(rand_means))],
                          seed=args.seed + 12, tag="imm-rand")
    print("imm-rand done")
    v1_log, _ = run_agent(model, wh, fresh(), device, args.refine_steps,
                          r_obs=0.1, q_drift=0.1, sigma0=0.3,
                          seed=args.seed + 13, act="map")
    print("v1 agent done")
    sgd_log = run_sgd_tracker(model, wh, fresh(), device, args.refine_steps,
                              seed=args.seed + 14)
    reset_log = run_sgd_tracker(model, wh, fresh(), device, args.refine_steps,
                                seed=args.seed + 15, reset_on_spike=True)
    oracle_log = run_fixed(model, fresh(), device, lambda t: oracles[t])
    static_log = run_fixed(model, fresh(), device, lambda t: static)
    print("baselines done")

    methods = {"imm": imm_log, "imm_random_init": rimm_log,
               "v1_agent": v1_log, "sgd_tracker": sgd_log,
               "reset_on_spike": reset_log, "oracle": oracle_log,
               "static": static_log}

    # ---- metrics -----------------------------------------------------------
    def mean_acc(lg):
        return sum(x["acc"] for x in lg) / len(lg)
    oaccs = [x["acc"] for x in oracle_log]
    tasks_at = [x["task"] for x in oracle_log]

    def split_recoveries(lg):
        accs = [x["acc"] for x in lg]
        first, revisit, seen = [], [], set()
        for sp in switch_points:
            task = tasks_at[sp]
            if sp == 0:
                seen.add(task)
                continue
            rec = None
            for i in range(sp, len(accs)):
                lo = max(i - 4, sp)
                if (sum(accs[lo:i + 1]) / (i + 1 - lo)
                        >= sum(oaccs[lo:i + 1]) / (i + 1 - lo) - 0.05):
                    rec = i - sp
                    break
            rec = rec if rec is not None else len(accs) - sp
            (revisit if task in seen else first).append(rec)
            seen.add(task)
        return (sum(first) / max(len(first), 1),
                sum(revisit) / max(len(revisit), 1))

    summary = {}
    for name, lg in methods.items():
        fv, rv = split_recoveries(lg) if name != "oracle" else (0, 0)
        summary[name] = {"mean_acc": mean_acc(lg),
                         "regret": mean_acc(oracle_log) - mean_acc(lg),
                         "first_visit_recovery": fv, "revisit_recovery": rv}
    print(json.dumps(summary, indent=2))

    from sklearn.metrics import adjusted_rand_score
    task_ids = [x["task"] for x in imm_log]
    winners = [x["winner"] for x in imm_log]
    ari_imm = adjusted_rand_score(task_ids, winners)
    rw = [x["winner"] for x in rimm_log]
    ari_rand = adjusted_rand_score([x["task"] for x in rimm_log], rw)
    early = 150
    ari_imm_early = adjusted_rand_score(task_ids[:early], winners[:early])
    ari_rand_early = adjusted_rand_score([x["task"] for x in rimm_log][:early],
                                         rw[:early])

    verdicts = {
        "P-1 imm beats reset on mean acc":
            summary["imm"]["mean_acc"] > summary["reset_on_spike"]["mean_acc"],
        "P-2 imm revisit recovery < 10":
            summary["imm"]["revisit_recovery"] < 10,
        "P-3 imm first-visit within ±20% of reset":
            abs(summary["imm"]["first_visit_recovery"]
                - summary["reset_on_spike"]["first_visit_recovery"])
            <= 0.2 * max(summary["reset_on_spike"]["first_visit_recovery"], 1e-9),
        "P-4 responsibility-regime ARI > 0.8": ari_imm > 0.8,
        "P-6 medoid init aligns faster than random":
            ari_imm_early > ari_rand_early,
    }
    report = {"summary": summary, "switch_points": switch_points,
              "ari": {"imm": ari_imm, "imm_random": ari_rand,
                      "imm_early150": ari_imm_early,
                      "imm_random_early150": ari_rand_early},
              "births_imm": imm_belief.births,
              "verdicts": verdicts,
              "logs": {k: v for k, v in methods.items()}}
    (out / "report_mvo_v2.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"ari": report["ari"], "births": report["births_imm"],
                      "verdicts": verdicts}, indent=2))
    print(f"wrote {out / 'report_mvo_v2.json'}")


if __name__ == "__main__":
    main()
