#!/usr/bin/env python3
"""Experiment C — the codebook (phase-4 spec §4). No training.

Maps the fitted field library to whitened tangent vectors, clusters in
(a) whitened L2 and (b) raw sitewise d_geo (control), referees both against
behavioral clustering (pairwise symmetrized KL on a shared probe set), and
reports ARIs, the rate–distortion curve, and actuator-profile labels.

    python scripts/codebook.py --fields-dir runs/e3strong \
        --base runs/e3strong/base.pt --whitening runs/w0_e3/whitening.json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lpm import LatentProgramModel, ProgramField  # noqa: E402
from lpm.quaternion import d_geo, q_angle2, q_normalize  # noqa: E402
from lpm.tasks import E3Vocab, e3_examples  # noqa: E402
from lpm.utils import get_device, set_seed  # noqa: E402
from lpm.whitening import Whitening  # noqa: E402
from e3_common import load_base  # noqa: E402


def kmedoids(D: torch.Tensor, k: int, seed: int = 0, iters: int = 50):
    """PAM-lite on a precomputed distance matrix."""
    n = D.shape[0]
    g = torch.Generator().manual_seed(seed)
    med = [int(torch.randint(0, n, (1,), generator=g))]
    while len(med) < k:  # greedy max-min init
        dmin = D[:, med].min(dim=1).values
        med.append(int(dmin.argmax()))
    med = list(dict.fromkeys(med))
    while len(med) < k:
        med.append(int(torch.randint(0, n, (1,), generator=g)))
    for _ in range(iters):
        assign = D[:, med].argmin(dim=1)
        new_med = []
        for c in range(k):
            idx = (assign == c).nonzero().flatten()
            if len(idx) == 0:
                new_med.append(med[c])
                continue
            sub = D[idx][:, idx]
            new_med.append(int(idx[sub.sum(dim=1).argmin()]))
        if new_med == med:
            break
        med = new_med
    assign = D[:, med].argmin(dim=1)
    return assign.tolist(), med


def silhouette(D, labels):
    import numpy as np
    from sklearn.metrics import silhouette_score
    labels = np.array(labels)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(silhouette_score(D.cpu().numpy(), labels, metric="precomputed"))


def g2_metric_rematch(args, spec, wh, fields, names, report_path, out):
    """Phase-4 v2 §1: fit the M-metric on the library + existing symKL,
    run the G1 LOO gate (T22) and the G2 clustering rematch."""
    from sklearn.metrics import adjusted_rand_score
    from lpm.metric import MetricM, loo_stress
    from lpm.quaternion import q_log, q_normalize
    prev = json.loads(Path(report_path).read_text())
    D_beh = torch.tensor(prev["D_beh"])
    assert prev["names"] == names, "field library changed since behavioral run"

    def tangent_flat(f):
        vs = []
        for k in spec.site_keys():
            q = q_normalize(f.q(*k).detach().float().cpu()).reshape(-1, 4)
            vs.append(q_log(q))
        return torch.cat(vs).flatten()

    V = torch.stack([tangent_flat(fields[n]) for n in names])
    c_vec = wh.c_vec
    g1 = loo_stress(V, D_beh, c_vec)
    print(f"G1 (T22 LOO): M {g1['spearman_M']:.3f} raw {g1['spearman_raw']:.3f} "
          f"whitened {g1['spearman_whitened']:.3f} pass={g1['G1_pass']}")
    m = MetricM.fit(V, D_beh, c_vec, meta={"probe_set": "copy/seed7/n200",
                                           "library": names})
    m.save(str(out / "metric.json"))
    D_M = m.pdist2(V).sqrt()
    D_M.fill_diagonal_(0)
    rows = []
    for k in range(2, 7):
        lm, _ = kmedoids(D_M, k, seed=0)
        lb = prev["per_k"][k - 2]["labels_behavioral"]
        lr_ari = prev["per_k"][k - 2]["ari_raw_vs_behavioral"]
        lw_ari = prev["per_k"][k - 2]["ari_whitened_vs_behavioral"]
        rows.append({"k": k, "ari_M_vs_behavioral": adjusted_rand_score(lb, lm),
                     "ari_raw_vs_behavioral": lr_ari,
                     "ari_whitened_vs_behavioral": lw_ari})
        print(rows[-1])
    mean_m = sum(r["ari_M_vs_behavioral"] for r in rows) / len(rows)
    mean_r = sum(r["ari_raw_vs_behavioral"] for r in rows) / len(rows)
    mean_w = sum(r["ari_whitened_vs_behavioral"] for r in rows) / len(rows)
    g2 = {"per_k": rows, "mean_ari": {"M": mean_m, "raw": mean_r,
                                      "whitened": mean_w},
          "G2_M_gt_raw": mean_m > mean_r, "G2_M_gt_whitened": mean_m > mean_w,
          "G1": g1}
    (out / "report_g2.json").write_text(json.dumps(g2, indent=2))
    print(f"G2: ARI_M {mean_m:.3f} vs raw {mean_r:.3f} (pass={mean_m > mean_r}) "
          f"vs whitened {mean_w:.3f} (pass={mean_m > mean_w})")
    print(f"wrote {out / 'metric.json'} and report_g2.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields-dir", default="runs/e3strong")
    ap.add_argument("--base", default="runs/e3strong/base.pt")
    ap.add_argument("--whitening", required=True)
    ap.add_argument("--out", default="runs/codebook")
    ap.add_argument("--n-probe", type=int, default=200)
    ap.add_argument("--g2", action="store_true",
                    help="fit the M-metric + G1/G2 gates from the existing "
                         "behavioral report (no new probes)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = Path(args.fields_dir)

    model = LatentProgramModel(load_base(args.base, device))
    spec = model.spec
    wh = Whitening.load(args.whitening, spec)
    vocab = E3Vocab()

    # library: atoms a–f + all "<x><y>" pipeline fields present
    names = []
    for p in sorted(d.glob("z_*.pt")):
        stem = p.stem[2:]
        if len(stem) <= 2 and stem.isalpha():
            names.append(stem)
    fields = {n: ProgramField.load(str(d / f"z_{n}.pt"),
                                   trainable=False).to(device) for n in names}
    print(f"library: {len(names)} fields: {names}")

    if args.g2:
        g2_metric_rematch(args, spec, wh, fields, names,
                          out / "report_codebook.json", out)
        return

    # 1. whitened vectors + raw d_geo matrix + actuator profiles
    X = torch.stack([wh.field_to_whitened(fields[n]).flatten() for n in names])
    n_f = len(names)
    D_white = torch.cdist(X, X)
    D_white.fill_diagonal_(0)
    D_raw = torch.zeros(n_f, n_f)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i:
                continue
            xs = [d_geo(q_normalize(fields[a].q(*k).detach().float()),
                        q_normalize(fields[b].q(*k).detach().float())).mean()
                  for k in spec.site_keys()]
            D_raw[i, j] = D_raw[j, i] = float(torch.stack(xs).mean())
    profiles = {}
    for n in names:
        prof = {}
        for gname in spec.site_names():
            xs = [q_angle2(q_normalize(fields[n].q(l, s).detach().float())).mean()
                  for (l, s) in spec.site_keys() if s == gname]
            prof[gname] = float(torch.stack(xs).mean())
        tot = sum(prof.values())
        profiles[n] = {k: v / max(tot, 1e-12) for k, v in prof.items()}

    # 2. behavioral matrix: symmetrized KL on shared probes
    ids, _, attn = e3_examples("copy", args.n_probe, vocab, seed=7)
    ids, attn = ids.to(device), attn.to(device)
    logps = []
    with torch.no_grad():
        for n in names:
            with model.program(fields[n]):
                lg = model(input_ids=ids, attention_mask=attn).logits
            logps.append(F.log_softmax(lg.float(), dim=-1)[attn.bool()].cpu())
    D_beh = torch.zeros(n_f, n_f)
    for i in range(n_f):
        for j in range(i + 1, n_f):
            li, lj = logps[i], logps[j]
            kij = (li.exp() * (li - lj)).sum(-1).mean()
            kji = (lj.exp() * (lj - li)).sum(-1).mean()
            D_beh[i, j] = D_beh[j, i] = float(0.5 * (kij + kji))

    # 3–5. cluster at k = 2..6, ARIs vs behavioral, rate–distortion
    from sklearn.metrics import adjusted_rand_score
    rows = []
    for k in range(2, 7):
        lw, mw = kmedoids(D_white, k, seed=args.seed)
        lr, _ = kmedoids(D_raw, k, seed=args.seed)
        lb, _ = kmedoids(D_beh, k, seed=args.seed)
        cover = float(D_white[:, mw].min(dim=1).values.max())
        rows.append({
            "k": k,
            "sil_whitened": silhouette(D_white, lw),
            "sil_raw": silhouette(D_raw, lr),
            "sil_behavioral": silhouette(D_beh, lb),
            "ari_whitened_vs_behavioral": adjusted_rand_score(lb, lw),
            "ari_raw_vs_behavioral": adjusted_rand_score(lb, lr),
            "whitened_covering_radius": cover,
            "labels_whitened": lw, "labels_raw": lr, "labels_behavioral": lb,
        })
        print({kk: (round(vv, 4) if isinstance(vv, float) else vv)
               for kk, vv in rows[-1].items() if not kk.startswith("labels")})

    # C-3 operationalization: pairs co-clustered geometrically (raw) yet in
    # different behavioral clusters, at each k
    c3 = []
    for r in rows:
        lr, lb = r["labels_raw"], r["labels_behavioral"]
        bad = [(names[i], names[j], float(D_beh[i, j]))
               for i in range(n_f) for j in range(i + 1, n_f)
               if lr[i] == lr[j] and lb[i] != lb[j]]
        bad.sort(key=lambda t: -t[2])
        c3.append({"k": r["k"], "n_merged_distinct": len(bad),
                   "worst": bad[:3]})

    mean_ariw = sum(r["ari_whitened_vs_behavioral"] for r in rows) / len(rows)
    mean_arir = sum(r["ari_raw_vs_behavioral"] for r in rows) / len(rows)
    report = {
        "names": names, "profiles": profiles, "per_k": rows, "c3": c3,
        "mean_ari": {"whitened": mean_ariw, "raw": mean_arir},
        "C1_pass": mean_ariw > mean_arir,
        "D_white": D_white.tolist(), "D_raw": D_raw.tolist(),
        "D_beh": D_beh.tolist(),
    }
    (out / "report_codebook.json").write_text(json.dumps(report, indent=2))
    print(f"\nC-1 (ARI whitened {mean_ariw:.3f} > raw {mean_arir:.3f}): "
          f"{report['C1_pass']}")
    print(f"wrote {out / 'report_codebook.json'}")


if __name__ == "__main__":
    main()
