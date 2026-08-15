#!/usr/bin/env python3
"""A1 — D3 flagship: can conjugation re-key a token inside a mechanism?
(steering-workstreams-a-b.md A1)

1. Fit atom a′ = prepend-⟨B⟩ (behavior "prependB", field z_f).
2. Fit oracle z_fb on f_then_b = reverse(x)+⟨B⟩ (payload-first).
3. Zero-shot candidates on f_then_b:
   (3) token-keying probe: Δ_A ⊗ z_b, Δ_A = z_ab ⊗ z_b* (≡ z_ab exactly) —
       is z_ab's tweak token-generic ("emit prepended token last") or keyed
       to ⟨A⟩? Prediction: near-perfect except ⟨A⟩ at the marker slot.
   (4) transported tweak: (g ⊗ Δ_A ⊗ g*) ⊗ z_b with g = z_f ⊗ z_a* —
       does the adjoint action re-address ⟨A⟩ → ⟨B⟩ inside the mechanism?

Metrics: exact/token acc PLUS the marker-slot distribution P(⟨A⟩), P(⟨B⟩),
P(other) at the marker position (teacher-forced gold prefix), both candidates
and the oracle/z_b references. Hard pass: (4) ≥ 0.5 exact. Soft pass:
P(⟨B⟩) under (4) ≥ 10× P(⟨B⟩) under (3) and ≥ 0.1 absolute.

    python scripts/a1_rekey.py --dir runs/e3strong
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

from lpm import (LatentProgramModel, ProgramField, compose, increment,  # noqa: E402
                 invert_field)
from lpm.tasks import E3Vocab, e3_output  # noqa: E402
from lpm.utils import get_device, set_seed  # noqa: E402
from e3_common import fit_field, load_base, ordered_accuracy  # noqa: E402


@torch.no_grad()
def marker_slot_distribution(model, field, vocab, device, n=200, seed=99):
    """Teacher-forced: feed [BOS] x [SEP] reverse(x), read the next-token
    softmax at the marker slot of f_then_b (= reverse(x) + ⟨B⟩)."""
    import random
    rng = random.Random(seed)
    pa = pb = po = 0.0
    with model.program(field):
        for _ in range(n):
            x = vocab.payload(rng)
            prefix = [vocab.BOS] + x + [vocab.SEP] + list(reversed(x))
            ids = torch.tensor([prefix], device=device)
            probs = F.softmax(model(input_ids=ids).logits[0, -1].float(), dim=-1)
            pa += float(probs[vocab.A])
            pb += float(probs[vocab.B])
            po += float(1.0 - probs[vocab.A] - probs[vocab.B])
    return {"P(A)": pa / n, "P(B)": pb / n, "P(other)": po / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/e3strong")
    ap.add_argument("--fit-steps", type=int, default=3000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    d = Path(args.dir)
    vocab = E3Vocab()
    model = LatentProgramModel(load_base(str(d / "base.pt"), device))

    za = ProgramField.load(str(d / "z_a.pt"), trainable=False).to(device)
    zb = ProgramField.load(str(d / "z_b.pt"), trainable=False).to(device)
    zab = ProgramField.load(str(d / "z_ab.pt"), trainable=False).to(device)

    # 1. atom a′ = prepend-⟨B⟩
    zf_path = d / "z_f.pt"
    if zf_path.exists():
        zf = ProgramField.load(str(zf_path), trainable=False).to(device)
    else:
        zf = ProgramField.randn_near_identity(model.spec, sigma=1e-3,
                                              trainable=True).to(device)
        zf = fit_field(model, zf, "prependB", vocab, device,
                       steps=args.fit_steps, seed=args.seed + 60, tag="z_f")
        zf.save(str(zf_path))
    e_f = ordered_accuracy(model, zf, "prependB", vocab, device)

    # 2. oracle z_fb on f_then_b
    zfb_path = d / "z_fb.pt"
    if zfb_path.exists():
        zfb = ProgramField.load(str(zfb_path), trainable=False).to(device)
    else:
        zfb = ProgramField.randn_near_identity(model.spec, sigma=1e-3,
                                               trainable=True).to(device)
        zfb = fit_field(model, zfb, "f_then_b", vocab, device,
                        steps=args.fit_steps, seed=args.seed + 61, tag="z_fb")
        zfb.save(str(zfb_path))
    e_fb = ordered_accuracy(model, zfb, "f_then_b", vocab, device)

    # 3. candidates
    delta_A = increment(zab, zb)                       # z_ab ⊗ z_b*
    cand3 = compose(delta_A, zb)                       # ≡ z_ab (sanity anchor)
    g = compose(zf, invert_field(za))                  # z_f ⊗ z_a*
    delta_transported = compose(compose(g, delta_A), invert_field(g))
    cand4 = compose(delta_transported, zb)

    rows = {}
    for name, f in (("(3) token-keying Δ_A⊗z_b", cand3),
                    ("(4) transported g Δ_A g*⊗z_b", cand4),
                    ("oracle z_fb", zfb), ("z_b alone", zb)):
        e = ordered_accuracy(model, f, "f_then_b", vocab, device)
        slot = marker_slot_distribution(model, f, vocab, device)
        rows[name] = {"exact": e[0], "token": e[1], "marker_slot": slot}
        print(name, json.dumps(rows[name]))

    p3b = rows["(3) token-keying Δ_A⊗z_b"]["marker_slot"]["P(B)"]
    p4b = rows["(4) transported g Δ_A g*⊗z_b"]["marker_slot"]["P(B)"]
    hard = rows["(4) transported g Δ_A g*⊗z_b"]["exact"] >= 0.5
    soft = (p4b >= 10 * max(p3b, 1e-9)) and (p4b >= 0.1)
    report = {"z_f_prependB_exact": e_f[0], "z_fb_oracle_exact": e_fb[0],
              "candidates": rows, "hard_pass": hard, "soft_pass": soft}
    (d / "report_a1.json").write_text(json.dumps(report, indent=2))
    print(f"\nA1: atom a' {e_f[0]:.3f}, oracle z_fb {e_fb[0]:.3f}")
    print(f"A1 verdict: hard_pass={hard} soft_pass={soft} "
          f"(P(B): cand3 {p3b:.4f} -> cand4 {p4b:.4f})")


if __name__ == "__main__":
    main()
