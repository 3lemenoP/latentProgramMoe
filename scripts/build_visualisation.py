#!/usr/bin/env python3
"""Build the self-contained project visualisation.

Reads scripts/visualisation_template.html, inlines every document listed in
MANIFEST (as JSON, rendered client-side by the page's own markdown renderer)
and writes docs/visualisation/index.html — a single file that opens from disk
with no network dependency beyond web fonts.

    python scripts/build_visualisation.py                 # -> docs/visualisation/index.html
    python scripts/build_visualisation.py --artifact out.html   # body-only variant (no <html> wrapper)

Add a document by appending a row to MANIFEST; order is reading order.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts" / "visualisation_template.html"
OUT = ROOT / "docs" / "visualisation" / "index.html"
PLACEHOLDER = "/*__DOCS_JSON__*/[]"

# (group, path, date, short title, one-line summary) — reading order.
MANIFEST = [
    ("Start here", "docs/story.md", "2026-09-26", "The story so far",
     "A reading guide: the idea, the campaign and where it stands, with every number sourced to a report below."),

    ("Origin", "docs/latent-program-moe-spec.md", "2026-08-12", "Specification v0.1",
     "The founding document: conventions, program fields, sandwiched modules, the dead-frame theorem, training phases, tests T1–T11, experiments E0–E4."),
    ("Origin", "README.md", "2026-08-16", "README and decisions log",
     "Layout, quickstart, milestone runbook and the running log of decisions (gains off, E1–E3 results, qk_rel placement, protocol notes)."),

    ("E-suite", "docs/reports/e0-mirsky.md", "2026-08-13", "E0 · Mirsky audit",
     "Fine-tuning barely moves singular spectra: median rotation-unreachable fraction 0.028 vs threshold 0.3. Gains off."),
    ("E-suite", "docs/reports/e1-experts.md", "2026-08-15", "E1 · Expert reconstruction",
     "Programs recover 85 / 84 / 100 / 96% of four LoRA experts' gap on GPT-2 small; random-orthogonal controls recover nothing."),
    ("E-suite", "docs/reports/e2-merge-french-caps.md", "2026-08-15", "E2 · Geodesic merging",
     "Weight-linear beats program slerp at the midpoint; the program french expert costs half the neutral-text collateral of the true fine-tune."),
    ("E-suite", "docs/reports/e3-order.md", "2026-08-15", "E3 · Order sensitivity",
     "Atoms fit to 1.000 exact; Hamilton composition scores 0.0 in both orders and is behaviourally order-blind."),
    ("E-suite", "e-suite-analysis-e3prime.md", "2026-08-15", "E-suite analysis and the E3′ suite",
     "Reads E0/E2/E3, proposes the disjoint-support mechanism, and specifies the D1–D6 diagnostic-and-rescue suite with its decision tree."),

    ("Diagnosis", "docs/reports/d1-diagnose.md", "2026-08-15", "D1 · Diagnostic battery",
     "Support overlap 0.39 (not disjoint); commutator geometrically at its ceiling but behaviourally silent (KL 0.005)."),
    ("Diagnosis", "handoff-d1-d2-status.md", "2026-08-15", "Handoff after D1/D2",
     "The unified diagnosis: the regulariser linearised the group; every program lives at ~3% activity where Hamilton products reduce to task arithmetic. Plan P0–P5."),
    ("Diagnosis", "docs/reports/p0-expo-probe.md", "2026-08-15", "P0 · Exponentiation probe",
     "Amplifying the fields wakes the commutator on the predicted schedule, yet composition stays 0.0 at every strength."),
    ("Diagnosis", "docs/reports/p1-strong-field.md", "2026-08-15", "P1 · Strong-field refits",
     "At 2× activity the commutator stays at its kinematic ceiling; the oracle pipeline sits nearer identity than the Hamilton prediction, geometrically and behaviourally."),
    ("Diagnosis", "docs/reports/p2-warm-starts.md", "2026-08-15", "P2 · Warm starts",
     "A Hamilton warm start converges slower than identity. Falsifier complete: the product rule is wrong in this gauge. z_ba plateaus at 0.82 regardless of init."),
    ("Diagnosis", "docs/reports/p3-p4-transport-strength.md", "2026-08-15", "P3/P4 · Transport controls and strength scan",
     "Transport controls score 0.0; no intermediate-strength sweet spot. The offset-reversal ceiling is sighted a third time."),

    ("Workstreams A and B", "steering-workstreams-a-b.md", "2026-08-15", "Steering: workstreams A and B",
     "Two parallel studios: A closes the composition question (A1 flagship, A2 ladder, A3-controlled D6); B tests the positional half of the idea on a rotary base."),
    ("Workstreams A and B", "docs/reports/workstream-a-final.md", "2026-08-15", "Workstream A · final",
     "A1: the pipeline's tweak is token-keyed and conjugation cannot re-key it. A2: the offset law. D6: learned operators fail flat. Composition retired."),
    ("Workstreams A and B", "docs/reports/b2-leverage.md", "2026-08-15", "B2 · Leverage ratio",
     "Conjugation fields are saturating small-signal actuators; the axis field is the only compounding, position-coupled one."),
    ("Workstreams A and B", "docs/reports/b3-rotary-ladder.md", "2026-08-15", "B3 · The axis field cracks the ceiling",
     "Offset rungs lift by +12 / +11.5 / +22; a tenfold lr erases the ceiling. Conjugation = content actuator, axis field = position actuator."),
    ("Workstreams A and B", "docs/reports/b4-pythia-style-skill.md", "2026-08-15", "B4 · Style skill on Pythia-410m",
     "The program recovers ~120% of its LoRA teacher's gap; a style skill ignores the axis sites."),

    ("Phase 4", "phase4-belief-differentiation-spec.md", "2026-08-15", "Phase 4 v1 spec · belief and differentiation",
     "Turn the substrate into an agent: whitened tangent Gaussian belief over the program, codebook experiment C, minimal viable organism D."),
    ("Phase 4", "docs/reports/phase4-w0-c-d.md", "2026-08-15", "Phase 4 v1 · W0, C, D",
     "The dead sea; C-2/C-3 confirmed, C-1 failed; D-2/D-4 confirmed, D-1 failed: the unimodal belief averages across regimes."),
    ("Phase 4", "phase4-v2-spec.md", "2026-08-15", "Phase 4 v2 spec · behavioural metric and mixture belief",
     "Constraint 0: no decision-bearing quantity may be manifold-native. Learned M-metric, reanalyses R1–R3, IMM mixture belief with a birth rule."),
    ("Phase 4", "docs/reports/phase4-v2-gates-reanalyses.md", "2026-08-16", "Phase 4 v2 · R1–R3 and metric gates",
     "R2/R3 confirmed; the learned metric fails leave-one-out (G1, G2). Only direct behavioural measurement survives."),
    ("Phase 4", "docs/reports/phase4-v2-mvo.md", "2026-08-16", "Phase 4 v2 · the IMM agent",
     "Memory works directionally (revisits 3× faster, medoid init decisive) but loses to reset-on-spike because toy refits are cheap. Toy MVO retired."),

    ("Gate 1", "gate1-steering.md", "2026-08-16", "Gate-1 steering",
     "One decision: worth scaling. G1-A controls, G1-B breadth, G1-C pruning, the quarter-rotary bridge, and the proposed Gate-2 Qwen run."),
    ("Gate 1", "skill-report-card-spec.md", "2026-08-16", "Skill report card spec",
     "Every fitted program emits an audit card: fidelity, gentleness, four certificates, commitment map, fingerprint, fixed limits text."),
    ("Gate 1", "docs/reports/gate1-bridge.md", "2026-08-16", "Gate 1 · quarter-rotary bridge",
     "Axis lift survives partial coverage (+5 / +10 / +21.5); axis-only collapses. Necessity survives, sufficiency does not."),
    ("Gate 1", "docs/reports/gate1-g1a-controls.md", "2026-08-16", "Gate 1 · G1-A controls",
     "The budget-matched teacher overfits catastrophically; the program recovers 120.6% [114, 128] with a 3× smaller generalisation gap."),
    ("Gate 1", "docs/reports/gate1-final.md", "2026-08-16", "Gate 1 · final scorecard",
     "Breadth: jsonish 104%, hedge 89%, formal 83%, medical 4%. Pruning: dense, 462 KB fp16. Gate 2 proceeds."),
    ("Gate 1", "registry/hedge/0.1.0/card.md", "2026-08-16", "Example card · hedge v0.1.0",
     "The checked-in report card for the hedge skill on Pythia-410m."),
]


def slug(path: str) -> str:
    stem = pathlib.Path(path).stem
    s = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-").lower()
    return s or "doc"


def build_docs() -> list[dict]:
    docs = []
    seen = set()
    for group, path, date, title, summary in MANIFEST:
        p = ROOT / path
        md = p.read_text(encoding="utf-8")
        sid = slug(path)
        if sid in seen:
            sid = sid + "-" + re.sub(r"[^a-z0-9]+", "-", group.lower()).strip("-")
        seen.add(sid)
        docs.append({"id": sid, "path": path, "date": date, "group": group,
                     "title": title, "summary": summary, "md": md})
    return docs


def inline_json(docs: list[dict]) -> str:
    payload = json.dumps(docs, ensure_ascii=False)
    # keep the inline <script> safe whatever the documents contain
    return payload.replace("</", "<\\/").replace("<!--", "<\\!--")


STANDALONE_HEAD = (
    "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1,viewport-fit=cover\">\n"
    "<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}"
    "img{max-width:100%}[hidden]{display:none!important}</style>\n</head>\n<body>\n"
)
STANDALONE_TAIL = "\n</body>\n</html>\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", type=pathlib.Path, default=None,
                    help="also write a body-only copy (no html/head/body wrapper) to this path")
    args = ap.parse_args()

    template = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise SystemExit(f"placeholder {PLACEHOLDER!r} not found in {TEMPLATE}")
    docs = build_docs()
    body = template.replace(PLACEHOLDER, inline_json(docs))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(STANDALONE_HEAD + body + STANDALONE_TAIL, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.0f} KB, {len(docs)} documents)")
    if args.artifact:
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        args.artifact.write_text(body, encoding="utf-8")
        print(f"wrote {args.artifact}  ({args.artifact.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
