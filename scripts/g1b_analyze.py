#!/usr/bin/env python3
"""G1-B — skill breadth: cards + gentleness + registered predictions
(gate1-steering §1). Also runs the studio-side card tests TC-4/TC-5.

Expects, per skill s in {formal, jsonish, medical, hedge} (+ french from
G1-A): experts-2000/<s>/merged and runs/g1b/z_<s>_conj_only.pt
(jsonish additionally z_jsonish_conj_axis.pt).

    python scripts/g1b_analyze.py --base EleutherAI/pythia-410m \
        --experts experts-2000 --programs runs/g1b \
        --leverage runs/w0_pythia/whitening.json --out runs/g1b
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lpm import LatentProgramModel, ProgramField  # noqa: E402
from lpm.card_render import render_html, render_md  # noqa: E402
from lpm.report_card import build_card, ce_per_block  # noqa: E402
from lpm.tasks import TASKS, _wikitext  # noqa: E402
from lpm.utils import get_device, pack_texts  # noqa: E402

SKILLS = ["formal", "jsonish", "medical", "hedge"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="EleutherAI/pythia-410m")
    ap.add_argument("--experts", default="experts-2000")
    ap.add_argument("--programs", default="runs/g1b")
    ap.add_argument("--leverage", default="runs/w0_pythia/whitening.json")
    ap.add_argument("--french-program", default="runs/g1a/z_french_conj_only.pt")
    ap.add_argument("--out", default="runs/g1b")
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = get_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = LatentProgramModel.from_pretrained(args.base).to(device)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    leverage = json.loads(Path(args.leverage).read_text())["leverage"]
    neutral = pack_texts(_wikitext("eval", 400), tok, seq_len=128,
                         max_blocks=args.n_eval)

    # library for fingerprints: every program in --programs + french
    lib_paths = {p.stem: p for p in sorted(Path(args.programs).glob("z_*.pt"))}
    fp = Path(args.french_program)
    if fp.exists():
        lib_paths["z_french_conj_only"] = fp
    library = {n: ProgramField.load(str(p), trainable=False).to(device)
               for n, p in lib_paths.items()}

    table, cards = {}, {}
    for skill in SKILLS:
        pp = Path(args.programs) / f"z_{skill}_conj_only.pt"
        tp = Path(args.experts) / skill / "merged"
        if not pp.exists() or not tp.exists():
            print(f"SKIP {skill}: missing artifacts")
            continue
        field = library[pp.stem]
        teacher = AutoModelForCausalLM.from_pretrained(
            str(tp), dtype=torch.float32).to(device).eval()
        task_blocks = pack_texts(TASKS[skill].texts("eval", 400), tok,
                                 seq_len=128, max_blocks=args.n_eval)
        lib_others = {n: f for n, f in library.items() if n != pp.stem}
        card = build_card(model=model, field=field, base_id=args.base,
                          program_path=str(pp), device=device,
                          skill_name=skill, teacher=teacher,
                          task_eval_blocks=task_blocks,
                          neutral_blocks=neutral,
                          probe_suites={f"task:{skill}": task_blocks,
                                        "neutral:wikitext": neutral},
                          leverage=leverage, library=lib_others,
                          provenance={"teacher": str(tp), "steps": 2000,
                                      "protocol": "G1-B budget-matched"})
        cdir = out / "cards" / skill
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "card.json").write_text(json.dumps(card, indent=2,
                                                   sort_keys=True))
        (cdir / "card.md").write_text(render_md(card), encoding="utf-8")
        (cdir / "card.html").write_text(render_html(card), encoding="utf-8")
        cards[skill] = card
        rec = card["fidelity"]["recovery"]
        gent = card["gentleness"]
        table[skill] = {
            "recovery": rec["recovery"], "recovery_ci95": rec["ci95"],
            "neutral_ppl": gent["neutral_ppl"],
            "gentler_than_teacher": gent["neutral_ppl"]["program"]
                <= gent["neutral_ppl"]["teacher"],
            "rope_ax_work_share": card["commitment"]["work_shares"].get(
                "rope_ax", 0.0),
            "actuator": card["commitment"]["actuator_class"],
            "certificates_pass": card["all_certificates_pass"]}
        print(skill, json.dumps(table[skill]))
        del teacher
        torch.cuda.empty_cache()

    # axis arm for the format skill
    ap_json = Path(args.programs) / "z_jsonish_conj_axis.pt"
    if ap_json.exists() and "jsonish" in cards:
        f_axis = ProgramField.load(str(ap_json), trainable=False).to(device)
        task_blocks = pack_texts(TASKS["jsonish"].texts("eval", 400), tok,
                                 seq_len=128, max_blocks=args.n_eval)
        ce_axis = float(ce_per_block(model, task_blocks, device,
                                     field=f_axis).mean())
        table["jsonish"]["conj_axis_task_ce"] = ce_axis

    recs = sorted(t["recovery"] for t in table.values())
    med = recs[len(recs) // 2] if recs else None
    rope_shares = {s: t["rope_ax_work_share"] for s, t in table.items()}
    registered = {
        "median_recovery_>=90% (prior 0.60)": med is not None and med >= 0.90,
        "all_gentler_than_teacher (prior 0.70)":
            all(t["gentler_than_teacher"] for t in table.values()),
        "format_highest_rope_share (prior 0.55)":
            max(rope_shares, key=rope_shares.get) == "jsonish"
            if rope_shares else False,
    }

    # TC-4 / TC-5 (studio-side card tests)
    tc = {}
    if "z_french_conj_only" in library and cards:
        any_card = next(iter(cards.values()))
        # TC-5 on the french program's commitment
        fr_card = build_card(model=model, field=library["z_french_conj_only"],
                             base_id=args.base, program_path=str(fp),
                             device=device, skill_name="french",
                             leverage=leverage, neutral_blocks=neutral)
        ws = fr_card["commitment"]["work_shares"]
        tc["TC-5 french qk_rel work > ffn_hidden"] = \
            ws.get("qk_rel", 0) > ws.get("ffn_hidden", 0)
        for skill, card in cards.items():
            fpb = card.get("fingerprint", {})
            if "neighbors_all" in fpb:
                tc[f"fingerprint_{skill}_neighbors"] = fpb["neighbors_top3"]
    report = {"table": table, "median_recovery": med,
              "registered": registered, "card_tests": tc}
    (out / "report_g1b.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"median_recovery": med, "registered": registered,
                      "card_tests": {k: v for k, v in tc.items()
                                     if k.startswith("TC")}}, indent=2))
    print(f"wrote {out / 'report_g1b.json'}")


if __name__ == "__main__":
    main()
