#!/usr/bin/env python3
"""scripts/make_card.py — skill report card CLI (skill-report-card-spec §3).

    python scripts/make_card.py --base EleutherAI/pythia-410m \
        --program runs/g1b/z_formal_conj_only.pt --skill formal \
        --teacher experts-2000/formal/merged --task formal \
        --leverage runs/w0_pythia/whitening.json \
        --library runs/g1b --out registry/formal/0.1.0

Exit code is nonzero when any certificate fails (TC-2).
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
from lpm.card_render import render_html, render_md  # noqa: E402
from lpm.report_card import build_card  # noqa: E402
from lpm.tasks import TASKS  # noqa: E402
from lpm.utils import get_device, pack_texts  # noqa: E402


def load_leverage(path):
    if not path:
        return None
    blob = json.loads(Path(path).read_text())
    return blob.get("leverage", blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--program", required=True)
    ap.add_argument("--skill", required=True)
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--task", default=None, choices=sorted(TASKS) + [None])
    ap.add_argument("--neutral", default="wikitext",
                    help="'wikitext' or 'none'")
    ap.add_argument("--leverage", default=None)
    ap.add_argument("--library", default=None,
                    help="dir of other z_*.pt programs for the fingerprint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--provenance", default=None, help="JSON string")
    args = ap.parse_args()
    device = get_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = LatentProgramModel.from_pretrained(args.base)
    model.to(device)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    field = ProgramField.load(args.program, trainable=False).to(device)
    teacher = None
    if args.teacher:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher, dtype=torch.float32).to(device).eval()

    task_blocks = None
    suites = {}
    if args.task:
        task_blocks = pack_texts(TASKS[args.task].texts("eval", 400), tok,
                                 seq_len=args.seq_len, max_blocks=args.n_eval)
        suites[f"task:{args.task}"] = task_blocks
    neutral_blocks = None
    if args.neutral == "wikitext":
        from lpm.tasks import _wikitext
        neutral_blocks = pack_texts(_wikitext("eval", 400), tok,
                                    seq_len=args.seq_len,
                                    max_blocks=args.n_eval)
        suites["neutral:wikitext"] = neutral_blocks

    library = None
    if args.library:
        library = {}
        for p in sorted(Path(args.library).glob("z_*.pt")):
            if Path(p).resolve() == Path(args.program).resolve():
                continue
            try:
                library[p.stem] = ProgramField.load(str(p),
                                                    trainable=False).to(device)
            except Exception:
                pass

    card = build_card(model=model, field=field, base_id=args.base,
                      program_path=args.program, device=device,
                      skill_name=args.skill, version=args.version,
                      teacher=teacher, teacher_id=args.teacher,
                      task_eval_blocks=task_blocks,
                      neutral_blocks=neutral_blocks,
                      probe_suites=suites or None,
                      leverage=load_leverage(args.leverage),
                      library=library,
                      provenance=(json.loads(args.provenance)
                                  if args.provenance else
                                  {"teacher": args.teacher,
                                   "base": args.base}),
                      seed=args.seed)
    (out / "card.json").write_text(json.dumps(card, indent=2, sort_keys=True))
    (out / "card.md").write_text(render_md(card), encoding="utf-8")
    (out / "card.html").write_text(render_html(card), encoding="utf-8")
    print(render_md(card))
    print(f"\nwrote {out}/card.[json|md|html]")
    if not card["all_certificates_pass"]:
        print("CERTIFICATE FAILURE", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
