#!/usr/bin/env python3
"""Build a pythia-410m whitening.json (operating-strength leverage table)
from the B2 measurement — the card generator's work-share input.

mlp_io was not measured in B2 (both io groups are d_model interface
sandwiches; toy W0 measured them within ~10% of each other), so
mlp_io := attn_io, flagged in meta.

    python scripts/b2_to_whitening.py --b2 runs/b2/report_b2.json \
        --out runs/w0_pythia/whitening.json
"""
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from lpm.field import FieldSpec  # noqa: E402
from lpm.whitening import Whitening  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b2", default="runs/b2/report_b2.json")
    ap.add_argument("--base", default="EleutherAI/pythia-410m")
    ap.add_argument("--sbar", default="0.01")
    ap.add_argument("--out", default="runs/w0_pythia/whitening.json")
    args = ap.parse_args()
    b2 = json.loads(Path(args.b2).read_text())
    lev = {g: b2["leverage"][f"{g}@{args.sbar}"]
           for g in ("rope_ax", "qk_rel", "ffn_hidden", "attn_io")}
    lev["mlp_io"] = lev["attn_io"]
    from transformers import AutoConfig
    spec = FieldSpec.from_hf_config(AutoConfig.from_pretrained(args.base))
    wh = Whitening(spec, lev, meta={"source": "B2 @sbar=" + args.sbar,
                                    "mlp_io": "copied from attn_io",
                                    "base": args.base})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wh.save(args.out)
    print(json.dumps({"leverage": lev, "c": wh.c}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
