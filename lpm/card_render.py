"""lpm/card_render.py — card.json → card.md (and a minimal one-page HTML)."""
from __future__ import annotations

import json
from typing import Dict


def _fmt(v, nd=4):
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render_md(card: Dict) -> str:
    i = card["identity"]
    lines = [f"# Skill card — {i['skill']} v{i['version']}",
             "",
             f"Base `{i['base']}` · program `{i['program_hash']}` · "
             f"{i['n_quats']} quats · {i['bytes_fp16']/1024:.1f} KB fp16 · "
             f"{i['bytes_8bit_sparse']/1024:.1f} KB 8-bit sparse "
             f"({i['active_sites@1e-4']} active sites)",
             ""]
    fid = card.get("fidelity")
    if isinstance(fid, dict):
        ce = fid["task_ce"]
        row = " / ".join(f"{k} {_fmt(v)}" for k, v in ce.items())
        lines += ["## Fidelity", f"- task CE: {row}"]
        rec = fid.get("recovery")
        if isinstance(rec, dict):
            lines += [f"- recovery: **{100*rec['recovery']:.1f}%** "
                      f"(95% CI [{100*rec['ci95'][0]:.1f}, "
                      f"{100*rec['ci95'][1]:.1f}]%, n={rec['n']})"]
        else:
            lines += [f"- recovery: {rec}"]
        lines += [""]
    gent = card.get("gentleness")
    if isinstance(gent, dict):
        pp = gent["neutral_ppl"]
        lines += ["## Gentleness",
                  "- neutral ppl: " + " / ".join(f"{k} {_fmt(v, 2)}"
                                                 for k, v in pp.items())]
        cr = gent.get("collateral_ratio")
        if cr:
            lines += [f"- collateral ratio: {_fmt(cr['value'], 3)}"]
        lines += [""]
    lines += ["## Certificates"]
    for name, c in card["certificates"].items():
        if isinstance(c, dict) and "pass" in c:
            mark = "PASS" if c["pass"] else "**FAIL**"
            lines += [f"- {name}: {mark}"]
        elif isinstance(c, dict):
            lines += [f"- {name}: {c.get('statement', 'see json')}"]
    lines += [""]
    com = card["commitment"]
    lines += ["## Commitment (behavioral work shares)",
              "| group | work share | raw activity (diag) |", "|---|---:|---:|"]
    for g, v in sorted(com["work_shares"].items(), key=lambda t: -t[1]):
        lines += [f"| {g} | {v:.3f} | "
                  f"{com['raw_activity_diagnostic'][g]:.5f} |"]
    lines += [f"", f"Actuator class: **{com['actuator_class']}**", ""]
    act = card["activity"]
    lines += ["## Activity", f"- mean {act['mean']:.5f}, max {act['max']:.4f}"]
    if "prunability_curve" in act:
        lines += ["- prunability: " + json.dumps(act["prunability_curve"])]
    fp = card.get("fingerprint")
    if fp:
        lines += ["", "## Fingerprint",
                  "- probe losses: " + ", ".join(
                      f"{k} {_fmt(v, 3)}" for k, v in fp["probe_losses"].items())]
        if "neighbors_top3" in fp:
            lines += ["- nearest: " + ", ".join(
                f"{n['name']} ({n['sym_kl']:.3f})" for n in fp["neighbors_top3"])]
    lines += ["", "## Limits", card["limits"], ""]
    if card.get("provenance"):
        lines += ["## Provenance", "```json",
                  json.dumps(card["provenance"], indent=1), "```"]
    return "\n".join(lines)


def render_html(card: Dict) -> str:
    body = render_md(card).replace("&", "&amp;").replace("<", "&lt;")
    ok = card.get("all_certificates_pass", False)
    color = "#2e7d32" if ok else "#c62828"
    return (f"<meta charset='utf-8'><title>{card['identity']['skill']} card"
            f"</title><body style='font-family:monospace;max-width:820px;"
            f"margin:2em auto'><div style='border-left:6px solid {color};"
            f"padding-left:1em'><pre style='white-space:pre-wrap'>{body}"
            "</pre></div></body>")
