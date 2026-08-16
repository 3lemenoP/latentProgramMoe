"""lpm/report_card.py — skill report card computation (skill-report-card-spec).

Pure functions over existing instruments: every fitted program emits an
audit card. The card certifies stability, size, scope of modification, and
reversibility — never content alignment, factuality, or safety of outputs.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .compose import compose, invert_field
from .field import ProgramField
from .quaternion import q_angle2, q_normalize, q_to_R

LIMITS_TEXT = (
    "This card certifies stability, size, scope of modification, and "
    "reversibility of the program. It does NOT certify content alignment, "
    "factuality, or safety of outputs — bounded modification class is not "
    "benign behavior. Unvalidated verbs: multi-skill stacking (composition "
    "of programs is retired); validated verbs: select, slerp, swap, undo.")

SCHEMA_VERSION = "1.0.0"
HIST_BINS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# eval primitives
# ---------------------------------------------------------------------------
@torch.no_grad()
def ce_per_block(model, blocks: torch.Tensor, device, field=None,
                 batch: int = 8) -> torch.Tensor:
    """Per-block mean CE. model: LatentProgramModel (field=None → base) or a
    raw HF model (field ignored)."""
    import torch.nn.functional as F
    out = []
    is_lpm = hasattr(model, "program")
    for i in range(0, blocks.shape[0], batch):
        b = blocks[i:i + batch].to(device)
        if is_lpm:
            with model.program(field):
                logits = model(input_ids=b).logits
        else:
            logits = model(input_ids=b).logits
        lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = b[:, 1:]
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        out.append(nll.mean(dim=-1).cpu())
    return torch.cat(out)


def paired_bootstrap(a: torch.Tensor, b: torch.Tensor, n_boot: int = 2000,
                     seed: int = 0):
    """CI on mean(a − b) over paired per-block values."""
    g = torch.Generator().manual_seed(seed)
    n = a.shape[0]
    d = a - b
    means = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        means.append(float(d[idx].mean()))
    means.sort()
    return {"mean": float(d.mean()),
            "ci95": [means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]],
            "n": n}


def recovery_bootstrap(base: torch.Tensor, prog: torch.Tensor,
                       teach: torch.Tensor, n_boot: int = 2000, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    n = base.shape[0]
    recs = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g)
        gap = float(base[idx].mean() - teach[idx].mean())
        recs.append(float(base[idx].mean() - prog[idx].mean()) / max(gap, 1e-9))
    recs.sort()
    gap = float(base.mean() - teach.mean())
    return {"recovery": float(base.mean() - prog.mean()) / max(gap, 1e-9),
            "ci95": [recs[int(0.025 * n_boot)], recs[int(0.975 * n_boot)]],
            "n": n}


# ---------------------------------------------------------------------------
# card blocks
# ---------------------------------------------------------------------------
def sizes_block(field: ProgramField, active_tau: float = 1e-4) -> Dict:
    n_quats, active = 0, 0
    for k in field.spec.site_keys():
        s = q_angle2(q_normalize(field.q(*k).detach().float())).reshape(-1)
        n_quats += s.numel()
        active += int((s > active_tau).sum())
    return {"n_quats": n_quats, "raw_params": n_quats * 4,
            "bytes_fp16": n_quats * 4 * 2,
            "active_sites@1e-4": active,
            # 8-bit sparse: 3-byte quantized vector part + 4-byte index
            "bytes_8bit_sparse": active * 7 + 16}


def spectrum_certificate(field: ProgramField, base=None, tol_ortho: float = 1e-5,
                         norm_band=(0.5, 2.0), seed: int = 0) -> Dict:
    """Structural spectrum guarantee: normalized rotations orthogonal within
    tol; raw quaternion norms inside a tamper band (raw storage is
    normalize-in-forward, so legitimate drift is small — a wild norm marks
    tampering or corruption); 3 SVD spot-checks on effective matrices."""
    worst_ortho, worst_norm_lo, worst_norm_hi = 0.0, 1.0, 1.0
    for k in field.spec.site_keys():
        q = field.q(*k).detach().float().cpu().reshape(-1, 4)
        n = q.norm(dim=-1)
        worst_norm_lo = min(worst_norm_lo, float(n.min()))
        worst_norm_hi = max(worst_norm_hi, float(n.max()))
        R = q_to_R(q_normalize(q))
        eye = torch.eye(3)
        worst_ortho = max(worst_ortho, float(
            (R.transpose(-1, -2) @ R - eye).abs().max()))
    ortho_ok = worst_ortho < tol_ortho
    band_ok = norm_band[0] <= worst_norm_lo and worst_norm_hi <= norm_band[1]
    svd_ok, svd_worst, svd_n = True, 0.0, 0
    if base is not None:
        g = torch.Generator().manual_seed(seed)
        d = field.spec.d_model
        mats = [(name, p) for name, p in base.named_parameters()
                if p.ndim == 2 and p.shape == (d, d)]
        for _ in range(min(3, len(mats))):
            name, W = mats[int(torch.randint(0, len(mats), (1,), generator=g))]
            layer = 0
            import re
            m = re.search(r"\.(\d+)\.", name)
            if m:
                layer = min(int(m.group(1)), field.spec.n_layers - 1)
            q = q_normalize(field.q(layer, "attn_io").detach().float().cpu())
            R3 = q_to_R(q)
            B = torch.eye(d)
            for i in range(R3.shape[0]):
                B[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R3[i]
            Wf = W.detach().float().cpu()
            s0 = torch.linalg.svdvals(Wf)
            s1 = torch.linalg.svdvals(B @ Wf @ B.T)
            rel = float((torch.sort(s0).values - torch.sort(s1).values
                         ).abs().max() / s0.max().clamp_min(1e-9))
            svd_worst = max(svd_worst, rel)
            svd_n += 1
        svd_ok = svd_worst < 1e-4
    return {"pass": bool(ortho_ok and band_ok and svd_ok),
            "ortho_worst": worst_ortho, "ortho_tol": tol_ortho,
            "raw_norm_range": [worst_norm_lo, worst_norm_hi],
            "norm_band": list(norm_band),
            "svd_spot_checks": svd_n, "svd_worst_rel": svd_worst}


@torch.no_grad()
def identity_and_undo_certificates(model, field: ProgramField, probe: torch.Tensor,
                                   device, tol: float = 1e-3) -> Dict:
    ids = probe.to(device)
    base_logits = model(input_ids=ids).logits
    with model.program(model.identity_field().to(ids.device)):
        id_logits = model(input_ids=ids).logits
    id_delta = float((id_logits - base_logits).abs().max())
    undone = compose(invert_field(field), field)
    with model.program(undone):
        un_logits = model(input_ids=ids).logits
    un_delta = float((un_logits - base_logits).abs().max())
    scale = float(base_logits.abs().max())
    return {"identity_check": {"pass": id_delta < tol * max(scale, 1.0),
                               "max_abs_delta": id_delta, "logit_scale": scale},
            "undo_attestation": {"pass": un_delta < tol * max(scale, 1.0),
                                 "max_abs_delta": un_delta}}


def commitment_block(field: ProgramField, leverage: Optional[Dict[str, float]],
                     top_k: int = 10) -> Dict:
    spec = field.spec
    n_layers = spec.n_layers
    bands = {"early": range(0, n_layers // 3 or 1),
             "mid": range(n_layers // 3 or 1, 2 * n_layers // 3 or 2),
             "late": range(2 * n_layers // 3 or 2, n_layers)}
    lev = leverage or {g: 1.0 for g in spec.site_names()}
    work, raw_act, per_band, tops = {}, {}, {}, []
    for gname in spec.site_names():
        acts = []
        for (l, n) in spec.site_keys():
            if n != gname:
                continue
            s = q_angle2(q_normalize(field.q(l, n).detach().float()))
            acts.append(float(s.mean()))
            band = next(b for b, rng in bands.items() if l in rng)
            per_band.setdefault(gname, {}).setdefault(band, 0.0)
            per_band[gname][band] += float(s.mean()) * lev.get(gname, 1.0)
            flat = s.reshape(-1)
            for idx in torch.topk(flat, min(3, flat.numel())).indices.tolist():
                tops.append((float(flat[idx]) * lev.get(gname, 1.0),
                             f"L{l}.{n}[{idx}]"))
        raw_act[gname] = sum(acts) / max(len(acts), 1)
        work[gname] = raw_act[gname] * lev.get(gname, 1.0)
    tot = sum(work.values())
    shares = {g: v / max(tot, 1e-12) for g, v in work.items()}
    rope_share = shares.get("rope_ax", 0.0)
    actuator = ("position" if rope_share >= 0.25 else
                "content" if rope_share <= 0.05 else "composite")
    tops.sort(reverse=True)
    return {"work_shares": shares, "raw_activity_diagnostic": raw_act,
            "per_band_work": per_band,
            "top_sites": [t[1] for t in tops[:top_k]],
            "actuator_class": actuator,
            "actuator_thresholds": {"position": ">=0.25 rope_ax share",
                                    "content": "<=0.05 rope_ax share"},
            "leverage_used": lev}


def activity_block(field: ProgramField, prune_curve=None) -> Dict:
    xs = []
    for k in field.spec.site_keys():
        xs.append(q_angle2(q_normalize(
            field.q(*k).detach().float().cpu())).reshape(-1))
    s = torch.cat(xs)
    hist = torch.histogram(s, bins=torch.tensor(HIST_BINS)).hist.tolist()
    out = {"mean": float(s.mean()), "max": float(s.max()),
           "histogram_bins": HIST_BINS, "histogram": hist}
    if prune_curve is not None:
        out["prunability_curve"] = prune_curve
    return out


@torch.no_grad()
def fingerprint_block(model, field, suites: Dict[str, torch.Tensor], device,
                      library: Optional[Dict[str, ProgramField]] = None,
                      kl_blocks: Optional[torch.Tensor] = None) -> Dict:
    import torch.nn.functional as F
    vec = {name: float(ce_per_block(model, blocks, device, field=field).mean())
           for name, blocks in suites.items()}
    out = {"probe_losses": vec}
    if library and kl_blocks is not None:
        ids = kl_blocks.to(device)
        with model.program(field):
            lp_self = F.log_softmax(model(input_ids=ids).logits.float(), dim=-1)
        neigh = []
        for name, other in library.items():
            with model.program(other):
                lp_o = F.log_softmax(model(input_ids=ids).logits.float(), dim=-1)
            kij = float((lp_self.exp() * (lp_self - lp_o)).sum(-1).mean())
            kji = float((lp_o.exp() * (lp_o - lp_self)).sum(-1).mean())
            neigh.append({"name": name, "sym_kl": 0.5 * (kij + kji)})
        neigh.sort(key=lambda r: r["sym_kl"])
        out["neighbors_top3"] = neigh[:3]
        out["neighbors_all"] = neigh
    return out


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def build_card(*, model, field: ProgramField, base_id: str, program_path: str,
               device, skill_name: str, version: str = "0.1.0",
               teacher=None, teacher_id: Optional[str] = None,
               task_eval_blocks: Optional[torch.Tensor] = None,
               neutral_blocks: Optional[torch.Tensor] = None,
               probe_suites: Optional[Dict[str, torch.Tensor]] = None,
               leverage: Optional[Dict[str, float]] = None,
               library: Optional[Dict[str, ProgramField]] = None,
               prune_curve=None, provenance: Optional[Dict] = None,
               seed: int = 0) -> Dict:
    card = {"schema": SCHEMA_VERSION,
            "identity": {"skill": skill_name, "version": version,
                         "base": base_id,
                         "program_hash": (sha256_file(program_path)
                                          if Path(program_path).exists() else "n/a"),
                         "seeds": {"card": seed},
                         **sizes_block(field)}}

    if task_eval_blocks is not None:
        base_ce = ce_per_block(model, task_eval_blocks, device, field=None)
        prog_ce = ce_per_block(model, task_eval_blocks, device, field=field)
        fid = {"task_ce": {"base": float(base_ce.mean()),
                           "program": float(prog_ce.mean())}}
        if teacher is not None:
            teach_ce = ce_per_block(teacher, task_eval_blocks, device)
            fid["task_ce"]["teacher"] = float(teach_ce.mean())
            fid["recovery"] = recovery_bootstrap(base_ce, prog_ce, teach_ce,
                                                 seed=seed)
            fid["margin_teacher_minus_program"] = paired_bootstrap(
                teach_ce, prog_ce, seed=seed)
        else:
            fid["recovery"] = "N/A (teacherless skill; absolute metrics only)"
        card["fidelity"] = fid
    else:
        card["fidelity"] = "N/A"

    if neutral_blocks is not None:
        nb = ce_per_block(model, neutral_blocks, device, field=None)
        np_ = ce_per_block(model, neutral_blocks, device, field=field)
        gent = {"neutral_ppl": {"base": math.exp(float(nb.mean())),
                                "program": math.exp(float(np_.mean()))}}
        if teacher is not None:
            nt = ce_per_block(teacher, neutral_blocks, device)
            gent["neutral_ppl"]["teacher"] = math.exp(float(nt.mean()))
            dp, dt = np_ - nb, nt - nb
            gent["collateral_ratio"] = {
                "value": float(dp.mean()) / max(float(dt.mean()), 1e-9),
                "ci_note": paired_bootstrap(dp, dt, seed=seed)}
        card["gentleness"] = gent
    else:
        card["gentleness"] = "N/A"

    probe = (task_eval_blocks if task_eval_blocks is not None else
             neutral_blocks)
    certs = {"spectrum": spectrum_certificate(field, base=model.base
                                              if hasattr(model, "base") else None,
                                              seed=seed),
             "lipschitz": {"statement": "program-independent bound by "
                                        "construction (rotations are "
                                        "isometries; see tests T1/T6)",
                           "test_ref": "tests/test_quaternion.py"}}
    if probe is not None:
        certs.update(identity_and_undo_certificates(
            model, field, probe[:2], device))
    card["certificates"] = certs

    card["commitment"] = commitment_block(field, leverage)
    card["activity"] = activity_block(field, prune_curve)
    if probe_suites:
        card["fingerprint"] = fingerprint_block(
            model, field, probe_suites, device, library=library,
            kl_blocks=probe[:8] if probe is not None else None)
    card["limits"] = LIMITS_TEXT
    card["provenance"] = provenance or {}
    cert_pass = all(v.get("pass", True) for v in card["certificates"].values()
                    if isinstance(v, dict) and "pass" in v)
    card["all_certificates_pass"] = bool(cert_pass)
    return card
