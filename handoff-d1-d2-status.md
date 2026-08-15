# Handoff — project status after D1/D2 (for Claude Code)

**Date: 2026-08-15. Extends: `latent-program-moe-spec.md` v0.1 and `e-suite-analysis-e3prime.md` v0.1. Where this document conflicts with those, this document wins.**

---

## 0. Current position, in four sentences

E3 failed (0.0% ordered accuracy, both algebras), D1 then **rejected** the disjoint-support explanation, and D2's oracle results split (one pipeline representable, one suspect). The unified diagnosis across all three: **λ_reg has linearized the group** — every fitted program lives at ~3% activity, where commutators are second-order, Hamilton products reduce to first-order BCH (rotation vectors adding), and the framework is behaviorally operating as a task-vector method, which is already known not to compose ordered pipelines. The algebra has not been disproven; it has not yet been driven. The plan below (P0–P5) drives it: a free exponentiation probe, then strong-field (λ_reg = 0) refits of everything, then the reruns — with D2's automated verdict overruled and its gating logic to be fixed.

---

## 1. Results ledger (all runs to date, facts only)

| run | result | status |
|---|---|---|
| E0 Mirsky audit (GPT-2, 4 experts) | overall median ratio **0.028** (threshold 0.3); worst: `attn.c_proj` all layers, late `mlp.c_fc` ~0.04–0.07 | PASS — `enable_gains=false` |
| E1 (via E2 endpoints) | program endpoints trail true experts by ~0.19 nats (french) / ~0.09 (caps) | partial recovery; full E1/E4 reports not yet reviewed |
| E2 merge (french+caps) | weight-linear beats program slerp at midpoint; **program french expert: neutral ppl 121.3 vs true expert 231.9** (caps ~parity); early→A schedule works | pairwise-merge claim reframed; gentleness + control confirmed |
| E3 order sensitivity (4-layer from-scratch base: copy/prepend/reverse) | 0.0% ordered exact both algebras; compose(b,a) ≈ compose(a,b) behaviorally (token accs 0.170/0.174 and 0.263/0.265) | FAIL — diagnosed below |
| E3 refit (fields used by D1) | z_a prepend **1.000** exact (loss →0.015); z_b reverse **0.970** (loss →~0.044) | fields at `runs/e3/z_a.pt`, `z_b.pt` |
| D1 diagnostic battery | support cosine **0.3924**, OVL **0.4798**; mean activity **0.0248 / 0.0302**; κ mean **0.003066**, κ max **0.335** (layer 2); site groups: ffn_hidden 0.033/0.044, qk_rel 0.036/0.025, attn_io 0.004/0.006, mlp_io 0.003/0.004; behavioral KL compose(b,a)‖compose(a,b) = **0.00494** | run; interpretation in §2.1 |
| D2 oracle pipelines | z_ab (prepend→reverse): **0.955 exact / 0.987 token**, smooth loss →0.056. z_ba (reverse→prepend): **0.775 / 0.929**, noisy plateau 0.13–0.16 at step 1400. Distances (mean per-site geodesic): d(hamilton(b,a), z_ab)=**0.1447**; d(z_a, z_ab)=0.1759; d(z_b, z_ab)=**0.1020**; d(id, z_ab)=**0.1373** | script printed blanket FAIL — **overruled**, see §2.2 |

---

## 2. Interpretation state (what we believe and why)

### 2.1 D1: disjoint-support rejected; second-order strength suppression confirmed

Cosine 0.39 / OVL 0.48 is substantial overlap — the §3.2 mechanism from the companion doc is dead. The live mechanism is arithmetic: mean activities s = 0.025/0.030 give site angles θ = 2·arcsin(√s) ≈ 0.317/0.348 rad; for small rotations the group-commutator angle is ceilinged at θ_c ≈ θ_A·θ_B (orthogonal axes) ≈ 0.110 rad, i.e. κ_ceiling = sin²(θ_c/2) ≈ **0.0030**. Measured κ mean: **0.00307**. At mean-field level (caveat: heavy tails — κ max 0.335 means one site pair carries a ~71° commutator) the commutator runs at its kinematic ceiling given the strengths: **axes are near-maximally misaligned where both skills are active; amplitude is the only suppressor.** A ~6° average field difference between the two composition orders is behaviorally invisible (KL 0.005). Note also: the `diagnose_e3.py` auto-reading line printed the disjoint-support parenthetical, contradicting its own numbers — fix in §3.

Secondary finding: skills live almost entirely in `ffn_hidden` and `qk_rel`; `attn_io`/`mlp_io` idle at ~0.003–0.006. Implications in §7.

### 2.2 D2: the automated verdict is wrong; per-pipeline reality

The banner "single program cannot represent the pipeline — pause composition" is an overclaim: the gate fired on z_ba only. **z_ab passed at 0.955**, and that is itself a headline substrate result: a frozen base executing a novel two-stage behavior it was never trained on, via pure rotation fields. z_ba at 0.775 has an unconverged signature (loss still oscillating 0.132→0.161→0.133 at step 1400; reverse-type fits were already the slow ones). Capacity vs optimization is unresolved; P2 resolves it. Do **not** treat the tree's "capacity" terminal as reached.

### 2.3 The distance table: what it shows and what confounds it

The Hamilton prediction is *farther from the oracle than the identity program* (0.145 > 0.137), and the oracle's nearest neighbor is z_b alone (0.102): z_ab is "reverse plus a small tweak," and adding z_a's rotation moves *away* from it. Mechanistic hypothesis (plausible, unverified): z_a implements emit-⟨A⟩-**first**, while prepend-then-reverse needs ⟨A⟩ emitted **last** — shared semantics, disjoint mechanism, so the component rotation is directionally useless for the composite. Two confounds before treating this as final: (i) program-space distance ≠ behavioral distance (stabilizer degeneracy: many z realize one behavior); (ii) the oracle was fitted under λ_reg and is therefore the *minimum-norm* representative, systematically sheared toward identity and away from any product of full-strength components. Both confounds are removed by P1's λ_reg = 0 refits plus the new report columns in §3.

### 2.4 The unified diagnosis

One regularizer, three symptoms: E3's order-blindness (commutator second-order at weak field), D1's κ-at-ceiling-but-tiny, D2's min-norm oracle shear. In the weak-field limit the group is its own tangent space and the method degenerates to task arithmetic. Everything downstream (D3 transport, D4 gate, D6 operator) must run on **strong-field fits** or it tests the abelian shadow again.

### 2.5 Open puzzle (logged, not theorized)

z_ba should naively be the easier direction (its ⟨A⟩ lands at constant position — token 1) yet fits worse. It is also the pipeline requiring a position-*conditional* program (prepend-mode at position 1, reverse-mode after). P2's failure decode will identify whether the miss is first-token or reversal structure. Do not build on this hypothesis until decoded.

---

## 3. Code changes required (do these first)

1. **`lpm/quaternion.py` — add `q_pow`** (new kernel; not yet implemented):
   ```python
   def q_pow(q, lam, eps=1e-8):
       # canonicalize FIRST: q and -q are the same rotation, but powering
       # must take the shortest representative or results diverge
       q = torch.where(q[..., :1] < 0, -q, q)
       w, v = q[..., 0], q[..., 1:]
       vn = v.norm(dim=-1)
       theta = 2.0 * torch.atan2(vn, w)                    # in [0, pi]
       nhat = v / vn.clamp_min(eps).unsqueeze(-1)
       half = 0.5 * lam * theta
       out = torch.cat([torch.cos(half).unsqueeze(-1),
                        torch.sin(half).unsqueeze(-1) * nhat], dim=-1)
       ident = torch.zeros_like(q); ident[..., 0] = 1.0
       return torch.where((vn < eps).unsqueeze(-1), ident, out)
   ```
   **T12 tests:** `q_pow(q,1) == canonical(q)`; `q_pow(q,0) == id`; `q_pow(q,2) == ±hamilton(q,q)`; `q_pow(-q,lam) == q_pow(q,lam)` (sign invariance); `q_to_R(q_pow(q,lam))` continuous in lam.
2. **`scripts/e3prime.py`:** (a) `.detach()` before the `float()` conversions (current UserWarning); (b) `oracle_pass` must be **per-pipeline**, and the FAIL message must name which pipeline failed — never claim both; (c) add anchor distances `d(id, z_a)`, `d(id, z_b)`, `d(id, hamilton(b,a))`; (d) add a **behavioral column**: mean KL(candidate-model ‖ oracle-model) on the probe set for candidates {hamilton(b,a), z_a, z_b, id}, so geometric and behavioral distance are separated; (e) dump 20 decoded z_ba failures classified as {first-token error, reversal error, length error, other}.
3. **`scripts/diagnose_e3.py`:** fix the auto-reading line — it printed the disjoint-support parenthetical against contradicting numbers. Correct logic: report which of three regimes the data indicates: (i) disjoint support (low cosine AND κ far below the §2.1 ceiling), (ii) **second-order suppression** (κ ≈ ceiling given activities — compute and print the ceiling), (iii) live-but-stabilizer-bound (κ ≫ ceiling-scale yet behavioral KL ≈ 0). Add anchor KLs: compose-vs-base and z_a-model-vs-z_b-model, so the order-KL has a denominator.

---

## 4. Amended plan (ordered; P0 is minutes, P1 is the decisive run)

**P0 — exponentiation probe (no training).** For λ ∈ {1, 1.25, 1.5, 2}: build z_a^λ, z_b^λ via `q_pow`. Gate 1: own-task retention — each amplified skill keeps ≥ 0.9 of its λ=1 exact score (amplification overshoots the task optimum; retention is the risk that decides the branch). If retained: rerun the composition table, κ, and order-KL vs λ. Predictions under §2.1: commutator angle ~λ² (κ(λ=2) ≈ 0.05); order-KL leaves zero roughly quadratically. Outcomes: κ grows on schedule AND order-KL leaves zero → strength was the whole story, proceed to P1 with confidence; κ grows but order-KL stays pinned → commutator is genuinely stabilizer-bound → weight shifts to D3/D6; skills break at λ > 1 → P1 is the only path (probe inconclusive, not negative).

**P1 — strong-field refits (the de-confounder).** Refit z_a, z_b, z_ab, z_ba with **λ_reg = 0** (secondary sweep: 1e-5), 3000 steps, cosine lr decay. Then rerun: D1 battery (expect activity ≫ 0.03, κ mean ≫ 0.003), the full distance table with §3 anchors, and the E3 composition eval — reported side-by-side weak vs strong. **The sharpened falsifier:** if strong-field oracles still sit closer to identity than to the Hamilton prediction, AND warm-started fits (P2) converge easily from it, the product rule is wrong rather than under-driven — composition claims move entirely to D3/D6.

**P2 — z_ba disambiguation.** Three warm starts, same budget: from identity (control), from hamilton(a,b) (the algebraic guess), from z_b. Report steps-to-0.9-exact per init — this converts the capacity question into a *fine-tune-distance* metric, which is the number we actually want. Plus: 2× steps at decayed lr, 3 restarts, overfit-32-examples check (if it can't memorize 32, capacity is real). Run the §3.2(e) failure decode.

**P3 — D3 transport, unchanged in design, run on strong-field fits only.** Fit z_c (append ⟨B⟩) and oracle z_cb at λ_reg = 0; evaluate both transport candidates (left: Δ_{b|a} ⊗ z_c; adjoint: g Δ_{b|a} g\* ⊗ z_c with g = z_c ⊗ z_a\*). Pass: either ≥ 0.5 exact zero-shot. Still the highest-value single result.

**P4 — D5 re-aimed.** The companion doc's downward scan (slerp toward identity) is the **wrong direction** — fitted programs are already in the abelianized regime. New D5 = strength scan upward: q_pow λ ∈ [1, 2.5] on strong-field fits + the λ_reg sweep at fit time. **D4 status:** shared-mask variant (iii) retired — overlap exists; variants (i)/(ii) are subsumed by P1.

**P5 — D6 unchanged, contingent on P1–P3 outcomes.** Note the pre-registered risk: the constrained operator C(y,x)=g₁⊗y⊗g₂⊗x⊗g₃ is a rigid motion of its inputs and cannot *suppress* a component (which §2.3 suggests composition may require); if only the unconstrained MLP works, that is the "coordinates, not semantics" verdict, not a bug.

---

## 5. Decision tree — current position

```
D2 oracle ── z_ab PASS (0.955) ── z_ba 0.775, UNRESOLVED
                  │
        ►► YOU ARE HERE: un-drawn branch —
           "one-sided oracle miss; optimization and regularization unresolved"
                  │
        P0 probe ──┬─ order-KL wakes with λ ──► strength story confirmed
                   ├─ κ grows, KL pinned ─────► stabilizer-bound → D3/D6 decide
                   └─ skills break at λ>1 ────► P1 only
        P1 strong-field refits ── rerun D1/D2/E3 tables
                   ├─ composition works strong-field ──► amend Phase-1 (no λ_reg
                   │                                     for composable skills),
                   │                                     unblock Phase 3
                   └─ falsifier holds (§4 P1) ─────────► product rule wrong;
                                                         all weight on D3/D6
```

Do NOT: treat the D2 banner as gate-final; run D5 downward; fit anything destined for composition with λ_reg on; report κ without its activity-ceiling denominator.

---

## 6. Claims register deltas (vs companion doc §5)

**Add to Established:** a frozen base can execute a novel two-stage pipeline it was never trained on via pure rotation fields (z_ab, 0.955 exact) — substrate expressivity independent of the composition algebra.
**Amend At-risk:** zero-shot algebraic composition is untested-at-strength, not disproven; the tested regime was the abelian tangent space.
**Unchanged:** gentleness (E2), control surface (E2), spectrum guarantees, scale/QRoPE/Phase-4 untested.

---

## 7. Architecture notes surfaced by D1

`attn_io`/`mlp_io` idle at ~0.003–0.006 while `ffn_hidden`/`qk_rel` carry the skills (0.025–0.044): (a) interpretability nugget — skills = hidden-frame reorientation + QK-plane selection; (b) add config flag `drop_interface_fields` (30% fewer sites) as a cost lever, default off pending strong-field rerun — the interface fields may only participate at higher strength; (c) if kept, give `attn_io`/`mlp_io` their own lr multiplier so P1 tests participation, not initialization inertia.
