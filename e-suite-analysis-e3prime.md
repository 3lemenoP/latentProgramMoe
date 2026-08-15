# E-suite results — analysis, E3′ specification, todo

**Version 0.1 — companion to `latent-program-moe-spec.md` (v0.1).**
Covers: analysis of E0/E2/E3 as reported; the E3′ diagnostic-and-rescue suite; deltas to the main spec; prioritized todo. E1/E4 reports not yet reviewed here — E2's endpoints already carry E1 signal (see §2.2); record actual E1 numbers when available. E4 is orthogonal to the E3 outcome.

---

## 1. Executive summary

E0 passed emphatically: fine-tuning barely moves singular spectra (median Mirsky ratio 0.028 vs threshold 0.3), so gains stay off and the necessary condition for conjugation-expressivity holds with room to spare. E2 lost the pairwise-merge benchmark to weight-linear interpolation — the pre-registered linear-mode-connectivity risk — but produced two real findings: conjugated experts cause roughly half the collateral damage of the true fine-tune on neutral text (ppl 121 vs 232 at the french endpoint), and per-layer slerp schedules work as a control surface. E3 failed at 0.0% ordered accuracy for both algebras, **but the failure pattern is diagnostic**: the two composition orders produced behaviorally identical models, meaning the commutator of the fitted programs is behaviorally silent — almost certainly because the two skills were fitted onto near-disjoint site supports, a regime our own regularizer encourages and in which any algebra abelianizes. The non-abelian hypothesis has not yet been tested; only its abelian shadow has. E3′ (§4) is the suite that actually engages it, with a decision tree (§4.7) mapping every outcome to a project narrative.

---

## 2. Results analysis

### 2.1 E0 — Mirsky audit: PASS

Median ratio 0.026–0.033 per expert, 0.028 overall: at most ~3–7% of any expert's weight displacement is provably unreachable by rotation. Decision per spec §10: `enable_gains = false`.

Two qualifications. First, Mirsky is necessary, not sufficient — it bounds the *full* orthogonal conjugation orbit, while the architecture has only the block-diagonal one; actual reachability is measured by E1-style fits, and E2's endpoints show a residual gap (§2.2). Second, the unreachable mass is not uniform: the worst matrices are consistently `attn.c_proj` (all layers) and late-layer `mlp.c_fc` (esp. sentiment: h.9–h.11 c_fc at 0.05+). If gains are ever revisited, they go there first — per-site gains on `attn_io` exit and late `ffn_hidden` — not everywhere.

### 2.2 E2 — geodesic merging: REFRAME

At the midpoint, weight-linear beats program slerp on both task CEs and neutral ppl. Two confounds and one finding:

- **Endpoint gap (E1 signal).** Program endpoints trail the true experts by ~0.19 nats (french) and ~0.09 nats (caps). The merge comparison starts from behind; midpoint deltas relative to own endpoints are closer but weight-linear still degrades least. Same-base GPT-2 fine-tunes sit in one linearly connected basin — the easiest possible setting for weight interpolation. The pairwise-merge claim should not lead the story at this scale.
- **The gentleness result.** The french fine-tune costs 231.9 neutral ppl; the program reconstruction of it costs 121.3 — half the collateral for 0.19 nats of task CE (caps: 43.6 vs 45.5, parity). This is the spectrum-preservation guarantee showing up behaviorally: conjugated experts are regularized experts. No weight-space format has this property by construction. This is now a headline claim.
- **Control surface confirmed.** early→A holds CE A to 4.11 at α=0.5 (vs 4.26 uniform) at legible cost to CE B and neutral. The per-layer dial does what it says.

Honest E2 headline: *not a better pairwise merger; a safer expert format with per-layer control.* The unique-operations story (inverses, transport, composition) transfers its burden to E3′.

### 2.3 E3 — order sensitivity: FAIL, with mechanism

The table contains its own diagnosis:

- Individual skills fit perfectly (1.000 exact both).
- `compose(b,a)` and `compose(a,b)` are behaviorally **identical**: 0.170 vs 0.174 token acc on `a_then_b`; 0.263 vs 0.265 on `b_then_a`. The order information in the Hamilton product produced no behavioral difference. Both compositions behave like the same unordered blend, slightly closer to `b_then_a` in token overlap.
- The abelian baseline scores the same pattern — consistent with the non-abelian machinery having been idle, not defeated.

---

## 3. Diagnosis

### 3.1 The commutator is behaviorally silent

compose(b,a) and compose(a,b) differ exactly by the group commutator of the fitted fields. Their behavioral identity means that commutator lies (approximately) in the behavioral stabilizer — the cryptic directions predicted earlier in the design. E3 therefore did not measure "can non-abelian structure carry order"; it measured "do these two particular programs have a live commutator," and the answer was no.

### 3.2 The disjoint-support mechanism

Sitewise, `hamilton(x, y) = y` whenever `x = identity`. If skill A rotates one set of sites and skill B a disjoint set — plausible for mechanistically different skills like prepend (positional/early) and reverse (attention-pattern) — then at every site at least one factor is ≈ identity, the sitewise products commute **exactly**, both compositions collapse to the union-of-rotations blend, and the observed table follows in full: 0.0 exact on both pipelines, partial token overlap with each, order-blindness, abelian parity.

### 3.3 Our regularizer manufactured the regime

Phase-1's λ_reg pulls every site toward identity: each skill is pushed to rotate as few sites as little as possible. Independently fitted skills therefore carve out disjoint minimal supports — the exact condition of §3.2. The fitting objective abelianized the library before the algebra was ever engaged. This is an artifact hypothesis, and it is checkable in minutes (D1).

### 3.4 Strong vs weak composition hypotheses

- **Strong (now disconfirmed at this distance/regime):** the group product of *independently fitted* programs equals behavioral pipelining, zero-shot. The deep reason it was always fragile: conjugation composes as an automorphism, `f_{R₂R₁} ≠ f_{R₂} ∘ f_{R₁}` for nonlinear f — the symmetry-breaking nonlinearity that makes programs expressive is exactly what denies algebra→semantics for free.
- **Weak (untested):** the program manifold is the right *coordinate system* for composition — semantic composition is a gauged/learned operation that is low-dimensional in z (e.g., a Hamilton product after a learned gauge alignment), and relative increments Δ = z_ab ⊗ z_a* transport across contexts. E3′ tests the weak hypothesis directly. The ledger, inverse, and transport claims survive if and only if some version of the weak hypothesis holds.

---

## 4. E3′ specification

All experiments reuse the Phase-1 fitting machinery and the saved `ProgramField` format. Quaternion math per main spec §1; new kernels in §4.0. "Skill c" below is a third synthetic skill, `append token ⟨B⟩` (composes non-trivially with reverse).

### 4.0 Kernel additions (`lpm/quaternion.py`)

```python
def q_conjugate(q):          # (w, x, y, z) -> (w, -x, -y, -z); inverse for unit q
def q_angle2(q):             # sign-invariant activity s = 1 - w^2 = sin^2(theta/2), in [0, 1]
def q_commutator(a, b):      # a ⊗ b ⊗ a* ⊗ b*  (unit inputs)
def d_geo(a, b):             # arccos(|<a, b>|), reporting metric (losses stay chordal)
```

New tests: **T11** `q_to_R(q_conjugate(q)) == q_to_R(q).T`; `hamilton(q, q_conjugate(q)) == ±identity`; `q_commutator` of coaxial quaternions == identity.

### 4.1 D1 — diagnostic battery (no training; run first)

`scripts/diagnose_e3.py --fields z_a.pt z_b.pt --probe <prompts>` → markdown report:

1. **Activity profiles**: per-site s = q_angle2(q) for each skill; per-layer/per-site-group summaries.
2. **Support overlap**: cosine ⟨s_A, s_B⟩/(‖s_A‖‖s_B‖) and OVL = Σ min(s_A,s_B)/min(Σs_A, Σs_B).
3. **Geometric commutator**: κ_i = q_angle2(q_commutator(a_i, b_i)); report mean/max, per layer.
4. **Behavioral commutator**: mean KL between compose(b,a)-model and compose(a,b)-model outputs on probe prompts.

Predictions under the §3.2 mechanism: κ_mean ≈ 0, low support cosine, behavioral KL ≈ 0. D1 is report-only (no gate); it selects between §3.2 (artifact) and "overlapping but still silent" (deeper problem).

### 4.2 D2 — oracle pipelines (capacity control)

Fit `z_ab` on `a_then_b` data and `z_ba` on `b_then_a` data, identical hyperparameters to the z_a/z_b fits. Metrics: exact match (expect ≈ 1.0); then geodesic distances d_geo(hamilton(z_b, z_a), z_ab) per layer, against baselines d_geo(z_a, z_ab), d_geo(z_b, z_ab), d_geo(id, z_ab).

- Oracle < 0.9 exact → **capacity problem**: a single program cannot represent the pipeline; all composition claims paused pending site/layer scaling study.
- Oracle ≥ 0.9 and Hamilton prediction *near* z_ab but behavior wrong → behavior map is sharp along specific directions; report which sites carry the error.
- Oracle ≥ 0.9 and Hamilton *far* → the rule is wrong as stated; proceed to D3/D6 (gauge story).

### 4.3 D3 — relative increments and transport (the reframed flagship)

Define the increment **Δ_{b|a} = z_ab ⊗ z_a\*** ("b in the context of a"; sanity: Δ_{b|a} ⊗ z_a ≡ z_ab, algebraic identity). Fit `z_c` and oracle `z_cb` (c then b). Evaluate two zero-shot candidates for `z_cb`:

- **Left transport**: ẑ = Δ_{b|a} ⊗ z_c (increments are absolute).
- **Adjoint transport**: g = z_c ⊗ z_a\*; ẑ = (g ⊗ Δ_{b|a} ⊗ g\*) ⊗ z_c (increments are frame-relative — the b⊗a⊗b\* mechanism).

Metrics: exact match of each ẑ on `c_then_b`, plus d_geo(ẑ, z_cb). **Pass: either candidate ≥ 0.5 exact zero-shot** (baseline from E3 is 0.0). Which candidate wins is itself a result: absolute vs frame-relative skill increments.

### 4.4 D4 — shared-support refit (engage the algebra)

Refit z_a, z_b **jointly** under one of: (i) λ_reg = 0; (ii) overlap bonus −λ_o Σ_i s_A,i · s_B,i added to the joint loss; (iii) hard shared mask: both skills restricted to the top-K sites by z_ab activity (from D2). Then rerun the E3 composition table and D1.

**Gate: ordered margin > 20 points over the abelian baseline (refit under the same constraint), with κ_mean now materially > 0.** If D4 passes, Phase 3 unblocks with the amended fitting protocol (§6), and §3.3 is confirmed as the E3 root cause.

### 4.5 D5 — strength scan (validity radius)

For λ ∈ {0.25, 0.5, 0.75, 1.0}: z(λ) = slerp(identity, z, λ) sitewise for both skills; evaluate composition quality vs λ. Near identity, commutators are second-order (⟦exp(εX), exp(εY)⟧ = exp(ε²[X,Y] + O(ε³))), so order effects must vanish at small λ; if the algebra is semantically valid anywhere, quality peaks at intermediate strength. Output: the curve; informs the operating strength for D3/D4/D6.

### 4.6 D6 — learned composition operator (semantics vs coordinates)

Build a library of ~5 basic skills and oracle-fit ~10–20 ordered pipelines (cheap at this scale). Train two operators on triples (z_x, z_y, z_yx-oracle), test on held-out pairs:

- **Constrained**: C(y, x) = g₁ ⊗ y ⊗ g₂ ⊗ x ⊗ g₃ with three learned gauge fields (reduces to plain Hamilton at g = id).
- **Unconstrained**: MLP on concatenated raw quaternions → raw output, normalized.

Decision: constrained ≈ unconstrained ≫ zero-shot Hamilton → composition is a *gauged product*; the algebraic story survives in gauged form and the operator ships as `compose_learned()`. Unconstrained ≫ constrained → the group is a parameterization, not semantics; retire zero-shot-algebra claims. Both fail → composition is not low-dimensional in z; retire composition claims entirely.

### 4.7 Decision tree

```
D2 oracle fails ──────────────► capacity problem: pause composition claims,
                                run site/layer scaling study
D2 passes
 ├─ D1 shows κ≈0 (disjoint support)
 │   └─ D4 refit
 │       ├─ margin > 20 ──────► algebra works when engaged: amend Phase-1
 │       │                      objective (§6), unblock Phase 3, rerun E3
 │       └─ still fails ──────► D6 decides:
 │            ├─ gauged product works ─► ship compose_learned; ledger/
 │            │                          transport claims survive (gauged)
 │            ├─ only MLP works ───────► group = coordinates; keep safety/
 │            │                          control claims, drop algebra claims
 │            └─ both fail ────────────► retire composition; substrate claims
 │                                       (E0, gentleness, control, Phase 4) stand
 └─ D3 transport (independent axis): either candidate ≥ 0.5 exact
     ────────────────────────► transport is real — headline result regardless
                                of raw-product status; winner identifies
                                absolute vs frame-relative increments
```

---

## 5. Claims register (post E0/E2/E3)

**Established:** identity/spectrum/Lipschitz guarantees (T-suite); fine-tune spectra barely move (E0, 0.028); conjugated experts are gentler — ~half the neutral-text damage of the true fine-tune (E2); per-layer slerp schedules as a control surface (E2); ~86KB spectrum-safe skill artifact.

**Weakened / reframed:** pairwise merging — parity-at-best against weight-linear in the linearly-connected regime; the merge story moves to safety + per-layer control + k>2 + group operations.

**At risk pending E3′:** zero-shot algebraic composition; order semantics; skill ledgers and exact un-skilling *as behavioral claims* (the inverse is exact in the group; whether removal behaves is D3-adjacent); conjugation transport (now D3, the reframed E5 flagship).

**Untested regardless of E3′:** scale trend (E1 recovery vs model size); QRoPE/axis field (needs rotary base — Pythia-410m); Phase 4 belief layer; per-token routing.

---

## 6. Deltas to `latent-program-moe-spec.md`

1. **§7 Phase 1**: add warning — λ_reg induces disjoint supports across independently fitted skills and abelianizes the library (§3.3 here); when skills are destined for composition, fit jointly with a shared-support term or mask (D4 protocol). Record the D4-winning variant as the default.
2. **§7 Phase 3 gate**: replace "E3 passes" with "D4 gate passes, or D6 ships a composition operator."
3. **§8 kernels**: add `q_conjugate`, `q_angle2`, `q_commutator`, `d_geo`; test T11.
4. **§10**: E3 marked superseded by E3′ (this document §4); E5 (conjugation transport) merged into D3.
5. **Appendix A**: no change (gains off per E0).

---

## 7. Todo (priority order)

1. **[~30 min]** D1 on the existing fitted fields (`diagnose_e3.py`). Decides artifact-vs-deeper before anything else runs.
2. **[~15 min]** Kernel additions + T11 (§4.0).
3. **[hours]** D2 oracle fits z_ab, z_ba; distance report.
4. **[hours]** D3: fit z_c, z_cb; run both transport candidates. Highest-value single result in the suite.
5. **[hours]** D4 shared-support refit; rerun composition table; gate check.
6. **[~1 day]** D5 strength scan (can run parallel to 3–5).
7. **[1–2 days]** D6 operator study (needs the D2/D3 oracle fits as training triples; batch the remaining pipeline fits).
8. **[15 min]** Apply §6 deltas to the main spec.
9. **[after D1–D4]** Update pitch/positioning from the claims register (§5): lead with gentleness + control + safety today; algebra claims move to "contingent on E3′" until the tree resolves.
10. **Unchanged/deferred:** E1/E4 report review when available; Pythia-410m QRoPE run; Phase 4 (Bingham/Thompson) build.
