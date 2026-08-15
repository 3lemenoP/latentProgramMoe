# Phase 4 specification — belief, de novo learning, and differentiation over program fields

**Version 0.1, 2026-08-15. Companion: `latent-program-fields-report.md` (results it builds on) and `latent-program-moe-spec.md` (kernels and field machinery it reuses). Target implementer: Claude Code.**

This layer turns the validated substrate into an agent: a belief over its own program, updated from experience, with differentiation (specialization) as the observable. It is designed strictly inside what the campaign established.

---

## 0. Design constraints inherited from the campaign (binding)

1. **No program algebra.** Composition by Hamilton products is retired (four independent falsifications). The orchestration layer composes by **scheduling only**: select, hot-swap, slerp-interpolate, exact-undo. No code path in this layer may multiply two *skills'* programs to obtain a third skill. (Hamilton products remain legal for geometry: undo, drift steps, exp/log maps.)
2. **Whitened geometry only.** Geometric proximity is not behavioral proximity (three sightings, report §5.3). Every distance, prior, covariance, and cluster in this layer lives in **whitened tangent coordinates** (§2), calibrated per base (§3). Raw d_geo may appear in reports as a diagnostic column, never in a decision.
3. **Perturbative regime is the operating point.** Fitted fields live at ~3–6% activity; beliefs will concentrate near identity. Tangent-space (local) machinery is therefore the *correct* v1 choice, not a shortcut. Global (Bingham-on-S³) beliefs are a v2 upgrade with a defined migration path (§2.4).
4. **Exploration is safe by theorem and by measurement.** Every sampled program is a rotation field: spectrum-preserving, Lipschitz-bounded, and empirically gentle (B2 ran random fields at all strengths; E2 measured ≈½ collateral damage). Thompson sampling needs no safety wrapper; the loop logs KL-to-base as a monitored invariant, not a gate.
5. **Two actuators.** Site groups differ in kind (content vs position) and in leverage by up to ~40×. Beliefs, whitening, and the cell-identity observables are all site-group-aware.

---

## 1. State space and maps

A program is a field z = {q_i} over sites i (site groups g(i) ∈ {attn_io, mlp_io, ffn_hidden, qk_rel, rope_ax}, per the existing `ProgramField`). This layer works in the **tangent space at identity** via the canonical maps (reuse `q_pow`'s canonicalization; sign-invariance mandatory):

```
log: unit quaternion → rotation vector    v = θ·n̂,  θ = 2·atan2(‖vec‖, w) after w≥0 canonicalization
exp: rotation vector → unit quaternion    q = (cos(‖v‖/2), sin(‖v‖/2)·v/‖v‖),  identity at v=0
```

Kernel additions to `lpm/quaternion.py`: `q_log(q) -> (...,3)`, `q_exp(v) -> (...,4)`. Test **T19**: exp∘log round-trip to 1e-9 (canonical representatives); log(−q) = log(q); log(identity) = 0.

## 2. The belief representation (v1: whitened tangent Gaussian)

### 2.1 Form

Per site i, belief over the rotation vector: v_i ~ N(μ_i, Σ_i), maintained in **whitened coordinates** ṽ_i = c_{g(i)}·v_i (so Σ is stored whitened; mean-field independence across sites — assumption stated, tested by T20's convergence sanity). The deployed program is the MAP field exp(μ).

Operations (all closed-form):
- **Fusion** (evidence with mean m, precision R): Λ ← Λ + R, Λμ ← Λμ + Rm. Additive, commutative — this is the multi-agent consensus mechanism inherited from the Bingham design, preserved exactly (**T16**: fusion order-invariance).
- **Prediction / drift**: Σ ← Σ + Q (whitened isotropic Q = q_drift²·I). Alternative for abrupt regimes: precision forgetting Λ ← λΛ, 0<λ<1. v1 default: Q-inflation; λ-forgetting behind a flag.
- **Prior**: de novo = broad isotropic N(0, σ₀²·I) in whitened coords (the proper "uninformative over experts" the compact manifold licenses, expressed locally).
- **Sampling**: draw ṽ, unwhiten, exp to a field. **T17**: sampled fields at whitened norm ρ produce KL-to-base within a declared envelope (safety observable).
- **Potency**: per-site entropy H_i = ½·log det(2πe·Σ_i); group and total potency = sums. This is the differentiation observable.

### 2.2 Why Gaussian and not Bingham (v1)

The campaign's perturbative-regime finding makes near-identity locality the *measured* truth, not an assumption; a tangent Gaussian is the Laplace form of the antipodally-symmetric Bingham there, with identical fusion additivity. **T21 (migration invariant)**: for concentrations in the operating range, Bingham-fused and Gaussian-fused posteriors agree on MAP to <1° per site — documents that v2's global upgrade changes nothing in-regime.

### 2.3 What the whitening is

For small fields, per-site behavioral displacement is quadratic in θ with a site-group coefficient measured by B2: KL ≈ L_g·s̄ with s̄ = sin²(θ/2) ≈ θ²/4. Define per-group whitening c_g² = L_g / (4·n_g) so that Σ_i‖ṽ_i‖² ≈ KL-to-base under linear-response additivity (assumption A2, gated below). Only *relative* scale across groups matters; sanity check: the resulting per-site weights must reproduce B2's per-dim leverage ordering (qk_rel ≫ attn_io ≈ ffn_hidden > rope_ax on quarter-rotary Pythia).

## 3. W0 — whitening calibration (gate before anything else runs)

B2's leverage table is base- and rotary-fraction-specific. For each working base (the rotary toy first):
1. Rerun the B2 protocol: matched-activity random fields per site group, s̄ ∈ {0.01, 0.03}, KL-to-base on ≥100 probe sequences; fit L_g (and optionally per-layer L_{g,ℓ}).
2. Set c_g; **gate T18**: random fields at equal *whitened* norm across groups produce KL within ×2 of each other. If the gate fails at group level, refine per-layer; if it still fails, the linear-response additivity assumption is broken at that strength — reduce ρ and re-gate.
3. Emit `whitening.json` (consumed by everything downstream). Deliverable: one table, one gate verdict.

---

## 4. Experiment C — the codebook (existing artifacts, no training)

**Question: do cell types already exist in the fitted library, and does the whitened metric see them?**

Data: the ~26 strong-fit fields on hand (6 atoms a–f, 20 ordered pipelines). Procedure:
1. Map every field to whitened tangent vectors (one long vector per field, ordered by site).
2. **Geometric clustering**: k-medoids over k = 2..6 in (a) whitened L2 and (b) raw sitewise d_geo (the control metric). Report silhouettes.
3. **Behavioral clustering** (the referee): pairwise symmetrized KL between program models on a shared 200-probe set; k-medoids at the same k.
4. Compare partitions by adjusted Rand index: ARI(whitened, behavioral) vs ARI(raw, behavioral).
5. Report the rate–distortion curve: whitened covering radius vs k (the requisite-variety readout — how many specialists this task measure needs).
6. Label clusters by actuator profile (mean activity share per site group) and task family.

Pre-registered predictions: **C-1** ARI_whitened > ARI_raw (the whitening fix is real) — prior 0.70. **C-2** clusters organize by actuator profile and task family, with the displacement family separating — prior 0.65. **C-3** the never-clean pair: geometric-only clustering (raw) merges behaviorally distinct fields at least once (a fourth sighting of the standing law) — prior 0.6.

**Gate**: C-1 failing means the whitening proxy is wrong at field scale → recalibrate per-layer (W0 step 2) before building the organism. C feeds D.

## 5. Experiment D — the minimal viable organism (MVO)

**Question: does a belief over one's own program buy anything over plain online SGD — and do differentiation dynamics (canalization, dedifferentiation, plasticity cost) appear as predicted?**

### 5.1 Environment

Task stream over the existing pipeline family on the rotary toy: 6 tasks (include one **never-fitted holdout pipeline** for the de novo rung), regime switches every N episodes (N ~ Uniform[20, 60], unknown to the agent), episode = one train batch + one eval batch. Streams are seeded and replayable.

### 5.2 Agent (ADF loop)

Per episode: **predict** (Σ += Q) → **act** (Thompson-sample z, or deploy MAP — flag) → **observe** (eval loss ℓ) → **refine** (T ∈ {0, 25} gradient steps from the sample on the train batch, whitened parameterization) → **fuse** (refined point as a Gaussian measurement with fixed observation covariance R_obs = r²·I; r and q_drift from a small grid on a validation stream). Log per episode: eval exact, KL-to-base of deployed program, potency (total and per group), commitment map.

### 5.3 Baselines (the experiment is the comparison)

(a) **Oracle**: per-task fixed strong-fit programs with known switch times — ceiling. (b) **Static**: one program fit on the task mixture — floor. (c) **SGD tracker**: continual gradient steps on the current field, no belief, no inflation — **the null hypothesis this experiment exists to beat**. (d) **Reset-on-spike**: loss-spike detector + refit from identity. Same compute budget per episode for (c), (d), and the agent.

### 5.4 Metrics and pre-registered predictions

Regret vs oracle; recovery time after each switch (episodes to within 5 points of oracle); potency trajectory; de novo rung: episodes-to-threshold on the holdout task starting from (i) fresh broad prior and (ii) a prior converged on another task.

- **D-1** agent beats SGD tracker on mean recovery time — prior 0.60.
- **D-2** potency tracks regime structure: falls within regimes (canalization), rises at switches under Q-inflation (dedifferentiation) — prior 0.70.
- **D-3** converged commitment maps match the corresponding direct-fit skills' activity profiles (cosine > 0.8) — the cell-identity check — prior 0.75.
- **D-4** de novo from a committed prior is slower than from a fresh prior (measured plasticity cost — Waddington canalization as a number) — prior 0.60.
- **D-5** safety invariant: no deployed or sampled program exceeds the T17 KL envelope at any point — prior 0.95 (reported, not gated).

### 5.5 Multi-agent stub (v1.1, stretch)

Two agents + an allocator routing each episode by belief-predicted loss (each agent's expected eval under its MAP). Fusion of beliefs on any shared task uses §2.1 additivity verbatim. Metric: specialization index = JS divergence between the two agents' normalized commitment maps over time. **M-1**: specialization emerges under competence routing and not under random allocation — prior 0.65. Nothing else from the orchestration vision (EFE planning, codebook-driven spawning, VSM wiring) is in scope for v1.

---

## 6. Code plan

```
lpm/whitening.py      # calibrate + load whitening.json; whiten/unwhiten helpers
lpm/belief.py         # WhitenedGaussianBelief: predict, fuse, sample, entropy,
                      #   map_field (exp(μ)), commitment_map, potency
lpm/quaternion.py     # + q_log, q_exp (T19)
envs/task_stream.py   # seeded regime-switching streams over the pipeline family
scripts/calibrate_whitening.py   # W0
scripts/codebook.py              # Experiment C (reads runs/e3strong + D6 fields)
scripts/mvo.py                   # Experiment D (+ baselines, shared budget harness)
tests/test_belief.py             # T16–T21
```

Tests: **T16** fusion additivity/order-invariance; **T17** sample-safety envelope (KL vs whitened norm curve, monotone, bounded); **T18** whitening gate (equal whitened norm ⇒ KL within ×2 across groups); **T19** exp/log round-trip + sign-invariance; **T20** stationary-task sanity (posterior concentrates on a fixed task; potency monotone-decreasing after burn-in; MAP within 5 points of direct fit); **T21** Bingham–Gaussian agreement in-regime.

## 7. Do-not list

- Do not compose skills by Hamilton products anywhere in this layer (scheduling only; geometry ops exempt).
- Do not use raw d_geo in any decision; whitened coordinates only (raw allowed as a report column).
- Do not run C or D before the W0/T18 gate is green on the working base.
- Do not compare the agent to baselines at unequal per-episode compute.
- Do not begin with Bingham-on-S³; tangent Gaussian is v1 by measured regime (§2.2), Bingham is the v2 global upgrade behind T21.

## 8. Definition of done

W0: leverage table + T18 verdict + `whitening.json`. C: cluster report with both ARIs, rate–distortion curve, actuator-labeled clusters, C-1..C-3 scored. D: regret/recovery table across agent + 4 baselines, potency and commitment-map trajectories, de novo comparison, D-1..D-5 scored. Reports in the established format; every prediction scored inline against its prior.
