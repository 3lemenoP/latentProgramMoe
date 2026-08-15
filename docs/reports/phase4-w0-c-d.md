# Phase 4 v1 — W0 whitening, Experiment C (codebook), Experiment D (MVO)

L4 studio runs 2026-08-15. Library: regenerated deterministically on the L4
(28 fields: atoms a–f, 20 pipelines, oracles z_ab/z_ba; original Studio-1
checkpoints unavailable — same seeds, same protocol, self-consistent).
Predictions scored inline against `phase4-belief-differentiation-spec.md`.

## W0 — whitening calibration (gate)

The naive protocol (single L_g from small-s̄ random probes) failed T18 at
**×105**: on the toy base the leverage ordering *inverts* vs Pythia
(attn_io/mlp_io L≈24–26; ffn_hidden L≈0.47; qk_rel L≈1.1 — no rotary, so
qk_rel loses its position coupling), and the weak groups respond
superlinearly in s̄. Two substrate facts fell out:

1. **Random ffn_hidden rotations are behaviorally almost free even at
   half-turn strength** (KL 0.03 at θ≈3.3 rad) while fitted fields
   concentrate their activity exactly there: the group has an enormous
   behavioral stabilizer; gradients find the rare live directions.
2. **KL at fixed (group, s̄) is heavily direction-dependent** — single-draw
   calibration cannot meet a ×2 gate.

Fixes (spec §3 escalations + additions): calibrate L_g **at the operating
strength** (self-consistent iteration at gate ρ) and **average ≥4 random
draws** per measurement. Result: **T18 PASS on both bases** (e3 toy:
max/min 1.99 @ρ=0.1; rotary toy: pass after seed-averaging).

## Experiment C — codebook

- **C-1 (ARI_whitened > ARI_raw, prior 0.70): FAILED** on the mean over k
  (0.097 vs 0.187), though whitened wins at fine k (k=5: 0.24 vs 0.20;
  k=6: 0.32 vs 0.23). Coherent with W0's stabilizer finding: whitening is
  calibrated on *random* directions, but fitted fields live in exactly the
  non-random directions random probes cannot price. The v2 fix is
  curvature-aware (Fisher/behavioral) whitening, not per-layer refinement.
- **C-2 (clusters organize by task family, displacement family separates,
  prior 0.65): CONFIRMED** in the behavioral partition: k=4 gives
  {displacement/offset family: ba, be, cb, da, db}, {rotl family: d, cd,
  dc, de, ed}, {reverse-content family: b, ab, ac, ad, bc, c, ca, eb, fb},
  {light/marker family: a, e, f, ae, bd, ce, ea, ec}.
- **C-3 (geometric clustering merges behaviorally distinct fields, prior
  0.60): CONFIRMED in bulk** — 50–108 merged-distinct pairs per k; worst
  offenders e.g. `c` vs `cb` at sym-KL 2.5 while co-clustered raw. Fifth
  sighting of "geometric proximity ≠ behavioral proximity."
- Rate–distortion: whitened covering radius declines gently (1.55 → 1.40
  over k=2..6) — no sharp elbow; the library does not compress to a few
  specialists under this metric.

## Experiment D — minimal viable organism

Environment sane (oracle 0.941, static floor 0.184). Grid winner
r_obs=0.1·σ₀, q_drift=0.1·σ₀. Two v1 scale lessons were required to make
the agent function at all (both now in code): whitened-isotropic priors
put radian-scale raw rotations on low-leverage groups — KL-safe but far
from task solutions — so per-site prior widths are capped at raw θ≤0.3;
and refine must optimize in raw rotation-vector geometry (Adam in whitened
coords maps a uniform lr to ~lr/c_g raw steps).

| method | mean acc | regret | mean recovery (eps) |
|---|---:|---:|---:|
| oracle (ceiling) | 0.941 | 0 | — |
| reset-on-spike | 0.713 | 0.228 | **41.5** |
| SGD tracker (null) | 0.710 | 0.232 | 47.4 |
| **agent (Thompson-act)** | 0.575 | 0.366 | 53.1 |
| static (floor) | 0.184 | 0.757 | 251 |

- **D-1 (agent beats SGD tracker, prior 0.60): FAILED** in the
  Thompson-act configuration — acting on posterior samples pays sampling
  variance at eval every episode. (MAP-acting variant run separately —
  appended below.)
- **D-2 (potency falls in-regime, rises at switches, prior 0.70):
  CONFIRMED** — canalization and Q-inflation dedifferentiation both appear.
- **D-3**: not auto-scored in this run (commitment maps logged every 25
  episodes in `runs/mvo/report_mvo.json`; scoring deferred).
- **D-4 (de novo slower from committed prior, prior 0.60): CONFIRMED** —
  14 vs 9 episodes to threshold: Waddington canalization as a number.
- **D-5**: KL envelope logged as activity spot-checks only in v1 (not
  gated, per spec).

## Reading

The differentiation *observables* behave exactly as predicted (D-2, D-4,
and C-2's family structure), while the two *proxy assumptions* — random
leverage as behavioral metric (C-1) and belief-as-planner beating SGD
(D-1) — failed informatively. Both failures point the same direction:
random-direction calibration cannot see the fitted-direction geometry
where skills actually live. v2 needs behavioral/Fisher whitening; the
belief layer's value shows in its observables, not yet in control.
