# Phase 4 v2 — R1–R3 reanalyses + M-metric gates (G1/G2)

Runs 2026-08-16 (local + phase4 studio, cached v1 artifacts). Predictions
scored inline against `phase4-v2-spec.md`.

## R1 — behavioral rate–distortion (registered: elbow at k≈4, prior 0.60): FAILED narrowly

symKL covering radius: 2.08 (k=2) → 1.556 (k=3) → 1.541 (k=4) → 1.321
(k=5) → 1.321 (k=6). Largest relative drop at **k=3** (25.2%); k=4 adds
0.95%. The v1 "no elbow" verdict is replaced: the behavioral measure wants
**~3 specialists**, one fewer than C-2's four families (the light/marker
family folds into a neighbor).

## R2 — revisit-split recovery (registered: no revisit advantage, prior 0.80): CONFIRMED

| method | first-visit mean | revisit mean | advantage |
|---|---:|---:|---:|
| v1 agent | 53.2 | 53.1 | **+0.1** |
| SGD tracker | 46.2 | 48.0 | −1.8 |
| reset-on-spike | 44.4 | 40.0 | +4.4 |

The compromise posterior erases regime memory exactly as diagnosed. The
IMM benchmark is set: its entire edge must appear in the revisit column
(P-2 target < 10 episodes vs ~41–53).

## R3 — D-3 scoring in behavioral work shares (re-registered, prior 0.75): CONFIRMED

Converged v1-agent commitment maps vs direct-fit skill profiles, leverage ×
activity shares: cosines prepend 0.910, reverse 0.988, a_then_b 0.981,
b_then_a 0.997, c_then_b 0.996, e_then_d 0.988 — **mean 0.977, all > 0.8**.
The cell-identity check passes: even the underperforming v1 agent's
components acquire the same actuator profile as the direct-fit skills.

## G1 (T22 LOO gate, learned metric beats baselines): FAILED

Leave-one-out Spearman vs symKL: **M −0.213** vs raw-geodesic 0.438 vs
random-whitened 0.527. The learned diagonal-in-PCA metric overfits 351
pairs from 27 fields and inverts out of sample (the same construction
passes at 0.9+ on synthetic libraries with a planted low-dim live subspace
— a sample-size/structure failure, not an implementation one).

## G2 (C-1 rematch under M, priors 0.70/0.80): FAILED

Mean ARI vs behavioral clustering: **M −0.062** vs raw 0.187 vs whitened
0.097. Both registered predictions fail.

## Reading — Constraint 0, sharpened

v1 showed manifold-native proxies fail; v2 shows a *learned* geometric
proxy fails too at this library size. Behavioral structure over 27 fields
supports reliable *clustering* (C-2) but not a generalizing *quadratic
form*. Constraint 0's operational content is therefore stronger than
written: at current scale, decisions must use direct behavioral
measurement — there is no shortcut through any static metric, learned or
not. The IMM (whose regime decisions are measured probe losses) is
unaffected by the metric failure; the metric survives only as diagnostics.
`metric.json` ships with FAILED gate flags and must not be consumed
downstream (v2.1 revisit gated on library growth ≥ ×4).
