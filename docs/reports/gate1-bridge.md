# Gate-1 §2 — the quarter-rotary bridge, scored

L4 run 2026-08-16. NeoX toy twin at rotary_pct = 0.25 (rotary_ndims = 6,
two 3-blocks per head), four-arm offset ladder, 3000-step protocol.

## Exact per rung per arm (rope_ax activity in parens)

| rung | conj_only | conj+axis | axis_only | lr×10 |
|---|---:|---:|---:|---:|
| k=0 | 0.855 | 0.880 (0.07) | 0.285 (—) | 0.880 (0.15) |
| k=1 ⟨A⟩ | 0.725 | **0.775** (0.16) | 0.000 | 0.765 (0.28) |
| k=1 ⟨B⟩ | 0.645 | **0.745** (0.16) | 0.000 | 0.730 (0.28) |
| k=2 | 0.385 | **0.600** (0.19) | 0.000 | 0.570 (0.48) |

Best-arm lifts on k ≥ 1: **+5.0 / +10.0 / +21.5** — growing with offset,
replicating the full-rotary pattern at reduced magnitude.

## Registered predictions

- **Gate passes (prior 0.60): CONFIRMED.** The axis-field claims survive at
  quarter coverage — they are NOT scoped to full-rotary bases.
- **"lr×10 is necessary at partial coverage" (prior 0.65): REFUTED,
  instructively.** The naive joint fit beats lr×10 on every rung (5.0 vs
  4.0, 10.0 vs 8.5, 21.5 vs 18.5). At quarter coverage the naive fit
  already routes into rope_ax (0.16–0.19 activity at k ≥ 1 — a *higher*
  share than full-rotary naive fits took), and forcing more (0.28–0.48)
  slightly overshoots. The lazy-routing hazard scales with how substitutable
  the axis dims are: with only two 3-blocks per head, each is load-bearing
  and gradients find them unaided.

## New scoping fact: sufficiency shrinks, necessity survives

axis_only collapses at quarter coverage (0.285 on plain reverse vs 0.850 at
full rotary; 0.000 on every offset rung): a quarter of head dims cannot
carry re-indexing alone — content actuators must participate. The axis
field's *lift* (necessity) survives partial coverage; its *sufficiency*
does not. For real bases (partial rotary is the norm), the axis field is an
amplifier of conjugation programs, not a standalone actuator.
