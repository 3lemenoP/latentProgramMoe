# B3 (Q2) — the axis field cracks the offset ceiling

Studio 2 run 2026-08-15. Fresh 4-layer GPT-NeoX toy, **full rotary**
(rotary_pct=1.0, d_head=24 ⇒ rope_ax [4, 4, 8]), trained from scratch on
copy/prepend/reverse/append. A2's ladder fitted per rung under two arms
(v1): conjugation-only (rope_ax frozen at identity) vs conjugation+axis.
3000-step strong-fit protocol throughout.

## Result: PASS — and the lift grows with offset

| rung | conj-only | conj+axis | lift (pts) | rope_ax activity (joint fit) |
|---|---:|---:|---:|---:|
| k=0 reverse | 0.875 | 0.930 | +5.5 | 0.130 |
| k=1 pad ⟨A⟩ | 0.745 | 0.865 | **+12.0** | 0.165 |
| k=1 pad ⟨B⟩ (control) | 0.770 | 0.885 | **+11.5** | 0.195 |
| k=2 pad ⟨A⟩ | 0.535 | 0.755 | **+22.0** | 0.248 |

Gate (best arm ≥ 5 points on every k ≥ 1 rung): **PASS**, by 2–4×.

## Pre-registered predictions, scored

- **B3 "axis lifts k≥1 ceilings by ≥5 points" (prior 0.55): CONFIRMED** —
  +12 / +11.5 / +22, and monotone in offset: the treatment is largest
  exactly where the disease is worst.
- **Refined prediction (steering follow-up): CONFIRMED** — fitted rope_ax
  activity comes out at 0.13–0.25, ~3–5× the typical conjugation-site
  activity, and **monotone in k**. The task demands the positional actuator
  in proportion to the re-indexing displacement. The lazy-routing false
  negative did not materialize on this base — note the economics differ
  from Pythia: with full rotary the axis field touches all 24 head dims, so
  the B2 leverage asymmetry (measured at rotary_pct=0.25, 240 dims) does
  not bind here. The axis-only and lr×10 arms (v2, in flight) complete the
  controls.
- Consistency detail: even k=0 lifts (+5.5) — plain reversal is itself a
  position re-indexing, so this is coherence, not leakage. Failure profiles
  shrink specifically in the reversal-tail class (k=2: 28→20 reversal,
  8→3 length), i.e. the axis field fixes the *predicted* error mode.
- Pad-identity control: equal lift under ⟨A⟩ vs ⟨B⟩ pads (12.0 vs 11.5) —
  offset, not marker identity, throughout.

## The closed loop

A2 (Studio 1): constant conjugation fields fail at position re-indexing,
monotonically in displacement — the law. B3 (Studio 2): giving the program
access to the rotary generators (Ω → RΩRᵀ) repairs exactly that failure,
in proportion to the displacement, while the fields *choose* to use the
axis sites more the more re-indexing the task needs.

**This is the initial idea's first positive, mechanism-specific result:
the latent program as a parameter of the model's rotational positional
machinery does something constant conjugation provably cannot.**

## v2 four-arm controls (final)

Exact per rung per arm (rope_ax fitted activity in parens):

| rung | conj_only | conj+axis | axis_only | conj+axis, rope_ax lr×10 |
|---|---:|---:|---:|---:|
| k=0 | 0.875 (0) | 0.930 (0.13) | 0.850 (0.35) | **0.955** (0.40) |
| k=1 ⟨A⟩ | 0.745 (0) | 0.865 (0.16) | 0.195 (0.45) | **0.925** (0.44) |
| k=1 ⟨B⟩ | 0.770 (0) | 0.885 (0.19) | 0.000 (0.40) | **0.930** (0.44) |
| k=2 | 0.535 (0) | 0.755 (0.25) | 0.000 (0.47) | **0.880** (0.52) |

1. **The lazy-routing worry was right in degree.** The naive joint fit
   under-uses rope_ax: boosting its lr ×10 doubles the fitted axis activity
   (0.25 → 0.52 at k=2) and buys another **+12.5 points** on top of the
   naive arm — k=2 lands at 0.880, statistically at the k=0 conj-only level
   (0.875). Best-arm lifts vs conj-only: **+18 / +16 / +34.5** on the k≥1
   rungs. The offset ceiling is effectively erased, not just lifted.
2. **Division of labor is clean.** axis_only nearly solves plain reversal
   (0.850 — a pure positional program almost executes reverse by itself,
   with zero content-site activity) but collapses whenever a *marker token*
   must be emitted (k=1 ⟨B⟩: 0.0 with 59 first-token errors; k=2: length
   errors) — the axis field re-indexes positions but cannot insert content.
   Conjugation inserts content but cannot re-index (A2). The composite is
   what works: **conjugation = content actuator, axis = position actuator.**
3. reversal-tail failures under lr×10 drop to 6–8 per 100 at every rung
   (conj-only: 14–28) — the axis field fixes the predicted error class.

Fitted ffn_hidden activity also drops in the axis arms at every rung (e.g.
k2: 0.212 → 0.190) — the conjugation sites *offload* the positional work
they were imitating.

## Next (proposed, not run)

B4 stretch (natural-language skill on Pythia with/without axis sites) and
the pre-registered low-frequency-only axis probe from B2.
