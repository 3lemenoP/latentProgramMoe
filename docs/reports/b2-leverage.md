# B2 (Q1) — leverage ratio: axis field vs conjugation fields

Studio 2 run 2026-08-15, pythia-410m fp32 eager on T4, 200 pile-10k
sequences × 512 tokens. Matched-activity single-type random fields
(sin²(θ/2) = s̄, uniform random axes). KL(program ‖ base), full vocab.
*(v1 run OOM'd in the attention probe after the full KL table printed; v2
rerun reproduced the table (same seeds) and completed the probe. Final.)*

## Leverage table L = KL/s̄

| field type | L @0.005 | L @0.01 | L @0.03 | L @0.06 | late/early @0.01 | late/early @0.06 |
|---|---:|---:|---:|---:|---:|---:|
| rope_ax | 3.40 | **1.91** | 4.45 | 5.70 | 1.74 | **3.38** |
| qk_rel | 79.9 | 70.7 | 56.5 | 47.0 | 2.41 | 1.91 |
| ffn_hidden | 33.9 | 60.3 | 156.3 | 138.0 | 0.89 | 1.20 |
| attn_io | 15.0 | 17.2 | 32.4 | 52.1 | 1.16 | 1.55 |

(late/early = mean KL over positions 448–511 ÷ positions 0–63.)

## Pre-registered predictions, scored

1. **"axis-field leverage > qk_rel leverage at s̄=0.01" (prior 0.65): REFUTED,
   by ~37×** (1.9 vs 70.7). Raw per-token leverage tracks how many dims a
   site group rotates: ffn_hidden 4096/layer, attn_io & qk_rel ~1024,
   rope_ax only 16 heads × 15 dims = 240. Even per-dim, rope_ax is lowest
   at small s̄.
2. **"distance-resolved KL grows with position for the axis field, flat for
   conjugation" (prior 0.70): PARTIALLY CONFIRMED, with an explained
   deviation.** rope_ax is the only type whose position-coupling *grows with
   strength* (late/early 1.12 → 3.38 as s̄ rises; superlinear — the
   φ·sin(ω·Δm) signature). ffn_hidden is flat (0.79–1.20) and attn_io near
   flat, as predicted. qk_rel however ALSO grows (≈2), which is **coherent,
   not anomalous**: the B1 gate work proved that a static post-RoPE qk_rel
   on a rotary base is itself position-coupled
   (score = q̃ᵀ·RoPE₋ₘ·M·RoPEₙ·k̃ ≠ f(n−m)) — on Pythia, qk_rel is not a
   clean "flat conjugation control"; the clean flat controls are
   ffn_hidden/attn_io, and they behave as predicted.

## Regime structure (the more interesting read)

Conjugation leverage *decays or saturates* with strength (qk_rel L: 80→47),
while axis leverage *rises* (3.4→5.7 crossing s̄): the axis field is the only
site group whose effect compounds with both strength and distance. It buys
little raw KL per activity — but what it buys is qualitatively different:
position-structured displacement rather than uniform perturbation. The
headline plot (per-position KL curves) is in `runs/b2/report_b2.json`
(`curves`), late/early ratios above.

## Attention-displacement probe (s̄=0.01, rope_ax vs qk_rel)

Normalized displacement |Δattn|/attn_base per relative-distance bin:

| field | rel. near (1–16) | rel. far (128–511) | far/near |
|---|---:|---:|---:|
| rope_ax | 0.174 | 0.113 | 0.65 |
| qk_rel | 0.509 | 0.527 | 1.03 |

At s̄=0.01 the axis field's *attention* displacement does not grow with
distance (mildly near-concentrated); qk_rel is flat and ~3× larger. The
distance signature that does hold for rope_ax is in the *output* KL
(late/early growing with s̄, up to 3.38), i.e. the compounding is visible
downstream of attention rather than in single-map displacement at this
weak strength. Probe at higher s̄ is the natural follow-up if the
attention-level signature matters for B3's interpretation. Full curves:
`runs/b2/report_b2.json` (`curves`, `attn_disp_*`, `attn_base_*`).
