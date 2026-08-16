# Gate 1 — final scorecard (bridge, G1-A, G1-B, G1-C)

L4 runs 2026-08-16, single studio per directive. Companions:
`gate1-bridge.md`, `gate1-g1a-controls.md`. Every registered prediction
scored inline. Cards for all four G1-B skills generated; example card
checked in at `registry/hedge/0.1.0/`.

## G1-B — skill breadth (recovery vs BEST teacher per skill)

The G1-A memorization finding generalized immediately: **3 of 4
budget-matched (2000-step) teachers overfit-inverted** (formal −0.58,
hedge −0.26, medical −0.94 eval gap; only synthetic jsonish +1.50). Best
teachers found by budget search: jsonish@2000, hedge@800, formal@800,
medical@400.

| skill | teacher gap | recovery | 95% CI | neutral ppl b/t/p | certs |
|---|---:|---:|---|---|---|
| jsonish (format) | +1.50 | **103.9%** | [103.0, 104.8] | 29.1 / **218.7** / 155.8 | ✓ |
| hedge | +0.65 | 88.7% | [84.9, 92.4] | 29.1 / 24.5 / 25.1 | ✓ |
| formal | +0.41 | 83.2% | [76.8, 90.7] | 29.1 / 23.3 / 23.9 | ✓ |
| medical | +0.21 | **3.9%** | [−17.8, 20.2] | 29.1 / 31.7 / 37.1 | ✗ |

- **jsonish is the cleanest beats-teacher datapoint of the campaign**: its
  teacher did NOT overfit (+1.50 gap), and the program still exceeds it
  (CI floor 103.0%). Note the gentleness asymmetry: the format teacher
  devastates neutral text (ppl 29 → 219) while the program does a third
  of that damage.
- **medical is the first outright distillation failure**: recovery ≈ 0
  (CI spans 0), collateral without benefit, and its spectrum certificate
  tripped (min raw norm 0.463 < the 0.5 tamper band — fit drift, not
  tampering; the band needs calibration, but the card correctly refused to
  certify an unhealthy program). Weakest teacher (+0.21) and the only
  genuinely out-of-distribution corpus (PubMed abstracts).
- Registered: median recovery ≥90% (prior 0.60): **FAILED** — median 85.9%
  (88.7% excluding the medical failure; near miss). All four gentler than
  teacher (0.70): **FAILED** as registered — but the comparison is
  confounded for formal/hedge, whose wikitext-derived task corpora make
  their teachers *improve* neutral wikitext ppl. Format skill highest
  rope_ax share (0.55): **NOT SCOREABLE as designed** — the breadth arms
  froze rope_ax (conj_only), so shares are 0 by construction; the jsonish
  conj+axis arm measures rope_ax at 1.9% work share (content-dominant).
  Protocol error recorded, not a prediction outcome.
- Card tests: **TC-5 PASS** (french qk_rel work share > ffn_hidden — the
  result the raw activity table hid). TC-4 deferred (needs a French v2).

## G1-C — pruning (french program)

| τ | kept | task CE | neutral ppl | 8-bit sparse |
|---:|---:|---:|---:|---:|
| 0 | 96.8% | 2.796 | 36.8 | 391 KB |
| 1e-4 | 95.1% | 2.797 | 36.9 | 384 KB |
| 3e-4 | 90.4% | 2.794 | 36.8 | 365 KB |
| 1e-3 | 72.4% | **2.790** | **35.5** | 292 KB |

- Registered ≥90% prunable at ≤0.01 nat (0.60): **FAILED** — the program
  is dense (activity spread across sites), only ~28% prunable at the
  tested τ. <100 KB at 8-bit (0.55): **FAILED** (292 KB best).
- Two silver linings: pruning at τ=1e-3 *improves* both task CE and
  neutral ppl (pruning as regularizer — worth a deeper τ sweep), and the
  absolute size story survives without sparsity: the dense fp16 artifact
  is 462 KB vs ~10 MB for the LoRA teacher — 20× smaller as-is.

## Combined Gate-1 reading

- **Bridge: PASS** — axis-field claims hold at quarter rotary; lr×10
  unnecessary there; axis is an amplifier, not a standalone actuator.
- **G1-A: headline upgraded** — programs beat their teachers (120.6%, CI
  excluding 100%), and at matched budget the LoRA memorizes while the
  manifold regularizes (gen-gap 0.42 vs 1.19/3.67).
- **G1-B: the honest breadth picture** — 2 strong (jsonish 104%, hedge
  89%), 1 decent (formal 83%), 1 failure (medical 4%). The manifold's
  regularization is also its ceiling: skills whose teachers barely beat
  the base (weak learnable signal) do not survive distillation.
- **G1-C: sparsity predictions failed; absolute size holds** (0.46 MB
  dense, 20× under the teacher).

**Gate-2 posture:** the steering's own rule says Gate 2 proceeds
regardless (its recovery gate is 85% — which 3 of 4 breadth skills meet
or approach, and french exceeds). Watch-items for the Qwen run: pick
skills with demonstrable teacher gaps (the medical lesson), calibrate the
certificate norm band, and run axis-enabled arms if the format-skill axis
question matters.
