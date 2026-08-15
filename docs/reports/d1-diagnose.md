# E3′ D1 — diagnostic battery

Studio run 2026-08-15. Fields `runs/e3/z_a.pt` (prepend, exact 1.000) and
`runs/e3/z_b.pt` (reverse, exact 0.970).

## Support overlap

- activity cosine ⟨s_A, s_B⟩: **0.3924**
- OVL: **0.4798**
- mean activity A / B: 0.0248 / 0.0302

## Geometric commutator

- κ mean: **0.003066**   max: **0.335275**
- Layer-2 tail (κ max 0.335) is the only material hotspot.

| site group | mean s_A | mean s_B |
|---|---:|---:|
| attn_io | 0.00366 | 0.00560 |
| ffn_hidden | 0.03287 | 0.04428 |
| mlp_io | 0.00270 | 0.00361 |
| qk_rel | 0.03574 | 0.02496 |

## Behavioral commutator

- mean KL compose(b,a) ‖ compose(a,b): **0.004942**

**Reading:** not the pure disjoint-support artifact (cosine 0.39, OVL 0.48).
Both skills are still *tiny* rotations (mean s ≈ 0.03), so commutators are
second-order on average (κ mean 0.003) and **behaviorally silent** (KL 0.005).
A few layer-2 sites have a live geometric commutator that does not show up
in outputs. Branch: **overlapping but silent** — D2 capacity check next,
then D3/D4/D5; D4 is still motivated (activity is λ_reg-shrunk).
