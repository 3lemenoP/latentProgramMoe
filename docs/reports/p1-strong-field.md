# P1 — strong-field refits (handoff §4 P1)

Studio run 2026-08-15. Refits of z_a, z_b, z_ab, z_ba at 3000 steps (2× the
weak fits), cosine lr decay, same frozen base (`runs/e3/base.pt` copied to
`runs/e3strong/`). Note: the E3 fit path never had an explicit λ_reg term —
the handoff's "λ_reg = 0" condition was already true at 1500 steps; what P1
actually varies is optimization budget. The weak-field regime therefore came
from tiny init (σ=1e-3) + implicit regularization, not an explicit pull.

## Atoms

z_a prepend **1.000** exact; z_b reverse **0.980** (weak: 0.970, loss
plateau 0.037 vs 0.044).

## D1 battery, weak vs strong

| metric | weak (1500) | strong (3000) |
|---|---:|---:|
| mean activity A / B | 0.0248 / 0.0302 | 0.0573 / 0.0544 |
| activity cosine | 0.392 | 0.440 |
| OVL | 0.480 | 0.468 |
| κ mean | 0.00307 | 0.01267 |
| κ ceiling (θ_A·θ_B) | 0.0030 | 0.0129 |
| κ / ceiling | ~1.0 | 0.98 |
| order-KL | 0.0049 | 0.0151 |
| anchor KL compose‖base | — | 0.542 |
| anchor KL z_a‖z_b | — | 0.813 |

Activity doubles, κ quadruples, and the commutator **stays pinned to its
kinematic ceiling** — axes remain near-maximally misaligned; amplitude is
still the only suppressor. Order-KL now has denominators: 0.015 against a
0.54 compose-vs-base displacement (~3%). Site profile unchanged: skills live
in `ffn_hidden` (0.077/0.082) and `qk_rel` (0.084/0.042); `attn_io`/`mlp_io`
still idle (≤0.007) — the §7 interface-field note survives at 2× strength.

## Composition eval (strong atoms)

compose(zb, za) and compose(za, zb): **0.0 exact** on both pipelines at
λ ∈ {1, 1.25, 1.5} (q_pow-amplified). Token acc 0.18–0.29. Strength does
not rescue zero-shot Hamilton composition.

## D2 oracles, strong

| | weak | strong |
|---|---:|---:|
| z_ab exact | 0.955 | **0.970** |
| z_ba exact | 0.775 | **0.820** |

Same one-sided miss (gate ≥0.9 fails on z_ba only); z_ba failure signature
unchanged: 0 first-token / 18 reversal / 18 other over 200 probes.

## Distance table + behavioral column (strong)

| candidate | d_geo to z_ab | KL(cand ‖ z_ab) |
|---|---:|---:|
| hamilton(b,a) | 0.199 | 1.060 |
| z_a | 0.232 | 1.437 |
| z_b | **0.126** | **0.525** |
| identity | 0.167 | 1.013 |

Anchors: d(id, z_a)=0.142, d(id, z_b)=0.137, d(id, hamilton)=0.197.

**The §4 P1 falsifier's first condition holds at strength:** the oracle
remains closer to identity than to the Hamilton prediction, and the new
behavioral column removes the stabilizer-degeneracy confound — Hamilton's
composite is behaviorally no better than the base model (KL 1.06 vs 1.01),
while z_b alone is twice as close. "z_ab = reverse plus a tweak" is
confirmed geometrically AND behaviorally. The falsifier completes only if
P2's warm-started fits converge easily from the Hamilton init — P2 running.
