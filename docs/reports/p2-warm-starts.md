# P2 — z_ba disambiguation + falsifier completion (handoff §4 P2)

Studio run 2026-08-15, on the strong-field fits (`runs/e3strong`, 3000-step
budget per fit, eval every 250 steps on 50 decodes).

## z_ab falsifier leg (handoff §4 P1)

| init | steps to 0.9 exact | final exact |
|---|---:|---:|
| identity | **250** | 0.960 |
| hamilton(b,a) | 500 | 0.940 |

The algebraic guess gives **no convergence head start** — the identity
control is faster. With P1's distance table (oracle nearer identity than
Hamilton; KL(hamilton‖z_ab) 1.06 ≈ KL(id‖z_ab) 1.01), the sharpened
falsifier of handoff §4 P1 now holds in full:

**The product rule is wrong in the current gauge, not under-driven.
Composition claims move entirely to D3/D6.**

## z_ba warm starts

| init | steps to 0.9 | plateau (exact) |
|---|---:|---:|
| identity | never | 0.76 |
| hamilton(a,b) | never | 0.82–0.84 |
| z_b | never | 0.82 |

The plateau is init-independent — the miss is not a fine-tune-distance
problem. Warm starts from the algebraic guess and from the nearest atom buy
the early steps (0.48 / 0.62 exact at step 250 vs 0.06 from identity) but
converge to the same ceiling.

## Overfit-32 capacity check

Final loss 0.0287, teacher-forced sequence exact **1.000** — the field
parameterization can memorize 32 fixed b_then_a examples. The handoff's
"capacity is real" terminal is **not** reached in the memorization sense;
the z_ba gap is distribution-level: a single constant field cannot (or
optimization cannot find one that can) represent full-distribution
reverse-then-prepend beyond ~0.82–0.84 exact, regardless of init.

## Decision tree position

```
P0: order-KL wakes with λ  ──► strength story confirmed
P1: strong fields, compose still 0.0; oracle nearer id than Hamilton
P2: Hamilton init no head start  ──► FALSIFIER HOLDS
                                     product rule wrong (current gauge)
                                     ──► all weight on D3 transport / D6 operator
```
