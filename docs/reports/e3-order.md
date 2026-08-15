# E3 — order sensitivity

Studio run 2026-08-15. Tiny from-scratch GPT-2 (`n_embd=96`, 4 layers).
Atomic skills fitted to 100% exact. Composition is zero-shot.

| condition | exact | token |
|---|---:|---:|
| z_a on prepend | 1.000 | 1.000 |
| z_b on reverse | 1.000 | 1.000 |
| compose(b,a) on a_then_b | 0.000 | 0.170 |
| compose(a,b) on b_then_a | 0.000 | 0.263 |
| compose(b,a) on b_then_a (swapped) | 0.000 | 0.265 |
| compose(a,b) on a_then_b (swapped) | 0.000 | 0.174 |
| abelian ta on prepend | 1.000 | 1.000 |
| abelian tb on reverse | 1.000 | 1.000 |
| abelian compose(b,a) on a_then_b | 0.000 | 0.231 |
| abelian compose(a,b) on b_then_a | 0.000 | 0.267 |

Non-abelian − abelian on the intended pairings: **−6.1** and **−0.4**
points (spec wants **> 20**). Both compose orders score ~0.17 on
`a_then_b` and ~0.26 on `b_then_a` — behaviorally order-blind.
Swapped `compose(b,a)` is *better* than the intended pairing.

**Decision: E3 does not pass.** Atomic programs work; the Hamilton
product does not implement ordered skill composition.
