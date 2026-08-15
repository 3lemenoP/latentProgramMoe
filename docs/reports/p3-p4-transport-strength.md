# P3 (D3 transport) + P4 (D5 upward scan) — strong-field runs

Studio runs 2026-08-15, on `runs/e3strong` (3000-step fits for D3).

## P3 — D3 relative-increment transport: CONTROLS FAIL (flagship pending)

**Amended per `steering-workstreams-a-b.md` §0:** this run used c =
append-⟨B⟩, which puts the marker first in c_then_b and structurally
mismatches a_then_b — these are the transport *controls*, not the flagship.
The token-keying probe and the b⊗a⊗b\* adjoint re-keying candidate (A1)
never ran. Do not read this section as "transport failed."

Δ_{b|a} = z_ab ⊗ z_a\* transported onto z_c (append, fresh fit 0.995 exact),
target pipeline c_then_b (oracle fit: 0.815 exact — the ~0.82 plateau again;
see P2's z_ba note).

| candidate | exact on c_then_b | d_geo to z_cb |
|---|---:|---:|
| left transport Δ_{b|a} ⊗ z_c | **0.000** | 0.315 |
| adjoint g Δ g\* ⊗ z_c | **0.000** | 0.317 |

Gate was ≥ 0.5 exact for either. Both candidates are ~2× farther from the
oracle than the atoms are from identity (0.14). Transport does not survive
the gauge — consistent with P1/P2's product-rule falsification: increments
are not absolute, and the z_a-conjugation does not repair them.

Side-fact: the "second pipeline fits worse" ceiling recurs — z_cb plateaus
at 0.815 like z_ba (0.82), while z_ab reaches 0.97. Both underperformers
put the *reversed payload late* behind a fixed first token (⟨A⟩/⟨B⟩ at
position 1); z_ab emits the reversed payload immediately. A single constant
field appears to struggle with reverse-after-marker structure specifically.

## P4 — D5 upward strength scan (q_pow on strong fits)

| λ | prepend | reverse | exact ab | exact ba |
|---:|---:|---:|---:|---:|
| 1.00 | 1.000 | 0.980 | 0.0 | 0.0 |
| 1.25 | 0.995 | 0.965 | 0.0 | 0.0 |
| 1.50 | 0.985 | 0.895 | 0.0 | 0.0 |
| 1.75 | 0.865 | 0.630 | 0.0 | 0.0 |

Composition is 0.0 exact at every strength; retention decays smoothly past
λ = 1.5. There is no intermediate-strength sweet spot — the companion doc's
"if the algebra is semantically valid anywhere, quality peaks at
intermediate strength" prediction fails. Consistent with the falsifier.

## Position in the tree

D2 (one-sided pass) → falsifier holds (P1+P2) → D3 *controls* fail,
flagship (A1 token re-keying) pending → D6 running under A3 evaluation
controls (5-atom library a–e incl. new rotl/swap2; all 20 ordered pipelines
oracle-fitted; held-out {d_then_b, c_then_a, e_then_d, a_then_e}).
