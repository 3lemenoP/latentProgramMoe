# P0 — exponentiation probe (handoff §4 P0)

Studio run 2026-08-15, no training. Weak fields `runs/e3/z_a.pt` (prepend,
1.000), `runs/e3/z_b.pt` (reverse, 0.970), amplified via `q_pow`.

| λ | prepend | reverse | retained | κ mean | κ max | order-KL |
|---:|---:|---:|:--:|---:|---:|---:|
| 1.00 | 1.000 | 0.970 | ✓ | 0.00307 | 0.335 | 0.00615 |
| 1.25 | 1.000 | 0.965 | ✓ | 0.00686 | 0.621 | 0.01643 |
| 1.50 | 1.000 | 0.910 | ✓ | 0.01278 | 0.880 | 0.03430 |
| 2.00 | 0.950 | 0.445 | ✗ | 0.03154 | 0.955 | 0.08573 |

Composition exact stays 0.000 at every λ (both orders); compose(b,a) token
acc drifts 0.313 → 0.175 as λ rises, compose(a,b) 0.229 → 0.278.

**Reading (branch 1 of handoff §4 P0, within the retained band λ ≤ 1.5):**
κ mean tracks the ~λ⁴ small-angle schedule (×4.2 at λ=1.5 vs ×5.1 predicted;
saturating at λ=2 as the layer-2 big-angle sites clip), and the order-KL
leaves zero on the same schedule (0.0061 → 0.0343, ×5.6). The commutator is
**not stabilizer-bound — it was amplitude-starved**. Strength story
confirmed; P1 proceeds with confidence. Amplification alone does not compose
the pipeline (exact 0.0 everywhere), so strong-field *refits*, not amplified
weak fits, are the decisive test. Retention breaks between λ=1.5 and 2
(reverse first), bounding the usable free-exponentiation range.

## z_ba failure decode (handoff §3.2e / §2.5 puzzle)

200 probes on the weak `runs/e3/z_ba.pt` (0.775 exact): **first-token 0,
reversal 22, length 0, other 23**. The "other" dumps are also deep-tail
scrambles / substitutions in the reversed payload (long inputs, repeated
tokens). ⟨A⟩ lands correctly at position 1 in every failure — the
position-conditional-program hypothesis (§2.5) is dead. The miss is
reversal-tail structure, consistent with optimization difficulty on
reverse-type fits, not a first-token capacity wall.
