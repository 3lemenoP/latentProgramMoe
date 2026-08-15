# Workstream A — final: A1 re-keying, A2 ladder, A3-controlled D6

Studio 1 runs 2026-08-15, all on `runs/e3strong` (strong-fit protocol).
Scores the pre-registered predictions of `steering-workstreams-a-b.md`.

## A1 — D3 flagship: token re-keying under conjugation

Atom a′ = prepend-⟨B⟩ fits at **1.000** (prepend-type easy, as predicted);
oracle z_fb (f_then_b = reverse(x)+⟨B⟩, payload-first) at **0.975** —
payload-first escapes the 0.82 class again, on a fourth pipeline.

Marker-slot distribution (teacher-forced, 200 probes):

| candidate | exact | token | P(⟨A⟩) | P(⟨B⟩) | P(other) |
|---|---:|---:|---:|---:|---:|
| (3) token-keying Δ_A⊗z_b (≡ z_ab) | 0.0 | 0.882 | **0.961** | 0.003 | 0.036 |
| (4) transported g Δ_A g\*⊗z_b | 0.0 | 0.876 | 0.740 | 0.031 | 0.228 |
| oracle z_fb | 0.975 | 0.996 | 0.001 | 0.940 | 0.060 |
| z_b alone | 0.0 | 0.778 | 0.000 | 0.004 | 0.996 |

- **Prediction A1(3) (prior 0.85): CONFIRMED.** The probe is near-perfect
  except the marker slot, where it emits ⟨A⟩ at P=0.961 — z_ab's tweak is
  token-keyed ("emit ⟨A⟩ last"), not token-generic.
- **Prediction A1(4) (prior 0.30): FAILED, with a directional residue.**
  Hard pass no (0.0 exact). Soft pass no: P(⟨B⟩) moves 11.7× (clears the
  ≥10× relative gate) but only reaches 0.031 (< 0.1 absolute). The adjoint
  *de-keys* ⟨A⟩ (0.96 → 0.74) but the displaced mass scatters to "other"
  (0.23) instead of re-addressing to ⟨B⟩. Conjugation perturbs the keying
  mechanism; it does not perform token substitution inside it.

## A2 — offset-reversal ladder: the ceiling is a law

| rung | behavior | exact | failure profile |
|---|---|---:|---|
| k=0 | reverse | **0.980** | — |
| k=1, pad ⟨A⟩ | b_then_a | **0.820** | 0 first-token, reversal-tail |
| k=1, pad ⟨B⟩ (control) | c_then_b | **0.815** | 0 first-token, reversal-tail |
| k=2, pad ⟨A⟩ | rev_pad2 | **0.445** | 0 first-token, 60 reversal / 13 length / 38 other |

- **Prediction A2 (prior 0.75): CONFIRMED.** Strictly monotone in k, steeply
  (0.98 → 0.82 → 0.445); pad-identity control null (Δ = 0.005 — offset, not
  marker identity); pads themselves always emitted (0 first-token errors at
  every k), errors concentrate in the reversal tail and grow with k.
- Corroborating D6 sightings on displacement-graded variants: cyclic-shift
  reversal (b_then_d) 0.730, first-two-transposed (b_then_e) 0.780,
  unshifted-reversal pipelines all ≥ 0.925.

**Law: a constant per-sequence field cannot re-index positional attention;
the achievable exact rate degrades monotonically with the displacement of
the reversal.** Workstream B3 tests whether the axis field is the cure.

## A3-controlled D6 — final verdict: meaningful both-fail

All 20 pipeline oracles fit 0.73–1.0 (substrate expressivity headline; the
low tail is exactly the A2 displacement family). Operators on 16 triples,
held-out {d_then_b, c_then_a, e_then_d, a_then_e}:

| model | held-out exact | note |
|---|---:|---|
| hamilton | 0.005 | signal only on the commuting pair c_then_a (0.68 token) |
| gauged product | 0.000 | pre-registered to fail ✓ |
| per-site MLP (+group emb) | 0.000 | **learning curve flat: {8: 0.0, 12: 0.0, 16: 0.0}** |
| flattened MLP control | 0.000 | train chordal loss → 0.000000: pure memorization ✓ |

The per-site operator is consistently the geometrically nearest candidate
(d_geo 0.15–0.26 vs hamilton 0.24–0.39) while behaviorally dead — geometric
proximity to a program is again no evidence of behavioral proximity.
**Verdict (tree-final): both fail with a flat learning curve — retire
zero-shot program-space composition on this substrate; substrate claims
(expressivity, gentleness, control) stand.**

- Prediction A3 (prior 0.80): constrained fails ✓; per-site > flattened
  held out only geometrically (both 0.0 behaviorally) — partial.

## Where this leaves the tree

Program-space composition: retired (D2 falsifier + D3 controls + A1
flagship + D6 both-fail, each independent). The one live mechanism-level
positive direction is Workstream B: the axis field as the positional
actuator that constant conjugation fields provably lack (A2 law ↔ B2
regime structure ↔ B3 causal test, running).
