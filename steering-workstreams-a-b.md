# Steering — parallel workstreams A (toy/D6) + B (Pythia axis field)

**Date: 2026-08-15. Extends `handoff-d1-d2-status.md`; where this conflicts with any earlier doc, this wins. Two GPU studios run in parallel: Studio 1 = Workstream A, Studio 2 = Workstream B. No cross-studio dependencies.**

---

## 0. Position after P0–P4

Closed: product rule (falsifier complete — Hamilton init strictly slower than identity), strength axis (no sweet spot at any λ), D3 *controls* (both 0.000, as pre-registered). **Not closed, despite the P3 report's tree line: D3 transport.** The run used c = append-⟨B⟩ from the original spec, which puts the marker *first* in c_then_b and structurally mismatches a_then_b; candidates (3)/(4) from the amended table never ran, and the D6 atom library (a–e) contains no prepend-⟨B⟩ atom, so they aren't covered there either. Correct tree state: *D3 controls failed; D3 flagship pending (A1 below).*

New confirmed finding: the **offset-reversal ceiling** — pipelines placing reversed payload behind a fixed first token plateau at ~0.82 (z_ba 0.82–0.84, z_cb 0.815) while payload-first pipelines reach 0.97+ (z_ab 0.970, z_b 0.980). Three sightings. Working hypothesis: a constant per-sequence field cannot re-index positional attention (reversal shifted by the marker offset). This is the empirical bridge to Workstream B: position re-indexing is precisely what the axis field modulates and what static conjugation fields cannot.

---

## Workstream A (Studio 1)

### A1 — D3 flagship: can conjugation re-key a token inside a mechanism?

Setup (all at the 3000-step strong-fit protocol):
1. Fit new atom **a′ = prepend ⟨B⟩** (expect ~1.000; prepend-type fits are the easy ones).
2. Fit oracle **z_a′b** on a′_then_b = reverse(x)+⟨B⟩ (payload-first ⇒ expect ~0.97, unlike the 0.82-class).
3. Evaluate two zero-shot candidates on a′_then_b:
   - **(3) token-keying probe:** Δ_A ⊗ z_b, where Δ_A = z_ab ⊗ z_b\*. Algebraically this IS z_ab — the probe asks whether z_ab's tweak is token-generic ("emit the prepended token last") or keyed to ⟨A⟩. Prediction: near-perfect everywhere except the marker slot, where it emits ⟨A⟩.
   - **(4) transported tweak (the b⊗a⊗b\* flagship):** (g ⊗ Δ_A ⊗ g\*) ⊗ z_b with g = z_a′ ⊗ z_a\*. The question in its sharpest form: does the adjoint action re-address ⟨A⟩ → ⟨B⟩ inside the mechanism?

Metrics — this is the part that must not be skipped: exact and token acc as usual, **plus the marker-slot distribution** P(⟨A⟩), P(⟨B⟩), P(other) at the marker position for both candidates. Gradated outcomes: **hard pass** = (4) ≥ 0.5 exact; **soft pass** = P(⟨B⟩) under (4) ≥ 10× P(⟨B⟩) under (3) and ≥ 0.1 absolute. A soft pass is a real transport result — probability mass moving A→B under conjugation is the signature, even at 0.0 exact. Report the full marker-slot softmax for both candidates regardless of outcome.

### A2 — offset-reversal ladder (turn the ceiling into a law)

Pipelines rev_k: output = PAD×k + reverse(x), PAD a fixed neutral token, k ∈ {0, 1, 2}. Oracle-fit each at 3000 steps. Predictions: exact ≈ {0.97, ~0.82, lower}, monotone in k. Add one control at k=1 with a *different* pad token to confirm the ceiling tracks offset, not marker identity. Decode errors per k (expect reversal-tail scrambles growing with k, zero pad-token errors). Deliverable: the exact-vs-k curve — if monotone, "constant fields fail at position re-indexing" is a law, and Workstream B's Q2 tests the cure.

### A3 — D6 evaluation controls (apply before reading any D6 result)

1. **Primary metric = task exact of the emitted program on the held-out pipeline.** Oracle-distance (d_geo, KL-to-oracle) is secondary only, and must carry a weak-oracle flag on every offset-shaped pipeline (its oracle is itself ~0.82-quality; disagreement with a bad oracle is not failure).
2. **Primary unconstrained model = per-site weight-shared operator:** one small MLP (q_x, q_y[, site-group embedding]) → ℝ⁴ → normalize, applied at every site. This is the right inductive bias (composition, if it exists, should be approximately site-local) and converts 16 training pipelines × thousands of sites into a real dataset. The flattened-field MLP is demoted to an overfitting control, not a verdict-bearing model.
3. Constrained gauged product C(y,x) = g₁⊗y⊗g₂⊗x⊗g₃ as specced; pre-registered to fail held-out (a rigid motion cannot mechanism-substitute).
4. Report a learning curve over training-pipeline count (8, 12, 16) so a both-fail verdict has sample-size context. Both-fail is only meaningful if the per-site operator fails *with a flat learning curve*.
5. If D6 already completed under the old evaluation: re-evaluate existing checkpoints under task-exact first; retrain only the per-site operator.

---

## Workstream B (Studio 2) — the axis field / QRoPE half, first contact with data

This is the *initial idea* finally meeting an experiment: the latent program as a parameter of the model's rotational positional machinery. Design principle carried over from the vertical half: at identity program, the base is bit-exact.

### B0 — construction

**Base: Pythia-410m** (rotary; verify from config at runtime — numbers below assume hidden 1024, 24 layers, 16 heads, d_head 64, `rotary_pct` 0.25 ⇒ `rotary_ndims` = 16 per head; GELU MLP; `use_parallel_residual=True`, so attention and MLP branches are wrapped independently off their own LNs and summed — the sandwich wrapping is per-branch and unaffected).

**Axis field.** Standard rotary applies exp(m·Ω) to the first 16 dims of each q/k head, Ω block-diagonal skew (8 frequency pairs). The program conjugates the generators: Ω → R Ω Rᵀ, implemented without touching Ω at all, as a sandwich around the existing rotary call:

```
q[..., :nd] ← R_ax · RoPE_m( R_axᵀ · q[..., :nd] );   same R_ax for k
```

with R_ax built from a new site group `rope_ax`: quaternion field of shape **[L=24, H=16, 5]** (16 rotary dims = five 3-blocks + 1 identity dim) — 1,920 quaternions, 7,680 raw params. Properties this construction guarantees (and tests must verify): identity program ⇒ exact base (T13); **relative-position preservation** — logits are invariant to a constant shift of position_ids, because q(m)ᵀk(n) reduces to content·R·RoPE(n−m)·Rᵀ·content (T14); frequency spectrum of the relative rotation unchanged — conjugation reorients the coupling *planes*, never the frequencies (T15). Same R_ax on q and k is mandatory; different rotations would break relativity.

Engineering notes: pin the transformers version and force eager attention for correctness runs (rotary monkeypatching is sdpa/flash-version-sensitive); intercept at `apply_rotary_pos_emb`; fields fp32, cast at apply; the pass-through dims [16:64] are untouched. The v1 conjugation fields (attn_io, mlp_io, ffn_hidden, qk_rel) port to Pythia unchanged via the spec formulas — qk_rel (static, post-RoPE) is retained deliberately as the axis field's degenerate control.

### B1 (Q0) — wiring gates

T4-Pythia (identity program ≡ base, all fields), T5 dead-frame still holds, T13/T14/T15 above. **No experiment below runs until these are green.**

### B2 (Q1) — leverage ratio: the headline measurement

Hypothesis from the whole arc: conjugation fields' behavioral effect is capped by their own amplitude (the perturbative regime), but the axis field is *position-leveraged* — a small axis tilt φ produces effects ~φ·sin(ω·Δm), multiplied by relative distance.

Protocol: random fields at matched mean activity s̄ ∈ {0.005, 0.01, 0.03, 0.06} (uniform random axis per site, deterministic angle with sin²(θ/2) = s̄), one field type at a time ∈ {rope_ax, qk_rel, ffn_hidden, attn_io}. Measure KL(program ‖ base) on ~200 Pile-validation sequences, length 512. Report:
1. **Leverage L = KL/s̄** at s̄ = 0.01 (linear-response regime) per field type, plus the full curve.
2. **Distance-resolved KL**: per-position KL vs position. The signature prediction: flat-in-position for conjugation fields, **growing-in-position for the axis field**. This plot is the experiment.
3. Attention-displacement probe: mean |Δattention| vs relative distance, axis vs qk_rel.

### B3 (Q2) — does the axis field crack the offset ceiling? (causal test)

The toy base has no rotary, so: retrain the 4-layer toy **with rotary** (rotary_pct = 1.0 on its head dims for maximal effect), same tasks. Rerun A2's offset ladder twice: programs = conjugation fields only, then conjugation + axis field. Prediction: the axis field lifts the k ≥ 1 ceilings toward the k = 0 level. This closes the loop on the finding: constant fields fail at re-indexing (A2) ⇒ the positional program fixes it (B3). If B3 holds, the initial idea has its first positive, mechanism-specific result.

### B4 (Q3, stretch) — natural-language skill on Pythia

Fit one style expert (caps or French) on Pythia-410m with and without the axis field enabled; report recovery delta and where activity lands (does a natural-language skill *use* the axis sites?). Also logs a second point for the E1-recovery-vs-scale trend. Run only after B2/B3 report.

---

## Pre-registered predictions (score these when reports land)

| item | prediction | prior |
|---|---|---:|
| A1 (3) | near-perfect except ⟨A⟩ emitted at marker slot (token-keyed tweak) | 0.85 |
| A1 (4) | soft pass (marker mass moves A→B ≥10×, ≥0.1 abs) | 0.30 |
| A2 | exact monotone-decreasing in offset k; pad-identity control null | 0.75 |
| A3 | constrained operator fails held-out; per-site operator > flattened MLP on held-out | 0.80 |
| B2 | axis-field leverage > qk_rel leverage at s̄=0.01 | 0.65 |
| B2 | distance-resolved KL grows with position for axis field, flat for conjugation | 0.70 |
| B3 | axis field lifts k≥1 offset ceilings by ≥5 points | 0.55 |

## Do-not list (additions)

- Do not mark D3 transport as failed in any tree/summary — controls failed; flagship is A1.
- Do not read D6 through oracle-distance on offset-shaped pipelines (weak-oracle confound).
- Do not run any B experiment before B1 gates are green; do not use sdpa/flash paths for correctness gates.
- Do not apply different axis rotations to q and k (breaks positional relativity by construction).

## Definition of done

Studio 1: A1 marker-slot table + verdict; A2 exact-vs-k curve + error decode; A3 re-evaluated D6 verdict with learning curve. Studio 2: B1 gates green; B2 leverage table + distance-resolved plot; B3 twin ladders. Each as a short .md report in the established format, predictions table scored inline.
