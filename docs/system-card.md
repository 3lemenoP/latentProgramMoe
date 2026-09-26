# Latent-Program MoE — system card

**Scope.** A complete technical description of what exists in this repository as of 2026-09-26: the mathematical objects, the kernels and forward passes as implemented, the invariants and their proofs, every operation and objective in code, the diagnostic and belief-layer mathematics, the test suite, the measured constants, and the empirical regularities those constants support. Nothing here is narrative; every number is a measurement recorded in `docs/reports/`. Intended to give a reader with no repository access high-fidelity knowledge of the system.

---

## 1. The object

A frozen causal transformer `f_θ` (base) is reprogrammed by a **program field** `z`: one unit quaternion per SO(3) block at each of a fixed set of sites. The program is applied as an **activation sandwich** around every nonlinearity, `R · g(Rᵀ · x)`, where `R` is the block-diagonal rotation built from the site's quaternions and `g` is the base sub-computation. Base weights are never modified or materialised in rotated form. A program is a point on `(S³)^{N_sites}` modulo sign; a skill is one such point, saved as a file.

### 1.1 Block partition

For a feature dimension `dim`: `n3 = ⌊dim/3⌋` SO(3) blocks over dims `[0, 3·n3)`; the remainder `rem = dim − 3·n3 ∈ {0,1,2}` passes through untouched (identity).

### 1.2 Site groups (per layer ℓ)

| group | quaternion tensor shape | attaches to | present on |
|---|---|---|---|
| `attn_io` | `(n3(d_model),)` | residual-interface sandwich around attention | all bases |
| `qk_rel` | `(H, n3(d_head))` | relative key/query frame inside attention logits | all bases |
| `mlp_io` | `(n3(d_model),)` | residual-interface sandwich around the MLP | all bases |
| `ffn_hidden` | `(n3(d_ff),)` | sandwich around the elementwise activation | all bases |
| `rope_ax` | `(H, n3(rotary_ndims))` | conjugation of the RoPE generators | rotary bases only (`rotary_ndims > 0`) |

`tie_qk_across_heads` reduces `qk_rel` to `(1, n3)`; `rope_ax` is never tied across heads but the same rotation is applied to q and k of a head (mandatory, §4.5). Site keys are `(layer, name)`; the flat site order used everywhere is `[(ℓ, n) for ℓ in layers for n in site_names]`.

### 1.3 Parameter counts

| base | d_model / L / H / d_head / d_ff / rotary | quaternions per layer | total quaternions | raw params | fp16 bytes |
|---|---|---|---|---|---|
| GPT-2 small | 768 / 12 / 12 / 64 / 3072 / 0 | 256 + 252 + 256 + 1024 = 1,788 | 21,456 | 85,824 | 171,648 |
| Pythia-410m | 1024 / 24 / 16 / 64 / 4096 / 16 | 341 + 336 + 341 + 1365 + 80 = 2,463 | 59,112 | 236,448 | 472,896 |

Manifold dimension is 3 per quaternion (`3 · N`). LoRA r=8 on the same six matrices of GPT-2 small is ≈1.33M parameters. Optional gains (§3.6) add one scalar per quaternion.

### 1.4 What is not rotated

Token embeddings, positional embeddings, the unembedding, all LayerNorm/RMSNorm parameters. These stay in the unrotated residual "lab frame". Per-coordinate LayerNorm gains are symmetry breakers by design and must not be made rotation-covariant.

---

## 2. Conventions

- Quaternion order `q = (w, x, y, z)`, scalar first. Raw (unnormalised) parameters are stored; normalisation happens in forward.
- `q` and `−q` are the same rotation. Every loss and metric is sign-invariant: chordal `d²(a,b) = 1 − ⟨a,b⟩²`, never `‖a−b‖` or unsigned `arccos⟨a,b⟩`.
- Composition convention (fixed project-wide): applying skill `a` first, then `b`, is `q = hamilton(q_b, q_a)` per site, so `R = R_b R_a`.
- All quaternion mathematics in fp32 regardless of model dtype; rotation matrices are cast to the activation dtype at application.
- Field initialisation: identity plus Gaussian noise `σ = 1e-3` on the vector part.

---

## 3. Kernels (`lpm/quaternion.py`; everything else imports from here)

```
q_normalize(q)      = q / max(‖q‖, ε)                                   projection to S³
q_to_R(q)           = [[1−2(y²+z²), 2(xy−wz), 2(xz+wy)],
                       [2(xy+wz), 1−2(x²+z²), 2(yz−wx)],
                       [2(xz−wy), 2(yz+wx), 1−2(x²+y²)]]                unit q → SO(3)
hamilton(a,b)       : w = a_w b_w − a_v·b_v ;  v = a_w b_v + b_w a_v + a_v × b_v      R(a⊗b) = R(a)R(b)
d2_chord(a,b)       = 1 − ⟨a,b⟩²                                          ∈ [0,1], sign-invariant
q_conjugate(q)      = (w, −x, −y, −z)                                    inverse for unit q
q_angle2(q)         = 1 − w²  = sin²(θ/2)                                 "activity" s ∈ [0,1]
q_commutator(a,b)   = a ⊗ b ⊗ a* ⊗ b*
q_pow(q, λ)         : canonicalise w ≥ 0; θ = 2·atan2(‖v‖, w); q^λ = (cos(λθ/2), sin(λθ/2)·v̂)
q_log(q)            : canonicalise w ≥ 0; v = θ·v̂,  θ = 2·atan2(‖v‖, w)   rotation vector, log(−q) = log(q)
q_exp(v)            = (cos(‖v‖/2), sin(‖v‖/2)·v̂);  identity at v = 0
d_geo(a,b)          = arccos(|⟨a,b⟩|) ∈ [0, π/2]                          reporting only, never a loss
slerp(a,b,t)        : b ← −b if ⟨a,b⟩<0; ω = arccos|⟨a,b⟩|; (sin((1−t)ω)a + sin(tω)b)/sin ω; normalised lerp for ω→0
apply_rot(x,R,part,inverse)      : x[..., :3n3] reshaped (…, n3, 3); y = einsum('...ni,nji->...nj', x, R or Rᵀ); remainder concatenated
apply_rot_head(x,R,part,inverse) : x (B,H,T,d_head), R (H,n3,3,3); einsum 'bhtni,hnji->bhtnj'
```

Numerically certified in numpy before the torch implementation (20/20 property checks at ~1e-15).

---

## 4. Sandwiched forwards (`lpm/sandwich.py`; module-forward level wrapping, never weight level)

Let `R = R(attn_io)`, `M = R(qk_rel)` per head, `A = R(rope_ax)` per head, `R' = R(mlp_io)`, `S = R(ffn_hidden)`.

### 4.1 Attention (GPT-2, Llama-family, GPT-NeoX)

```
u        = Rᵀ x                                   x = post-LayerNorm input
q, k, v  = base projections of u, split to heads
[rotary bases]  q[:, :nd] ← A · RoPE_m(Aᵀ q[:, :nd]) ;  k[:, :nd] ← A · RoPE_m(Aᵀ k[:, :nd])   (same A on q and k)
q        ← Mᵀ q                                   query-side relative transport, after RoPE
attn     = softmax(q kᵀ / √d_h + mask) v          base path untouched; v carries no frame
o        = W_o · merge_heads(attn)
out      = R o
```

The spec writes `k ← M k`; the implementation uses the numerically identical query-side form `q ← Mᵀ q` (`qᵀ(Mk) = (Mᵀq)ᵀk`, tested). Reason: the qk_rel rotation never enters the KV cache, and under GQA the per-attention-head field applies to q, which always has the full head count. M is applied after RoPE (2-blocks and 3-blocks do not commute; the order is a fixed convention). The KV cache remains program-dependent through `attn_io` and upstream layers.

### 4.2 MLP, GELU variant (GPT-2, NeoX)

```
u = R'ᵀ x ;  a = S · W_up u ;  h = gelu(a) ;  y = W_down · Sᵀ h ;  out = R' y
```

### 4.3 MLP, SwiGLU variant (Llama-family)

```
u = R'ᵀ x ;  g = silu(S · W_gate u) ;  a = S · W_up u ;  h = Sᵀ (g ⊙ a) ;  y = W_down h ;  out = R' y
```

### 4.4 GPT-NeoX specifics

Parallel residual (`use_parallel_residual=True`): attention and MLP branches are wrapped independently off their own LayerNorms and summed. Partial rotary: `rotary_ndims = d_head · rotary_pct` (Pythia-410m: 16 of 64 dims per head ⇒ five 3-blocks + one identity dim). Correctness runs use eager attention.

### 4.5 Axis field (`rope_ax`) properties

Standard rotary applies `exp(m·Ω)` to the first `nd` dims of each q/k head, `Ω` block-diagonal skew with `nd/2` frequency pairs. The sandwich `A · RoPE_m(Aᵀ ·)` realises `Ω → A Ω Aᵀ` without touching `Ω`. Guarantees (tested T13–T15): identity program ⇒ bit-exact base; **relative-position preservation** — with the same `A` on q and k, `q(m)ᵀk(n) = q̃ᵀ A RoPE(n−m) Aᵀ k̃`, invariant to a constant shift of position ids; frequency spectrum of the relative rotation unchanged (conjugation reorients coupling planes, never frequencies). Different rotations on q and k would break relativity. A static post-RoPE `qk_rel` on a rotary base is itself position-coupled (`q̃ᵀ RoPE₋ₘ M RoPEₙ k̃ ≠ f(n−m)`) and serves as the axis field's degenerate control.

### 4.6 Abelian gains (`enable_gains`, default off)

`g = exp(ρ)`, one scalar per 3-block per site, applied immediately after the corresponding forward rotation as `x ← x ⊙ repeat_interleave(g, 3)`; SwiGLU applies the `ffn_hidden` gain once, on the up branch. Gains commute with rotations, compose additively in `ρ`, slerp linearly in `ρ`, preserve identity at `ρ = 0` (T10). Decision rule for enabling: Mirsky audit median ratio > 0.3 (§9.1); measured 0.028 ⇒ off.

### 4.7 Abelian baseline field (`AbelianProgramField`)

One scalar angle `θ_i` per site with a shared fixed axis bank `n̂_i`; `q_i = (cos θ_i/2, sin θ_i/2 · n̂_i)`. All such fields commute sitewise. Used as the E3 control; two theta-fields and the bank are fitted jointly, never sequentially.

---

## 5. Invariants (theorems with one-line proofs; each has a test)

1. **Identity program = base model.** All `q = (1,0,0,0)` ⇒ `R = I` at every site ⇒ every sandwich is the base sub-computation. Tested to max |Δlogit| < 1e-4 (fp32) on real GPT-2 (T4) and Pythia (T13).
2. **Spectrum preservation.** The effective weight under a program is `R W Rᵀ` (or `R₁ W R₂ᵀ` across a nonlinearity boundary), a product with orthogonal matrices; singular values and eigenvalues are program-independent; hence the network Lipschitz bound is program-independent (T6; certified per card by unit-norm and orthogonality checks plus three SVD spot-checks).
3. **Automorphism constraint.** `(R A Rᵀ)(R B Rᵀ) = R (A B) Rᵀ`: any purely linear composite of conjugated weights collapses to a single conjugation. All functional diversity flows through nonlinearities (GELU/SwiGLU, softmax mixing, LayerNorm gains).
4. **Dead-frame theorem.** A per-sequence value/output frame `C` (`W_v → C W_v`, `W_o → W_o Cᵀ`) cancels exactly because attention mixing is linear in `v`: `Cᵀ Σⱼ aᵢⱼ C vⱼ = Σⱼ aᵢⱼ vⱼ`. Same for any frame between two adjacent linear maps. Not a program parameter; tested numerically (T5, max |Δlogit| < 5e-4 on real GPT-2 fp32).
5. **Composition homomorphism.** `q_to_R(hamilton(a,b)) = q_to_R(a) q_to_R(b)` (T2); `R(q*) = R(q)ᵀ`; `q ⊗ q* = ±1` (T11).
6. **Sign invariance.** `R(−q) = R(q)`; model logits identical under a global sign flip of a field (T3).
7. **Relative-position preservation and frequency preservation of the axis field** (T14, T15; §4.5).
8. **Exact undo.** `compose(invert_field(z), z)` is the identity program sitewise; attested per card by max |Δlogit| < 1e-3·scale.

Consequences: rotations cannot change the singular spectrum, so an expert whose fine-tune moved spectra is unreachable by conjugation alone — Mirsky's inequality bounds the unreachable mass per matrix: `min_R ‖R W₀ Rᵀ − W_k‖_F ≥ ‖σ(W₀) − σ(W_k)‖₂` (necessary condition only; the architecture has the block-diagonal orbit, not the full one). Near identity the group is its own tangent space: for site angles `θ_A, θ_B` with orthogonal axes the commutator angle is ceilinged at `θ_A θ_B`, so `κ_ceiling = sin²(θ_A θ_B / 2)`.

---

## 6. Field-level operations (`lpm/compose.py`)

| operation | definition | gains | validation status |
|---|---|---|---|
| `compose(z_b, z_a)` | sitewise `q_b ⊗ q_a` (a first, then b), `R = R_b R_a` | `ρ_a + ρ_b` | geometry only; **not validated as skill stacking** (§10.4) |
| `invert_field(z)` | sitewise `q*` | `−ρ` | validated (undo attestation) |
| `increment(z_yx, z_x)` | `Δ_{y|x} = z_yx ⊗ z_x*` (identity `Δ ⊗ z_x ≡ z_yx`) | — | geometry only; transport not validated |
| `pow_field(z, λ)` | sitewise `q^λ` via `q_pow` | `λρ` | geometry (strength scans) |
| `slerp_field(z₀, z₁, α)` | sitewise sign-aligned slerp; `α` scalar, per-layer `[L]`, or dict by `(layer, name)` | linear in `ρ` | validated (E2: smooth continuum, per-layer schedule works) |
| `mean_field([z_k], w)` | sign-align all to `z_0` sitewise, weighted sum, normalise (chordal mean) | weighted mean `ρ` | implemented, tested (T-compose), not evaluated behaviourally |
| select / hot-swap | install a different `ProgramField` via `model.program(z)` | — | validated |

Validated verbs: **select, slerp, swap, undo**. Hamilton products remain legal for geometry (undo, drift, exp/log).

---

## 7. Objectives and training routines in code

### 7.1 Phase 1, distillation from a teacher (`scripts/fit_expert.py`)

```
L = KL( p_teacher(·|x) ‖ p_program(·|x) )                     batchmean over tokens, expert-domain text
  + λ_h  Σ_ℓ MSE(hidden_ℓ^teacher, hidden_ℓ^program)           λ_h = 0.1, summed over layers
  + λ_reg Σ_sites d²_chord(q, identity)                        λ_reg = 1e-4
```

AdamW, lr 1e-3, cosine schedule with 50 warmup steps, grad-clip 1.0, batch 8 × 128 tokens, 2000–3000 steps; base frozen, field fp32. Options: freeze site groups to identity (`--freeze-sites rope_ax`), per-group lr multipliers (`--rope-lr-mult`). Random-orthogonal control field reported alongside. Metric: `recovery = (CE_base − CE_program) / (CE_base − CE_teacher)` on held-out blocks.

### 7.2 Direct fit on task data (`scripts/e3_common.py::fit_field`, all toy experiments)

`L = CE(labels | program)` teacher-forced, optionally `− λ_o Σ_i s_A,i · s_B,i` (overlap bonus) and `+ λ_reg Σ d²_chord`. AdamW lr 1e-3, cosine, batch 64, 1500 (weak) or 3000 (strong) steps; `fit_fields_joint` fits several fields under one loss. No teacher.

### 7.3 Few-shot refinement (`lpm/encoder.py::refine`)

Adam on raw field parameters against the demo loss from any starting field (encoder output or identity), `T` steps, lr 1e-3; returns a detached trainable copy.

### 7.4 Whitened online refinement (`scripts/mvo.py::whitened_refine`)

`T` Adam steps on the raw rotation vector `v_raw = ṽ / c_g` against the current train batch's CE; interface stays whitened. Optimisation in raw geometry is mandatory (a whitened-parameterised Adam maps a uniform lr to `lr/c_g` raw steps).

### 7.5 Encoder (`lpm/encoder.py`, `scripts/train_encoder.py`)

Input: K demos formatted `"INPUT: {x}\nOUTPUT: {y}\n<sep>"`, concatenated, truncated to 1024 tokens, base tokenizer. Trunk: 4-layer transformer encoder, `d_enc = 512`, 8 heads, ff 2048, mean-pool. Heads: per base layer `512 → 64 → 4·quats(ℓ)`, zero-initialised weights with bias tiled `(1,0,0,0)` ⇒ emits the identity program at init. Loss:

```
L = λ_geo Σ_sites d²_chord(q̂(demos), q_k)   (regression to Phase-1 targets, λ_geo = 1)
  + λ_task CE(task batch | program = q̂)     (end-to-end through the frozen sandwiched base, λ_task = 1)
```

K ∈ {4, 16, 64}; gradients flow through `apply_rot` into the encoder only. Optional compositional episodes supervise toward `compose(z_b, z_a)`. Implemented and unit-tested; never trained at scale (E4 not run).

---

## 8. Diagnostics in code

- **Activity** `s_i = sin²(θ_i/2)`; per-group means; support overlap between two fields: cosine `⟨s_A, s_B⟩/(‖s_A‖‖s_B‖)` and `OVL = Σ min(s_A, s_B) / min(Σ s_A, Σ s_B)`.
- **Geometric commutator** `κ_i = q_angle2(q_commutator(a_i, b_i))`, reported with its kinematic ceiling `sin²(θ_A θ_B/2)`; three regimes: disjoint support (`κ ≪` ceiling, low cosine), second-order suppression (`κ ≈` ceiling), stabiliser-bound (`κ ≫` ceiling, behavioural KL ≈ 0).
- **Behavioural commutator** mean `KL(compose(b,a) ‖ compose(a,b))` on probes; anchors `KL(compose ‖ base)`, `KL(z_a ‖ z_b)`.
- **Leverage** `L_g = KL(program ‖ base) / s̄` for a random single-group field at matched mean activity `s̄` (uniform random axes, deterministic angle `sin²(θ/2) = s̄`); position coupling = late/early ratio of per-position KL (positions 448–511 ÷ 0–63); attention-displacement probe `|Δattn|/attn` by relative distance.
- **Mirsky ratio** per matrix `‖σ(W₀) − σ(W_k)‖₂ / ‖W_k − W₀‖_F`.
- **Failure decode** of exact-match misses into {first-token, reversal, length, other}.
- **Symmetrised KL** between program models on a fixed 200-probe set (the behavioural distance used for clustering, fingerprints and the metric fit).

---

## 9. Phase 4 mathematics (belief over programs)

### 9.1 Whitened tangent coordinates (`lpm/whitening.py`)

Per site `v_i = q_log(q_i) ∈ ℝ³`; whitened `ṽ_i = c_{g(i)} v_i` with `c_g² = L_g / (4 n_g)` (`n_g` = quaternion count of the group), so that `Σ_i ‖ṽ_i‖² ≈ KL-to-base` under linear response (`KL ≈ L_g s̄`, `s̄ ≈ θ²/4`). `L_g` from a W0 calibration: matched-activity random fields per group **at the operating strength**, averaged over ≥ 4 random draws (single-draw and small-`s̄` calibration fail the gate). Gate T18: random fields at equal whitened norm across groups give KL within ×2.

### 9.2 Gaussian belief (`lpm/belief.py`)

Per site `v_i ~ N(μ_i, σ_i² I₃)` in whitened coordinates, mean-field across sites; stored as precision `Λ` and `η = Λμ`.

```
predict:  Σ ← Σ + Q (Q = q_drift² I)       or  Λ ← λΛ, η ← λη   (precision forgetting, flag)
fuse:     Λ ← Λ + R,  η ← η + R m           (evidence mean m, precision R = r_obs⁻²; additive, commutative — T16)
sample:   ṽ = μ + σ ε ;  field = q_exp(ṽ / c)
MAP:      field = q_exp(μ / c)
potency:  H_i = (3/2) log(2πe σ_i²); per-group and total sums (the differentiation observable)
commitment map: per-group mean activity of the MAP field
```

Engineering constraints (binding): prior widths capped so the implied raw per-site angle ≤ 0.3 rad (`σ_cap = c_g · 0.3/√3`); refinement in raw geometry (§7.4). Drift and observation noise are set as fractions of the per-site prior width. Grid winner on the toy: `r_obs = q_drift = 0.1·σ₀`.

### 9.3 Mixture belief, IMM (`lpm/mixture_belief.py`)

K named components plus one novelty component, each a per-site Gaussian in **raw** rotation-vector coordinates (σ_init = 0.05 rad, novelty σ = 0.3, per-dim σ/√3).

```
responsibilities: r_k ∝ exp(−β · loss_k) · prior_k,   prior = p_stay on the previous winner, (1−p_stay)/(K−1) elsewhere
                  loss_k = measured probe loss of component k's MAP program on the current train batch (never geometric)
act:              deploy the winner's MAP (Thompson within-component behind a flag)
fuse:             into the winner only if r_winner > τ; else responsibility-weighted across components; never component-into-component (T27)
drift:            Q-inflation on all components
birth:            novelty has won for t_birth consecutive episodes with loss below its streak-start loss ⇒ it becomes a named component; a fresh novelty respawns (T26)
optional gate:    if min named loss > novelty_gate, novelty wins regardless of softmax
defaults:         β = 5, p_stay = 0.9, τ = 0.7, t_birth = 10
```

Components initialise from behavioural codebook medoids (k-medoids on symmetrised KL).

### 9.4 Learned behavioural metric (`lpm/metric.py`)

`V ∈ ℝ^{n×D}` flattened raw tangent vectors of the library (`D = 3N`); PCA to rank `r` keeping 95% variance (`r ≤ n−1`); nonnegative diagonal weights `w` in PCA coordinates fitted by projected gradient to `Σ_ij (Σ_c w_c ΔZ_ij,c² − symKL_ij)² + ridge‖w‖²`; out-of-subspace residual priced at W0 whitening weights:

```
d_M(a,b)² = Σ_c w_c (z_a − z_b)_c²  +  Σ_i c_{g(i)}² ‖r_a,i − r_b,i‖²
```

Gate T22 (leave-one-out Spearman of `d_M` against symKL must beat raw-L2 and random-whitened baselines). **Status: the gate failed on the real library (n = 27); `metric.json` ships with FAILED flags and is not consumed by any decision.** Same construction passes at ρ > 0.9 on synthetic libraries with a planted low-rank live subspace.

### 9.5 Constraint 0 (binding design rule of the belief layer)

No decision-bearing quantity may be manifold-native. Regime decisions and routing use measured probe loss; distances derived from geometry (raw geodesic, random-whitened, learned) appear only as diagnostics.

---

## 10. Report card (`lpm/report_card.py`, `lpm/card_render.py`, `scripts/make_card.py`)

Every fitted program emits `card.json` (+ md/html) with:

1. **Identity/sizes.** Base id, program hash, quaternion count, raw params, `bytes_fp16 = 8·N`, active sites at `s > 1e-4`, `bytes_8bit_sparse = 7·active + 16`.
2. **Fidelity.** Task CE per block for base/teacher/program; `recovery` with paired bootstrap 95% CI (2000 resamples, n blocks); teacherless skills report absolute metrics only.
3. **Gentleness.** Neutral-corpus perplexity base/teacher/program; collateral ratio `(ppl_program − ppl_base)/(ppl_teacher − ppl_base)` (degenerate when the teacher improves neutral text).
4. **Certificates.** Identity check (identity program through the serving path, max |Δlogit| < 1e-3·scale); spectrum (every rotation orthogonal within 1e-5, raw norms inside the tamper band [0.5, 2.0], three SVD spot-checks rel. error < 1e-4); Lipschitz statement (by construction, cites T1/T6); undo attestation (apply then conjugate, max |Δlogit| < 1e-3·scale).
5. **Commitment map.** Behavioural **work share** per group = `leverage_g × mean activity_g`, normalised; per layer-band (early/mid/late thirds); top-k sites; actuator class: `position` if rope_ax share ≥ 0.25, `content` if ≤ 0.05, else `composite`. Raw activity is a diagnostic column only.
6. **Activity statistics.** Mean, max, histogram with bins `[0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 1]`; prunability curve when available.
7. **Fingerprint.** Probe-suite loss vector; symmetrised KL to every library entry; top-3 neighbours.
8. **Limits (fixed text).** Certifies stability, size, scope of modification and reversibility; does not certify content alignment, factuality or safety of outputs. Unvalidated verb: multi-skill stacking.
9. **Provenance.** Teacher, steps, protocol, versions.

Tests TC-1 (identity program ⇒ all certificates green), TC-2 (one non-unit quaternion ⇒ spectrum certificate fails, nonzero exit), TC-3 (byte-identical determinism), TC-5 (French program: qk_rel work share > ffn_hidden) pass; TC-4 deferred.

---

## 11. Test suite (`tests/`, pytest; `-m gpt2` for real-checkpoint runs)

| id | statement | tolerance |
|---|---|---|
| T1 | `RᵀR = I`, `det R = +1` on random unit q | 1e-6 |
| T2 | `q_to_R(a⊗b) = q_to_R(a) q_to_R(b)` | 1e-5 |
| T3 | `R(−q) = R(q)`; logits identical under global field sign flip | exact / 1e-6 |
| T4 | identity program == base on real text (tiny and real GPT-2) | < 1e-4 fp32 |
| T5 | per-sequence value/output frame leaves logits unchanged | < 1e-4 tiny; < 5e-4 real GPT-2 |
| T6 | singular values of `R W Rᵀ` equal those of `W` | 1e-5 |
| T7 | field with vector noise σ = 0.3 changes logits materially | mean |Δ| > 0.1 |
| T8 | slerp endpoints, unit norm, antipodal inputs finite; gradients finite | 1e-6 |
| T9 | `compose(b,a) ≠ compose(a,b)` generically; matches `R_b R_a`; differs behaviourally | chordal > 0.01 |
| T10 | ρ = 0 preserves T4; gains compose additively | as T4 |
| T11 | `R(q*) = R(q)ᵀ`; `q⊗q* = ±1`; coaxial commutator = identity | 1e-6 |
| T12 | `q_pow(q,1) = canonical q`; `q_pow(q,0) = 1`; `q_pow(q,2) = ±q⊗q`; sign-invariant; continuous in λ | — |
| T13 | identity program == base on Pythia with `rope_ax` present | — |
| T14 | logits invariant to constant position-id shift under a random axis field | — |
| T15 | relative rotation frequency spectrum unchanged under conjugation | — |
| T16 | Gaussian fusion additive and order-invariant | — |
| T17 | sampled-field KL-to-base monotone and bounded in whitened norm | — |
| T18 | equal whitened norm ⇒ KL within ×2 across groups (calibration gate) | ×2 |
| T19 | `exp∘log` round-trip; `log(−q) = log(q)`; `log(1) = 0` | 1e-9 |
| T20 | stationary task: posterior concentrates, potency decreases, MAP within 5 points of direct fit | — |
| T21 | Bingham-fused and Gaussian-fused MAP agree in-regime | < 1° per site |
| T22 | learned metric LOO beats raw and random-whitened baselines | (fails on real library) |
| T23 | hybrid-metric consistency: dead-sea directions priced at W0 weights; live round-trip lossless | — |
| T24 | stationary: one IMM component takes responsibility ≥ 0.9, others untouched | — |
| T25 | scripted A→B→A stream: revisit recovery < first-visit | — |
| T26 | exactly one birth per novel regime | — |
| T27 | fusion isolation | — |
| misc | q-side qk_rel equivalence (± gains); KV-cache parity under a program; `generate` identity parity; padded-mask parity; hidden-state capture; program context restores previous; base params frozen; save/load round-trip; rotation cache invalidation; dtype cast refused | — |

---

## 12. Measured constants

All on the bases stated; CE in nats; "exact" = exact-match rate on 200 decodes; "recovery" as defined in §7.1.

### 12.1 Spectrum drift (Mirsky ratio, median over matrices)

| base | expert | median ratio | worst matrices |
|---|---|---|---|
| GPT-2 small | french / caps / jsonish / sentiment (LoRA r=8, 800 steps) | 0.033 / 0.026 / 0.024 / 0.032; overall **0.028** | `attn.c_proj` all layers; late `mlp.c_fc` (≤ 0.07) |
| Pythia-410m | french LoRA 800 / 2000 steps | 0.038 / 0.080 | — |

Threshold for gains: 0.3.

### 12.2 Distillation recovery

| base | skill | CE base / teacher / program | recovery | 95% CI | n |
|---|---|---|---|---|---|
| GPT-2 | french | 4.933 / 3.676 / 3.864 | 85.1% | — | — |
| GPT-2 | caps | 3.819 / 3.281 / 3.370 | 83.6% | — | — |
| GPT-2 | jsonish | 2.289 / 0.400 / 0.407 | 99.6% | — | — |
| GPT-2 | sentiment | 5.011 / 4.004 / 4.043 | 96.1% | — | — |
| GPT-2 | random-orthogonal controls | CE 15–18 | −7 to −24% | — | — |
| Pythia-410m | french (teacher 800 steps) | 3.266 / 2.876 / 2.796 | 120.6% | [114.0, 127.8] | 64 |
| Pythia-410m | jsonish (teacher 2000) | teacher gap +1.50 | 103.9% | [103.0, 104.8] | 64 |
| Pythia-410m | hedge (teacher 800) | 3.496 / 2.848 / 2.921 | 88.7% | [84.9, 92.4] | 64 |
| Pythia-410m | formal (teacher 800) | teacher gap +0.41 | 83.2% | [76.8, 90.7] | 64 |
| Pythia-410m | medical (teacher 400) | teacher gap +0.21 | 3.9% | [−17.8, 20.2] | 64 |

Pythia french generalisation gaps (train CE − eval CE): base −0.01; teacher 800 steps +1.19; teacher 2000 steps +3.67 (train 0.290, eval 3.960); program +0.42 (train 2.371, eval 2.796). The 2000-step teacher's eval CE is worse than the base. Axis-field arms on the french skill: conj_only 119.8%, conj+axis 119.5%, conj+axis lr×10 119.7%; rope_ax activity 0 / 0.0034 / 0.0064.

### 12.3 Neutral-text perplexity (gentleness)

| base | skill | base / teacher / program |
|---|---|---|
| GPT-2 | french endpoint (E2) | — / 231.9 / 121.3 |
| GPT-2 | caps endpoint (E2) | — / 45.5 / 43.6 |
| Pythia-410m | jsonish | 29.1 / 218.7 / 155.8 |
| Pythia-410m | hedge | 29.1 / 24.5 / 25.1 |
| Pythia-410m | formal | 29.1 / 23.3 / 23.9 |
| Pythia-410m | medical | 29.1 / 31.7 / 37.1 |

### 12.4 Merging (GPT-2, french + caps; CE A / CE B / neutral ppl at α = 0.5)

program slerp 4.256 / 3.631 / 52.2; program slerp early→A 4.107 / 4.003 / 75.9; weight linear 3.918 / 3.491 / 47.9; weight slerp 3.796 / 3.601 / 60.8. Program slerp is a smooth continuum with no midpoint collapse.

### 12.5 Toy composition suite (4-layer from-scratch GPT-2 / NeoX toys; atoms prepend ⟨A⟩, reverse, append ⟨B⟩, rotl, swap2, prepend ⟨B⟩)

- Atoms: prepend 1.000, reverse 0.970 (weak, 1500 steps) / 0.980 (strong, 3000 steps), append 0.995, prepend-⟨B⟩ 1.000.
- Zero-shot `compose(z_b, z_a)` and `compose(z_a, z_b)`: 0.000 exact on both pipelines at every strength λ ∈ {1, 1.25, 1.5, 1.75, 2}; token accuracy 0.17–0.29; both orders behaviourally identical (order-KL 0.005 weak, 0.015 strong). Abelian baseline: same pattern (non-abelian − abelian = −6.1 and −0.4 points).
- D1 battery, weak → strong: mean activity 0.025/0.030 → 0.057/0.054; support cosine 0.392 → 0.440; OVL 0.480 → 0.468; κ mean 0.00307 → 0.01267 with κ/ceiling ≈ 1.0 → 0.98; κ max 0.335 (layer 2). Skills live in `ffn_hidden` and `qk_rel`; `attn_io`/`mlp_io` ≤ 0.007.
- Direct pipeline oracles (strong): z_ab (prepend then reverse) 0.970; z_ba 0.820; z_cb 0.815; z_fb (payload-first) 0.975; all 20 ordered pipelines of the 5-atom library 0.73–1.0.
- Distances to z_ab: hamilton(b,a) d_geo 0.199 / KL 1.060; z_a 0.232 / 1.437; z_b 0.126 / 0.525; identity 0.167 / 1.013.
- Warm starts to z_ab, steps to 0.9 exact: identity 250, hamilton 500. z_ba plateau 0.76–0.84 from identity / hamilton / z_b inits; overfit-32 check reaches 1.000.
- Transport candidates on c_then_b: left `Δ ⊗ z_c` 0.000; adjoint `gΔg* ⊗ z_c` 0.000 (d_geo to oracle 0.315 / 0.317).
- Token re-keying (A1): `Δ_A ⊗ z_b` exact 0.0, token 0.882, marker slot P(⟨A⟩) 0.961; transported `gΔ_Ag* ⊗ z_b` exact 0.0, P(⟨A⟩) 0.740, P(⟨B⟩) 0.031, P(other) 0.228; oracle P(⟨B⟩) 0.940.
- Learned operators (D6, 16 training triples, 4 held-out): hamilton 0.005; gauged product `g₁⊗y⊗g₂⊗x⊗g₃` 0.000; per-site MLP (+group embedding) 0.000 with learning curve {8: 0, 12: 0, 16: 0}; flattened MLP 0.000 (train chordal loss → 0).
- Offset ladder (constant conjugation fields, no rotary): k=0 0.980; k=1 ⟨A⟩ 0.820; k=1 ⟨B⟩ 0.815; k=2 0.445; errors in the reversal tail, zero pad-token errors.

### 12.6 Leverage (Pythia-410m, 200 × 512-token sequences, KL/s̄)

| group | s̄ = 0.005 | 0.01 | 0.03 | 0.06 | late/early @0.01 | @0.06 |
|---|---|---|---|---|---|---|
| rope_ax | 3.40 | 1.91 | 4.45 | 5.70 | 1.74 | 3.38 |
| qk_rel | 79.9 | 70.7 | 56.5 | 47.0 | 2.41 | 1.91 |
| ffn_hidden | 33.9 | 60.3 | 156.3 | 138.0 | 0.89 | 1.20 |
| attn_io | 15.0 | 17.2 | 32.4 | 52.1 | 1.16 | 1.55 |

Attention displacement at s̄ = 0.01, far/near: rope_ax 0.65, qk_rel 1.03 (qk_rel ~3× larger magnitude). Toy base (no rotary) leverage ordering inverts: attn_io/mlp_io ≈ 24–26, ffn_hidden 0.47, qk_rel 1.1; random ffn_hidden rotations give KL 0.03 at θ ≈ 3.3 rad.

### 12.7 Rotary offset ladders (exact; rope_ax fitted activity in parentheses)

Full rotary toy (d_head 24, rotary_pct 1.0):

| rung | conj_only | conj+axis | axis_only | conj+axis, rope_ax lr×10 |
|---|---|---|---|---|
| k=0 | 0.875 | 0.930 (0.13) | 0.850 (0.35) | 0.955 (0.40) |
| k=1 ⟨A⟩ | 0.745 | 0.865 (0.16) | 0.195 (0.45) | 0.925 (0.44) |
| k=1 ⟨B⟩ | 0.770 | 0.885 (0.19) | 0.000 (0.40) | 0.930 (0.44) |
| k=2 | 0.535 | 0.755 (0.25) | 0.000 (0.47) | 0.880 (0.52) |

Quarter rotary toy (rotary_ndims 6):

| rung | conj_only | conj+axis | axis_only | lr×10 |
|---|---|---|---|---|
| k=0 | 0.855 | 0.880 (0.07) | 0.285 | 0.880 (0.15) |
| k=1 ⟨A⟩ | 0.725 | 0.775 (0.16) | 0.000 | 0.765 (0.28) |
| k=1 ⟨B⟩ | 0.645 | 0.745 (0.16) | 0.000 | 0.730 (0.28) |
| k=2 | 0.385 | 0.600 (0.19) | 0.000 | 0.570 (0.48) |

### 12.8 Phase 4 (toy bases)

- W0: naive single-`L_g` calibration fails T18 at ×105; operating-strength, seed-averaged calibration passes (max/min 1.99 at ρ = 0.1).
- Codebook (28 fields): ARI vs behavioural clustering, mean over k = 2..6: whitened 0.097, raw 0.187, learned M −0.062; whitened wins at k = 5, 6 only. Behavioural k = 4 families: {ba, be, cb, da, db}, {d, cd, dc, de, ed}, {b, ab, ac, ad, bc, c, ca, eb, fb}, {a, e, f, ae, bd, ce, ea, ec}. Geometric clustering co-clusters 50–108 behaviourally distinct pairs per k (e.g. c vs cb at symKL 2.5). symKL covering radius: 2.08 / 1.556 / 1.541 / 1.321 / 1.321 for k = 2..6.
- Metric LOO Spearman: M −0.213, raw 0.438, random-whitened 0.527.
- MVO v1 (600-episode stream, 25 refine steps): oracle 0.941; reset-on-spike 0.713 (recovery 41.5 episodes); SGD tracker 0.710 (47.4); agent Thompson-act 0.575 (53.1), MAP-act 0.596; static 0.184. De novo episodes to threshold: fresh prior 9, committed prior 14. Revisit-split recovery: agent 53.2 first / 53.1 revisit; tracker 46.2 / 48.0; reset 44.4 / 40.0. Commitment-map cosines vs direct fits (work shares): 0.910–0.997, mean 0.977.
- MVO v2 (recur-3 stream, codebook medoids {bc, d, ba, e}): oracle 0.891; SGD tracker 0.551 (57.8 / 54.9); reset-on-spike 0.526 (42.0 / 46.5); v1 agent MAP 0.450 (61.4 / 152.5); IMM medoid init 0.371 (142.2 / 48.7); IMM novelty-gated 0.113 (27 births); IMM random init 0.030; static 0.075. Responsibility ARI 0.77 early → 0.57 late (medoid) vs 0.22 (random).

### 12.9 Pruning (Pythia french program)

| τ | sites kept | task CE | neutral ppl | 8-bit sparse |
|---|---|---|---|---|
| 0 | 96.8% | 2.796 | 36.8 | 391 KB |
| 1e-4 | 95.1% | 2.797 | 36.9 | 384 KB |
| 3e-4 | 90.4% | 2.794 | 36.8 | 365 KB |
| 1e-3 | 72.4% | 2.790 | 35.5 | 292 KB |

Dense fp16 artifact 462 KB; LoRA r=8 teacher ≈ 10 MB.

---

## 13. Empirical regularities (stated as laws, with the measurements they rest on)

1. **Perturbative regime.** Fitted fields live at mean activity 0.003–0.06 (§12.5, §12.2); λ_reg and tiny init both produce it. Near identity the group is its own tangent space; sitewise Hamilton products reduce to first-order addition of rotation vectors and commutators are second-order (κ ≈ ceiling, §12.5).
2. **Automorphism obstruction to zero-shot composition.** `f_{R₂R₁} ≠ f_{R₂} ∘ f_{R₁}` for nonlinear `f`; measured: zero-shot products 0.0 exact at every strength, oracle nearer identity than the product, product warm-start slower than identity, transport 0.0, learned operators 0.0 with flat learning curves (§12.5).
3. **Offset re-indexing law.** A constant per-sequence conjugation field cannot re-index positional attention; exact match falls monotonically with the displacement of reversed content (0.98 → 0.82 → 0.445), independent of marker identity; errors concentrate in the reversal tail.
4. **Two actuators.** Conjugation fields (`attn_io`, `mlp_io`, `ffn_hidden`, `qk_rel`) are content actuators: saturating small-signal (leverage falls with strength), cannot re-index positions, carry ≈100% of a style skill. The axis field (`rope_ax`) is a position actuator: compounding structural (leverage and position-coupling grow with strength), repairs the offset law in proportion to displacement (+12/+11.5/+22, ceiling erased at lr×10), cannot insert content (axis-only collapses whenever a marker must be emitted), and on partial-rotary bases is an amplifier of conjugation programs, not a standalone actuator (axis-only 0.285 on plain reverse at quarter rotary).
5. **Lazy routing.** Naive joint fits under-use `rope_ax` when the axis dims are substitutable (full rotary: lr×10 adds +12.5 points at k=2); at quarter rotary each 3-block is load-bearing and the naive fit routes into it unaided (lr×10 slightly overshoots).
6. **Dead sea.** In high-capacity groups random rotations are behaviourally almost free even at half-turn strength while fitted skills concentrate exactly there; KL at fixed (group, s̄) is strongly direction-dependent. Consequences: random-probe calibration prices the wrong subspace; black-box search over the raw field is sample-inefficient.
7. **Geometric proximity is not behavioural proximity.** Six independent sightings: hamilton nearer than z_b geometrically but worse behaviourally; per-site operator geometrically nearest but behaviourally dead; raw co-clustering of behaviourally distinct fields (50–108 pairs per k); random-whitened ARI below raw; learned metric LOO negative. No static metric over programs (random-calibrated, geodesic or learned) has survived out-of-sample behavioural validation at n ≈ 27.
8. **Manifold regularisation.** At matched budget a LoRA teacher memorises (gen gap +3.67, eval worse than base) while the program distilled from an earlier teacher generalises (gap +0.42) and exceeds the better teacher (120.6%, CI excluding 100%). Skills whose teachers barely beat the base do not survive distillation (medical).
9. **Gentleness.** Conjugated experts cause less neutral-text collateral than the fine-tune they imitate where the teacher is destructive (GPT-2 french 121 vs 232; Pythia jsonish 156 vs 219); parity where the teacher is not.
10. **Regime memory on this substrate is representable but, at toy refit cost, not profitable.** IMM revisit recovery 48.7 vs first-visit 142.2; medoid init ARI 0.77 vs random 0.22; every belief architecture loses to spike-triggered refitting when a refit takes ~40 episodes.
11. **Unimodal fusion across regimes compromises.** The single-Gaussian agent shows no revisit advantage (+0.1 episodes); the switch-triggered forgetting it lacks is what reset-on-spike supplies.
12. **Differentiation observables.** Potency falls within a regime and rises at switches under Q-inflation; de novo learning from a committed prior is slower than from a fresh one (14 vs 9 episodes); converged commitment maps match direct-fit actuator profiles (cosine 0.977).

---

## 14. Operational constraints

- Load bases fp32 with eager attention for correctness runs (rotary interception is sdpa/flash-version-sensitive); transformers ≥ 5.
- Never swap programs mid-generation with a live KV cache (the cache is program-dependent through `attn_io` and upstream layers). Single-program cache parity is tested.
- With a program installed, `output_attentions` returns empty (the sandwich bypasses the base attention class's recording); `output_hidden_states` works.
- Do not `save_pretrained` a wrapped model; `ProgramField.save` (a dict of raw quaternion tensors keyed by site, plus optional `ρ`) is the shipped artifact.
- λ_reg pushes independently fitted skills toward minimal, near-disjoint supports and shrinks activity; fits destined for comparison of commutators must use λ_reg = 0 and the 3000-step protocol.
- Report κ only next to its activity ceiling; report geometric distances only next to a behavioural column.
- Routing and regime decisions use measured probe loss (Constraint 0).
- Certificate raw-norm band [0.5, 2.0] is uncalibrated: a healthy-but-drifted fit tripped it at 0.463.
- The safety-posture (hedge) skill is a register/style skill and is not built or evaluated as a guardrail-modification capability; every card carries the fixed limits text.

---

## 15. Code layout

```
lpm/quaternion.py     kernels (§3)                        lpm/whitening.py       c_g, flatten/unflatten (§9.1)
lpm/field.py          FieldSpec, ProgramField (cached      lpm/belief.py          WhitenedGaussianBelief (§9.2)
                      rotations, save/load), AxisBank,    lpm/mixture_belief.py  IMMBelief (§9.3)
                      AbelianProgramField                  lpm/metric.py          MetricM, loo_stress (§9.4)
lpm/sandwich.py       GPT-2 / NeoX / Llama forwards (§4)   lpm/report_card.py     card blocks (§10)
lpm/model_wrap.py     LatentProgramModel, program() ctx    lpm/card_render.py     md/html rendering
lpm/gains.py          apply_gain(_head) (§4.6)             lpm/tasks.py           french/caps/jsonish/sentiment/formal/medical/hedge
lpm/compose.py        §6 operations                                               + toy E3 vocab and behaviours
lpm/encoder.py        LPNEncoder, refine (§7.3, §7.5)      envs/task_stream.py    seeded regime-switching streams
lpm/config.py         LPMConfig ↔ configs/*.yaml           registry/hedge/0.1.0/  example card (json/md/html)
scripts/              verify_math, audit_mirsky, make_experts, fit_expert, train_encoder, eval_merge, eval_order,
                      eval_encoder, e3_common, e3prime, diagnose_e3, fit_e3_atoms, p0_expo_probe, a1_rekey, a2_ladder,
                      d6_operator, b2_leverage, b2_to_whitening, b3_rotary_ladder, calibrate_whitening, codebook, mvo,
                      mvo_v2, reanalyze_mvo, g1a_analyze, g1b_analyze, g1b_best_teachers, g1c_prune, make_card, gate1_chain.sh
```

Config defaults (`configs/base.yaml`): `base_model gpt2`, `enable_gains false`, `tie_qk_across_heads false`, `field_init_sigma 1e-3`, `lambda_h 0.1`, `lambda_reg 1e-4`, `lambda_geo 1.0`, `lambda_task 1.0`, `refine_steps 0`, `refine_lr 1e-3`; optimizer lr 1e-3, weight decay 0, betas (0.9, 0.999), grad clip 1.0, 2000 steps, 50 warmup, cosine, batch 8.

Bases exercised: GPT-2 small; Pythia-410m; 4-layer from-scratch GPT-2 and GPT-NeoX toys (`n_embd 96`; rotary at pct 1.0 with d_head 24, and pct 0.25). Llama-family wrappers (SwiGLU, GQA, RMSNorm) are implemented and unit-tested on tiny random configs only.

---

## 16. Implemented but not evaluated; not implemented

- Implemented, not run at scale: LPN encoder and E4 (encoder-only / encoder+refine / refine-from-identity / oracle / LoRA-per-task conditions); `mean_field` behaviour; Llama-family bases on real checkpoints; the compositional encoder curriculum (gated off).
- Not implemented (spec v2 backlog): per-token quaternion fields and the relative transport `R(zᵢ)ᵀR(zⱼ)` in logits (kernels already broadcast over leading dims); path-ordered transport / curvature; Bingham posterior on S³ (tangent Gaussian is the in-regime Laplace form, T21); unequal block partitions tied to head boundaries; re-basin alignment for heterogeneous checkpoints; distillation from experts not fine-tuned from the base; an adolescence period for newborn IMM components; teacherless natural-language fitting paths (direct CE, neutral-text KL term, prompt-distillation, preference objectives) in `fit_expert.py`; the Gate-2 library and dispatcher benchmark on a 7B base.
