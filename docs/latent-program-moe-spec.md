# Latent-program MoE — implementation specification

**Version 0.1 — SO(3) conjugation hypernetwork over a frozen transformer.**
Target implementer: Claude Code. Language: Python / PyTorch ≥ 2.2, HuggingFace `transformers`.

---

## 0. Summary and scope

A single frozen base transformer is reprogrammed by a **per-layer, per-block quaternion field**. Each 3-dimensional feature block carries a unit quaternion; the induced block-diagonal SO(3) rotation conjugates the layer's computation via an **activation sandwich** `R · f(Rᵀ · x)` — base weights are never modified or materialized in rotated form. Experts are points `z` on the program manifold; skills compose by **ordered Hamilton products** (non-abelian); merging is **slerp** (the exact geodesic). A small **latent program network (LPN) encoder** amortizes inference of `z` from few-shot demos, with optional test-time refinement.

Core guarantees the implementation MUST preserve:

1. **Identity program = base model.** The all-identity field reproduces base logits exactly (up to float tolerance).
2. **Spectrum preservation.** Conjugation is a similarity transform by an orthogonal matrix: singular values and eigenvalues of every effective weight are program-independent, hence the network Lipschitz bound is program-independent. Program switching cannot destabilize activations.
3. **The automorphism constraint.** Conjugation is an automorphism of the matrix algebra: `(R A Rᵀ)(R B Rᵀ) = R (A B) Rᵀ`. Any purely linear composite of conjugated weights collapses to a single conjugation. All functional diversity flows through the **nonlinearities** (GELU/SwiGLU, softmax mixing, per-coordinate LayerNorm gains). Consequence: some frames are provably dead (§3.4) and MUST NOT be implemented as program parameters; equivariant/norm-gated activations MUST NOT be introduced (they would make the program cancel).

Out of scope for v1 (listed in §12): per-token frames / gauge transport in attention, path-ordered curvature, Bingham variational posterior, cross-family merging (re-basin), heterogeneous checkpoints.

---

## 1. Mathematical conventions

All quaternion math in **fp32** regardless of model dtype; cast rotation matrices to activation dtype at application time.

### 1.1 Quaternions

Order convention `q = (w, x, y, z)`, scalar first. Unit norm enforced by normalize-in-forward:

```python
def q_normalize(q, eps=1e-8):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
```

Rotation matrix (for **unit** q):

```python
def q_to_R(q):
    # q: (..., 4) unit quaternions -> (..., 3, 3)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y),
          2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x),
          2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R
```

Hamilton product (composition; `R(a ⊗ b) = R(a) R(b)`):

```python
def hamilton(a, b):
    # (...,4) x (...,4) -> (...,4), a ⊗ b
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    w = aw*bw - (av*bv).sum(-1, keepdim=True)
    v = aw*bv + bw*av + torch.cross(av, bv, dim=-1)
    return torch.cat([w, v], dim=-1)
```

**Sign invariance is load-bearing.** `q` and `−q` give the same `R`. Every downstream use of a quaternion MUST be sign-invariant: either it goes through `q_to_R`, or it uses the chordal form

```python
def d2_chord(a, b):
    # sign-invariant squared distance in [0, 1]
    return 1.0 - (a * b).sum(-1) ** 2
```

Never use raw `‖a − b‖` or unsigned `arccos⟨a,b⟩` in a loss.

Slerp (geodesic interpolation), sign-aligned:

```python
def slerp(a, b, t, eps=1e-6):
    dot = (a * b).sum(-1, keepdim=True)
    b = torch.where(dot < 0, -b, b)          # antipodal alignment
    dot = dot.abs().clamp(max=1 - 1e-7)
    omega = torch.acos(dot)
    small = omega < eps
    so = torch.sin(omega)
    out = (torch.sin((1 - t) * omega) * a + torch.sin(t * omega) * b) / so
    lerp = q_normalize((1 - t) * a + t * b)
    return torch.where(small, lerp, out)
```

### 1.2 Block partition

For a feature dimension `dim`: `n3 = dim // 3` SO(3) blocks over dims `[0, 3·n3)`; the remainder `rem = dim − 3·n3 ∈ {0, 1, 2}` passes through **untouched (identity)** in v1. Provide:

```python
def make_partition(dim) -> Partition:  # records n3, rem, and the reshape views
```

Applying a field to activations (the only hot kernel):

```python
def apply_rot(x, R, partition, inverse=False):
    # x: (..., dim); R: (n3, 3, 3) fp32, cast to x.dtype at call
    xb = x[..., : 3 * partition.n3].reshape(*x.shape[:-1], partition.n3, 3)
    Rm = R.transpose(-1, -2) if inverse else R      # inverse = Rᵀ (orthogonal)
    yb = torch.einsum('...ni,nji->...nj', xb, Rm)
    return torch.cat([yb.reshape(*x.shape[:-1], 3 * partition.n3),
                      x[..., 3 * partition.n3:]], dim=-1)
```

Per-head variant for shape `(B, H, T, d_h)` uses `R: (H, n_h, 3, 3)` and einsum `'bhtni,hnji->bhtnj'`.

### 1.3 Composition convention

Applying skill **a first, then b**: per site `q_composed = hamilton(q_b, q_a)`, so `R_composed = R_b R_a`. This is fixed project-wide; write it in the docstring of `compose()` and test it (§9, T9). Optional abelian gains (§4) compose by elementwise multiplication.

---

## 2. Program fields

### 2.1 Sites

Per transformer layer `ℓ ∈ {0..L−1}`, four field groups (decision: **full per-block field, per layer**):

| name        | shape (quats)        | attaches to                                   |
|-------------|----------------------|-----------------------------------------------|
| `attn_io`   | `n3(d_model)`        | residual-interface sandwich around attention  |
| `qk_rel`    | `H × n3(d_head)`     | relative key transport inside attention logits|
| `mlp_io`    | `n3(d_model)`        | residual-interface sandwich around the MLP    |
| `ffn_hidden`| `n3(d_ff)`           | sandwich around the elementwise activation    |

A `ProgramField` is a dict `{(layer, name): tensor(..., 4)}` of **raw** (unnormalized) quaternions; normalization happens in forward. Provide `ProgramField.identity(cfg)` (all `(1,0,0,0)`), `ProgramField.randn_near_identity(cfg, sigma)`, save/load, and a cached `.rotations()` that builds all `R` tensors once per field-set and invalidates on parameter update (training) or field swap (inference).

### 2.2 Initialization

Fields init to identity + noise `σ = 1e−3` on the vector part. At init the model MUST be behaviorally the base model (test T4).

### 2.3 What is deliberately NOT rotated

Token embeddings, positional embeddings, the unembedding, and all LayerNorm/RMSNorm parameters stay in the unrotated residual "lab frame" and are frozen with the base. Per-coordinate LN gains are symmetry breakers by design — do not make them rotation-covariant.

---

## 3. Sandwiched modules

Wrap at the **module-forward level**, never the weight level (this makes HF quirks like GPT-2's `Conv1D` irrelevant). Implementation: wrapper `nn.Module`s that hold references to the frozen base submodules plus the current `ProgramField`; a context manager `with model.program(field): ...` installs/uninstalls it. Base parameters all have `requires_grad=False`.

### 3.1 MLP (GELU variant, GPT-2)

Base: `y = W_down · gelu(W_up · u)`. Sandwiched, with `R = R(mlp_io)`, `S = R(ffn_hidden)`:

```
u  = apply_rot(x, R, inverse=True)      # enter module frame
a  = W_up(u)
a  = apply_rot(a, S)
h  = gelu(a)
h  = apply_rot(h, S, inverse=True)
y  = W_down(h)
out = apply_rot(y, R)                   # exit to residual frame
```

Both `R` and `S` are live: each sandwiches a nonlinearity. Cost: 4 block-diagonal rotations per MLP ≈ O(d + d_ff) mults — negligible next to the matmuls.

### 3.2 MLP (SwiGLU variant, Llama-family)

`h = silu(W_gate u) ⊙ (W_up u)`. Rotate **both** branches by `S` before the elementwise ops, inverse after the product:

```
g = silu(apply_rot(W_gate(u), S))
a = apply_rot(W_up(u), S)
h = apply_rot(g * a, S, inverse=True)
y = W_down(h); out = apply_rot(y, R)
```

### 3.3 Attention

With `R = R(attn_io)`, `M = R(qk_rel)` (per head):

```
u       = apply_rot(x, R, inverse=True)          # x is the post-LN input
q, k, v = base projections of u, split to heads  # (B, H, T, d_h)
k       = apply_rot_head(k, M)                   # AFTER base RoPE, if any
attn    = softmax(q kᵀ / sqrt(d_h) + mask) v     # base path, untouched
o       = W_o(merge_heads(attn))
out     = apply_rot(o, R)
```

Notes:
- `M` enters the logits as `qᵀ (M k)` — the gauge-invariant content of separate query/key frames (`(A,B) → (GA,GB)` leaves `AᵀB` unchanged, so we parameterize the relative element directly, one field per head).
- If the base uses RoPE, apply `M` **after** RoPE (2-blocks and 3-blocks don't commute; the order is a convention, fix it and document it).
- `qk_rel` is per-head by default; config flag `tie_qk_across_heads` reduces it to one shared field.

### 3.4 The dead-frame theorem (MUST NOT implement as a program parameter)

A per-sequence **value/output frame** `C` (i.e., `W_v → C W_v`, `W_o → W_o Cᵀ`) cancels exactly: attention mixing `Σⱼ aᵢⱼ vⱼ` is linear in `v`, so `Cᵀ (Σⱼ aᵢⱼ C vⱼ) = Σⱼ aᵢⱼ vⱼ`. Same for any frame inserted between two adjacent linear maps with no nonlinearity between them. Do not add such parameters; add test T5 asserting the cancellation numerically (it doubles as a check that the sandwich machinery is correctly wired). Per-token frames break this cancellation — that is the v2 gauge-transport feature, not v1.

### 3.5 Precision and caching

Build all `R` in fp32; cast at `apply_rot`. During inference the field is constant: build once per `program()` entry. During field training rebuild each step (cheap: ≤ ~2k tiny 3×3s per layer). `torch.compile` the wrapped forward if convenient; the einsums fuse well.

---

## 4. Abelian gains (v1.1, gated on the Mirsky audit)

Conjugation preserves singular spectra, so it cannot reach an expert whose fine-tuning moved the spectrum. Mirsky's inequality gives the reachability floor per matrix:

```
min_R ‖ R W₀ Rᵀ − W_k ‖_F  ≥  ‖ σ(W₀) − σ(W_k) ‖₂
```

If experiment E0 (§10) shows material spectrum drift, enable **scalar-per-block gains**: `g ∈ ℝ^{n3}` per site group, applied as `x ← x · repeat_interleave(g, 3)` immediately after the corresponding forward rotation. Parameterize `g = exp(ρ)`, `ρ` init 0. Gains live in the trivial irrep: they commute with all rotations, compose multiplicatively (`ρ` adds), slerp linearly in `ρ`, and preserve the identity-program guarantee at init. The program becomes polar-decomposed: non-abelian rotation part + abelian gain part.

---

## 5. LPN encoder

### 5.1 Architecture

- Input: K few-shot demos, formatted `"INPUT: {x}\nOUTPUT: {y}\n<sep>"`, concatenated (truncate to 1024 tokens).
- Trunk: 4-layer transformer, `d_enc = 512`, 8 heads, trained from scratch (own small vocab/tokenizer = base model's tokenizer), mean-pool over tokens.
- Heads: per base-layer low-rank heads `512 → r=64 → 4 · sites(ℓ)`, output reshaped to raw quaternions per site.
- **Init for identity**: final projection weights zero, bias = tiled `(1,0,0,0)` — the encoder emits the identity program at init, so encoder training starts from base-model behavior.
- If gains are enabled, parallel heads emit `ρ` (zero-init).

### 5.2 Posterior (deferred)

v1 is a point estimate (MAP). v2: antipodally-symmetric variational posterior (Bingham, or projected normal with sign-invariant losses). Nothing in v1 may depend on quaternion sign (§1.1), so this upgrade is drop-in.

### 5.3 Test-time refinement

`refine(field, demos, steps=T, lr)` — Adam on the raw field params against the demo loss, starting from the encoder's output. Default T = 0 (off) for eval parity; E4 sweeps T ∈ {0, 50}.

---

## 6. Composition and merging API

```python
compose(field_b, field_a) -> ProgramField      # apply a first, then b: q = b ⊗ a per site; gains: ρ_a + ρ_b
slerp_field(f0, f1, alpha) -> ProgramField     # alpha: scalar | per-layer tensor [L] | dict by (layer, name)
mean_field(fields, weights) -> ProgramField    # k>2: sign-align all to fields[0], weighted sum, normalize (chordal mean)
```

Per-layer `alpha` is the point of per-layer fields: depth-wise slerp schedules (e.g., interpolate style layers while pinning early layers to expert A). `slerp_field` MUST use the sign-aligned slerp of §1.1 sitewise.

---

## 7. Training pipeline

Base frozen throughout. Field/gain params fp32, AdamW, grad-clip 1.0.

**Phase 0 — Mirsky audit (E0).** Before any training, run `scripts/audit_mirsky.py` on an existing fine-tune family; decide gains on/off.

**Phase 1 — per-expert direct fit.** For each expert `k` (a LoRA or full fine-tune of the base, made by `scripts/make_experts.py`), learn `z_k` by distillation:

```
L = KL( p_expert(·|x) ‖ p_program(·|x) )            on expert-domain data
  + λ_h  Σ_ℓ MSE(hidden_ℓ^expert, hidden_ℓ^program)   (λ_h = 0.1; layerwise signal, cheap)
  + λ_reg Σ_sites d2_chord(q, identity)                (λ_reg = 1e-4; keeps programs near base →
                                                        emergent sharing across experts)
```

**Warning (E3′ §3.3):** λ_reg pushes each independently fitted skill toward a *minimal* support. Two skills then tend to occupy disjoint sites, the sitewise Hamilton product abelianizes (`q ⊗ I = q`), and zero-shot composition is vacuously order-blind. When skills are destined for composition, fit them jointly with a shared-support term or mask (E3′ D4). Record the D4-winning variant as the Phase-1 default if that gate passes.

lr 1e-3, cosine, 1–3k steps typical. Deliverable: `z_k` checkpoints + recovered-performance report.

**Phase 2 — encoder amortization.** Meta-train over the task distribution:

```
L = λ_geo Σ_sites d2_chord(q̂(demos), q_k)   (regression to Phase-1 targets, λ_geo = 1.0)
  + λ_task CE(task batch | program = q̂)      (end-to-end through the frozen sandwiched base, λ_task = 1.0)
```

Sample K ∈ {4, 16, 64} demos per episode. Gradients flow through `apply_rot` into the encoder only.

**Phase 3 — compositional curriculum (after E3′ D4 passes, or D6 ships a composition operator).** Episodes of ordered task pairs (a then b): supervise `q̂` toward `compose(z_b, z_a)` and/or end-task loss on composed behavior. Order must be sampled both ways. The original E3 zero-shot-Hamilton gate is superseded (`e-suite-analysis-e3prime.md`).

---

## 8. Reference kernels — file `lpm/quaternion.py`

Contains exactly: `q_normalize`, `q_to_R`, `hamilton`, `d2_chord`, `slerp`, `q_conjugate`, `q_angle2`, `q_commutator`, `d_geo`, `make_partition`, `apply_rot`, `apply_rot_head`. All pure functions, no module state, property-tested (§9). Everything else imports from here; no re-implementations elsewhere in the repo.

---

## 9. Tests and acceptance criteria (`tests/`, pytest)

| id | test | pass criterion |
|----|------|----------------|
| T1 | `q_to_R` orthogonality: `RᵀR = I`, `det R = +1` on random unit q | atol 1e-6 |
| T2 | homomorphism: `q_to_R(hamilton(a,b)) == q_to_R(a) @ q_to_R(b)` | atol 1e-5 |
| T3 | sign invariance: `q_to_R(-q) == q_to_R(q)`; model logits identical under global sign flip of a field | exact / atol 1e-6 |
| T4 | **identity program == base model**: `ProgramField.identity` vs unwrapped base on real text | max abs logit diff < 1e-4 (fp32), < 3e-2 (bf16) |
| T5 | **dead value frame**: test-harness-only per-sequence conjugation of `W_v`/`W_o` leaves logits unchanged | max abs diff < 1e-4 (tiny); < 5e-4 on real GPT-2 small fp32 (12-layer residual sits at ~1e-4) |
| T6 | spectrum preservation: materialize `R W Rᵀ` for a random field (test only); singular values match `W`'s | atol 1e-5 |
| T7 | program is live: field with vector-part noise σ=0.3 changes logits materially | mean abs diff > 0.1 |
| T8 | slerp endpoints: `slerp(a,b,0)=a`, `slerp(a,b,1)=±b`; output unit-norm; antipodal inputs don't NaN | atol 1e-6 |
| T9 | composition order: `compose(b,a) != compose(a,b)` for generic fields (chordal distance > 0.01 at some site); matches convention `R_b R_a` | — |
| T10 | gains identity: ρ=0 preserves T4; gains compose additively | atol as T4 |
| T11 | conjugate is inverse: `R(q*)=R(q)ᵀ`, `q⊗q*=±I`; coaxial commutator is identity | atol 1e-6 |

T4 and T5 are the theory-in-code checks; if either fails, the wiring is wrong — stop and fix before training anything.

---

## 10. Experiments

**E0 — Mirsky audit** (`scripts/audit_mirsky.py --base <model> --experts <ckpts...>`). Per weight matrix report `‖σ(W₀)−σ(W_k)‖₂ / ‖W_k−W₀‖_F` (the fraction of the fine-tune displacement that rotation provably cannot express). Output: markdown table per layer/matrix-type + summary. **Decision rule: median ratio > 0.3 → enable gains (§4) for all subsequent phases.**

**E1 — expert reconstruction.** Base: GPT-2 small (d=768 — divisible by 3, convenient). Experts: 3–4 LoRA fine-tunes on contrastive tasks (e.g., French continuation, all-caps style, JSON-ish formatting, sentiment-steered). Fit `z_k` (Phase 1). Metric: fraction of the expert-vs-base performance gap recovered. Target ≥ 80% with gains on; report with gains off; control: random-orthogonal field (should recover ~0%).

**E2 — geodesic merging.** Pairs of E1 experts: `slerp_field` at α ∈ {0, .25, .5, .75, 1} vs weight-space linear and weight-space slerp of the LoRA-merged checkpoints. Metrics: both tasks' scores at midpoint + perplexity spike on neutral text. Sweep one per-layer schedule (early layers pinned to A, late interpolated).

**E3 — order sensitivity (superseded as a gate by E3′).** Original protocol: small from-scratch transformer; independently fitted `z_a` (prepend) / `z_b` (reverse); zero-shot Hamilton compose vs abelian baseline. Studio result 2026-08-15: atoms 1.0 exact, compose 0.0 exact and order-blind. See `docs/reports/e3-order.md`. **E3′** (`e-suite-analysis-e3prime.md` §4) is the composition suite: D1 support/commutator diagnostic, D2 oracle pipelines, D3 transport, D4 shared-support refit (new Phase-3 gate), D5 strength scan, D6 learned operator. E5 (conjugation transport) is D3.

**E4 — encoder generalization.** Held-out task suite; conditions: encoder-only (T=0), encoder+refine (T=50), refine-from-identity (no encoder), per-task direct fit (oracle), LoRA-per-task (upper baseline). Report vs K demos.

---

## 11. Repository layout, milestones, config

```
latent_program_moe/
  lpm/
    quaternion.py      # §8 kernels only
    field.py           # ProgramField, sites, init, save/load, rotation cache
    sandwich.py        # wrapped attention + MLP forwards (§3)
    model_wrap.py      # HF model wrapping, program() context manager
    gains.py           # §4 (behind config flag)
    encoder.py         # §5
    compose.py         # §6
  scripts/
    audit_mirsky.py  make_experts.py  fit_expert.py  train_encoder.py
    eval_merge.py    eval_order.py    eval_encoder.py
  tests/               # T1–T10
  configs/             # yaml per experiment
```

Config (dataclass, mirrored in yaml): `base_model`, `enable_gains`, `tie_qk_across_heads`, `field_init_sigma`, `lambda_h`, `lambda_reg`, `lambda_geo`, `lambda_task`, `refine_steps`, optimizer block, per-experiment data paths.

Milestones (each gated on its tests):
- **M0** kernels + T1/T2/T3/T8/T9 green.
- **M1** GPT-2 wrapping + T4/T5/T6/T7/T10 green.
- **M2** E0 audit report; gains decision recorded in the config default.
- **M3** experts made; Phase 1 fits; E1 report.
- **M4** merge + composition; E2/E3 reports.
- **M5** encoder; Phase 2 (+3 if E3 passed); E4 report.

Dependencies: `torch>=2.2`, `transformers`, `datasets`, `peft` (for making experts), `numpy`, `pytest`. Single-GPU sufficient for everything at GPT-2-small scale.

---

## 12. Explicitly out of scope (v2 backlog)

Per-token quaternion fields and the relative transport `R(zᵢ)ᵀR(zⱼ)` in logits (gauge connection over the sequence; `apply_rot` already broadcasts over leading `(B,T)` dims — design for it, don't build it). Path-ordered transport / curvature. Bingham posterior. Unequal block partitions tied to head boundaries (the Cauchy–Schwarz backbone trade). Re-basin alignment for heterogeneous checkpoints. Distillation from experts that were not fine-tuned from this base.

---

## Appendix A — parameter counts (GPT-2 small: d=768, L=12, H=12, d_h=64, d_ff=3072)

Per layer: `attn_io` 256 + `mlp_io` 256 + `ffn_hidden` 1024 + `qk_rel` 12×21 = 252 → **1,788 quaternions = 7,152 raw params** (manifold dim 5,364). Total: **21,456 quaternions = 85,824 raw params** (manifold 64,368); optional gains +21,456. Compare LoRA r=8 on the same six matrices ≈ 1.33M — the program is ~15× smaller, and spectrum-safe. Encoder dominates trainable params (~12M trunk + ~6M low-rank heads); the program itself is the artifact that ships per skill.

Note: d_h = 64 leaves 1 identity dim per head; d = 768 and d_ff = 3072 partition exactly.

---

## Appendix B — certification status (added after M0 pre-work)

The §8 kernel formulas were numerically certified before implementation via a numpy twin (`verify_math.py`, ships alongside this spec): 17/17 property checks pass at ~1e-15, covering T1/T2/T3/T6/T8/T9 math plus the composition convention, the automorphism identity, and the dead value-frame cancellation. Drop-in files provided: `quaternion.py` (→ `lpm/quaternion.py`), `test_quaternion.py` (→ `tests/`), `audit_mirsky.py` (→ `scripts/`). The torch code is line-for-line the certified math; run `pytest tests/test_quaternion.py` first in the target environment, then proceed to M1.
