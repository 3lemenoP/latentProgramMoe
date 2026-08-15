# Latent-program MoE

**SO(3) conjugation hypernetwork over a frozen transformer** — implementation of
[`docs/latent-program-moe-spec.md`](docs/latent-program-moe-spec.md) (v0.1).

A single frozen base transformer is reprogrammed by a per-layer, per-block
quaternion field. Each 3-dimensional feature block carries a unit quaternion;
the induced block-diagonal SO(3) rotation conjugates the layer's computation
via an activation sandwich `R · f(Rᵀ · x)` — base weights are never modified.
Experts are points `z` on the program manifold; skills compose by ordered
Hamilton products (non-abelian); merging is slerp (the exact geodesic); a small
LPN encoder amortizes inference of `z` from few-shot demos.

Core guarantees (enforced by tests, see below):

1. **Identity program = base model** (T4) — the all-identity field reproduces
   base logits exactly.
2. **Spectrum preservation** (T6) — conjugation is a similarity transform by an
   orthogonal matrix; program switching cannot destabilize activations.
3. **The automorphism constraint** — purely linear composites of conjugated
   weights collapse; all functional diversity flows through the nonlinearities.
   Dead frames (per-sequence value/output rotations) cancel exactly (T5) and
   are not program parameters.

## Layout

```
lpm/
  quaternion.py    reference kernels (§8) — certified, everything imports from here
  field.py         FieldSpec, ProgramField, AbelianProgramField (E3 baseline)
  sandwich.py      wrapped attention + MLP forwards (§3; GPT-2 and Llama-family)
  model_wrap.py    LatentProgramModel + program() context manager
  gains.py         abelian scalar-per-block gains (§4, behind enable_gains)
  encoder.py       LPN encoder + test-time refinement (§5)
  compose.py       compose / slerp_field / mean_field (§6)
  config.py        LPMConfig dataclass, mirrored in configs/*.yaml
  tasks.py, utils.py   experiment support for scripts/
scripts/
  verify_math.py   numpy certification of the kernel formulas (17 checks)
  audit_mirsky.py  E0 — reachability floor, gates enable_gains (§4/§10)
  make_experts.py  LoRA experts on contrastive tasks (E1 prerequisites)
  fit_expert.py    phase 1 — per-expert direct fit by distillation (§7)
  train_encoder.py phase 2 (+3) — encoder amortization / compositional curriculum
  eval_merge.py    E2 — geodesic merging vs weight-space baselines
  eval_order.py    E3 — order sensitivity falsifier (+ abelian baseline)
  eval_encoder.py  E4 — encoder generalization on held-out tasks
tests/             T1–T10 (pytest; `-m gpt2` for real-checkpoint acceptance runs)
configs/           yaml per experiment
```

## Install

```bash
pip install -e .[dev]          # torch>=2.2, transformers v5, datasets, peft
```

## Quickstart

```python
import torch
from lpm import LatentProgramModel, ProgramField, compose, slerp_field

model = LatentProgramModel.from_pretrained("gpt2")   # frozen, fp32, eager
field = ProgramField.randn_near_identity(model.spec, sigma=1e-3, trainable=True)

with model.program(field):                 # sandwiched forward
    out = model(input_ids=ids)
out = model(input_ids=ids)                 # base model, untouched path

z = compose(z_b, z_a)                      # apply a first, then b (R_b R_a)
z = slerp_field(z_a, z_b, alpha=0.5)       # geodesic merge (per-layer alpha ok)
field.save("z_task.pt")                    # the program IS the shipped artifact
```

Conventions (fixed project-wide, spec §1): `q = (w, x, y, z)` scalar-first;
`q` and `−q` are the same rotation and every loss must be sign-invariant
(`d2_chord`, never raw distances); "a then b" composes as `hamilton(q_b, q_a)`;
all quaternion math in fp32.

## Test gates

```bash
python scripts/verify_math.py        # M0 pre-gate: 17/17 numpy certification
pytest -q -m "not gpt2"              # fast offline suite (tiny random models)
pytest -q -m gpt2                    # M1 acceptance on real GPT-2 small
```

T4 (identity program == base) and T5 (dead value frame cancels) are the
theory-in-code checks: if either fails, the wiring is wrong — stop and fix
before training anything.

## Milestone runbook

CPU is fine through M1; **run M2–M5 on a LightningAI GPU studio** (single GPU
is sufficient at GPT-2-small scale).

### LightningAI setup (once per studio)

```bash
git clone https://github.com/3lemenoP/latentProgramMoe.git
cd latentProgramMoe
pip install -e .[dev]
python scripts/verify_math.py && pytest -q      # sanity gate on the studio
```

### M2 — E0 Mirsky audit → gains decision

```bash
python scripts/make_experts.py --base gpt2 --out experts \
    --tasks french caps jsonish sentiment --steps 800
python scripts/audit_mirsky.py --base gpt2 \
    --experts experts/french/merged experts/caps/merged \
              experts/jsonish/merged experts/sentiment/merged \
    --out report_e0.md
```

Decision rule: median ratio > 0.3 → set `enable_gains: true` in
`configs/*.yaml` for all subsequent phases (record it in `configs/base.yaml`).

### M3 — phase-1 fits + E1 report

```bash
for t in french caps jsonish sentiment; do
  python scripts/fit_expert.py --config configs/e1_experts.yaml \
      --task $t --experts-dir experts --out runs/e1 --control
done
```

Target: ≥80% of the expert-vs-base gap recovered (with gains on, if enabled);
the random-orthogonal control should recover ~0%.

### M4 — merging + composition

```bash
python scripts/eval_merge.py --config configs/e2_merge.yaml \
    --task-a french --task-b caps --experts-dir experts --z-dir runs/e1
python scripts/eval_order.py --config configs/e3_order.yaml --out report_e3.md
```

E3 pass (non-abelian − abelian ordered accuracy > 20 points) unlocks the
phase-3 compositional curriculum.

### M5 — encoder + E4

```bash
# leave-one-out: train on three tasks, hold out the fourth
python scripts/train_encoder.py --config configs/e4_encoder.yaml \
    --tasks french caps jsonish --out runs/encoder \
    $( [ -f report_e3.json ] && echo --compositional )
python scripts/eval_encoder.py --config configs/e4_encoder.yaml \
    --encoder runs/encoder/encoder.pt --task sentiment
```

## Notes / decisions log

- **E0 / enable_gains (2026-08-13)**: Mirsky audit on the four LoRA experts
  (`french`, `caps`, `jsonish`, `sentiment`; 800 steps, GPT-2 small) gave
  overall median ratio **0.028** (per-task medians 0.024–0.033) against the
  spec threshold 0.3. Decision: **gains OFF** — recorded in
  `configs/base.yaml`. Report: [`docs/reports/e0-mirsky.md`](docs/reports/e0-mirsky.md).
- **E1 (2026-08-15)**: phase-1 fits recovered **85% / 84% / 100% / 96%**
  of the expert–base CE gap on french / caps / jsonish / sentiment
  (gains off). Random-orthogonal controls all largely *negative*.
  Report: [`docs/reports/e1-experts.md`](docs/reports/e1-experts.md).
- **E2 (2026-08-15)**: french+caps midpoint — weight-space linear beats
  program slerp on CE A, CE B, and neutral ppl. Program slerp is smooth
  (no collapse) but is not the better merge. Report:
  [`docs/reports/e2-merge-french-caps.md`](docs/reports/e2-merge-french-caps.md).
- **E3 (2026-08-15)**: atomic prepend/reverse programs are exact (1.0);
  zero-shot Hamilton compose is exact 0.0 and order-blind (non-abelian
  − abelian ≈ −6 / 0 points; spec wants > 20). Report:
  [`docs/reports/e3-order.md`](docs/reports/e3-order.md). Phase-3
  compositional curriculum stays **off**.
- **transformers v5** (≥5.x) module interfaces; base loaded fp32 +
  `attn_implementation="eager"` by default. T4/T5 assert parity with the
  unwrapped base, so interface drift fails loudly.
- **qk_rel placement**: the spec writes the relative transport as `k ← M k`
  inside the logits; both adapters implement the numerically identical
  query-side form `q ← Mᵀ q` (after RoPE where the base has RoPE), with gains
  as `q ← Mᵀ(g ⊙ q)`. Why q-side: the qk_rel rotation never enters the KV
  cache, and under GQA the per-attention-head field applies to q, which always
  has the full head count. Equivalence `qᵀ(g ⊙ Mk) = (Mᵀ(g ⊙ q))ᵀk` is tested
  at the kernel level (`test_qk_rel_query_side_equivalence`). The cache
  remains program-dependent through `attn_io` and upstream layers — never
  swap programs mid-generation with a live cache.
- **`output_attentions` limitation**: with a program installed, returned
  attentions are empty (transformers v5 records them from the base attention
  class, which the sandwich path bypasses); `output_hidden_states` works and
  is tested.
- **SwiGLU gains**: the `ffn_hidden` gain applies once, on the up branch, so
  the gated product carries the gain exactly once (spec §4 is single-branch).
- **E3 abelian baseline protocol**: the two theta-fields and the shared
  `AxisBank` are fitted JOINTLY (`fit_fields_joint`), never sequentially — a
  second sequential fit would move the shared axes out from under the first
  field's frozen thetas, contaminating the baseline with a protocol artifact.
  (A reduced-scale probe measured only ~1.5° mean axis drift, but the
  falsifier must not depend on drift staying benign at full fit length.)
- Do not `save_pretrained` a wrapped model; programs (`ProgramField.save`) are
  the artifact that ships. The base stays pristine.
- v2 backlog (spec §12): per-token fields/gauge transport, path-ordered
  curvature, Bingham posterior, re-basin merging.
