# The story so far — from the initial idea to the current state

**Written 2026-09-26 as a reading guide to the documents in this repository.** Every number below is taken from a report in `docs/reports/` or a steering document at the repo root; the documents themselves follow this guide in chronological order.

---

## 1. The idea (2026-08-12)

A fine-tune changes a transformer's weights. This project asked whether a skill could instead be expressed as a **rotation of the model's activations**, leaving every weight untouched.

The construction is small. Cut every feature dimension into blocks of three. Give each block a unit quaternion, so each block carries a rotation in SO(3). At every point where a layer applies a nonlinearity, rotate the activations into the block frame, apply the base computation, and rotate back: `R · f(Rᵀ · x)`. The set of all these quaternions across all layers is the **program field**, and a skill is a point `z` on that manifold.

Three things are then true by construction rather than by training:

1. **The identity program is the base model.** All quaternions `(1,0,0,0)` reproduce base logits exactly (test T4).
2. **Spectra are preserved.** Conjugating a weight by an orthogonal matrix leaves its singular values alone, so a program can never destabilise the network (T6).
3. **Only the nonlinearities carry the program.** A rotation between two adjacent linear maps cancels exactly, which is why a value/output frame in attention is provably dead (T5) and is not a parameter.

On GPT-2 small the whole program is 21,456 quaternions, about 86 KB of fp32, roughly fifteen times smaller than a rank-8 LoRA on the same matrices.

The idea came with an algebra attached. Because rotations form a group, skills should compose by ordered Hamilton products (`a` then `b` is `q_b ⊗ q_a`), merge along exact geodesics (slerp), and undo exactly (the conjugate). A small encoder would infer `z` from a handful of demonstrations. The spec (`docs/latent-program-moe-spec.md`) laid out milestones M0 to M5 and experiments E0 to E4 to test all of this, with the kernel maths certified in numpy (20/20 checks) before any torch code existed.

## 2. The substrate holds (E0, E1, E2 — 2026-08-13 to 15)

**E0** asked whether rotation could reach fine-tunes at all. Mirsky's inequality gives a floor on how much of a weight change no rotation can express; across four LoRA experts the median was 0.028 against a decision threshold of 0.3. Fine-tuning barely moves singular spectra, so the optional abelian gains stayed off.

**E1** fitted a program to each of four LoRA experts by distillation and recovered 85%, 84%, 100% and 96% of the expert-versus-base gap. Random-orthogonal controls recovered nothing. The substrate was expressive.

**E2** tested geodesic merging against weight-space interpolation and lost: at the midpoint, plain weight averaging beat program slerp on both tasks and on neutral perplexity. But the endpoints carried a finding that became a headline. The French fine-tune drove neutral-text perplexity from about 45 to 232; the program that imitates it stopped at 121. Conjugated experts are gentler, at about half the collateral damage, and that follows from the spectrum guarantee. Per-layer slerp schedules also worked as a control surface.

## 3. The algebra fails, and the failure is diagnosed (E3 to Workstream A — 2026-08-15)

**E3** was the falsifier for ordered composition. On a tiny from-scratch transformer, two skills (prepend a marker; reverse the sequence) each fitted to 100% exact match. Their Hamilton product scored 0.0 in both orders, and the two orders were behaviourally identical.

What followed was a week of diagnosis compressed into a day, each step a pre-registered prediction scored against its prior:

- `e-suite-analysis-e3prime.md` proposed that the regulariser had pushed the two skills onto disjoint sites, where any algebra commutes. **D1** measured the overlap and rejected this: support cosine 0.39. The skills overlapped but their commutator was behaviourally silent.
- `handoff-d1-d2-status.md` unified the picture: every fitted program lived at about 3% activity, where the group is its own tangent space and Hamilton products reduce to adding rotation vectors. The method had been operating as task arithmetic, which is already known not to compose ordered pipelines. **D2** also showed a frozen base could execute a two-stage pipeline it had never seen (`z_ab` at 0.955 exact) when fitted directly, so capacity was not the problem.
- **P0** amplified the fields and watched the commutator wake up on schedule, but composition stayed at 0.0. **P1** refitted everything at full strength: the oracle pipeline sat closer to the identity program than to the Hamilton prediction. **P2** warm-started from the Hamilton guess and found it gave no head start over identity. The falsifier was complete: the product rule is wrong in this gauge, not under-driven.
- **P3** tried transporting a skill increment between contexts; the controls scored 0.0. **A1**, the flagship, showed that the pipeline's correction was keyed to a specific token, and that conjugation could de-key it but not re-address it. **D6** trained learned composition operators on 16 oracle pipelines and both failed on held-out pairs with a flat learning curve.

Four independent falsifications later, zero-shot program-space composition was retired. The substrate claims (expressivity, gentleness, control) were untouched.

A side finding turned out to matter more than the failure. Pipelines that put a reversed payload behind a fixed first token plateaued at about 0.82 exact while payload-first pipelines reached 0.97. **A2** turned this into a ladder and a law: a constant per-sequence field cannot re-index positional attention, and accuracy degrades monotonically with the displacement (0.98, 0.82, 0.445 for offsets 0, 1, 2).

## 4. The positional half of the idea, finally tested (Workstream B — 2026-08-15)

The original idea had always had a second half: the latent program as a parameter of the model's rotary positional machinery. Standard RoPE rotates query and key pairs by position; conjugating the rotary generators, `Ω → R Ω Rᵀ`, reorients the coupling planes without changing the frequencies or breaking relative-position invariance. This needed a rotary base, so Workstream B moved to Pythia-410m and to a rotary twin of the toy.

**B2** measured how much behavioural change each site group buys per unit of activity. The axis field had the lowest raw leverage (it touches few dimensions) but was the only field type whose leverage and whose position-coupling grew with strength: conjugation fields are saturating small-signal actuators; the axis field is a compounding structural one.

**B3** was the causal test. Rerunning A2's offset ladder with the axis field available lifted the offset rungs by +12, +11.5 and +22 points, largest exactly where the disease was worst, and the fitted axis activity rose monotonically with the offset. Giving the axis sites a tenfold learning rate erased the ceiling entirely (k=2 landed at 0.880, the level plain reversal had without an axis field). The axis-only arm nearly solved plain reversal but collapsed whenever a marker token had to be emitted. That settled the **two-actuator law**: conjugation is the content actuator and cannot re-index; the axis field is the position actuator and cannot insert content.

**B4** fitted a French style skill on Pythia-410m and found two things: a style skill ignores the axis sites completely, and the program recovered about 120% of the teacher's gap. At 410m parameters the rotation field beat the fine-tune it was distilled from.

## 5. A belief over one's own program (Phase 4 — 2026-08-15 to 16)

With composition retired, the orchestration layer could only select, swap, slerp and undo. Phase 4 asked whether an agent that maintains a belief over its own program could learn and specialise from a stream of regime-switching tasks.

**v1** built whitened tangent-space Gaussians. Calibration exposed the "dead sea": random rotations in high-capacity groups are behaviourally almost free even at half-turn strength, while fitted skills concentrate exactly there. The codebook experiment found real task-family structure in the fitted library (C-2 confirmed) but random-calibrated geometry could not see it (C-1 failed). The minimal organism showed the predicted differentiation dynamics (potency falls within a regime and rises at switches; a committed prior learns new tasks slower) but lost to a crude reset-on-spike baseline, because a unimodal Gaussian averages evidence across regimes into a compromise program.

**v2** tried a learned behavioural metric and a mixture belief indexed by the codebook's medoids. The learned metric failed out of sample at 27 fields (G1 and G2), which sharpened the campaign's binding rule, Constraint 0: at this scale, no static metric over programs, random-calibrated, geodesic or learned, has survived behavioural validation; decisions must use direct measurement. The mixture agent's memory worked directionally (revisits recovered three times faster than first visits, assignment was family-correct, medoid initialisation was worth everything) but still lost to reset-on-spike, because on the toy a refit from scratch is cheap. The toy organism was retired with the open question stated: belief over programs must pay where refitting is expensive.

## 6. Gate 1: is it worth scaling? (2026-08-16)

Gate 1 was designed to make the scaling decision from days of work on current hardware.

- **Bridge.** The axis-field lift survived a quarter-rotary toy (+5, +10, +21.5), so the claims are not scoped to full-rotary bases. Axis-only collapsed there: on real bases the axis field is an amplifier of conjugation programs, not a standalone actuator.
- **G1-A.** The 120% from B4 was interrogated. Training the LoRA teacher to the program's own budget made it catastrophically overfit (train CE 0.29, held-out 3.96, worse than the base). Against the better 800-step teacher the program recovered 120.6% with a 95% interval of [114, 128]. The program's generalisation gap was a third of its teacher's. The manifold constraint regularises where unconstrained LoRA memorises.
- **G1-B.** Four new skills gave the honest breadth picture: jsonish 104% (the cleanest beats-teacher datapoint, against a teacher that did not overfit), hedge 89%, formal 83%, medical 4%. The failure had the weakest teacher signal and the only out-of-distribution corpus; the regularisation that helps elsewhere is also a ceiling.
- **G1-C.** Programs are dense (only about 28% of sites prunable at the tested thresholds), so the sparse-artifact predictions failed, but the dense fp16 artifact is 462 KB against roughly 10 MB for the LoRA teacher.
- **Report cards.** Every fitted program now emits an audit card with fidelity, gentleness, four certificates (identity, spectrum, Lipschitz, exact undo), a commitment map by actuator, a behavioural fingerprint and fixed limits text. The hedge card is checked in under `registry/`.

## 7. Where it stands now

**Established:** the identity, spectrum and Lipschitz guarantees; substrate expressivity, including novel two-stage pipelines on a frozen base; gentleness at about half the collateral of a fine-tune; per-layer slerp as a control surface; the two-actuator law and the offset-re-indexing law with its axis-field cure; programs beating their teachers at 410m with a regularisation mechanism; a sub-megabyte auditable artifact per skill.

**Retired:** zero-shot program composition by Hamilton products; static program-space metrics as decision procedures at current library scale; the toy belief organism.

**Open:** whether belief over programs pays when refits are expensive; the scale trend beyond 410m; the encoder (E4) which was never run because the campaign turned toward the composition question.

**Proposed next:** Gate 2, a Qwen-class 7B run that fits a six-skill library with cards and benchmarks a probe-router against refit-per-switch and prompt-routing on a regime-switching stream, with adaptation compute metered. The validated verbs going in are select, slerp, swap and undo; stacking skills is not one of them.
