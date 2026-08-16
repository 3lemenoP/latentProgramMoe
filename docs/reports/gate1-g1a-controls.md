# Gate-1 G1-A — the B4 controls: the 120% interrogated

L4 run 2026-08-16. Fresh 800- and 2000-step French LoRA teachers
(pythia-410m, r=8), fresh conj_only program (2000-step distillation from
the 800-step teacher, B4 protocol). n = 64 eval blocks, paired bootstrap.

## The table (French CE)

| model | train CE | eval CE | generalization gap |
|---|---:|---:|---:|
| base | 3.277 | 3.266 | −0.01 |
| teacher, 800 steps | 1.684 | 2.876 | **+1.19** |
| teacher, 2000 steps (budget-matched) | **0.290** | **3.960** | **+3.67** |
| program (2000-step distill) | 2.371 | **2.796** | **+0.42** |

## What the controls found

1. **The budget-matched control INVERTED rather than deflated.** At the
   program's own budget the LoRA teacher catastrophically overfits —
   train CE 0.29 vs eval 3.96, *worse than the raw base* on held-out
   French. The "undertrained-teacher" confound is refuted in the strongest
   available way: more teacher budget makes the teacher worse.
2. **Recovery vs the (better) 800-step teacher: 120.6%, 95% CI
   [114.0%, 127.8%]** — the interval excludes 100%. Margin teacher−program
   = +0.080 nats, significant. (Recovery "vs the 2000-step teacher" is
   degenerate — its gap is negative — and is reported as such, not used.)
3. **The program is the regularized learner**: its generalization gap
   (0.42) is 3× smaller than its own teacher's (1.19). This is the
   orbit-projection/filter story's independent leg measured directly: the
   rotation manifold cannot represent the memorization component, so
   distillation through it filters to the generalizing part.
4. **Mirsky**: median ratio 0.038 (800-step) / 0.080 (2000-step) — small
   spectrum drift, and the memorizing teacher moves spectra 2× more.
   Both far below the 0.3 gains threshold.
5. Reproducibility: this fresh L4 fit gives 120.6% where the original T4
   fit gave 119.8% — the number survives hardware and re-seeding.

## Registered predictions scored

- controls preserve ≥100% (prior 0.55): **CONFIRMED** (vs the only
  non-degenerate teacher; CI excludes 100%).
- teacher positive train/held-out gap (prior 0.60): **CONFIRMED,
  emphatically** (+1.19 and +3.67).
- Mirsky median ≤ 0.03 (prior 0.60): **narrowly FAILED** (0.038).

## Decision rule → headline

**"Programs beat their teachers"** — upgraded with a mechanism clause: at
matched budget, unconstrained LoRA memorizes while the rotation-manifold
constraint regularizes. The honest fine print: on tasks/budgets where the
teacher does not overfit, the margin may close; the 800-step teacher
(gap +1.19) already overfits substantially, and the program beats it
anyway.
