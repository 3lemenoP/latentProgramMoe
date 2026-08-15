# E1 — expert reconstruction

Studio run 2026-08-13–15. Base `gpt2`, `enable_gains: false`, phase-1
distillation 3000 steps (`configs/e1_experts.yaml`). Teachers: LoRA r=8,
800 steps. Metric: fraction of the expert-vs-base eval CE gap recovered
by the fitted program. Target ≥ 80%. Random-orthogonal control should
recover ~0%.

| task | CE base | CE expert | CE program | gap recovered | control CE | control gap |
|---|---:|---:|---:|---:|---:|---:|
| french | 4.933 | 3.676 | 3.864 | **85.1%** | 18.20 | −10.55 |
| caps | 3.819 | 3.281 | 3.370 | **83.6%** | 16.97 | −24.46 |
| jsonish | 2.289 | 0.400 | 0.407 | **99.6%** | 15.82 | −7.16 |
| sentiment | 5.011 | 4.004 | 4.043 | **96.1%** | 15.21 | −10.13 |

**Decision: E1 pass** on all four tasks with gains off. Programs live in
`runs/e1/z_<task>.pt` on the studio (gitignored).
