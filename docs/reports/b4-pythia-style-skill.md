# B4 (Q3) — natural-language style skill on Pythia-410m, ± axis field

Studio 2 run 2026-08-15. French style expert: LoRA r=8 on pythia-410m
(800 steps), CE on held-out french text 3.266 → 2.877 (**gap 0.389 nats**).
Program fits: E1 protocol (distill KL + hidden-MSE + λ_reg 1e-4), 2000
steps, three arms mirroring B3's design.

## Results

| arm | gap recovered | program CE | rope_ax act. | qk_rel act. | attn_io | ffn_hidden |
|---|---:|---:|---:|---:|---:|---:|
| conj_only (rope_ax frozen) | **119.8%** | 2.800 | 0 | 0.0058 | 0.0052 | 0.0035 |
| conj + axis | 119.5% | 2.802 | 0.0034 | 0.0056 | 0.0052 | 0.0035 |
| conj + axis, lr×10 | 119.7% | 2.801 | 0.0064 | 0.0055 | — | — |

## Findings

1. **The scale headline: the program BEATS its teacher.** All arms recover
   ~120% of the expert gap — program CE 2.800 vs the LoRA expert's 2.877.
   On GPT-2 (E1) programs trailed their experts by ~5–16%; at 410m the
   distilled rotation field surpasses the fine-tune it imitates
   (KL-to-teacher + hidden anchors act as a regularizer over the teacher).
   E1-recovery-vs-scale now has two points and the trend is UP with scale.
2. **Actuator specificity confirmed from the other side.** A style skill is
   a pure content-actuator skill: the axis field contributes nothing
   (Δrecovery ≈ −0.3% to −0.1%, noise), the skill allocates rope_ax only
   0.0034 activity when free, and even forcing it (lr×10 doubles rope_ax to
   0.0064) changes recovery by nothing. Exactly the complement of B3, where
   the *positional* task demanded the axis field in proportion to offset.
   The two-actuator picture holds in both directions.
3. Activity lands in `qk_rel` ≈ `attn_io` > `ffn_hidden` on Pythia — a
   different landing profile than the toy fits (which loaded ffn_hidden),
   consistent with B2's per-base leverage differences.

**Prediction B4 (implicit in steering §B4): "does a natural-language skill
use the axis sites?" — answer: no, and it costs nothing to offer them.**
