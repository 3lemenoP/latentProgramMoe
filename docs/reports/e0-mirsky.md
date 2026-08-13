# E0 — Mirsky audit

Base: `gpt2`

Studio run 2026-08-13. Experts: LoRA r=8, 800 steps, tasks french/caps/jsonish/sentiment.
Overall median ratio **0.028** (threshold 0.3) → **gains OFF**.

## `experts/caps/merged`

- matrices changed: 48
- median ratio: **0.026**   mean: 0.029

| matrix type | count | median ratio |
|---|---|---|
| c_attn | 12 | 0.023 |
| c_fc | 12 | 0.028 |
| c_proj | 24 | 0.030 |

worst 10 (most rotation-unreachable):

| name | layer | ‖ΔW‖_F | Mirsky lower | ratio |
|---|---|---|---|---|
| transformer.h.11.attn.c_proj.weight | 11 | 2.0144 | 0.1318 | 0.065 |
| transformer.h.2.attn.c_proj.weight | 2 | 1.6011 | 0.0624 | 0.039 |
| transformer.h.4.attn.c_proj.weight | 4 | 2.1069 | 0.0798 | 0.038 |
| transformer.h.10.mlp.c_fc.weight | 10 | 9.0867 | 0.3400 | 0.037 |
| transformer.h.10.attn.c_proj.weight | 10 | 2.7032 | 0.1007 | 0.037 |
| transformer.h.0.attn.c_proj.weight | 0 | 1.3889 | 0.0517 | 0.037 |
| transformer.h.9.attn.c_proj.weight | 9 | 2.1099 | 0.0779 | 0.037 |
| transformer.h.5.attn.c_proj.weight | 5 | 1.7928 | 0.0660 | 0.037 |
| transformer.h.6.attn.c_proj.weight | 6 | 2.2584 | 0.0829 | 0.037 |
| transformer.h.3.attn.c_proj.weight | 3 | 1.9378 | 0.0690 | 0.036 |

## `experts/french/merged`

- matrices changed: 48
- median ratio: **0.033**   mean: 0.033

| matrix type | count | median ratio |
|---|---|---|
| c_attn | 12 | 0.025 |
| c_fc | 12 | 0.038 |
| c_proj | 24 | 0.033 |

worst 10 (most rotation-unreachable):

| name | layer | ‖ΔW‖_F | Mirsky lower | ratio |
|---|---|---|---|---|
| transformer.h.11.attn.c_proj.weight | 11 | 4.3615 | 0.3061 | 0.070 |
| transformer.h.10.attn.c_proj.weight | 10 | 4.1405 | 0.2336 | 0.056 |
| transformer.h.10.mlp.c_fc.weight | 10 | 8.3546 | 0.3638 | 0.044 |
| transformer.h.7.mlp.c_proj.weight | 7 | 4.5883 | 0.1973 | 0.043 |
| transformer.h.2.attn.c_proj.weight | 2 | 2.1969 | 0.0912 | 0.041 |
| transformer.h.7.mlp.c_fc.weight | 7 | 8.7726 | 0.3632 | 0.041 |
| transformer.h.4.attn.c_proj.weight | 4 | 3.1112 | 0.1266 | 0.041 |
| transformer.h.8.mlp.c_fc.weight | 8 | 9.2784 | 0.3727 | 0.040 |
| transformer.h.3.attn.c_proj.weight | 3 | 2.6125 | 0.1047 | 0.040 |
| transformer.h.0.attn.c_proj.weight | 0 | 1.5739 | 0.0630 | 0.040 |

## `experts/jsonish/merged`

- matrices changed: 48
- median ratio: **0.024**   mean: 0.027

| matrix type | count | median ratio |
|---|---|---|
| c_attn | 12 | 0.022 |
| c_fc | 12 | 0.028 |
| c_proj | 24 | 0.035 |

worst 10 (most rotation-unreachable):

| name | layer | ‖ΔW‖_F | Mirsky lower | ratio |
|---|---|---|---|---|
| transformer.h.0.attn.c_proj.weight | 0 | 1.3382 | 0.0601 | 0.045 |
| transformer.h.9.attn.c_proj.weight | 9 | 1.8305 | 0.0742 | 0.041 |
| transformer.h.10.attn.c_proj.weight | 10 | 2.0821 | 0.0842 | 0.040 |
| transformer.h.11.attn.c_proj.weight | 11 | 2.1795 | 0.0840 | 0.039 |
| transformer.h.7.attn.c_proj.weight | 7 | 1.4478 | 0.0539 | 0.037 |
| transformer.h.6.attn.c_proj.weight | 6 | 1.1299 | 0.0420 | 0.037 |
| transformer.h.8.attn.c_proj.weight | 8 | 1.6875 | 0.0624 | 0.037 |
| transformer.h.9.mlp.c_proj.weight | 9 | 3.2731 | 0.1203 | 0.037 |
| transformer.h.1.attn.c_proj.weight | 1 | 0.8589 | 0.0311 | 0.036 |
| transformer.h.2.attn.c_proj.weight | 2 | 0.7870 | 0.0284 | 0.036 |

## `experts/sentiment/merged`

- matrices changed: 48
- median ratio: **0.032**   mean: 0.033

| matrix type | count | median ratio |
|---|---|---|
| c_attn | 12 | 0.026 |
| c_fc | 12 | 0.040 |
| c_proj | 24 | 0.034 |

worst 10 (most rotation-unreachable):

| name | layer | ‖ΔW‖_F | Mirsky lower | ratio |
|---|---|---|---|---|
| transformer.h.11.attn.c_proj.weight | 11 | 2.4384 | 0.1464 | 0.060 |
| transformer.h.10.mlp.c_fc.weight | 10 | 13.0982 | 0.7209 | 0.055 |
| transformer.h.11.mlp.c_fc.weight | 11 | 12.0603 | 0.6230 | 0.052 |
| transformer.h.9.mlp.c_fc.weight | 9 | 11.6289 | 0.5965 | 0.051 |
| transformer.h.7.mlp.c_fc.weight | 7 | 10.2791 | 0.5024 | 0.049 |
| transformer.h.8.mlp.c_fc.weight | 8 | 10.9616 | 0.5153 | 0.047 |
| transformer.h.6.mlp.c_proj.weight | 6 | 3.3998 | 0.1411 | 0.041 |
| transformer.h.2.attn.c_attn.weight | 2 | 3.3798 | 0.1374 | 0.041 |
| transformer.h.0.attn.c_proj.weight | 0 | 1.5970 | 0.0646 | 0.040 |
| transformer.h.3.mlp.c_fc.weight | 3 | 7.6712 | 0.3086 | 0.040 |

---

**Overall median ratio: 0.028**  (threshold 0.3)

**Decision: gains OFF (pure conjugation suffices)**
