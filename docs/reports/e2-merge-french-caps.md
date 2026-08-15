# E2 — geodesic merging: `french` + `caps`

Studio run 2026-08-15. Base `gpt2`; program slerp vs weight-space baselines.

| method | alpha | CE A | CE B | neutral ppl |
|---|---|---|---|---|
| program slerp | 0.0 | 3.8120 | 4.4414 | 121.34 |
| program slerp | 0.25 | 3.9464 | 3.9335 | 69.02 |
| program slerp | 0.5 | 4.2556 | 3.6309 | 52.19 |
| program slerp | 0.75 | 4.6255 | 3.4455 | 44.85 |
| program slerp | 1.0 | 5.0407 | 3.3720 | 43.55 |
| program slerp (early→A) | 0.0 | 3.8120 | 4.4414 | 121.34 |
| program slerp (early→A) | 0.25 | 3.9025 | 4.1855 | 92.14 |
| program slerp (early→A) | 0.5 | 4.1072 | 4.0029 | 75.86 |
| program slerp (early→A) | 0.75 | 4.3854 | 3.8853 | 66.83 |
| program slerp (early→A) | 1.0 | 4.6982 | 3.8320 | 63.50 |
| weight linear | 0.0 | 3.6235 | 4.7467 | 231.93 |
| weight linear | 0.25 | 3.7054 | 3.9222 | 77.85 |
| weight linear | 0.5 | 3.9179 | 3.4913 | 47.93 |
| weight linear | 0.75 | 4.3249 | 3.3203 | 41.74 |
| weight linear | 1.0 | 5.0459 | 3.2844 | 45.51 |
| weight slerp | 0.0 | 3.6235 | 4.7467 | 231.93 |
| weight slerp | 0.25 | 3.6519 | 4.1168 | 112.40 |
| weight slerp | 0.5 | 3.7955 | 3.6013 | 60.84 |
| weight slerp | 0.75 | 4.1815 | 3.3469 | 46.90 |
| weight slerp | 1.0 | 5.0459 | 3.2844 | 45.51 |

Midpoint (α = 0.5): weight linear dominates program slerp on CE A, CE B,
and neutral ppl. Program slerp is a smooth, well-behaved continuum (no
midpoint collapse) but is not a better merge geometry than weight
arithmetic on this pair.

**Decision: E2 does not pass** the “geodesic beats weight-space” claim
for french+caps.
