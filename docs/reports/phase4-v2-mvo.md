# Phase 4 v2 — the IMM agent (mvo_v2), scored

L4 runs 2026-08-16, e3-base recur3 stream (6 tasks × ≥3 regimes each;
holdout rev_pad2 never direct-fit), matched per-episode compute (25 refine
steps), codebook k=4 behavioral medoids {bc, d, ba, e} as components.

## The table

| method | mean acc | first-visit recovery | revisit recovery |
|---|---:|---:|---:|
| oracle | 0.891 | — | — |
| SGD tracker | 0.551 | 57.8 | 54.9 |
| reset-on-spike | 0.526 | 42.0 | 46.5 |
| v1 agent (MAP-act) | 0.450 | 61.4 | 152.5 |
| **IMM (medoid init)** | 0.371 | 142.2 | 48.7 |
| IMM, novelty-gated control | 0.113 | 357.4 | 183.4 |
| IMM (random init) | 0.030 | 468.0 | 183.4 |
| static | 0.075 | 468.0 | 183.4 |

## Predictions scored

- **P-1 (IMM > reset on mean acc, prior 0.65): FAILED** (0.371 vs 0.526).
- **P-2 (revisit recovery < 10, prior 0.70): FAILED** (48.7) — though the
  IMM is the only learner whose revisit column (48.7) beats its own
  first-visit column (142.2) by ~3×: the memory *works directionally*,
  the components just aren't good enough to convert it.
- **P-3 (first-visit ≈ reset ±20%, prior 0.75): FAILED** (142.2 vs 42.0) —
  a novel regime forces a mediocre family component or slow novelty.
- **P-4 (responsibility ARI > 0.8, prior 0.70): FAILED at task grain
  (0.57), but the assignment is family-correct**: late-run winners map
  `ba` → the whole displacement family (b_then_a, c_then_b, rev_pad2),
  `born_t134` → both marker atoms, exactly the codebook's C-2 partition.
  ARI degrades 0.77 → 0.57 as winner-drag blurs components.
- **P-5 (per-component canalization + novelty totipotency, prior 0.75):
  PARTIAL** — active components canalize hard (ba potency −5.7k → −10.8k),
  unused ones dedifferentiate under drift (d/e rise), but novelty
  totipotency FAILS: below-τ soft fusion leaks evidence into it and it
  canalizes (−0.9k → −5.3k). v2.1 fix: exclude novelty from soft fusion.
- **P-6 (medoid init > random init, prior 0.60): CONFIRMED, decisively** —
  early ARI 0.77 vs 0.22; random-init IMM lands at the static floor
  (0.030). The codebook is worth ~everything the IMM has.

## The gated control (diagnosis, not a fix)

Enforcing spec §3.2's sentence directly (novelty wins when all named fit
poorly) produced a **birth runaway** (27 components, births every ~10
episodes, acc 0.113): at 25 refine-steps/episode a newborn needs ~20+
episodes to mature, and the gate re-fires before it can, spawning
duplicates. Ungated = drag + birth starvation; gated = runaway. The
missing mechanism is an adolescence period (newborns exempt from gate
competition while their loss falls) — noted for v2.1, not run.

## The deepest reading

Reset-on-spike keeps winning for a structural reason the campaign should
own: **on this toy, refitting from scratch is cheap** (~40 episodes to
oracle-tolerance), so memory has little to buy — the revisit economy only
pays when refit cost is high. The IMM's directional signature (revisit 3×
faster than first-visit, family-correct assignment, medoid init worth 25×
over random) shows the architecture functions; the environment prices its
advantage at near zero. The decisive v3 experiment is an environment where
refit is expensive (larger base, harder tasks, smaller per-episode budget)
— that is where a codebook-as-memory must either pay or be retired.

## Claims-register update (one paragraph)

Established, unchanged: substrate expressivity, gentleness, control, the
two-actuator law (content/position), the offset-re-indexing law and its
axis-field cure, program-beats-teacher at 410m. Newly established: regime
memory in program space is *representable* (family-correct IMM assignment;
medoid init decisive) but *not yet profitable* (every belief architecture
tested loses to spike-triggered refitting at toy refit costs); no static
metric over programs — random-calibrated, geodesic, or learned — has
survived out-of-sample behavioral validation at n=27 (three failures:
C-1, G1, G2), making direct behavioral measurement the only validated
decision procedure over programs. Retired: zero-shot composition (v1),
static-metric program geometry at current library scale (v2). Open: the
refit-cost frontier — whether belief-over-programs pays when programs are
expensive to refit.
