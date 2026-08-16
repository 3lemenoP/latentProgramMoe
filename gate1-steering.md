# Gate-1 steering — controls, breadth, bridge, and the Qwen run

**Date: 2026-08-16. Extends `latent-program-fields-report.md`, `phase4-v2-spec.md`, and the v2-MVO results; where this conflicts with earlier docs, this wins. The toy MVO is retired (v2-MVO verdict: memory is representable but cannot be profitable at toy refit costs); its v2.1 fixes are notes, not work items.**

Everything here serves one decision: **worth scaling.** Gate 1 is days on current hardware, two studios in parallel. Gate 2 is one Qwen-class run that is simultaneously the scale datapoint, the belief layer's fair trial, and the product demo. The report-card generator (separate spec, `skill-report-card-spec.md`) must exist before Gate 2's library fitting completes, so every skill ships with its card.

---

## 1. Studio 2 (Pythia-410m)

### G1-A — the B4 controls (interrogate the 120%)

1. **Budget-matched teacher:** train the French LoRA to 2000 steps (the program's budget); recompute gap and recovery against it. This is the confound that decides the headline.
2. **Teacher overfit check:** teacher train-CE vs held-out-CE on French (and same for the program). A positive teacher generalization gap is the "noise to filter" the orbit-projection story requires.
3. **Statistics:** paired bootstrap CI on the program-vs-teacher CE margin over eval sequences; report n. The 0.077-nat margin needs an interval before it appears anywhere.
4. **Mirsky audit on this exact LoRA** (`audit_mirsky.py --base pythia-410m --experts french-lora-merged`): small spectrum drift is the filter story's independent leg.

**Decision rule for the headline:** recovery vs the 2000-step teacher ≥100% with CI excluding <95% → "programs beat their teachers." 90–100% → "parity at ~30× smaller, with guarantees." <90% → the scale-trend claim reverts to "improving with scale"; Gate 2 proceeds regardless (its own recovery gate is 85%).

Registered: controls preserve ≥100% (prior **0.55**); teacher shows a positive train/held-out gap (0.60); this LoRA's median Mirsky ratio ≤ 0.03 (0.60).

### G1-B — skill breadth (n=1 → a table)

Four new skills, budget-matched LoRA teachers (r=8, 2000 steps) from the start: **persona/formal-register**, **JSON/format-structured output**, **domain style** (legal or medical corpus), **safety-posture register** (hedging/refusal *style*). Protocol: conj_only arms suffice per B4, except the **format skill also runs conj+axis** — it is the one plausibly position-flavored skill of the four. Per skill: recovery %, gentleness triple (neutral ppl base/teacher/program), whitened commitment map (behavioral work shares), report card.

Registered: median recovery ≥90% (0.60); all four gentler than their teachers on neutral text (0.70); the format skill shows the highest rope_ax work share of the four (0.55).

Scope note, binding: the safety-posture skill is a register/style skill. It MUST NOT be framed, built, or evaluated as a guardrail-modification capability — bounded modification class is not benign behavior, and its card carries the standard limits text like every other skill.

### G1-C — pruning report (the artifact-size story)

On the French program: activity histogram; prune sites below τ ∈ {1e-4, 3e-4, 1e-3} to identity; re-eval task CE and neutral ppl per τ. Deliverable: the size-vs-recovery curve, with bytes at fp16 and 8-bit sparse encodings. Registered: ≥90% of sites prunable at ≤0.01-nat CE cost (0.60); effective artifact <100 KB at 8-bit sparse (0.55).

## 2. Studio 1 (rotary toy) — the quarter-rotary bridge

As specified in the consolidated report §7.1, unchanged: rotary_pct = 0.25 twin of the NeoX toy (rotary_ndims = 6 ⇒ two 3-blocks per head), the four-arm offset ladder at k ∈ {0, 1, 2}. Gate: best-arm lift ≥5 points on every k ≥ 1 rung. Registered: gate passes (0.60); the lr×10 arm is *necessary* at partial coverage — naive joint fit under-uses rope_ax (0.65). A failed bridge scopes all axis-field claims to full-rotary bases; it does not touch product v1.

## 3. Gate 2 — the Qwen run (scale datapoint + dispatcher benchmark + product demo)

**Base:** a Qwen-class 7B open-weights checkpoint (verify config at runtime; expect GQA — `qk_rel` sites are per **KV** head, n_kv × n3(d_head) per layer; SwiGLU MLP wrapper per main spec §3.2; axis-field sites per the config's rotary dims; RMSNorm needs nothing). Prefer the base (non-instruct) checkpoint for continuity with the campaign; the chat-tuned-geometry question is a recorded open caveat — run one skill on the instruct variant as a probe, report only.

**Part 1 — the library.** Six skills (G1-B's four + French + one more domain), budget-matched distillation, a report card each. Gates: median recovery ≥85%; gentleness — program neutral-ppl ≤ teacher's for ≥5 of 6.

**Part 2 — the dispatcher benchmark.** Regime-switching stream over the six skills plus one never-fitted holdout; regimes recur ≥3×; switch times hidden; labels logged for scoring only. Methods, with **adaptation compute metered in GPU-seconds** and matched accounting:

- oracle-router (labels — ceiling);
- **probe-router**: per episode, evaluate every skill's cached program on a mini-probe from the current stream; route to argmin with a sticky prior. Library immutable — no fusion, no drag (the v2-MVO lesson made structural). Optional arm: probe-router + fork-refine (T steps on a copied program);
- refit-per-switch: spike detector triggers a full 2000-step fit — the honest cost of "just fine-tune again";
- prompt-router: same probe detection, adaptation = swapping a per-skill system prompt (written once per skill) — the fair cheap baseline;
- static base (floor).

Metrics: per-regime task quality; router accuracy vs hidden labels; jump latency (episodes to within 5 points of oracle after a switch); **cost per behavior change**; total adaptation compute. These are the three deck numbers plus their audit trail.

Registered: probe-router within 5 points of oracle mean quality at ≤10⁻³ the adaptation compute of refit-per-switch (**0.60**); probe-router ≥ prompt-router on quality (0.55 — prompts may be strong; the comparison is the point); router accuracy ≥0.9 at regime grain (0.65).

## 4. Sequencing, do-nots, done

**Sequence:** G1-A/B/C and §2 start now, in parallel. §3 Part 1 begins as soon as G1-A's budget-matched protocol is confirmed (its decision rule only calibrates the headline, not the work); Part 2 needs Part 1's library plus the card generator.

**Do-nots:** no static program metric in any decision — routing is measured probe loss only (Constraint 0 carries over). No stacking multiple skills on one request anywhere (unvalidated; validated verbs remain select / slerp / swap / undo). No method comparisons at unmatched adaptation budgets. No further toy-MVO work. No framing of the safety-posture skill beyond register/style.

**Definition of done:** G1-A decision rule executed with CI; four breadth cards; pruning curve; bridge verdict; Qwen library gates checked; dispatcher table with all five methods, three deck numbers highlighted; every registered prediction above scored inline. Combined reading: Gate 1 green + Gate 2 gates met ⇒ the scale decision makes itself; controls deflating to parity changes the headline, not the decision; a failed bridge scopes the axis story and nothing else.
