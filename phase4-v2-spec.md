# Phase 4 v2 specification — learned behavioral metric, mixture beliefs, and the codebook-as-memory

**Version 2.0, 2026-08-15. Supersedes `phase4-belief-differentiation-spec.md` (v1) where they conflict; v1's kernels, environment harness, baselines, and tests T16–T21 carry over. Grounded in the v1 results (`phase4-w0-c-d.md`): C-2/C-3/D-2/D-4 confirmed; C-1 and D-1 failed informatively. Target implementer: Claude Code.**

---

## 0. The binding law, and what v2 changes

Every failure in this program — Hamilton composition, random-leverage whitening, linear-Gaussian fusion — has been one failure: **letting the manifold's intrinsic structure stand in for behavioral structure without measuring the map.** Every success measured behavior directly. v2 elevates this to a design constraint:

> **Constraint 0.** No decision-bearing quantity may be manifold-native. Distances that matter are learned from behavior (§1). Discrete decisions (which regime, which component) are made from measured loss, never from geometric likelihood (§3). Manifold-native quantities (raw d_geo, random-whitened norms) may appear in reports as diagnostics only.

Two v1 findings drive the design. **The dead sea** (W0): random rotations in high-capacity groups are behaviorally near-free even at half-turn strength, while fitted skills concentrate exactly there — skills live on a low-dimensional live manifold inside a behaviorally null bulk, so random-probe calibration prices the wrong subspace (this is why C-1 failed). **The structural mismatch** (D-1): a unimodal Gaussian filtering a regime-*switching* environment accumulates cross-regime evidence into a compromise program; the environment has a discrete latent the belief lacked (this is why reset-on-spike — the degenerate one-component mixture with forgetting — beat the agent).

v2 therefore has three parts: a **learned behavioral metric** (§1), three **reanalyses from existing artifacts** (§2), and a **mixture belief** whose components are the codebook's cell types (§3).

---

## 1. The learned behavioral metric (M-metric)

### 1.1 Construction

Inputs already on disk: N = 28 fitted fields (tangent-mapped via q_log) and their N×N symmetrized-KL matrix on the fixed, versioned 200-probe set.

1. **Live subspace:** PCA the N tangent vectors to r ≤ N−1 components (keep 95% variance). This is the empirical skill subspace — the live manifold's local span.
2. **Metric fit:** learn a diagonal weight vector w in PCA coordinates such that weighted distances match behavioral distances: minimize Σ_{ij} ( d_w(i,j)² − symKL(i,j) )² with nonnegativity and ridge regularization. (Diagonal-in-PCA is the right capacity for N = 28; full or low-rank M is v2.1, gated on library growth.)
3. **The hybrid rule for out-of-subspace mass:** directions orthogonal to the live subspace get the operating-strength *random-calibrated* group weights from v1's W0. This is now principled rather than a fallback: random calibration was wrong for skills precisely because skills live in the subspace; for the dead sea it is the correct price. One metric, two regimes, each measured where it is valid.
4. Emit `metric.json` (PCA basis, w, dead-sea weights, probe-set version). The metric refreshes whenever the library grows by ≥25%; distances are never compared across metric versions.

### 1.2 Gates and registered predictions

- **G1 (validity):** leave-one-out — refit the metric without field i, check d_w(i, ·) against symKL(i, ·); LOO stress must beat both raw-geodesic and random-whitened baselines. Test **T22**.
- **G2 (the C-1 rematch):** re-run Experiment C's clustering under M. Registered: **ARI_M > ARI_raw** (prior 0.70) and **ARI_M > ARI_random-whitened** (prior 0.80). C-1's v1 failure stands as scored; this is a new, differently-constructed prediction.
- **T23:** hybrid consistency — dead-sea directions price at W0 weights; live-subspace round-trip through PCA is lossless for the library fields.

### 1.3 Commitment maps (B4 amendment, binding)

All commitment maps and potency reports are stated in **behavioral work shares**, not raw activity: v2 default = leverage × activity per site group (operating-strength L_g); v2.1 = M-norms once G1/G2 are green. Raw activity may appear as a diagnostic column. (Motivating case: B4's raw table hid that qk_rel does 2–4× the behavioral work of the other groups.)

---

## 2. Reanalyses from existing artifacts (run first; no training)

- **R1 — behavioral rate–distortion.** Recompute the covering-radius-vs-k curve in symKL (the matrix exists). v1's "no elbow" verdict was computed in the metric C-1 falsified and is hereby reopened. Registered: **elbow at k ≈ 4**, matching C-2's family structure (prior 0.60).
- **R2 — revisit-split recovery.** From `runs/mvo/report_mvo.json`: split every recovery event into first-visit vs revisit of the regime, per method. Registered: the v1 agent shows **no revisit advantage** (the compromise posterior erases regime memory; prior 0.80); reset-on-spike shows none by construction. This figure is the IMM's motivation and its benchmark: a memory architecture must show its entire edge in the revisit column.
- **R3 — D-3 scoring.** Score the deferred D-3 from the logged commitment maps, in behavioral work shares per §1.3: converged agent maps vs direct-fit skill maps, cosine > 0.8 (re-registered, prior 0.75).

---

## 3. The mixture belief (IMM agent) — the codebook as memory

### 3.1 Structure

Belief = K components + 1 novelty component. Each component k is a v1-style per-site Gaussian **in raw rotation-vector coordinates with operating-strength group scaling** (v1 engineering lessons are binding: priors capped at raw θ ≤ 0.3 per site; all refinement in raw geometry). The M-metric of §1 governs initialization, diagnostics, and clustering — **never** the discrete regime decision, which is behavioral by Constraint 0:

- **Responsibilities are measured, not geometric.** Each episode, evaluate every component's MAP program on a mini-probe of the current train batch (K ≤ 6 keeps this cheap); responsibility ∝ exp(−β·loss_k) × sticky transition prior (stay-probability p_stay). This is an IMM filter whose observation model is the task itself.
- **Act:** deploy the winning component's MAP (MAP-act is the v2 default; within-component Thompson behind a flag — v1 measured the sampling cost at ~0.02 and it bought nothing).
- **Refine:** T gradient steps from the winner's MAP on the train batch, raw geometry.
- **Fuse into the winner only** (hard assignment when top responsibility > τ = 0.7; soft-weighted fusion below τ). Cross-component fusion is prohibited — it is exactly the v1 pathology reinstated (**T27** enforces isolation).
- **Drift:** mild Q-inflation on all components; λ-forgetting *within* the winner behind a flag.

### 3.2 Initialization, birth, and lineage

Components initialize from **C-2's four behavioral medoids** (their fitted fields as means; σ_init = 0.05 rad per site), plus one broad novelty component (σ = 0.3, capped). The novelty path is the de novo mechanism: when all named components' probe losses are high, novelty wins, refinement starts from its broad prior, and a **birth rule** converts it into a new named component once it has held responsibility for T_birth = 10 consecutive episodes with falling loss — then a fresh novelty component respawns (**T26**: exactly one birth per novel regime, no duplicates). Potency is reported **per component**: named components canalize while novelty stays totipotent — the lineage picture becomes a plotted object.

### 3.3 Environment and baselines

v1 harness carries over with two changes: the schedule guarantees each task recurs ≥ 3 times (revisits are where memory must pay), and one never-fitted holdout task is included for the birth test. True regime labels are logged, hidden from the agent, used only for scoring. Baselines at matched per-episode compute: oracle, reset-on-spike (the null to beat), SGD tracker, static, v1 single-Gaussian agent (now a baseline), and one ablation — **random-init IMM** (medoids replaced by random draws) to measure what the codebook initialization is actually worth.

### 3.4 Registered predictions

- **P-1** IMM beats reset-on-spike on mean accuracy (prior 0.65).
- **P-2** revisit recovery < 10 episodes (component jump), vs reset's ~41 (prior 0.70).
- **P-3** first-visit recovery ≈ reset-on-spike ± 20% — no free lunch on genuinely novel regimes (prior 0.75).
- **P-4** responsibility-to-regime alignment ARI > 0.8 against hidden labels (prior 0.70).
- **P-5** per-component canalization + persistent novelty totipotency; D-2/D-4 observables reproduce at component level (prior 0.75).
- **P-6** C-medoid init reaches P-4 alignment faster than random-init IMM (prior 0.60).

## 4. Code plan and tests

```
lpm/metric.py            # fit/load M: PCA basis, diagonal fit to symKL, hybrid dead-sea weights (T22, T23)
lpm/mixture_belief.py    # IMMBelief: components, behavioral responsibilities, winner-fusion,
                         #   birth rule, per-component potency/commitment (T24–T27)
scripts/reanalyze_mvo.py # R2 revisit split + R3 D-3 scoring (existing logs)
scripts/codebook.py      # + behavioral rate–distortion (R1); C rematch under M (G2)
scripts/mvo_v2.py        # §3 agent + baselines + ablation, matched compute
```

Tests: **T22** metric LOO gate; **T23** hybrid-metric consistency; **T24** stationary sanity (one component takes responsibility ≥ 0.9, others' means untouched); **T25** scripted A→B→A stream: revisit recovery < first-visit; **T26** single-birth rule; **T27** fusion isolation.

## 5. Do-not list

- No manifold-native quantity in any decision (Constraint 0); geometric numbers are report columns.
- No cross-component fusion, ever.
- No Thompson-act by default; MAP-act unless the flag is deliberately set.
- No comparing distances across metric versions; `metric.json` is versioned with its probe set.
- No skipping R1–R3: the reanalyses are ordered before any v2 training because two of them re-score standing conclusions from data already on disk.

## 6. Definition of done

R1–R3 reports with predictions scored; metric fitted with G1/G2 verdicts; mvo_v2 table across six methods + ablation with P-1..P-6 scored inline; per-component potency and commitment trajectories (behavioral work shares) plotted; one paragraph updating the standing-claims register. Same report format as the campaign; every claim lands next to its prior.
