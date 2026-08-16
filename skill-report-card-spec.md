# Skill report card — generator specification

**Version 0.1, 2026-08-16. Target: Claude Code. The card is the product's differentiator: every fitted program emits an auto-generated audit card as a side effect of fitting. It wraps instruments that already exist in the repo — nothing here invents a new measurement.**

---

## 1. Inputs

`make_card.py --base <ckpt-or-hash> --program <field.pt> [--teacher <ckpt>] --evals <evals.yaml> --leverage <whitening.json> --out <dir>`

- Base checkpoint (hash recorded), program file (`ProgramField`), optional teacher checkpoint.
- `evals.yaml`: versioned pointers to the task eval set, the neutral corpus, and the standard probe suite. Cards are only comparable within an eval version.
- Operating-strength leverage table for this base (W0 protocol output) — required for behavioral work shares.

## 2. The card (card.json + rendered card.md/html)

1. **Identity.** Skill name, semver, base hash, program hash, created-at, fit-protocol hash, seeds; sizes: quaternion count, raw params, bytes fp16, bytes 8-bit-sparse (if pruning was run, the pruned variant's numbers alongside).
2. **Fidelity.** Task metric for base / teacher / program; recovery % with paired-bootstrap CI and n. Teacherless skills report absolute task metrics only — the card never fabricates a recovery number.
3. **Gentleness.** Neutral-corpus perplexity base / teacher / program; collateral ratio (program−base)/(teacher−base) with CI. This is the E2 measurement productized.
4. **Certificates** (the guarantee block — each a named check with pass/fail and tolerance):
   - *Identity check*: with the identity program installed through the same runtime path, max |Δlogits| vs raw base < tol — proves the serving wiring, per test T4.
   - *Spectrum certificate*: structural — verify every stored quaternion is unit-norm and its rotation orthogonal within tol (this **is** the spectrum guarantee, exactly), plus full-SVD spot-checks on 3 randomly sampled effective matrices as belt-and-braces. Cheap at any scale because the guarantee is by construction, not by exhaustive measurement.
   - *Lipschitz statement*: program-independent bound holds by the same construction; card cites the test hash.
   - *Exact-undo attestation*: apply program, apply its sitewise conjugate, max |Δlogits| vs base < tol. This line is the erasure/compliance artifact.
5. **Commitment map.** Behavioral work shares (leverage × activity) per site group × layer band, rendered as a heatmap; top-k sites; **actuator classification** — content / position / composite — by rope_ax work share against declared thresholds. Raw activity appears as a diagnostic column only.
6. **Activity statistics.** Mean/max activity, histogram, prunability curve when available.
7. **Behavioral fingerprint.** The probe-suite loss vector ("the metric is a benchmark suite" doctrine, the only decision procedure that survived validation). Enables registry-side routing, duplicate detection, and nearest-neighbor display: sym-KL to existing library entries, top-3 neighbors named.
8. **Limits.** Fixed text, non-editable: the card certifies *stability, size, scope of modification, and reversibility*. It does **not** certify content alignment, factuality, or safety of outputs — bounded modification class is not benign behavior. Lists the unvalidated verbs (no multi-skill stacking).
9. **Provenance.** Teacher hash, data description, steps, hardware, wall-clock, library/metric/eval versions.

## 3. Implementation

```
lpm/report_card.py    # compute all blocks from inputs; pure functions over existing instruments
lpm/card_render.py    # card.json -> card.md and a one-page HTML
scripts/make_card.py  # CLI; also invoked automatically at the end of fit_expert
registry/{skill}/{version}/{program.pt, card.json, card.md}   # immutable once written
```

Compute notes: the heavy items are the eval passes (fidelity, gentleness, fingerprint) — batchable, and shared probe caches mean a library-wide card refresh reuses base/teacher numbers. Certificates are near-free (structural checks + 3 SVDs). Cards are deterministic given inputs (fixed seeds for probe order); `card.json` carries its own schema semver.

## 4. Tests

- **TC-1** identity program → card renders, certificates all green, fidelity block marked N/A, undo attestation trivially passes.
- **TC-2** tampered program (one non-unit quaternion) → spectrum certificate fails, card renders with a red block; exit nonzero.
- **TC-3** determinism: same inputs → byte-identical card.json.
- **TC-4** fingerprint sanity: French-v2's nearest neighbor is French-v1, and its sym-KL to the JSON skill exceeds it by ≥5×.
- **TC-5** work-share sanity: on the B4 French program, qk_rel's behavioral work share exceeds ffn_hidden's (reproduces the known result the raw table hid).

## 5. Definition of done

`fit_expert` emits a card automatically; the four G1-B skills and the six Gate-2 library skills each ship with one; a rendered example card is checked into the repo; TC-1..TC-5 green. From that point, every experimental run produces sales collateral as a side effect, and every registry entry is auditable by construction.
