---
description: Rigorously validate a baseline/solution against a data object — sealed holdout vs purged K-fold CV, calibration, fairness, leakage — and emit the VERDICT deploy asserts.
argument-hint: <a dataset_ref pathspec (+ target, optional --solution run, optional --sensitive column)>
---

# /loom-validate — rigorous validation with a trustworthy VERDICT (read-only)

Validate a candidate against a data object with the rigor a **promotion decision**
needs, so the user (and `/loom-deploy`) can trust the number before shipping. Not a
loose `cross_val_score`: a sealed holdout in **no** CV fold, stratified/purged
K-fold CV, probability calibration (reliability curve + Brier), per-slice/fairness
metrics, and leakage flags so an implausibly perfect score is explained, not
trusted. The metric is the spec.

Validate: $@

## 1. Intake — pin the spec (refuse without a target)
- **Data** — a `dataset_ref` pathspec. Never a raw S3 URI / loose file.
- **Target** — the column to evaluate against. **Refuse without it** (a
  wrong/missing target silently validates the wrong thing); only fall back to the
  schema target if the user confirms.
- **Solution (optional)** — a prior `loom_run` run pathspec via `solution`; absent
  → a gradient-boosted-trees baseline is fit to produce the numbers.
- **Sensitive (optional)** — a column for per-slice / fairness metrics.

## 2. Plan — read-only tier
The evaluation reads the data object read-only and trains/scores a baseline only in
this run's own workspace; it never prompts. State: "I'll validate `<dataset_ref>`
against target `<col>` — sealed holdout + K-fold CV + calibration + fairness +
leakage — and hand back the run + `@card` + `VERDICT`."

## 3. Run — call the `loom_validate` tool
Call `loom_validate` with `dataset` (+ `target`, `solution`, `sensitive`). It runs
as a Metaflow run through the MLOps interface.

## 4. Verify — the rigor IS the verifier
Confirm `status` and read the **run pathspec** + `card_path`. The report summary is
small derived data (CV/holdout scores, calibration, slice metrics, leakage,
verdict); raw rows never leave Metaflow. The **sealed holdout** is the lineage
guarantee for the headline number.

## 5. Deliver — narrate, return run + summary + VERDICT
- Walk through the **CV mean ± std** and **sealed-holdout** score; **calibration**
  (flag a mis-calibrated model); the **lift** table; **per-slice/fairness** gaps;
  and crucially the **leakage flags** (explain an implausibly perfect score before
  trusting it).
- Hand back the run + `@card` + typed summary with its **`VERDICT`** line.
- **Next step:** if `VERDICT == PASS` and the number meets the bar, offer
  `/loom-deploy`; if `REVIEW` (leakage) or sub-threshold, flag exactly what to fix
  **before** any deploy.

## Composition / exit gate
Produces a `VERDICT` (`PASS`/`REVIEW`) + the sealed-holdout score + a `leakage`
boolean — the gate `/loom-deploy` asserts. A leaky/sub-threshold validation must
not read `PASS`; leakage forces `REVIEW`, which **blocks** deploy.
