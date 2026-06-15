---
name: loom-validate
description: Rigorously validate a model/baseline against a data object through Loom — a sealed holdout distinct from a stratified/purged K-fold CV, probability calibration (curve + Brier), per-slice / fairness metrics, and leakage flags — then narrate the @card and its VERDICT. Use when the user asks "is this good enough", "validate this model", "check the calibration / fairness", "can we ship this", or right before loom-deploy to gate promotion on a trustworthy held-out number. Trains/scores only in its own workspace; never mutates the data.
when_to_use: "validate a candidate before promotion, check held-out / CV performance, check calibration, check per-slice / fairness gaps, gate a deploy on a trustworthy metric"
when_not_to_use: "to spend a search budget finding a better solution, use loom-optimize; to profile an unprofiled dataset first, use loom-eda; to ship a validated model, use loom-deploy."
argument-hint: "<a dataset_ref pathspec (+ target, optional prior optimize run, optional sensitive column)>"
---

# loom-validate

Validate a candidate against a data object with the rigor a **promotion decision**
needs, so a human (and `loom-deploy`) can trust the number before shipping. This is
a **planned, gated, lineage-grounded run** — not a loose `cross_val_score` snippet:
it carves a **sealed holdout** that appears in **no** CV fold, runs
stratified/purged **K-fold CV**, measures **probability calibration** (a reliability
curve + the Brier score), computes **per-slice / fairness** metrics when a sensitive
column is given, and surfaces **leakage flags** so an implausibly perfect score is
explained, not trusted. **The metric is the spec.** Stay domain-neutral — never
assume a task type, column meaning, or vertical; reflect back only what the data and
the declared target actually say.

## When to use

- The user points at a `dataset_ref` and asks "is this good enough?", "validate
  this model", "what's the held-out number?", "check calibration / fairness".
- Immediately before `loom-deploy`, to produce the typed `VERDICT` the deploy gate
  asserts.
- To sanity-check a `loom-optimize` winner against a sealed holdout (pass its run
  pathspec via `--solution`).

## When NOT to use

- To *search for a better solution* against a metric — hand off to
  **`loom-optimize`**.
- To *profile an unprofiled dataset* (shape / missingness / balance) — run
  **`loom-eda`** first.
- To *ship* a validated model — hand off to **`loom-deploy`** (which asserts this
  verb's `VERDICT`).

## 1. Intake — pin the spec (refuse without a measurable target)

Pin the inputs in the user's own terms and write them back for confirmation:

- **Data** — the **`dataset_ref`**: a Metaflow **pathspec** (e.g.
  `IngestDataset/123`) produced by `loom-connect` / `loom ingest`. Take a pathspec,
  **never a raw S3 URI or a loose local file** as the source of truth.
- **Target** — the column to evaluate against. **Refuse to start without it** (a
  wrong/missing target silently validates the wrong thing); the flow falls back to
  the data object's recorded schema target only when the user is content with that —
  confirm it.
- **Solution (optional)** — a prior `loom-optimize` run pathspec via `--solution`
  whose best solution to evaluate. Absent → a sensible **gradient-boosted-trees
  baseline** is fit to produce the numbers.
- **Sensitive (optional)** — a column for **per-slice / fairness** metrics via
  `--sensitive` (the holdout metric computed within each of its values).

## 2. Plan — show the plan + tier (workspace-write, light)

Validate is the **workspace-write tier** of the approval matrix (see
`CONVENTIONS.md`): the evaluation is **read-only over the data object** (it never
mutates the data), and it trains/scores a baseline **only in this run's own Metaflow
workspace** — a light, no-prompt workspace-write, not a pure read and not an
expensive/mutate op. Briefly state the plan before running: "I'll validate
`<dataset_ref>` against target `<col>` (workspace-write, own workspace) — sealed
holdout + K-fold CV + calibration + fairness + leakage — and hand back the run +
`@card` and its `VERDICT`." Name the exact `dataset_ref` and that the data stays in
the user's own Metaflow perimeter, not exfiltrated. If the user adjusts the
target/solution/sensitive column, re-state the plan.

## 3. Run — call Loom's MLOps INTERFACE (the `loom` CLI), never the backend

Speak only Loom's interface — shell out to the `loom` CLI, which resolves the MLOps
provider by name (default **Metaflow**, swappable by config) and runs the validation
flow through the interface's `run_flow` seam. **Never call Metaflow or AIDE directly,
and never touch raw S3** — the data object is read only through the Client API; the
datastore is the interface's opaque concern.

```bash
loom validate --dataset <PATHSPEC> [--target <COL>] [--solution <RUN>] [--sensitive <COL>]
```

- The work executes as a **Metaflow run**; the input is the data object by
  `dataset_ref`, read via the Client API.
- Lifecycle flows need the **metaflow** MLOps provider — the `local` dev provider
  cannot run them (it will say so, pointing at `--mlops metaflow`).
- Secrets/endpoints (the Metaflow profile) come from the **environment** only —
  never put them on the command line or in the transcript.

## 4. Verify — assert lineage; the rigor IS the verifier

- The command returns a **run pathspec** and the **`@card` reference**; confirm it
  reported success. The report summary is a small *derived* dict (CV/holdout
  scores, calibration, slice metrics, leakage, verdict) — raw rows never leave
  Metaflow.
- The **sealed holdout** is the lineage guarantee for the headline number: it is
  carved off *before* any CV fold, so the reported holdout score never peeked at the
  rows it scores. Every figure traces back to the run's pathspec + the `dataset_ref`
  it read.

## 5. Deliver — narrate the @card, return run + summary + VERDICT, append a learnings row

- **Narrate the `@card`:** walk the user through the report — the **CV mean ± std**
  and the **sealed-holdout** score in the task's metric (ROC AUC for binary,
  accuracy for multiclass, RMSE for regression); the **calibration** (Brier +
  reliability curve — flag a mis-calibrated model whose probabilities are
  untrustworthy even if its ranking is fine); the **lift table** (decile lift); the
  **per-slice / fairness** gaps when a sensitive column was given (flag a large
  per-group gap); and crucially the **LEAKAGE flags** with the caution to *explain
  an implausibly perfect score before trusting it*.
- **Hand back the mandated artifact:** the versioned **Metaflow run + `@card`**,
  plus the typed report summary the CLI prints, with its **`VERDICT`** line.
- **Learnings:** the run appends one `command="validate"` row to the flywheel corpus
  (`learnings/rollouts.jsonl`) — data-object ref · target · holdout metric · verdict ·
  leakage flag · run + card pathspecs — sanitized, no raw rows, no secrets. The CLI
  does this; do not hand-write the row.
- **Next step:** if `VERDICT: PASS` and the number meets the bar, offer
  `loom-deploy`; if `VERDICT: REVIEW` (leakage present) or the holdout is
  sub-threshold, flag exactly what to fix (resolve the leak, re-engineer features,
  recalibrate) **before** any deploy.

## Composition — machine-checkable exit gate (executable self-test)

- **Consumes:** a `dataset_ref` pathspec (from `loom-connect`/`loom-eda`) and,
  optionally, a `loom-optimize` run pathspec via `--solution`.
- **Exit gate:** the report's typed summary carries a **`verdict`
  (`PASS`/`REVIEW`)** plus the **sealed-holdout score** and **`leakage` boolean** —
  the gate `loom-deploy` asserts. A leaky or sub-threshold validation must **not**
  read `PASS`: leakage forces `REVIEW`, which **blocks** `loom-deploy`.
- **Self-test:** the gate has an executable self-test on a known-leaky fixture —
  `tests/test_validate.py::test_validate_flags_leakage_and_sets_review_verdict`
  asserts a near-perfect predictor is flagged **and** the verdict is forced to
  `REVIEW` (the "fails-open" failure mode is exactly what it guards against), and
  `::test_validate_clean_data_passes` asserts clean data gates to `PASS`.

---

## Acceptance test (the bar before this verb joins the pack)

1. **Speaks only the interface** — shells out to `loom validate` (the MLOps
   interface, provider-by-name), never Metaflow/AIDE directly, never raw S3; input
   is a `dataset_ref` pathspec read via the Client API.
2. **Output is a versioned run + `@card`** — not a chat transcript or a loose
   `cross_val_score` snippet.
3. **Approval tier is correct** — workspace-write tier, light/auto (the read-only
   evaluation runs in its own workspace and does not prompt); no
   `disable-model-invocation` needed.
4. **Writes a learnings row** — the run appends a sanitized `command="validate"`
   row to `learnings/rollouts.jsonl` (the CLI does this every run).
5. **Exit gate has a self-test** — the `VERDICT`/leakage gate is covered by the
   `tests/test_validate.py` leakage/verdict tests above.
6. **Single free-text arg** — one `dataset_ref` (plus an optional target, prior
   run, and sensitive column).
7. **Dual-invocation** — works user-typed (`/loom-validate`) and model-auto-loaded
   on the `description` / `when_to_use` match; safe to auto-fire (own-workspace,
   non-destructive).
