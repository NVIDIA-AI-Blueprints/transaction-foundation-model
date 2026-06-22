---
name: loom-eda
description: Profile a Metaflow data object through Loom — shape, dtypes, missingness, numeric summary, target balance, top feature correlations, and simple leakage smells — then narrate the resulting @card. Read-only. Use when the user points at a registered data object (a dataset_ref pathspec) and asks "what's in here", "profile this data", "is this ready for Loom", "check for leakage", or right before loom-optimize so the goal/metric phrasing is grounded in the real data. Never modifies the data.
when_to_use: "profile a data object, explore a dataset by pathspec, check class balance, check for leakage before features, get suggested goal/metric phrasing"
when_not_to_use: "to register/ingest a source into a data object (get a dataset_ref), use loom-connect first; to spend a search budget against a metric, use loom-optimize."
argument-hint: "<a dataset_ref pathspec (+ optional target column)>"
---

# loom-eda

Profile a data object so a human (and the `loom-optimize` skill) can write a good
goal/metric and spot problems — missingness, imbalance, **leakage** — before
committing a search budget. This is a **read-only** reconnaissance step run
*through Loom's MLOps interface*: it materializes the data object, profiles it,
and hands back a versioned Metaflow run + an `@card`. It never mutates, cleans,
or writes back to the data.

Loom is domain-neutral: do not assume any task type, column meaning, or vertical.
Describe only what the data and the (optional) declared target actually say.

## When to use

- The user points at a **registered data object** (a `dataset_ref` pathspec, e.g.
  `IngestDataset/123`) and asks what's in it / whether it's ready for a run.
- They want a leakage / balance / missingness check before `loom-features` or
  `loom-optimize`.
- Immediately before `loom-optimize`, to ground the plan and metric in real data.

## When NOT to use

- To *register/ingest* a source into a data object (turn a path into a
  `dataset_ref`) — hand off to **`loom-connect`** first.
- To spend a search budget against a metric — hand off to **`loom-optimize`**.

## 1. Intake — pin the data object (refuse if there is no pathspec)

Pin the inputs in the user's own terms and write them back for confirmation:

- **Data** — the **`dataset_ref`**: a Metaflow **pathspec** (e.g.
  `IngestDataset/123`) produced by `loom-connect` / `loom ingest`. Take a
  pathspec, **never a raw S3 URI or a loose local file** as the source of truth.
- **Target (optional)** — the column the user believes is the label/target. When
  omitted, the profile infers one (a train-only column vs the test split, else a
  literal `target`/`label` column) and marks it as inferred.

If the user hands a **local path instead of a pathspec**, do not read the file
directly — route them through `loom-connect` to register it first, then profile
the resulting `dataset_ref`. A read-only profile may proceed with just a
`dataset_ref`.

## 2. Plan — show the plan + tier (read-only)

EDA is the **read-only tier** of the approval matrix (see `CONVENTIONS.md`):
profiling is non-destructive and runs in the data object's own perimeter, so it
**never prompts**. Briefly state the plan before running: "I'll profile
`<dataset_ref>` (read-only) — schema, missingness, target balance, correlations,
and leakage smells — and hand back the run + `@card`." Name the exact
`dataset_ref` that will be read and that the data stays in the user's own Metaflow
perimeter, not exfiltrated.

## 3. Run — call Loom's MLOps INTERFACE (the `loom` CLI), never the backend

Speak only Loom's interface — shell out to the `loom` CLI, which resolves the
MLOps provider by name (default **Metaflow**, swappable by config) and runs the
read-only EDA flow through the interface's `run_flow` seam. **Never call Metaflow
or AIDE directly, and never touch raw S3** — the data object is read only through
the Client API; the datastore is the interface's opaque concern.

```bash
loom eda --dataset <PATHSPEC> [--target <COL>]
```

- The work executes as a **Metaflow run**; the input is the data object by
  `dataset_ref`, read via the Client API.
- Lifecycle flows need the **metaflow** MLOps provider — the `local` dev provider
  cannot run them (it will say so, pointing at `--mlops metaflow`). EDA is not a
  candidate-exec dev task, so run it on metaflow.
- Secrets/endpoints (the Metaflow profile) come from the **environment** only —
  never put them on the command line or in the transcript.

## 4. Verify — assert lineage; large output already stays in Metaflow

- The command returns a **run pathspec** and the **`@card` reference**; confirm it
  reported success. The profile summary is a small *derived* dict (schema/flags) —
  the bulk data stays in Metaflow as the data object's artifacts, never inlined.
- Every figure the profile reports (row count, missingness, balance, correlations,
  leakage flags) traces back to that run's pathspec + the `dataset_ref` it read.

## 5. Deliver — narrate the @card, return run + summary, append a learnings row

- **Narrate the `@card`:** walk the user through the profile — shape; per-column
  dtypes and **% missing**; numeric summary; **target & balance** (flag severe
  imbalance) or that no target was identifiable (list candidates rather than
  guessing); top feature correlations; and crucially the **LEAKAGE flags** (a
  feature near-perfectly predictive of the target, or a duplicate-of-target) with
  the explicit caution to resolve them *before* `loom-features`.
- **Hand back the mandated artifact:** the versioned **Metaflow run + `@card`**
  (the shareable render), plus the typed profile summary the CLI prints.
- **Suggest goal/metric phrasing** the way `loom-optimize` consumes it (e.g.
  "Predict `<target>` from the remaining columns." / "Maximize ROC AUC on a
  held-out split.") — as suggestions to confirm or edit, not a started search.
- **Learnings:** the run appends one `command="eda"` row to the flywheel corpus
  (`learnings/rollouts.jsonl`) — data-object ref · resolved target · leakage flag ·
  run + card pathspecs — sanitized, no raw rows, no secrets. The CLI does this; do
  not hand-write the row.
- **Next step:** offer `loom-optimize` if the data looks ready, or flag what to
  fix first (leakage, imbalance, missingness) before features/optimize.

## Composition — machine-checkable exit gate (executable self-test)

- **Consumes:** a `dataset_ref` pathspec (from `loom-connect`).
- **Exit gate:** the profile's typed summary carries a **`leakage` boolean +
  `leakage_flags`** that gate `loom-features` — a data object with leakage flags
  must be resolved before features are built on it.
- **Self-test:** the leakage gate has an executable self-test on a known-leaky
  fixture — `tests/test_eda.py::test_profile_flags_near_perfect_predictor` and
  `::test_profile_flags_duplicate_of_target` assert the gate **flags** the leak
  (and `::test_profile_id_column_not_flagged_as_duplicate` guards against the
  "fails-open / false-positive" failure mode).

---

## Acceptance test (the bar before this verb joins the pack)

1. **Speaks only the interface** — shells out to `loom eda` (the MLOps interface,
   provider-by-name), never Metaflow/AIDE directly, never raw S3; input is a
   `dataset_ref` pathspec read via the Client API.
2. **Output is a versioned run + `@card`** — not a chat transcript or a loose
   pandas snippet.
3. **Approval tier is correct** — read-only tier, never prompts; no
   `disable-model-invocation` needed.
4. **Writes a learnings row** — the run appends a sanitized `command="eda"` row to
   `learnings/rollouts.jsonl` (the CLI does this every run).
5. **Exit gate has a self-test** — the leakage gate is covered by the
   `tests/test_eda.py` leakage tests above.
6. **Single free-text arg** — one `dataset_ref` (plus an optional target column).
7. **Dual-invocation** — works user-typed (`/loom-eda`) and model-auto-loaded on
   the `description` / `when_to_use` match; safe to auto-fire (read-only).
