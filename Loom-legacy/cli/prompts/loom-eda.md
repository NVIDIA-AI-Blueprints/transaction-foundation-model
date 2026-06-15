---
description: Profile an ingested data object (read-only) — shape, missingness, target balance, correlations, leakage smells — then narrate the @card.
argument-hint: <a dataset_ref pathspec (+ optional target column)>
---

# /loom-eda — profile a data object (read-only)

Profile a registered Metaflow data object so you (and the user) can write a good
goal/metric and spot problems — missingness, imbalance, **leakage** — before
committing a search budget. Domain-neutral: describe only what the data and the
(optional) declared target actually say.

Run EDA on: $@

## 1. Intake — pin the data object (refuse without a pathspec)
- **Data** — a `dataset_ref` **pathspec** (e.g. `IngestDataset/123`). Take a
  pathspec, never a raw S3 URI or a loose local file. If the user hands a local
  path, route them through `/loom-ingest` (the `loom_ingest` tool) first, then
  profile the resulting `dataset_ref`.
- **Target (optional)** — the believed label column; inferred when omitted.

## 2. Plan — read-only tier
EDA never prompts. Briefly state: "I'll profile `<dataset_ref>` (read-only) —
schema, missingness, target balance, correlations, leakage smells — and hand back
the run + `@card`. Data stays in your Metaflow perimeter."

## 3. Run — call the `loom_eda` tool
Call `loom_eda` with `dataset` (and `target` if given). It runs as a Metaflow run
through the MLOps interface; you never touch Metaflow/AIDE or raw S3 directly.

## 4. Verify — assert lineage
Confirm `status == ok` and that `pathspec` + `card_path` came back. The profile in
`details.summary` is small derived data (schema/flags) — the bulk data stays in
Metaflow. Every figure traces to the run pathspec + the `dataset_ref`.

## 5. Deliver — narrate, return run + summary, flag leakage
- Walk the user through shape; per-column dtypes + % missing; numeric summary;
  **target & balance** (flag severe imbalance, or list candidates if none was
  identifiable); top correlations; and crucially **`summary.leakage_flags`** —
  resolve these *before* `/loom-features`.
- Hand back the Metaflow run + `@card` + the typed summary.
- Suggest goal/metric phrasing for `/loom-run` (as suggestions to confirm, not a
  started search).
- **Next step:** offer `/loom-run` if ready, or flag leakage/imbalance/missingness
  to fix first.

## Composition / exit gate
The `summary.leakage` boolean + `leakage_flags` gate `/loom-features` — a data
object with leakage must be resolved (or its columns dropped via `--from`) before
features are built on it.
