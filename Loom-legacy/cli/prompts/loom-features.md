---
description: Build engineered features into a NEW versioned data object — domain-neutral transforms; drops leakage columns flagged by eda via --from.
argument-hint: <a dataset_ref pathspec (+ optional target, --from eda-run, --recipe)>
---

# /loom-features — build engineered features (workspace-write)

Build engineered features from a data object and hand back a **new, versioned data
object** the rest of the lifecycle can consume — not a notebook cell that
evaporates. Domain-neutral transforms only (numeric scaling/interactions,
categorical encoding, datetime parts, a simple group-mean aggregation); preserve
the target untouched.

Build features for: $@

## 1. Intake — pin the data object (refuse without a pathspec)
- **Data** — a `dataset_ref` pathspec (`IngestDataset/123`, or a `FeaturesFlow/<id>`
  to chain). Never a raw S3 URI or loose file.
- **Target (optional)** — the column to preserve untouched; inferred when omitted.
- **`--from` (optional)** — an upstream `loom_eda` run pathspec; its
  leakage-flagged columns are **dropped** before building (the `eda → features`
  edge). **Gate-assert:** if a prior EDA flagged leakage, pass that run via `from`.
- **`--recipe` (optional)** — `minimal` (scaling + encoding) or `full` (default).

## 2. Plan — workspace-write, light
Reads the source read-only and writes a **new** data object only into this run's
own workspace (network off). State: "I'll build features from `<dataset_ref>` —
scaling, encoding, datetime parts, interactions — preserving target `<col>`,
dropping any leakage columns from `<eda-run>`, and hand back the NEW
`FeaturesFlow/<id>` + `@card`."

## 3. Run — call the `loom_features` tool
Call `loom_features` with `dataset` (+ `target`, `from`, `recipe`). It runs as a
Metaflow run; the engineered artifacts ARE the new data object.

## 4. Verify — assert lineage
Confirm `status == ok`, read the **new pathspec** (`FeaturesFlow/<id>`) and
`card_path`. The build summary is small derived data (feature counts, dropped
columns, null/variance stats); engineered rows stay in Metaflow.

## 5. Deliver — narrate, return the new data object
- Walk the user through before → after feature count and the engineered families;
  null %/variance (flag constant/all-null columns); and the **leakage handling** —
  exactly which columns were dropped (or that none were).
- Hand back the NEW `FeaturesFlow/<id>` + `@card` + the typed summary (`VERDICT: BUILT`).
- **Next step:** offer `/loom-pipeline` / `/loom-run` / `/loom-validate` against the
  new data object.

## Composition / exit gate
Produces a new `dataset_ref` + `fingerprint` + `refused_leakage`; when an EDA run
is passed via `from`, the flagged columns are dropped so leakage is never silently
engineered into the feature set.
