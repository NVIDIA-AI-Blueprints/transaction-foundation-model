---
name: loom-features
description: Build engineered features from a Metaflow data object through Loom and WRITE them as a NEW versioned data object — domain-neutral transforms (numeric scaling/interactions, categorical encoding, datetime parts, simple aggregations) — then narrate the @card. Composes with loom-eda — pass an EDA run via --from and its leakage-flagged columns are DROPPED before building. Use when the user says "engineer features", "build features for this", "make a feature set", "prep features before training". Reads the source read-only; writes only a new data object in its own workspace.
when_to_use: "engineer features from a data object, build a reusable feature set, encode/scale/transform columns before optimize/validate, drop leakage columns flagged by eda"
when_not_to_use: "to profile an unprofiled dataset / check leakage first, use loom-eda; to chain profile->features->optimize->validate in one run, use loom-pipeline; to search solution code against a metric, use loom-optimize."
argument-hint: "<a dataset_ref pathspec (+ optional target, --from eda-run, --recipe)>"
---

# loom-features

Build engineered features from a data object and hand back a **new, versioned data
object** the rest of the lifecycle can consume — so feature work is a
lineage-grounded artifact, not a notebook cell that evaporates. This is a **planned,
gated run through Loom's MLOps interface**: it materializes the source data object
read-only, applies only the domain-neutral transforms the column dtypes justify
(numeric scaling/interactions, categorical encoding, datetime parts, a simple
group-mean aggregation), and writes the engineered table as a brand-new Metaflow
data object whose pathspec (`FeaturesFlow/<id>`) every downstream verb takes via
`--dataset`. Stay domain-neutral — never assume a task type, column meaning, or
vertical; apply only the transforms the data justifies and preserve the target
untouched.

## When to use

- The user points at a `dataset_ref` and asks to "engineer features", "build a
  feature set", "scale/encode the columns", or "prep features before training".
- Right after `loom-eda` flagged leakage: pass the EDA run via `--from` so the
  flagged columns are **dropped** before features are built on them.

## When NOT to use

- To *profile* an unprofiled dataset / check for leakage first — run **`loom-eda`**.
- To chain *profile → features → optimize → validate* in one gated run — use
  **`loom-pipeline`**.
- To *search solution code* against a metric — hand off to **`loom-optimize`**.

## 1. Intake — pin the data object (refuse if there is no pathspec)

Pin the inputs in the user's own terms and write them back for confirmation:

- **Data** — the **`dataset_ref`**: a Metaflow **pathspec** (e.g. `IngestDataset/123`
  or another `FeaturesFlow/<id>` to chain builds). Take a pathspec, **never a raw S3
  URI or a loose local file** as the source of truth.
- **Target (optional)** — the column to **preserve untouched** so it is never
  engineered as a feature; inferred from the data object's schema when omitted.
- **`--from` (optional)** — an upstream `loom-eda` run pathspec; its leakage-flagged
  columns are **DROPPED** before building (the `eda → features` composition edge).
- **`--recipe` (optional)** — `minimal` (scaling + encoding only) or `full`
  (default; adds datetime parts, interactions, a group-mean aggregation).

## 2. Plan — show the plan + tier (workspace-write, light)

Features is the **workspace-write tier** of the approval matrix (see
`CONVENTIONS.md`): it reads the source data object **read-only** and writes a **new
data object only into this run's own Metaflow workspace** — a light, no-prompt
workspace-write with **network off**. Briefly state the plan before running: "I'll
build features from `<dataset_ref>` (workspace-write, own workspace) — scaling,
encoding, datetime parts, interactions — preserving target `<col>`, dropping any
leakage columns from `<eda-run>`, and hand back the NEW `FeaturesFlow/<id>` data
object + `@card`." Name the exact `dataset_ref` and that the data stays in the
user's own Metaflow perimeter, not exfiltrated. Re-state the plan if the user
adjusts the target / recipe / `--from`.

## 3. Run — call Loom's MLOps INTERFACE (the `loom` CLI), never the backend

Speak only Loom's interface — shell out to the `loom` CLI, which resolves the MLOps
provider by name (default **Metaflow**, swappable by config) and runs the
feature-build flow through the interface's `run_flow` seam. **Never call Metaflow or
AIDE directly, and never touch raw S3** — the data object is read only through the
Client API; the datastore is the interface's opaque concern.

```bash
loom features --dataset <PATHSPEC> [--target <COL>] [--from <EDA-RUN>] [--recipe minimal|full]
```

- The work executes as a **Metaflow run**; the input is the source data object by
  `dataset_ref`, read via the Client API. The engineered `train`/`test`/`schema`/
  `fingerprint` artifacts **are** the new data object.
- Lifecycle flows need the **metaflow** MLOps provider — the `local` dev provider
  cannot run them (it will say so, pointing at `--mlops metaflow`).
- Secrets/endpoints (the Metaflow profile) come from the **environment** only.

## 4. Verify — assert lineage; the new data object stays in Metaflow

- The command returns the **new data object's pathspec** (`FeaturesFlow/<id>`) and
  the **`@card` reference**; confirm it reported success. The build summary is a
  small *derived* dict (feature counts, dropped columns, null/variance stats) — the
  engineered rows stay in Metaflow as the new data object's artifacts, never inlined.
- The new data object carries a **content fingerprint** so a downstream verb can
  assert it consumed exactly this feature build; every figure traces back to the run
  pathspec + the source `dataset_ref` it read.

## 5. Deliver — narrate the @card, return the new data object + summary, append a learnings row

- **Narrate the `@card`:** walk the user through the build — the **before → after
  feature count** and the engineered families (scaling, encoding, datetime parts,
  interactions, the group-mean aggregation); the **null %/variance** of the new
  columns (flag a constant or all-null engineered column); and the **leakage
  handling** — exactly which columns were dropped because the upstream EDA flagged
  them (or that none were).
- **Hand back the mandated artifact:** the **NEW `FeaturesFlow/<id>` data object** +
  the versioned `@card`, plus the typed build summary the CLI prints with its
  `VERDICT: BUILT` line.
- **Learnings:** the run appends one `command="features"` row to the flywheel corpus
  (`learnings/rollouts.jsonl`) — source + new data-object refs · target · recipe ·
  feature counts · dropped/leakage columns · fingerprint · run + card pathspecs —
  sanitized, no raw rows, no secrets. The CLI does this; do not hand-write the row.
- **Next step:** offer `loom-pipeline` / `loom-optimize` / `loom-validate` against
  the new `FeaturesFlow/<id>` data object.

## Composition — machine-checkable exit gate (executable self-test)

- **Consumes:** a `dataset_ref` pathspec (from `loom-connect`/`loom-eda`) and,
  optionally, a `loom-eda` run pathspec via `--from` (the `eda → features` edge).
- **Exit gate:** the build's typed summary carries a **new `dataset_ref` +
  `fingerprint` + `refused_leakage`** the downstream verbs consume; when an EDA run
  is passed via `--from`, the EDA-flagged columns are **dropped** so leakage is never
  silently engineered into the feature set.
- **Self-test:** the leakage-drop is covered by an executable self-test on a known-
  leaky fixture — `tests/test_features.py::test_build_features_drops_leakage_columns`
  asserts the flagged column is dropped (and never engineered), and the build-family
  / fingerprint tests guard the rest of the contract.

---

## Acceptance test (the bar before this verb joins the pack)

1. **Speaks only the interface** — shells out to `loom features` (the MLOps
   interface, provider-by-name), never Metaflow/AIDE directly, never raw S3; input
   is a `dataset_ref` pathspec read via the Client API.
2. **Output is a versioned run + `@card`** (and a NEW data object) — not a chat
   transcript or a loose pandas snippet.
3. **Approval tier is correct** — workspace-write tier, light/auto, network off (it
   reads read-only and writes only into its own workspace); no
   `disable-model-invocation` needed.
4. **Writes a learnings row** — the run appends a sanitized `command="features"` row
   to `learnings/rollouts.jsonl` (the CLI does this every run).
5. **Exit gate has a self-test** — the leakage-drop is covered by the
   `tests/test_features.py` drop-leakage test above.
6. **Single free-text arg** — one `dataset_ref` (plus an optional target, `--from`
   eda-run, and recipe).
7. **Dual-invocation** — works user-typed (`/loom-features`) and model-auto-loaded
   on the `description` / `when_to_use` match; safe to auto-fire (own-workspace,
   non-destructive over the source).
