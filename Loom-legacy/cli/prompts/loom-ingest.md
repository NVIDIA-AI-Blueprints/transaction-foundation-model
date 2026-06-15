---
description: Register a local source (a directory of CSVs or a single CSV) as a Metaflow data object addressable by pathspec.
argument-hint: <a source path to register>
---

# /loom-ingest — register a source as a data object (workspace-write)

Data access is the front door to every other Loom verb: nothing downstream runs
until the data is a **Metaflow data object** addressable by pathspec. This verb
does that one job — register a source once into Loom's MLOps interface. To *list*
what is already registered, use `/loom-datasets`.

Register: $@

## 1. Intake — pin the source
- A **local path**: a directory containing `train.csv` (and optionally
  `test.csv`), or a single `.csv`. Take a path, never a raw S3 URI or a loose
  in-memory frame. If the path is missing or does not exist, **ask — do not
  guess**. One source per registration; do not batch.

## 2. Plan — workspace-write / light
Listing is a pure read; registering writes only *into* the MLOps interface's own
datastore (a one-time ingest of the source the user pointed at — not a mutate of
their source). It never prompts. State: "I'll register `<path>` once into a
Metaflow data object and hand back its pathspec. Data is staged into your own
Metaflow perimeter, not exfiltrated."

## 3. Run — call the `loom_ingest` tool
Call `loom_ingest` with `source` (and `name` if given). It prints the new data
object's **pathspec** (e.g. `IngestDataset/123`) — the `dataset_ref` every
downstream verb consumes. Speak only the interface; never touch Metaflow/raw S3.

## 4. Verify — confirm the data object exists
Confirm `status == ok` and a `pathspec` came back. Optionally call `loom_datasets`
to show it now appears in the catalog — its presence is the lineage check.

## 5. Deliver — return the data object + pathspec
- Give the **pathspec** plainly and state it is the `dataset_ref` for the next verb.
- Hand back the Metaflow data object (the `IngestDataset` run by pathspec), not a
  loose CSV path.
- **Next step:** offer `/loom-eda --dataset <pathspec>` to profile it.

## Composition / exit gate
Produces a `dataset_ref` pathspec of shape `<FlowName>/<run_id>`; every downstream
verb refuses to start without a well-formed one.
