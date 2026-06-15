---
description: Register a local source (a directory of CSVs or a single CSV) as a versioned, content-addressed data object addressable by pathspec.
argument-hint: <a source path to register>
---

# /loom-ingest — register a source as a data object (workspace-write)

Data access is the front door to every other Loom verb: nothing downstream runs
until the data is a **versioned, content-addressed data object** addressable by
pathspec. This verb does that one job — register a source once into the workspace
and run the schema sniff + the advisory **EDA leakage scan**.

Register: $@

## 1. Intake — pin the source
- A **local path**: a directory containing `train.csv` (and optionally
  `test.csv`), or a single `.csv`. Take a path, never a raw S3 URI or a loose
  in-memory frame. If the path is missing or does not exist, **ask — do not
  guess**. One source per registration; do not batch.
- If the user knows the framing, take `--entity` (the grouping column a sequence
  belongs to), `--event` (what a row is), and `--target` (the label, if any, so
  the scan runs the target-leakage check).

## 2. Plan — workspace-write / light
Registering writes only *into* the workspace (a one-time ingest of the source the
user pointed at — never a mutate of their source) and never prompts. State: "I'll
register `<path>` once as a versioned data object and hand back its pathspec.
Ingest is idempotent — the same source + spec returns the same object."

## 3. Run — call the `loom_ingest` tool
Call `loom_ingest` with `--in <path>` (and `--name`/`--entity`/`--event`/`--target`
if given). It registers the source, sniffs the schema, runs the advisory **EDA
leakage scan** (flagging id-shaped / near-unique / target-correlated columns), and
prints the new data object's **pathspec** (e.g. `IngestDataset/1`) — the
`dataset_ref` every downstream verb consumes. Speak only the interface; never
expose internals.

## 4. Verify — confirm the data object exists
Confirm `status == ok` and a `pathspec` came back. Read `data.schema` (rows/cols)
and `data.eda.flags` — `verdict=REVIEW` means the scan flagged columns; that is
advisory, not a failure. A flagged column that is the grouping entity should be
passed as `--entity` (it is then never tokenized as a feature); a flagged column
that is not the entity should be dropped before tokenizing.

## 5. Deliver — return the data object + pathspec
- Give the **pathspec** plainly and state it is the `--in` ref for the next verb.
- Hand back the data object (the `IngestDataset/<n>` by pathspec), not a loose
  CSV path.
- Summarize the schema and the leakage flags in plain words.
- **Next step:** offer `/loom-tokenize --in <pathspec>` to compile a
  contract-checked Corpus, and `/loom-baseline --in <pathspec>` to set the bar.

## Composition / exit gate
Produces a `dataset_ref` pathspec of shape `IngestDataset/<n>`; every downstream
verb refuses to start without a well-formed one.
