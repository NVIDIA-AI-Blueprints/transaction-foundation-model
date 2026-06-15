---
name: loom-connect
description: Data access for Loom — register a source (a local directory of CSVs or a single CSV) as a Metaflow data object addressable by pathspec, and list the data objects already ingested. Use when the user wants to point Loom at data and get a dataset_ref ("connect to this data", "ingest this dataset", "register this CSV", "what datasets do I have", "list my data objects"). Read-only/light: it registers a source once into the MLOps interface and reads back the catalog; it never optimizes, profiles deeply, or mutates the source.
when_to_use: "connect to / ingest a dataset, register a source as a data object, get a dataset_ref pathspec, list ingested datasets"
when_not_to_use: "to profile what is IN a registered data object (shape/missingness/leakage), use loom-eda; to optimize against a metric, use loom-optimize."
argument-hint: "<a source path to register, or 'list' to see ingested data objects>"
---

# loom-connect

Data access is the #1 daily DS pain and the front door to every other Loom verb:
nothing downstream runs until the data is a **Metaflow data object** addressable
by **pathspec**. This verb does exactly that one job — register a source once into
Loom's MLOps interface (so it becomes a versioned, content-addressed, profile-backed
data object) and surface the catalog of what is already registered. It is the
boundary where outside data crosses *into* Metaflow; from then on every verb reads
it only through the Client API, never as a loose file.

Loom is domain-neutral: reflect back only the source the user actually pointed at;
never assume a task type, column meaning, or vertical.

## When to use

- The user wants to point Loom at data: "connect to this data", "ingest this
  dataset", "register this CSV/folder", "make a dataset_ref out of this".
- The user wants to see what is already registered: "what datasets do I have",
  "list my data objects", "show me the ingested data".

## When NOT to use

- To profile what is *inside* a registered data object (shape, dtypes,
  missingness, target balance, leakage smells) — hand off to **`loom-eda`**.
- To optimize solution code against a metric — hand off to **`loom-optimize`**.

## 1. Intake — pin the source

Pin one input in the user's own terms and write it back for confirmation:

- **Register a source** — a **local path**: a directory containing `train.csv`
  (and optionally `test.csv`), or a single `.csv` file. Take a path, never a raw
  S3 URI or a loose in-memory frame as the source of truth.
- **List datasets** — no input; the user just wants the catalog.

If the user wants to register but gives no usable path (or a path that does not
exist), **ask — do not guess**. A single source per registration; do not batch.

## 2. Plan — show the plan + tier (read-only / light)

This verb is the **read-only tier** of the approval matrix (see
`CONVENTIONS.md`): listing is a pure Client-API read, and registering writes only
*into* the MLOps interface's own datastore (a one-time ingest of the source the
user explicitly pointed at — not a mutate of their source, not compute spend).
**It never prompts.** Still, briefly state the plan before acting:

- **List:** "I'll list the ingested data objects via the Client API."
- **Register:** "I'll register `<path>` once into a Metaflow data object and hand
  back its pathspec." Name which source path will be read and that the data is
  staged into the user's own Metaflow perimeter (on a remote profile), not
  exfiltrated.

## 3. Run — call Loom's MLOps INTERFACE (the `loom` CLI), never the backend

Speak only Loom's interface — shell out to the `loom` CLI, which resolves the
MLOps provider by name (default **Metaflow**, swappable by config). **Never call
Metaflow directly and never touch raw S3** — the datastore is an opaque detail the
interface owns.

- **Register a source** (the one external→Metaflow boundary):

  ```bash
  loom ingest --source <PATH> [--name <NAME>]
  ```

  It prints the new data object's **pathspec** (e.g. `IngestDataset/123`) — that
  is the `dataset_ref` every downstream verb consumes.

- **List ingested data objects** (a pure Client-API read):

  ```bash
  loom datasets
  ```

  It prints one row per data object: `pathspec · name · nrows/schema`.

Secrets/endpoints (the Metaflow profile, any datastore credentials) come from the
**environment** only — never put them on the command line or in the transcript.

## 4. Verify — confirm the data object exists

After a register, confirm the ingest reported success and a pathspec came back
(the CLI fails fast and non-zero if the flow did not complete). Optionally run
`loom datasets` to show the new data object now appears in the catalog — its
presence *is* the lineage check (a versioned run by pathspec, not a loose file).

## 5. Deliver — narrate, return the data object + pathspec, append a learnings row

- **Narrate:** for a register, give the user the **pathspec** plainly and state it
  is the `dataset_ref` for the next verb; for a list, summarize the catalog (how
  many data objects, names, row counts).
- **Hand back the mandated artifact:** the **Metaflow data object** (the
  `IngestDataset` run, addressable by pathspec) — the versioned, shareable
  reference, not a loose CSV path.
- **Learnings:** `loom datasets` is a pure read and records nothing; `loom ingest`
  registers the data object as the durable record. (Deep profiling and its
  learnings row are `loom-eda`'s job.) Never persist secrets or raw rows.
- **Next step:** offer to hand off to **`loom-eda`** to profile the new data
  object (`loom eda --dataset <pathspec>`).

## Composition — machine-checkable exit gate (executable self-test)

- **Consumes:** a source path (register) or nothing (list).
- **Produces (the gate output the next verb asserts):** a **`dataset_ref`
  pathspec** of shape `<FlowName>/<run_id>` (e.g. `IngestDataset/123`). Every
  downstream verb (`loom-eda`, `loom-optimize`) refuses to start without one — a
  missing/malformed `dataset_ref` BLOCKS them (`loom.dataio.resolve_run` raises on
  a non `<flow>/<run_id>` reference).
- **Self-test:** the `dataset_ref` contract is guarded by the executable test
  `tests/test_dataio.py::test_resolve_run_rejects_non_run_pathspecs`, which asserts
  that empty / malformed references (e.g. `""`, `"IngestDataset"`, `"a/b/c"`)
  **raise** rather than silently resolving — i.e. a bad `dataset_ref` blocks the
  downstream read.

---

## Acceptance test (the bar before this verb joins the pack)

1. **Speaks only the interface** — shells out to `loom ingest` / `loom datasets`
   (the MLOps interface, provider-by-name), never Metaflow directly, never raw S3.
2. **Output is a versioned data object** — a Metaflow `IngestDataset` run
   addressable by pathspec, not a loose CSV path.
3. **Approval tier is correct** — read-only/light tier, never prompts; no
   `disable-model-invocation` needed (it neither spends compute nor writes
   irreversibly to the user's source).
4. **Writes a learnings row** — n/a for the pure read (`loom datasets`); the
   registration's durable record is the data object itself, and deep-profile
   learnings belong to `loom-eda`.
5. **Exit gate has a self-test** — the `dataset_ref` shape gate is covered by
   `tests/test_dataio.py::test_resolve_run_rejects_non_run_pathspecs`.
6. **Single free-text arg** — one source path (or "list").
7. **Dual-invocation** — works user-typed (`/loom-connect`) and model-auto-loaded
   on the `description` / `when_to_use` match; safe to auto-fire (read-only/light).
