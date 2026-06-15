---
name: loom-ops
description: Monitor Loom's own runs and data objects through the MLOps interface — recent run health (successes/failures, recency), the leaderboard, schedule/run health, and a simple data-object DRIFT check (compare a data object's schema/summary stats to a reference) — then narrate the @card. Read-only. Use when the user says "what's running", "show me run health", "did the last runs pass", "is the data drifting", "check the leaderboard". Trains nothing, writes nothing back, never prompts.
when_to_use: "check run health, see successes/failures and recency, read the leaderboard, check schedule health, detect data drift vs a reference data object"
when_not_to_use: "to ship/promote a validated model, use loom-deploy; to assemble a shareable bundle of a run, use loom-collab; to rigorously evaluate a candidate, use loom-validate."
argument-hint: "<a --flow NAME, an --experiment ID, or a --dataset + --reference drift pair>"
---

# loom-ops

Monitor Loom's runs and data objects so a human can see **what passed, what failed,
and whether the data moved** — without touching or mutating anything. This is a
**read-only run through Loom's MLOps interface**: it reads finished runs and data-
object schemas/stats through the Client API and hands back a versioned Metaflow run +
an `@card` (run health, leaderboard, a drift table). It trains nothing, writes
nothing back, and **never prompts**. Stay domain-neutral — it reports only what the
runs and the data's summary stats actually say, never a vertical.

## When to use

- The user asks "what's running?", "show me run health", "did the last runs pass?",
  "check the leaderboard", or "is the data drifting?".
- A scheduled flow's health (recency + success rate) needs a quick read.

## When NOT to use

- To *ship / promote* a validated model — use **`loom-deploy`**.
- To *assemble a shareable bundle* of a run for a teammate — use **`loom-collab`**.
- To *rigorously evaluate* a candidate (CV / sealed holdout / calibration) — use
  **`loom-validate`**.

## 1. Intake — pin the monitoring view

Pin the inputs in the user's own terms and write them back for confirmation. Give one
of:

- **`--flow NAME`** — a flow's **recent run health** (e.g. `ValidateFlow`):
  successes/failures, recency, the most-recent outcome.
- **`--experiment ID`** — an experiment's runs + **leaderboard** (the
  `loom_experiment:<id>` tag).
- **`--dataset PATHSPEC --reference PATHSPEC`** — a **DRIFT** check: compare a current
  data object's schema / summary stats to a reference data object. Take pathspecs,
  **never a raw S3 URI or a loose local file**.

A read-only monitor may proceed with just one of these; if none is given the CLI
refuses with a hint.

## 2. Plan — show the plan + tier (read-only)

Ops is the **read-only tier** of the approval matrix (see `CONVENTIONS.md`): all of
it is non-destructive reads through the Client API, so it **never prompts**. Briefly
state the plan before running: "I'll read run health for `<flow/experiment>` (and/or
drift of `<dataset_ref>` vs `<reference>`) — read-only — and hand back the run +
`@card`." Name exactly which flow/experiment/data objects will be read and that
nothing is written outside this run.

## 3. Run — call Loom's MLOps INTERFACE (the `loom` CLI), never the backend

Speak only Loom's interface — shell out to the `loom` CLI, which resolves the MLOps
provider by name (default **Metaflow**, swappable by config) and runs the read-only
ops flow through the interface's `run_flow` seam. **Never call Metaflow or AIDE
directly, and never touch raw S3** — runs and data objects are read only through the
Client API; the datastore is the interface's opaque concern.

```bash
loom ops --flow <NAME>                                  # run health for a flow
loom ops --experiment <ID>                              # runs + leaderboard for an experiment
loom ops --dataset <PATHSPEC> --reference <PATHSPEC>    # data-object drift check
```

- The work executes as a **Metaflow run**; recent runs + the two data objects'
  frames are read via the Client API.
- Lifecycle flows need the **metaflow** MLOps provider — the `local` dev provider
  cannot run them (it will say so, pointing at `--mlops metaflow`).
- Secrets/endpoints (the Metaflow profile) come from the **environment** only.

## 4. Verify — assert lineage; large output stays in Metaflow

- The command returns a **run pathspec** and the **`@card` reference**; confirm it
  reported success. The summary is a small *derived* dict (health counts, the
  leaderboard, the drift flags) — the data objects' rows stay in Metaflow, never
  inlined.
- Every health/drift figure traces back to the runs + data objects it read by
  pathspec.

## 5. Deliver — narrate the @card, return run + summary, append a learnings row

- **Narrate the `@card`:** walk the user through **run health** (run/success/failure
  counts, success rate, whether the most-recent run succeeded — flag a `DEGRADED`
  latest-failed status); the **leaderboard** (scored runs best-first); and the
  **drift table** — `STABLE` vs `DRIFT` with the per-column smells (a numeric
  mean-shift, a null-rate shift, or a schema add/remove). Flag exactly which columns
  moved.
- **Hand back the mandated artifact:** the versioned **Metaflow run + `@card`** plus
  the typed summary the CLI prints, with its overall `VERDICT`
  (`OK`/`ATTENTION`/`EMPTY`).
- **Learnings:** the run appends one `command="ops"` row to the flywheel corpus
  (`learnings/rollouts.jsonl`) — flow/experiment/data-object refs · run-health status ·
  drift status · run + card pathspecs — sanitized, no raw rows, no secrets. The CLI
  does this; do not hand-write the row.
- **Next step:** on `ATTENTION` (degraded health or detected drift), point at what to
  do — re-run the failed flow, or re-validate/re-feature against the drifted data; on
  `OK`, nothing to do.

## Composition — machine-checkable exit gate (executable self-test)

- **Consumes:** a `--flow`, an `--experiment`, or a `--dataset` + `--reference` drift
  pair (data-object pathspecs from `loom-connect`/`loom-features`).
- **Exit gate:** the summary's typed **`status`** (`OK`/`ATTENTION`/`EMPTY`) and the
  **drift `status`** (`STABLE`/`DRIFT`) are the signals a human (or a downstream
  re-validate) reads — a detected drift or degraded run health surfaces as
  `ATTENTION`.
- **Self-test:** the drift smell test has an executable self-test —
  `tests/test_ops.py::test_compute_drift_identical_frames_is_stable` asserts identical
  frames are `STABLE` (the false-positive failure mode), and
  `::test_compute_drift_flags_numeric_mean_shift` /
  `::test_compute_drift_flags_null_rate_shift` /
  `::test_compute_drift_reports_schema_add_remove` assert a shifted/null/changed
  frame flags `DRIFT`. The `summarize_ops` tests pin the run-health status.

---

## Acceptance test (the bar before this verb joins the pack)

1. **Speaks only the interface** — shells out to `loom ops` (the MLOps interface,
   provider-by-name), never Metaflow/AIDE directly, never raw S3; runs + data objects
   are read via the Client API.
2. **Output is a versioned run + `@card`** — not a chat transcript or a loose
   metadata dump.
3. **Approval tier is correct** — read-only tier, never prompts; no
   `disable-model-invocation` needed.
4. **Writes a learnings row** — the run appends a sanitized `command="ops"` row to
   `learnings/rollouts.jsonl` (the CLI does this every run).
5. **Exit gate has a self-test** — the drift smell test (stable vs shifted) is covered
   by the `tests/test_ops.py` drift tests above.
6. **Single free-text arg** — one monitoring view: a `--flow`, an `--experiment`, or a
   `--dataset` + `--reference` drift pair.
7. **Dual-invocation** — works user-typed (`/loom-ops`) and model-auto-loaded on the
   `description` / `when_to_use` match; safe to auto-fire (read-only).
