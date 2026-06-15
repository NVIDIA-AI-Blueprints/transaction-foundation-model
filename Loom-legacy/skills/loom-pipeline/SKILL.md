---
name: loom-pipeline
description: Run the end-to-end DS lifecycle as ONE gated Metaflow run through Loom — profile -> features -> a bounded candidate/optimize step -> validate — where each stage asserts the prior stage's VERDICT before running (leakage blocks/handles features; a sub-threshold validate marks the run FAIL). Use when the user says "run the whole pipeline", "do the end-to-end thing", "profile, feature, train and validate this", "give me a baseline pipeline". Workspace-write, but ESCALATES to expensive at the train/optimize stage — gate before the costly step.
when_to_use: "run the full lifecycle in one shot, end-to-end profile->features->optimize->validate, get a gated baseline pipeline, reproducible ingest->feature->train->eval DAG"
when_not_to_use: "to only profile a dataset, use loom-eda; to only build a feature set, use loom-features; to only rigorously validate an existing solution, use loom-validate; to deeply tree-search solution code, use loom-optimize."
argument-hint: "<a dataset_ref pathspec + a goal sentence (+ optional target)>"
---

# loom-pipeline

Run the back-half of the DS lifecycle — profile, feature-engineer, fit/optimize a
bounded candidate, then rigorously validate — as **one gated Metaflow run**, so the
whole chain is a single reproducible artifact with cross-stage gates instead of four
loose scripts. This is a **planned, gated run through Loom's MLOps interface**: each
stage asserts the prior stage's `VERDICT` before it runs (a profile that flags
**leakage** blocks/handles the features stage; a sub-threshold **validate** marks the
whole run `FAIL` — the same exit-gate vocabulary `loom-deploy` reuses), and the
costly optimize stage is held to a **declared budget**. **The metric is the spec.**
Stay domain-neutral — never assume a task type, column meaning, or vertical; reflect
back only what the data and the goal actually say.

## When to use

- The user points at a `dataset_ref` + a goal and asks to "run the whole pipeline",
  "do the end-to-end thing", or "profile, feature, train and validate this".
- They want a **gated baseline pipeline** in one shot rather than driving each verb
  by hand.

## When NOT to use

- To *only profile* a dataset (shape / leakage) — run **`loom-eda`**.
- To *only build a feature set* (a new data object) — use **`loom-features`**.
- To *only rigorously validate* an existing solution — use **`loom-validate`**.
- To *deeply tree-search* solution code against a metric — use **`loom-optimize`**.

## 1. Intake — pin the spec (refuse without a measurable goal)

Pin the inputs in the user's own terms and write them back for confirmation:

- **Data** — the **`dataset_ref`**: a Metaflow **pathspec** (e.g.
  `IngestDataset/123`). Take a pathspec, **never a raw S3 URI or a loose local
  file**.
- **Goal** — one natural-language sentence of what the solution should achieve. The
  pipeline is domain-neutral; the goal is carried into the candidate stage + the
  learnings row.
- **Target (optional)** — the column the validate stage scores against; inferred
  from the data object's schema when omitted. Without a resolvable target the
  validate stage cannot score and the run is marked `FAIL`.

If the goal/target is vague, **ask — do not guess**.

## 2. Plan — show the plan + tier, STOP at the gate before the costly stage

Pipeline is the **workspace-write tier** that **escalates to EXPENSIVE** at its
train/optimize stage (see `CONVENTIONS.md`): the profile/features stages are light
workspace-writes (network off), but the optimize stage **spends real compute**, so
**stop at the approval gate before that stage**. Show the plan: "I'll run
`<dataset_ref>` end-to-end — profile → features → optimize (≤ N candidate steps,
the declared budget) → validate — gated per stage, and hand back the composite run +
`@card` and its `VERDICT`." Surface the **cost shape** (the optimize budget) and the
**data scope** (the exact `dataset_ref`, staged into the user's own perimeter). Only
the *taste* decisions (goal, target, budget) need the user; everything mechanical is
autonomous within the declared budget. Re-plan if the user adjusts budget / target /
goal.

## 3. Run — call Loom's MLOps INTERFACE (the `loom` CLI), never the backend

Speak only Loom's interface — shell out to the `loom` CLI, which resolves the MLOps
provider by name (default **Metaflow**, swappable by config) and runs the lifecycle
flow through the interface's `run_flow` seam. **Never call Metaflow or AIDE directly,
and never touch raw S3** — the data object is read only through the Client API.

```bash
loom pipeline --dataset <PATHSPEC> --goal "<one sentence>" [--target <COL>]
```

- The work executes as a single **Metaflow run** chaining all four stages; each
  stage asserts the prior stage's `VERDICT` before running.
- Lifecycle flows need the **metaflow** MLOps provider — the `local` dev provider
  cannot run them (it will say so, pointing at `--mlops metaflow`).
- Secrets/endpoints (the Metaflow profile) come from the **environment** only.

## 4. Verify — assert lineage; large output stays in Metaflow

- The command returns a **run pathspec** and the **`@card` reference**; confirm it
  reported success. The composite summary is a small *derived* dict (per-stage
  status + gate decisions + the headline verdict) — the engineered rows / fold
  scores stay in Metaflow, never inlined.
- Every stage's outcome traces back to that run's pathspec + the `dataset_ref` it
  read; the **gate decisions** are the cross-stage composition trail.

## 5. Deliver — narrate the @card, return run + summary + VERDICT, append a learnings row

- **Narrate the `@card`:** walk the user through the stages — whether each **ran /
  was skipped / was BLOCKED** and why; how **leakage was handled** (flagged columns
  dropped before features); the **bounded optimize budget**; and crucially the
  validate stage's **sealed-holdout `VERDICT`**. If the run is `FAIL`, name the
  **failed stage** exactly (e.g. "blocked at features: leakage with no droppable
  columns" / "validate sub-threshold").
- **Hand back the mandated artifact:** the versioned **Metaflow run + `@card`** plus
  the typed composite summary the CLI prints, with its headline `VERDICT`
  (`PASS`/`REVIEW`/`FAIL`).
- **Learnings:** the run appends one `command="pipeline"` row to the flywheel corpus
  (`learnings/rollouts.jsonl`) — data-object ref · goal · target · leakage-handled ·
  failed stage · verdict · run + card pathspecs — sanitized, no raw rows, no secrets.
  The CLI does this; do not hand-write the row.
- **Next step:** on `VERDICT: PASS`, offer `loom-validate` for a deeper standalone
  check or `loom-deploy` to gate promotion; on `REVIEW`/`FAIL`, flag exactly what to
  fix (resolve leakage, re-engineer features, raise the holdout) before any deploy.

## Composition — machine-checkable exit gate (executable self-test)

- **Consumes:** a `dataset_ref` pathspec (from `loom-connect`/`loom-eda`/
  `loom-features`).
- **Exit gate:** each stage asserts the prior stage's `VERDICT` before running, and
  the run's headline `VERDICT` (`PASS`/`REVIEW`/`FAIL`) is the typed gate the next
  verb (`loom-deploy`) asserts. Leakage blocks features when undroppable; a
  sub-threshold (higher-is-better) holdout downgrades the run to `FAIL`.
- **Self-test:** the cross-stage ordering is covered by executable self-tests on stub
  stage results — `tests/test_pipeline.py::test_orchestrate_gate_decisions_are_ordered`
  asserts the gates fire in `profile → features → optimize → validate` order,
  `::test_orchestrate_undroppable_leakage_blocks_features` asserts leakage blocks the
  features stage, and `::test_orchestrate_sub_threshold_metric_downgraded_to_fail`
  asserts a sub-threshold holdout marks the run `FAIL` (the "fails-open" mode it
  guards against).

---

## Acceptance test (the bar before this verb joins the pack)

1. **Speaks only the interface** — shells out to `loom pipeline` (the MLOps
   interface, provider-by-name), never Metaflow/AIDE directly, never raw S3; input
   is a `dataset_ref` pathspec read via the Client API.
2. **Output is a versioned run + `@card`** — not a chat transcript or a loose
   script.
3. **Approval tier is correct** — workspace-write that **escalates to expensive** at
   the optimize stage; the skill stops at the gate before the costly stage (the
   optimize budget is declared and bounded). No `disable-model-invocation` needed
   (the gate is at the costly stage, not the whole verb).
4. **Writes a learnings row** — the run appends a sanitized `command="pipeline"` row
   to `learnings/rollouts.jsonl` (the CLI does this every run).
5. **Exit gate has a self-test** — the per-stage gate ordering is covered by the
   `tests/test_pipeline.py` orchestration tests above.
6. **Single free-text arg** — one `dataset_ref` + a goal sentence (plus an optional
   target).
7. **Dual-invocation** — works user-typed (`/loom-pipeline`) and model-auto-loaded
   on the `description` / `when_to_use` match; the model proposes the plan and stops
   at the gate before the expensive optimize stage.
