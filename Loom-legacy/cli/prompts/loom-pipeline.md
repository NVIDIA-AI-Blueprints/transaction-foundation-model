---
description: Run the end-to-end lifecycle (profile -> features -> optimize -> validate) as one gated run; escalates to expensive at the optimize step.
argument-hint: <a dataset_ref pathspec + a goal sentence (+ optional target)>
---

# /loom-pipeline — the gated end-to-end lifecycle (workspace-write → expensive)

Run the back-half of the DS lifecycle — profile → features → a bounded
optimize step → validate — as **one gated Metaflow run**, so the whole chain is a
single reproducible artifact with cross-stage gates instead of four loose scripts.
Each stage asserts the prior stage's `VERDICT` (leakage blocks/handles features; a
sub-threshold validate marks the run `FAIL`). The metric is the spec.

Run the pipeline for: $@

## 1. Intake — pin the spec (refuse without a measurable goal)
- **Data** — a `dataset_ref` pathspec. Never a raw S3 URI / loose file.
- **Goal** — one natural-language sentence of what the solution should achieve.
- **Target (optional)** — the column the validate stage scores against; inferred
  when omitted (without a resolvable target the run is `FAIL`).

If the goal/target is vague, **ask — do not guess**.

## 2. Plan — STOP at the gate before the costly stage
Workspace-write that **escalates to EXPENSIVE** at the optimize stage. The
profile/features stages are light workspace-writes (network off), but optimize
spends real compute — so give the user a heads-up before firing. Show the plan:
"I'll run `<dataset_ref>` end-to-end — profile → features → optimize (≤ N steps) →
validate — gated per stage, and hand back the composite run + `@card` + `VERDICT`."
Surface the cost shape (the optimize budget) and the data scope. Re-plan if the
user adjusts budget / target / goal.

## 3. Run — call the `loom_pipeline` tool
Call `loom_pipeline` with `dataset` and `goal` (+ `target`). It runs as a single
Metaflow run chaining all four stages, each asserting the prior stage's `VERDICT`.

## 4. Verify — assert lineage
Confirm `status` and read the **run pathspec** + `card_path`. The composite summary
is small derived data (per-stage status + gate decisions + the headline verdict);
fold scores / engineered rows stay in Metaflow.

## 5. Deliver — narrate, return run + summary + VERDICT
- Walk through each stage (ran / skipped / **BLOCKED** and why); how leakage was
  handled; the bounded optimize budget; and the validate stage's sealed-holdout
  `VERDICT`. On `FAIL`, name the failed stage exactly.
- Hand back the run + `@card` + typed summary with the headline `VERDICT`
  (`PASS`/`REVIEW`/`FAIL`).
- **Next step:** on `PASS`, offer `/loom-validate` (deeper standalone check) or
  `/loom-deploy` (gate promotion); on `REVIEW`/`FAIL`, flag exactly what to fix.

## Composition / exit gate
Each stage asserts the prior stage's `VERDICT`; the headline `VERDICT`
(`PASS`/`REVIEW`/`FAIL`) is the typed gate `/loom-deploy` asserts.
