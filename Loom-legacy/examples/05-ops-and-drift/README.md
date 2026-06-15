# 05-ops-and-drift -- monitoring: run health + data drift

## Use case

This example walks through Loom's **read-only monitoring** tier: watching the
health of your runs and catching when a data feed's distribution **drifts** away
from a known-good reference. Ops is the safe-by-default tier of the approval
matrix -- it trains nothing, writes nothing back, and never prompts; it only
*reads* finished runs and data-object schemas through the Metaflow Client API.

The synthetic data (`make_data.py`) is two sibling, domain-neutral frames with
the **identical schema** -- `id`, `feature_0 .. feature_7`, and a `target` in
`{0, 1}`:

* **`reference/train.csv`** -- the baseline: features drawn standard-normal.
* **`shifted/train.csv`** -- the same schema, but every numeric `feature_*`
  column has a **planted distribution shift** (an additive mean offset plus a
  variance inflation). The move is deliberately large enough to clear Loom's
  relative-mean-shift threshold (`0.25`) on every feature, so the drift check
  fires unambiguously.

It is deterministic, seeded, and generated offline (no downloads); the reference
and shifted frames use distinct derived seeds so they are independent draws, not
the same rows.

## Ask Loom

The natural-language line you would type at the `loom` agent -- the product UX:

```
loom "is my latest validate run healthy, and has this new batch drifted from last week's data?"
```

> Needs a model key. The agent plans, then runs the keyless `ops` verbs in
> "Step by step" under the hood (a run-health read for the flow + a drift
> comparison of the two data objects). The key-gated verbs (`loom run` /
> `loom optimize`, the pipeline's optimize stage) would only enter if you then
> asked it to *act* on the drift (e.g. re-validate or rebuild) -- they are **not**
> part of this monitoring recipe and **not** in the asserted `run.sh`.

## Step by step

The explicit, **keyless** verb sequence the agent runs under the hood. Each line
is one `$LOOM <verb> ... --json` with a one-line explanation and the expected
outcome (the `summary` field / VERDICT `run.sh` asserts on).

1. **`$LOOM ingest --source <dir>/reference --name <unique> --json`** -- the
   one external->Metaflow boundary, for the baseline. Expected: `status == "ok"`
   and a `pathspec` (the reference `dataset_ref`).
2. **`$LOOM ingest --source <dir>/shifted --name <unique>-shifted --json`** --
   ingest the shifted variant as its own data object. Expected: `status == "ok"`
   and a `pathspec` (the shifted `dataset_ref`).
3. **`$LOOM validate --dataset <reference> --target target --json`** -- a keyless
   baseline validate, run here only to **land a real `ValidateFlow` run** so the
   run-health view has something to count. Expected: `status == "ok"` and a
   `VERDICT` (`PASS`/`REVIEW`).
4. **`$LOOM ops --flow ValidateFlow --json`** -- the **run-health** view: reads
   recent `ValidateFlow` runs via the Client API and rolls up success/failure
   counts. Expected: `status == "ok"`; `summary.health` is a dict scoped to
   `ValidateFlow` with `n_runs >= 1` and `n_runs == n_successful + n_failed`; the
   `VERDICT` (the overall ops status) is one of `OK` / `ATTENTION` / `EMPTY`.
5. **`$LOOM ops --dataset <shifted> --reference <reference> --json`** -- the
   **drift check**: materializes both data objects via the Client API and
   compares their summary stats. Expected: `status == "ok"`; `summary.drift` is a
   dict with `drift == True`, `status == "DRIFT"`, and a non-empty `drift_flags`
   list naming the shifted `feature_*` columns. A detected drift degrades the
   overall ops `VERDICT` to `ATTENTION`.

_(Aside: in a real session you might follow a `DRIFT` finding with a key-gated
`loom run`/optimize to rebuild against the new distribution. That step needs a
model key, so it is omitted from the asserted recipe.)_

## What this proves

The monitoring contract stays intact end to end:

* **`ops --flow` always returns a run-health rollup** -- a `summary.health` dict
  with reconciling counts (`n_runs == n_successful + n_failed`), scoped to the
  requested flow, and a status `VERDICT` -- after a real run has landed.
* **`ops --dataset --reference` always catches a real distribution shift** -- a
  `summary.drift` dict with `drift == True` / `status == "DRIFT"` and a non-empty
  `drift_flags` list, when the current data object has moved past the threshold
  relative to its reference, and it degrades the overall `VERDICT` to
  `ATTENTION`.
* **Ops stays read-only and keyless** -- it counts runs and compares schemas
  through the Client API without training, writing back, or needing a model key.

This is the invariant `tests/test_examples.py` guards by replaying `run.sh`.
