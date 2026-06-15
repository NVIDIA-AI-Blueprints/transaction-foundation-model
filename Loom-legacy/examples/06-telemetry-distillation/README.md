# 06-telemetry-distillation -- the moat capture: rollouts -> a redacted SFT corpus

## Use case

This example demonstrates Loom's **moat capture**: every keyless verb you run
records a command-level **rollout**, the telemetry layer assembles those
rollouts (joined with the telemetry events + proxy LLM calls) into ordered
**trajectories**, and `telemetry export` distills them into **LOOM-DS-1** -- the
supervised-fine-tuning / teacher corpus that trains Loom's own data-science
model. Two hard invariants ride along: the **IP boundary** (only
`owned_by == "general"` trajectories are exportable -- tenant-owned work is never
in the cross-tenant set) and **prompt hygiene** (content is **redacted by
default** to a `<REDACTED:...>` sentinel; raw rows never enter the corpus).

The synthetic data (`make_data.py`) is incidental -- it is just fuel for the
verbs that generate the rollouts. It is a clean, learnable, domain-neutral binary
classification table: an `id` column, generic `feature_0 .. feature_11` columns
(six informative, the rest redundant/noise), and an integer `target` in `{0, 1}`
with a mild 60/40 imbalance. It is deterministic, seeded (`--seed 0`), and
generated offline. Because it is clean and learnable, the two verbs succeed
(`eda` finds no leakage, `validate` returns `PASS`), so the rollouts they record
carry a usable reward.

The example runs the verbs against a **private, isolated telemetry corpus**: it
points the events / trajectories / rollouts / proxy-calls paths at a scratch dir
(`LOOM_TELEMETRY_PATH`, `LOOM_TRAJECTORIES_PATH`, `LOOM_LEARNINGS_PATH`,
`LOOM_PROXY_LOG_PATH`) and enables capture with `LOOM_TELEMETRY=1`, so it sees
only the trajectories its own verbs produce -- no collision with a developer's
real corpus or a concurrent run.

## Ask Loom

The natural-language line you would type at the `loom` agent -- the product UX:

```
loom "summarize the training-data corpus my runs have produced and export the general, redacted SFT set"
```

> Needs a model key. The agent plans, then runs the verbs in "Step by step"
> under the hood. The key-gated verbs (`loom run` / `loom optimize`, the
> pipeline's optimize stage) appear here in prose only -- they are **not** in the
> asserted `run.sh`. In a real session those LLM verbs are exactly what *adds the
> richest trajectories* to the corpus (a full agent rollout with LLM I/O); the
> keyless lifecycle verbs (`eda`, `validate`, ...) also each record a rollout, and
> those are what this keyless example distills.

## Step by step

The explicit, **keyless** verb sequence the agent runs under the hood. Each line
is one `$LOOM <verb> ... --json` with a one-line explanation and the expected
outcome (the field `run.sh` asserts on). First the run isolates + enables the
telemetry corpus (`export LOOM_TELEMETRY=1` + the four `LOOM_*_PATH` vars into a
scratch dir) so only this run's rollouts are seen.

1. **`$LOOM ingest --source <dir> --name <unique> --json`** -- the
   external->Metaflow boundary. Expected: `status == "ok"`, a `pathspec` (the
   `dataset_ref`). Ingest records no rollout; it just lands the data object.
2. **`$LOOM eda --dataset <dataset> --target target --json`** -- a read-only
   profile that **records a `command="eda"` rollout** keyed on this dataset_ref.
   Expected: `status == "ok"`, `summary.nrows > 0`, `leakage_flags == []` (clean
   data). This rollout becomes one trajectory.
3. **`$LOOM ingest --source <dir> --name <unique>-b --json`** -- a **second**
   data object (a distinct pathspec, so a distinct `experiment_id`). Expected:
   `status == "ok"`. (eda + validate on the *same* dataset would collide into one
   trajectory; two datasets give two.)
4. **`$LOOM validate --dataset <dataset-2> --target target --json`** -- fits a
   baseline and **records a `command="validate"` rollout** with a real reward (the
   holdout score / verdict). Expected: `status == "ok"`, `VERDICT in {PASS,
   REVIEW}`. This is the second trajectory.
5. **`$LOOM telemetry status --json`** -- the read-only corpus snapshot. Expected:
   `verb == "telemetry-status"`, `status == "ok"`, `summary.n_trajectories >= 2`,
   and `summary.ip_boundary == {general: >=2, tenant_owned: 0}` -- the IP split.
   No VERDICT (a pure read).
6. **`$LOOM telemetry export --out <file>`** -- distills the trajectories into the
   LOOM-DS-1 SFT corpus. **It has no `--json` flag**, so the recipe parses its
   prose (`examples written: N`, `content : REDACTED (default)`) and reads the
   `--out` JSONL. Expected: `>= 1` example, every row `owned_by == "general"`,
   content redacted (a `<REDACTED:...>` sentinel present, never raw rows), and the
   SFT shape `context` / `teacher_output` / `tools_trajectory` / `reward` /
   `weight`.

_(Optional aside: a key-gated `loom run` / `optimize` step would slot in here in
a real session and add the richest trajectory -- a full agent rollout with the
proxy LLM I/O stitched in. It is omitted from the asserted recipe because it
needs a model key; the keyless rollouts above distill the same way.)_

## What this proves

The moat-capture pipeline holds end to end and keeps its two invariants:

* **Capture -> assemble -> distill works keyless.** Keyless verbs (`eda`,
  `validate`) each record a rollout; `telemetry status` assembles them into
  `>= 2` trajectories; `telemetry export` writes `>= 1` LOOM-DS-1 SFT example with
  the stable shape (`context` / `teacher_output` / `tools_trajectory` / `reward` /
  `weight`).
* **The IP boundary is enforced.** Every exported example is `owned_by ==
  "general"`; `tenant_owned == 0`. Tenant-confidential work never enters the
  cross-tenant set.
* **Prompt hygiene is on by default.** Content is redacted to the
  `<REDACTED:...>` sentinel unless an operator explicitly opts in -- raw rows
  never reach the corpus.

This is the contract `tests/test_examples.py` guards by replaying `run.sh`: a
regression in the telemetry summary shape, the IP-boundary filter, the
default-redaction, or the SFT export shape exits nonzero.
