# 06 - Monitoring data drift in production

A beginner-friendly tutorial on Loom's **monitoring** tier: how to watch the
health of your runs and catch when a live data feed has **drifted** away from
the data your model was built on.

By the end you will have run two monitoring checks against a real local
Metaflow + minio datastore -- with **no model key and no cost** -- and you will
understand exactly what each one tells you.

---

## Why this matters

A model is only as good as the data it sees. You train and validate it on one
distribution (say, last month's traffic), ship it, and then the world moves on:
a new sensor comes online, a marketing push changes who shows up, a currency
re-denominates, an upstream bug doubles a column. The model keeps returning
confident answers -- but they are quietly getting worse, because the inputs no
longer look like what it learned from. This silent failure mode is called
**data drift**, and catching it early is the whole job of production monitoring.

Loom gives you two read-only monitoring views, both reachable through one verb,
`ops`:

1. **Run health** -- "are my pipelines actually finishing, or are runs failing?"
2. **Data drift** -- "does this new batch of data still look like the baseline I
   trust, or has its distribution moved?"

Both are in Loom's safe-by-default **read-only tier**: they train nothing, write
nothing back, and never need an LLM key. They only *read* finished runs and
data-object schemas through the Metaflow Client API. That makes them perfectly
suited to run on a schedule, in CI, or as a pre-deploy gate.

---

## The data

This tutorial generates its own data inline -- no downloads -- so it is fully
reproducible. It builds **two frames with the identical schema**:

| column                | meaning                                            |
| --------------------- | -------------------------------------------------- |
| `id`                  | row identifier                                     |
| `feature_0 .. feature_7` | eight numeric features                          |
| `target`              | a binary label in `{0, 1}`                         |

- **`reference/train.csv` (v1)** -- the **baseline**. Features are drawn from a
  standard normal distribution (mean 0, spread 1). This represents the data your
  model was built and validated on -- your "known good" snapshot.

- **`current/train.csv` (v2)** -- a **new production batch** with the *same
  columns* but a **planted distribution shift**: every `feature_*` column gets a
  `+3.0` mean offset and a `1.8x` spread inflation. The shift is deliberately
  large -- well past Loom's relative-mean-shift threshold (`0.25`) -- so the
  drift check fires clearly and unambiguously.

Both frames are seeded, so a repeat run is byte-identical, and they use distinct
derived seeds so they are independent draws (not the same rows shifted). The
schema is intentionally **domain-neutral** (`feature_*` / `target`) -- the point
is the *shape of the data*, not any particular vertical.

> In real life, v1 and v2 are the same pipeline's output a week (or a deploy)
> apart: v1 is the snapshot you trusted, v2 is what is flowing through right now.

---

## The verb steps, explained

Each step below is one keyless `loom <verb> ... --json` call. The `--json` flag
makes the verb print a machine-readable envelope (`status`, `VERDICT`,
`pathspec`, `summary`, ...) on top of the human-readable output, which is what
`run.sh` parses and asserts on.

The entrypoint is the venv module form -- substitute your own if you have the
`loom` console script on your PATH:

```bash
LOOM="/Users/anub/Work/Loom/.venv/bin/python -m loom"
```

### 1. Ingest the reference (v1)

```bash
$LOOM ingest --source <work>/reference --name <unique>-v1 --json
```

`ingest` is the one boundary where external files cross into Metaflow: it reads
the CSV and stores it as an immutable **data object**, handing you back a
`pathspec` like `IngestDataset/1781045726396495`. That pathspec is the stable
handle every later verb uses.

**What to look for:** `status == "ok"` and a non-empty `pathspec`. That pathspec
is your **reference handle** -- the baseline you will compare against.

### 2. Ingest the current batch (v2)

```bash
$LOOM ingest --source <work>/current --name <unique>-v2 --json
```

Exactly the same, for the new (shifted) batch. It becomes its own independent
data object with its own pathspec -- your **current handle**.

**What to look for:** `status == "ok"` and a second, distinct `pathspec`.

### 3. Land one run so run-health has something to count

```bash
$LOOM validate --dataset <reference> --target target --json
```

`ops --flow` reports on *finished runs*, so we first need at least one finished
run to exist. `validate` is keyless (it does cross-validated, leakage-checked
model validation without any LLM), so we use it purely to **land one real
`ValidateFlow` run** on the baseline. (It is also a genuinely useful baseline
check -- you will see a `VERDICT` of `PASS` or `REVIEW` and cross-validated
metrics.)

**What to look for:** `status == "ok"` and a `VERDICT`.

### 4. Run health: `ops --flow`

```bash
$LOOM ops --flow ValidateFlow --json
```

This is the **run-health** view. It reads recent `ValidateFlow` runs through the
Metaflow Client API and rolls them up into a health block under
`summary.health`:

```jsonc
"health": {
  "flow_name": "ValidateFlow",
  "n_runs": 46,
  "n_successful": 45,
  "n_failed": 1,
  "success_rate": 0.978,
  "status": "DEGRADED"   // OK | DEGRADED | EMPTY
}
```

**What to look for:**

- `summary.health` is scoped to the flow you asked about (`flow_name`).
- The counts **reconcile**: `n_runs == n_successful + n_failed`.
- The overall `VERDICT` is `OK`, `ATTENTION`, or `EMPTY`. (Failed runs in the
  history pull health to `DEGRADED`, which surfaces as an `ATTENTION` verdict --
  that is the monitor doing its job.)

This is your "are my pipelines healthy?" dashboard line, on demand.

### 5. Data drift: `ops --dataset ... --reference ...`

```bash
$LOOM ops --dataset <current-v2> --reference <reference-v1> --json
```

This is the headline of the tutorial. `ops` materializes **both** data objects,
computes per-column summary statistics for each, and compares them. Because v2's
numeric features all moved far past the threshold, it reports drift under
`summary.drift`:

```jsonc
"drift": {
  "n_shared_columns": 10,
  "added": [],
  "removed": [],
  "drift_flags": [
    { "column": "feature_0", "kind": "mean_shift",
      "detail": "mean 0.0087 -> 3.040 (rel 347.9)", "ref": 0.0087, "cur": 3.040 },
    { "column": "feature_1", "kind": "mean_shift", "...": "..." }
    // ... one flag per drifted column
  ],
  "drift": true,
  "status": "DRIFT"
}
```

**What to look for:**

- `n_shared_columns` -- how many columns the two frames have in common (the
  comparison is like-for-like). `added` / `removed` would list schema changes.
- `drift == true` and `status == "DRIFT"` -- the headline verdict.
- `drift_flags` -- **one entry per drifted column**, each with the `kind` of
  drift (`mean_shift`), a human-readable `detail`, and the `ref` (baseline) vs
  `cur` (current) statistic so you can see *how much* it moved.
- The top-level `VERDICT` degrades to `ATTENTION` -- a single field your
  automation can gate on.

> If the data had *not* drifted, you would instead see `drift == false`,
> `status == "OK"`, an empty `drift_flags` list, and an `OK` verdict.

---

## What to expect when you run it

The script prints a running log of each verb, a green `ok:` line per assertion,
and finishes with:

```
== PASS: 06-monitoring-drift
```

and exit code `0`. If any check regresses -- ingest fails, run-health counts
don't reconcile, or the drift isn't caught -- it prints an `ASSERT FAIL` with the
offending JSON and **exits nonzero**. That is the contract: a green run means the
whole monitoring path is intact.

---

## How to run it

The local Metaflow + minio datastore must already be up (the script sources
`/tmp/loom-cluster-env.sh` for you if it exists, so it is self-contained):

```bash
cd /Users/anub/Work/Loom
bash tutorials/06-monitoring-drift/run.sh
```

It uses a **unique** dataset name (timestamp + pid + random suffix) on every run,
so you can run it repeatedly or concurrently without collisions, and it cleans up
its scratch directory on exit.

---

## Next step (needs an LLM key)

Detecting drift is the *signal*; the *response* is to rebuild against the new
distribution. In a real session you would follow a `DRIFT` finding by asking the
Loom agent to act on it -- for example:

```
loom "my latest batch has drifted from the baseline -- re-validate on the new
      data and, if it holds up, rebuild the model against it"
```

The agent would then reach for the key-gated verbs (`loom run` / `loom optimize`,
the pipeline's optimize stage) that search for and build a model. **Those verbs
call an LLM and cost money, so they are intentionally NOT part of this keyless,
self-checking tutorial.** Everything in `run.sh` stays in the free, read-only
monitoring tier -- exactly the part you would safely run on a schedule.
