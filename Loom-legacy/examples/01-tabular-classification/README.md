# 01-tabular-classification -- the core Loom lifecycle on a labeled table

## Use case

The canonical end-to-end lifecycle on a clean, labeled tabular dataset: **ingest
-> profile -> build features -> validate a baseline.** This is the path every
other example branches off of.

`make_data.py` synthesizes a deterministic, **domain-neutral** binary
classification table (seeded, offline -- no downloads) via
`sklearn.datasets.make_classification`:

* `id` -- a row identifier.
* `feature_0 .. feature_19` -- 20 float features; 8 carry real signal, the rest
  are redundant/noise so a baseline is non-trivial but learnable.
* `target` -- the integer label in `{0, 1}`, with a mild 60/40 imbalance so
  ROC-AUC stays meaningful.

The data is deliberately **clean** -- there is no planted leak (that is example
[`02-leakage-detection`](../02-leakage-detection/)). So `eda` should report
`leakage_flags == []` and `validate` should come back with a healthy VERDICT.

## Ask Loom

The natural-language line you would type at the `loom` agent -- the product UX:

```
loom "ingest this table, profile it for leakage, build features, and validate a baseline on target"
```

> Needs a model key. The agent plans, then runs the verbs in "Step by step"
> under the hood. The key-gated verbs -- `loom run` / `loom optimize` (which
> would search for a *better-than-baseline* solution on top of this validated
> baseline), and the agentic NL flow itself -- appear here in prose only; they
> are **not** in the asserted `run.sh`.

## Step by step

The explicit, **keyless** verb sequence the agent runs under the hood. Each line
is one `$LOOM <verb> ... --json` with the expected outcome (the field `run.sh`
asserts on). `$LOOM` defaults to `/Users/anub/Work/Loom/.venv/bin/python -m loom`.

1. **`$LOOM ingest --source <dir> --name <unique> --json`** -- the one
   external->Metaflow boundary; stages `train.csv` into a Metaflow data object.
   Expected: `status == "ok"` and a `pathspec` (the `IngestDataset/<id>` ref
   everything downstream consumes).
2. **`$LOOM datasets --json`** -- list the catalog of ingested data objects.
   Expected: `status == "ok"` and our just-ingested `pathspec` + unique `name`
   appear in `summary["datasets"]` (the data object really landed).
3. **`$LOOM eda --dataset <pathspec> --target target --json`** -- read-only
   profile: shape, dtypes, target balance, top correlations, leakage flags.
   Expected: `status == "ok"`, `summary["nrows"] > 0`, `summary["target"] ==
   "target"`, a non-empty `summary["target_balance"]`, and -- because the data
   is clean -- `summary["leakage_flags"] == []`. (EDA carries no VERDICT;
   `VERDICT` is `null` by contract.)
4. **`$LOOM features --dataset <pathspec> --target target --json`** -- build
   engineered features into a **new** data object.
   Expected: `status == "ok"`, `VERDICT == "BUILT"`, a fresh `pathspec`
   (`FeaturesFlow/<id>`, distinct from the source), and
   `summary["n_features_after"] > 0` with `summary["refused_leakage"] == false`.
5. **`$LOOM validate --dataset <features-pathspec> --target target --json`** --
   fit a gradient-boosted-trees baseline and evaluate it rigorously (stratified
   CV + a sealed holdout).
   Expected: `status == "ok"`, `VERDICT in ("PASS", "REVIEW")`, and a `summary`
   carrying `cv` (with a `mean`), `holdout` (with a `score`), and
   `holdout_fraction`.

_(Optional aside: a key-gated `loom optimize`/`loom run` step would slot in
after validate in a real session -- it searches for a solution that beats this
baseline -- but it is omitted from the asserted recipe because it needs a model
key.)_

## What this proves

The core lifecycle stays wired and its `--json` contract is stable:

* `ingest` lands a readable, named data object that `datasets` then lists.
* `eda` always reports `nrows`/`ncols`/`target` and -- on clean data -- flags
  **no** leakage.
* `features` always produces a **new** `FeaturesFlow` pathspec (distinct from
  the source) with a `BUILT` verdict and a positive feature count.
* `validate` always emits a `PASS`/`REVIEW` VERDICT with both **CV** and
  **sealed-holdout** numbers present.

`tests/test_examples.py` guards this by replaying `run.sh` and asserting exit 0;
any drift in those fields exits nonzero.
