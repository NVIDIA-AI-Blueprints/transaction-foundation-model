# 02 - Predicting a number: regression

> A beginner-friendly walkthrough of the **core Loom lifecycle on a continuous
> target**. It is the sibling of tutorial
> [`01`](../01-tabular-classification/) (which predicts a *label*); here we
> predict a *number*. Read 01 first if you haven't -- this tutorial highlights
> only what changes.

## The use case -- and why it matters

Lots of real questions are not "which bucket?" but "how much?":

* How many units will this store sell next week?
* What is this apartment worth?
* How many minutes until this delivery arrives?
* What will tomorrow's temperature be?

These are **regression** problems. The thing you predict (`target`) is a
*continuous number* on a scale -- not one of a handful of categories. That one
difference ripples through the whole workflow: you can't talk about
"accuracy" or a "60/40 class balance" anymore, and you need a yardstick that
measures *how far off* your predictions are, in the units of the thing you're
predicting.

The good news: in Loom you run the **exact same verbs** as the classification
tutorial. Loom looks at your target, notices it's a continuous number, and
quietly switches the machinery underneath -- a regression model and a
regression metric -- without you having to say so. This tutorial shows that,
end to end, and explains the one thing you most need to internalize: **the
metric, and which direction is "good," flips.**

## The data

`make_data.py` synthesizes a deterministic, **domain-neutral** table (seeded,
fully offline -- no downloads) using `sklearn.datasets.make_regression`:

* `id` -- a row identifier.
* `feature_0 .. feature_11` -- 12 float features. 6 of them carry real signal
  (they genuinely move the target); the other 6 are pure noise, so a baseline
  has to actually *learn* which features matter.
* `target` -- **a continuous float.** It is a linear combination of the
  informative features plus Gaussian noise. The noise is deliberate: it keeps
  the relationship real but *imperfect*, so the best achievable error is
  greater than zero (a perfect score would be suspicious -- see leakage below).

The single change from tutorial 01 is `target`: there it was an integer label
in `{0, 1}`; here it ranges over hundreds of distinct float values. That is
literally all it takes to make this a regression task -- and to make Loom treat
it as one.

The data is deliberately **clean**: there is no planted leak (a feature that
secretly encodes the answer). So `eda` should report no leakage, and `validate`
should come back with a healthy `PASS` and a sensible, finite error.

## Classification vs. regression -- the one mental shift

| | Classification (tutorial 01) | Regression (this tutorial) |
|---|---|---|
| `target` | a label, e.g. `{0, 1}` | a continuous number |
| Loom's model | a gradient-boosted **classifier** | a gradient-boosted **regressor** |
| The metric | **ROC-AUC** (or accuracy) | **RMSE** (root mean squared error) |
| Range | 0..1 | 0..infinity, in the target's own units |
| Which way is good? | **higher is better** (1.0 is perfect) | **lower is better** (0.0 is perfect) |
| Reading it | "ranks positives above negatives well" | "typical prediction is off by ~this much" |

The headline: **RMSE is an *error*, so smaller is better.** That is the
opposite direction from the ROC-AUC you saw in tutorial 01, and it's the single
most common thing beginners trip on. RMSE also lives *in the units of your
target* -- if you're predicting dollars, an RMSE of 30 means your typical miss
is roughly 30 dollars. (Loom auto-detects regression by a simple rule: a
*numeric* target with *many distinct values* is treated as continuous; a
small handful of distinct values means classification.)

## Ask Loom (the product UX)

The natural-language line you'd type at the `loom` agent:

```
loom "ingest this table, profile it, build features, and validate a regression baseline that predicts target"
```

> **Needs a model key.** The agent plans, then runs the same keyless verbs shown
> below under the hood. The key-gated verbs -- `loom run` / `loom optimize`
> (which would *search for a model that beats this baseline's RMSE*) and the
> agentic NL flow itself -- need an LLM key and **cost money**, so they appear
> here in prose only and are **not** in the runnable `run.sh`.

## Step by step (all keyless)

Each line is one `$LOOM <verb> ... --json` with the outcome `run.sh` asserts on.
`$LOOM` defaults to `/Users/anub/Work/Loom/.venv/bin/python -m loom`.

1. **`$LOOM ingest --source <dir> --name <unique> --json`** -- the one
   external-to-Metaflow boundary; stages `train.csv` into a Metaflow data
   object.
   *Look for:* `status == "ok"` and a `pathspec` (the `IngestDataset/<id>`
   reference everything downstream consumes).

2. **`$LOOM datasets --json`** -- list the catalog of ingested data objects.
   *Look for:* our just-ingested `pathspec` and unique `name` appear in
   `summary["datasets"]` -- proof the data really landed and is named.

3. **`$LOOM eda --dataset <pathspec> --target target --json`** -- a read-only
   profile: shape, dtypes, per-column statistics, and leakage flags.
   *Look for:* `summary["nrows"] > 0`, `summary["target"] == "target"`, and --
   because the target is continuous -- a `summary["numeric_describe"]["target"]`
   block with `mean`/`std`/`min`/`max`. That `std > 0` confirms the target
   actually *varies* (a constant target would be unlearnable). And because the
   data is clean, `summary["leakage_flags"] == []`. (EDA carries no `VERDICT` by
   contract.)
   *New vs. 01:* for a label, EDA shows you a tidy class balance; for a
   continuous target there's no such thing, so you read the target's
   *distribution* (its spread) instead.

4. **`$LOOM features --dataset <pathspec> --target target --json`** -- build
   engineered features into a **new** data object.
   *Look for:* `VERDICT == "BUILT"`, a fresh `pathspec` (`FeaturesFlow/<id>`,
   distinct from the source), `summary["n_features_after"] > 0`, and
   `summary["refused_leakage"] == false`. Feature building is task-agnostic --
   it works the same whether you're heading toward a label or a number.

5. **`$LOOM validate --dataset <features-pathspec> --target target --json`** --
   **the step where regression shows up.** Loom inspects the target, sees a
   continuous number, fits a gradient-boosted **regressor** (instead of a
   classifier), and scores it with **RMSE** via stratified-free K-fold
   cross-validation plus a sealed holdout.
   *Look for:* `status == "ok"`, `VERDICT in ("PASS", "REVIEW")`, and in
   `summary`: `task_type == "regression"`, `metric == "rmse"`, a `cv` block
   with a numeric `mean` (the cross-validated RMSE), a `holdout` block with a
   `score` (the sealed-holdout RMSE), and a `holdout_fraction`. Remember:
   these RMSE numbers are *errors in the target's units* -- **lower is better**,
   and on this clean-but-noisy data they settle around ~30 (the noise floor),
   not near 0.

6. **`$LOOM report --runs <validate-run-pathspec> --json`** -- assemble the
   validate run into a read-only model-card: its metric, success, and lineage.
   *Look for:* `summary["n_runs"] >= 1`, `summary["n_successful"] >= 1`, a
   `best_metric` (here, the holdout RMSE), and our validate run on the
   `leaderboard`. This is how you'd later compare several runs side by side.

_(Optional aside: a key-gated `loom optimize` / `loom run` would slot in after
validate in a real session -- it searches for a model that **lowers** this
baseline's RMSE -- but it's omitted from the runnable recipe because it needs a
model key.)_

## What to expect when you run it

A clean pass. Concretely (numbers are seed-stable but may vary slightly by
sklearn version):

* `eda` -> no leakage; the target's `std` is large (hundreds), confirming a wide
  continuous spread.
* `features` -> `BUILT`, growing 14 columns to ~30 engineered ones.
* `validate` -> `task_type = regression`, `metric = rmse`, **VERDICT PASS**,
  with CV RMSE ~30 +/- ~1 and a sealed-holdout RMSE ~29. (Because we added
  noise on purpose, ~29-30 *is* a good score here -- it's near the irreducible
  floor.)
* `report` -> 1 run, 1 successful, `best_metric` ~29, on the leaderboard.

The script prints `== PASS: 02-regression` and exits 0. If any field in the
`--json` contract drifts -- including a silent flip back to classification --
an assertion fails and the script exits nonzero.

## How to run it

The local Metaflow + minio datastore must be up (the `run.sh` sources
`/tmp/loom-cluster-env.sh` automatically if it's present, so the script is
self-contained):

```bash
cd /Users/anub/Work/Loom
bash tutorials/02-regression/run.sh
```

It generates fresh synthetic data, ingests it under a **unique** name (a
pid+random suffix, so repeat or concurrent runs never collide), runs the keyless
sequence above, asserts every outcome inline, prints clear `ok:`/`PASS` lines,
and cleans up its scratch directory on exit.

## What this proves

The core lifecycle and its `--json` contract stay stable **for a continuous
target**, with regression detected automatically:

* `ingest` lands a readable, named data object that `datasets` then lists.
* `eda` profiles a continuous target as a numeric column (`mean`/`std`/...) and,
  on clean data, flags **no** leakage.
* `features` produces a **new** `FeaturesFlow` pathspec with a `BUILT` verdict.
* `validate` **auto-detects `task_type == "regression"`**, scores with **`rmse`
  (lower-is-better)**, and emits a `PASS`/`REVIEW` VERDICT with both CV and
  sealed-holdout error numbers.
* `report` gathers the run and surfaces its RMSE as `best_metric` on a
  leaderboard.

The same verbs, the same envelope -- just a number instead of a label, and a
metric that points the other way.
