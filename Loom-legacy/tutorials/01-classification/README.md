# 01 · Your first model: tabular classification

> A beginner-friendly, end-to-end walkthrough of the Loom lifecycle on a labeled
> table. Everything here is **keyless** — it runs against your live local
> Metaflow + minio datastore with no API key and no cost. Copy-paste the commands,
> or just run the tested script at the bottom.

## What you'll build

You have a table. Each row is an example, and one column — the **target** — is a
yes/no label you want to predict for *future* rows. That is **binary
classification**, the single most common task in applied data science: will this
customer churn? is this transaction fraud? did the patient respond? will the
lead convert?

By the end you'll have taken a raw table all the way to a **validated baseline
model with a model-card**, using six Loom verbs in sequence. You'll learn what
each verb does, what to look for in its output, and where the (key-gated) "make
it better" step would slot in.

## Why this lifecycle matters

The hard part of classification isn't fitting a model — `scikit-learn` does that
in one line. The hard part is *trusting the number*. A model that scores 0.99 in
a notebook and 0.62 in production usually got there one of two ways:

- **Leakage** — a feature that secretly encodes the answer (a column derived from
  the target, a timestamp that only exists after the outcome is known). It looks
  brilliant offline and collapses live.
- **Optimistic evaluation** — scoring on data the model effectively trained on,
  so the number reflects memorization, not generalization.

Loom's lifecycle is built to catch both *before* you ship: it profiles for
leakage, engineers features in a leakage-aware way, and validates with
cross-validation **plus a sealed holdout** the model never touches. Each step
emits a structured `--json` result you (or a script) can assert on.

## The data

We synthesize a small, clean, deterministic table inline — **no downloads** — with
`sklearn.datasets.make_classification`:

| column                | meaning                                                            |
| --------------------- | ------------------------------------------------------------------ |
| `id`                  | a row identifier (carries no signal)                               |
| `feature_0 … feature_11` | 12 float features; ~5 carry real signal, the rest are redundant/noise |
| `target`              | the integer label in `{0, 1}` — what we predict                    |

It's 1,500 rows with a mild **~60/40 class imbalance** (so ROC-AUC stays
meaningful) and a touch of label noise (`flip_y=0.02`, so a perfect score is
impossible and the baseline is honest). It is deliberately **clean** — no planted
leak — so EDA should report *no leakage* and validation should come back healthy.
A fixed seed makes every run byte-identical.

## Setup

The keyless verbs talk to your already-running local Metaflow + minio datastore.
Make that environment available, then call Loom via the venv:

```bash
source /tmp/loom-cluster-env.sh
LOOM="/Users/anub/Work/Loom/.venv/bin/python -m loom"
```

Every verb below takes `--json`, which prints one machine-readable object to
stdout (human-readable prose goes to stderr). The stable fields you'll see are
`status`, `VERDICT`, `pathspec` (a `Flow/<run-id>` reference to the Metaflow run
this verb produced), and `summary` (the verb's typed payload).

## Step by step

### 1. `ingest` — bring the table into Loom

```bash
$LOOM ingest --source <dir-with-train.csv> --name my-first-classifier --json
```

This is the one boundary that crosses from "a file on disk" into Loom: it stages
your CSV into a **Metaflow data object** that every downstream verb consumes.

**What to look for:** `status == "ok"` and a `pathspec` like
`IngestDataset/1781045646677783`. *Copy that pathspec* — it's the handle you pass
to the next verb. Give each ingest a unique `--name` so it's easy to find in the
catalog (`$LOOM datasets --json` lists everything you've ingested).

### 2. `eda` — profile it and scan for leakage

```bash
$LOOM eda --dataset IngestDataset/<id> --target target --json
```

A **read-only** profile: shape, dtypes, **target balance**, top correlations, and
a leakage scan. Declaring `--target` is what arms the leakage check — Loom looks
for columns that predict the target *too* well (a near-perfect predictor, a
duplicate of the label).

**What to look for:**

- `summary.nrows` / `summary.ncols` — does the shape match what you ingested?
- `summary.target_balance` — e.g. `0=905, 1=595`. Severe imbalance changes how
  you'd read the score later.
- `summary.leakage_flags` — on our clean data this is **empty `[]`**. If it
  weren't, those columns would be your prime suspects, and the next step would
  drop them. (EDA is read-only, so it carries no `VERDICT` — that's by design.)

Keep EDA's `pathspec` (an `EdaFlow/<id>` run) — the next step composes on it.

### 3. `features` — engineer a new, leakage-aware feature object

```bash
$LOOM features --dataset IngestDataset/<id> --target target --from EdaFlow/<id> --json
```

This builds engineered features into a **brand-new data object** — your source
table is never mutated. The `--from <eda-run>` is the important part: it makes
feature-building **leakage-aware**. Features reads the upstream EDA profile's
`leakage_flags` and *refuses* (drops) anything that was flagged **before** it
builds, so a leak caught in step 2 can't sneak back in here.

**What to look for:**

- `VERDICT == "BUILT"` — the feature build succeeded.
- A fresh `pathspec` (`FeaturesFlow/<id>`) that is **different** from the source —
  proof it wrote a new object rather than editing yours in place.
- `summary.n_features_after` grew (in our run, 14 → 30 columns).
- `summary.refused_leakage == false` — on clean data nothing had to be dropped.
  On a *leaky* table this would be `true` with the dropped columns listed.

Hand the new `FeaturesFlow/<id>` pathspec to validation.

### 4. `validate` — a rigorous, honest baseline

```bash
$LOOM validate --dataset FeaturesFlow/<id> --target target --json
```

This fits a gradient-boosted-trees **baseline** and grades it the way a skeptic
would: **stratified cross-validation** (so every row gets a turn as test data)
**plus a sealed holdout** that the CV never saw. Two independent reads on
generalization — if they disagree, something's off.

**What to look for:**

- `VERDICT` — `PASS` (healthy) or `REVIEW` (look closer; e.g. score too low, or
  leakage detected at validation time).
- `summary.cv.mean` — the cross-validated score (our run: ROC-AUC ≈ 0.965).
- `summary.holdout.score` — the sealed-holdout score (≈ 0.961). It being *close*
  to the CV mean is the good sign — the model generalizes, it didn't memorize.
- `summary.holdout_fraction` — how much data was sealed away for that final check.

This is your **baseline**: the number every fancier model must beat to earn its
complexity.

### 5. `report` — assemble a model-card

```bash
$LOOM report --runs ValidateFlow/<id> --json
```

A **read-only** assembly step: it gathers one or more runs — their pathspecs,
metrics, and lineage — into a single structured **model-card** (and a Metaflow
`@card` you can open in a browser). It trains nothing and writes nothing back; it
just stitches together what already happened so you have one artifact to share or
archive. Pass several runs as `--runs A,B,C` to get a leaderboard across them.

**What to look for:**

- `VERDICT == "OK"` — at least one successful run made it into the card.
- `summary.n_runs` / `summary.n_successful` — how many runs were assembled.
- `summary.best_run` and `summary.leaderboard` — your validated baseline should be
  there, with its metric.

## What to expect when you run it

A healthy run profiles 1,500 rows with no leakage, grows the feature set, and
validates to **`PASS`** with CV and holdout ROC-AUC both around **0.96** and in
close agreement. The report comes back **`OK`** with your baseline on the
leaderboard. (Exact decimals vary with library versions; the *shape* — PASS, no
leakage, agreeing CV/holdout — is what matters.)

## Run the whole thing (tested)

`run.sh` is the executable, self-checking version of this walkthrough. It
generates the data, runs all six verbs with `--json`, **asserts every outcome**,
prints a clear `PASS`/`FAIL`, and exits nonzero on any regression. It uses a
unique dataset name each run, so you can run it repeatedly (or alongside others)
without collisions, and it cleans up its scratch directory on exit.

```bash
cd /Users/anub/Work/Loom
bash tutorials/01-classification/run.sh
```

You should see a stream of `ok:` lines and, at the end:

```
== PASS: tutorial 01-classification
```

## Next step — make it *better* (needs an LLM key)

Everything above is keyless and free. The natural next move — searching for a
model that **beats** your validated baseline — is the one part that needs a model
key, because it drives an LLM-guided search (AIDE) over candidate solutions:

```bash
# Needs an LLM key — costs money. NOT run by this tutorial.
$LOOM optimize --dataset FeaturesFlow/<id> --target target \
  --goal "beat the baseline ROC-AUC on target" --metric roc_auc
```

In the product, you'd just say it in natural language and the agent plans and
runs the verbs for you:

```bash
loom "ingest this table, check it for leakage, build features, validate a baseline,
      then search for something that beats it"
```

Under the hood that's exactly steps 1–5 above (keyless), followed by the key-gated
`optimize`/`run` search. The discipline you practiced here — profile, de-leak,
validate against a sealed holdout — is what keeps that search **honest**: it has a
real baseline to beat and a holdout it can't cheat on. Start keyless; reach for the
key only when you have a number worth beating.

## See also

- [`examples/01-tabular-classification`](../../examples/01-tabular-classification/)
  — the terse regression-eval version of this same lifecycle.
- [`examples/02-leakage-detection`](../../examples/02-leakage-detection/) — what
  steps 2–3 look like when the data *does* contain a planted leak.
- [`examples/04-validate-and-gated-deploy`](../../examples/04-validate-and-gated-deploy/)
  — gating a deploy on the `validate` VERDICT.
