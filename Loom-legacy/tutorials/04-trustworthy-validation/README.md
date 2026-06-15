# Tutorial 04 — Validation You Can Trust + a Model Card

> A hands-on, **keyless** Loom tutorial. Everything here runs against the live
> local Metaflow + minio datastore with **no API key** and **no cost**.

## What you'll learn

A model that scores 0.95 on "the test set" is worthless if that test set leaked,
or if the score collapses on a slice of users you didn't look at, or if the
predicted probabilities are wildly over-confident. **Trustworthy validation is
not one number — it's a bundle of evidence.**

In this tutorial you'll take a tabular classification dataset that carries a
**sensitive attribute** (a `group` column) and put it through Loom's rigorous
evaluation, then assemble a **model card** you could hand to a reviewer:

1. **`loom ingest`** — register the dataset as a versioned Metaflow data object.
2. **`loom validate --sensitive group`** — the heart of the tutorial. One verb
   runs *all* of the following and returns them in a single JSON envelope:
   - **Cross-validation (CV)** — stratified K-fold, reported as a mean **and a
     standard deviation** so you can see fold-to-fold stability, not just a
     lucky split.
   - **A sealed holdout** — a slice set aside and *never* touched by CV, scored
     once at the end. If the holdout score tracks the CV mean, you didn't
     overfit your evaluation.
   - **Probability calibration** — reliability bins + a **Brier score**. This
     tells you whether "the model said 0.8" actually means "happens 80% of the
     time."
   - **Fairness / per-slice metrics** — the score computed *separately* for each
     value of your sensitive column, so disparate performance is visible
     instead of averaged away.
   - **Leakage checks** — automatic flags for features that are suspiciously
     predictive of the label (a near-perfect "feature" is usually the answer in
     disguise).
   - A single **`VERDICT`** (`PASS` / `REVIEW` / `FAIL`) that summarizes whether
     the evidence holds together.
3. **`loom report`** — assemble the validated run into a **model card** (an
   `@card` HTML artifact) with the metrics, lineage, and leaderboard.

## Why it matters

The hard failures in production ML are almost never "the model wasn't accurate
enough." They're "we measured accuracy wrong." A leaked column, an
un-stratified split, a score that's really just the majority class, a model
that's accurate overall but 15 points worse for one group — each of these
passes a naive `model.score(X_test, y_test)` and then blows up later. Loom's
`validate` is designed so the *evidence travels with the model*: the same JSON
that says `PASS` also carries the CV spread, the sealed-holdout number, the
calibration curve, the per-slice scores, and the leakage flags. The model card
makes that evidence reviewable by a human.

## The data (synthetic, deterministic, generated inline)

`run.sh` writes a small CSV — no downloads — with:

| Column        | Meaning                                                              |
|---------------|----------------------------------------------------------------------|
| `feature_0..4`| Informative numeric features that drive a logit.                     |
| `group`       | The **sensitive attribute** (`"A"` / `"B"`), drawn *independently* of the label — so it is not predictive and is not a leak. This is the column we slice fairness on. |
| `target`      | A binary label: a seeded Bernoulli draw on the logit, so the signal is **real but not perfect** (a healthy `PASS`, not a suspiciously perfect score). |

Because the label is a *probabilistic* function of the features, an honest
baseline lands around ROC-AUC ≈ 0.81 — strong enough to be useful, far from the
~1.0 you'd see if a column had leaked.

## Run it

```bash
cd /Users/anub/Work/Loom
bash tutorials/04-trustworthy-validation/run.sh
```

The script is self-contained: it sources `/tmp/loom-cluster-env.sh` if present,
generates the data, ingests it under a **unique** dataset name (so repeat or
concurrent runs never collide), runs the verbs with `--json`, asserts every
signal inline, prints a line of `ok:` checks, and ends with `== PASS`. It
**exits non-zero** on any regression and cleans up its scratch directory on
exit.

## Step by step — what to look for

### 1. Ingest

```bash
loom ingest --source <dir> --name <unique-name> --json
```

Registers the CSV as a Metaflow data object and returns a `pathspec` like
`IngestDataset/1781045768637407`. Everything downstream refers to the dataset by
this pathspec, so the data is versioned and the evaluation is reproducible.
**Look for:** `status: "ok"` and a non-empty `pathspec`.

### 2. Validate (the rigorous evaluation)

```bash
loom validate --dataset <pathspec> --target target --sensitive group --json
```

This is a Metaflow run that fits a gradient-boosted-trees **baseline** and
subjects it to the full battery. The `--json` envelope's `summary` carries each
signal — here's what the tutorial asserts and what each one tells you:

| `summary` key        | What it is                                              | What "good" looks like |
|----------------------|---------------------------------------------------------|------------------------|
| `cv.scores` / `cv.mean` / `cv.std` | Stratified K-fold scores, their mean, and their spread. | A small `std` relative to the `mean` (stable across folds). |
| `holdout.score` / `holdout.n`      | Score on the sealed holdout + its size.                 | `holdout.score` close to `cv.mean` (no eval overfit). |
| `calibration.bins` / `calibration.brier` | Reliability curve + Brier score.                | Predicted probabilities track observed frequencies; lower Brier is better. |
| `slice_metrics.A` / `slice_metrics.B`     | **Fairness**: the score *per sensitive-group value*. | The two slice scores are close to each other. |
| `leakage_flags` / `leakage`        | Columns flagged as suspiciously predictive.             | An **empty** list and `leakage: false` on a clean dataset. |
| `VERDICT` (top level)              | The overall call: `PASS` / `REVIEW` / `FAIL`.           | `PASS` when the evidence holds together. |

In this tutorial the clean, honest dataset yields a `PASS`: CV ≈ 0.81 with a
tight std, a holdout that tracks it, both fairness slices within a couple of
points, and **zero** leakage flags. The human-readable summary on stderr prints
all of this at a glance:

```
  CV roc_auc : 0.815137 +/- 0.01333 (5-fold)
  holdout     : 0.806806 (sealed, n=320)
  calibration : Brier=0.191698
  fairness    : A=0.815599, B=0.795946
  VERDICT     : PASS
```

> **Try it:** add a near-perfect column (e.g. `df["leak"] = target + tiny_noise`)
> and re-run. You'll see `leakage_flags` name that column and the `VERDICT` drop
> to `REVIEW` — exactly the safety net you want before anything ships. (Tutorial
> 02 and example `04-validate-and-gated-deploy` walk through the leak branch.)

### 3. Report (the model card)

```bash
loom report --runs <validate-run-pathspec> --json
```

Assembles the validated run's metrics, tags, lineage, and a leaderboard into a
model-card `@card`. **Look for:** `VERDICT: "OK"`, a non-empty `card_path`
(the HTML model card), `n_successful: 1`, and your validate run listed in the
`leaderboard`. Open the `card_path` HTML to read the card.

## Next step — the optimize loop (needs an LLM key, NOT run here)

This tutorial validates a **baseline**. The natural next move is to let Loom's
agentic search propose and evaluate better solutions, then validate the winner
the same rigorous way and gate a deploy on its `VERDICT == PASS`:

```bash
# NEEDS AN LLM KEY + COSTS MONEY — described here, intentionally not in run.sh:
loom optimize --dataset <pathspec> --goal "maximize roc_auc" --metric roc_auc
loom validate --dataset <pathspec> --solution <optimize-run> --sensitive group
loom deploy   --validate <validate-run>          # gates on the PASS verdict
```

Because `optimize` / `run` drive an LLM-based search, they require an API key
and incur cost, so they are kept out of the keyless, free-to-run `run.sh`. The
validation and model-card steps above are exactly what you'd run on the
optimizer's winning solution — the trust machinery doesn't change.

## Verbs exercised by `run.sh`

`ingest` → `validate` (`--sensitive`) → `report`. All keyless, all asserted.
