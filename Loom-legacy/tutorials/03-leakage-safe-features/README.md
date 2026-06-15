# 03 - Engineering features without leaking

A beginner-friendly walkthrough of **leakage-aware feature engineering** with Loom.
You will plant a leak in a dataset on purpose, watch Loom *catch* it, watch it get
*dropped* before any model sees it, and confirm the cleaned-up features give you an
**honest** score instead of a fake one.

Everything here is **keyless** -- it runs against the live local Metaflow + minio
datastore and never calls a paid model. (The one model-using step, searching for an
actual model, is described at the end as a next step but is *not* run.)

---

## What is "leakage", and why should you care?

**Data leakage** is when a feature in your training table secretly contains
information you would *not* have at prediction time. The classic giveaway: a column
whose value is only known *after* the thing you are trying to predict has already
happened.

If a leaky column sneaks into your features:

- Your offline evaluation looks **incredible** -- near-perfect accuracy, an ROC-AUC
  pushing 1.0. It feels like you nailed it.
- Then you ship the model and it is **useless**, because that magic column simply
  does not exist when you score a fresh, real example. The model was "predicting" the
  past from the future.

Leakage is one of the most common and most expensive mistakes in applied ML, and it
is sneaky precisely because the metrics *look great*. The goal of this tutorial is to
make catching it boring and automatic.

---

## The scenario and the data

We use a small, made-up **loan-default** table. We are trying to predict
`defaulted` (did this loan go bad: 1 = yes, 0 = no) from information available *at the
time someone applies*.

`make_data.py` writes a deterministic, seeded, offline `train.csv` (2000 rows -- no
real people, no PII, no network access):

| column | role | known at application time? |
| --- | --- | --- |
| `application_id` | benign row index, unique per row | n/a (an ID, carries no signal) |
| `income`, `loan_amount`, `credit_score`, `age`, `debt_ratio` | honest application-time features; a few drive `defaulted` through a logistic link | **yes** |
| `recovery_amount` | **PLANTED LEAK** -- money a collections team recovered *after* a default; built as `defaulted` + a tiny bit of noise so `|corr|` with the target is ~0.99 | **no** -- only exists after a default |
| `collections_status` | **PLANTED LEAK** -- a back-office status (`in_collections` / `current`) stamped on the account *after* the outcome; a 1:1 relabel of the target | **no** -- only exists after the outcome |
| `defaulted` | the binary 0/1 target | -- |

The two leaks are the kind you meet in real life: `recovery_amount` and
`collections_status` only get populated *because* a loan defaulted. At the moment you
score a brand-new application they are blank. Train on them and your model essentially
reads the answer off the page.

`application_id` is the **control**: it is unique per row, so a naive "is this column
suspicious?" rule might fear it -- but it carries no information about the target, so a
good leakage detector must leave it alone. We assert that Loom does.

---

## The lifecycle, step by step

These are the four **keyless** verbs the tutorial runs, in order. Each is one
`loom <verb> ... --json`; `run.sh` parses the JSON and asserts the outcome, so if any
step regresses the script fails loudly.

### 1. `ingest` -- bring the data into Metaflow

```
loom ingest --source <dir> --name <unique-name> --json
```

This is the one place raw files cross into Loom. It writes a Metaflow **data object**
and hands you back a `pathspec` (something like `IngestDataset/1781045704132479`) that
every later verb consumes.

**What to look for:** `status == "ok"` and a non-empty `pathspec`. The tutorial uses a
unique, timestamped dataset name so repeat or concurrent runs never collide.

### 2. `eda` -- profile the data and surface the leak

```
loom eda --dataset <dataset> --target defaulted --json
```

`eda` is read-only -- it profiles the table and never modifies it. The important part:
**declare the target** with `--target defaulted`. That switches on Loom's leakage
check, which compares every column against the target and flags any that are
suspiciously predictive.

In the JSON envelope, look at `summary`:

- `summary['leakage']` flips to `true` -- a leak is present.
- `summary['leakage_flags']` is a list, one entry per suspect column, each with a
  `column` name and a `kind`:
  - `recovery_amount` trips **`near_perfect_predictor`** -- its absolute correlation
    with the target is ~0.99 (the run prints `|corr| with target = 0.9992`).
  - `collections_status` trips **`duplicate_of_target`** -- 100% of its values map to a
    single target class.
- `application_id` is **not** in the list -- the benign ID is correctly left alone.

The envelope's `pathspec` is this EDA run (e.g. `EdaFlow/...`). We capture it, because
the next step builds *on top of it*.

> This is the moment leakage gets caught -- **before** any feature engineering or
> modeling, while it is still cheap to fix.

### 3. `features --from <eda-run>` -- build features, drop the leaks

```
loom features --dataset <dataset> --target defaulted --from <eda-run> --json
```

This is the **composition gate**, the heart of the tutorial. By passing
`--from <eda-run>`, you tell `features` to read the upstream EDA run's
`leakage_flags` and **drop exactly those columns** before building anything. It then
engineers features from the *remaining, honest* columns and writes a **new** clean
data object.

In the JSON envelope:

- `VERDICT == "BUILT"` -- the feature set was produced.
- `summary['refused_leakage'] is True` -- it actively refused to engineer the leaks.
- `summary['dropped_columns']` lists both `recovery_amount` and `collections_status`.
- A fresh `pathspec` (e.g. `FeaturesFlow/...`) -- the leak-free data object the next
  step validates.

The run prints it plainly:
`LEAKAGE: dropped 2 flagged column(s): collections_status, recovery_amount`.

> Without `--from`, `features` would happily engineer the leaky columns straight into
> your model inputs. The `eda -> features` handoff is what makes leakage-removal
> automatic instead of a thing you hope you remembered.

### 4. `validate` -- confirm the clean features score honestly

```
loom validate --dataset <features> --target defaulted --json
```

Finally, validate the **leak-free** feature set. This runs cross-validation plus a
sealed holdout and emits a VERDICT.

In the JSON envelope:

- `status == "ok"` and a `VERDICT` of `PASS` or `REVIEW`.
- `summary['cv']` and `summary['holdout']` carry real numbers.

**What to look for -- the punchline:** the score is *believable*, not perfect. On this
synthetic data you will see something like CV ROC-AUC ~0.86 and a similar holdout. That
is a genuinely-good-but-not-magical model. Had the leaks survived, you would have seen
a near-1.0 score that would have collapsed the instant the model met real applications.
An honest 0.86 you can ship beats a fake 0.99 you cannot.

---

## What to expect when you run it

`run.sh` prints a running log of each verb plus an `ok:` line for every assertion it
checks, and ends with:

```
== PASS: 03-leakage-safe-features
```

and exits `0`. If anything regresses -- a leak slips past `eda`, the gate stops
dropping it, or the JSON shape drifts -- it prints an `ASSERT FAIL: ...` line and exits
nonzero. That is the point: this tutorial is also a test.

---

## How to run it

The local Metaflow + minio datastore should already be up. From the repo root:

```bash
cd /Users/anub/Work/Loom
bash tutorials/03-leakage-safe-features/run.sh
```

The script is self-contained: it sources `/tmp/loom-cluster-env.sh` (the datastore
endpoint + credentials) if present, generates the data inline, uses a unique dataset
name, and cleans up its scratch directory on exit.

---

## Next step (needs an LLM key -- not run here)

Once you trust the clean feature set, the natural next move is to actually **search for
a model** on it:

```
loom run --dataset <features> --goal "predict loan default" --metric roc_auc
```

`loom run` (and `loom optimize`, and the agent's natural-language flow) drive a real
model search through an LLM, so they **require an API key and cost money**. They are
intentionally left out of this tutorial's tested script. The keyless lifecycle above is
exactly the data-readiness work you want to finish *first* -- so that when you do spend
on a model search, you are searching over honest, leak-free features instead of
manufacturing a score you cannot ship.

## The takeaway

Catch leakage in `eda`, drop it automatically via the `eda -> features` handoff, and
let `validate` confirm an honest baseline -- all before you spend a cent on a model.
That is the whole discipline of leakage-safe feature engineering, made routine.
