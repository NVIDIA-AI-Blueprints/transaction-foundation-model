# 02-leakage-detection -- catch a planted leak, drop it, validate clean

## Use case

Data readiness and the **`eda -> features` composition gate**. Before you model,
Loom should notice when a column trivially gives away the answer (a "leak") and
*refuse* to engineer it into the feature set. This example plants two leaks, then
shows the lifecycle catching and removing them.

`make_data.py` writes a deterministic, seeded, offline `train.csv` (2000 rows,
domain-neutral columns -- no customer / vertical / PII content):

| column | role |
| --- | --- |
| `id` | benign high-cardinality row index (must **not** be mistaken for a leak) |
| `feature_0 .. feature_5` | honest numeric signal; the first three drive `target` through a logistic link, so a clean model is *moderate*, not perfect |
| `leak_score` | **planted leak** -- a numeric near-duplicate of `target` (target + tiny seeded noise); `|corr|` with the target is ~0.99 |
| `leak_flag` | **planted leak** -- a categorical deterministic relabel of `target` (`"positive"`/`"negative"`); each value maps 1:1 to a class |
| `target` | the binary 0/1 label |

The two leaks trip the two domain-neutral checks Loom's EDA runs: `leak_score`
trips **`near_perfect_predictor`** (absolute Pearson correlation with the target
at/above the 0.98 flag threshold) and `leak_flag` trips
**`duplicate_of_target`** (a feature whose value-groups each map to a single
target value). `id` is high-cardinality, so it is correctly excluded from the
duplicate check -- a leak detector that flagged the row index would be useless.

## Ask Loom

The natural-language line you would type at the `loom` agent -- the product UX:

```
loom "profile this data and flag any leakage before we model it, then build features without the leaks"
```

> Needs a model key. The agent plans, then runs the verbs in "Step by step"
> under the hood. The key-gated verbs (`loom run` / `loom optimize`, the
> pipeline's optimize stage) appear here in prose only -- they are **not** in the
> asserted `run.sh`. In a real session the agent would, after the clean
> `validate` below, hand the leak-free feature set to `loom run` to actually
> search for a model; that step is omitted from the eval bed because it needs a
> key.

## Step by step

The explicit, **keyless** verb sequence the agent runs under the hood. Each line
is one `$LOOM <verb> ... --json` with a one-line explanation and the expected
outcome (the VERDICT / summary field `run.sh` asserts on).

1. **`$LOOM ingest --source <dir> --name <unique> --json`** -- the one
   external->Metaflow boundary. Expected: `status == "ok"` and a `pathspec` (the
   `IngestDataset/<id>` `dataset_ref` everything downstream consumes).
2. **`$LOOM eda --dataset <dataset> --target target --json`** -- read-only
   profile with the target declared, so the leakage check fires. Expected:
   `status == "ok"`, `summary['leakage'] is True`, and
   `summary['leakage_flags']` contains **both** planted columns -- `leak_score`
   (`kind == "near_perfect_predictor"`) and `leak_flag`
   (`kind == "duplicate_of_target"`) -- while `id` is **not** flagged. The
   envelope's `pathspec` is this `EdaFlow/<id>` run, captured for the next step.
3. **`$LOOM features --dataset <dataset> --target target --from <eda-run> --json`**
   -- the composition gate: `--from` points at the EDA run, so `features` reads
   its `leakage_flags` and DROPS exactly those columns before building, then
   writes a NEW data object. Expected: `VERDICT == "BUILT"`,
   `summary['refused_leakage'] is True`, both leaks in
   `summary['dropped_columns']`, and a new `FeaturesFlow/<id>` `pathspec`.
4. **`$LOOM validate --dataset <features> --target target --json`** -- validate
   the leak-free feature set. Expected: `status == "ok"`, a `VERDICT`
   (`PASS` / `REVIEW`), and CV + holdout numbers present -- a *real* baseline
   (here CV ROC-AUC ~0.74, holdout ~0.79), not the ~perfect score the leaks
   would have manufactured.

_(Optional aside: a key-gated `loom run`/optimize step would slot in after step
4 in a real session, searching for a model on the clean features; it is omitted
from the asserted recipe because it needs a model key.)_

## What this proves

The leakage gate holds end to end, as a black-box contract over the `--json`
envelopes:

* `eda --target` always flags a planted leak (`leakage_flags` is non-empty and
  names the leak columns), distinguishes the two leak `kind`s, and does **not**
  false-positive a benign high-cardinality `id`;
* `features --from <eda-run>` honours those flags -- it DROPS the flagged columns
  (`dropped_columns` names them, `refused_leakage is True`) and emits a new
  `FeaturesFlow` data object rather than silently engineering the leak in;
* `validate` on the de-leaked set returns a real VERDICT with CV + holdout
  numbers.

If any of those regress -- a leak slips past EDA, the composition gate stops
dropping it, or the envelope shape drifts -- `run.sh` exits nonzero and
`tests/test_examples.py` (which replays it) fails.
