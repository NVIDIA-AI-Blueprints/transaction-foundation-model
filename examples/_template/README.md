<!--
examples/_template/README.md -- the CANONICAL walkthrough template.

Every examples/<NN-name>/README.md uses EXACTLY these four `##` section
headings, in this order:

  ## Use case
  ## Ask Loom
  ## Step by step
  ## What this proves

Fill each section per the guidance below. Keep it domain-neutral. The
"Step by step" verbs must be EXACTLY the keyless verb sequence run.sh asserts;
the only LLM (key-gated) step appears under "Ask Loom" (and optionally as a
clearly-marked aside in "Step by step"), never in the asserted run.sh.
-->

# NN-name -- one-line title

## Use case

What this example demonstrates and the synthetic data behind it. Describe the
generated dataset (`make_data.py`): its shape, columns (generic `feature_*` /
`target` / `event` -- domain-neutral), and the planted structure the verbs are
expected to find (e.g. a leak column, a Markov chain, a distribution shift).
Note it is deterministic + seeded + offline.

## Ask Loom

The natural-language line you would type at the `loom` agent -- the product UX:

```
loom "profile this data and flag any leakage before we model it"
```

> Needs a model key. The agent plans, then runs the verbs in "Step by step"
> under the hood. The key-gated verbs (`loom run` / `loom optimize`, the
> pipeline's optimize stage) appear here in prose only -- they are **not** in
> the asserted `run.sh`.

## Step by step

The explicit, **keyless** verb sequence the agent runs under the hood. Each line
is one `$LOOM <verb> ... --json` with a one-line explanation and the expected
outcome (the VERDICT / summary field run.sh asserts on).

1. **`$LOOM ingest --source <dir> --name <unique> --json`** -- the one
   external->Metaflow boundary. Expected: `status == "ok"`, a `pathspec`
   (the `dataset_ref` everything downstream consumes).
2. **`$LOOM <verb> --dataset <pathspec> ... --json`** -- what it does.
   Expected: `status == "ok"` and `<the summary field / VERDICT this step asserts>`.
3. ... one bullet per verb in the sequence ...

_(Optional aside: a key-gated `loom run`/optimize step would slot in here in a
real session; it is omitted from the asserted recipe because it needs a model
key.)_

## What this proves

The regression invariant this example pins -- stated as the thing that must stay
true for the suite to be green (e.g. "eda always reports `nrows`/`target` and
flags the planted leak; features drops it; validate emits a VERDICT with CV +
holdout numbers"). This is the contract `tests/test_examples.py` guards by
replaying `run.sh`.
