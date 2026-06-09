# 04-validate-and-gated-deploy -- the cross-verb exit gate (safe-by-default deploy)

## Use case

This example demonstrates Loom's **cross-verb exit gate**: `loom deploy` will not
promote a model unless the upstream `loom validate` run earned a `PASS` VERDICT,
and even then the real external action is **OFF by default**. A sub-threshold
validation (`REVIEW` / `FAIL`, or any leakage) **BLOCKS** the deploy. This is the
machine-checkable safety boundary between "we evaluated something" and "we shipped
something."

The generator (`make_data.py`) is deterministic, seeded, and offline, and writes
**two** domain-neutral binary-classification variants under the scratch dir:

- **`clean/train.csv`** -- `feature_0..feature_5` drive a moderate-strength logit;
  the `target` is a seeded Bernoulli draw on it. A baseline learns a real (not
  perfect) signal, so `validate` returns **`PASS`**.
- **`leaky/train.csv`** -- the same frame plus a planted `leak_feature` column
  equal to the target plus tiny deterministic jitter (a near-perfect predictor,
  `|corr| >= 0.98`, the validate leakage threshold). `validate` flags it and
  returns **`REVIEW`**.

The two variants exercise the two branches of the gate (ALLOW and BLOCK) in one run.

## Ask Loom

The natural-language line you would type at the `loom` agent -- the product UX:

```
loom "validate this candidate and only stage a deploy if it passes"
```

> Needs a model key. The agent plans, then runs the verbs in "Step by step"
> under the hood. The key-gated verbs (`loom run` / `loom optimize`, the
> pipeline's optimize stage) appear here in prose only -- they are **not** in
> the asserted `run.sh`. (A real session might first `loom optimize` to produce a
> candidate solution, then `loom validate --solution <run>` instead of the
> baseline; the gate logic is identical.)

## Step by step

The explicit, **keyless** verb sequence the agent runs under the hood. Each line
is one `$LOOM <verb> ... --json` with a one-line explanation and the expected
outcome (the VERDICT / gate / summary field `run.sh` asserts on).

1. **`$LOOM ingest --source <dir>/clean --name <unique>-clean --json`** -- the
   external->Metaflow boundary for the clean variant. Expected: `status == "ok"`,
   a `pathspec` (the `dataset_ref`).
2. **`$LOOM ingest --source <dir>/leaky --name <unique>-leaky --json`** -- same for
   the planted-leak variant. Expected: `status == "ok"`, a `pathspec`.
3. **`$LOOM validate --dataset <clean-ref> --target target --json`** -- rigorous
   validation (sealed holdout + stratified CV + leakage scan) of the clean data.
   Expected: `status == "ok"`, `VERDICT == "PASS"`, no `leakage_flags`, and both
   `summary.cv` and `summary.holdout` present. Capture its `pathspec`
   (`ValidateFlow/<id>`).
4. **`$LOOM deploy --validate <clean-validate-run> --json`** -- the exit gate, on a
   `PASS`. **No `--apply`.** Expected: `gate.decision == "ALLOW"`,
   `VERDICT == "STAGED"`, `summary.status == "PLANNED"` (a staged manifest, no
   external mutation), and `summary.apply is False`.
5. **`$LOOM validate --dataset <leaky-ref> --target target --json`** -- validate the
   leaky data. Expected: `status == "ok"`, `VERDICT == "REVIEW"`, and
   `summary.leakage_flags` non-empty naming `leak_feature`. Capture its `pathspec`.
6. **`$LOOM deploy --validate <leaky-validate-run> --json`** -- the exit gate, on a
   `REVIEW`. **No `--apply`.** Expected: `gate.decision == "BLOCK"` with non-empty
   `gate.reasons`, `VERDICT == "BLOCKED"`, `summary.status == "BLOCKED"`, and
   `summary.apply is False`.

_(Optional aside: in a real session a key-gated `loom optimize` would produce the
candidate solution that step 3/5 validates via `--solution <run>`; it is omitted
from the asserted recipe because it needs a model key. The gate is identical.)_

## What this proves

The regression invariant this example pins: **the deploy gate decision always
tracks the upstream validate VERDICT, and `--apply` is off by default.** A `PASS`
validate ALLOWs the deploy and stages it (`STAGED` / `PLANNED`); a `REVIEW` (or
`FAIL` / leaky) validate BLOCKs it (`BLOCKED`) with explicit reasons. In both
branches `summary.apply` is `False` and the default run performs **no external
mutation** -- the safe-by-default posture. This is the contract
`tests/test_examples.py` guards by replaying `run.sh`.
