---
description: Compute the control a model must beat — popularity, repeat-last-item, next-side, next-amount — via a leave-one-last-out hold-out.
argument-hint: <IngestDataset/n pathspec>
---

# /loom-baseline — compute the control a model must beat (workspace-write)

Before celebrating any model, you need the bar. This verb computes the cheap,
honest baselines — what you'd get with no foundation model at all — so a later
result is a real comparison, not a vibe. Wallets/users are habitual, so these
baselines are often strong.

Baseline: $@

## 1. Intake — pin the dataset
- `--in <IngestDataset/n>` (the pathspec from `/loom-ingest`). Optional:
  `--k` (the Prec@K cutoff, default 5), `--kind popularity|repeat-last-item|both`.
- If no resolvable input, **ask — do not guess**.

## 2. Plan — workspace-write / cheap (CPU, no GPU)
State: "I'll compute the popularity / repeat-last-item / next-side / next-amount
controls on `<pathspec>` via a leave-one-last-out temporal hold-out." Columns
(entity / item / time / side / amount) are auto-inferred from the dataset.

## 3. Run — call the `loom_baseline` tool
Call `loom_baseline` with `--in` (and `--k`/`--kind` if given). It builds a
leave-one-last-out hold-out per entity (entities with ≥2 events), fits popularity
on history only (no target leak), and scores the controls.

## 4. Verify — read the metrics
Confirm `status == ok`. Read `data.metrics`: popularity Prec@K, repeat-last-item
Prec@1, next-side accuracy, next-amount MAE — and the `n` of evaluated entities.
A `REFUSED_CONTRACT` (C6) means no input / no rows / no entity+item columns /
zero multi-event entities — surface the reason and stop.

## 5. Deliver — return the bar
- Give the **`Baseline/<n>`** pathspec and the metric values plainly, sliced by
  what's available.
- State the headline: **this is the number a foundation model must beat** — and
  the strongest control (often repeat-last-item) is the one to clear.

## Composition / exit gate
Produces a `Baseline/<n>` referenced by `--experiment`; a model's later evaluation
is only a win if it beats these controls on the same hold-out.
