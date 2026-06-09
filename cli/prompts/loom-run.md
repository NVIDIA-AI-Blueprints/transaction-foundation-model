---
description: Run a task end-to-end through the configured providers (the AIDE optimize loop) — turn a goal + metric into the best solution + leaderboard.
argument-hint: <a dataset/data path + a goal sentence + a metric sentence>
---

# /loom-run — optimize a task against a metric (EXPENSIVE)

Loom's ML-iteration verb — the AIDE-search slice of the lifecycle. Hand it a
dataset, a goal, and an evaluation metric, and it searches for solution code that
maximizes that metric, runs each candidate in a real execution environment, and
returns the best solution plus a leaderboard. **The metric is the spec.**

Run/optimize: $@

If the data has not been profiled, run (or suggest) `/loom-eda` first so the goal
and metric are grounded in real columns.

## 1. Intake — metric-is-the-spec (refuse if vague)
Pin three things in the user's own terms and write them back:
- **Data** — `--data` path or a `--dataset` pathspec.
- **Goal** — one sentence of what a solution should achieve.
- **Metric** — one sentence of how a solution is scored, with an unambiguous
  direction ("Maximize ROC AUC on a held-out split", "Minimize RMSE on validation").

If any is missing or vague, **ask — do not guess a metric**. A wrong metric
silently optimizes the wrong thing.

## 2. Plan + heads-up — EXPENSIVE
This spends real cost (model tokens + compute) and reads real data, so it is the
**expensive tier**. Before firing, give the user a heads-up with:
- **Cost shape** — roughly how many candidate executions the chosen `steps` imply
  (more steps → more cost); on `metaflow` it consumes the user's own compute.
- **Data scope** — exactly which data path / dataset will be read.
- Recommend starting with `mlops=local` and a small `steps` to validate cheaply
  before any larger or `metaflow` run.

## 3. Run — call the `loom_run` tool
Call `loom_run` with `goal` and `metric` (+ `data`/`dataset`, `steps`, `mlops`,
`search`, the `*-provider` flags, `experiment-id`). Secrets/endpoints come from the
environment — never put keys on a flag or in the transcript.

## 4 & 5. Verify + narrate the result
- **Best metric** and whether it meets the goal / beats any named target.
- **Leaderboard** — a short ranked table of what Loom tried (the spread).
- **Artifacts** — the run pathspec / `card_path`; the best-solution / journal /
  tree references so the user can inspect or reuse them.
- **Next step** — more `steps`, switch `local → metaflow`, refine the metric, or
  accept the result. If no viable solution was found, say so and suggest a change.

## Guardrails
The metric is the spec — confirm it precisely; never substitute your own. Prefer
the cheap path first. Never print or pass secrets. Stay domain-neutral.
