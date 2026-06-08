---
name: loom-optimize
description: Drive a Loom run end to end — turn a goal + evaluation metric into a plan, gate it behind an explicit cost/data approval, invoke the `loom run` CLI, then narrate the best solution and leaderboard. The metric is the spec. Use when the user wants Loom to find/optimize solution code for a dataset ("optimize this", "run Loom on X", "find the best model for this metric", "beat <score>"). Pair with loom-eda first if the data is unprofiled.
---

# loom-optimize

Loom is a general-purpose, domain-neutral automated ML engine: hand it a
dataset, a goal, and an evaluation metric, and it searches for solution code that
maximizes that metric, runs each candidate in a real execution environment, and
returns the best solution plus a leaderboard.

**The metric is the spec.** Your job in this skill is to turn a loosely stated
request into a precise, runnable Loom invocation — and to stop at an approval
gate before spending budget or touching data, because a search run costs model
tokens and compute.

Stay domain-neutral. Do not assume a task type or vertical; reflect back only
what the data and the user's metric actually say.

## When to use

- The user wants Loom to optimize/solve a task against a measurable metric.
- They say things like "run Loom on this", "optimize for AUC", "find the best
  model", "beat my current score of X".

If the data has not been profiled, run (or suggest) `loom-eda` first so the goal
and metric are grounded in real columns.

## Workflow

### 1. Metric-is-the-spec intake

Pin down three things, in the user's own terms, and write them back for
confirmation:

- **Data** — the path passed to `--data` (a directory, matching Loom's layout).
- **Goal** — one natural-language sentence of what a solution should achieve
  (`--goal`).
- **Metric** — one natural-language sentence of how a solution is scored, stated
  so the optimization direction is unambiguous (`--metric`), e.g. "Maximize ROC
  AUC on a held-out split" or "Minimize RMSE on the validation set".

If any of the three is missing or vague, ask — do not guess a metric. A wrong
metric silently optimizes the wrong thing.

### 2. Plan

Propose a concrete run plan and show it before doing anything:

- **Providers**: search brain (`--search`, default `aide`) and MLOps muscle
  (`--mlops`): `local` for a fast, Metaflow-free dev path, or `metaflow` to run
  each candidate through the static `EvalCandidate` flow on the user's own
  Metaflow endpoint (their data stays in their perimeter).
- **Budget**: number of search steps (`--steps`) — more steps = more candidates
  = more tokens/compute. Recommend a small step count for a first/dev run and a
  larger one only once the local path is known to work.
- **Models**: note that code/feedback models and the routing endpoint come from
  config/env (`LOOM_CODE_MODEL`, `LOOM_FEEDBACK_MODEL`, `OPENAI_BASE_URL` /
  NIM, or Claude) — surface what is configured; never print key material.
- **Where output lands**: best metric, journal/tree artifacts, the corpus JSONL,
  and a leaderboard.

### 3. Approval gate (required)

**Stop and get explicit user approval before invoking the CLI.** This is a hard
gate because a run spends real cost and reads real data. Present:

- **Cost shape** — roughly how many candidate executions/model calls the chosen
  `--steps` and budget imply (more steps → more model + compute cost), and that
  on `metaflow` it consumes the user's own compute.
- **Data scope** — exactly which `--data` path will be read, and (on `metaflow`)
  that the data is staged into the user's perimeter, not exfiltrated.
- **The exact command** you are about to run.

Recommend starting with `--mlops local` and a small `--steps` to validate cheaply
before any larger or `metaflow` run. Do not proceed until the user confirms. If
they adjust budget/providers/metric, re-plan and re-present the gate.

### 4. Invoke the `loom run` CLI

After approval, run the command. The canonical form:

```bash
loom run \
  --data <DIR> \
  --goal "<goal sentence>" \
  --metric "<metric sentence>" \
  --steps <N> \
  --mlops <local|metaflow> \
  --search <aide>
```

- Secrets/endpoints come from the environment (`.env`/env) — never put keys on
  the command line or in the transcript.
- Let it run to completion. The CLI streams progress and, at the end, prints the
  best metric, the artifact paths, and a short leaderboard.

### 5. Narrate the result

Summarize for the user:

- **Best metric** achieved and whether it meets the goal / beats any target the
  user named.
- **Leaderboard** — a short ranked table of what Loom tried (top runs by metric),
  so the user sees the spread, not just the winner.
- **Artifacts** — absolute paths to the best solution code, the journal, and the
  search tree, so the user can inspect or reuse them.
- **Next step** — e.g. spend more `--steps`, switch `local → metaflow`, refine
  the metric, or accept the result. If no viable solution was found, say so and
  suggest what to change (often the metric phrasing, the budget, or the data).

## Guardrails

- **Always pass the approval gate before running.** Never launch a search
  silently.
- **The metric is the spec** — confirm it precisely; never substitute your own.
- **Never print or pass secrets**; they live in env/config only.
- **Domain-neutral** — do not tailor the plan to a customer or vertical.
- Prefer the cheap path first (`local`, small `--steps`) before scaling up.
