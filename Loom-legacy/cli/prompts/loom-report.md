---
description: Assemble an experiment's runs + metrics + lineage into a structured model-card / report (read-only) — you narrate the prose.
argument-hint: <an experiment id, or a comma list of run pathspecs>
---

# /loom-report — assemble a shareable experiment write-up (read-only)

Assemble a shareable write-up of a Loom experiment so a human (or reviewer) can see
what was tried, how it scored, and how it traces back — without re-running
anything. Read-only assembly: it gathers the experiment's runs, their metrics, and
their lineage into a structured analysis / model-card and a `@card`. The flow
assembles the structure; the **narrative prose is your job**.

Report on: $@

## 1. Intake — pin what to report on (exactly one)
- **Experiment** — an `experiment` id (the `loom_experiment:<id>` tag a `loom_run`
  grouped its runs under). Gathers every run with that tag + the learnings rows.
- **Runs** — an explicit comma list of run pathspecs (e.g.
  `EvalCandidate/1,EvalCandidate/2,ValidateFlow/7`).

If neither is given, ask which experiment or runs.

## 2. Plan — read-only tier
Reads finished runs only; never prompts. State: "I'll assemble a report for
`<experiment>` (read-only) — its runs, metrics, leaderboard, and lineage — and hand
back the run + `@card`."

## 3. Run — call the `loom_report` tool
Call `loom_report` with `experiment` **or** `runs`. It reads runs (Flow/Run + tags)
+ the learnings corpus only through the Client API.

## 4. Verify — assert lineage
Confirm `status` and read the **run pathspec** + `card_path`. The summary is small
derived data (run/success counts, best metric + run, spread, a capped leaderboard);
the full run set stays in Metaflow.

## 5. Deliver — narrate the report (the prose is yours)
- Turn the assembled structure into a readable write-up: how many runs (and how
  many succeeded), the **best metric and its run**, the **spread** (min/mean/max),
  the **leaderboard** (top runs best-first), and the **lineage** — and the headline
  `VERDICT` (`OK` when ≥1 successful run, else `EMPTY`).
- Hand back the run + `@card` + the typed summary.
- **Next step:** offer `/loom-viz` for a visual, or `/loom-validate` if a candidate
  needs a trustworthy held-out number before the report can recommend it.

## Composition / exit gate
Produces a `verdict` (`OK`/`EMPTY`) + a `best_run` pathspec; a downstream verb can
assert `OK` and hand `best_run` to `/loom-validate` / `/loom-viz`.
