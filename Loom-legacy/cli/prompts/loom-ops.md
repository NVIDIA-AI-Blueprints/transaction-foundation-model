---
description: Monitor run health, the leaderboard, and data drift (read-only) — see what passed, what failed, and whether the data moved.
argument-hint: <a --flow NAME, an --experiment ID, or a --dataset + --reference drift pair>
---

# /loom-ops — monitor runs and data drift (read-only)

Monitor Loom's runs and data objects so the user can see **what passed, what
failed, and whether the data moved** — without touching or mutating anything.
Read-only: it reads finished runs and data-object schemas/stats through the Client
API and hands back a versioned run + `@card` (run health, leaderboard, a drift
table). It trains nothing, writes nothing back, never prompts.

Monitor: $@

## 1. Intake — pin the monitoring view (one of)
- **`--flow NAME`** — a flow's recent run health: successes/failures, recency, the
  most-recent outcome.
- **`--experiment ID`** — an experiment's runs + leaderboard.
- **`--dataset PATHSPEC --reference PATHSPEC`** — a **drift** check: compare a
  current data object's schema / summary stats to a reference. Take pathspecs.

If none is given, the tool refuses with a hint — ask the user which view.

## 2. Plan — read-only tier
All non-destructive reads; never prompts. State exactly which
flow/experiment/data-objects will be read.

## 3. Run — call the `loom_ops` tool
Call `loom_ops` with `flow` / `experiment` / (`dataset` + `reference`). It runs as a
Metaflow run; recent runs + the data objects' frames are read via the Client API.

## 4. Verify — assert lineage
Confirm `status` and read the **run pathspec** + `card_path`. The summary is small
derived data (health counts, leaderboard, drift flags); rows stay in Metaflow.

## 5. Deliver — narrate, return run + summary
- Walk through **run health** (counts, success rate, whether the latest run
  succeeded — flag a `DEGRADED` latest-failed); the **leaderboard**; and the
  **drift table** — `STABLE` vs `DRIFT` with per-column smells (mean-shift,
  null-rate shift, schema add/remove). Flag exactly which columns moved.
- Hand back the run + `@card` + typed summary with its overall `VERDICT`
  (`OK`/`ATTENTION`/`EMPTY`).
- **Next step:** on `ATTENTION` (degraded health / drift), point at what to do —
  re-run the failed flow, or re-validate/re-feature against the drifted data.

## Composition / exit gate
Produces a `status` (`OK`/`ATTENTION`/`EMPTY`) + a drift `status` (`STABLE`/`DRIFT`)
the human (or a downstream re-validate) reads.
