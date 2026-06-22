---
description: Plot a data object or a run (read-only) — distributions, correlation heatmap, target-vs-feature, metric-over-nodes, leaderboard — as @card images.
argument-hint: <a dataset_ref pathspec OR a run pathspec (+ optional --kind / --target)>
---

# /loom-viz — source-grounded plots (read-only)

Produce a source-grounded visual of a data object or a run so the user can *see*
the shape of the data or the search — not just read numbers. Read-only: it renders
matplotlib figures and hands back a Metaflow run + an `@card` with the figures
embedded. Every figure traces to the `dataset_ref` / run pathspec it was drawn from.

Plot: $@

## 1. Intake — pin what to plot (exactly one source)
- **Dataset** — a `dataset_ref` pathspec for data plots: feature **distributions**,
  a **correlation heatmap**, **target-vs-feature**. Optional `target` and `kind`
  (`distributions`/`correlation`/`target`/`all`, default `all`).
- **Run** — a run pathspec for run plots: **metric-over-nodes** (the search
  trajectory) and a **leaderboard** bar.

Take a pathspec, never a raw S3 URI / loose file. If neither is given, ask which.

## 2. Plan — read-only tier
Never prompts. State: "I'll plot `<ref>` (read-only) — `<kind>` for a dataset, or
metric-over-nodes + leaderboard for a run — and hand back the run + `@card`."

## 3. Run — call the `loom_viz` tool
Call `loom_viz` with `dataset` (+ `target`, `kind`) **or** `run`. Figures render
headlessly inside the flow; data is read only through the Client API.

## 4. Verify — figures live in the card
Confirm `status` and read the **run pathspec** + `card_path`. The summary is a small
descriptor (the plot kind, the source, the list of plot names) — the figures live
in the `@card`, never inlined as bytes.

## 5. Deliver — narrate, return run + summary
- Walk through the figures: for a dataset, what the distributions show (skew,
  multi-modality), the strongest correlations, any clear target separation; for a
  run, the metric-over-nodes trajectory and the leaderboard spread. Describe only
  what the chart shows; don't over-read.
- Hand back the run + `@card` (figures embedded) + the typed descriptor.
- **Next step:** offer `/loom-eda` for the numeric profile, `/loom-report` to fold
  charts into a write-up, or another `kind`.

## Composition / exit gate
Produces a `plots` list (named figures); an empty list means nothing was plottable
(no numeric columns / no scored runs) rather than a silent success.
