---
name: loom-viz
description: Generate standard, lineage-grounded plots through Loom — from a data object (feature distributions, correlation heatmap, target-vs-feature) or a run (metric-over-nodes, leaderboard) — emitted as @card images, then narrate what the charts show. Use when the user asks "plot this", "show me the distributions", "chart the correlations", "visualize the leaderboard", or "graph the search". Read-only: renders figures, mutates nothing.
when_to_use: "plot a dataset's distributions / correlations / target-vs-feature, visualize a run's leaderboard or metric-over-nodes, get a source-grounded chart"
when_not_to_use: "to profile a dataset's stats / leakage in text, use loom-eda; to write up an experiment in prose, use loom-report; to compute validation metrics, use loom-validate."
argument-hint: "<a dataset_ref pathspec OR a run pathspec (+ optional kind / target)>"
---

# loom-viz

Produce a **source-grounded visual** of a data object or a run so a human can *see*
the shape of the data or the search — not just read numbers. This is a **read-only**
plotting run *through Loom's MLOps interface*: it renders standard matplotlib
figures and hands back a Metaflow run + an `@card` with the figures embedded as
images. It never mutates, cleans, or writes back to the data. **The chart is
grounded in lineage** — every figure traces to the `dataset_ref` / run pathspec it
was drawn from. Stay domain-neutral — never assume a task type, column meaning, or
vertical; plot only what the data and the (optional) declared target say.

## When to use

- The user points at a `dataset_ref` and asks to "plot this", "show the
  distributions", "chart the correlations", "show `target` vs the features".
- They point at a **run** and ask to "visualize the leaderboard", "graph the search",
  "show the metric over nodes".
- Alongside `loom-eda` (the numeric profile) when a *visual* is more useful than a
  table, or alongside `loom-report` to add charts to a write-up.

## When NOT to use

- To *profile a dataset's stats* (schema, missingness, balance, leakage flags) **in
  text** — that is **`loom-eda`**.
- To *write up an experiment* in prose / a model card — that is **`loom-report`**.
- To *compute validation metrics* (CV / holdout / calibration) — that is
  **`loom-validate`**.

## 1. Intake — pin what to plot

Pin the inputs in the user's own terms and write them back for confirmation. Give
**exactly one** source:

- **Dataset** — a **`dataset_ref`** pathspec (e.g. `IngestDataset/123`) for data
  plots: feature **distributions** (histograms), a **correlation heatmap**, and
  **target-vs-feature** views. Optionally pass a `--target` (inferred from the data
  object's schema when omitted) and a `--kind`
  (`distributions`/`correlation`/`target`/`all`, default `all`).
- **Run** — a **run pathspec** (e.g. `EvalCandidate/42`) for run plots:
  **metric-over-nodes** (the search trajectory) and a **leaderboard** bar.

Take a pathspec, **never a raw S3 URI or a loose local file** as the source of
truth. A read-only viz may proceed with just one source; if the user gives neither,
ask which dataset or run.

## 2. Plan — show the plan + tier (read-only)

Viz is the **read-only tier** of the approval matrix (see `CONVENTIONS.md`):
plotting is non-destructive and reads only the data object / run, so it **never
prompts**. Briefly state the plan before running: "I'll plot `<ref>` (read-only) —
`<kind>` for a dataset, or metric-over-nodes + leaderboard for a run — and hand back
the run + `@card` with the figures." Name the exact `dataset_ref` / run pathspec and
that the data stays in the user's own Metaflow perimeter, not exfiltrated.

## 3. Run — call Loom's MLOps INTERFACE (the `loom` CLI), never the backend

Speak only Loom's interface — shell out to the `loom` CLI, which resolves the MLOps
provider by name (default **Metaflow**, swappable by config) and runs the viz flow
through the interface's `run_flow` seam. The figures render **headlessly inside the
flow** and the data is read **only through the Client API**; **never call Metaflow
or AIDE directly, and never touch raw S3.**

```bash
loom viz --dataset <PATHSPEC> [--target <COL>] [--kind distributions|correlation|target|all]
loom viz --run <PATHSPEC>
```

- The work executes as a **Metaflow run**; the input is the data object / run, read
  via the Client API.
- Lifecycle flows need the **metaflow** MLOps provider — the `local` dev provider
  cannot run them (it will say so, pointing at `--mlops metaflow`).
- Secrets/endpoints (the Metaflow profile) come from the **environment** only.

## 4. Verify — assert lineage; figures are embedded in the card, not inlined

- The command returns a **run pathspec** and the **`@card` reference**; confirm it
  reported success. The summary is a small *derived* descriptor (the plot kind, the
  source, and the list of plot names) — the **figures live in the `@card`** (the
  shareable render), never inlined as bytes into the transcript.
- Every figure traces back to the run's pathspec + the `dataset_ref` / run it was
  drawn from.

## 5. Deliver — narrate the @card, return run + summary, append a learnings row

- **Narrate the `@card`:** walk the user through the figures — for a dataset, what
  the **distributions** show (skew, multi-modality), the **strongest correlations**
  in the heatmap, and any clear **target-vs-feature** separation; for a run, the
  **metric-over-nodes** trajectory (did the search improve?) and the
  **leaderboard** spread. Describe only what the chart shows; don't over-read it.
- **Hand back the mandated artifact:** the versioned **Metaflow run + `@card`**
  (with the figures embedded), plus the typed plot descriptor the CLI prints.
- **Learnings:** the run appends one `command="viz"` row to the flywheel corpus
  (`learnings/rollouts.jsonl`) — source ref · plot kind · plot names · run + card
  pathspecs — sanitized, no raw rows, no secrets. The CLI does this; do not
  hand-write the row.
- **Next step:** offer `loom-eda` for the numeric profile behind the charts,
  `loom-report` to fold the charts into a write-up, or another `--kind` if the user
  wants a different view.

## Composition — machine-checkable exit gate (executable self-test)

- **Consumes:** a `dataset_ref` pathspec or a run pathspec (e.g. a `loom-report`
  `best_run` via `--from`).
- **Exit gate:** the viz's typed descriptor carries the **`plots` list** (the named
  figures produced) — a downstream verb / CI can assert the expected figures were
  rendered (and that an empty `plots` list means there was nothing plottable: no
  numeric columns / no scored runs), rather than trusting a silent success.
- **Self-test:** the plotting gate has an executable self-test —
  `tests/test_viz.py::test_plot_run_metrics_empty_leaderboard_no_figures` asserts an
  empty / unscored leaderboard produces **no** figures (the "draws an empty chart and
  claims success" failure mode), and `::test_plot_dataframe_all_kinds_returns_figures`
  + `::test_plot_dataframe_saves_paths_when_save_dir` assert the expected figures are
  produced (and saved) for a real dataset.

---

## Acceptance test (the bar before this verb joins the pack)

1. **Speaks only the interface** — shells out to `loom viz` (the MLOps interface,
   provider-by-name), never Metaflow/AIDE directly, never raw S3; input is a
   pathspec read via the Client API.
2. **Output is a versioned run + `@card`** — the figures are embedded as card
   images, not inlined into a chat transcript or saved as loose local PNGs as the
   deliverable.
3. **Approval tier is correct** — read-only tier, never prompts; no
   `disable-model-invocation` needed.
4. **Writes a learnings row** — the run appends a sanitized `command="viz"` row to
   `learnings/rollouts.jsonl` (the CLI does this every run).
5. **Exit gate has a self-test** — the `plots` descriptor gate is covered by the
   `tests/test_viz.py` empty/produced tests above.
6. **Single free-text arg** — one source (a `dataset_ref` or a run pathspec), plus
   an optional `--kind` / `--target`.
7. **Dual-invocation** — works user-typed (`/loom-viz`) and model-auto-loaded on the
   `description` / `when_to_use` match; safe to auto-fire (read-only).
