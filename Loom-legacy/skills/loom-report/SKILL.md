---
name: loom-report
description: Assemble a shareable, lineage-grounded report of a Loom experiment through Loom — gather its runs, their metrics, and their lineage (Flow/Run + tags + the learnings rows) into a structured analysis/model-card and a @card, then narrate the write-up. Use when the user asks "summarize what Loom did", "write up this experiment", "make a model card", "what's the leaderboard for run X", or "report on experiment <id>". Read-only: trains nothing, writes nothing back.
when_to_use: "summarize / write up an experiment, build a model card, gather a run's metrics + lineage, produce a shareable report of what Loom did and why"
when_not_to_use: "to compute new validation metrics (CV / holdout / calibration), use loom-validate; to draw plots/charts, use loom-viz; to search for a solution, use loom-optimize."
argument-hint: "<an experiment id, or a comma list of run pathspecs>"
---

# loom-report

Assemble a **shareable write-up** of a Loom experiment so a human (or a reviewer)
can see what was tried, how it scored, and how it traces back — without re-running
anything. This is a **read-only** assembly run *through Loom's MLOps interface*: it
gathers the experiment's runs, their metrics, and their lineage (Flow/Run + tags +
the `learnings/rollouts.jsonl` rows) into a **structured analysis / model-card** and
a `@card`. The flow assembles the **structured data + the card**; the **narrative
prose is this skill's job** — you turn the assembled structure into the readable
report. Stay domain-neutral — never assume a task type, column meaning, or vertical;
report only what the runs and learnings actually carry.

## When to use

- The user asks to "summarize what Loom did", "write up this experiment", "make a
  model card", or "report on experiment `<id>`".
- They want the leaderboard + lineage for a specific set of runs (pass the
  pathspecs).
- After a `loom-optimize` / `loom-validate` sweep, to produce the shareable artifact
  that records the decision.

## When NOT to use

- To *compute new validation metrics* (CV / sealed holdout / calibration /
  fairness) — that is **`loom-validate`**; report only *reads* metrics that already
  exist on the runs.
- To *draw plots* (distributions, leaderboard bars) — that is **`loom-viz`**.
- To *search for a solution* — that is **`loom-optimize`**.

## 1. Intake — pin what to report on

Pin the inputs in the user's own terms and write them back for confirmation. Give
**exactly one** of:

- **Experiment** — an **`experiment_id`** (the stable `loom_experiment:<id>` tag a
  `loom run` / `loom-optimize` grouped its runs under). The flow gathers every run
  carrying that tag, plus the learnings rows that reference it.
- **Runs** — an explicit **comma list of run pathspecs** (e.g.
  `EvalCandidate/1,EvalCandidate/2,ValidateFlow/7`) when the user wants a specific
  set rather than a whole experiment.

A report may proceed with just one of these — it reads existing runs, so there is no
metric to refuse over. If the user gives neither, ask which experiment or runs.

## 2. Plan — show the plan + tier (read-only)

Report is the **read-only tier** of the approval matrix (see `CONVENTIONS.md`):
assembly is non-destructive and reads finished runs only, so it **never prompts**.
Briefly state the plan before running: "I'll assemble a report for
`<experiment_id>` (read-only) — its runs, metrics, leaderboard, and lineage — and
hand back the run + `@card`." Name the exact experiment / pathspecs that will be read
and that nothing is trained or written back.

## 3. Run — call Loom's MLOps INTERFACE (the `loom` CLI), never the backend

Speak only Loom's interface — shell out to the `loom` CLI, which resolves the MLOps
provider by name (default **Metaflow**, swappable by config) and runs the report
flow through the interface's `run_flow` seam. The flow reads the runs **only through
the Client API** (Flow/Run + tags) and the learnings corpus; **never call Metaflow
or AIDE directly, and never touch raw S3.**

```bash
loom report --experiment <ID>
loom report --runs <PATHSPEC,PATHSPEC,...>
```

- The work executes as a **Metaflow run**; its inputs are other runs, read via the
  Client API.
- Lifecycle flows need the **metaflow** MLOps provider — the `local` dev provider
  cannot run them (it will say so, pointing at `--mlops metaflow`).
- Secrets/endpoints (the Metaflow profile) come from the **environment** only.

## 4. Verify — assert lineage; large output stays in Metaflow

- The command returns a **run pathspec** and the **`@card` reference**; confirm it
  reported success. The report summary is a small *derived* dict (run/success
  counts, best metric + run, metric spread, a capped leaderboard, compacted
  learnings) — the full run set stays in Metaflow, referenced by pathspec.
- Every figure in the report (best metric, spread, each leaderboard row) traces back
  to a specific run's pathspec and the learnings rows that reference the experiment.

## 5. Deliver — narrate the report, return run + summary, append a learnings row

- **Narrate (the prose is yours):** turn the assembled structure into a readable
  write-up — how many runs (and how many succeeded), the **best metric and its
  run**, the **spread** (min/mean/max, so the reader sees the distribution, not just
  the winner), the **leaderboard** (top runs best-first), and the **lineage**
  (which command produced what, the data ref, the artifacts) — and the headline
  `VERDICT` (`OK` when at least one successful run exists, else `EMPTY`).
- **Hand back the mandated artifact:** the versioned **Metaflow run + `@card`**
  (the shareable render), plus the typed report summary the CLI prints.
- **Learnings:** the run appends one `command="report"` row to the flywheel corpus
  (`learnings/rollouts.jsonl`) — experiment id · run count · best metric · verdict ·
  run + card pathspecs — sanitized, no raw rows, no secrets. The CLI does this; do
  not hand-write the row.
- **Next step:** offer `loom-viz` for a visual of the leaderboard / a run, or
  `loom-validate` if a candidate needs a trustworthy held-out number before the
  report can recommend it.

## Composition — machine-checkable exit gate (executable self-test)

- **Consumes:** an `experiment_id` (the `loom_experiment` tag) or an explicit list
  of run pathspecs.
- **Exit gate:** the report's typed summary carries a **`verdict` (`OK`/`EMPTY`)**
  and the **`best_run` pathspec** — a downstream verb can assert `OK` (the
  experiment produced at least one successful run) and hand the `best_run` to
  `loom-validate` / `loom-viz` via `--from`.
- **Self-test:** the assembly gate has an executable self-test —
  `tests/test_report.py::test_assemble_empty_experiment_is_empty_verdict` asserts an
  experiment with no successful runs gates to `EMPTY` (it must not falsely report
  `OK`), and `::test_assemble_counts_and_best` asserts the best run / counts are
  computed correctly from the gathered runs.

---

## Acceptance test (the bar before this verb joins the pack)

1. **Speaks only the interface** — shells out to `loom report` (the MLOps
   interface, provider-by-name), never Metaflow/AIDE directly, never raw S3; reads
   runs + learnings via the Client API.
2. **Output is a versioned run + `@card`** — not a chat transcript or a loose
   markdown dump.
3. **Approval tier is correct** — read-only tier, never prompts; no
   `disable-model-invocation` needed.
4. **Writes a learnings row** — the run appends a sanitized `command="report"` row
   to `learnings/rollouts.jsonl` (the CLI does this every run).
5. **Exit gate has a self-test** — the `verdict` gate is covered by the
   `tests/test_report.py` empty/counts tests above.
6. **Single free-text arg** — one experiment id (or a comma list of run pathspecs).
7. **Dual-invocation** — works user-typed (`/loom-report`) and model-auto-loaded on
   the `description` / `when_to_use` match; safe to auto-fire (read-only).
