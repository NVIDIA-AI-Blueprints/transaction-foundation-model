---
name: loom-eda
description: Quick, read-only data profile for a Loom task — shape, columns, dtypes, missingness, target balance, and leakage smells — before you spend a search budget. Use when the user points at a dataset and asks "what's in here", "profile this data", "is this ready for Loom", or right before loom-optimize so the plan and the metric phrasing are grounded in the real data. Never modifies the data.
---

# loom-eda

Profile a dataset so a human (and the `loom-optimize` skill) can write a good
goal/metric and spot problems before committing a search budget. This is a
**read-only** reconnaissance step: look, summarize, recommend — never mutate,
clean, or write back to the data.

Loom is domain-neutral: do not assume any task type, column meaning, or
vertical. Describe only what is actually in the data.

## When to use

- The user points at a directory or file and asks what's in it / whether it's
  ready for an automated ML run.
- Immediately before `loom-optimize`, to ground the plan in real columns and
  shapes.

## Inputs

- A data path: a directory (preferred — mirrors Loom's `--data DIR`) or a single
  tabular file (`.csv`, `.parquet`, `.json`/`.jsonl`).
- Optional: the column the user believes is the target/label.

## Procedure

1. **Locate the data.** List the path. For a directory, enumerate files and
   sizes and pick the primary table(s) (largest / clearly-named `train`/`test`).
   Note any obvious `train`/`validation`/`test` split already present.

2. **Profile each primary table (read-only).** Prefer a tiny pandas snippet run
   through the environment's Python; keep it to a sampled/`nrows`-bounded read
   for large files so profiling itself is cheap. Report:
   - shape (rows × columns) and on-disk size;
   - per-column dtype, % missing, and cardinality (n unique);
   - for numeric columns: min / median / max and a couple of outlier flags;
   - for categorical/text columns: top few values and their frequencies;
   - a small head() sample (redact anything that looks like a secret/credential
     or direct personal identifier — show the shape of the value, not the value).

3. **Target & balance.** If a target column is named or obvious, report its
   distribution: class balance for classification (flag severe imbalance), or
   range/skew for regression. If no target is identifiable, say so plainly and
   list the plausible candidates rather than guessing one.

4. **Readiness & smells.** Call out things that would bite a Loom run:
   - leakage smells (a column that is a near-perfect proxy for the target,
     IDs/timestamps that could leak split membership);
   - columns that are constant, all-unique (ID-like), or almost entirely missing;
   - train/test schema mismatches;
   - whether an explicit validation split exists or one must be created.

5. **Summarize for the next step.** Produce a compact profile and, crucially, a
   **suggested goal sentence and a suggested evaluation-metric sentence** phrased
   the way `loom-optimize` consumes them (e.g. "Predict `<target>` from the
   remaining columns." / "Maximize ROC AUC on a held-out split."). Present these
   as suggestions for the user to confirm or edit — do not start a search.

## Output

A short markdown report: dataset overview → per-table column profile → target
distribution → readiness flags → suggested goal + metric phrasing. End by
offering to hand off to `loom-optimize` if the user wants to run a search.

## Guardrails

- **Read-only.** Never write, clean, re-encode, or move the data.
- **Cheap.** Sample/limit rows for large files; do not load enormous data fully
  just to profile it.
- **No secrets in output.** Redact anything that looks like a key/token or a
  direct identifier.
- **No domain assumptions.** Describe the data as found; don't invent meaning.
