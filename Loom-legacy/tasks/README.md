# Loom tasks

Self-contained, **domain-neutral** demo tasks used to exercise the Loom engine
end to end. Each task is a `data_dir` of input files plus a `task.md` describing
the goal and evaluation metric — the exact shape Loom consumes (`data_dir`,
`goal`, `eval`). These exist to prove the engine; **none of them is fit to any
customer, dataset, or vertical**.

## Task layout

A Loom task is the trio the controller hands to the providers:

- **`data_dir`** — a directory of input files. An
  [`ExecutionProvider`](../loom/providers/__init__.py) stages it into the
  workspace as `./input` and creates an empty `./working` alongside it. A
  solution reads from `./input` and writes its prediction to
  `./working/submission.csv`.
- **`goal`** — a natural-language description of what to predict.
- **`eval`** — a natural-language description of the validation metric.

`task.md` carries the `## Goal` / `## Evaluation` / `## Data description`
sections, so it can either be passed to AIDE as a description file or have its
goal/eval read out and handed to Loom directly.

## Available tasks

### `generic_demo/`

The internal v0.1 engine proof: a synthetic tabular **binary classification**
task. `prepare_data.py` generates an abstract dataset via
`sklearn.datasets.make_classification` (no network/download) and writes
`train.csv`, `test.csv`, and `sample_submission.csv`. The features are generic
`feature_0 … feature_N` columns and the label is `target ∈ {0, 1}`; solutions are
scored by **ROC-AUC**. See [`generic_demo/task.md`](generic_demo/task.md) for the
full spec.

Generate the data, then run the task locally (Metaflow-free dev path):

```bash
# 1) materialize the dataset into tasks/generic_demo/input/
python tasks/generic_demo/prepare_data.py

# 2) run Loom over it (see the top-level README for full CLI options)
loom run \
  --data tasks/generic_demo/input \
  --goal "Predict the binary target column for each row in the test set." \
  --metric "ROC-AUC between predicted positive-class probability and true target." \
  --mlops local
```

Swap `--mlops local` for `--mlops metaflow` to run candidates through the
Metaflow execution provider instead.

## Adding a task

1. Create a new folder under `tasks/`.
2. Add a generator that writes the input files into that folder's `./input`
   (no external downloads — bundle or synthesize the data).
3. Add a `task.md` with `## Goal`, `## Evaluation`, and `## Data description`.
4. Keep it **domain-neutral**: a task here is an engine smoke-test, not a
   model of any real-world problem.
