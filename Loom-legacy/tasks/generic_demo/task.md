## Goal

Predict the `target` column for every row in the test set. `target` is a binary
label in `{0, 1}`. Train a model on `train.csv` (which contains the features and
the known `target`), then produce a prediction for each row in `test.csv` (which
contains the features but no `target`).

Write your predictions to `./working/submission.csv` with exactly two columns:

```
id,target
0,0.87
1,0.12
2,0.55
etc.
```

`id` joins each prediction back to the corresponding row in `test.csv`. `target`
should be the **predicted probability** of the positive class (a float in
`[0, 1]`), not a hard 0/1 label.

## Background

This is a generic, domain-neutral tabular dataset used to smoke-test the Loom
engine end to end. The features are abstract numeric measurements with **no
real-world meaning** — there is no vertical, customer, or use case attached. The
only objective is to learn the relationship between the feature columns and the
binary `target` as well as possible.

## Evaluation

Submissions are evaluated by **ROC-AUC** (area under the receiver operating
characteristic curve) between the predicted positive-class probabilities and the
true `target` labels. Higher is better; a perfect model scores `1.0` and a
random model scores about `0.5`. Because the classes are mildly imbalanced,
ROC-AUC is preferred over plain accuracy. Validate locally with a held-out split
or cross-validation on `train.csv`.

## Data description

- **train.csv** — the training set. Columns: `id`, `feature_0` … `feature_19`
  (float features), and `target` (the integer label in `{0, 1}` to learn).
- **test.csv** — the test set. Same feature columns as `train.csv` plus `id`,
  but **without** the `target` column — that is what you must predict.
- **sample_submission.csv** — a correctly formatted example submission
  (`id,target`) showing the exact output shape expected in
  `./working/submission.csv`. Its `target` values are placeholders, not labels.

Generate the data with `python prepare_data.py` (writes the files above into
`./input` next to the script using `sklearn.datasets.make_classification`; no
external download).
