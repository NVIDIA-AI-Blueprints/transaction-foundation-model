"""Generate the Loom v0.1 generic demo dataset.

This is a **domain-neutral smoke-test** for the Loom engine, not a model of any
real-world problem. It synthesizes an abstract tabular binary-classification
dataset using :func:`sklearn.datasets.make_classification` (no network access,
no external download) and writes it into the ``./input`` layout that a Loom
:class:`~loom.providers.ExecutionProvider` stages for a task:

    <input_dir>/
        train.csv             # labelled rows (features + target)
        test.csv              # unlabelled rows (features only)
        sample_submission.csv  # expected submission shape (id + target)

The columns are deliberately generic -- ``id``, ``feature_0 .. feature_{N-1}``,
and ``target`` -- so the task exercises the engine end to end without tying it to
any vertical, customer, or use case. A solution is expected to train on
``train.csv``, predict the ``target`` for every row in ``test.csv``, and write
``./working/submission.csv`` with columns ``id,target`` (probabilities scored by
ROC-AUC; see ``task.md``).

Usage::

    python tasks/generic_demo/prepare_data.py --out-dir path/to/input

By default the data is written next to this script under ``./input`` so the task
folder is self-contained and can be pointed at directly as a Loom ``data_dir``.
This module only *writes* files; it does not train or evaluate anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Fixed defaults keep the demo reproducible across machines and runs.
DEFAULT_SEED = 0
DEFAULT_N_SAMPLES = 2000
DEFAULT_N_FEATURES = 20
DEFAULT_N_INFORMATIVE = 8
DEFAULT_TEST_FRACTION = 0.25

# Stable column names. Generic on purpose -- no domain semantics.
ID_COLUMN = "id"
TARGET_COLUMN = "target"


def _default_out_dir() -> Path:
    """Return the default output directory (``./input`` beside this script)."""
    return Path(__file__).resolve().parent / "input"


def generate_frame(
    n_samples: int = DEFAULT_N_SAMPLES,
    n_features: int = DEFAULT_N_FEATURES,
    n_informative: int = DEFAULT_N_INFORMATIVE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Build a generic tabular binary-classification dataset.

    Args:
        n_samples: Total number of rows to generate (before the train/test
            split).
        n_features: Number of feature columns (named ``feature_0`` ..
            ``feature_{n_features-1}``).
        n_informative: Number of features that actually carry signal; the rest
            are redundant/noise so the task is non-trivial but learnable.
        seed: Random seed controlling :func:`make_classification` for
            reproducibility.

    Returns:
        A :class:`pandas.DataFrame` with an ``id`` column, ``n_features`` float
        feature columns, and an integer ``target`` column in ``{0, 1}``.
    """
    features, labels = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(0, n_features // 4),
        n_classes=2,
        weights=[0.6, 0.4],  # mild class imbalance to make ROC-AUC meaningful
        flip_y=0.02,
        random_state=seed,
    )

    feature_names = [f"feature_{i}" for i in range(n_features)]
    frame = pd.DataFrame(features, columns=feature_names)
    frame.insert(0, ID_COLUMN, range(len(frame)))
    frame[TARGET_COLUMN] = labels.astype(int)
    return frame


def write_dataset(
    out_dir: Path,
    n_samples: int = DEFAULT_N_SAMPLES,
    n_features: int = DEFAULT_N_FEATURES,
    n_informative: int = DEFAULT_N_INFORMATIVE,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    """Generate the dataset and write the ``./input`` CSV files.

    Args:
        out_dir: Directory to write ``train.csv``, ``test.csv`` and
            ``sample_submission.csv`` into. Created if it does not exist.
        n_samples: Total rows before splitting (see :func:`generate_frame`).
        n_features: Number of feature columns (see :func:`generate_frame`).
        n_informative: Number of informative features (see
            :func:`generate_frame`).
        test_fraction: Fraction of rows held out as the (unlabelled) test set.
        seed: Random seed for both generation and the train/test split.

    Returns:
        A mapping of logical name (``"train"``, ``"test"``,
        ``"sample_submission"``) to the written file path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = generate_frame(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        seed=seed,
    )

    train_df, test_df = train_test_split(
        frame,
        test_size=test_fraction,
        random_state=seed,
        stratify=frame[TARGET_COLUMN],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # The test set is published WITHOUT the target so a solution must predict it.
    feature_names = [f"feature_{i}" for i in range(n_features)]
    test_public = test_df[[ID_COLUMN, *feature_names]]

    # The sample submission documents the exact expected output shape. The
    # placeholder target is a constant 0.5 probability -- it is not a label.
    sample_submission = pd.DataFrame(
        {ID_COLUMN: test_df[ID_COLUMN], TARGET_COLUMN: 0.5}
    )

    paths = {
        "train": out_dir / "train.csv",
        "test": out_dir / "test.csv",
        "sample_submission": out_dir / "sample_submission.csv",
    }

    train_df.to_csv(paths["train"], index=False)
    test_public.to_csv(paths["test"], index=False)
    sample_submission.to_csv(paths["sample_submission"], index=False)

    return paths


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the data generator."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the Loom v0.1 generic, domain-neutral demo dataset "
            "(synthetic tabular binary classification)."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_default_out_dir(),
        help="Directory to write train.csv/test.csv/sample_submission.csv into "
        "(default: ./input next to this script).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Total rows before splitting (default: {DEFAULT_N_SAMPLES}).",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=DEFAULT_N_FEATURES,
        help=f"Number of feature columns (default: {DEFAULT_N_FEATURES}).",
    )
    parser.add_argument(
        "--n-informative",
        type=int,
        default=DEFAULT_N_INFORMATIVE,
        help=f"Number of informative features (default: {DEFAULT_N_INFORMATIVE}).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help=f"Held-out test fraction (default: {DEFAULT_TEST_FRACTION}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: generate the dataset and report what was written."""
    args = _parse_args(argv)
    paths = write_dataset(
        out_dir=args.out_dir,
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=args.n_informative,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print(f"Wrote generic demo dataset to: {args.out_dir}")
    for name, path in paths.items():
        print(f"  {name:18s} -> {path}")


if __name__ == "__main__":
    main()
