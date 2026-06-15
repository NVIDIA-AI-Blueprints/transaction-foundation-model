"""tutorials/02-regression/make_data.py -- the deterministic regression generator.

The sibling of ``examples/01-tabular-classification/make_data.py``, but for a
**continuous** target. It synthesizes a clean, learnable, domain-neutral
regression table and honors the same scaffold invariants:

* **DOMAIN-NEUTRAL.** Generic ``id`` / ``feature_*`` / ``target`` columns only --
  no customer, vertical, or PII content. An abstract "predict a number"
  scenario.
* **DETERMINISTIC + SEEDED.** A fixed ``--seed`` (default 0) drives
  :func:`sklearn.datasets.make_regression`, so a repeat run is byte-identical.
  No randomness escapes the seed.
* **NO DOWNLOADS.** Synthesized in-process; never touches the network.
* **WRITES ONLY.** It only writes ``train.csv`` into ``--out-dir``; it
  trains/evaluates nothing.

The difference from example 01 is the **target**: instead of an integer label in
``{0, 1}``, ``target`` is a *continuous float* produced by a linear combination
of the informative features plus Gaussian noise. That single change is what
flips Loom's downstream auto-detection from classification (ROC-AUC) to
regression (RMSE): ``validate`` infers the task type from the target's
cardinality -- a numeric target with many distinct values is treated as
regression.

The data is deliberately **clean** -- there is no planted leak. The
``n_informative`` features carry real signal; the rest are pure noise, so a
baseline is non-trivial but learnable. This is what makes the downstream ``eda``
report ``leakage_flags == []`` and ``validate`` come back ``PASS`` with a finite
RMSE.

Usage (called by run.sh)::

    python make_data.py --out-dir <dir> [--seed 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_regression

# Fixed defaults keep the tutorial reproducible across machines and runs.
DEFAULT_SEED = 0
DEFAULT_N_ROWS = 2000
DEFAULT_N_FEATURES = 12
DEFAULT_N_INFORMATIVE = 6
DEFAULT_NOISE = 12.0

# Stable, GENERIC column names -- no domain semantics.
ID_COLUMN = "id"
TARGET_COLUMN = "target"


def generate_frame(
    n_rows: int = DEFAULT_N_ROWS,
    seed: int = DEFAULT_SEED,
    n_features: int = DEFAULT_N_FEATURES,
    n_informative: int = DEFAULT_N_INFORMATIVE,
    noise: float = DEFAULT_NOISE,
) -> pd.DataFrame:
    """Build the deterministic, domain-neutral tabular **regression** frame.

    A clean (no planted leak) regression table: ``n_informative`` of the
    ``n_features`` columns drive a linear target, the rest are pure noise, and
    Gaussian ``noise`` is added so the relationship is real but not perfect
    (RMSE > 0 and a baseline is non-trivial). Everything is driven by ``seed`` so
    a repeat run is byte-identical.

    Args:
        n_rows: Number of rows to synthesize.
        seed: Random seed controlling all generation (reproducibility).
        n_features: Number of feature columns (``feature_0 .. feature_{N-1}``).
        n_informative: Number of features that actually carry signal.
        noise: Std-dev of the Gaussian noise added to the continuous target.

    Returns:
        A :class:`pandas.DataFrame` with an ``id`` column, ``n_features`` float
        ``feature_*`` columns, and a **continuous float** ``target`` column.
    """
    features, response = make_regression(
        n_samples=n_rows,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=seed,
    )

    feature_names = [f"feature_{i}" for i in range(n_features)]
    frame = pd.DataFrame(features, columns=feature_names)
    frame.insert(0, ID_COLUMN, range(len(frame)))
    # A continuous float target -- this is what makes the task REGRESSION.
    frame[TARGET_COLUMN] = response.astype(float)
    return frame


def write_dataset(
    out_dir: Path, n_rows: int = DEFAULT_N_ROWS, seed: int = DEFAULT_SEED
) -> dict[str, Path]:
    """Generate the frame and write ``train.csv`` into ``out_dir``.

    Args:
        out_dir: Directory to write ``train.csv`` into. Created if absent.
        n_rows: Rows to synthesize (see :func:`generate_frame`).
        seed: Random seed (see :func:`generate_frame`).

    Returns:
        A mapping of logical name to the written file path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = generate_frame(n_rows=n_rows, seed=seed)
    paths = {"train": out_dir / "train.csv"}
    frame.to_csv(paths["train"], index=False)
    return paths


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the data generator."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the 02-regression deterministic, domain-neutral synthetic "
            "dataset (clean continuous-target regression)."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write the CSV(s) into (the run.sh scratch dir).",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=DEFAULT_N_ROWS,
        help=f"Rows to synthesize (default: {DEFAULT_N_ROWS}).",
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
    paths = write_dataset(out_dir=args.out_dir, n_rows=args.n_rows, seed=args.seed)
    print(f"Wrote tutorial dataset to: {args.out_dir}")
    for name, path in paths.items():
        print(f"  {name:18s} -> {path}")


if __name__ == "__main__":
    main()
