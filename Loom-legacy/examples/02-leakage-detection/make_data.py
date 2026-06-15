"""examples/02-leakage-detection/make_data.py -- deterministic synthetic generator.

Builds a domain-neutral binary-classification table with TWO **planted leak
columns** so the EDA leakage check has something concrete to catch and the
``eda -> features`` composition gate has something concrete to drop:

* ``leak_score`` -- a numeric **near-duplicate of the target** (``target`` plus a
  tiny amount of seeded noise). Its absolute Pearson correlation with the target
  is ~0.99, so :func:`flows.eda.profile_dataframe` flags it
  ``kind="near_perfect_predictor"``.
* ``leak_flag`` -- a low-cardinality categorical that is a **deterministic
  relabel of the target** (one value per class). Its value-groups each map to a
  single target value, so EDA flags it ``kind="duplicate_of_target"``.

Everything else is honest signal: ``feature_0 .. feature_{K-1}`` are drawn from a
seeded normal, a handful of them drive the target through a logistic link, and an
``id`` column is a benign row index (high-cardinality, so it is correctly NOT
mistaken for a leak). The point of the example is that EDA flags exactly the two
planted leaks (and nothing else), ``features --from`` drops them, and the
post-drop ``validate`` is clean.

Contract honored (the SCAFFOLD invariants):

* **DOMAIN-NEUTRAL.** Generic ``id`` / ``feature_*`` / ``leak_*`` / ``target``
  columns only -- no customer, vertical, or PII content.
* **DETERMINISTIC + SEEDED.** A fixed ``--seed`` (default 0); a repeat run is
  byte-identical. All randomness comes from one seeded ``numpy`` generator.
* **NO DOWNLOADS.** Synthesized in-process with ``numpy`` / ``pandas``; never
  touches the network.
* **WRITES ONLY.** It only writes ``train.csv`` into ``--out-dir``; it
  trains/evaluates nothing.

Usage (called by run.sh)::

    python make_data.py --out-dir <dir> [--seed 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed defaults keep the example reproducible across machines and runs.
DEFAULT_SEED = 0
DEFAULT_N_ROWS = 2000
DEFAULT_N_FEATURES = 6

# Stable, GENERIC column names -- no domain semantics.
ID_COLUMN = "id"
TARGET_COLUMN = "target"

# The two PLANTED leak columns (named so the README + run.sh can assert on them).
LEAK_NUMERIC_COLUMN = "leak_score"  # near-perfect numeric predictor of the target
LEAK_CATEGORICAL_COLUMN = "leak_flag"  # deterministic categorical relabel of target

#: The full set of planted leaks EDA must flag and features must drop.
PLANTED_LEAK_COLUMNS = (LEAK_NUMERIC_COLUMN, LEAK_CATEGORICAL_COLUMN)

# Standard deviation of the noise added to the target to build the numeric leak.
# Small enough that |corr(leak_score, target)| stays well above EDA's 0.98 flag
# threshold, but nonzero so the column is not a literal copy of the target.
_LEAK_NOISE_STD = 0.02


def generate_frame(
    n_rows: int = DEFAULT_N_ROWS,
    n_features: int = DEFAULT_N_FEATURES,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Build the deterministic, domain-neutral table with two planted leaks.

    Args:
        n_rows: Number of rows to synthesize.
        n_features: Number of honest ``feature_*`` columns.
        seed: Random seed controlling all generation (reproducibility).

    Returns:
        A :class:`pandas.DataFrame` with columns ``id``, ``feature_0`` ..
        ``feature_{n_features-1}``, ``leak_score`` (numeric near-perfect
        predictor), ``leak_flag`` (categorical relabel), and ``target`` (0/1).
    """
    rng = np.random.default_rng(seed)

    # Honest features: a seeded standard normal block.
    features = rng.normal(size=(n_rows, n_features))

    # A logistic link over the first three features generates a learnable -- but
    # NOT trivially separable -- binary target (so a clean model is moderate, not
    # perfect, once the leaks are removed).
    logit = 0.9 * features[:, 0] - 0.7 * features[:, 1] + 0.5 * features[:, 2]
    probs = 1.0 / (1.0 + np.exp(-logit))
    target = (rng.uniform(size=n_rows) < probs).astype(int)

    frame = pd.DataFrame(
        {f"feature_{i}": features[:, i] for i in range(n_features)}
    )
    frame.insert(0, ID_COLUMN, range(n_rows))  # benign high-cardinality row index

    # PLANTED LEAK 1 -- numeric near-duplicate of the target (|corr| ~ 0.99).
    frame[LEAK_NUMERIC_COLUMN] = target + rng.normal(
        scale=_LEAK_NOISE_STD, size=n_rows
    )

    # PLANTED LEAK 2 -- categorical deterministic relabel of the target.
    frame[LEAK_CATEGORICAL_COLUMN] = np.where(target == 1, "positive", "negative")

    frame[TARGET_COLUMN] = target
    return frame


def write_dataset(
    out_dir: Path,
    n_rows: int = DEFAULT_N_ROWS,
    n_features: int = DEFAULT_N_FEATURES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    """Generate the frame and write ``train.csv`` into ``out_dir``.

    Args:
        out_dir: Directory to write ``train.csv`` into. Created if absent.
        n_rows: Rows to synthesize (see :func:`generate_frame`).
        n_features: Honest feature columns (see :func:`generate_frame`).
        seed: Random seed (see :func:`generate_frame`).

    Returns:
        A mapping of logical name to the written file path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = generate_frame(n_rows=n_rows, n_features=n_features, seed=seed)
    paths = {"train": out_dir / "train.csv"}
    frame.to_csv(paths["train"], index=False)
    return paths


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the data generator."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the leakage-detection example's deterministic, "
            "domain-neutral synthetic dataset (two planted leak columns)."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write train.csv into (the run.sh scratch dir).",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=DEFAULT_N_ROWS,
        help=f"Rows to synthesize (default: {DEFAULT_N_ROWS}).",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=DEFAULT_N_FEATURES,
        help=f"Honest feature columns (default: {DEFAULT_N_FEATURES}).",
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
        n_rows=args.n_rows,
        n_features=args.n_features,
        seed=args.seed,
    )
    print(f"Wrote example dataset to: {args.out_dir}")
    print(f"  planted leaks      -> {', '.join(PLANTED_LEAK_COLUMNS)}")
    for name, path in paths.items():
        print(f"  {name:18s} -> {path}")


if __name__ == "__main__":
    main()
