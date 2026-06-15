"""tutorials/03-leakage-safe-features/make_data.py -- deterministic data with a planted leak.

This tutorial teaches **leakage-aware feature engineering**. To do that it needs a
table where two columns secretly give away the answer -- a "leak" -- so we can watch
Loom's ``eda`` flag them and ``features --from <eda-run>`` drop them before they
poison the model.

The framing is a small, made-up **loan-default** table (it is easier to reason about
"a column you only know *after* the loan defaults" than abstract ``feature_*`` noise),
but the data is 100% synthetic, seeded, and offline -- no real people, no PII, no
network.

Columns written to ``train.csv``:

* ``application_id`` -- a benign high-cardinality row index. A naive "is this column
  unique per row?" check might fear it, but it carries no signal, so EDA must **not**
  flag it. (This is the false-positive guard.)
* ``income``, ``loan_amount``, ``credit_score``, ``age``, ``debt_ratio`` -- honest
  application-time features. A few of them drive the ``defaulted`` label through a
  logistic link, so an *honest* model is moderately good, never perfect.
* ``recovery_amount`` -- **PLANTED LEAK 1 (numeric)**. The amount a collections team
  recovered *after* a default. You only ever observe this once the loan has already
  defaulted, so at prediction time (loan-application time) it does not exist. It is
  built as ``defaulted`` plus a tiny bit of seeded noise, so its absolute Pearson
  correlation with the target is ~0.99 and EDA flags it
  ``kind="near_perfect_predictor"``.
* ``collections_status`` -- **PLANTED LEAK 2 (categorical)**. A back-office status
  stamped on the account *after* the outcome is known ("in_collections" vs "current").
  It is a deterministic relabel of the target, so each value maps 1:1 to a class and
  EDA flags it ``kind="duplicate_of_target"``.
* ``defaulted`` -- the binary 0/1 target (1 = the loan defaulted).

Why a leak is dangerous: if you train on ``recovery_amount`` / ``collections_status``
your offline scores look *amazing* (~perfect), then the model is useless in production
because those columns are not known when you actually score a fresh application. The
whole tutorial is about catching that *before* you model.

Contract (same invariants as the eval-bed examples):

* DOMAIN-FLAVORED BUT SYNTHETIC -- no real customers / PII; all values generated.
* DETERMINISTIC + SEEDED -- a fixed ``--seed`` (default 0); repeat runs are identical.
* NO DOWNLOADS -- synthesized in-process with numpy / pandas.
* WRITES ONLY -- only writes ``train.csv`` into ``--out-dir``.

Usage (called by run.sh)::

    python make_data.py --out-dir <dir> [--seed 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed defaults keep the tutorial reproducible across machines and runs.
DEFAULT_SEED = 0
DEFAULT_N_ROWS = 2000

ID_COLUMN = "application_id"
TARGET_COLUMN = "defaulted"

# The two PLANTED leak columns (named so the README + run.sh can assert on them).
LEAK_NUMERIC_COLUMN = "recovery_amount"        # numeric near-perfect predictor
LEAK_CATEGORICAL_COLUMN = "collections_status"  # categorical relabel of the target

#: The full set of planted leaks EDA must flag and features must drop.
PLANTED_LEAK_COLUMNS = (LEAK_NUMERIC_COLUMN, LEAK_CATEGORICAL_COLUMN)

# Noise on the numeric leak: small enough to stay above EDA's ~0.98 correlation
# flag threshold, but nonzero so the column is not a literal copy of the target.
_LEAK_NOISE_STD = 0.02


def generate_frame(
    n_rows: int = DEFAULT_N_ROWS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Build the deterministic loan table with two planted leaks.

    Args:
        n_rows: Number of rows (loan applications) to synthesize.
        seed: Random seed controlling all generation (reproducibility).

    Returns:
        A :class:`pandas.DataFrame` with the honest application-time features,
        the two planted leak columns, and the ``defaulted`` 0/1 target.
    """
    rng = np.random.default_rng(seed)

    # --- Honest, application-time features (known when you score a new applicant) ---
    # Drawn from seeded distributions, then lightly standardized where it helps the
    # logistic link read cleanly. Nothing here trivially determines the outcome.
    income = rng.normal(loc=60_000, scale=18_000, size=n_rows).clip(min=8_000)
    loan_amount = rng.normal(loc=15_000, scale=6_000, size=n_rows).clip(min=500)
    credit_score = rng.normal(loc=680, scale=60, size=n_rows).clip(min=300, max=850)
    age = rng.integers(low=21, high=70, size=n_rows).astype(float)
    debt_ratio = (loan_amount / income).clip(max=2.0)

    # A logistic link over a few honest features generates a learnable -- but NOT
    # trivially separable -- default probability. Lower credit score, higher debt
    # ratio, and larger loans raise default risk; higher income lowers it.
    logit = (
        -0.9
        - 2.4 * ((credit_score - 680) / 60)   # better credit -> less default
        + 1.6 * (debt_ratio - debt_ratio.mean())
        - 0.6 * ((income - 60_000) / 18_000)
        + 0.4 * ((loan_amount - 15_000) / 6_000)
    )
    probs = 1.0 / (1.0 + np.exp(-logit))
    target = (rng.uniform(size=n_rows) < probs).astype(int)

    frame = pd.DataFrame(
        {
            ID_COLUMN: range(n_rows),  # benign high-cardinality row index
            "income": income.round(2),
            "loan_amount": loan_amount.round(2),
            "credit_score": credit_score.round(1),
            "age": age,
            "debt_ratio": debt_ratio.round(4),
        }
    )

    # --- PLANTED LEAK 1 -- numeric near-duplicate of the target (|corr| ~ 0.99) ---
    # "recovery_amount" is only populated AFTER a default, so it does not exist at
    # application time. Built as target + tiny seeded noise.
    frame[LEAK_NUMERIC_COLUMN] = (target + rng.normal(scale=_LEAK_NOISE_STD, size=n_rows)).round(4)

    # --- PLANTED LEAK 2 -- categorical deterministic relabel of the target ---
    # A back-office status stamped after the outcome is known.
    frame[LEAK_CATEGORICAL_COLUMN] = np.where(target == 1, "in_collections", "current")

    frame[TARGET_COLUMN] = target
    return frame


def write_dataset(
    out_dir: Path,
    n_rows: int = DEFAULT_N_ROWS,
    seed: int = DEFAULT_SEED,
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
            "Generate the leakage-safe-features tutorial's deterministic, "
            "synthetic loan-default dataset (two planted leak columns)."
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
    print(f"  planted leaks      -> {', '.join(PLANTED_LEAK_COLUMNS)}")
    for name, path in paths.items():
        print(f"  {name:18s} -> {path}")


if __name__ == "__main__":
    main()
