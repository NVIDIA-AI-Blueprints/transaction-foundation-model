"""examples/04-validate-and-gated-deploy/make_data.py -- deterministic generator.

The deploy-gate use case needs TWO validate outcomes so we can prove BOTH
branches of the cross-verb exit gate in one self-checking run:

* a **clean** tabular classification dataset -> ``loom validate`` returns
  VERDICT ``PASS`` -> the deploy gate ALLOWs (STAGED);
* a near-identical dataset with a **planted leak** column (a near-perfect
  predictor of the target, |corr| >= 0.98) -> ``loom validate`` returns
  VERDICT ``REVIEW`` -> the deploy gate BLOCKs.

Both are written under ``--out-dir`` in their own subdirectory so ``run.sh``
can ingest each as a separate Metaflow data object:

    <out-dir>/clean/train.csv     # PASS -> ALLOW
    <out-dir>/leaky/train.csv     # REVIEW -> BLOCK

DOMAIN-NEUTRAL (generic ``feature_*`` / ``target`` / planted ``leak_feature``),
DETERMINISTIC + SEEDED, and OFFLINE -- it only synthesizes and writes CSVs.

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
N_FEATURES = 6

TARGET_COLUMN = "target"
# The planted near-perfect predictor. Validate flags any numeric feature whose
# |corr| with the encoded target is >= 0.98 (flows/validate.py
# _LEAKAGE_CORR_THRESHOLD), so this column forces a REVIEW verdict downstream.
LEAK_COLUMN = "leak_feature"


def generate_frame(
    n_rows: int = DEFAULT_N_ROWS,
    seed: int = DEFAULT_SEED,
    *,
    with_leak: bool = False,
) -> pd.DataFrame:
    """Build a deterministic, domain-neutral binary-classification frame.

    A handful of informative ``feature_*`` columns drive a logit; the label is a
    seeded Bernoulli draw on that logit so a baseline learns a real (not perfect)
    signal -> a healthy ``PASS`` validate. When ``with_leak`` is set, a
    ``leak_feature`` column equal to the target plus tiny deterministic noise is
    appended -- a near-perfect predictor (|corr| ~ 1.0) that validate flags,
    pushing its verdict to ``REVIEW`` and BLOCKING the downstream deploy gate.

    Args:
        n_rows: Number of rows to synthesize.
        seed: Random seed controlling all generation (reproducibility).
        with_leak: If True, append the planted ``leak_feature`` leak column.

    Returns:
        A :class:`pandas.DataFrame` with ``feature_0..feature_{N-1}``, a binary
        ``target``, and (when ``with_leak``) a ``leak_feature`` column.
    """
    rng = np.random.default_rng(seed)

    # Informative features and a moderate-strength linear signal -> a learnable
    # but NOT perfect target (so the clean dataset validates to PASS, not REVIEW).
    X = rng.normal(size=(n_rows, N_FEATURES))
    weights = np.array([1.4, -1.1, 0.9, -0.7, 0.5, -0.3])[:N_FEATURES]
    logit = X @ weights
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n_rows) < prob).astype(int)

    frame = pd.DataFrame(
        X, columns=[f"feature_{i}" for i in range(N_FEATURES)]
    )
    frame[TARGET_COLUMN] = y

    if with_leak:
        # A near-perfect predictor: the target plus tiny seeded jitter. |corr| is
        # ~1.0 (>= the 0.98 leakage threshold), so validate flags it -> REVIEW.
        jitter = rng.normal(scale=1e-3, size=n_rows)
        frame[LEAK_COLUMN] = y.astype(float) + jitter

    return frame


def write_dataset(
    out_dir: Path, n_rows: int = DEFAULT_N_ROWS, seed: int = DEFAULT_SEED
) -> dict[str, Path]:
    """Generate the clean + leaky frames and write each into its own subdir.

    Args:
        out_dir: Directory under which ``clean/train.csv`` and ``leaky/train.csv``
            are written. Created if absent.
        n_rows: Rows to synthesize (see :func:`generate_frame`).
        seed: Random seed (see :func:`generate_frame`).

    Returns:
        A mapping of logical name (``"clean"`` / ``"leaky"``) to the written path.
    """
    out_dir = Path(out_dir)
    paths: dict[str, Path] = {}

    clean_dir = out_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    generate_frame(n_rows=n_rows, seed=seed, with_leak=False).to_csv(
        clean_dir / "train.csv", index=False
    )
    paths["clean"] = clean_dir / "train.csv"

    leaky_dir = out_dir / "leaky"
    leaky_dir.mkdir(parents=True, exist_ok=True)
    generate_frame(n_rows=n_rows, seed=seed, with_leak=True).to_csv(
        leaky_dir / "train.csv", index=False
    )
    paths["leaky"] = leaky_dir / "train.csv"

    return paths


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the data generator."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the deploy-gate example's deterministic, domain-neutral "
            "datasets: a clean (PASS) and a planted-leak (REVIEW) variant."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write clean/train.csv and leaky/train.csv into.",
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
    """CLI entry point: generate the datasets and report what was written."""
    args = _parse_args(argv)
    paths = write_dataset(out_dir=args.out_dir, n_rows=args.n_rows, seed=args.seed)
    print(f"Wrote example dataset to: {args.out_dir}")
    for name, path in paths.items():
        print(f"  {name:18s} -> {path}")


if __name__ == "__main__":
    main()
