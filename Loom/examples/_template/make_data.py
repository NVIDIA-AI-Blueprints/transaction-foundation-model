"""examples/_template/make_data.py -- the deterministic synthetic-generator STUB.

Every ``examples/<NN-name>/make_data.py`` is a copy of this stub with
:func:`generate_frame` specialized to the use case. The contract every example's
generator MUST honor (these are the SCAFFOLD invariants):

* **DOMAIN-NEUTRAL.** Generic ``feature_*`` / ``target`` / ``event`` columns
  only -- no customer, vertical, or PII content. Abstract
  classification / sequence / drift scenarios, nothing else.
* **DETERMINISTIC + SEEDED.** A fixed ``--seed`` (default 0) so a repeat run is
  byte-identical. No randomness that escapes the seed.
* **NO DOWNLOADS.** Synthesize in-process (``numpy`` / ``sklearn.datasets`` /
  plain ``pandas``); never reach the network.
* **WRITES ONLY.** It only writes CSVs into ``--out-dir``; it trains/evaluates
  nothing.

It mirrors ``tasks/generic_demo/prepare_data.py``: write ``train.csv`` (+
optionally ``test.csv`` / ``sample_submission.csv``) into the given dir so
``loom ingest --source <dir>`` picks it up.

Usage (called by run.sh)::

    python make_data.py --out-dir <dir> [--seed 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Fixed defaults keep the example reproducible across machines and runs.
DEFAULT_SEED = 0
DEFAULT_N_ROWS = 2000

# Stable, GENERIC column names -- no domain semantics.
ID_COLUMN = "id"
TARGET_COLUMN = "target"


def generate_frame(n_rows: int = DEFAULT_N_ROWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Build the example's deterministic, domain-neutral DataFrame.

    SPECIALIZE THIS per use case (classification / planted-leak / sequence /
    reference-vs-shifted / ...). Keep it seeded and domain-neutral.

    Args:
        n_rows: Number of rows to synthesize.
        seed: Random seed controlling all generation (reproducibility).

    Returns:
        A :class:`pandas.DataFrame` with generic columns (``id``, ``feature_*``,
        ``target`` -- or the per-example schema).
    """
    raise NotImplementedError(
        "examples/_template/make_data.py is a stub -- copy it into "
        "examples/<NN-name>/make_data.py and specialize generate_frame()."
    )


def write_dataset(out_dir: Path, n_rows: int = DEFAULT_N_ROWS, seed: int = DEFAULT_SEED) -> dict[str, Path]:
    """Generate the frame and write the CSV(s) into ``out_dir``.

    Args:
        out_dir: Directory to write ``train.csv`` (+ optional ``test.csv`` /
            ``sample_submission.csv``) into. Created if absent.
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
        description="Generate this example's deterministic, domain-neutral synthetic dataset."
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
    print(f"Wrote example dataset to: {args.out_dir}")
    for name, path in paths.items():
        print(f"  {name:18s} -> {path}")


if __name__ == "__main__":
    main()
