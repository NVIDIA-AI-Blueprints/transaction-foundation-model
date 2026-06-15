"""examples/06-telemetry-distillation/make_data.py -- the deterministic generator.

Specialized from ``examples/_template/make_data.py`` for the **telemetry
distillation** use case: the moat-capture story. The example does not need an
exotic dataset -- it needs a clean, learnable, domain-neutral table that two
keyless verbs (``eda`` + ``validate``) can run against to GENERATE ROLLOUTS,
which the telemetry layer then assembles into trajectories and distills into the
LOOM-DS-1 SFT corpus. So this generator emits exactly that: a clean binary
classification frame (the same shape as 01), deterministic and offline.

It honors the scaffold invariants:

* **DOMAIN-NEUTRAL.** Generic ``id`` / ``feature_*`` / ``target`` columns only --
  no customer, vertical, or PII content. An abstract classification scenario.
* **DETERMINISTIC + SEEDED.** A fixed ``--seed`` (default 0) drives
  :func:`sklearn.datasets.make_classification`, so a repeat run is
  byte-identical. No randomness escapes the seed.
* **NO DOWNLOADS.** Synthesized in-process; never reaches the network.
* **WRITES ONLY.** It only writes ``train.csv`` into ``--out-dir``; it
  trains/evaluates nothing.

The data is deliberately CLEAN and learnable so the upstream verbs succeed
(``eda`` reports ``leakage_flags == []``, ``validate`` comes back ``PASS``) and
the rollouts they record carry a usable reward -- which is what makes the
distilled SFT examples meaningful. The telemetry layer is the subject here; the
data is just the fuel that two verbs burn to produce trajectories.

It mirrors ``tasks/generic_demo/prepare_data.py`` but writes only the single
labelled ``train.csv`` into ``--out-dir`` so ``loom ingest --source <dir>``
picks it up.

Usage (called by run.sh)::

    python make_data.py --out-dir <dir> [--seed 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

# Fixed defaults keep the example reproducible across machines and runs. A small
# row count keeps the two verb runs (eda + validate) fast -- this example is
# about the telemetry capture, not about a heavy model.
DEFAULT_SEED = 0
DEFAULT_N_ROWS = 600
DEFAULT_N_FEATURES = 12
DEFAULT_N_INFORMATIVE = 6

# Stable, GENERIC column names -- no domain semantics.
ID_COLUMN = "id"
TARGET_COLUMN = "target"


def generate_frame(
    n_rows: int = DEFAULT_N_ROWS,
    seed: int = DEFAULT_SEED,
    n_features: int = DEFAULT_N_FEATURES,
    n_informative: int = DEFAULT_N_INFORMATIVE,
) -> pd.DataFrame:
    """Build the deterministic, domain-neutral classification frame to profile.

    A clean (no planted leak) binary-classification table: ``n_informative`` of
    the ``n_features`` columns carry signal, the rest are redundant/noise, and a
    mild 60/40 class imbalance keeps ROC-AUC meaningful. Everything is driven by
    ``seed`` so a repeat run is byte-identical. The point is simply that ``eda``
    and ``validate`` can run against it cleanly and record rollouts.

    Args:
        n_rows: Number of rows to synthesize.
        seed: Random seed controlling all generation (reproducibility).
        n_features: Number of feature columns (``feature_0 .. feature_{N-1}``).
        n_informative: Number of features that actually carry signal.

    Returns:
        A :class:`pandas.DataFrame` with an ``id`` column, ``n_features`` float
        ``feature_*`` columns, and an integer ``target`` column in ``{0, 1}``.
    """
    features, labels = make_classification(
        n_samples=n_rows,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(0, n_features // 4),
        n_classes=2,
        weights=[0.6, 0.4],  # mild class imbalance -> ROC-AUC is meaningful
        flip_y=0.02,
        random_state=seed,
    )

    feature_names = [f"feature_{i}" for i in range(n_features)]
    frame = pd.DataFrame(features, columns=feature_names)
    frame.insert(0, ID_COLUMN, range(len(frame)))
    frame[TARGET_COLUMN] = labels.astype(int)
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
            "Generate the 06-telemetry-distillation deterministic, domain-neutral "
            "synthetic dataset (a clean binary-classification table the keyless "
            "verbs profile to generate rollouts)."
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
    print(f"Wrote example dataset to: {args.out_dir}")
    for name, path in paths.items():
        print(f"  {name:18s} -> {path}")


if __name__ == "__main__":
    main()
