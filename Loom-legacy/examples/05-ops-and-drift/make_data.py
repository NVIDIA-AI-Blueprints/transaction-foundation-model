"""examples/05-ops-and-drift/make_data.py -- deterministic synthetic generator.

The "ops & drift" use case needs TWO data objects to compare: a **reference**
(baseline) frame and a **shifted** variant whose numeric feature distributions
have moved enough to trip Loom's drift smell test. This generator writes both as
sibling subdirectories of ``--out-dir`` so ``run.sh`` can ingest each as its own
Loom dataset::

    <out-dir>/
        reference/train.csv   # the baseline distribution
        shifted/train.csv     # the same schema, distributions moved

Both frames share the EXACT same columns (``id`` + ``feature_0 .. feature_{N-1}``
+ ``target``) so the drift check compares like-for-like; only the *values* of the
numeric features move in the shifted frame. The shift is a deterministic additive
mean offset + variance inflation on the feature columns -- large enough that the
relative-mean-shift exceeds Loom's ``_DRIFT_MEAN_SHIFT_THRESHOLD`` (0.25) on
multiple columns, so ``ops --dataset <shifted> --reference <reference>`` reports
``status == "DRIFT"`` with a non-empty ``drift_flags`` list.

Contract (the SCAFFOLD invariants -- see examples/_template/make_data.py):

* **DOMAIN-NEUTRAL.** Generic ``feature_*`` / ``target`` columns only -- no
  customer, vertical, or PII content. An abstract distribution-shift scenario.
* **DETERMINISTIC + SEEDED.** A fixed ``--seed`` (default 0); a repeat run is
  byte-identical. The reference and the shifted frame use distinct, derived seeds
  so they are independent draws of the same generator, not the same rows.
* **NO DOWNLOADS.** Synthesized in-process with numpy; never touches the network.
* **WRITES ONLY.** It only writes the two ``train.csv`` files; trains/evaluates
  nothing.

Usage (called by run.sh)::

    python make_data.py --out-dir <dir> [--seed 0] [--n-rows 2000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed defaults keep the example reproducible across machines and runs.
DEFAULT_SEED = 0
DEFAULT_N_ROWS = 2000
DEFAULT_N_FEATURES = 8

# Stable, GENERIC column names -- no domain semantics.
ID_COLUMN = "id"
TARGET_COLUMN = "target"

# The deterministic shift applied to the SHIFTED frame's numeric features. The
# additive offset is a multiple of each feature's baseline scale so the relative
# mean shift comfortably clears Loom's 0.25 drift threshold on every feature; the
# variance is inflated too so the move is unambiguous.
_SHIFT_MEAN_OFFSET = 3.0   # added to every feature column in the shifted frame
_SHIFT_STD_SCALE = 1.8     # feature spread multiplier in the shifted frame


def _build_frame(
    n_rows: int,
    n_features: int,
    seed: int,
    *,
    mean_offset: float = 0.0,
    std_scale: float = 1.0,
) -> pd.DataFrame:
    """Build one domain-neutral tabular frame (a single distribution draw).

    The reference frame uses ``mean_offset=0`` / ``std_scale=1``; the shifted
    frame uses the module's shift constants so its numeric features move. Both
    share the identical schema (``id`` + ``feature_*`` + ``target``).

    Args:
        n_rows: Rows to synthesize.
        n_features: Number of ``feature_*`` columns.
        seed: Random seed controlling this frame's draw (reproducibility).
        mean_offset: Additive offset applied to every feature column (the shift).
        std_scale: Multiplier on each feature column's spread (the shift).

    Returns:
        A :class:`pandas.DataFrame` with ``id``, ``n_features`` float feature
        columns, and an integer ``target`` column in ``{0, 1}``.
    """
    rng = np.random.default_rng(seed)

    # Base standard-normal features, then apply the (possibly identity) shift.
    base = rng.standard_normal(size=(n_rows, n_features))
    features = base * float(std_scale) + float(mean_offset)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    frame = pd.DataFrame(features, columns=feature_names)
    frame.insert(0, ID_COLUMN, range(n_rows))

    # A simple, learnable target off the (pre-shift) signal so the schema carries
    # a target column like the rest of the suite; ops never trains on it -- it is
    # only there to keep the schema realistic and shared across both frames.
    logits = base[:, 0] - 0.5 * base[:, 1]
    labels = (logits > np.median(logits)).astype(int)
    frame[TARGET_COLUMN] = labels
    return frame


def write_dataset(
    out_dir: Path,
    n_rows: int = DEFAULT_N_ROWS,
    n_features: int = DEFAULT_N_FEATURES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    """Generate the reference + shifted frames and write the two ``train.csv`` files.

    Writes ``<out_dir>/reference/train.csv`` (baseline) and
    ``<out_dir>/shifted/train.csv`` (distributions moved). Each subdirectory is a
    self-contained Loom ingest source.

    Args:
        out_dir: Parent directory; the two variant subdirs are created under it.
        n_rows: Rows per frame (see :func:`_build_frame`).
        n_features: Number of feature columns (see :func:`_build_frame`).
        seed: Base random seed; the two frames use derived seeds (independent
            draws of the same generator).

    Returns:
        A mapping of logical name (``"reference"``, ``"shifted"``) to the written
        ``train.csv`` path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = out_dir / "reference"
    shifted_dir = out_dir / "shifted"
    ref_dir.mkdir(parents=True, exist_ok=True)
    shifted_dir.mkdir(parents=True, exist_ok=True)

    reference = _build_frame(n_rows, n_features, seed=seed)
    shifted = _build_frame(
        n_rows,
        n_features,
        seed=seed + 1,  # an independent draw, not the same rows
        mean_offset=_SHIFT_MEAN_OFFSET,
        std_scale=_SHIFT_STD_SCALE,
    )

    paths = {
        "reference": ref_dir / "train.csv",
        "shifted": shifted_dir / "train.csv",
    }
    reference.to_csv(paths["reference"], index=False)
    shifted.to_csv(paths["shifted"], index=False)
    return paths


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the data generator."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the ops-and-drift example's deterministic, domain-neutral "
            "reference + shifted datasets (synthetic tabular distribution shift)."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Parent directory; reference/ and shifted/ subdirs are written under it.",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=DEFAULT_N_ROWS,
        help=f"Rows per frame (default: {DEFAULT_N_ROWS}).",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=DEFAULT_N_FEATURES,
        help=f"Number of feature columns (default: {DEFAULT_N_FEATURES}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base random seed for reproducibility (default: {DEFAULT_SEED}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: generate the two datasets and report what was written."""
    args = _parse_args(argv)
    paths = write_dataset(
        out_dir=args.out_dir,
        n_rows=args.n_rows,
        n_features=args.n_features,
        seed=args.seed,
    )
    print(f"Wrote ops-and-drift example datasets to: {args.out_dir}")
    for name, path in paths.items():
        print(f"  {name:12s} -> {path}")


if __name__ == "__main__":
    main()
