"""examples/03-sequence-embeddings/make_data.py -- the per-account EVENT-SEQUENCE generator.

Specializes the ``examples/_template/make_data.py`` stub for the model-builder
(CPU) use case. It synthesizes a domain-neutral, per-account event-sequence
fixture with a PLANTED next-event signal: **positive accounts follow a Markov
chain over an abstract event alphabet; negative accounts emit random events.**
That planted sequential structure is exactly what the PPMI+SVD backbone the
``local`` model-builder produces is meant to capture (it is the lift the
conformance suite proves embeddings beat a raw per-row baseline), so it is the
right shape to drive ``loom train``.

The schema mirrors the train-flow fixture convention
(``tests/test_train.py::_planted_fixture``):

    account   -- the grouping key (``acct-NNN``); rows are grouped into sequences.
    t         -- the within-account ordering step (0, 1, 2, ...).
    event     -- an abstract categorical event in {A, B, C, D, E} (NO domain
                 semantics): the token whose co-occurrence carries the signal.
    amount    -- a generic numeric field (bucketized into the shared vocab).
    label     -- the per-account binary target (1 = follows the chain, 0 = random).

The contract from the scaffold (honored here):

* **DOMAIN-NEUTRAL.** Abstract ``event`` letters + a generic ``amount`` -- no
  customer, vertical, or PII content.
* **DETERMINISTIC + SEEDED.** A fixed ``numpy`` ``default_rng(seed)`` drives every
  draw (the Markov transitions, the sequence lengths, the amounts), so a repeat
  run is byte-identical.
* **NO DOWNLOADS.** Synthesized in-process with ``numpy`` / ``pandas`` only.
* **WRITES ONLY.** It writes ``train.csv`` into ``--out-dir``; it trains nothing.

Usage (called by run.sh)::

    python make_data.py --out-dir <dir> [--n-rows N] [--seed 0]

``--n-rows`` is interpreted as an approximate row budget: it is converted into a
number of accounts (each account contributes ~12 rows on average) so the
``--n-rows`` knob the template passes through stays meaningful for a per-account
fixture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed defaults keep the example reproducible across machines and runs.
DEFAULT_SEED = 0
DEFAULT_N_ROWS = 2000

# The abstract event alphabet + the planted first-order Markov chain positive
# accounts follow. NO domain semantics -- these are opaque event tokens whose
# *transition structure* (A->B->C->D->E->A) is the only signal in the data.
_ALPHABET = ["A", "B", "C", "D", "E"]
_CHAIN = {"A": "B", "B": "C", "C": "D", "D": "E", "E": "A"}

# Probability a positive account follows the chain at each step (the rest is a
# uniform random jump), and the per-account sequence-length range.
_CHAIN_PROB = 0.85
_MIN_SEQ_LEN = 8
_MAX_SEQ_LEN = 16
# Average rows an account contributes, used to turn an approximate --n-rows row
# budget into an account count.
_AVG_SEQ_LEN = (_MIN_SEQ_LEN + _MAX_SEQ_LEN) / 2.0

# Stable, GENERIC column names -- no domain semantics.
ACCOUNT_COLUMN = "account"
TIME_COLUMN = "t"
EVENT_COLUMN = "event"
AMOUNT_COLUMN = "amount"
LABEL_COLUMN = "label"


def generate_frame(n_rows: int = DEFAULT_N_ROWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Build the per-account event-sequence DataFrame with a planted next-event signal.

    Positive accounts (``label == 1``) follow the first-order Markov chain
    :data:`_CHAIN` with probability :data:`_CHAIN_PROB` at each step (a uniform
    random event otherwise); negative accounts (``label == 0``) emit i.i.d. random
    events. Half the accounts are positive (``acct % 2 == 0``), so the target is
    balanced. Everything is driven by a single seeded ``numpy`` RNG -> a repeat run
    is byte-identical.

    Args:
        n_rows: Approximate total row budget; converted into a number of accounts
            (each ~:data:`_AVG_SEQ_LEN` rows) so the knob stays meaningful for a
            per-account fixture.
        seed: Random seed controlling all generation (reproducibility).

    Returns:
        A :class:`pandas.DataFrame` with columns ``account`` / ``t`` / ``event`` /
        ``amount`` / ``label`` (generic, domain-neutral).
    """
    rng = np.random.default_rng(seed)
    n_accounts = max(2, int(round(n_rows / _AVG_SEQ_LEN)))

    rows: list[dict] = []
    for acct in range(n_accounts):
        positive = acct % 2 == 0
        seq_len = int(rng.integers(_MIN_SEQ_LEN, _MAX_SEQ_LEN))
        if positive:
            cur = _ALPHABET[int(rng.integers(0, len(_ALPHABET)))]
            events = [cur]
            for _ in range(seq_len - 1):
                if rng.random() < _CHAIN_PROB:
                    cur = _CHAIN[cur]
                else:
                    cur = _ALPHABET[int(rng.integers(0, len(_ALPHABET)))]
                events.append(cur)
        else:
            events = [_ALPHABET[int(rng.integers(0, len(_ALPHABET)))] for _ in range(seq_len)]
        for t, ev in enumerate(events):
            rows.append(
                {
                    ACCOUNT_COLUMN: f"acct-{acct:03d}",
                    TIME_COLUMN: t,
                    EVENT_COLUMN: ev,
                    AMOUNT_COLUMN: float(rng.normal(10.0, 2.0)),
                    LABEL_COLUMN: int(positive),
                }
            )
    return pd.DataFrame(rows)


def write_dataset(out_dir: Path, n_rows: int = DEFAULT_N_ROWS, seed: int = DEFAULT_SEED) -> dict[str, Path]:
    """Generate the frame and write ``train.csv`` into ``out_dir``.

    Args:
        out_dir: Directory to write ``train.csv`` into. Created if absent.
        n_rows: Approximate row budget (see :func:`generate_frame`).
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
            "Generate this example's deterministic, domain-neutral per-account "
            "event-sequence dataset (a planted next-event Markov signal)."
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
        help=f"Approximate total rows to synthesize (default: {DEFAULT_N_ROWS}).",
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
