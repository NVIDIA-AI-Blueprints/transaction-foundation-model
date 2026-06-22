"""Strategy implementations for the Loom tokenizer engine (CPU/pandas).

Each :class:`~loom.engine.api.Strategy` declares a contiguous id block via two
config-only, data-free, deterministic operations:

  * ``build_vocab(strategy)`` → the ORDERED list of local token strings (local
    indices ``0..count-1``). The vocabulary is a pure function of the strategy's
    config — no fitting, no data dependence (C2 determinism).
  * ``transform(strategy, series)`` → a pandas Series of token STRINGS (the
    ``prefix_<...>`` form) for a column of raw/preprocessed values.

The reference tokenizers (``src/tokenizer/*.py``) are GPU-only (cuDF/cupy). Loom
reimplements the same vocab/grammar on CPU; conformance is on vocab size,
ordering, injectivity and grammar — NOT on the merchant hash-bucket identity
(the reference's cuDF ``hash_values`` is a different, non-portable hash).
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .api import (
    FixedVocab,
    Hash,
    MappingDirect,
    MappingPassthrough,
    MappingRange,
    Strategy,
    TimeDelta,
)

# Matches the reference TimeDeltaTokenizer (timedelta.py).
_SECONDS_PER_JULIAN_YEAR = 31556951.999999996


# ---------------------------------------------------------------------------
# KMer — a GENERIC fixed-alphabet sequence strategy (the DNA generality probe).
#
# This Strategy is defined HERE (not in api.py) on purpose: adding a new domain
# (DNA/RNA/protein) must require only a new Strategy + the field-map/compile
# wiring — NO change to the harness or contracts (the generality test, build
# brief Fix 2). ``build_vocab``/``count``/``transform`` dispatch on it by
# isinstance exactly like every existing strategy, so it flows through the LOCKED
# ``compile_spec`` + C1/C2/C3 unchanged: the k-mer vocab is injective + dense
# (it is the full ``len(alphabet)**k`` enumeration in a fixed lexical order, OR a
# config-frozen observed subset), the grammar/chunk derive normally.
#
# Unlike the tabular strategies (one row → one token), a KMer field tokenizes a
# whole SEQUENCE STRING into a per-position k-mer token SEQUENCE — "one input
# sequence → many tokens". That fan-out is NOT in ``transform`` (which stays a
# 1:1 value→token map like every other strategy); it happens in the preprocess
# (``spec.preprocess_field_map``), which EXPLODES one sequence row into one row
# per k-mer position before the per-step ``transform`` runs. So each k-mer
# position is just one single-token "event" and the existing corpus grammar
# (``<bos> KMER_x <sep> KMER_y … <eos>``) materializes it with zero new machinery.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KMer:
    """A fixed-alphabet k-mer vocabulary (DNA/RNA/protein, or any string over a
    closed alphabet). Contributes one contiguous id block of EXACTLY
    ``len(alphabet)**k`` tokens ``prefix_<KMER>`` (e.g. ``KMER_AAAA`` …
    ``KMER_TTTT`` for DNA k=4), in a fixed lexical order over ``sorted(alphabet)``
    — a pure function of config (alphabet, k), so it is C2-deterministic (no
    fitting, no data dependence). The fixed alphabet enumerates a perfect-hash code:
    the vocab is injective + dense by construction, so C1/C2/C3 derive unchanged.

    ``stride`` and ``overlapping`` describe how the *preprocess* slices a sequence
    string into k-mer positions; they do NOT change the vocabulary (the set of
    possible k-mers is the same regardless of stride). ``observed`` optionally
    pins the vocab to a config-frozen subset of k-mer strings (still C2-clean — the
    list is part of the spec, not fitted at materialize time); when ``None`` the
    vocab is the full dense ``alphabet**k`` enumeration.

    There is NO out-of-vocab/UNK bucket: the preprocess (``split_kmers``) only emits
    windows whose every symbol is in the alphabet (and, when ``observed`` is set,
    only k-mers in that subset), so every materialized window is in-vocab — the
    vocab stays EXACTLY the k-mer set, keeping ``count`` == ``alphabet**k`` for the
    full enumeration (or ``len(observed)`` for a pinned subset)."""

    prefix: str
    k: int
    alphabet: tuple[str, ...]
    stride: int = 1
    overlapping: bool = True
    observed: tuple[str, ...] | None = None

    def count(self) -> int:
        from . import strategies
        return strategies.count(self)

    def tokens(self) -> list[str]:
        from . import strategies
        return strategies.build_vocab(self)


def _kmer_strings(s: KMer) -> list[str]:
    """The ordered k-mer STRINGS (without the prefix) this strategy enumerates.

    Full enumeration is the cartesian product of ``sorted(alphabet)`` repeated
    ``k`` times in lexical order (dense ``len(alphabet)**k``); a pinned
    ``observed`` set is sorted (deduped) for a deterministic, config-only order.
    No default/out-of-vocab token is appended — the vocab is EXACTLY the k-mer set
    (the preprocess guarantees every emitted window is in-vocab)."""
    if s.observed is not None:
        return sorted(dict.fromkeys(str(m) for m in s.observed))
    alpha = sorted(set(s.alphabet))
    return ["".join(t) for t in itertools.product(alpha, repeat=int(s.k))]


def _kmer_tokens(s: KMer) -> list[str]:
    return [f"{s.prefix}_{m}" for m in _kmer_strings(s)]


# ---------------------------------------------------------------------------
# Deterministic, cross-process stable hash for the Hash strategy.
# ---------------------------------------------------------------------------


def stable_bucket(value: str, buckets: int) -> int:
    """Map ``value`` to ``0..buckets-1`` via a stable (hashlib) hash.

    Uses blake2b over the UTF-8 bytes so the bucket is reproducible across runs
    and processes — unlike Python's salted ``hash()``. This DIFFERS from the
    reference cuDF ``hash_values`` by design: Loom is the product; conformance is
    on vocab/grammar/injectivity, not merchant-bucket identity (build brief)."""
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % buckets


# ---------------------------------------------------------------------------
# build_vocab — the ORDERED local token strings for a strategy (config-only).
# ---------------------------------------------------------------------------


def _fixed_vocab_tokens(s: FixedVocab) -> list[str]:
    # 0-based local index i in 0..count-1 maps to raw value (min_val + i).
    return [f"{s.prefix}_{(s.min_val + i):0{s.pad_width}d}" for i in range(s.count())]


def _hash_tokens(s: Hash) -> list[str]:
    return [f"{s.prefix}_{i}" for i in range(s.buckets)]


def _mapping_range_tokens(s: MappingRange) -> list[str]:
    labels = sorted({lbl for _, _, lbl in s.ranges})
    labels.append(s.default)
    labels = list(dict.fromkeys(labels))  # dedupe, preserve order
    return [f"{s.prefix}_{lbl}" for lbl in labels]


def _mapping_direct_tokens(s: MappingDirect) -> list[str]:
    labels = sorted(set(s.mapping.values()))
    labels.append(s.default)
    labels = list(dict.fromkeys(labels))
    return [f"{s.prefix}_{lbl}" for lbl in labels]


def _mapping_passthrough_tokens(s: MappingPassthrough) -> list[str]:
    labels = sorted({str(v) for v in s.values})
    labels.append(s.default)
    labels = list(dict.fromkeys(labels))
    return [f"{s.prefix}_{lbl}" for lbl in labels]


def _time_delta_tokens(s: TimeDelta) -> list[str]:
    return [f"{s.special_token}_{i}" for i in range(s.num_bins)]


def build_vocab(strategy: Strategy) -> list[str]:
    """Return the ordered local token strings (local indices 0..count-1)."""
    if isinstance(strategy, FixedVocab):
        return _fixed_vocab_tokens(strategy)
    if isinstance(strategy, Hash):
        return _hash_tokens(strategy)
    if isinstance(strategy, MappingRange):
        return _mapping_range_tokens(strategy)
    if isinstance(strategy, MappingDirect):
        return _mapping_direct_tokens(strategy)
    if isinstance(strategy, MappingPassthrough):
        return _mapping_passthrough_tokens(strategy)
    if isinstance(strategy, TimeDelta):
        return _time_delta_tokens(strategy)
    if isinstance(strategy, KMer):
        return _kmer_tokens(strategy)
    raise TypeError(f"unknown strategy: {strategy!r}")


def count(strategy: Strategy) -> int:
    """Number of contiguous ids this strategy contributes."""
    if isinstance(strategy, FixedVocab):
        return strategy.max_val - strategy.min_val + 1
    if isinstance(strategy, Hash):
        return strategy.buckets
    if isinstance(strategy, (MappingRange, MappingDirect, MappingPassthrough)):
        return len(build_vocab(strategy))
    if isinstance(strategy, TimeDelta):
        return strategy.num_bins
    if isinstance(strategy, KMer):
        return len(build_vocab(strategy))
    raise TypeError(f"unknown strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# transform — raw/preprocessed column → token-string Series (CPU/pandas).
# ---------------------------------------------------------------------------


def _transform_fixed(s: FixedVocab, series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").fillna(s.min_val)
    vals = vals.round().astype("int64").clip(s.min_val, s.max_val)
    return vals.map(lambda v: f"{s.prefix}_{int(v):0{s.pad_width}d}")


def _transform_hash(s: Hash, series: pd.Series) -> pd.Series:
    return series.astype(str).map(
        lambda v: f"{s.prefix}_{stable_bucket(v, s.buckets)}"
    )


def _transform_range(s: MappingRange, series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").fillna(-1).astype("int64").to_numpy()
    out = np.full(len(vals), f"{s.prefix}_{s.default}", dtype=object)
    for lo, hi, label in s.ranges:
        mask = (vals >= lo) & (vals <= hi)
        out[mask] = f"{s.prefix}_{label}"
    return pd.Series(out, index=series.index)


def _transform_direct(s: MappingDirect, series: pd.Series) -> pd.Series:
    mapped = series.astype(str).map(s.mapping).fillna(s.default)
    return s.prefix + "_" + mapped.astype(str)


def _transform_passthrough(s: MappingPassthrough, series: pd.Series) -> pd.Series:
    allowed = set(str(v) for v in s.values)
    raw = series.astype(str)
    mapped = raw.where(raw.isin(allowed), s.default)
    return s.prefix + "_" + mapped.astype(str)


def _transform_time_delta(s: TimeDelta, series: pd.Series) -> pd.Series:
    """Log-spaced binning matching the reference TimeDeltaTokenizer.

    ``series`` carries the inter-event gap in SECONDS. We clamp to
    ``[0, max_horizon]``, take ``log(x+1)``, and ``digitize`` against
    ``num_bins+1`` linearly spaced boundaries over ``[0, log_max]``, clamped to
    ``0..num_bins-1`` — the same arithmetic the cuDF reference uses."""
    max_horizon = int(s.max_years * _SECONDS_PER_JULIAN_YEAR)
    log_max = math.log(float(max_horizon) + 1.0)
    boundaries = np.linspace(0.0, log_max, s.num_bins + 1)
    secs = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(0, max_horizon)
    log_vals = np.log(secs.to_numpy(dtype="float64") + 1.0)
    bins = np.clip(np.digitize(log_vals, boundaries), 0, s.num_bins - 1)
    return pd.Series(
        [f"{s.special_token}_{int(b)}" for b in bins], index=series.index
    )


def _transform_kmer(s: KMer, series: pd.Series) -> pd.Series:
    """Map a column of k-mer STRINGS (one per exploded sequence position) → tokens.

    The preprocess (``split_kmers``) has already sliced each sequence row into one
    in-vocab k-mer-string per position (the "one sequence → many tokens" fan-out
    lives there, and it filters to alphabet/observed windows) — so here this is a
    clean 1:1 value→token map like every other ``transform``: ``prefix_<kmer>``.
    A stray value that is somehow not a vocab k-mer maps to the ``<unk>`` special
    (id 4, always in the vocab) rather than minting an out-of-vocab token — keeping
    the vocab EXACTLY the k-mer set (injective; no UNK k-mer token is minted)."""
    allowed = set(_kmer_strings(s))
    raw = series.astype(str)
    return raw.map(lambda v: f"{s.prefix}_{v}" if v in allowed else "<unk>")


def transform(strategy: Strategy, series: pd.Series) -> pd.Series:
    """Map a column of raw/preprocessed values to token strings."""
    if isinstance(strategy, FixedVocab):
        return _transform_fixed(strategy, series)
    if isinstance(strategy, Hash):
        return _transform_hash(strategy, series)
    if isinstance(strategy, MappingRange):
        return _transform_range(strategy, series)
    if isinstance(strategy, MappingDirect):
        return _transform_direct(strategy, series)
    if isinstance(strategy, MappingPassthrough):
        return _transform_passthrough(strategy, series)
    if isinstance(strategy, TimeDelta):
        return _transform_time_delta(strategy, series)
    if isinstance(strategy, KMer):
        return _transform_kmer(strategy, series)
    raise TypeError(f"unknown strategy: {strategy!r}")


def split_kmers(
    seq: str,
    k: int,
    stride: int,
    alphabet: set[str],
    observed: set[str] | None = None,
) -> list[str]:
    """Slice ONE sequence string into its ordered in-vocab k-mer windows (config-only).

    Upper-cases and keeps only alphabet symbols (a defensive clean so whitespace /
    FASTA gaps don't shift the frame), then takes length-``k`` windows at the given
    ``stride``. A window containing any out-of-alphabet symbol cannot occur (those
    symbols are stripped first); when ``observed`` pins a vocab subset, a window not
    in that subset is dropped (so every emitted window is in-vocab). Pure +
    deterministic — no data fitting. This is the per-row primitive the preprocess
    uses to EXPLODE a sequence row into one row per position (one sequence → many
    tokens)."""
    s = "".join(ch for ch in str(seq).upper() if ch in alphabet)
    if k <= 0 or len(s) < k or stride <= 0:
        return []
    windows = [s[i : i + k] for i in range(0, len(s) - k + 1, stride)]
    if observed is not None:
        windows = [w for w in windows if w in observed]
    return windows


__all__ = ["build_vocab", "count", "transform", "stable_bucket", "KMer", "split_kmers"]
