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
import math
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
    raise TypeError(f"unknown strategy: {strategy!r}")


__all__ = ["build_vocab", "count", "transform", "stable_bucket"]
