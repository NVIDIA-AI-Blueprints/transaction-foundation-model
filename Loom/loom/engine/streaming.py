"""Streaming (constant-memory) column statistics for vocab learning.

The :func:`stream_stats` collector walks a corpus LARGER THAN RAM in ONE bounded
pass and produces a :class:`StreamingStats` bundle that is a drop-in replacement
for the two whole-frame reads the in-RAM path makes today:

  * ``ingest``'s :func:`loom.verbs.ingest._sniff_schema` (dtype / null-frac /
    cardinality) — reproduced byte-for-byte by :meth:`StreamingStats.to_schema`,
    PLUS a real ``min``/``max`` on numeric columns (streaming-only enrichment).
  * ``ingest``'s :func:`loom.eda.leakage_scan` (identity / target leakage) —
    reproduced by :meth:`StreamingStats.to_eda_diagnostics`.
  * ``propose``'s :func:`loom.engine.propose._observed_values` (the complete
    distinct value LIST a low-card Mapping needs) and
    :func:`loom.engine.propose._is_numeric_coercible_string` (the GAP-1 amount
    detector) — fed from :class:`ColStat` via the new ``col_stats`` kw on
    :func:`loom.engine.propose.propose_spec`.

Why streaming at all: a 20 % HEAD sample of the real TabFormer corpus
(24,386,901 rows) saw only 214 of 223 merchant-states — the 10 rarest states
clustered in the tail collapsed to UNK. A constant-memory pass over EVERY row
recovers the complete vocab without ever materializing the frame.

The collector is CPU-only, deterministic, and stdlib + pandas only (no torch, no
new deps). It NEVER prints. The thresholds it routes on (``LOW_CARD_MAX`` etc.)
are IMPORTED from :mod:`loom.engine.propose` — never redefined here — so the
streamed and in-RAM paths can never drift.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional

import pandas as pd

# Single source of truth for the cutoffs / regexes — imported, never redefined,
# so the streamed path routes on EXACTLY the numbers the in-RAM path uses.
from .propose import (
    HIGH_CARD_MAX,
    LOW_CARD_MAX,
    NUMERIC_STRING_FRACTION,
    _NEAR_UNIQUE_RATIO,
    _NUMERIC_FORMAT_RE,
    _NUMERIC_STRIP_RE,
)
from .propose import _name_looks_like_identity  # mirrors eda.py's name test

#: default rows-per-chunk for the disk reader (env-tunable so an OOM-tight box can
#: shrink it without a redeploy). Used by the ingest verb's chunked ``read_csv``.
CHUNK_ROWS = int(os.environ.get("LOOM_STREAM_CHUNK_ROWS", "200000") or "200000")

#: |corr| above which a numeric feature is flagged target-leaking — mirrors
#: ``loom.eda._HIGH_CORR`` (kept in lockstep; eda.py does not export it).
_HIGH_CORR = 0.95

#: the determinism label cap: once a categorical value has been seen mapping to
#: more than this many distinct target labels it can NEVER "perfectly determine"
#: the target, so we stop growing its label set (bounded memory). The cap is 2 —
#: one extra label beyond the first is already enough to prove non-determinism.
_DETERM_LABEL_CAP = 2

# The numeric-string predicates, factored to a single place so they match
# `_is_numeric_coercible_string` cell-for-cell (same strip → coerce → mark).


def _ns_numeric_ok(stripped: "pd.Series") -> "pd.Series":
    """Boolean mask: which stripped cells coerce to a number (NUMERIC_STRING)."""
    return pd.to_numeric(stripped, errors="coerce").notna()


# ---------------------------------------------------------------------------
# The dtype-promotion lattice.
#
# Each column reads as STRING (the chunked reader uses dtype=str). To make
# `to_schema()` byte-match a single `pd.read_csv` (which infers dtypes), we
# classify every non-null cell into a coarse "kind" and combine kinds with a
# sticky lattice, then resolve the final pandas dtype STRING from the surviving
# lattice state + whether any null was seen. The targets (pandas 3.x +
# future.infer_string=True, the repo's pin) are:
#   pure ints, no null         -> "int64"
#   ints with >=1 null         -> "float64"
#   any float/decimal          -> "float64"
#   only nulls (all-blank)      -> "float64"
#   pure bool literals, no null -> "bool"
#   bool literals with a null   -> "object"
#   anything mixed / textual    -> "str"
# ---------------------------------------------------------------------------

# lattice states (ordered loosely by "promotion strength")
_K_UNKNOWN = 0  # only nulls seen so far
_K_INT = 1
_K_FLOAT = 2
_K_BOOL = 3
_K_OBJECT = 4  # absorbing: textual / mixed

# the exact bool literal set pandas' C parser accepts (case-insensitive forms).
_BOOL_TOKENS = frozenset({"True", "False", "TRUE", "FALSE", "true", "false"})


def _classify_token(tok: str) -> int:
    """Classify ONE already-stripped, non-null cell string into a lattice kind.

    Mirrors pandas' read_csv inference: a token that is a bool literal → bool; a
    bare integer → int; anything else numeric (decimal / exponent / leading-zero
    overflow handled by pandas as int, but we treat ``.``/``e`` as float) → float;
    everything else → object.
    """
    if tok in _BOOL_TOKENS:
        return _K_BOOL
    # integer? (optional sign, all digits). pandas reads leading-zero "01" as int.
    t = tok
    if t and t[0] in "+-":
        t = t[1:]
    if t.isdigit():
        return _K_INT
    # float? let pandas' own float parser decide via to_numeric on a scalar-ish
    # check — but avoid per-cell overhead: a token with a '.', 'e'/'E', or 'inf'/
    # 'nan' shape that coerces is float; otherwise object.
    try:
        float(tok)
    except (TypeError, ValueError):
        return _K_OBJECT
    return _K_FLOAT


def _promote(state: int, kind: int) -> int:
    """Combine the running lattice ``state`` with a new cell ``kind`` (sticky).

    Rules (commutative, monotone toward _K_OBJECT):
      unknown absorbs into whatever the first real kind is;
      int + float        -> float;
      int/float + bool   -> object  (number and bool can't share a column);
      anything + object  -> object  (object is absorbing);
      bool + bool        -> bool;  int + int -> int;  float + float -> float.
    """
    if state == _K_OBJECT or kind == _K_OBJECT:
        return _K_OBJECT
    if state == _K_UNKNOWN:
        return kind
    if state == kind:
        return state
    # mixed numeric: int+float -> float
    if {state, kind} == {_K_INT, _K_FLOAT}:
        return _K_FLOAT
    # bool mixed with any number -> object
    return _K_OBJECT


def _resolve_dtype(state: int, saw_null: bool, saw_any: bool) -> str:
    """The final pandas dtype STRING for a column from its lattice state.

    Matches a single ``pd.read_csv`` under the repo's pandas pin
    (future.infer_string=True)."""
    if not saw_any or state == _K_UNKNOWN:
        # column was entirely null/blank -> all-NaN float64 (matches pandas).
        return "float64"
    if state == _K_INT:
        return "float64" if saw_null else "int64"
    if state == _K_FLOAT:
        return "float64"
    if state == _K_BOOL:
        return "object" if saw_null else "bool"
    return "str"  # _K_OBJECT


# ---------------------------------------------------------------------------
# ColStat — one column's bounded accumulator.
# ---------------------------------------------------------------------------


@dataclass
class ColStat:
    """Bounded, mergeable statistics for ONE column over a streamed corpus.

    The distinct-value collection is TWO-TIER and capped so memory stays
    constant regardless of cardinality:

      * a free ``_distinct`` set of stringified non-null values is kept ONLY while
        it stays under :data:`LOW_CARD_MAX`. The first time it would reach the cap
        the value LIST is FREED (``values`` becomes ``None``) and ``hash_bound``
        flips True — the proposer then routes the column to Hash, identical to the
        in-RAM Mapping↔Hash decision (Hash only needs ``n_unique >= LOW_CARD_MAX``).
      * a SECOND probe set ``_probe`` keeps the EXACT distinct COUNT past the
        LOW_CARD cap, up to ``HIGH_CARD_MAX + 1``; at that point it is frozen and
        ``over_high_card`` flips True. ``n_unique`` is therefore the TRUE distinct
        count (not a sentinel) for every column up to ``HIGH_CARD_MAX + 1`` — this
        is what makes :meth:`StreamingStats.to_eda_diagnostics`' ``unique_ratio`` /
        ``near_unique`` and the proposer's own near-unique gate match the in-RAM
        path EXACTLY, including for a high-cardinality near-unique key (the value
        LIST is freed at ``LOW_CARD_MAX`` for memory, but the COUNT stays accurate).
        A column whose TRUE distinct exceeds ``HIGH_CARD_MAX`` reports the
        ``HIGH_CARD_MAX + 1`` sentinel and ``over_high_card`` — both paths then DROP
        it as free-text, so the residual count beyond the cap is never load-bearing.

    Numeric range (``vmin``/``vmax``) is tracked for any column whose cells coerce
    to a number (streaming-only enrichment feeding ``to_schema``'s min/max).

    The three numeric-STRING counters (``ns_non_null`` / ``ns_parse_ok`` /
    ``ns_currency_mark``) reproduce :func:`_is_numeric_coercible_string`'s two
    fractions exactly.

    Leakage accumulators (Pearson sufficient stats + capped per-value target
    labels) are filled ONLY when a ``target`` is set; they reduce to the same
    :class:`~loom.types.Diagnostic` cards :func:`loom.eda.leakage_scan` emits.
    """

    name: str
    dtype: str = "float64"
    total_count: int = 0
    null_count: int = 0
    n_non_null: int = 0
    n_unique: int = 0

    # complete distinct non-null value list, present ONLY while < LOW_CARD_MAX.
    values: Optional[list[str]] = None
    hash_bound: bool = False
    over_high_card: bool = False

    # numeric range (None until a coercible numeric cell is seen).
    vmin: Optional[float] = None
    vmax: Optional[float] = None

    # numeric-string fractions (mirror _is_numeric_coercible_string).
    ns_non_null: int = 0
    ns_parse_ok: int = 0
    ns_currency_mark: int = 0

    # leakage: Pearson sufficient stats (Welford-stable co-moments).
    corr_n: int = 0
    corr_mean_x: float = 0.0
    corr_mean_y: float = 0.0
    corr_m2_x: float = 0.0
    corr_m2_y: float = 0.0
    corr_cxy: float = 0.0

    # leakage: per categorical value -> set of distinct target labels (capped).
    determ_labels: dict[str, set] = field(default_factory=dict)
    determ_tracked: bool = True

    # --- internal, NOT serialized: the live distinct + probe sets + lattice ---
    _distinct: Optional[set] = field(default=None, repr=False, compare=False)
    _probe: Optional[set] = field(default=None, repr=False, compare=False)
    _lattice: int = field(default=_K_UNKNOWN, repr=False, compare=False)
    _saw_null: bool = field(default=False, repr=False, compare=False)
    _saw_any: bool = field(default=False, repr=False, compare=False)
    _determ_capped: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._distinct is None and not self.hash_bound:
            self._distinct = set()
        if self._probe is None and not self.over_high_card:
            self._probe = set()

    # -- the ONE shared null predicate -------------------------------------

    @staticmethod
    def is_null_str(text: "pd.Series") -> "pd.Series":
        """The single null predicate shared by every counter: a cell is NULL iff
        it is NaN/None OR empty after stripping surrounding whitespace.

        Matches ``s.dropna()`` semantics on a frame the in-RAM path reads, plus the
        chunked reader's ``na_values=['']`` (a literally-empty field is already
        NaN; this also drops a whitespace-only field, which the in-RAM path treats
        as a present blank string — but ``''`` after strip is the agreed null per
        the build brief's "empty-after-strip = null" rule)."""
        isna = text.isna()
        stripped = text.fillna("").astype("string").str.strip()
        return isna | (stripped == "")

    # -- distinct-set maintenance (the bounded two-tier collector) ----------

    def _observe_distinct(self, vals: "pd.Series") -> None:
        """Fold a chunk's non-null stringified values into the two-tier sets.

        ``_probe`` is the EXACT distinct-count source: it keeps growing up to
        ``HIGH_CARD_MAX + 1`` regardless of whether the low-card value LIST was
        already freed, so ``finalize`` can report the TRUE ``n_unique`` (needed for
        leakage / near-unique parity with the in-RAM path) for any column whose true
        cardinality is ``<= HIGH_CARD_MAX``."""
        # tier 2: the EXACT distinct-count probe (cap HIGH_CARD_MAX + 1, then frozen).
        if self._probe is not None:
            for v in vals:
                self._probe.add(v)
                if len(self._probe) > HIGH_CARD_MAX:
                    self.over_high_card = True
                    self._probe = None  # freeze + free (count is now the sentinel)
                    break
        elif not self.over_high_card:
            self.over_high_card = True

        # tier 1: the complete low-card VALUE LIST (cap LOW_CARD_MAX, then freed).
        # Only the value LIST is freed here — the COUNT lives on in ``_probe`` above,
        # and determinism tracking is NOT abandoned (it is bounded by ``_probe``'s
        # HIGH_CARD cap, freed in ``_observe_determ`` when ``over_high_card``), so a
        # high-card near-unique column still raises target_determines like in-RAM.
        if self._distinct is not None:
            for v in vals:
                self._distinct.add(v)
                if len(self._distinct) >= LOW_CARD_MAX:
                    self.hash_bound = True
                    self._distinct = None  # free the LIST — bounded memory
                    break

    # -- leakage accumulators ----------------------------------------------

    def _observe_corr(self, x: "pd.Series", y: "pd.Series") -> None:
        """Update the Pearson co-moments from a chunk's paired numeric values
        (Welford / Chan parallel-merge stable, matching pandas' pairwise corr)."""
        xv = x.to_numpy(dtype="float64")
        yv = y.to_numpy(dtype="float64")
        nb = xv.shape[0]
        if nb == 0:
            return
        # chunk moments
        mx = float(xv.mean())
        my = float(yv.mean())
        dx = xv - mx
        dy = yv - my
        m2x_b = float((dx * dx).sum())
        m2y_b = float((dy * dy).sum())
        cxy_b = float((dx * dy).sum())
        na = self.corr_n
        if na == 0:
            self.corr_n = nb
            self.corr_mean_x = mx
            self.corr_mean_y = my
            self.corr_m2_x = m2x_b
            self.corr_m2_y = m2y_b
            self.corr_cxy = cxy_b
            return
        nt = na + nb
        delta_x = mx - self.corr_mean_x
        delta_y = my - self.corr_mean_y
        self.corr_m2_x += m2x_b + delta_x * delta_x * na * nb / nt
        self.corr_m2_y += m2y_b + delta_y * delta_y * na * nb / nt
        self.corr_cxy += cxy_b + delta_x * delta_y * na * nb / nt
        self.corr_mean_x += delta_x * nb / nt
        self.corr_mean_y += delta_y * nb / nt
        self.corr_n = nt

    def _observe_determ(self, col_vals: "pd.Series", tgt_vals: "pd.Series") -> None:
        """Track, per distinct categorical value, the set of target labels seen
        (capped at :data:`_DETERM_LABEL_CAP`). Once any value maps to >1 label the
        column cannot determine the target.

        NOT abandoned at the LOW_CARD value-list cap — a high-card near-unique column
        legitimately determines the target in-RAM (each near-unique value maps to one
        label), so we keep tracking past ``hash_bound`` to preserve that parity. The
        per-value label dict is bounded by the HIGH_CARD probe: once the column's true
        distinct crosses ``HIGH_CARD_MAX`` (``over_high_card``) the column drops as
        free-text in BOTH paths, so determinism is moot — we abandon + free the dict
        there to keep memory bounded."""
        if not self.determ_tracked:
            return
        if self.over_high_card:
            # column drops as free-text in both paths — determinism is moot; free.
            self.determ_tracked = False
            self.determ_labels = {}
            return
        # group this chunk locally, then merge the small per-value label sets.
        try:
            pairs = pd.DataFrame({"c": col_vals.to_numpy(), "t": tgt_vals.to_numpy()})
        except Exception:  # pragma: no cover - defensive
            return
        for cval, sub in pairs.groupby("c", sort=False):
            key = str(cval)
            labels = self.determ_labels.setdefault(key, set())
            for tval in sub["t"].unique():
                if len(labels) >= _DETERM_LABEL_CAP:
                    break
                labels.add(str(tval))
            # bound the dict itself by the distinct cap — if it ever explodes,
            # the hash_bound path above will have freed it; this is belt-and-braces.

    # -- finalize -----------------------------------------------------------

    def finalize(self) -> None:
        """Freeze derived fields from the live sets before serialization.

        ``n_unique`` is the TRUE distinct count for parity with the in-RAM path's
        ``nunique()`` — taken from ``_probe`` (exact up to ``HIGH_CARD_MAX + 1``),
        NOT a ``LOW_CARD_MAX`` sentinel. This keeps ``to_schema`` /
        ``to_eda_diagnostics`` ``unique_ratio`` (and the proposer's own near-unique
        gate) byte-identical to the in-RAM path, including for a high-cardinality
        near-unique key. The value LIST is freed at the LOW_CARD cap (so a capped
        column carries ``values=None`` → Hash routing), but the COUNT survives."""
        self.dtype = _resolve_dtype(self._lattice, self._saw_null, self._saw_any)
        if self.hash_bound or self._distinct is None:
            # value list freed (capped) — route to Hash via values=None. The COUNT
            # is the exact probe size, or the >HIGH_CARD_MAX sentinel once frozen.
            self.values = None
            if self._probe is not None:
                self.n_unique = len(self._probe)
            else:
                # probe frozen: true distinct exceeds HIGH_CARD_MAX → drop sentinel
                # (> HIGH_CARD_MAX, exactly what the in-RAM count would route on).
                self.n_unique = HIGH_CARD_MAX + 1
        else:
            # sorted EXACTLY like _observed_values: sorted(set(str(v))).
            self.values = sorted(self._distinct)
            self.n_unique = len(self.values)
        # drop the now-frozen live sets so to_dict carries no transient state.
        self._distinct = None
        self._probe = None


# ---------------------------------------------------------------------------
# StreamingStats — the whole-corpus bundle.
# ---------------------------------------------------------------------------


@dataclass
class StreamingStats:
    """The per-corpus statistics bundle (JSON-serializable; stored INLINE on
    ``IngestDataset.extras['col_stats']``)."""

    n_rows: int
    n_cols: int
    column_order: list[str]
    source_fingerprint: str
    columns: dict[str, ColStat]

    # -- schema (byte-compatible with ingest._sniff_schema + min/max) -------

    def to_schema(self) -> dict[str, Any]:
        """Reproduce :func:`loom.verbs.ingest._sniff_schema` EXACTLY, plus a real
        ``min``/``max`` on numeric columns (the streaming-only enrichment that lets
        ``propose._int_range`` use real ranges).

        Shape: ``{n_rows, n_cols, columns: {col: {dtype, null_frac (6dp),
        n_unique[, min, max]}}}`` — column order is ``column_order``."""
        cols: dict[str, Any] = {}
        nr = max(self.n_rows, 1)
        for name in self.column_order:
            cs = self.columns[name]
            null_frac = (cs.null_count / nr) if self.n_rows else 0.0
            entry: dict[str, Any] = {
                "dtype": cs.dtype,
                "null_frac": round(float(null_frac), 6),
                "n_unique": int(cs.n_unique),
            }
            # min/max enrichment: only for columns whose final dtype is numeric.
            if cs.dtype in ("int64", "float64") and cs.vmin is not None and cs.vmax is not None:
                if cs.dtype == "int64":
                    entry["min"] = int(cs.vmin)
                    entry["max"] = int(cs.vmax)
                else:
                    entry["min"] = float(cs.vmin)
                    entry["max"] = float(cs.vmax)
            cols[name] = entry
        return {"n_rows": int(self.n_rows), "n_cols": int(self.n_cols), "columns": cols}

    # -- leakage (byte-compatible with eda.leakage_scan) --------------------

    def to_eda_diagnostics(self, target: Optional[str] = None) -> list[dict[str, Any]]:
        """Reduce the streamed accumulators into the SAME ``Diagnostic`` card dicts
        :func:`loom.eda.leakage_scan` emits (``data.kind`` ∈ ``identity_like`` /
        ``target_correlated`` / ``target_determines``), in ``column_order``.

        Returns serialized dicts (``Diagnostic.to_dict()`` shape) so the verb layer
        can build live ``Diagnostic`` objects from them exactly as it does for the
        persisted ``ingest_report``."""
        diags: list[dict[str, Any]] = []
        if self.n_rows == 0:
            return diags
        target_l = str(target) if target is not None else None
        for col in self.column_order:
            if col == target_l:
                continue
            cs = self.columns[col]
            n_non_null = cs.n_non_null
            if n_non_null == 0:
                continue
            n_unique = int(cs.n_unique)
            unique_ratio = n_unique / n_non_null
            name_hit = _name_looks_like_identity(col)
            near_unique = unique_ratio >= _NEAR_UNIQUE_RATIO

            # --- identity / near-unique key ---------------------------------
            if name_hit or near_unique:
                if name_hit and near_unique:
                    why = "id-shaped name AND near-unique values"
                elif name_hit:
                    why = "id-shaped column name"
                else:
                    why = "near-unique values (looks like a row/entity key)"
                diags.append(
                    {
                        "contract": "EDA",
                        "severity": "warning",
                        "message": (
                            f"column {col!r} looks identity-like ({why}): "
                            f"{n_unique}/{n_non_null} distinct "
                            f"({unique_ratio:.0%} unique)"
                        ),
                        "fix": (
                            f"if {col!r} is the grouping entity, pass it as --entity "
                            f"(it is then never tokenized as a feature, T2); "
                            f"otherwise drop it before tokenizing"
                        ),
                        "data": {
                            "column": col,
                            "kind": "identity_like",
                            "n_unique": n_unique,
                            "n_non_null": n_non_null,
                            "unique_ratio": round(unique_ratio, 4),
                            "name_match": name_hit,
                            "near_unique": near_unique,
                        },
                    }
                )

            # --- target leakage ---------------------------------------------
            if target_l is not None and target_l in self.columns:
                # numeric Pearson correlation from the streamed co-moments.
                if cs.corr_n >= 3 and cs.corr_m2_x > 0 and cs.corr_m2_y > 0:
                    corr = cs.corr_cxy / math.sqrt(cs.corr_m2_x * cs.corr_m2_y)
                    if corr == corr and abs(corr) >= _HIGH_CORR:  # corr==corr: not NaN
                        diags.append(
                            {
                                "contract": "EDA",
                                "severity": "warning",
                                "message": (
                                    f"column {col!r} is {abs(corr):.0%}-correlated with "
                                    f"target {target_l!r} — possible leakage / a derived label"
                                ),
                                "fix": (
                                    f"confirm {col!r} is a legitimate pre-event feature, "
                                    f"not computed from {target_l!r}; drop it if it is"
                                ),
                                "data": {
                                    "column": col,
                                    "kind": "target_correlated",
                                    "target": target_l,
                                    "abs_corr": round(abs(float(corr)), 4),
                                },
                            }
                        )
                        continue
                # categorical determinism: each value maps to exactly one label.
                if cs.determ_tracked and cs.dtype not in ("int64", "float64", "bool"):
                    labels = cs.determ_labels
                    if len(labels) > 1 and all(len(v) <= 1 for v in labels.values()):
                        diags.append(
                            {
                                "contract": "EDA",
                                "severity": "warning",
                                "message": (
                                    f"column {col!r} perfectly determines target "
                                    f"{target_l!r} (each value maps to one label) — likely leakage"
                                ),
                                "fix": (
                                    f"verify {col!r} is observable before the event; "
                                    f"drop it if it encodes {target_l!r}"
                                ),
                                "data": {
                                    "column": col,
                                    "kind": "target_determines",
                                    "target": target_l,
                                },
                            }
                        )
        return diags

    # -- serialization (round-trips through extras['col_stats'] JSON) -------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "loom-streaming-stats/1",
            "n_rows": int(self.n_rows),
            "n_cols": int(self.n_cols),
            "column_order": list(self.column_order),
            "source_fingerprint": self.source_fingerprint,
            "columns": {name: _colstat_to_dict(cs) for name, cs in self.columns.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StreamingStats":
        cols = {name: _colstat_from_dict(name, cd) for name, cd in (d.get("columns") or {}).items()}
        return cls(
            n_rows=int(d.get("n_rows", 0) or 0),
            n_cols=int(d.get("n_cols", 0) or 0),
            column_order=list(d.get("column_order") or []),
            source_fingerprint=str(d.get("source_fingerprint", "")),
            columns=cols,
        )


def _colstat_to_dict(cs: ColStat) -> dict[str, Any]:
    """JSON-able form of a finalized ColStat (the live sets are already dropped)."""
    return {
        "name": cs.name,
        "dtype": cs.dtype,
        "total_count": int(cs.total_count),
        "null_count": int(cs.null_count),
        "n_non_null": int(cs.n_non_null),
        "n_unique": int(cs.n_unique),
        "values": list(cs.values) if cs.values is not None else None,
        "hash_bound": bool(cs.hash_bound),
        "over_high_card": bool(cs.over_high_card),
        "vmin": cs.vmin,
        "vmax": cs.vmax,
        "ns_non_null": int(cs.ns_non_null),
        "ns_parse_ok": int(cs.ns_parse_ok),
        "ns_currency_mark": int(cs.ns_currency_mark),
        "corr_n": int(cs.corr_n),
        "corr_mean_x": cs.corr_mean_x,
        "corr_mean_y": cs.corr_mean_y,
        "corr_m2_x": cs.corr_m2_x,
        "corr_m2_y": cs.corr_m2_y,
        "corr_cxy": cs.corr_cxy,
        "determ_labels": {k: sorted(v) for k, v in cs.determ_labels.items()},
        "determ_tracked": bool(cs.determ_tracked),
    }


def _colstat_from_dict(name: str, d: dict[str, Any]) -> ColStat:
    cs = ColStat(name=name, hash_bound=bool(d.get("hash_bound", False)),
                 over_high_card=bool(d.get("over_high_card", False)))
    # __post_init__ may have allocated live sets; clear them — this is a finalized
    # rehydration, not a live collector.
    cs._distinct = None
    cs._probe = None
    cs.dtype = str(d.get("dtype", "float64"))
    cs.total_count = int(d.get("total_count", 0) or 0)
    cs.null_count = int(d.get("null_count", 0) or 0)
    cs.n_non_null = int(d.get("n_non_null", 0) or 0)
    cs.n_unique = int(d.get("n_unique", 0) or 0)
    vals = d.get("values")
    cs.values = list(vals) if vals is not None else None
    cs.vmin = d.get("vmin")
    cs.vmax = d.get("vmax")
    cs.ns_non_null = int(d.get("ns_non_null", 0) or 0)
    cs.ns_parse_ok = int(d.get("ns_parse_ok", 0) or 0)
    cs.ns_currency_mark = int(d.get("ns_currency_mark", 0) or 0)
    cs.corr_n = int(d.get("corr_n", 0) or 0)
    cs.corr_mean_x = float(d.get("corr_mean_x", 0.0) or 0.0)
    cs.corr_mean_y = float(d.get("corr_mean_y", 0.0) or 0.0)
    cs.corr_m2_x = float(d.get("corr_m2_x", 0.0) or 0.0)
    cs.corr_m2_y = float(d.get("corr_m2_y", 0.0) or 0.0)
    cs.corr_cxy = float(d.get("corr_cxy", 0.0) or 0.0)
    cs.determ_labels = {k: set(v) for k, v in (d.get("determ_labels") or {}).items()}
    cs.determ_tracked = bool(d.get("determ_tracked", True))
    return cs


# ---------------------------------------------------------------------------
# The collector entrypoint.
# ---------------------------------------------------------------------------


def stream_stats(
    chunk_iter: Iterable["pd.DataFrame"],
    *,
    column_order: list[str],
    target: Optional[str] = None,
) -> StreamingStats:
    """Fold an iterator of string-dtype chunks into a :class:`StreamingStats`.

    Each chunk is a ``pandas.DataFrame`` read with ``dtype=str`` (every cell a
    string or NaN). Per chunk, per column we update: total/null counts (the ONE
    shared null predicate), the bounded two-tier distinct set, the dtype-promotion
    lattice, the numeric range, the three numeric-string counters, and — when a
    ``target`` is set — the Pearson co-moments + the capped determinism labels.

    ``column_order`` fixes the column order of the resulting schema/diagnostics
    (the same order ``_collect_source_files`` + ``read_csv`` would yield). The
    collector is deterministic: the same chunks in the same order → an identical
    bundle. It NEVER prints.
    """
    cols: dict[str, ColStat] = {c: ColStat(name=c) for c in column_order}
    n_rows = 0
    target_l = str(target) if target is not None else None

    for chunk in chunk_iter:
        if chunk is None or len(chunk) == 0:
            continue
        n_rows += int(len(chunk))

        # precompute the target's numeric coercion ONCE per chunk (for corr) and
        # the target's null mask (for determinism pairing).
        tgt_num = None
        tgt_raw = None
        tgt_notnull = None
        if target_l is not None and target_l in chunk.columns:
            tgt_raw = chunk[target_l]
            tgt_num = pd.to_numeric(tgt_raw, errors="coerce")
            tgt_notnull = ~ColStat.is_null_str(tgt_raw)

        for col in column_order:
            cs = cols[col]
            if col not in chunk.columns:
                # a ragged file missing a column: count rows as null for this col.
                cs.total_count += int(len(chunk))
                cs.null_count += int(len(chunk))
                cs._saw_null = True
                continue
            series = chunk[col]
            cs.total_count += int(len(series))

            null_mask = ColStat.is_null_str(series)
            n_null = int(null_mask.sum())
            cs.null_count += n_null
            if n_null:
                cs._saw_null = True

            present = series[~null_mask]
            if len(present) == 0:
                continue
            cs._saw_any = True

            # stringify present values EXACTLY like _observed_values: str(v) on the
            # cell. The chunk is dtype=str already, but a stray NaN-free cast keeps
            # behavior identical to sorted({str(v) for v in unique()}).
            text = present.astype("string")
            # the value strings used for distinct/determinism are str(cell) — for a
            # str-dtype frame this is the cell itself (no float repr surprises).
            str_vals = text.astype(str)

            # 1) distinct two-tier collection (over the chunk's UNIQUE strings).
            cs._observe_distinct(pd.unique(str_vals.to_numpy()))

            # 2) dtype-promotion lattice (over the chunk's distinct stripped tokens
            #    — classification depends only on the token shape, so unique is safe
            #    and bounded). Strip surrounding whitespace to match read_csv, which
            #    trims unquoted numeric fields.
            for tok in pd.unique(str_vals.str.strip().to_numpy()):
                cs._lattice = _promote(cs._lattice, _classify_token(tok))
                if cs._lattice == _K_OBJECT:
                    # absorbing — no token can change it; stop early this chunk.
                    break
            else:
                pass

            # 3) numeric range (only meaningful for numeric columns; cheap + safe
            #    on any column — non-numeric coerces to all-NaN and is skipped).
            coerced = pd.to_numeric(str_vals.str.strip(), errors="coerce")
            cvalid = coerced.dropna()
            if len(cvalid) > 0:
                cmin = float(cvalid.min())
                cmax = float(cvalid.max())
                cs.vmin = cmin if cs.vmin is None else min(cs.vmin, cmin)
                cs.vmax = cmax if cs.vmax is None else max(cs.vmax, cmax)

            # 4) the three numeric-string counters (mirror _is_numeric_coercible_string:
            #    strip via _NUMERIC_STRIP_RE then to_numeric; mark via _NUMERIC_FORMAT_RE).
            cs.ns_non_null += int(len(present))
            stripped = str_vals.str.replace(_NUMERIC_STRIP_RE, "", regex=True).str.strip()
            cs.ns_parse_ok += int(_ns_numeric_ok(stripped).sum())
            cs.ns_currency_mark += int(str_vals.str.contains(_NUMERIC_FORMAT_RE, regex=True).sum())

            # 5) leakage accumulators (only when a target is present + this isn't it).
            if target_l is not None and col != target_l and tgt_raw is not None:
                # numeric Pearson: pair this col's coerced numeric with the target's,
                # over rows where BOTH are present + numeric (matches eda's dropna).
                col_num_full = pd.to_numeric(series, errors="coerce")
                pair_mask = col_num_full.notna() & tgt_num.notna()
                if int(pair_mask.sum()) > 0:
                    cs._observe_corr(col_num_full[pair_mask], tgt_num[pair_mask])
                # categorical determinism: pair present col value -> present target
                # raw value, over rows where BOTH are present (matches eda's dropna).
                if cs.determ_tracked:
                    dmask = (~null_mask) & tgt_notnull
                    if int(dmask.sum()) > 0:
                        cs._observe_determ(
                            series[dmask].astype(str),
                            tgt_raw[dmask].astype(str),
                        )

    # finalize derived fields + the non-null count.
    for cs in cols.values():
        cs.n_non_null = cs.total_count - cs.null_count
        cs.finalize()

    return StreamingStats(
        n_rows=int(n_rows),
        n_cols=int(len(column_order)),
        column_order=list(column_order),
        source_fingerprint="",
        columns=cols,
    )


__all__ = [
    "ColStat",
    "StreamingStats",
    "stream_stats",
    "CHUNK_ROWS",
]
