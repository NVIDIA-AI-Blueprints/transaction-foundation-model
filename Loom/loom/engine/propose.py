"""The ``propose`` field→strategy classifier — bring-your-own-schema authoring.

A first-time user with their OWN tabular schema (e.g. ``account_id, txn_amount,
mcc, channel, dr_cr, balance, txn_ts``) cannot tokenize today: ``tokenize`` ships
only the two hardcoded presets (``financial``/``chain``). This module is the pure
AUTHORING surface that turns an ingested schema + EDA leakage flags into a
reviewable, editable tokenizer SPEC — the field-map — and compiles that field-map
into the existing :class:`~loom.engine.api.TokenizerSpec` so it flows through the
LOCKED compiler + C1/C2/C3 contracts unchanged.

One pure entrypoint (no verb/store/IO coupling — numpy/pandas/stdlib only):

  * :func:`propose_spec` — the classifier. ``(schema, eda_flags, entity, event,
    target, context_len, rows) -> SpecDraft``. Applies the exact rule set (the
    build brief's Ground 1) to every column: EXCLUDE the entity (T2) and the target
    (leakage) and any column failing the "earns a token" gate (recording WHY in
    the rationale, keyed off the EDA flags); pick a strategy + params for every
    surviving column; hand-count ``vocab_size`` / ``tokens_per_event`` and derive
    ``chunk_size = context_len // (tokens_per_event + 1)``.

The field-map this proposer emits is compiled into a :class:`TokenizerSpec` by the
SINGLE production compiler :func:`loom.engine.spec.spec_from_field_map` (exported as
``loom.engine.spec_from_field_map``) — the SAME function ``tokenize --spec`` uses,
so the proposal and the tokenized corpus can never diverge. This module owns NO
compiler of its own (a second, divergent field-map compiler here once masked a
mapping-collapse bug; there is now exactly one).

The rules and constants are the build brief's Ground 1, encoded EXACTLY. The
proposer does NOT re-run the leakage heuristics — it CONSUMES the ``Diagnostic``
cards ``ingest`` already emitted (``data.kind`` ∈ {identity_like,
target_correlated, target_determines}) and surfaces every consumed flag in the
proposal (HARD INVARIANT #4).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Constants — the exact cutoffs from the build brief §C (hardcode these).
# ---------------------------------------------------------------------------

#: bounded-int small-range threshold: ``max - min < SMALL_RANGE_MAX`` → FixedVocab.
SMALL_RANGE_MAX = 50
#: low-card / high-card categorical cutoff: ``< LOW_CARD_MAX`` → Mapping, else Hash.
LOW_CARD_MAX = 500
#: high-card upper bound / free-text cutoff: ``> HIGH_CARD_MAX`` distinct → drop.
HIGH_CARD_MAX = 100_000
#: continuous bins: default 8, clamped to ``[CONT_BINS_MIN, CONT_BINS_MAX]``.
CONT_BINS_DEFAULT = 8
CONT_BINS_MIN = 7
CONT_BINS_MAX = 32
#: TimeDelta (inter-event gap) bins.
TIMEDELTA_BINS = 32
#: hash buckets = clamp(round(corpus_events / HASH_DIVISOR), HASH_MIN, HASH_MAX).
HASH_DIVISOR = 10_000
HASH_MIN = 256
HASH_MAX = 65_536
#: "earns a token" occupancy floor — SAMPLE-AWARE, UNIFORM (the build brief's
#: Fix 1). Tokenizer design is a LOCAL laptop-SAMPLE activity: the old absolute
#: 1K floor assumed the full cloud corpus and wrongly nuked healthy low-card
#: categoricals on a 2.5K-row sample. The floor is now ``clamp(round(n_rows *
#: SAMPLE_OCC_FRACTION), MIN_OCC_FLOOR, MIN_OCC_CAP)`` — it grows gently with the
#: sample but stays LOW-CAPPED, so the intent (don't mint a token seen ~never) is
#: preserved while a value/bin seen a handful of times on a small sample is KEPT.
#: Applied UNIFORMLY across every strategy (no per-strategy exemption): vocab
#: safety for high-cardinality fields comes from the cardinality→strategy routing
#: (high-card → Hash), not from this gate dropping them.
#: ``MIN_OCC_FLOOR`` — the absolute minimum: a token must appear ≥ this many times.
MIN_OCC_FLOOR = 5
#: ``MIN_OCC_CAP`` — the low cap: the floor never exceeds this even on a big corpus.
MIN_OCC_CAP = 50
#: ``SAMPLE_OCC_FRACTION`` — scale the floor to the (sample) corpus size.
SAMPLE_OCC_FRACTION = 0.002
#: coverage floor: present on ~every event.
MIN_COVERAGE = 0.99
#: numeric-coercible-string detection floor (the build brief's GAP 1). A STRING
#: column whose values parse to a number — after stripping common formatting (a
#: leading/trailing ``$``, thousands ``,``, surrounding whitespace, a trailing
#: ``%``) — on ≥ this fraction of non-null rows is treated as CONTINUOUS (the
#: ``amount`` log-bin strategy), exactly like a float column. This recovers the
#: magnitude ordering of a ``$``-formatted ``amount`` column (e.g. "$12.50") the
#: schema sniff reports as a high-cardinality OBJECT/string (which would otherwise
#: be hashed, losing the ordering). It runs AFTER the id/entity/target/near-unique
#: exclusions so an id-shaped numeric-string (e.g. ``account_id``) is still
#: EXCLUDED, never coerced to continuous.
NUMERIC_STRING_FRACTION = 0.95
#: characters stripped before the numeric-coercibility test / the amount preprocess.
_NUMERIC_STRIP_RE = re.compile(r"[\s$,]|%$")
#: the numeric/currency FORMATTING marks that distinguish a magnitude-bearing amount
#: string (``"$12.50"``, ``"1,234.50"``, ``"3.2%"``) from a bare integer-CODE string
#: (``"3812"`` — a merchant/region id that must still HASH, not bin). The
#: numeric-string→continuous detector requires one of these marks on a high fraction
#: of rows, so a column of bare integer codes is NOT mistaken for a continuous amount.
_NUMERIC_FORMAT_RE = re.compile(r"[$,.%]")


def _occ_floor(corpus_events: int) -> int:
    """The sample-aware per-token occupancy floor for a corpus of ``corpus_events``.

    ``clamp(round(corpus_events * SAMPLE_OCC_FRACTION), MIN_OCC_FLOOR, MIN_OCC_CAP)``
    — a low, sample-aware floor (≈5 on a 2.5K-row sample, ≤50 on any larger corpus)
    rather than the old absolute 1K floor that assumed the full cloud corpus and
    dropped healthy low-cardinality categoricals on a local SAMPLE. Uniform across
    all strategies."""
    scaled = round(int(corpus_events) * SAMPLE_OCC_FRACTION)
    return int(max(MIN_OCC_FLOOR, min(MIN_OCC_CAP, scaled)))

#: number of special tokens (specials occupy ids 0-4 before any field block).
N_SPECIALS = 5

# Mirrors of the eda.py constants (reused, NOT redefined — kept in lockstep with
# loom/eda.py so the proposer reads the same id-name family / ratios).
_NEAR_UNIQUE_RATIO = 0.98
_HIGH_CORR = 0.95
_IDENTITY_NAME_RE = re.compile(
    r"(?:^|[_\W])(id|uuid|guid|user|cust|customer|wallet|account|acct|"
    r"address|addr|hash|email|ssn|pan|card_?number|primary_?key|pk)(?:$|[_\W])",
    re.IGNORECASE,
)

# Log-spaced default thresholds for the continuous (amount/balance/fee) binner —
# the worked AMT example: 8 bins from [0.001,0.01,0.1,1,10,100,1000] → tokens 0..7.
_LOG_THRESHOLDS_8 = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


# ---------------------------------------------------------------------------
# The SpecDraft shape — the reviewable, editable proposal.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldProposal:
    """One column's proposed tokenization (an INCLUDED field).

    ``name`` is the logical step name (lower-snake of the column); ``source`` is
    the raw column the preprocess reads; ``strategy`` is the field-map strategy
    keyword (``amount``/``mapping``/``hash``/``fixedvocab``/``calendar``/
    ``timedelta``); ``params`` carries the strategy params; ``token_count`` is the
    hand-counted id-block size this field contributes; ``rationale`` states WHY
    this strategy/sizing was chosen."""

    name: str
    source: str
    strategy: str
    params: dict[str, Any]
    token_count: int
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExclusionProposal:
    """One EXCLUDED column (never reaches the vocab), with WHY + any EDA card.

    ``reason`` is the machine code (``entity``/``target``/``near-unique``/
    ``sparse``/``near-constant``/``free-text``/``starved``); ``rationale`` is the
    human one-liner; ``eda`` embeds the originating ``Diagnostic`` dict (if any)
    so the emitted spec/card states the reasoning (HARD INVARIANT #4)."""

    name: str
    reason: str
    rationale: str
    eda: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpecDraft:
    """The reviewable proposal: an editable field-map + the hand-counted estimates.

    ``fieldmap`` is the declarative ``loom-fieldmap/1`` dict (the artifact a human
    tweaks and ``tokenize --spec`` compiles). ``fields`` / ``excluded`` carry the
    per-field rationale + the excluded list (with EDA cards). ``vocab_size`` /
    ``tokens_per_event`` / ``chunk_size`` are hand-computed (asserted later by
    C1/C3 on the compiled spec)."""

    fieldmap: dict[str, Any]
    fields: tuple[FieldProposal, ...]
    excluded: tuple[ExclusionProposal, ...]
    vocab_size: int
    tokens_per_event: int
    chunk_size: int
    context_len: int
    entity: Optional[str]
    event: Optional[str]
    target: Optional[str]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "fieldmap": self.fieldmap,
            "fields": [f.to_dict() for f in self.fields],
            "excluded": [e.to_dict() for e in self.excluded],
            "vocab_size": self.vocab_size,
            "tokens_per_event": self.tokens_per_event,
            "chunk_size": self.chunk_size,
            "context_len": self.context_len,
            "entity": self.entity,
            "event": self.event,
            "target": self.target,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# EDA-flag consumption — index the cards ingest emitted by column.
# ---------------------------------------------------------------------------


def _flag_data(flag: Any) -> dict[str, Any]:
    """Normalize one EDA flag to its ``data`` dict + a serializable card.

    Accepts either a live :class:`~loom.types.Diagnostic` or its ``to_dict()``
    serialized form (``extras["ingest_report"]["eda_diagnostics"]`` is the latter).
    Returns ``(data, card_dict)``.
    """
    if hasattr(flag, "to_dict"):
        card = flag.to_dict()
    elif isinstance(flag, dict):
        card = flag
    else:  # pragma: no cover - defensive
        card = {"data": {}}
    data = card.get("data") or {}
    return data, card


def _index_flags(eda_flags: Any) -> dict[str, list[dict[str, Any]]]:
    """Map column name → list of its EDA card dicts (kind/data preserved)."""
    by_col: dict[str, list[dict[str, Any]]] = {}
    for flag in eda_flags or []:
        data, card = _flag_data(flag)
        col = data.get("column")
        if col is None:
            continue
        by_col.setdefault(str(col), []).append(card)
    return by_col


def _name_looks_like_identity(name: str) -> bool:
    return bool(_IDENTITY_NAME_RE.search(str(name)))


def _is_datetime_col(name: str, meta: dict[str, Any], event: Optional[str]) -> bool:
    dtype = str(meta.get("dtype", "")).lower()
    if "datetime" in dtype or "date" in dtype or "time" in dtype and "timedelta" not in dtype:
        return True
    # event-time column by name (ingest's chosen event column or *_ts/*_time/date).
    lname = str(name).lower()
    if event and lname == str(event).lower():
        return True
    return bool(re.search(r"(?:^|[_\W])(ts|timestamp|datetime|date|time)(?:$|[_\W])", lname))


def _is_int_dtype(meta: dict[str, Any]) -> bool:
    dtype = str(meta.get("dtype", "")).lower()
    return "int" in dtype and "point" not in dtype


def _is_float_dtype(meta: dict[str, Any]) -> bool:
    dtype = str(meta.get("dtype", "")).lower()
    return "float" in dtype or "double" in dtype or "decimal" in dtype


def _step_name(col: str) -> str:
    """A clean lower-snake logical step name from a raw column name."""
    n = re.sub(r"[^0-9a-zA-Z]+", "_", str(col)).strip("_").lower()
    return n or "field"


# ---------------------------------------------------------------------------
# The classifier — propose_spec.
# ---------------------------------------------------------------------------


def propose_spec(
    *,
    schema: dict[str, Any],
    eda_flags: Any = None,
    entity: Optional[str] = None,
    event: Optional[str] = None,
    target: Optional[str] = None,
    context_len: int = 4096,
    rows: Any = None,
) -> SpecDraft:
    """Propose a tokenizer field-map for an arbitrary schema (the BYO-schema flow).

    ``schema`` is ingest's ``_sniff_schema`` dict: ``{"n_rows", "n_cols",
    "columns": {col: {"dtype", "null_frac", "n_unique"}}}``. ``eda_flags`` is the
    list of EDA :class:`~loom.types.Diagnostic` cards (or their serialized dicts)
    from ``leakage_scan`` — CONSUMED, not re-run. ``entity`` (the sequence owner /
    grouping key) and ``target`` (the label) are EXCLUDED from the vocab; ``event``
    names the per-event semantics. Returns a :class:`SpecDraft`.

    ``rows`` is the OPTIONAL ingested rows frame (a ``pandas.DataFrame`` read from
    the ``IngestDataset`` payload). When supplied, a low-cardinality categorical
    field carries the REAL observed value list (``values: [...]``) enumerated off
    the frame — so the compiled ``MappingPassthrough`` maps each distinct value to
    its own token instead of collapsing every value to the single ``_<default>``
    token (the schema sniff alone carries only ``n_unique``, a COUNT, which is
    insufficient to build a working mapping vocab). The value list keeps the vocab
    config-only (C2-clean: it is frozen into the spec, not fitted at materialize
    time). When ``rows`` is absent the field carries ``n_values`` only and the human
    must fill in the values before tokenize (the proposal is then advisory).

    Deterministic + pure: same inputs → identical draft (column order is the
    schema order; the enumerated values are sorted so the draft is stable).
    """
    columns: dict[str, Any] = dict(schema.get("columns", {}))
    n_rows = int(schema.get("n_rows", 0) or 0)
    corpus_events = max(n_rows, 1)
    flags_by_col = _index_flags(eda_flags)

    entity_l = str(entity).lower() if entity else None
    target_l = str(target).lower() if target else None

    fields: list[FieldProposal] = []
    excluded: list[ExclusionProposal] = []
    warnings: list[str] = []

    # The set of columns the EDA flagged as leakage (correlated / determines).
    def _leakage_card(col: str) -> Optional[dict[str, Any]]:
        for card in flags_by_col.get(col, []):
            if (card.get("data") or {}).get("kind") in ("target_correlated", "target_determines"):
                return card
        return None

    def _identity_card(col: str) -> Optional[dict[str, Any]]:
        for card in flags_by_col.get(col, []):
            if (card.get("data") or {}).get("kind") == "identity_like":
                return card
        return None

    for col, meta in columns.items():
        col_l = str(col).lower()
        n_unique = int(meta.get("n_unique", 0) or 0)
        null_frac = float(meta.get("null_frac", 0.0) or 0.0)
        coverage = 1.0 - null_frac
        # ingest's n_unique is over non-null; unique_ratio over the present rows.
        n_non_null = max(int(round(coverage * n_rows)), 0)
        unique_ratio = (n_unique / n_non_null) if n_non_null else 0.0

        # ---- A. Exclusion gate (run FIRST) ---------------------------------

        # A.1 — the entity column (T2). Hard rule; identity lives in the grouping.
        if entity_l is not None and col_l == entity_l:
            excluded.append(
                ExclusionProposal(
                    name=col,
                    reason="entity",
                    rationale="entity (T2): identity comes from history, not an embedding",
                    eda=_identity_card(col),
                )
            )
            continue

        # A.2 — the declared target/label (and EDA leakage cards).
        if target_l is not None and col_l == target_l:
            excluded.append(
                ExclusionProposal(
                    name=col,
                    reason="target",
                    rationale="target / label column (leakage if tokenized as a feature)",
                    eda=None,
                )
            )
            continue
        leak = _leakage_card(col)
        if leak is not None:
            kind = (leak.get("data") or {}).get("kind")
            if kind == "target_correlated":
                why = "target / leakage (|corr|>=0.95 with the label)"
            else:
                why = "target / leakage (perfectly determines the label)"
            excluded.append(
                ExclusionProposal(name=col, reason="target", rationale=why, eda=leak)
            )
            continue

        # A.3 — "earns a token" gate.
        # near-constant — no information.
        if n_unique <= 1:
            excluded.append(
                ExclusionProposal(
                    name=col,
                    reason="near-constant",
                    rationale=f"near-constant: {n_unique} distinct value(s) — no information",
                    eda=None,
                )
            )
            continue
        # coverage — not present on ~every event.
        if coverage < MIN_COVERAGE:
            excluded.append(
                ExclusionProposal(
                    name=col,
                    reason="sparse",
                    rationale=(
                        f"sparse: present on {coverage:.0%} of events "
                        f"(< {MIN_COVERAGE:.0%}) — does not earn a token"
                    ),
                    eda=None,
                )
            )
            continue
        # near-unique / id-shaped (non-entity). Drop by default; human may hash.
        # This is a CATEGORICAL/string id test — continuous numerics are near-unique
        # BY VALUE and are meant to be BINNED, not dropped, so they are exempt (the
        # gate applies only to non-numeric columns and id-shaped names).
        id_card = _identity_card(col)
        near_unique = unique_ratio >= _NEAR_UNIQUE_RATIO
        name_hit = _name_looks_like_identity(col)
        is_dt = _is_datetime_col(col, meta, event)
        is_numeric = _is_float_dtype(meta) or _is_int_dtype(meta)
        # GAP 1 — numeric-coercible STRING: a non-id-named OBJECT/string column whose
        # values parse to a number after stripping common formatting ($, thousands ',',
        # whitespace, trailing %) on a high fraction of non-null rows. This is a
        # ``$``-formatted ``amount`` the schema sniff reports as a high-cardinality
        # string; it is a CONTINUOUS feature (near-unique BY VALUE) and must be BINNED,
        # not hashed or dropped. We only probe non-datetime, non-numeric-dtype, NON-id
        # NAMED columns (so an ``account_id``-style numeric-string id stays excluded
        # by the id-name gate below and is NEVER coerced to continuous).
        is_numeric_string = False
        if not is_dt and not is_numeric and not name_hit:
            is_numeric_string = _is_numeric_coercible_string(rows, col)
        # A near-unique numeric (no id-shaped name) is a continuous feature → bin it;
        # the same exemption covers a numeric-coercible string (a ``$``-amount), so the
        # near-unique gate below never strips a currency/numeric-string amount.
        numeric_continuous = (is_numeric or is_numeric_string) and not name_hit
        if not is_dt and not numeric_continuous and (id_card is not None or near_unique or name_hit):
            if near_unique or (id_card and (id_card.get("data") or {}).get("near_unique")):
                why = (
                    f"near-unique (id-shaped): {unique_ratio:.0%} distinct — "
                    "excluded by default; keep only by hashing it explicitly"
                )
                reason = "near-unique"
            else:
                why = (
                    "id-shaped column name (not the entity) — excluded by default; "
                    "review and keep (e.g. hash) only if it is behavior-bearing"
                )
                reason = "near-unique"
            excluded.append(
                ExclusionProposal(name=col, reason=reason, rationale=why, eda=id_card)
            )
            warnings.append(f"{col}: review — {why}")
            continue

        # ---- B. Strategy decision table (surviving columns) -----------------

        proposal: Optional[FieldProposal] = None

        if is_dt:
            # datetime → calendar tokens (HOUR/DOW/MONTH) + inter-event TimeDelta.
            # These expand into MULTIPLE steps; we record one logical proposal per
            # part below by appending several FieldProposal entries.
            base = _step_name(col)
            cal_parts = [
                ("hour", 24, "calendar HOUR via FixedVocab(0,23)"),
                ("dow", 7, "calendar DOW via FixedVocab(0,6)"),
                ("month", 12, "calendar MONTH via FixedVocab 0-based (MONTH_00..MONTH_11)"),
            ]
            for part, cnt, why in cal_parts:
                fields.append(
                    FieldProposal(
                        name=f"{base}_{part}",
                        source=col,
                        strategy="calendar",
                        params={"part": part},
                        token_count=cnt,
                        rationale=(
                            f"timestamp {col!r}: {why} — calendar seasonality, "
                            "config-only (C2-clean)"
                        ),
                    )
                )
            fields.append(
                FieldProposal(
                    name=f"{base}_gap",
                    source=col,
                    strategy="timedelta",
                    params={"bins": TIMEDELTA_BINS},
                    token_count=TIMEDELTA_BINS,
                    rationale=(
                        f"timestamp {col!r}: inter-event TimeDelta "
                        f"({TIMEDELTA_BINS} log-bins, seconds→months) — the T1 gap"
                    ),
                )
            )
            continue

        if _is_float_dtype(meta) or is_numeric_string:
            # continuous → log-spaced deterministic threshold bins (C2-clean). Covers
            # both a float column and a numeric-coercible STRING (a ``$``-formatted
            # ``amount``); the generic ``amount`` preprocess strips ``$``/``,`` before
            # binning so the magnitude ordering is preserved either way (GAP 1).
            bins = CONT_BINS_DEFAULT
            # occupancy: corpus_events / bins must clear the sample-aware floor.
            occ_floor = _occ_floor(corpus_events)
            if corpus_events / max(bins, 1) < occ_floor:
                bins = max(CONT_BINS_MIN, corpus_events // max(occ_floor, 1))
                bins = min(bins, CONT_BINS_MAX)
            kind = "continuous float" if _is_float_dtype(meta) else (
                "numeric-coercible string (currency/numeric formatting stripped)"
            )
            proposal = FieldProposal(
                name=_step_name(col),
                source=col,
                strategy="amount",
                params={"bins": int(bins)},
                token_count=int(bins),
                rationale=(
                    f"{kind}: {int(bins)} log-spaced deterministic "
                    "threshold bins (preferred over a fitted quantile binner — "
                    "no fitted artifact, C2-clean)"
                ),
            )
        elif _is_int_dtype(meta):
            # bounded int small range → FixedVocab; else treat as categorical.
            rng = _int_range(meta, n_unique)
            if rng is not None and (rng[1] - rng[0]) < SMALL_RANGE_MAX:
                lo, hi = rng
                span = hi - lo
                proposal = FieldProposal(
                    name=_step_name(col),
                    source=col,
                    strategy="fixedvocab",
                    params={"min": 0, "max": int(span)},
                    token_count=int(span) + 1,
                    rationale=(
                        f"bounded int, range {lo}..{hi} (< {SMALL_RANGE_MAX}): "
                        f"FixedVocab with a 0-based shift (min=0, max={span}) — "
                        "avoids the MONTH_12/CARD_0 min_val>0 collision the doc warns of"
                    ),
                )
            else:
                proposal = _categorical_proposal(
                    col, n_unique, corpus_events, values=_observed_values(rows, col)
                )
        else:
            # string / object → categorical or free-text by cardinality.
            proposal = _categorical_proposal(
                col, n_unique, corpus_events, values=_observed_values(rows, col)
            )

        if proposal is None:
            # free text / > HIGH_CARD_MAX distinct → dropped by default (hashing
            # is a human opt-in, flagged). Recorded as an explicit exclusion.
            excluded.append(
                ExclusionProposal(
                    name=col,
                    reason="free-text",
                    rationale=(
                        f"free text / very high cardinality ({n_unique} distinct "
                        f"> {HIGH_CARD_MAX}): dropped by default — hash only on "
                        "explicit opt-in (flagged)"
                    ),
                    eda=None,
                )
            )
            warnings.append(
                f"{col}: dropped — free text ({n_unique} distinct > {HIGH_CARD_MAX})"
            )
            continue

        # B-gate (b): the "won't starve" occupancy check, applied AFTER a tentative
        # bin/strategy count is chosen — drop only if a token would be seen ~never
        # even on this (possibly small) SAMPLE. The floor is sample-aware (≈5 on a
        # 2.5K-row sample) and applied UNIFORMLY across every strategy — no field
        # type is special. A healthy low-cardinality categorical on a small sample
        # therefore SURVIVES (e.g. a 4-value field at ~640 occ/value clears 5).
        # High-cardinality fields are NOT dropped here: the cardinality→strategy
        # routing already sent them to Hash (a corpus-sized bucket vocab whose
        # collisions are expected), and the low floor lets that hash survive — the
        # routing, not this gate, is what keeps the vocab safe.
        occ_floor = _occ_floor(corpus_events)
        occ = corpus_events / max(proposal.token_count, 1)
        if occ < occ_floor:
            excluded.append(
                ExclusionProposal(
                    name=col,
                    reason="starved",
                    rationale=(
                        f"would starve: {proposal.token_count} tokens over "
                        f"{corpus_events} events = {occ:.0f} occ/token "
                        f"(< {occ_floor}, the sample-aware floor); a token would be "
                        "seen ~never even on this sample — not a healthy vocab"
                    ),
                    eda=None,
                )
            )
            warnings.append(
                f"{col}: dropped — {proposal.token_count} tokens starve at "
                f"{occ:.0f} occ/token (< {occ_floor})"
            )
            continue

        fields.append(proposal)

    # ---- E. Derived numbers (hand-computed; asserted later by C1/C3) --------

    tokens_per_event = len(fields)
    vocab_size = N_SPECIALS + sum(f.token_count for f in fields)
    # chunk_size = context_len // (tokens_per_event + 1) — the +1 is the <sep>.
    chunk_size = context_len // (tokens_per_event + 1) if tokens_per_event >= 0 else 0

    fieldmap = _build_fieldmap(
        fields, entity=entity, event=event, target=target, context_len=context_len
    )

    return SpecDraft(
        fieldmap=fieldmap,
        fields=tuple(fields),
        excluded=tuple(excluded),
        vocab_size=vocab_size,
        tokens_per_event=tokens_per_event,
        chunk_size=chunk_size,
        context_len=context_len,
        entity=entity,
        event=event,
        target=target,
        warnings=tuple(warnings),
    )


def _int_range(meta: dict[str, Any], n_unique: int) -> Optional[tuple[int, int]]:
    """The integer value range for a bounded-int column.

    ingest's ``_sniff_schema`` does not carry min/max, so we infer a conservative
    range from ``n_unique`` (the values are assumed contiguous 0..n_unique-1 when
    no explicit range is given). If the schema carries an explicit ``min``/``max``
    (a hand-edited field-map or an enriched sniff), use it.
    """
    if "min" in meta and "max" in meta:
        try:
            return int(meta["min"]), int(meta["max"])
        except (TypeError, ValueError):
            return None
    if n_unique <= 0:
        return None
    # Treat a small-cardinality int as a contiguous 0..n_unique-1 bounded range.
    return 0, n_unique - 1


def _observed_values(rows: Any, col: str) -> Optional[list[str]]:
    """The sorted distinct NON-null values of ``col`` in the rows frame, as strings.

    Returns ``None`` when no frame is available or the column is absent (the caller
    then falls back to an ``n_values``-only mapping). Sorted so the proposal is
    deterministic; stringified to match :class:`MappingPassthrough`'s string vocab.
    """
    if rows is None:
        return None
    try:
        if col not in getattr(rows, "columns", []):
            return None
        series = rows[col].dropna()
    except Exception:  # pragma: no cover - defensive (non-DataFrame rows)
        return None
    return sorted({str(v) for v in series.unique()})


def _is_numeric_coercible_string(rows: Any, col: str) -> bool:
    """True iff ``col`` is a FORMATTED numeric STRING — a magnitude-bearing amount
    the schema sniff reports as a high-cardinality OBJECT string (e.g. "$12.50",
    "1,234.50", "3.2%") — that should bin like a float, NOT a bare integer-CODE
    string (e.g. "3812", a merchant/region id that must still HASH).

    The test (GAP 1): on the NON-null rows, after stripping common formatting (``$``,
    thousands ``,``, surrounding whitespace, a trailing ``%``), ≥
    ``NUMERIC_STRING_FRACTION`` parse to a number AND ≥ ``NUMERIC_STRING_FRACTION``
    carry an actual numeric/currency formatting mark (``$ , . %``). The
    formatting-mark requirement is what separates a ``$``/decimal amount from a column
    of bare integer codes (which coerce but carry NO mark) — so an integer-string
    merchant/region id is NOT swallowed into the continuous binner.

    Returns ``False`` when no frame is available / the column is absent / the column
    is empty (the caller then falls back to the categorical path — advisory, exactly
    the existing ``n_values``-only degradation). Pure pandas, no fitting — inspects
    only the cell text (config-time, C2-clean)."""
    if rows is None:
        return False
    try:
        if col not in getattr(rows, "columns", []):
            return False
        import pandas as pd

        series = rows[col].dropna()
        if len(series) == 0:
            return False
        text = series.astype(str)
        stripped = text.str.replace(_NUMERIC_STRIP_RE, "", regex=True).str.strip()
        coerced = pd.to_numeric(stripped, errors="coerce")
        numeric_frac = float(coerced.notna().mean())
        formatted_frac = float(text.str.contains(_NUMERIC_FORMAT_RE, regex=True).mean())
        return numeric_frac >= NUMERIC_STRING_FRACTION and formatted_frac >= NUMERIC_STRING_FRACTION
    except Exception:  # pragma: no cover - defensive (non-DataFrame rows)
        return False


def _categorical_proposal(
    col: str,
    n_unique: int,
    corpus_events: int,
    *,
    values: Optional[list[str]] = None,
) -> Optional[FieldProposal]:
    """Categorical column → Mapping (low-card) / Hash (high-card) / drop (free-text).

    When ``values`` (the observed value list from the rows frame) is supplied for a
    low-card column, the proposal carries a REAL ``values: [...]`` list so the
    compiled ``MappingPassthrough`` maps each distinct value to its own token. Absent
    a frame it carries ``n_values`` only (a count) and the human must supply the
    values before tokenize — otherwise every value would collapse to the default."""
    name = _step_name(col)
    if n_unique < LOW_CARD_MAX:
        # Mapping: observed values + exactly 1 default. The token count is the count
        # of DISTINCT observed values + 1 default. When the rows frame gave us the
        # real value list, freeze it into the field-map (config-only, C2-clean) so
        # the MappingPassthrough has a real vocab; else carry n_values for the human.
        if values is not None:
            params: dict[str, Any] = {"values": list(values), "default": "UNK"}
            token_count = len(values) + 1
            note = "over the observed values + 1 default"
        else:
            params = {"n_values": int(n_unique), "default": "UNK"}
            token_count = int(n_unique) + 1
            note = "(supply the value list before tokenize) + 1 default"
        return FieldProposal(
            name=name,
            source=col,
            strategy="mapping",
            params=params,
            token_count=int(token_count),
            rationale=(
                f"categorical, {n_unique} distinct (< {LOW_CARD_MAX}): "
                f"Mapping {note} (size {token_count})"
            ),
        )
    if n_unique <= HIGH_CARD_MAX:
        buckets = _hash_buckets(corpus_events)
        return FieldProposal(
            name=name,
            source=col,
            strategy="hash",
            params={"buckets": int(buckets)},
            token_count=int(buckets),
            rationale=(
                f"high-cardinality, {n_unique} distinct (~merchant/counterparty): "
                f"Hash into {buckets} buckets (~corpus_events/{HASH_DIVISOR}) — "
                "expect collisions"
            ),
        )
    return None  # > HIGH_CARD_MAX → drop (handled by caller as free-text)


def _hash_buckets(corpus_events: int) -> int:
    """``clamp(round(corpus_events / 10_000), 256, 65_536)`` — the brief's rule."""
    raw = round(corpus_events / HASH_DIVISOR)
    return int(max(HASH_MIN, min(HASH_MAX, raw)))


# ---------------------------------------------------------------------------
# Field-map assembly — the editable ``loom-fieldmap/1`` dict.
# ---------------------------------------------------------------------------

FIELDMAP_VERSION = "loom-fieldmap/1"


def _build_fieldmap(
    fields: list[FieldProposal],
    *,
    entity: Optional[str],
    event: Optional[str],
    target: Optional[str],
    context_len: int,
) -> dict[str, Any]:
    """Assemble the declarative ``loom-fieldmap/1`` dict from the proposed fields.

    This is the artifact the ``propose`` verb persists (as YAML) and the SINGLE
    production compiler :func:`loom.engine.spec.spec_from_field_map` consumes via
    ``tokenize --spec``. Each entry carries the strategy + its params (notably the
    real ``values`` list for a ``mapping`` field, enumerated off the rows frame)."""
    fm_fields: list[dict[str, Any]] = []
    for f in fields:
        entry: dict[str, Any] = {"name": f.name, "source": f.source, "strategy": f.strategy}
        entry.update(f.params)
        fm_fields.append(entry)
    return {
        "version": FIELDMAP_VERSION,
        "entity": entity,
        "event": event,
        "target": target,
        "context_len": context_len,
        "fields": fm_fields,
    }


__all__ = [
    "SpecDraft",
    "FieldProposal",
    "ExclusionProposal",
    "propose_spec",
    "FIELDMAP_VERSION",
    # constants (exported so the verb layer / tests reference the same numbers).
    "SMALL_RANGE_MAX",
    "LOW_CARD_MAX",
    "HIGH_CARD_MAX",
    "CONT_BINS_DEFAULT",
    "TIMEDELTA_BINS",
    "MIN_OCC_FLOOR",
    "MIN_OCC_CAP",
    "SAMPLE_OCC_FRACTION",
    "MIN_COVERAGE",
    "NUMERIC_STRING_FRACTION",
    "N_SPECIALS",
]
