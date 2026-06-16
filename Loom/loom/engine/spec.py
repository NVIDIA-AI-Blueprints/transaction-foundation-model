"""Preset tokenizer specs + the financial CPU preprocess (build brief).

These factories build :class:`~loom.engine.api.TokenizerSpec` objects from the
exact reference constants. The ``financial`` preset compiles to ``vocab_size ==
6251`` (6283 with the time-delta field); the ``chain`` preset's vocab is DERIVED
(not hardcoded). The CPU preprocess reproduces the reference field derivations on
pandas — the reference ``src/tokenizer/*.py`` is cuDF-only and is used only as a
correctness oracle (Loom's clean-compile REPLACES hand-writing
``src/tokenizer/chain_pipeline.py`` for Phase 1; confirmed 2026-06-15).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .api import (
    AmountStrategy,
    FieldStep,
    FixedVocab,
    Hash,
    MappingDirect,
    MappingPassthrough,
    MappingRange,
    TimeDelta,
    TokenizerSpec,
)
from .strategies import KMer, split_kmers

# ── VERBATIM reference constants (build brief / financial_pipeline.py) ──────

AMOUNT_THRESHOLDS = [0, 10, 50, 100, 500, 1000, 5000]

INDUSTRY_RANGES: tuple[tuple[int, int, str], ...] = (
    (0, 1499, "AGRICULTURAL"),
    (1500, 2999, "CONTRACTED"),
    (3000, 3299, "AIRLINES"),
    (3300, 3499, "CAR_RENTAL"),
    (3500, 3999, "LODGING"),
    (4000, 4799, "TRANSPORTATION"),
    (4800, 4999, "UTILITIES"),
    (5000, 5599, "RETAIL"),
    (5600, 5699, "CLOTHING"),
    (5700, 7299, "MISC_STORES"),
    (7300, 7999, "BUSINESS"),
    (8000, 8999, "PROFESSIONAL"),
    (9000, 9999, "GOVERNMENT"),
)

KNOWN_MCCS = [
    -1, 1711, 3000, 3001, 3005, 3006, 3007, 3008, 3009, 3058, 3066,
    3075, 3132, 3144, 3174, 3256, 3260, 3359, 3387, 3389, 3390, 3393,
    3395, 3405, 3504, 3509, 3596, 3640, 3684, 3722, 3730, 3771, 3775,
    3780, 4111, 4112, 4121, 4131, 4214, 4411, 4511, 4722, 4784, 4814,
    4829, 4899, 4900, 5045, 5094, 5192, 5193, 5211, 5251, 5261, 5300,
    5310, 5311, 5411, 5499, 5533, 5541, 5621, 5651, 5655, 5661, 5712,
    5719, 5722, 5732, 5733, 5812, 5813, 5814, 5815, 5816, 5912, 5921,
    5932, 5941, 5942, 5947, 5970, 5977, 6300, 7011, 7210, 7230, 7276,
    7349, 7393, 7531, 7538, 7542, 7549, 7801, 7802, 7832, 7922, 7995,
    7996, 8011, 8021, 8041, 8043, 8049, 8062, 8099, 8111, 8931, 9402,
]

CHIP_MAPPING = {
    "SWIPE TRANSACTION": "SWIPE",
    "CHIP TRANSACTION": "CHIP",
    "ONLINE TRANSACTION": "ONLINE",
}

ALL_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "VI", "GU", "AS", "MP", "XX", "ONLINE",
]

# DEX next-trade venues (chain preset, Step 0.2/0.3).
DEX_VENUES = ["DEXETH", "DEXBASE", "DEXSOL"]
DEX_SIDE_MAPPING = {"BUY": "BUY", "SELL": "SELL"}

_MERCH_CLEAN_RE = re.compile(r"[^A-Z0-9\s\-]")

# ── Custom field-map (bring-your-own-schema) defaults ───────────────────────
# These mirror the Ground-1/A classifier constants. The continuous `amount`
# strategy compiles to a FixedVocab over the COUNT of log-spaced thresholds
# crossed (0..n_thresholds), i.e. `bins` tokens for `bins-1` thresholds — exactly
# the financial AMT family (7 tokens from [10,50,100,500,1000,5000] thresholds).
CONT_BINS_DEFAULT = 8
TIMEDELTA_BINS_DEFAULT = 32
# Default alphabet for the generic `kmer` strategy (DNA). RNA/protein/arbitrary
# alphabets are supplied per field via `alphabet:` — the strategy is fixed-alphabet
# generic, NOT DNA-special; DNA is only the default + the worked control.
DNA_ALPHABET: tuple[str, ...] = ("A", "C", "G", "T")
# Calendar parts → (min, max, pad_width). MONTH is 0-based (the sharp-edge-#9 /
# MONTH_12≡CARD_0 collision fix: shift the raw 1..12 month to 0..11 in preprocess).
_CALENDAR_PARTS: dict[str, tuple[int, int, int]] = {
    "hour": (0, 23, 2),
    "dow": (0, 6, 0),
    "month": (0, 11, 2),
}


# ── Preset factories ────────────────────────────────────────────────────────


def financial_spec(
    *,
    merchant_hash_size: int = 2000,
    amount_strategy: AmountStrategy = AmountStrategy.FIXED,
    include_time_delta: bool = False,
    drop_steps: tuple[str, ...] = (),
) -> TokenizerSpec:
    """The reference ``financial`` (TabFormer) 12-field spec → vocab 6251.

    With ``include_time_delta=True`` a 13th ``TDIF`` field (32 log-bins) is
    appended → vocab 6283, tokens_per_txn 13, chunk_size 292. ``drop_steps``
    removes named steps (e.g. ``("cust",)`` for T2 → vocab 3251)."""
    amt_fitted = amount_strategy in (AmountStrategy.QUANTILE, AmountStrategy.KMEANS)

    steps: list[FieldStep] = [
        FieldStep("amt", "amt_val", FixedVocab("AMT", 0, 6, 0), fitted=amt_fitted),
        FieldStep("merch", "merch", Hash("MERCH", merchant_hash_size)),
        FieldStep("cat", "mcc_int", MappingRange("CAT", INDUSTRY_RANGES, "GENERAL")),
        FieldStep(
            "mcc",
            "mcc_str",
            MappingPassthrough("MCC", tuple(str(m) for m in KNOWN_MCCS), "-1"),
        ),
        FieldStep("hour", "hour", FixedVocab("HOUR", 0, 23, 2)),
        FieldStep("dow", "dow", FixedVocab("DOW", 0, 6, 0)),
        FieldStep("month", "month", FixedVocab("MONTH", 1, 12, 2)),
        FieldStep("card", "card", FixedVocab("CARD", 0, 9, 0)),
        FieldStep("chip", "chip_upper", MappingDirect("CHIP", CHIP_MAPPING, "UNK")),
        FieldStep("zip3", "zip3", FixedVocab("ZIP3", 0, 999, 3)),
        FieldStep("state", "state_clean", MappingPassthrough("STATE", tuple(ALL_STATES), "XX")),
        FieldStep("cust", "cust", FixedVocab("CUST", 0, 2999, 0)),
    ]
    if include_time_delta:
        steps.append(FieldStep("tdif", "time_delta_s", TimeDelta("TDIF", 32, 10.0)))

    if drop_steps:
        drop = set(drop_steps)
        steps = [s for s in steps if s.name not in drop]

    return TokenizerSpec(
        steps=tuple(steps),
        preset="financial",
        entity="cust",
        event="transaction",
        amount_strategy=amount_strategy,
    )


def chain_spec(
    *,
    item_hash_size: int = 5000,
    size_bins: int = 8,
    include_identity_token: bool = False,
    drop_steps: tuple[str, ...] = (),
) -> TokenizerSpec:
    """The Phase-0 ``chain`` (DEX next-trade) spec; vocab is DERIVED.

    Fields: venue (FixedVocab, 3), side (MappingDirect BUY/SELL + default),
    item (Hash ~5000), size_usd (TimeDelta-style log bins ~8 via Hash-free
    FixedVocab? no — log bins), inter-trade gap (TimeDelta 32, T1 non-negotiable),
    hour (24), dow (7). NO wallet-identity token by default (T2). entity=wallet,
    event=trade; sort by [wallet, timestamp] (C6). tokens_per_event == 7 →
    chunk_size 512 at context_len 4096."""
    steps: list[FieldStep] = [
        FieldStep(
            "venue",
            "venue",
            FixedVocab("VENUE_DEX", 0, len(DEX_VENUES) - 1, 0),
        ),
        FieldStep("side", "side", MappingDirect("SIDE", DEX_SIDE_MAPPING, "UNK")),
        FieldStep("item", "item", Hash("ITEM", item_hash_size)),
        # size_usd binned with log-spaced TimeDelta-family bins (size_bins).
        FieldStep("size", "size_usd", TimeDelta("SIZE", size_bins, 10.0)),
        # inter-trade gap (T1 non-negotiable).
        FieldStep("gap", "gap_s", TimeDelta("GAP", 32, 10.0)),
        FieldStep("hour", "hour", FixedVocab("HOUR", 0, 23, 2)),
        FieldStep("dow", "dow", FixedVocab("DOW", 0, 6, 0)),
    ]
    if include_identity_token:
        # T2: kept OFF by default. A wallet-identity hash would leak identity.
        steps.append(FieldStep("wallet", "wallet", Hash("WALLET", 4096)))

    if drop_steps:
        drop = set(drop_steps)
        steps = [s for s in steps if s.name not in drop]

    return TokenizerSpec(
        steps=tuple(steps),
        preset="chain",
        entity="wallet",
        event="trade",
        amount_strategy=AmountStrategy.FIXED,
    )


# ── Custom field-map → TokenizerSpec (bring-your-own-schema compile path) ───


def _log_thresholds(n_bins: int) -> list[float]:
    """``n_bins`` tokens ⇒ ``n_bins - 1`` log-spaced thresholds (the AMT family).

    Mirrors the worked AMT example (8 tokens from the 7 thresholds
    ``[0.001,0.01,0.1,1,10,100,1000]`` → tokens 0..7): a value crossing ``k`` of
    the thresholds maps to token ``k`` in ``0..n_bins-1``. Deterministic, config
    only (C2-clean) — no fitting, no data. Thresholds span 10**-3 .. 10**(n-4) so
    8 bins reproduces the documented decade ladder."""
    n_thresh = max(1, int(n_bins) - 1)
    # Decade ladder starting two decades below 1.0 (matches the AMT worked example
    # at n_bins=8: exponents -3..3 → [0.001 … 1000]).
    start_exp = -(n_thresh // 2) - 1
    return [10.0 ** (start_exp + i) for i in range(n_thresh)]


def _unique_prefix(base: str, seen_prefixes: set[str], *, explicit: bool = False) -> str:
    """A unique UPPER token prefix so two fields never ACCIDENTALLY collide.

    C1 injectivity: two fields sharing a prefix over overlapping local indices emit
    the SAME ``PREFIX_<i>`` token strings and collide. When a field's AUTO-DERIVED
    prefix is already claimed (e.g. a second ``calendar`` field over a second
    timestamp column reuses ``HOUR``, or two amount fields named ``amt``), append a
    numeric suffix (``HOUR``→``HOUR1``→``HOUR2``) so each field gets a disjoint id
    block — the multi-timestamp / repeated-name schema compiles out of the box.

    An EXPLICIT user-written ``prefix:`` is honored VERBATIM (never suffixed): a
    human who deliberately writes two identical prefixes is authoring a genuinely
    bad spec, and C1 must REFUSE it with the named diff (the safety net, INVARIANT
    #2) rather than silently rewrite it into a different vocab than they declared.
    The prefix is still recorded so a later auto-derived one steps around it."""
    if explicit:
        seen_prefixes.add(base)
        return base
    p = base
    i = 1
    while p in seen_prefixes:
        p = f"{base}{i}"
        i += 1
    seen_prefixes.add(p)
    return p


def _field_step_from_entry(entry: dict, *, corpus_events: int, seen_prefixes: set[str]) -> FieldStep:
    """Translate ONE declarative field-map entry → a :class:`FieldStep`.

    Each ``entry`` is ``{name, source, strategy, ...params}`` where ``strategy`` is
    one of the Ground-1/A strategy names. This is a pure config→dataclass mapping;
    it instantiates an EXISTING :class:`Strategy` so the result flows through the
    LOCKED ``compile_spec`` + C1/C2/C3 unchanged. ``corpus_events`` sizes the Hash
    bucket count when a field-map omits an explicit ``buckets``. ``seen_prefixes``
    tracks claimed token prefixes so two fields with AUTO-DERIVED prefixes never
    accidentally collide (C1 — e.g. two timestamp/amount columns); an EXPLICIT
    ``prefix:`` is honored verbatim so a deliberate collision still trips C1."""
    name = str(entry["name"])
    source = str(entry.get("source", name))
    strat = str(entry["strategy"]).lower()
    has_explicit_prefix = "prefix" in entry
    prefix = _unique_prefix(
        str(entry.get("prefix", name)).upper(), seen_prefixes, explicit=has_explicit_prefix
    )

    if strat in ("fixedvocab", "fixed", "fixed_vocab"):
        min_val = int(entry.get("min", 0))
        max_val = int(entry["max"])
        pad = int(entry.get("pad_width", entry.get("pad", 0)))
        return FieldStep(name, source, FixedVocab(prefix, min_val, max_val, pad))

    if strat in ("mapping", "map"):
        values = tuple(str(v) for v in entry.get("values", ()))
        default = str(entry.get("default", "UNK"))
        return FieldStep(name, source, MappingPassthrough(prefix, values, default))

    if strat in ("hash", "categorical_hash"):
        buckets = entry.get("buckets")
        if buckets is None:
            buckets = _hash_buckets(corpus_events)
        return FieldStep(name, source, Hash(prefix, int(buckets)))

    if strat in ("amount", "continuous", "logbins", "numeric", "numerical"):
        # Log-threshold bins → FixedVocab over the bin index 0..bins-1. C2-clean
        # (deterministic thresholds, NO fitted artifact). The bin index is derived
        # by the preprocess from a recipe-encoded source (``__amt__<col>__<bins>``)
        # so the spec ALONE drives materialize — no separate field-map needed.
        bins = int(entry.get("bins", CONT_BINS_DEFAULT))
        bins = max(2, min(bins, 32))
        return FieldStep(name, f"__amt__{source}__{bins}", FixedVocab(prefix, 0, bins - 1, 0))

    if strat in ("timedelta", "time_delta", "gap"):
        # Inter-event gap (seconds) per entity, derived from the timestamp source.
        bins = int(entry.get("bins", TIMEDELTA_BINS_DEFAULT))
        return FieldStep(
            name,
            f"__gap__{source}",
            TimeDelta(prefix, bins, float(entry.get("max_years", 10.0))),
        )

    if strat in ("kmer", "k_mer", "kmers"):
        # A fixed-alphabet biological/string sequence → per-position k-mer tokens.
        # GENERIC: any alphabet (DNA {A,C,G,T} default / RNA / protein / arbitrary).
        # The recipe source ``__kmer__<col>__<k>__<stride>`` tells the preprocess to
        # EXPLODE one sequence row into one row per k-mer position (the "one sequence
        # → many tokens" fan-out); the KMer strategy then maps each window 1:1 to its
        # token. The vocab is the dense ``alphabet**k`` enumeration (or a config-pinned
        # ``observed`` subset) — injective + dense, so C1/C2/C3 derive unchanged.
        # ``prefix`` is the field name (a field named ``kmer`` → ``KMER_<kmer>``
        # tokens), uniquified so a second k-mer field gets a disjoint block (C1-clean).
        k = int(entry.get("k", 3))
        stride = int(entry.get("stride", 1))
        alphabet = entry.get("alphabet") or DNA_ALPHABET
        alpha = tuple(str(c).upper() for c in alphabet)
        observed = entry.get("observed")
        obs = tuple(str(m).upper() for m in observed) if observed else None
        return FieldStep(
            name,
            f"__kmer__{source}__{k}__{stride}",
            KMer(prefix, k, alpha, stride=stride, observed=obs),
        )

    raise ValueError(
        f"field {name!r}: unknown strategy {strat!r} "
        "(expected one of: fixedvocab, mapping, hash, amount, calendar, timedelta, kmer)"
    )


def _hash_buckets(corpus_events: int) -> int:
    """``buckets = clamp(round(corpus_events / 10_000), 256, 65_536)`` (Ground-1/B
    sizing rule, the "≈ corpus_events / 5K–50K" mid-divisor)."""
    raw = round(int(corpus_events) / 10_000) if corpus_events else 256
    return max(256, min(int(raw) or 256, 65_536))


def spec_from_field_map(field_map: dict, *, context_len: int = 4096) -> TokenizerSpec:
    """Build a :class:`TokenizerSpec` from a declarative FIELD-MAP (the SpecDraft).

    The field-map is a plain dict (parsed from the ``loom-fieldmap/1`` YAML the
    ``propose`` verb emits). Shape::

        entity: account_id          # EXCLUDED from the vocab (T2)
        event: txn
        target: is_fraud            # EXCLUDED (leakage)
        context_len: 4096
        fields:
          - {name: amt,  source: txn_amount, strategy: amount,   bins: 8}
          - {name: mcc,  source: mcc,        strategy: hash,      buckets: 4096}
          - {name: chan, source: channel,    strategy: mapping,   values: [...], default: UNK}
          - {name: drcr, source: dr_cr,      strategy: fixedvocab, min: 0, max: 1}
          - {name: ts,   source: txn_ts,     strategy: calendar}   # → HOUR+DOW+MONTH
          - {name: gap,  source: txn_ts,     strategy: timedelta, bins: 32}

    Each ``fields[]`` entry is instantiated as the matching existing
    :class:`Strategy` so the returned spec compiles through the LOCKED
    ``compile_spec`` + C1/C2/C3 **unchanged** — ARBITRARY columns now compile, and a
    bad field-map (e.g. two fields → colliding tokens) is REFUSED by C1 with the
    named diff (HARD INVARIANT #2). ``preset="custom"`` keeps the financial/chain
    dual-driver byte-identity untouched (HARD INVARIANT #1) while ``materialize``
    branches on it.

    HARD INVARIANT #4: the ``entity`` and any ``target`` column are NEVER emitted as
    field steps — the proposer excludes them upstream, and this compiler additionally
    refuses any ``fields[]`` entry whose ``source`` is the entity/target so a
    hand-edited field-map cannot smuggle an identity/leakage token back in.

    A ``calendar`` field with no ``part`` EXPANDS to three steps (HOUR, DOW, MONTH)
    sharing the timestamp source — the doc §2 calendar-token family; ``part: hour``
    emits just that one. MONTH is laid out 0-based (the MONTH_12≡CARD_0 fix)."""
    entity = field_map.get("entity")
    event = field_map.get("event")
    target = field_map.get("target")
    ctx_len = int(field_map.get("context_len", context_len))
    corpus_events = int(field_map.get("corpus_events", 0) or 0)

    excluded = {c for c in (entity, target) if c}

    steps: list[FieldStep] = []
    # Track the UPPER token prefixes already claimed so two fields can never emit
    # the same token strings (C1 injectivity). A schema with two timestamp columns
    # (created_at + updated_at) would otherwise emit two HOUR_* blocks that collide;
    # `_unique_prefix` disambiguates the second (HOUR, HOUR1, …) so disjoint blocks.
    seen_prefixes: set[str] = set()
    for entry in field_map.get("fields", ()):
        source = str(entry.get("source", entry.get("name", "")))
        if source in excluded or entry.get("name") in excluded:
            # T2 / leakage guard — never tokenize the entity or the target, even if
            # a hand-edited field-map lists it (HARD INVARIANT #4).
            raise ValueError(
                f"field {entry.get('name')!r} sources the "
                f"{'entity (T2)' if source == entity else 'target (leakage)'} "
                f"column {source!r}; it must NOT be tokenized — drop it from fields."
            )
        strat = str(entry["strategy"]).lower()
        if strat == "calendar":
            steps.extend(_calendar_steps(entry, source, seen_prefixes))
        else:
            steps.append(
                _field_step_from_entry(entry, corpus_events=corpus_events, seen_prefixes=seen_prefixes)
            )

    return TokenizerSpec(
        steps=tuple(steps),
        preset="custom",
        entity=entity,
        event=event,
        amount_strategy=AmountStrategy.FIXED,
    )


def _calendar_steps(entry: dict, source: str, seen_prefixes: set[str]) -> list[FieldStep]:
    """Expand a ``calendar`` field-map entry into FixedVocab calendar step(s).

    No ``part`` ⇒ the full HOUR(24)+DOW(7)+MONTH(12) family (doc §2). A single
    ``part`` (``hour``/``dow``/``month``) emits just that step. Each part is a
    FixedVocab whose preprocess column the field-map preprocess derives from the
    timestamp ``source``. The step name is ``"{name}_{part}"`` so two calendar
    fields over different timestamps never collide on a logical name; the TOKEN
    prefix is uniquified via ``_unique_prefix`` (``HOUR``→``HOUR1`` for the second
    timestamp column) so the two calendar blocks are DISJOINT and pass C1 — a
    multi-timestamp schema (created_at + updated_at) compiles out of the box."""
    base = str(entry.get("name", "ts"))
    part = entry.get("part")
    parts = [part.lower()] if part else ["hour", "dow", "month"]
    out: list[FieldStep] = []
    for p in parts:
        if p not in _CALENDAR_PARTS:
            raise ValueError(
                f"calendar field {base!r}: unknown part {p!r} (expected hour/dow/month)"
            )
        lo, hi, pad = _CALENDAR_PARTS[p]
        step_name = f"{base}_{p}"
        prefix = _unique_prefix(p.upper(), seen_prefixes)
        out.append(
            FieldStep(step_name, f"__cal__{source}__{p}", FixedVocab(prefix, lo, hi, pad))
        )
    return out


# ── The financial CPU preprocess (pandas; build brief constants) ────────────


def _amount_bin(amt: pd.Series) -> pd.Series:
    """``amt_bin = sum(amt >= t for t in [10,50,100,500,1000,5000])`` → 0..6."""
    f = (
        amt.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    f = pd.to_numeric(f, errors="coerce").fillna(0.0)
    bin_ = (
        (f >= 10).astype(int)
        + (f >= 50).astype(int)
        + (f >= 100).astype(int)
        + (f >= 500).astype(int)
        + (f >= 1000).astype(int)
        + (f >= 5000).astype(int)
    )
    return bin_.astype("int64")


def preprocess_financial(df: pd.DataFrame) -> pd.DataFrame:
    """Raw TabFormer-shaped frame → per-step columns the financial spec reads.

    Matches the conftest ``tabformer_df`` column naming (cust, card, amount, mcc,
    merchant, chip, zip, state, datetime) AND tolerates the reference TabFormer
    naming. Output columns: amt_val, merch, mcc_int, mcc_str, hour, dow, month,
    card, chip_upper, zip3, state_clean, cust, time_delta_s. CPU/pandas only.
    """
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_").lower() for c in df.columns]

    def col(*names: str, default=None) -> pd.Series:
        for n in names:
            if n in df.columns:
                return df[n]
        if default is not None:
            return pd.Series([default] * len(df), index=df.index)
        raise KeyError(f"none of {names} present in {list(df.columns)}")

    out = pd.DataFrame(index=df.index)

    # 1. amount
    out["amt_val"] = _amount_bin(col("amount", "amt"))

    # 2. merchant — uppercase, strip [^A-Z0-9\s-]; stable hash done in transform.
    merch = col("merchant", "merchant_name").astype(str).str.upper()
    out["merch"] = merch.map(lambda s: _MERCH_CLEAN_RE.sub("", s))

    # 3-4. mcc int + str
    mcc = pd.to_numeric(col("mcc"), errors="coerce").fillna(-1).astype("int64")
    out["mcc_int"] = mcc
    out["mcc_str"] = mcc.astype(str)

    # 5-7. temporal: hour / dow / month
    dt = _parse_datetime(df)
    out["hour"] = dt.dt.hour
    out["dow"] = dt.dt.dayofweek
    out["month"] = dt.dt.month

    # 8. card 0..9
    out["card"] = pd.to_numeric(col("card"), errors="coerce").fillna(0).astype("int64").clip(0, 9)

    # 9. chip — upper-cased raw mapping keys ("Swipe Transaction" → "SWIPE TRANSACTION")
    out["chip_upper"] = col("chip", "use_chip").astype(str).str.upper()

    # 10. zip3 — first 3 digits 0..999
    zip_col = col("zip", "merchant_zip", default="00000").astype(str).str.replace(".0", "", regex=False)
    out["zip3"] = (
        zip_col.str[:3].str.zfill(3).str.replace(r"\D", "0", regex=True).astype("int64").clip(0, 999)
    )

    # 11. state — upper/strip mapped
    state = col("state", "merchant_state", default="XX").astype(str).str.upper().str.strip()
    out["state_clean"] = state.where(state != "", "XX")

    # 12. cust 0..2999
    out["cust"] = pd.to_numeric(col("cust", "user"), errors="coerce").fillna(0).astype("int64").clip(0, 2999)

    # optional time_delta (seconds), grouped by [cust, card], sorted by time.
    tmp = out.copy()
    tmp["_dt"] = dt
    tmp = tmp.sort_values(["cust", "card", "_dt"], kind="stable")
    gap = tmp.groupby(["cust", "card"])["_dt"].diff().dt.total_seconds().fillna(0).clip(lower=0)
    out["time_delta_s"] = gap.reindex(out.index).fillna(0)

    return out.reset_index(drop=True)


def _parse_datetime(df: pd.DataFrame) -> pd.Series:
    """Parse a datetime from either a single ``datetime`` column or the reference
    Year/Month/Day/Time quad."""
    if "datetime" in df.columns:
        return pd.to_datetime(df["datetime"], errors="coerce")
    if {"year", "month", "day"} <= set(df.columns):
        date_str = (
            df["year"].astype(str)
            + "-"
            + df["month"].astype(str).str.zfill(2)
            + "-"
            + df["day"].astype(str).str.zfill(2)
            + " "
            + df.get("time", pd.Series(["00:00"] * len(df))).fillna("00:00").astype(str)
        )
        return pd.to_datetime(date_str, errors="coerce")
    # Fallback: a single epoch-zero so downstream bins are deterministic.
    return pd.to_datetime(pd.Series([0] * len(df)), unit="s")


def preprocess_chain(df: pd.DataFrame) -> pd.DataFrame:
    """Raw DEX trade frame → per-step columns the chain spec reads (CPU/pandas).

    Sorts by [wallet, timestamp] (C6), derives the inter-trade gap per wallet,
    and emits venue/side/item/size_usd/gap_s/hour/dow. The ``wallet`` column is
    carried for grouping but NEVER tokenized (T2)."""
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_").lower() for c in df.columns]
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["_ts"] = ts
    df = df.sort_values(["wallet", "_ts"], kind="stable").reset_index(drop=True)
    ts = df["_ts"]

    out = pd.DataFrame(index=df.index)
    out["wallet"] = df["wallet"].astype(str)
    # venue → 0-based index into DEX_VENUES (unknowns clip to 0).
    venue_index = {v: i for i, v in enumerate(DEX_VENUES)}
    out["venue"] = df["venue"].astype(str).str.upper().map(venue_index).fillna(0).astype("int64")
    out["side"] = df["side"].astype(str).str.upper()
    out["item"] = df["item"].astype(str).str.upper()
    out["size_usd"] = pd.to_numeric(df["size_usd"], errors="coerce").fillna(0.0).clip(lower=0)
    gap = df.groupby("wallet")["_ts"].diff().dt.total_seconds().fillna(0).clip(lower=0)
    out["gap_s"] = gap.values
    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek
    return out


def _amount_bin_thresholds(amt: pd.Series, thresholds: list[float]) -> pd.Series:
    """``bin = sum(amt >= t for t in thresholds)`` → 0..len(thresholds) (generic).

    The bring-your-own-schema analogue of :func:`_amount_bin`: log-spaced
    deterministic threshold bins (C2-clean, no fitted artifact). Tolerates a
    ``"$1,234.50"``-style string column the same way the financial preprocess does."""
    f = (
        amt.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    f = pd.to_numeric(f, errors="coerce").fillna(0.0)
    bin_ = pd.Series(0, index=amt.index, dtype="int64")
    for t in thresholds:
        bin_ = bin_ + (f >= t).astype("int64")
    return bin_


def _is_kmer_spec(spec: TokenizerSpec) -> bool:
    """True iff the spec is a SEQUENCE spec (any step sources a ``__kmer__`` recipe).

    A k-mer field tokenizes a whole sequence STRING into a per-position token
    sequence, so its preprocess EXPLODES one input row into many — fundamentally a
    different row cardinality than the tabular per-row fields. We therefore route a
    sequence spec to :func:`_preprocess_kmer`; the financial/chain/tabular-custom
    paths are byte-for-byte unchanged (HARD INVARIANT #1)."""
    return any(s.source.startswith("__kmer__") for s in spec.steps)


def _parse_kmer_source(src: str) -> tuple[str, int, int]:
    """``__kmer__<col>__<k>__<stride>`` → (col, k, stride). The column name may
    itself contain ``__``; k and stride are the last two ``__``-delimited fields."""
    body = src[len("__kmer__"):]
    rest, _, stride_s = body.rpartition("__")
    col, _, k_s = rest.rpartition("__")
    return col, int(k_s), int(stride_s)


# The synthetic per-sequence grouping key the k-mer preprocess emits — each input
# (sequence) row becomes its OWN corpus line. Private name so it can't clash with a
# user field/entity column (same discipline as the ``__group__`` carry in
# ``materialize_corpus_lines``).
KMER_SEQ_COL = "__seq__"


def _preprocess_kmer(df: pd.DataFrame, spec: TokenizerSpec) -> pd.DataFrame:
    """Explode each sequence row into one row per k-mer position (the sequence path).

    "One input sequence → many tokens": for every k-mer step, slice its sequence
    column into ordered length-``k`` windows (at the step's stride) and emit one
    output row per window, carrying the recipe-source column = the k-mer STRING for
    that position and a ``__seq__`` id = the originating input-row index (so every
    sequence becomes ONE corpus line, in position order). The per-step ``transform``
    then maps each window 1:1 to its ``KMER_<window>`` token — so the existing corpus
    grammar materializes ``<bos> KMER_x <sep> KMER_y … <eos>`` with no new machinery.

    All k-mer steps in one spec must share the SAME windowing (they tokenize the same
    sequence positions); the first step defines the row layout and the rest align to
    it. Pure pandas, CPU-only, config-only (no fitting). A row that yields no windows
    (sequence shorter than k) contributes no positions (and thus no line)."""
    kmer_steps = [s for s in spec.steps if s.source.startswith("__kmer__")]

    seq_ids: list[int] = []
    # One materialized column per step source (the k-mer string at each position).
    cols: dict[str, list[str]] = {s.source: [] for s in kmer_steps}
    # Carry the entity value (if a real entity column exists) per exploded position
    # so a caller that groups by a real entity still works; otherwise __seq__ groups.
    entity = spec.entity
    carry_entity = bool(entity and entity in df.columns)
    entity_vals: list = []

    for row_pos in range(len(df)):
        # Each step may slice a different sequence column / k / stride, but they must
        # agree on the NUMBER of positions to stay row-aligned. We window each step's
        # column and require equal length; the windows of the first step set the count.
        per_step_windows: dict[str, list[str]] = {}
        n_positions = None
        for step in kmer_steps:
            strat = step.strategy  # KMer
            col, k, stride = _parse_kmer_source(step.source)
            raw = df.iloc[row_pos][col] if col in df.columns else ""
            obs = set(strat.observed) if strat.observed is not None else None
            windows = split_kmers(raw, k, stride, set(strat.alphabet), observed=obs)
            per_step_windows[step.source] = windows
            if n_positions is None:
                n_positions = len(windows)
            else:
                n_positions = min(n_positions, len(windows))
        n_positions = n_positions or 0
        for pos in range(n_positions):
            seq_ids.append(row_pos)
            if carry_entity:
                entity_vals.append(df.iloc[row_pos][entity])
            for src, windows in per_step_windows.items():
                cols[src].append(windows[pos])

    out = pd.DataFrame({src: pd.Series(vals, dtype=object) for src, vals in cols.items()})
    out[KMER_SEQ_COL] = pd.Series(seq_ids, dtype="int64")
    if carry_entity:
        out[entity] = pd.Series(entity_vals)
    return out.reset_index(drop=True)


def preprocess_field_map(df: pd.DataFrame, spec: TokenizerSpec) -> pd.DataFrame:
    """Generic CPU preprocess for a custom field-map spec (the ``preset=="custom"``
    path). Pure pandas, CPU-only, no fitted state.

    Every step's ``source`` is honored generically:

      * a recipe source ``__amt__<col>__<bins>`` → log-threshold bin index 0..bins-1;
      * a recipe source ``__cal__<col>__<part>`` → the calendar part (hour/dow, or
        MONTH shifted to 0-based so MONTH never collides with a 0-based block);
      * a recipe source ``__gap__<col>`` → the per-ENTITY inter-event gap in seconds
        (sorted by ``[entity, timestamp]`` — the C6 discipline);
      * any other source → the RAW column passed through verbatim (FixedVocab on a
        bounded int, Mapping, Hash — the strategy's own ``transform`` does the rest).

    The entity column is carried under its raw name for grouping but is NEVER a
    field step (the proposer + ``spec_from_field_map`` exclude it — T2).

    A SEQUENCE spec (any ``kmer`` step) is routed to :func:`_preprocess_kmer`, which
    EXPLODES one sequence row into one row per k-mer position (one sequence → many
    tokens) — a different row cardinality than the tabular branch below, which is
    byte-for-byte unchanged."""
    if _is_kmer_spec(spec):
        return _preprocess_kmer(df, spec)

    out = pd.DataFrame(index=df.index)
    entity = spec.entity

    # Parse the timestamp column(s) once per distinct underlying source.
    ts_cache: dict[str, pd.Series] = {}

    def _ts(col: str) -> pd.Series:
        if col not in ts_cache:
            ts_cache[col] = pd.to_datetime(df[col], errors="coerce") if col in df.columns else pd.to_datetime(
                pd.Series([0] * len(df), index=df.index), unit="s"
            )
        return ts_cache[col]

    for step in spec.steps:
        src = step.source
        if src.startswith("__amt__"):
            body = src[len("__amt__"):]
            col, _, bins_s = body.rpartition("__")
            bins = int(bins_s)
            raw = df[col] if col in df.columns else pd.Series([0.0] * len(df), index=df.index)
            out[src] = _amount_bin_thresholds(raw, _log_thresholds(bins))
        elif src.startswith("__cal__"):
            body = src[len("__cal__"):]
            col, _, part = body.rpartition("__")
            dt = _ts(col)
            if part == "hour":
                out[src] = dt.dt.hour.fillna(0).astype("int64")
            elif part == "dow":
                out[src] = dt.dt.dayofweek.fillna(0).astype("int64")
            elif part == "month":
                # 0-based month (1..12 → 0..11): the MONTH_12≡CARD_0 collision fix.
                out[src] = (dt.dt.month.fillna(1).astype("int64") - 1).clip(0, 11)
            else:  # pragma: no cover - guarded in _calendar_steps
                out[src] = 0
        elif src.startswith("__gap__"):
            col = src[len("__gap__"):]
            dt = _ts(col)
            tmp = pd.DataFrame({"_dt": dt})
            if entity and entity in df.columns:
                tmp["_grp"] = df[entity].values
                tmp = tmp.sort_values(["_grp", "_dt"], kind="stable")
                gap = tmp.groupby("_grp")["_dt"].diff().dt.total_seconds()
            else:
                tmp = tmp.sort_values(["_dt"], kind="stable")
                gap = tmp["_dt"].diff().dt.total_seconds()
            out[src] = gap.reindex(out.index).fillna(0).clip(lower=0)
        else:
            # Plain pass-through: the strategy's transform handles dtype coercion.
            out[src] = df[src] if src in df.columns else pd.Series([None] * len(df), index=df.index)

    # Carry the grouping entity (never a field step) under its raw name.
    if entity and entity in df.columns and entity not in out.columns:
        out[entity] = df[entity].values
    return out


def materialize_corpus_lines(compiled, df: pd.DataFrame) -> tuple[list[str], int]:
    """Compile a raw frame into corpus lines for a :class:`CompiledTokenizer`.

    The high-level corpus assembler (DESIGN.md §0 C3): preprocess the rows for the
    compiled spec's preset, emit each step's token-string column, group by the
    entity, and slice into ``chunk_size`` windows as
    ``<bos> txn (<sep> txn)* <eos>`` lines. Returns ``(lines, n_txns)``.

    This is the public ``(compiled, df) -> lines`` entrypoint both the ``tokenize``
    verb and the conformance tests use; the low-level
    :func:`loom.engine.contracts.to_corpus_lines` (token-df in) is the inner step.
    """
    from . import strategies
    from .contracts import to_corpus_lines

    spec = compiled.spec
    if spec.preset == "chain":
        pre = preprocess_chain(df)
        group_source = "wallet"
    elif spec.preset == "custom":
        # Bring-your-own-schema: the generic field-map preprocess; the grouping key
        # is the declared entity (never a field step — T2). The financial/chain
        # branches below are byte-for-byte UNCHANGED (HARD INVARIANT #1).
        pre = preprocess_field_map(df, spec)
        # A SEQUENCE (k-mer) spec groups by the synthetic per-sequence id the explode
        # emitted — each input sequence row → ONE corpus line of its position tokens.
        if _is_kmer_spec(spec) and KMER_SEQ_COL in pre.columns:
            group_source = KMER_SEQ_COL
        else:
            group_source = spec.entity or "__none__"
    else:
        pre = preprocess_financial(df)
        group_source = "cust"

    token_cols: dict[str, pd.Series] = {}
    for step in spec.steps:
        if step.source in pre.columns:
            token_cols[step.name] = strategies.transform(step.strategy, pre[step.source])
    token_df = pd.DataFrame(token_cols)
    field_cols = [s.name for s in spec.steps if s.name in token_df.columns]

    # Carry the grouping key under a PRIVATE column name so it can never clobber a
    # field-token column whose step name collides with the entity (e.g. the CUST
    # field step is named "cust" and the financial entity is also "cust"; grouping
    # by the raw integer must NOT overwrite the tokenized CUST_* column).
    if group_source in pre.columns:
        group_col = "__group__"
        token_df[group_col] = pre[group_source].values
        lines = to_corpus_lines(token_df, [group_col], compiled.chunk_size, field_cols=field_cols)
    else:
        lines = to_corpus_lines(token_df, [], compiled.chunk_size, field_cols=field_cols)
    return lines, int(len(token_df))


def corpus_lines(compiled, df: pd.DataFrame) -> list[str]:
    """The public ``(compiled, df) -> list[str]`` corpus-line helper (DESIGN.md §0
    C3). Returns ONLY the corpus-line strings (the ``materialize_corpus_lines``
    workhorse also returns the transaction count for the verb's bookkeeping)."""
    lines, _ = materialize_corpus_lines(compiled, df)
    return lines


__all__ = [
    "financial_spec",
    "chain_spec",
    "spec_from_field_map",
    "preprocess_financial",
    "preprocess_chain",
    "preprocess_field_map",
    "materialize_corpus_lines",
    "corpus_lines",
    "AMOUNT_THRESHOLDS",
    "INDUSTRY_RANGES",
    "KNOWN_MCCS",
    "CHIP_MAPPING",
    "ALL_STATES",
    "DEX_VENUES",
    "CONT_BINS_DEFAULT",
    "TIMEDELTA_BINS_DEFAULT",
    "DNA_ALPHABET",
    "KMER_SEQ_COL",
]
