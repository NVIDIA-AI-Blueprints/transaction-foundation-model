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
    "preprocess_financial",
    "preprocess_chain",
    "materialize_corpus_lines",
    "corpus_lines",
    "AMOUNT_THRESHOLDS",
    "INDUSTRY_RANGES",
    "KNOWN_MCCS",
    "CHIP_MAPPING",
    "ALL_STATES",
    "DEX_VENUES",
]
