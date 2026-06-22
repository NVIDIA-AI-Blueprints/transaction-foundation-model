"""Shared pytest fixtures — tiny synthetic TabFormer + DEX sample frames.

These run on CPU in milliseconds with no network/GPU/cuDF. They exercise every
field family the financial (TabFormer) and chain (DEX) presets read, so the
tokenize/ingest/baseline implementers can assert vocab/grammar/EDA behavior
against deterministic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tabformer_df() -> pd.DataFrame:
    """A tiny synthetic TabFormer-shaped frame covering the 12 financial fields.

    Column names mirror the raw fields the financial preprocess reads (amount as a
    ``"$x"`` string, MCC as int, dates as parseable strings, chip as the raw
    upper-case mapping keys, etc.). Deterministic — no randomness."""
    rows = [
        # user, card, amount,    mcc,  merchant,        chip,                  zip,    state, dt
        (0, 0, "$12.50",  5411, "WHOLE FOODS #123",  "Swipe Transaction",  "94107", "CA", "2026-01-02 09:15:00"),
        (0, 1, "$4.00",   5814, "BLUE BOTTLE",       "Chip Transaction",   "94110", "CA", "2026-01-02 13:40:00"),
        (1, 0, "$1500.00",4111, "BART",              "Online Transaction", "94612", "CA", "2026-02-14 18:05:00"),
        (1, 0, "$0.99",   5942, "AMAZON",            "Online Transaction", "10001", "NY", "2026-03-30 23:59:00"),
        (2, 2, "$87.20",  5912, "CVS PHARMACY",      "Swipe Transaction",  "60601", "IL", "2026-12-25 07:30:00"),
    ]
    df = pd.DataFrame(
        rows,
        columns=["cust", "card", "amount", "mcc", "merchant", "chip", "zip", "state", "datetime"],
    )
    return df


@pytest.fixture
def dex_df() -> pd.DataFrame:
    """A tiny synthetic DEX trade frame covering the chain preset's fields.

    Sorted-ready by [wallet, timestamp] (C6). Covers venue/side/item/size/gap/
    hour/dow. The ``wallet`` column is the GROUPING entity, never tokenized (T2)."""
    rows = [
        # wallet, ts,                  venue,       side,  item,    size_usd
        ("0xa1", "2026-06-01 00:00:00", "DEXETH",   "BUY",  "WETH",  120.0),
        ("0xa1", "2026-06-01 00:05:00", "DEXETH",   "SELL", "WETH",  118.5),
        ("0xa1", "2026-06-02 12:00:00", "DEXBASE",  "BUY",  "USDC",  5000.0),
        ("0xb2", "2026-06-01 09:00:00", "DEXSOL",   "BUY",  "SOL",   42.0),
        ("0xb2", "2026-06-03 21:30:00", "DEXSOL",   "SELL", "SOL",   40.0),
    ]
    df = pd.DataFrame(
        rows,
        columns=["wallet", "timestamp", "venue", "side", "item", "size_usd"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture
def leaky_df() -> pd.DataFrame:
    """A frame with an obvious identity-like leakage column for the EDA scan
    (``user_id`` is unique per row ⇒ should be flagged)."""
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(8)],   # near-unique identity → flag
            "amount": np.arange(8, dtype=float),
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
