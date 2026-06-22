"""Shared leave-one-last-out temporal hold-out (C6) — the eval split `baseline`
and `evaluate` MUST agree on byte-for-byte.

This module is a **pure extraction** of the column-inference + temporal split that
used to live inline in :mod:`loom.verbs.baseline`. It is NOT a fork: the semantics
are identical — same column candidate tuples, the same
``sort_values([entity, time], kind="mergesort")`` with a time-None stable-index
fallback, the same ``groupby(entity, sort=True)`` ≥2-event filter (target =
``iloc[-1]``, prev = ``iloc[-2]``), the same history =
``work.drop(groupby(sort=False).tail(1).index)``, the same ``round(..., 6)`` and
the same ``$1,234.50`` amount coercion. ``baseline`` and ``evaluate`` both import
these so the strong repeat-last-item control is computed exactly once, one way.

CPU-only, deterministic, no GPU/torch/NeMo imports — engine code is contract logic.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

# Column-name candidates used to infer the entity / time / item / side / amount
# columns when they aren't pinned on the ingested object's extras.
_ENTITY_CANDS = ("wallet", "cust", "customer", "user", "account", "entity")
_TIME_CANDS = ("timestamp", "datetime", "ts", "time", "date", "event_time")
_ITEM_CANDS = ("item", "mcc", "merchant", "venue", "product", "token", "symbol")
_SIDE_CANDS = ("side", "direction", "action")
_AMOUNT_CANDS = ("size_usd", "amount", "amt", "size", "value", "notional")


def _infer_col(
    df: pd.DataFrame, pinned: Optional[str], candidates: tuple[str, ...]
) -> Optional[str]:
    if pinned and pinned in df.columns:
        return pinned
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def infer_cols(df: pd.DataFrame, extras: dict[str, Any]) -> dict[str, Optional[str]]:
    """Resolve the entity/time/item/side/amount column names for ``df``.

    ``extras`` is the ingested object's ``extras`` dict — its ``entity``/``event``
    pins take precedence over the candidate tuples (same precedence baseline used)."""
    extras = extras or {}
    return {
        "entity": _infer_col(df, extras.get("entity"), _ENTITY_CANDS),
        "time": _infer_col(df, None, _TIME_CANDS),
        "item": _infer_col(df, extras.get("event"), _ITEM_CANDS),
        "side": _infer_col(df, None, _SIDE_CANDS),
        "amount": _infer_col(df, None, _AMOUNT_CANDS),
    }


def load_rows(obj: Any) -> Optional[pd.DataFrame]:
    """Load the dataframe payload for an ingested object, if present.

    ``ingest`` writes CSV (Phase-0 dep set has no parquet engine), so CSV is tried
    first; parquet is a fallback for any object written by a future v0.2 path.
    """
    payload_path = getattr(obj, "payload_path", None)
    if not payload_path:
        return None
    try:
        return pd.read_csv(payload_path)
    except Exception:  # noqa: BLE001 — fall back to parquet if that's the payload
        try:
            return pd.read_parquet(payload_path)
        except Exception:  # noqa: BLE001
            return None


def prec_at_k(predicted_topk: list[Any], actual: Any) -> float:
    return 1.0 if actual in predicted_topk else 0.0


def coerce_amount(value: Any) -> Optional[float]:
    """Best-effort numeric coercion for an amount/size cell.

    Raw TabFormer amounts arrive as ``"$x"`` / ``"$1,234.50"`` strings; strip the
    currency formatting and parse a float. Returns ``None`` for missing or
    non-numeric values so the MAE baseline skips that pair rather than crashing
    (the previous ``float("$22.00")`` raised ``ValueError`` mid-verb)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace("$", "").replace(",", "")
    if s == "" or s.lower() in ("nan", "none"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def build_leave_one_last_out(
    work: pd.DataFrame, cols: dict[str, Optional[str]]
) -> tuple[list[dict[str, Any]], pd.DataFrame, int]:
    """Build the temporal leave-one-last-out eval targets + the history frame.

    ``work`` is the raw rows; ``cols`` is the :func:`infer_cols` mapping. Performs
    the deterministic per-entity ordering (by time when present, else stable index,
    C6), then for each entity with ≥2 events emits a target (last event) carrying
    its previous item/side/amount. ``history`` drops each entity's last row (so a
    popularity control fit on it never leaks the held-out target).

    Returns ``(targets, history, n_eval)``. This is the EXACT split baseline used —
    same mergesort ordering, same ``groupby(sort=True)`` over targets, same
    ``groupby(sort=False).tail(1)`` for the history drop.
    """
    entity_col = cols["entity"]
    time_col = cols["time"]
    item_col = cols["item"]
    side_col = cols["side"]
    amount_col = cols["amount"]

    work = work.copy()
    # Deterministic per-entity ordering: by time when present, else stable index (C6).
    if time_col is not None:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.sort_values([entity_col, time_col], kind="mergesort")
    else:
        work = work.sort_values([entity_col], kind="mergesort")
    work = work.reset_index(drop=True)

    # For each entity with >=2 events: history = all but the last; target = last.
    targets: list[dict[str, Any]] = []
    for ent, grp in work.groupby(entity_col, sort=True):
        if len(grp) < 2:
            continue
        last = grp.iloc[-1]
        prev = grp.iloc[-2]
        targets.append(
            {
                "entity": ent,
                "actual_item": last[item_col],
                "prev_item": prev[item_col],
                "actual_side": last[side_col] if side_col else None,
                "actual_amount": last[amount_col] if amount_col else None,
                "prev_amount": prev[amount_col] if amount_col else None,
            }
        )

    n_eval = len(targets)

    # History excludes each entity's held-out last event (avoids leaking the target
    # into a popularity/majority control fit on it).
    last_idx = work.groupby(entity_col, sort=False).tail(1).index
    history = work.drop(index=last_idx)

    return targets, history, n_eval
