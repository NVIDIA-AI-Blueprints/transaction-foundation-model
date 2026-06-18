"""``baseline`` — popularity + repeat-last-item baselines (DESIGN.md Phase-0 Step 0.6).

Computes the classical controls a model must beat (house rule #3 / §3.3) over an
ingested dataset's eval slice, on CPU/pandas, and persists a ``Baseline``
:class:`~loom.store.DataObject` referenced by experiment id (the join key, §7.7).

Baselines computed:
  * **popularity** — predict the globally most-frequent next item; scored as
    Prec@K over each entity's held-out last event.
  * **repeat-last-item** — predict the entity's own previous item; the strong
    sequential control (D5: next-trade Prec@5 0.31 vs repeat-last 0.22, §3.3).
  * **next-side majority** — when a ``side`` (BUY/SELL-like) column exists,
    accuracy of always predicting the majority side.
  * **next-amount last-value** — when an amount/size column exists, the
    last-value carry-forward error (MAE) for the held-out event.

Eval slice (C6): a **temporal** hold-out of each entity's last event (leave-one-
last-out), which is the cheap per-entity analogue of the temporal/entity-disjoint
split a real evaluate uses. Only entities with ≥2 events contribute a held-out
target. LOCKED params (``BASELINE_PARAMS``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from ..engine.holdout import (
    build_leave_one_last_out,
    coerce_amount,
    infer_cols,
    load_rows,
    prec_at_k,
)
from ..registry import VerbContext, register
from ..types import (
    CapabilityMode,
    CostPlan,
    Diagnostic,
    Severity,
    Status,
    Tier,
    Verdict,
    VerbResult,
)

BASELINE_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "in": {"type": "string", "description": "input Corpus/<n> or IngestDataset/<n> pathspec"},
        "task": {"type": "string", "description": "task: next-item | fraud-auprc"},
        "k": {"type": "integer", "description": "Prec@K cutoff for next-item baselines"},
        "kind": {"type": "string", "enum": ["popularity", "repeat-last-item", "both"],
                 "description": "which baseline(s) to compute (default both)"},
        "eval_split": {"type": "string", "description": "temporal | entity-disjoint (C6)"},
        "confirm_token": {"type": "string", "description": "agent second-call confirm token (§5.3)"},
    },
}

# Column inference, row loading, the leave-one-last-out split, Prec@K and amount
# coercion are SHARED with ``evaluate`` via :mod:`loom.engine.holdout` — the eval
# split the model must beat is computed exactly once, one way (no fork).


def _refused(msg: str, fix: str, experiment: Optional[str]) -> VerbResult:
    return VerbResult(
        verb="baseline",
        status=Status.REFUSED_CONTRACT,
        verdict=Verdict.FAIL,
        tier=Tier.WORKSPACE_WRITE,
        capability_mode=CapabilityMode.SEARCHABLE,
        summary=msg,
        diagnostics=[Diagnostic(contract="C6", severity=Severity.ERROR, message=msg, fix=fix)],
        experiment=experiment,
        cost_plan=CostPlan(),
    )


@register(
    "baseline",
    summary="compute popularity + repeat-last-item baselines (the control a model must beat)",
    tier=Tier.WORKSPACE_WRITE,
    capability_mode=CapabilityMode.SEARCHABLE,
    params=BASELINE_PARAMS,
)
def _baseline(args: dict[str, Any], ctx: VerbContext) -> VerbResult:
    from ..store import DataObject  # local import: store is the v0.2 seam

    in_spec = args.get("in") or ""
    k = int(args.get("k") or 5)
    kind = args.get("kind") or "both"
    eval_split = args.get("eval_split") or "temporal"
    experiment = ctx.experiment

    if not in_spec:
        return _refused(
            "baseline needs an input dataset (Corpus/<n> or IngestDataset/<n>)",
            "pass `loom baseline IngestDataset/<n> --experiment <id>`",
            experiment,
        )

    # --- resolve the input object + its rows ----------------------------------
    try:
        obj = ctx.store.get(in_spec)
    except (KeyError, ValueError) as exc:
        return _refused(
            f"baseline could not resolve input {in_spec!r}: {exc}",
            "ingest the dataset first; baseline reads an IngestDataset/<n>",
            experiment,
        )

    df = load_rows(obj)
    if df is None or len(df) == 0:
        return _refused(
            f"baseline found no rows on {in_spec} (no readable payload)",
            "re-ingest the source so the rows payload is persisted",
            experiment,
        )

    extras = getattr(obj, "extras", {}) or {}
    cols = infer_cols(df, extras)
    entity_col = cols["entity"]
    time_col = cols["time"]
    item_col = cols["item"]
    side_col = cols["side"]
    amount_col = cols["amount"]

    if entity_col is None or item_col is None:
        return _refused(
            f"baseline needs an entity and an item column "
            f"(entity={entity_col!r}, item={item_col!r} on {in_spec})",
            "pin --entity at ingest time and ensure an item-like column exists",
            experiment,
        )

    # --- build leave-one-last-out eval targets (temporal hold-out, C6) --------
    # SHARED with ``evaluate`` (loom.engine.holdout) so the control the model must
    # beat is computed exactly once: per-entity time/stable ordering, ≥2-event
    # filter, target=last/prev=second-last, history drops each entity's last event.
    targets, history, n_eval = build_leave_one_last_out(df, cols)
    if n_eval == 0:
        return _refused(
            f"baseline has no held-out targets on {in_spec}: every entity has <2 events",
            "ingest a dataset with multi-event entities (sequences), or widen the slice",
            experiment,
        )

    # Global popularity ranking is fit on the HISTORY only (exclude held-out last
    # events) to avoid leaking the target into the popularity control.
    pop_counts = history[item_col].value_counts()
    pop_topk = list(pop_counts.head(k).index)

    metrics: dict[str, Any] = {}

    # --- popularity: Prec@K of the global top-K -------------------------------
    if kind in ("popularity", "both"):
        hits = sum(prec_at_k(pop_topk, t["actual_item"]) for t in targets)
        metrics["popularity"] = {
            "metric": f"prec@{k}",
            "value": round(hits / n_eval, 6),
            "topk": [str(x) for x in pop_topk],
        }

    # --- repeat-last-item: predict the entity's previous item -----------------
    if kind in ("repeat-last-item", "both"):
        hits = sum(1.0 for t in targets if t["actual_item"] == t["prev_item"])
        metrics["repeat-last-item"] = {
            "metric": "prec@1",
            "value": round(hits / n_eval, 6),
        }

    # --- next-side majority (when a side column exists) -----------------------
    if side_col is not None:
        side_hist = history[side_col].dropna()
        if len(side_hist):
            majority_side = side_hist.value_counts().idxmax()
            side_hits = sum(1.0 for t in targets if t["actual_side"] == majority_side)
            metrics["next-side-majority"] = {
                "metric": "accuracy",
                "value": round(side_hits / n_eval, 6),
                "majority_side": str(majority_side),
            }

    # --- next-amount last-value carry-forward (when an amount column exists) ---
    if amount_col is not None:
        errs: list[float] = []
        for t in targets:
            a = coerce_amount(t["actual_amount"])
            p = coerce_amount(t["prev_amount"])
            if a is not None and p is not None:
                errs.append(abs(a - p))
        if errs:
            metrics["next-amount-last-value"] = {
                "metric": "mae",
                "value": round(float(np.mean(errs)), 6),
                "n": len(errs),
            }

    # --- persist a Baseline object referenced by the experiment ----------------
    ref = ctx.store.new_ref("Baseline")
    data_block = {
        "input": obj.pathspec,
        "entity_col": entity_col,
        "item_col": item_col,
        "side_col": side_col,
        "amount_col": amount_col,
        "time_col": time_col,
        "eval_split": eval_split,
        "n_entities_eval": n_eval,
        "n_rows": int(len(df)),
        "k": k,
        "metrics": metrics,
    }
    bobj = DataObject(
        ref=ref,
        kind="Baseline",
        content_id=f"{obj.content_id}:baseline:{kind}:{eval_split}:k{k}",
        parents=[obj.pathspec],
        producer_verb="baseline",
        producer_args={"in": in_spec, "kind": kind, "k": k, "eval_split": eval_split},
        signatures={"metrics": metrics, "eval_split": eval_split},
        verdict=Verdict.PASS,
        status=Status.OK,
        experiment=experiment,
        created_at=datetime.now(timezone.utc).isoformat(),
        extras={"baseline": data_block},
    )
    stored = ctx.store.put(bobj)

    # --- a readable one-line summary, leading with the strongest control -------
    parts: list[str] = []
    if "repeat-last-item" in metrics:
        parts.append(f"repeat-last-item prec@1={metrics['repeat-last-item']['value']}")
    if "popularity" in metrics:
        parts.append(f"popularity prec@{k}={metrics['popularity']['value']}")
    if "next-side-majority" in metrics:
        parts.append(f"next-side-majority acc={metrics['next-side-majority']['value']}")
    if "next-amount-last-value" in metrics:
        parts.append(f"next-amount MAE={metrics['next-amount-last-value']['value']}")
    summary = f"{stored.pathspec}  " + "  ".join(parts) + f"  (n={n_eval}, split={eval_split})"

    return VerbResult(
        verb="baseline",
        status=Status.OK,
        verdict=Verdict.PASS,
        tier=Tier.WORKSPACE_WRITE,
        capability_mode=CapabilityMode.SEARCHABLE,
        summary=summary,
        outputs=[stored.ref],
        diagnostics=[],
        data={"pathspec": stored.pathspec, **data_block},
        experiment=experiment,
        cost_plan=CostPlan(),
    )
