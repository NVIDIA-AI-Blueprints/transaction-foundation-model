"""EDA leakage gate (DESIGN.md §2.2 ingest, §5.1 step 3, item #2).

``ingest`` runs a schema sniff + an EDA leakage scan that flags identity-like
columns (the ``CUST_*`` family) and emits a VERDICT (PASS / REVIEW). The scan
returns a list of :class:`~loom.types.Diagnostic` cards (contract ``"EDA"``) that
travel on the ingest result envelope and the object's ``ingest_report``.

The heuristics are deliberately conservative and *advisory*: a flagged column is
surfaced as a REVIEW (WARNING severity), never a hard FAIL. The point is to make
the human/agent state their reasoning ("yes, ``cust`` is the grouping entity, not
a feature") rather than to silently ship a leaky report (house rule, §5.1).
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from .types import Diagnostic, Severity

# Column-name shapes that read as a per-entity / per-row identity. Matched
# case-insensitively against the *whole* name and common token boundaries.
_IDENTITY_NAME_RE = re.compile(
    r"(?:^|[_\W])(id|uuid|guid|user|cust|customer|wallet|account|acct|"
    r"address|addr|hash|email|ssn|pan|card_?number|primary_?key|pk)(?:$|[_\W])",
    re.IGNORECASE,
)

# Fraction of distinct values above which a column looks like a (near-)unique key.
_NEAR_UNIQUE_RATIO = 0.98
# |corr| above which a numeric feature is suspiciously predictive of the target.
_HIGH_CORR = 0.95


def _name_looks_like_identity(name: str) -> bool:
    return bool(_IDENTITY_NAME_RE.search(str(name)))


def leakage_scan(df: pd.DataFrame, target: Optional[str] = None) -> list[Diagnostic]:
    """Scan a dataframe for leakage / identity-like columns.

    Flags columns that look like per-entity identities (near-unique, id-shaped, or
    matching the entity key) as ``EDA`` warnings — these leak the label if used as
    features and drive the REVIEW verdict. When ``target`` is given, also flags
    columns suspiciously correlated with / derivable from it.

    Returns a list of named :class:`Diagnostic` cards (empty ⇒ clean ⇒ PASS).
    """
    diags: list[Diagnostic] = []
    n_rows = len(df)
    if n_rows == 0:
        return diags

    target_series = None
    if target is not None and target in df.columns:
        target_series = df[target]

    for col in df.columns:
        if col == target:
            continue  # the label itself is not "leakage"
        series = df[col]
        non_null = series.dropna()
        n_non_null = len(non_null)
        if n_non_null == 0:
            continue
        n_unique = int(non_null.nunique())
        unique_ratio = n_unique / n_non_null
        name_hit = _name_looks_like_identity(col)
        near_unique = unique_ratio >= _NEAR_UNIQUE_RATIO

        # --- identity / near-unique key -------------------------------------
        if name_hit or near_unique:
            if name_hit and near_unique:
                why = "id-shaped name AND near-unique values"
            elif name_hit:
                why = "id-shaped column name"
            else:
                why = "near-unique values (looks like a row/entity key)"
            diags.append(
                Diagnostic(
                    contract="EDA",
                    severity=Severity.WARNING,
                    message=(
                        f"column {col!r} looks identity-like ({why}): "
                        f"{n_unique}/{n_non_null} distinct "
                        f"({unique_ratio:.0%} unique)"
                    ),
                    fix=(
                        f"if {col!r} is the grouping entity, pass it as --entity "
                        f"(it is then never tokenized as a feature, T2); "
                        f"otherwise drop it before tokenizing"
                    ),
                    data={
                        "column": col,
                        "kind": "identity_like",
                        "n_unique": n_unique,
                        "n_non_null": n_non_null,
                        "unique_ratio": round(unique_ratio, 4),
                        "name_match": name_hit,
                        "near_unique": near_unique,
                    },
                )
            )

        # --- target leakage --------------------------------------------------
        if target_series is not None:
            # A non-target column that perfectly co-varies with the label leaks it.
            try:
                num_col = pd.to_numeric(series, errors="coerce")
                num_target = pd.to_numeric(target_series, errors="coerce")
                pair = pd.concat([num_col, num_target], axis=1).dropna()
                if len(pair) >= 3 and pair.iloc[:, 0].nunique() > 1 and pair.iloc[:, 1].nunique() > 1:
                    corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
                    if corr is not None and pd.notna(corr) and abs(corr) >= _HIGH_CORR:
                        diags.append(
                            Diagnostic(
                                contract="EDA",
                                severity=Severity.WARNING,
                                message=(
                                    f"column {col!r} is {abs(corr):.0%}-correlated with "
                                    f"target {target!r} — possible leakage / a derived label"
                                ),
                                fix=(
                                    f"confirm {col!r} is a legitimate pre-event feature, "
                                    f"not computed from {target!r}; drop it if it is"
                                ),
                                data={
                                    "column": col,
                                    "kind": "target_correlated",
                                    "target": target,
                                    "abs_corr": round(abs(float(corr)), 4),
                                },
                            )
                        )
                        continue
            except (TypeError, ValueError):
                pass
            # Categorical 1:1 determinism: each value of `col` maps to one label.
            if not pd.api.types.is_numeric_dtype(series):
                pair = pd.concat([series, target_series], axis=1).dropna()
                if len(pair) >= 3 and pair.iloc[:, 0].nunique() > 1:
                    per_value_labels = pair.groupby(pair.columns[0])[pair.columns[1]].nunique()
                    if (per_value_labels <= 1).all():
                        diags.append(
                            Diagnostic(
                                contract="EDA",
                                severity=Severity.WARNING,
                                message=(
                                    f"column {col!r} perfectly determines target "
                                    f"{target!r} (each value maps to one label) — likely leakage"
                                ),
                                fix=(
                                    f"verify {col!r} is observable before the event; "
                                    f"drop it if it encodes {target!r}"
                                ),
                                data={
                                    "column": col,
                                    "kind": "target_determines",
                                    "target": target,
                                },
                            )
                        )

    return diags
