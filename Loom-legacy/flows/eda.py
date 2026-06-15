"""Loom's read-only EDA (exploratory data analysis) Metaflow flow.

This module defines the single static ``FlowSpec`` -- :class:`EdaFlow` -- that the
``loom eda`` command runs (via the Metaflow MLOps interface) to **profile** an
ingested data object. EDA is the **read-only tier** of the approval matrix
(design-spec §3): it never prompts and never writes outside its own workspace --
it materializes the data object into a tmp ``./input``, profiles it, and emits a
Metaflow run + an ``@card``.

The input is a **Metaflow data object** referenced by ``dataset_ref`` (a pathspec
like ``"IngestDataset/123"`` produced by ``loom ingest``). The ``start`` step
materializes it into ``./input`` through the Metaflow **Client API** only
(:func:`loom.dataio.materialize_dataset`); Loom never touches the underlying
datastore (local or S3/minio) directly -- that is Metaflow's concern.

Flow shape::

    start --> profile --> end

* ``start``   -- materialize the data object's ``train`` (and optional ``test``)
                 CSVs into a tmp ``./input`` via the Client API and load them with
                 pandas.
* ``profile`` -- compute the profile dict (schema, dtypes, nrows, missingness,
                 numeric describe, target class balance, top-k correlations, and
                 simple LEAKAGE flags) via the pure, unit-testable
                 :func:`profile_dataframe`, store it on ``self.profile``, and
                 render an ``@card`` (Markdown + Tables) summarizing it.
* ``end``     -- carry ``self.profile`` forward so ``Run.data.profile`` exposes
                 it to the MLOps interface's Client-API read.

The profiling *logic* is factored into the module-level pure function
:func:`profile_dataframe` so it is unit-testable on a small in-memory DataFrame
with no Metaflow involved. The flow step is a thin wrapper that calls it.

Only standard Metaflow APIs are used (``FlowSpec``, ``@step``, ``Parameter``,
``@card``, and the ``current.card`` append API). ``pandas`` and ``loom`` are
imported *inside* the steps so the flow file parses even where they are not yet
importable until the Runner subprocess sets up the environment.
"""

from __future__ import annotations

from typing import Any

from metaflow import FlowSpec, Parameter, card, current, step

#: Number of most-correlated feature pairs to report.
_TOP_K_CORRELATIONS = 10

#: Absolute Pearson correlation between a feature and the (numeric-encoded)
#: target at/above which the feature is flagged as a LEAKAGE smell (near-perfect
#: predictor). Deliberately high so only genuinely suspicious features trip it.
_LEAKAGE_CORR_THRESHOLD = 0.98

#: Fraction of a feature's value-groups that must map to a single target value
#: for the feature to be flagged as a duplicate-of-target leakage smell.
_DUP_TARGET_THRESHOLD = 0.999

#: A feature whose distinct-value count is at/above this fraction of the row count
#: is treated as ID-like: it trivially "determines" the target (each value appears
#: ~once), so the duplicate-of-target check would false-positive. Such a column is
#: excluded from that check (it is an ID smell, not target leakage).
_ID_LIKE_CARDINALITY_FRACTION = 0.9


def _is_numeric_dtype(series: Any) -> bool:
    """Return whether a pandas Series has a numeric dtype (lazy pandas import)."""
    import pandas as pd

    return bool(pd.api.types.is_numeric_dtype(series))


def profile_dataframe(
    train: Any,
    test: Any = None,
    target: str | None = None,
) -> dict:
    """Profile a training DataFrame into a JSON-able EDA dict (pure function).

    This is the unit-testable core of :class:`EdaFlow`: given pandas DataFrames it
    computes the whole profile with no Metaflow involved. It is domain-neutral --
    it never assumes a task type, column meaning, or vertical; it reports only
    what the data and the (optional) declared target actually say.

    The returned dict is intentionally small and JSON-able (suitable for
    ``RunResult.summary`` and a learnings row): per-column stats and a handful of
    flags, never raw rows.

    Args:
        train: The training DataFrame.
        test: The optional test DataFrame (used only to *infer* the target when
            one is not declared: a column present in train but absent from test).
        target: The declared target column name. When ``None`` the function infers
            one (train-only column vs test, else a literal ``"target"``/``"label"``
            column) and reports it under ``target_inferred``.

    Returns:
        A JSON-able profile dict with keys: ``nrows``, ``ncols``, ``columns``,
        ``dtypes``, ``missingness`` (% per column), ``numeric_describe``,
        ``target``, ``target_inferred`` (bool), ``target_balance``,
        ``top_correlations`` (list of ``[a, b, corr]``), ``leakage_flags`` (list
        of ``{column, kind, detail}``), and ``leakage`` (bool, any flag present).
    """
    import pandas as pd  # noqa: F401  (used via the DataFrame API)

    columns = [str(c) for c in train.columns]
    nrows = int(len(train))
    ncols = len(columns)
    dtypes = {str(c): str(train[c].dtype) for c in train.columns}

    # Missingness as a percentage per column (0..100), rounded for compactness.
    missingness: dict[str, float] = {}
    for c in train.columns:
        frac = float(train[c].isna().mean()) if nrows else 0.0
        missingness[str(c)] = round(frac * 100.0, 4)

    # Numeric describe: count/mean/std/min/max per numeric column (JSON-able).
    numeric_cols = [c for c in train.columns if _is_numeric_dtype(train[c])]
    numeric_describe: dict[str, dict] = {}
    for c in numeric_cols:
        col = train[c].dropna()
        if col.empty:
            continue
        numeric_describe[str(c)] = {
            "count": int(col.count()),
            "mean": _finite_or_none(float(col.mean())),
            "std": _finite_or_none(float(col.std())) if col.count() > 1 else None,
            "min": _finite_or_none(float(col.min())),
            "max": _finite_or_none(float(col.max())),
        }

    # Resolve the target: declared > train-only-vs-test > literal target/label.
    target_col, target_inferred = _resolve_target(columns, test, target)

    # Target class balance: value -> count (capped to the most common classes so
    # a high-cardinality / continuous target does not bloat the summary).
    target_balance: dict[str, int] | None = None
    if target_col is not None and target_col in train.columns:
        counts = train[target_col].value_counts(dropna=False)
        target_balance = {
            str(k): int(v) for k, v in counts.head(50).items()
        }

    # Top-k feature correlations among numeric columns (absolute Pearson).
    top_correlations = _top_correlations(train, numeric_cols, _TOP_K_CORRELATIONS)

    # Simple leakage flags against the target.
    leakage_flags = _leakage_flags(train, target_col, numeric_cols)

    return {
        "nrows": nrows,
        "ncols": ncols,
        "columns": columns,
        "dtypes": dtypes,
        "missingness": missingness,
        "numeric_describe": numeric_describe,
        "target": target_col,
        "target_inferred": bool(target_inferred),
        "target_balance": target_balance,
        "top_correlations": top_correlations,
        "leakage_flags": leakage_flags,
        "leakage": bool(leakage_flags),
    }


def _finite_or_none(value: float) -> float | None:
    """Return ``value`` if finite, else ``None`` (keeps the summary JSON-able)."""
    import math

    return value if math.isfinite(value) else None


def _resolve_target(
    columns: list[str], test: Any, declared: str | None
) -> tuple[str | None, bool]:
    """Resolve the target column and whether it was inferred.

    Precedence: an explicitly declared ``target`` (if present in the columns) >
    a single column present in train but absent from test (the usual train/test
    asymmetry) > a column literally named ``target`` or ``label``.

    Args:
        columns: The train column names (as strings).
        test: The optional test DataFrame.
        declared: The user-declared target column name, or ``None``.

    Returns:
        ``(target_or_None, inferred)`` -- ``inferred`` is ``False`` only when the
        declared target was used as-is.
    """
    declared = (declared or "").strip()
    if declared and declared in columns:
        return declared, False

    if test is not None:
        test_cols = {str(c) for c in test.columns}
        only_in_train = [c for c in columns if c not in test_cols]
        if len(only_in_train) == 1:
            return only_in_train[0], True

    for candidate in ("target", "label"):
        if candidate in columns:
            return candidate, True

    return None, True


def _top_correlations(
    train: Any, numeric_cols: list[str], k: int
) -> list[list]:
    """Return the top-``k`` numeric feature pairs by absolute Pearson correlation.

    Args:
        train: The train DataFrame.
        numeric_cols: The numeric column names.
        k: How many pairs to keep.

    Returns:
        A list of ``[col_a, col_b, corr]`` rows (corr rounded), strongest first.
        Empty when fewer than two numeric columns exist.
    """
    if len(numeric_cols) < 2:
        return []

    import warnings

    import numpy as np

    # ``.corr()`` on a constant/degenerate column emits benign divide-by-zero /
    # DoF RuntimeWarnings; the non-finite results they produce are filtered below.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        corr = train[numeric_cols].corr(numeric_only=True)
    pairs: list[list] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iat[i, j]
            if value is None:
                continue
            try:
                fvalue = float(value)
            except (TypeError, ValueError):
                continue
            import math

            if not math.isfinite(fvalue):
                continue
            pairs.append([str(cols[i]), str(cols[j]), round(fvalue, 6)])

    pairs.sort(key=lambda row: abs(row[2]), reverse=True)
    return pairs[:k]


def _leakage_flags(
    train: Any, target_col: str | None, numeric_cols: list[str]
) -> list[dict]:
    """Detect simple leakage smells of a feature against the target.

    Two cheap, domain-neutral checks:

    * **near-perfect predictor** -- a numeric feature whose absolute Pearson
      correlation with the (numeric-encoded) target is at/above
      :data:`_LEAKAGE_CORR_THRESHOLD`;
    * **duplicate-of-target** -- a non-target column whose values map 1:1 onto the
      target (a per-row deterministic function of the target), measured as the
      fraction of target groups that contain exactly one distinct feature value.

    Args:
        train: The train DataFrame.
        target_col: The resolved target column, or ``None`` (then no flags).
        numeric_cols: The numeric column names.

    Returns:
        A list of ``{"column", "kind", "detail"}`` dicts (empty when clean).
    """
    if target_col is None or target_col not in train.columns:
        return []

    import pandas as pd

    flags: list[dict] = []
    target = train[target_col]
    n = int(len(train))

    # The duplicate-of-target check only makes sense when the target itself has
    # more than one value AND a feature's groups actually aggregate rows. A target
    # with a single class is trivially "determined" by everything.
    target_has_variation = int(target.nunique(dropna=False)) > 1

    # Numeric-encode the target once for the correlation check (factorize maps any
    # dtype to integer codes; for an already-numeric target this is a monotone
    # relabel that preserves a |corr|~1 perfect predictor relationship well enough
    # for a smell test).
    if _is_numeric_dtype(target):
        target_numeric = target
    else:
        codes, _ = pd.factorize(target)
        target_numeric = pd.Series(codes, index=target.index)

    for c in train.columns:
        col_name = str(c)
        if col_name == str(target_col):
            continue

        # near-perfect predictor (numeric features only).
        if c in numeric_cols and _is_numeric_dtype(target_numeric):
            import warnings

            import numpy as np

            try:
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore", RuntimeWarning)
                    corr = float(train[c].corr(target_numeric))
            except Exception:  # pragma: no cover - degenerate column
                corr = float("nan")
            import math

            if math.isfinite(corr) and abs(corr) >= _LEAKAGE_CORR_THRESHOLD:
                flags.append(
                    {
                        "column": col_name,
                        "kind": "near_perfect_predictor",
                        "detail": f"|corr| with target = {abs(corr):.4f}",
                    }
                )
                continue  # one flag per column is enough

        # duplicate-of-target: a feature whose value-groups each map to exactly one
        # target value (a deterministic function of the target). Skip when the
        # target has no variation, and skip ID-like features (cardinality ~= nrows)
        # which trivially "determine" the target because each value appears once.
        if n and target_has_variation:
            try:
                cardinality = int(train[c].nunique(dropna=False))
            except Exception:  # pragma: no cover - unhashable column
                cardinality = n
            id_like = cardinality >= max(2, int(n * _ID_LIKE_CARDINALITY_FRACTION))
            if cardinality > 1 and not id_like:
                try:
                    grouped = train.groupby(c, dropna=False)[target_col].nunique()
                    determined = int((grouped <= 1).sum())
                    frac = (
                        determined / float(len(grouped)) if len(grouped) else 0.0
                    )
                except Exception:  # pragma: no cover - degenerate column
                    frac = 0.0
                if frac >= _DUP_TARGET_THRESHOLD:
                    flags.append(
                        {
                            "column": col_name,
                            "kind": "duplicate_of_target",
                            "detail": (
                                f"{frac*100:.2f}% of {col_name} values map to a "
                                "single target value"
                            ),
                        }
                    )

    return flags


class EdaFlow(FlowSpec):
    """Read-only profile of an ingested Metaflow data object.

    Materializes the data object referenced by ``dataset_ref`` into a tmp
    ``./input`` via the Client API, profiles it with :func:`profile_dataframe`,
    and emits a Metaflow run + an ``@card``. The profile dict is carried on
    ``self.profile`` so the MLOps interface reads it back from ``Run.data``.
    """

    #: Metaflow **pathspec** of the ingested data object to profile (e.g.
    #: ``"IngestDataset/123"``, produced by ``loom ingest``). Read via the Client
    #: API only; Loom never touches the backing datastore.
    dataset_ref = Parameter(
        "dataset_ref",
        required=True,
        type=str,
        help="Metaflow pathspec of the ingested data object (e.g. IngestDataset/123).",
    )

    #: Optional declared target/label column. When empty the flow infers one.
    target = Parameter(
        "target",
        default="",
        type=str,
        help="Optional target/label column name (inferred when omitted).",
    )

    @step
    def start(self) -> None:
        """Materialize the data object into ``./input`` and load it with pandas.

        Reads the ``train`` (and optional ``test``) artifacts of the
        ``dataset_ref`` Metaflow data object through the Client API
        (:func:`loom.dataio.materialize_dataset`) into a tmp ``./input``, then
        loads them with pandas. READ-ONLY: nothing is written outside this
        workspace.
        """
        import os
        import tempfile

        import pandas as pd

        from loom.dataio import materialize_dataset

        workspace = tempfile.mkdtemp(prefix="loom-eda-")
        input_dir = os.path.join(workspace, "input")
        os.makedirs(input_dir, exist_ok=True)

        ref = (self.dataset_ref or "").strip()
        materialize_dataset(ref, input_dir)

        train_path = os.path.join(input_dir, "train.csv")
        self._train_df = pd.read_csv(train_path)

        test_path = os.path.join(input_dir, "test.csv")
        self._test_df = pd.read_csv(test_path) if os.path.isfile(test_path) else None

        self.workspace_dir = workspace
        self.next(self.profile)

    @card
    @step
    def profile(self) -> None:
        """Compute the profile dict and render the ``@card`` summarizing it.

        Delegates the whole computation to the pure :func:`profile_dataframe`, so
        the logic is unit-testable without Metaflow, then renders a Markdown +
        Tables ``@card`` from the resulting dict.
        """
        declared = (self.target or "").strip() or None
        self.profile = profile_dataframe(
            self._train_df, test=self._test_df, target=declared
        )
        self._render_card(self.profile)
        self.next(self.end)

    def _render_card(self, profile: dict) -> None:
        """Render a Markdown + Tables ``@card`` from the profile dict.

        Args:
            profile: The JSON-able profile dict from :func:`profile_dataframe`.
        """
        from metaflow.cards import Markdown, Table

        current.card.append(Markdown("# Loom EDA profile"))
        current.card.append(
            Markdown(
                f"**dataset_ref:** `{self.dataset_ref}`  \n"
                f"**rows:** {profile['nrows']}  \n"
                f"**columns:** {profile['ncols']}  \n"
                f"**target:** `{profile.get('target')}`"
                + (" _(inferred)_" if profile.get("target_inferred") else "")
            )
        )

        # Schema + missingness table.
        current.card.append(Markdown("## Columns"))
        schema_rows = [
            [
                col,
                profile["dtypes"].get(col, ""),
                f"{profile['missingness'].get(col, 0.0):.2f}%",
            ]
            for col in profile["columns"]
        ]
        current.card.append(
            Table(schema_rows, headers=["column", "dtype", "missing %"])
        )

        # Target balance.
        balance = profile.get("target_balance")
        if balance:
            current.card.append(Markdown("## Target balance"))
            current.card.append(
                Table(
                    [[k, v] for k, v in balance.items()],
                    headers=["value", "count"],
                )
            )

        # Top correlations.
        corrs = profile.get("top_correlations") or []
        if corrs:
            current.card.append(Markdown("## Top feature correlations"))
            current.card.append(
                Table(
                    [[a, b, f"{c:.4f}"] for a, b, c in corrs],
                    headers=["feature a", "feature b", "corr"],
                )
            )

        # Leakage flags.
        flags = profile.get("leakage_flags") or []
        current.card.append(Markdown("## Leakage flags"))
        if flags:
            current.card.append(
                Table(
                    [[f["column"], f["kind"], f["detail"]] for f in flags],
                    headers=["column", "kind", "detail"],
                )
            )
        else:
            current.card.append(Markdown("_No leakage smells detected._"))

    @step
    def end(self) -> None:
        """Carry ``self.profile`` forward so ``Run.data.profile`` exposes it.

        Metaflow persists step artifacts, so ``self.profile`` (set in
        ``profile``) is already on ``Run.data``; the MLOps interface reads it back
        for the command's summary. Nothing else to do.
        """
        pass


if __name__ == "__main__":
    EdaFlow()
