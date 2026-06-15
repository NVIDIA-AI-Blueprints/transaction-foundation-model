"""Loom's feature-engineering Metaflow flow -- writes a NEW data object.

This module defines the single static ``FlowSpec`` -- :class:`FeaturesFlow` -- that
the ``loom features`` command runs (via the Metaflow MLOps interface) to **build
engineered features** from an ingested data object and persist the result as a
*new* Metaflow data object. Features is the **workspace-write tier** of the approval
matrix (design-spec §3): it reads the source data object read-only (through the
Client API), builds features locally inside its own Metaflow workspace, and writes
a brand-new data object -- light, auto, no prompt, network off.

Like :class:`flows.ingest_dataset.IngestDataset`, this flow is one of the few that
*writes* a data object: its ``end`` step carries the engineered ``train`` (and
optional ``test``) DataFrames plus a ``schema`` dict and a content ``fingerprint``
as Metaflow artifacts, so the produced run's pathspec (``"FeaturesFlow/<run_id>"``)
is itself a ``dataset_ref`` every downstream verb (``loom pipeline`` /
``loom validate`` / ...) can consume via ``--dataset``. The same Client-API door
(:mod:`loom.dataio`) reads it back -- the artifact names (``train`` / ``test`` /
``schema``) deliberately match ``IngestDataset`` so :func:`loom.dataio.materialize_dataset`
and :func:`loom.dataio.dataset_schema` work against a FeaturesFlow run unchanged.

The input is a **Metaflow data object** referenced by ``dataset_ref`` (a pathspec
like ``"IngestDataset/123"``). The ``start`` step materializes it into ``./input``
through the Metaflow **Client API** only; Loom never touches the underlying
datastore (local or S3/minio) directly -- that is Metaflow's concern.

LEAKAGE composition: when an upstream ``loom eda`` run is supplied via
``eda_run``, its profile's ``leakage_flags`` are read (Client API) and the flagged
columns are DROPPED before feature building, and the run is marked
``refused_leakage`` so a downstream verb sees that leakage was handled rather than
silently engineered into the feature set. This is the ``eda -> features``
composition edge.

Flow shape::

    start --> build --> end

* ``start`` -- materialize the source data object's ``train`` (and optional
               ``test``) CSVs into a tmp ``./input`` via the Client API, load them
               with pandas, and read any upstream EDA leakage flags (Client API).
* ``build`` -- build engineered features via the pure, unit-testable
               :func:`build_features` (numeric scaling/interactions, categorical
               encoding, datetime parts, simple aggregations), compute the new
               ``schema`` + ``fingerprint``, store the engineered ``train``/``test``
               and a ``summary`` dict on ``self``, and render an ``@card`` (feature
               list, before/after schema, null/variance stats).
* ``end``   -- no-op; the ``train`` / ``test`` / ``schema`` / ``fingerprint`` /
               ``summary`` artifacts on ``self`` are the NEW data object.

The feature-building *logic* is factored into the module-level pure function
:func:`build_features` so it is unit-testable on a small in-memory DataFrame with
no Metaflow involved. The flow step is a thin wrapper that calls it. It is
domain-neutral: it never assumes a task type, column meaning, or vertical -- it
applies only generic transforms the data's dtypes justify.

Only standard Metaflow APIs are used (``FlowSpec``, ``@step``, ``Parameter``,
``@card``, and the ``current.card`` append API). ``pandas`` / ``numpy`` and
``loom`` are imported *inside* the steps (and the pure function) so the flow file
parses even where they are not yet importable until the Runner subprocess sets up
the environment.
"""

from __future__ import annotations

from typing import Any

from metaflow import FlowSpec, Parameter, card, current, step

#: A non-target object/categorical column with at/below this many distinct values
#: is one-hot encoded; above it the column is frequency-encoded (a domain-neutral
#: numeric stand-in that never explodes the feature width on high cardinality).
_ONEHOT_MAX_CARDINALITY = 12

#: The most-variable numeric feature pairs to combine into a multiplicative
#: interaction (kept small so the feature width stays bounded and the transform
#: stays cheap -- this is feature *engineering*, not an exhaustive cross).
_MAX_INTERACTIONS = 3

#: Datetime parts extracted from a detected datetime column (domain-neutral
#: calendar decomposition). Order is the emitted-column order.
_DATETIME_PARTS = ("year", "month", "day", "dayofweek", "hour")


def build_features(
    train: Any,
    test: Any = None,
    target: str | None = None,
    recipe: str | None = None,
    drop_columns: list[str] | None = None,
) -> dict:
    """Build engineered features from ``train`` (and ``test``) into a result dict (pure).

    This is the unit-testable core of :class:`FeaturesFlow`: given pandas
    DataFrames it builds the whole engineered feature set with no Metaflow
    involved. It is domain-neutral -- it never assumes a task type, column meaning,
    or vertical; it applies only generic transforms the column dtypes justify:

    * **numeric scaling** -- each numeric (non-target) column standardized to a
      ``<col>__z`` column (zero mean / unit std; constant columns map to 0);
    * **numeric interactions** -- the top :data:`_MAX_INTERACTIONS` most-variable
      numeric pairs combined multiplicatively into ``<a>__x__<b>`` columns;
    * **categorical encoding** -- low-cardinality object columns one-hot encoded
      (``<col>__<value>``), higher-cardinality ones frequency-encoded
      (``<col>__freq``);
    * **datetime parts** -- a parseable datetime column decomposed into
      ``<col>__{year,month,day,dayofweek,hour}`` integer parts;
    * **simple aggregations** -- a per-row ``<group>__<num>__grpmean`` (the group
      mean of a numeric column within each low-cardinality categorical) when both a
      low-cardinality categorical and a numeric column are present.

    The original columns are preserved alongside the engineered ones, and the
    ``target`` (when given/detected) is passed through untouched so the engineered
    table is a drop-in ``dataset_ref`` for a downstream verb. Any ``drop_columns``
    (e.g. EDA-flagged leakage columns) are removed from BOTH frames first.

    Args:
        train: The training DataFrame.
        test: The optional test DataFrame (the same transforms are applied; columns
            absent from test -- e.g. the target -- are simply skipped there).
        target: The declared target/label column name, preserved untouched. When
            ``None`` it is inferred (a train-only-vs-test column, else a literal
            ``"target"``/``"label"``) so it is never accidentally engineered.
        recipe: Optional named recipe selecting which transform families to apply
            (``"minimal"`` = scaling + encoding only; ``"full"``/``None`` = all).
            Domain-neutral knobs, never a vertical.
        drop_columns: Columns to drop from both frames before building (e.g. EDA
            leakage flags). Silently ignores names not present.

    Returns:
        A JSON-friendly result dict with keys: ``train`` (the engineered train
        DataFrame), ``test`` (the engineered test DataFrame or ``None``),
        ``schema`` (columns/dtypes/nrows/target of the engineered train, matching
        the ``IngestDataset`` schema shape), ``target``, ``recipe``,
        ``added_features`` (the new column names), ``dropped_columns`` (those
        removed), ``n_features_before`` / ``n_features_after`` (column counts), and
        ``null_stats`` (per-engineered-column null % + variance, JSON-able).

    Raises:
        ValueError: If ``train`` is empty (no rows or no columns).
    """
    import numpy as np  # noqa: F401  (used via the DataFrame API below)
    import pandas as pd

    if train is None or len(train.columns) == 0 or len(train) == 0:
        raise ValueError(
            "build_features requires a non-empty train DataFrame (got "
            f"{0 if train is None else len(train)} rows)."
        )

    recipe_name = (recipe or "full").strip().lower() or "full"
    minimal = recipe_name == "minimal"

    # Drop requested columns (e.g. EDA leakage flags) from both frames first.
    drop = [str(c) for c in (drop_columns or []) if str(c) in train.columns]
    train_df = train.drop(columns=drop) if drop else train.copy()
    test_df = None
    if test is not None:
        test_drop = [str(c) for c in drop if str(c) in test.columns]
        test_df = test.drop(columns=test_drop) if test_drop else test.copy()

    columns_before = [str(c) for c in train_df.columns]
    n_before = len(columns_before)

    target_col = _resolve_target(columns_before, test_df, target)

    # Feature columns: everything that is not the (preserved) target.
    feature_cols = [c for c in train_df.columns if str(c) != str(target_col)]
    numeric_cols = [c for c in feature_cols if _is_numeric_dtype(train_df[c])]
    object_cols = [
        c
        for c in feature_cols
        if not _is_numeric_dtype(train_df[c]) and not _looks_datetime(train_df[c])
    ]
    datetime_cols = [
        c for c in feature_cols if _looks_datetime(train_df[c])
    ]

    added: list[str] = []

    def _emit(frame_train: Any, frame_test: Any, name: str, fn) -> None:
        """Compute a derived column on train (and test when the source col exists)."""
        frame_train[name] = fn(frame_train)
        added.append(name)
        if frame_test is not None and _source_present(fn, frame_test):
            try:
                frame_test[name] = fn(frame_test)
            except Exception:  # pragma: no cover - test frame missing a source col
                pass

    # 1) numeric scaling (z-score) using TRAIN statistics for both frames.
    for c in numeric_cols:
        mean = float(train_df[c].mean())
        std = float(train_df[c].std())
        std = std if std and std == std else 0.0  # NaN-guard
        name = f"{c}__z"
        scale = std if std > 0 else 1.0
        _emit(
            train_df,
            test_df,
            name,
            (lambda col, m, s: (lambda f: (f[col] - m) / s))(c, mean, scale),
        )

    # 2) categorical encoding (one-hot for low cardinality, frequency otherwise).
    for c in object_cols:
        cardinality = int(train_df[c].nunique(dropna=True))
        if 0 < cardinality <= _ONEHOT_MAX_CARDINALITY:
            for value in sorted(
                str(v) for v in train_df[c].dropna().unique()
            ):
                name = f"{c}__{value}"
                _emit(
                    train_df,
                    test_df,
                    name,
                    (lambda col, val: (lambda f: (f[col].astype(str) == val).astype(int)))(
                        c, value
                    ),
                )
        else:
            freq = train_df[c].astype(str).value_counts(normalize=True).to_dict()
            name = f"{c}__freq"
            _emit(
                train_df,
                test_df,
                name,
                (lambda col, table: (lambda f: f[col].astype(str).map(table).fillna(0.0)))(
                    c, freq
                ),
            )

    # 3) datetime parts (calendar decomposition).
    if not minimal:
        for c in datetime_cols:
            for part in _DATETIME_PARTS:
                name = f"{c}__{part}"
                _emit(
                    train_df,
                    test_df,
                    name,
                    (lambda col, p: (lambda f: _datetime_part(f[col], p)))(c, part),
                )

    # 4) numeric interactions (top-variance pairs, multiplicative).
    if not minimal and len(numeric_cols) >= 2:
        ranked = sorted(
            numeric_cols,
            key=lambda c: float(train_df[c].std() or 0.0),
            reverse=True,
        )
        pairs = []
        for i in range(len(ranked)):
            for j in range(i + 1, len(ranked)):
                pairs.append((ranked[i], ranked[j]))
                if len(pairs) >= _MAX_INTERACTIONS:
                    break
            if len(pairs) >= _MAX_INTERACTIONS:
                break
        for a, b in pairs:
            name = f"{a}__x__{b}"
            _emit(
                train_df,
                test_df,
                name,
                (lambda ca, cb: (lambda f: f[ca] * f[cb]))(a, b),
            )

    # 5) simple aggregation: per-row group-mean of a numeric within a categorical.
    if not minimal and numeric_cols:
        low_card = [
            c
            for c in object_cols
            if 0 < int(train_df[c].nunique(dropna=True)) <= _ONEHOT_MAX_CARDINALITY
        ]
        if low_card:
            group_col = low_card[0]
            num_col = numeric_cols[0]
            group_means = (
                train_df.groupby(group_col, dropna=False)[num_col].mean().to_dict()
            )
            overall = float(train_df[num_col].mean())
            name = f"{group_col}__{num_col}__grpmean"
            _emit(
                train_df,
                test_df,
                name,
                (
                    lambda gc, table, default: (
                        lambda f: f[gc].map(table).fillna(default)
                    )
                )(group_col, group_means, overall),
            )

    columns_after = [str(c) for c in train_df.columns]
    schema = _build_schema(train_df, test_df, target_col)
    null_stats = _null_stats(train_df, added)

    return {
        "train": train_df,
        "test": test_df,
        "schema": schema,
        "target": target_col,
        "recipe": recipe_name,
        "added_features": added,
        "dropped_columns": drop,
        "n_features_before": n_before,
        "n_features_after": len(columns_after),
        "null_stats": null_stats,
    }


# ---------------------------------------------------------------------------
# Pure helpers (no Metaflow; importable + unit-testable on in-memory frames).
# ---------------------------------------------------------------------------


def _is_numeric_dtype(series: Any) -> bool:
    """Return whether a pandas Series has a numeric dtype (lazy pandas import)."""
    import pandas as pd

    return bool(pd.api.types.is_numeric_dtype(series))


def _looks_datetime(series: Any) -> bool:
    """Return whether a Series is datetime-typed or a parseable datetime object.

    A native datetime dtype is accepted directly. An object column is treated as
    datetime only when a strict parse of a small sample succeeds for the bulk of
    non-null values -- so a plain string column is never mistaken for a date.
    """
    import pandas as pd

    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (series.dtype == object):
        return False
    sample = series.dropna().head(20)
    if sample.empty:
        return False
    import warnings

    try:
        # A free-form date column triggers a benign "could not infer format"
        # UserWarning on the dateutil fallback; it is expected here (this is a
        # heuristic probe), so silence it rather than leak it to the caller.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce")
    except Exception:  # pragma: no cover - unparseable object column
        return False
    return float(parsed.notna().mean()) >= 0.8


def _datetime_part(series: Any, part: str) -> Any:
    """Extract a calendar ``part`` from a (possibly object) datetime Series.

    Returns an integer Series with ``-1`` where the value is missing/unparseable,
    so the engineered column is always numeric and NaN-free for a downstream model.
    """
    import warnings

    import pandas as pd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dt = pd.to_datetime(series, errors="coerce")
    values = getattr(dt.dt, part)
    return values.fillna(-1).astype(int)


def _source_present(fn: Any, frame: Any) -> bool:
    """Best-effort: can ``fn`` be evaluated on ``frame`` (its source col exists)?

    Evaluates the closure against the frame and reports whether it raised; used to
    skip a transform on the test frame when its source column is absent (e.g. the
    target). Pure and side-effect free aside from the discarded probe.
    """
    try:
        fn(frame)
        return True
    except Exception:
        return False


def _resolve_target(
    columns: list[str], test: Any, declared: str | None
) -> str | None:
    """Resolve the target column to preserve (declared > train-only-vs-test > literal).

    Mirrors the EDA flow's resolution so ``features`` preserves exactly the column a
    prior ``eda`` would have called the target, never engineering it as a feature.
    """
    declared = (declared or "").strip()
    if declared and declared in columns:
        return declared
    if test is not None:
        test_cols = {str(c) for c in test.columns}
        only_in_train = [c for c in columns if c not in test_cols]
        if len(only_in_train) == 1:
            return only_in_train[0]
    for candidate in ("target", "label"):
        if candidate in columns:
            return candidate
    return None


def _build_schema(train_df: Any, test_df: Any, target: str | None) -> dict:
    """Build the engineered data object's schema dict (IngestDataset shape).

    Records the engineered train columns + (stringified) dtypes, the row count, and
    the preserved ``target`` -- the same ``columns``/``dtypes``/``nrows``/``target``
    shape :class:`IngestDataset` writes, so :func:`loom.dataio.dataset_schema` reads
    a FeaturesFlow run identically.
    """
    columns = [str(c) for c in train_df.columns]
    dtypes = {str(c): str(train_df[c].dtype) for c in train_df.columns}
    return {
        "columns": columns,
        "dtypes": dtypes,
        "nrows": int(len(train_df)),
        "target": target,
    }


def _null_stats(train_df: Any, added: list[str]) -> dict:
    """Per-engineered-column null % and variance (JSON-able, for the card).

    Reports only the columns this flow ADDED (the before/after delta the card
    surfaces), never raw rows: ``{column: {"null_pct", "variance"}}``.
    """
    import math

    n = int(len(train_df))
    out: dict[str, dict] = {}
    for col in added:
        if col not in train_df.columns:
            continue
        series = train_df[col]
        null_pct = round(float(series.isna().mean()) * 100.0, 4) if n else 0.0
        try:
            variance = float(series.var())
        except Exception:  # pragma: no cover - non-numeric engineered column
            variance = float("nan")
        out[str(col)] = {
            "null_pct": null_pct,
            "variance": round(variance, 6) if math.isfinite(variance) else None,
        }
    return out


def fingerprint_frame(train: Any, schema: dict) -> str:
    """Compute a content fingerprint for an engineered data object (pure).

    A stable SHA-256 over the engineered train frame's shape + column names +
    dtypes + a hash of the cell values, so a downstream verb can assert it consumed
    a specific feature build. Content-addressed, not time-based, so re-running the
    same build on the same input yields the same fingerprint.

    Args:
        train: The engineered train DataFrame.
        schema: The engineered schema dict (columns/dtypes/nrows/target).

    Returns:
        A hex digest string (``"sha256:<hexdigest>"``).
    """
    import hashlib

    import pandas as pd  # noqa: F401  (used via the DataFrame API)

    hasher = hashlib.sha256()
    hasher.update(str(schema.get("columns")).encode("utf-8"))
    hasher.update(str(schema.get("dtypes")).encode("utf-8"))
    hasher.update(str(schema.get("nrows")).encode("utf-8"))
    hasher.update(str(schema.get("target")).encode("utf-8"))
    try:
        # ``pd.util.hash_pandas_object`` is order-stable and dtype-aware; summing the
        # per-row hashes keeps the digest cheap and content-addressed.
        row_hashes = pd.util.hash_pandas_object(train, index=False)
        hasher.update(str(int(row_hashes.sum())).encode("utf-8"))
    except Exception:  # pragma: no cover - unhashable cell content
        hasher.update(str(train.shape).encode("utf-8"))
    return f"sha256:{hasher.hexdigest()}"


def _leakage_columns_from_eda(eda_run: str | None) -> list[str]:
    """Read EDA-flagged leakage columns from an upstream ``eda`` run (Client API).

    Resolves ``eda_run`` (a ``"EdaFlow/<id>"`` pathspec) and reads its
    ``profile.leakage_flags`` through the Metaflow Client API, returning the flagged
    column names so the build can DROP them (the ``eda -> features`` composition
    edge). Best-effort: any failure (no run, no profile) yields ``[]`` so a missing
    upstream never fails the build. ``metaflow`` is imported lazily.

    Args:
        eda_run: Pathspec of the upstream EDA run, or ``None``.

    Returns:
        The flagged column names (empty when none / unreadable).
    """
    ref = (eda_run or "").strip()
    if not ref:
        return []
    try:
        from metaflow import Run

        data = Run(ref).data
        profile = getattr(data, "profile", None)
        if not isinstance(profile, dict):
            return []
        flags = profile.get("leakage_flags") or []
        return sorted({str(f.get("column")) for f in flags if f.get("column")})
    except Exception:  # noqa: BLE001 - upstream optional / unreadable
        return []


class FeaturesFlow(FlowSpec):
    """Build engineered features and WRITE them as a NEW Metaflow data object.

    Materializes the source data object referenced by ``dataset_ref`` into a tmp
    ``./input`` via the Client API, builds features with :func:`build_features`
    (dropping any EDA-flagged leakage columns when an ``eda_run`` is given), and
    emits a Metaflow run + an ``@card``. The engineered ``train`` / ``test`` /
    ``schema`` / ``fingerprint`` artifacts on ``self`` ARE the new data object, so
    this run's pathspec (``"FeaturesFlow/<run_id>"``) is a ``dataset_ref`` for the
    downstream verbs. Workspace-write tier: read-only over the source, writes only
    into this run's own workspace.
    """

    #: Metaflow **pathspec** of the source data object to build features from (e.g.
    #: ``"IngestDataset/123"``, produced by ``loom ingest`` -- or another
    #: ``"FeaturesFlow/<id>"`` to chain builds). Read via the Client API only.
    dataset_ref = Parameter(
        "dataset_ref",
        required=True,
        type=str,
        help="Metaflow pathspec of the source data object (e.g. IngestDataset/123).",
    )

    #: Optional declared target/label column, preserved untouched. Inferred when
    #: empty so it is never engineered as a feature.
    target = Parameter(
        "target",
        default="",
        type=str,
        help="Optional target/label column to preserve (inferred when omitted).",
    )

    #: Optional named transform recipe (``"minimal"`` = scaling + encoding only;
    #: ``"full"`` / empty = all families). Domain-neutral knobs only.
    recipe = Parameter(
        "recipe",
        default="",
        type=str,
        help="Optional transform recipe: minimal | full (default full).",
    )

    #: Optional pathspec of an upstream ``loom eda`` run whose leakage flags should
    #: be honoured -- the flagged columns are DROPPED before building (the
    #: ``eda -> features`` composition edge). Empty -> build all columns.
    eda_run = Parameter(
        "eda_run",
        default="",
        type=str,
        help="Optional upstream EdaFlow run; its leakage-flagged columns are dropped.",
    )

    @step
    def start(self) -> None:
        """Materialize the source data object and read any upstream EDA leakage flags.

        Reads the ``train`` (and optional ``test``) artifacts of the ``dataset_ref``
        Metaflow data object through the Client API
        (:func:`loom.dataio.materialize_dataset`) into a tmp ``./input``, loads them
        with pandas, and -- when an ``eda_run`` is given -- reads that run's leakage
        flags (Client API) so the flagged columns are dropped in ``build``.
        READ-ONLY over the source data.
        """
        import os
        import tempfile

        import pandas as pd

        from loom.dataio import dataset_schema, materialize_dataset

        workspace = tempfile.mkdtemp(prefix="loom-features-")
        input_dir = os.path.join(workspace, "input")
        os.makedirs(input_dir, exist_ok=True)

        ref = (self.dataset_ref or "").strip()
        materialize_dataset(ref, input_dir)

        train_path = os.path.join(input_dir, "train.csv")
        self._train_df = pd.read_csv(train_path)

        test_path = os.path.join(input_dir, "test.csv")
        self._test_df = pd.read_csv(test_path) if os.path.isfile(test_path) else None

        # Resolve the target: declared > the source data object's recorded schema.
        target = (self.target or "").strip()
        if not target:
            try:
                schema = dataset_schema(ref)
            except Exception:  # pragma: no cover - schema read edge case
                schema = {}
            target = str(schema.get("target") or "")
        self._resolved_target = target or None

        # Honour upstream EDA leakage flags (the eda -> features composition edge).
        self._leakage_drop = _leakage_columns_from_eda(
            (self.eda_run or "").strip() or None
        )

        self.workspace_dir = workspace
        self.next(self.build)

    @card
    @step
    def build(self) -> None:
        """Build features, write the new data-object artifacts, and render the ``@card``.

        Delegates the whole transform to the pure :func:`build_features` (so the
        logic is unit-testable without Metaflow), then persists the engineered
        ``train`` / ``test`` / ``schema`` / ``fingerprint`` as artifacts (the NEW
        data object, IngestDataset-shaped) plus a small ``summary`` dict the MLOps
        interface reads back, and renders a Markdown + Tables ``@card``
        (feature list, before/after schema, null/variance stats).
        """
        recipe = (self.recipe or "").strip() or None

        result = build_features(
            self._train_df,
            test=self._test_df,
            target=self._resolved_target,
            recipe=recipe,
            drop_columns=self._leakage_drop,
        )

        # The engineered frames + schema + fingerprint ARE the new data object.
        # Artifact names match IngestDataset so loom.dataio reads them unchanged.
        self.train = result["train"]
        self.test = result["test"]
        self.schema = result["schema"]
        self.fingerprint = fingerprint_frame(result["train"], result["schema"])
        self.dataset_name = f"features:{(self.dataset_ref or '').strip()}"

        # A small, JSON-able summary the MLOps interface surfaces (read back via the
        # 'summary' artifact name); bulk frames stay in Metaflow, never inlined.
        self.summary = {
            "source_dataset_ref": (self.dataset_ref or "").strip(),
            "target": result["target"],
            "recipe": result["recipe"],
            "fingerprint": self.fingerprint,
            "n_features_before": result["n_features_before"],
            "n_features_after": result["n_features_after"],
            "n_added": len(result["added_features"]),
            "added_features": result["added_features"],
            "dropped_columns": result["dropped_columns"],
            "refused_leakage": bool(result["dropped_columns"]),
            "null_stats": result["null_stats"],
            "verdict": "BUILT",
        }

        self._render_card(result)
        self.next(self.end)

    def _render_card(self, result: dict) -> None:
        """Render a Markdown + Tables ``@card`` from the build result dict.

        Args:
            result: The result dict from :func:`build_features`.
        """
        from metaflow.cards import Markdown, Table

        current.card.append(Markdown("# Loom feature build"))
        dropped = result.get("dropped_columns") or []
        current.card.append(
            Markdown(
                f"**source dataset_ref:** `{(self.dataset_ref or '').strip()}`  \n"
                f"**new dataset_ref:** `FeaturesFlow/<this run>`  \n"
                f"**target (preserved):** `{result.get('target')}`  \n"
                f"**recipe:** `{result.get('recipe')}`  \n"
                f"**fingerprint:** `{self.fingerprint}`  \n"
                f"**features:** {result.get('n_features_before')} -> "
                f"{result.get('n_features_after')} "
                f"(+{len(result.get('added_features') or [])})  \n"
                f"**VERDICT:** **{self.summary.get('verdict')}**"
            )
        )

        # Leakage handling (the eda -> features composition edge).
        current.card.append(Markdown("## Leakage handling"))
        if dropped:
            current.card.append(
                Markdown(
                    "Dropped "
                    + ", ".join(f"`{c}`" for c in dropped)
                    + " (flagged by the upstream EDA run before building)."
                )
            )
        else:
            current.card.append(
                Markdown("_No leakage columns dropped (none flagged upstream)._")
            )

        # Added features + their null/variance stats.
        null_stats = result.get("null_stats") or {}
        added = result.get("added_features") or []
        current.card.append(Markdown("## Engineered features"))
        if added:
            current.card.append(
                Table(
                    [
                        [
                            col,
                            f"{null_stats.get(col, {}).get('null_pct', 0.0):.2f}%",
                            (
                                f"{null_stats[col]['variance']:.6g}"
                                if null_stats.get(col, {}).get("variance") is not None
                                else "n/a"
                            ),
                        ]
                        for col in added
                    ],
                    headers=["feature", "null %", "variance"],
                )
            )
        else:
            current.card.append(
                Markdown("_No new features engineered (no eligible columns)._")
            )

        # Before/after schema sketch.
        current.card.append(Markdown("## Schema (after)"))
        schema = result.get("schema") or {}
        dtypes = schema.get("dtypes") or {}
        current.card.append(
            Table(
                [[col, dtypes.get(col, "")] for col in (schema.get("columns") or [])],
                headers=["column", "dtype"],
            )
        )

    @step
    def end(self) -> None:
        """Finalize the new data object.

        The engineered artifacts (``train`` / ``test`` / ``schema`` /
        ``fingerprint`` / ``dataset_name`` / ``summary``) are already persisted on
        ``self`` by Metaflow, so they are exposed on ``Run.data`` and this run's
        pathspec (``"FeaturesFlow/<run_id>"``) is a usable ``dataset_ref`` for the
        downstream verbs. Nothing else to do.
        """
        pass


if __name__ == "__main__":
    FeaturesFlow()
