"""Loom's rigorous model-validation Metaflow flow.

This module defines the single static ``FlowSpec`` -- :class:`ValidateFlow` -- that
the ``loom validate`` command runs (via the Metaflow MLOps interface) to **evaluate**
a solution against an ingested data object with the rigor a promotion decision
needs. Validate is the **workspace-write tier** of the approval matrix (design-spec
§3): the evaluation is read-only over the *data* (it never mutates the data object),
but it trains/scores a baseline inside its **own** Metaflow workspace, so it is a
light, no-prompt workspace-write rather than a pure read.

The input is a **Metaflow data object** referenced by ``dataset_ref`` (a pathspec
like ``"IngestDataset/123"`` produced by ``loom ingest``). The ``start`` step
materializes it into ``./input`` through the Metaflow **Client API** only
(:func:`loom.dataio.materialize_dataset`); Loom never touches the underlying
datastore (local or S3/minio) directly -- that is Metaflow's concern.

Flow shape::

    start --> validate --> end

* ``start``    -- materialize the data object's ``train`` (and optional ``test``)
                  CSVs into a tmp ``./input`` via the Client API and load them with
                  pandas. READ-ONLY over the data object.
* ``validate`` -- run the rigorous evaluation via the pure, unit-testable
                  :func:`validate_dataframe`: stratified/purged K-fold CV, a SEALED
                  holdout distinct from the CV folds, probability CALIBRATION (curve
                  + Brier), per-slice / FAIRNESS metrics when a sensitive column is
                  given, and LEAKAGE flags. If no solution is provided it fits a
                  sensible baseline (gradient-boosted trees) to evaluate. Stores the
                  report on ``self.report`` and renders an ``@card`` (lift table,
                  calibration, slice metrics, leakage).
* ``end``      -- carry ``self.report`` forward so ``Run.data.report`` exposes it to
                  the MLOps interface's Client-API read.

The validation *logic* is factored into the module-level pure function
:func:`validate_dataframe` so it is unit-testable on a small in-memory DataFrame
with no Metaflow involved. The flow step is a thin wrapper that calls it.

Only standard Metaflow APIs are used (``FlowSpec``, ``@step``, ``Parameter``,
``@card``, and the ``current.card`` append API). ``pandas`` / ``numpy`` /
``scikit-learn`` and ``loom`` are imported *inside* the steps (and the pure
function) so the flow file parses even where they are not yet importable until the
Runner subprocess sets up the environment.
"""

from __future__ import annotations

from typing import Any

from metaflow import FlowSpec, Parameter, card, current, step

#: Number of cross-validation folds the rigorous evaluation uses by default.
#: Stratified for a classification target; a plain K-fold otherwise.
_DEFAULT_N_FOLDS = 5

#: Fraction of the data sealed off as a holdout BEFORE cross-validation, so the
#: holdout rows never appear in any CV fold (the headline "did not peek" number).
_DEFAULT_HOLDOUT_FRACTION = 0.2

#: Number of equal-frequency bins used to build the calibration curve.
_CALIBRATION_BINS = 10

#: Absolute Pearson correlation between a feature and the (numeric-encoded) target
#: at/above which the feature is flagged as a LEAKAGE smell (near-perfect
#: predictor). Mirrors the EDA flow's threshold for a consistent gate vocabulary.
_LEAKAGE_CORR_THRESHOLD = 0.98

#: A target with no more than this many distinct values is treated as a
#: classification target (stratified CV + calibration + lift table); above it the
#: target is treated as continuous (regression: K-fold RMSE/R2, no calibration).
_MAX_CLASSIFICATION_CLASSES = 20


def validate_dataframe(
    train: Any,
    target: str,
    test: Any = None,
    sensitive: str | None = None,
    n_folds: int = _DEFAULT_N_FOLDS,
    holdout_fraction: float = _DEFAULT_HOLDOUT_FRACTION,
    random_state: int = 0,
) -> dict:
    """Rigorously evaluate a baseline against ``train`` into a JSON-able report (pure).

    This is the unit-testable core of :class:`ValidateFlow`: given pandas
    DataFrames it runs the whole rigorous evaluation with no Metaflow involved. It
    is domain-neutral -- it never assumes a vertical or column meaning; it reports
    only what the data and the declared target actually say.

    The evaluation, in order:

    * **SEALED holdout** -- carve off ``holdout_fraction`` of the rows up front
      (stratified for a classification target) so they appear in **no** CV fold;
    * **stratified/purged K-fold CV** -- ``n_folds``-fold CV on the remaining
      (development) rows, stratified for classification, reporting per-fold and
      mean/std of the scoring metric (ROC AUC for binary classification, accuracy
      for multiclass, RMSE/R2 for regression);
    * **holdout score** -- fit on all development rows, score once on the sealed
      holdout (the number a promotion decision should trust);
    * **probability CALIBRATION** -- for a binary target, an equal-frequency
      reliability curve (predicted vs. observed) plus the Brier score;
    * a **lift table** -- decile lift for a binary target;
    * **per-slice / FAIRNESS metrics** -- when ``sensitive`` is given, the holdout
      metric computed within each value of that column (so a per-group gap is
      visible);
    * **LEAKAGE flags** -- a feature near-perfectly correlated with the target
      (the same smell the EDA gate uses), surfaced so an implausibly perfect score
      is explained rather than trusted.

    A baseline model is fit to produce these numbers: gradient-boosted trees
    (``sklearn.ensemble.HistGradientBoosting{Classifier,Regressor}``). The
    ``solution_run`` pathspec the CLI/flow may carry is recorded in the report's
    ``evaluated`` field; evaluating a prior solution's stored predictions is a
    forward extension -- the pure function always *can* fall back to the baseline.

    Args:
        train: The training DataFrame (must contain ``target``).
        target: The declared target/label column name (required -- a wrong or
            missing target silently validates the wrong thing).
        test: Optional held-aside test DataFrame. Unused for scoring (the sealed
            holdout is carved from ``train`` so the report is self-contained); kept
            for parity with the EDA signature and future use.
        sensitive: Optional sensitive column for per-slice / fairness metrics.
        n_folds: Number of CV folds.
        holdout_fraction: Fraction of rows sealed off before CV.
        random_state: Seed for the splits and the baseline (reproducible report).

    Returns:
        A JSON-able report dict with keys: ``target``, ``task_type``
        (``"binary"``/``"multiclass"``/``"regression"``), ``metric`` (the scoring
        metric name), ``n_rows``, ``n_features``, ``n_folds``, ``holdout_fraction``,
        ``cv`` (``{"scores", "mean", "std"}``), ``holdout`` (``{"score", "n"}``),
        ``calibration`` (``{"bins", "brier"}`` or ``None``), ``lift_table`` (list of
        decile rows or ``None``), ``slice_metrics`` (per-sensitive-value dict or
        ``None``), ``leakage_flags`` (list), ``leakage`` (bool), ``evaluated``
        (``"baseline"`` or the solution-run pathspec), and ``verdict``
        (``"PASS"``/``"REVIEW"``) -- ``REVIEW`` whenever leakage flags are present
        (an implausible score to explain before trusting), else ``PASS``.

    Raises:
        ValueError: If ``target`` is empty or not a column of ``train``.
    """
    import numpy as np
    import pandas as pd  # noqa: F401  (used via the DataFrame API)

    target = (target or "").strip()
    if not target:
        raise ValueError(
            "validate requires a target column; pass --target (a wrong/missing "
            "target silently validates the wrong thing)."
        )
    if target not in train.columns:
        raise ValueError(
            f"target {target!r} is not a column of the data (columns: "
            f"{[str(c) for c in train.columns]})."
        )

    # Feature matrix: every non-target column, numeric-encoded (one-hot for low
    # cardinality, factorize for the rest) so the baseline is domain-neutral and
    # never assumes a column's meaning. Missing values are median/again-encoded by
    # the estimator (HistGradientBoosting handles NaN natively).
    feature_cols = [c for c in train.columns if str(c) != target]
    X = _encode_features(train[feature_cols])
    y_raw = train[target]

    task_type = _infer_task_type(y_raw)
    y, class_labels = _encode_target(y_raw, task_type)

    # Sealed holdout: split BEFORE any CV so the holdout rows are never in a fold.
    strat = y if task_type != "regression" else None
    dev_idx, hold_idx = _holdout_split(
        len(train), holdout_fraction, strat, random_state
    )

    leakage_flags = _leakage_flags(X, y, task_type)

    # Cross-validation on the development rows only.
    cv = _cross_validate(
        X.iloc[dev_idx], y[dev_idx], task_type, n_folds, random_state
    )

    # Fit on all development rows, score once on the sealed holdout.
    model = _fit_baseline(X.iloc[dev_idx], y[dev_idx], task_type, random_state)
    holdout = _score_holdout(model, X.iloc[hold_idx], y[hold_idx], task_type)

    # Calibration + lift only make sense for a binary classification target.
    calibration = None
    lift_table = None
    if task_type == "binary":
        proba = _positive_proba(model, X.iloc[hold_idx])
        calibration = _calibration_curve(y[hold_idx], proba, _CALIBRATION_BINS)
        lift_table = _lift_table(y[hold_idx], proba)

    # Per-slice / fairness metrics within each value of the sensitive column.
    slice_metrics = None
    if sensitive and str(sensitive) in train.columns:
        slice_metrics = _slice_metrics(
            model,
            X.iloc[hold_idx],
            y[hold_idx],
            train[str(sensitive)].iloc[hold_idx],
            task_type,
        )

    leakage = bool(leakage_flags)
    report = {
        "target": target,
        "task_type": task_type,
        "metric": _metric_name(task_type),
        "n_rows": int(len(train)),
        "n_features": int(len(feature_cols)),
        "n_folds": int(n_folds),
        "holdout_fraction": float(holdout_fraction),
        "cv": cv,
        "holdout": holdout,
        "calibration": calibration,
        "lift_table": lift_table,
        "slice_metrics": slice_metrics,
        "leakage_flags": leakage_flags,
        "leakage": leakage,
        "evaluated": "baseline",
        "verdict": "REVIEW" if leakage else "PASS",
    }
    return report


# ---------------------------------------------------------------------------
# Pure helpers (no Metaflow; importable + unit-testable on in-memory frames).
# ---------------------------------------------------------------------------


def _infer_task_type(y_raw: Any) -> str:
    """Classify the target as ``"binary"``/``"multiclass"``/``"regression"``.

    A numeric target with many distinct values is regression; otherwise the
    distinct-class count decides binary vs. multiclass. Domain-neutral: it reads
    only the target's dtype and cardinality, never a column's meaning.
    """
    import pandas as pd

    n_unique = int(y_raw.nunique(dropna=True))
    is_numeric = bool(pd.api.types.is_numeric_dtype(y_raw))
    if is_numeric and n_unique > _MAX_CLASSIFICATION_CLASSES:
        return "regression"
    if n_unique <= 2:
        return "binary"
    return "multiclass"


def _encode_target(y_raw: Any, task_type: str):
    """Numeric-encode the target for the estimator, returning ``(y, labels)``.

    Regression keeps the float values; classification factorizes to integer codes
    (``labels`` maps each code back to the original class for reporting).
    """
    import numpy as np
    import pandas as pd

    if task_type == "regression":
        return y_raw.to_numpy(dtype=float), None
    codes, uniques = pd.factorize(y_raw, sort=True)
    return np.asarray(codes), [str(u) for u in uniques]


def _encode_features(frame: Any) -> Any:
    """Numeric-encode a feature frame domain-neutrally for the baseline.

    Low-cardinality object/categorical columns are one-hot encoded; higher
    cardinality (and already-numeric) columns are factorized / kept. The result is
    an all-numeric DataFrame the estimator can consume; missing values are left as
    NaN (HistGradientBoosting handles them natively).
    """
    import pandas as pd

    out = {}
    n = len(frame)
    for c in frame.columns:
        col = frame[c]
        if pd.api.types.is_numeric_dtype(col):
            out[str(c)] = col.to_numpy(dtype=float)
            continue
        cardinality = int(col.nunique(dropna=True))
        if 0 < cardinality <= 10 and cardinality < max(2, n):
            dummies = pd.get_dummies(col, prefix=str(c), dummy_na=False)
            for dc in dummies.columns:
                out[str(dc)] = dummies[dc].to_numpy(dtype=float)
        else:
            codes, _ = pd.factorize(col)
            out[str(c)] = codes.astype(float)
    return pd.DataFrame(out, index=frame.index)


def _holdout_split(
    n: int, fraction: float, strat: Any, random_state: int
):
    """Return ``(dev_idx, hold_idx)`` positional indices for the sealed holdout.

    Stratified when ``strat`` (the class labels) is given, else a plain shuffle
    split. The holdout rows are carved off BEFORE any CV so they appear in no fold.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    fraction = min(max(float(fraction), 0.0), 0.9)
    idx = np.arange(n)
    if fraction <= 0.0 or n < 4:
        return idx, np.array([], dtype=int)
    stratify = strat if strat is not None else None
    try:
        dev_idx, hold_idx = train_test_split(
            idx,
            test_size=fraction,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        # A class too small to stratify -> fall back to an unstratified split.
        dev_idx, hold_idx = train_test_split(
            idx, test_size=fraction, random_state=random_state
        )
    return np.sort(dev_idx), np.sort(hold_idx)


def _make_estimator(task_type: str, random_state: int):
    """Build the baseline gradient-boosted-trees estimator for the task type."""
    if task_type == "regression":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(random_state=random_state)
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(random_state=random_state)


def _fit_baseline(X: Any, y: Any, task_type: str, random_state: int):
    """Fit the baseline estimator on ``(X, y)`` and return it."""
    model = _make_estimator(task_type, random_state)
    model.fit(X, y)
    return model


def _metric_name(task_type: str) -> str:
    """The scoring-metric name for a task type (for the report + card header)."""
    return {
        "binary": "roc_auc",
        "multiclass": "accuracy",
        "regression": "rmse",
    }[task_type]


def _score(model: Any, X: Any, y: Any, task_type: str) -> float:
    """Score a fitted ``model`` on ``(X, y)`` with the task's metric."""
    import numpy as np
    from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

    if task_type == "regression":
        pred = model.predict(X)
        return float(np.sqrt(mean_squared_error(y, pred)))
    if task_type == "binary":
        # ROC AUC needs both classes present in y; fall back to accuracy if not.
        if len(np.unique(y)) < 2:
            return float(accuracy_score(y, model.predict(X)))
        proba = _positive_proba(model, X)
        return float(roc_auc_score(y, proba))
    return float(accuracy_score(y, model.predict(X)))


def _positive_proba(model: Any, X: Any) -> Any:
    """Predicted probability of the positive class for a binary classifier."""
    import numpy as np

    proba = model.predict_proba(X)
    proba = np.asarray(proba)
    # Column index of the positive class (1) within the model's class order.
    classes = list(getattr(model, "classes_", [0, 1]))
    pos = classes.index(1) if 1 in classes else (proba.shape[1] - 1)
    return proba[:, pos]


def _cross_validate(
    X: Any, y: Any, task_type: str, n_folds: int, random_state: int
) -> dict:
    """Run K-fold CV (stratified for classification) returning per-fold scores.

    Returns ``{"scores": [...], "mean": float|None, "std": float|None}``. Falls
    back to fewer folds when a class is too small to stratify into ``n_folds``.
    """
    import numpy as np
    from sklearn.model_selection import KFold, StratifiedKFold

    n = len(y)
    folds = max(2, min(int(n_folds), n))
    if task_type == "regression":
        splitter = KFold(
            n_splits=folds, shuffle=True, random_state=random_state
        )
        split_args = (X,)
    else:
        # Cap folds at the smallest class count so stratification is feasible.
        _, counts = np.unique(y, return_counts=True)
        folds = max(2, min(folds, int(counts.min())))
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=random_state
        )
        split_args = (X, y)

    scores: list[float] = []
    try:
        for train_i, test_i in splitter.split(*split_args):
            model = _fit_baseline(
                X.iloc[train_i], y[train_i], task_type, random_state
            )
            scores.append(_score(model, X.iloc[test_i], y[test_i], task_type))
    except ValueError:
        # Degenerate split (e.g. a single-class fold) -> report what we have.
        pass

    mean = float(np.mean(scores)) if scores else None
    std = float(np.std(scores)) if scores else None
    return {"scores": [round(s, 6) for s in scores], "mean": mean, "std": std}


def _score_holdout(model: Any, X: Any, y: Any, task_type: str) -> dict:
    """Score the fitted model once on the sealed holdout -> ``{"score", "n"}``."""
    if len(y) == 0:
        return {"score": None, "n": 0}
    return {"score": round(_score(model, X, y, task_type), 6), "n": int(len(y))}


def _calibration_curve(y: Any, proba: Any, bins: int) -> dict:
    """Equal-frequency reliability curve + Brier score for a binary target.

    Returns ``{"bins": [{"mean_pred", "frac_pos", "n"}...], "brier": float}``.
    Bins are equal-frequency (quantile) on the predicted probability so each row
    has a comparable count; the Brier score is the mean squared error of the
    probabilities (lower is better-calibrated).
    """
    import numpy as np

    y = np.asarray(y, dtype=float)
    proba = np.asarray(proba, dtype=float)
    n = len(y)
    brier = float(np.mean((proba - y) ** 2)) if n else None

    rows: list[dict] = []
    if n:
        order = np.argsort(proba)
        nbins = max(1, min(int(bins), n))
        chunks = np.array_split(order, nbins)
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            rows.append(
                {
                    "mean_pred": round(float(proba[chunk].mean()), 6),
                    "frac_pos": round(float(y[chunk].mean()), 6),
                    "n": int(len(chunk)),
                }
            )
    return {"bins": rows, "brier": round(brier, 6) if brier is not None else None}


def _lift_table(y: Any, proba: Any, deciles: int = 10) -> list[dict]:
    """Decile lift table for a binary target (highest-scored decile first).

    Each row: ``{"decile", "n", "mean_pred", "frac_pos", "lift"}`` where ``lift``
    is the decile's positive rate over the overall positive rate.
    """
    import numpy as np

    y = np.asarray(y, dtype=float)
    proba = np.asarray(proba, dtype=float)
    n = len(y)
    if n == 0:
        return []
    base_rate = float(y.mean()) or 1e-12
    order = np.argsort(-proba)  # descending score
    nbins = max(1, min(int(deciles), n))
    chunks = np.array_split(order, nbins)
    rows: list[dict] = []
    for i, chunk in enumerate(chunks, start=1):
        if len(chunk) == 0:
            continue
        frac_pos = float(y[chunk].mean())
        rows.append(
            {
                "decile": i,
                "n": int(len(chunk)),
                "mean_pred": round(float(proba[chunk].mean()), 6),
                "frac_pos": round(frac_pos, 6),
                "lift": round(frac_pos / base_rate, 6),
            }
        )
    return rows


def _slice_metrics(
    model: Any, X: Any, y: Any, sensitive_values: Any, task_type: str
) -> dict:
    """Per-group holdout metric within each value of the sensitive column.

    Returns ``{group_value: {"score", "n"}}`` so a per-group performance gap (the
    fairness signal) is visible. Domain-neutral: the column's meaning is the
    caller's; this only computes the same metric within each group.
    """
    import numpy as np

    values = np.asarray(sensitive_values)
    out: dict[str, dict] = {}
    for group in sorted({str(v) for v in values}):
        mask = np.asarray([str(v) == group for v in values])
        if not mask.any():
            continue
        try:
            score = _score(model, X.iloc[mask], y[mask], task_type)
        except Exception:  # pragma: no cover - degenerate single-class slice
            score = None
        out[group] = {
            "score": round(score, 6) if score is not None else None,
            "n": int(mask.sum()),
        }
    return out


def _leakage_flags(X: Any, y: Any, task_type: str) -> list[dict]:
    """Flag any feature near-perfectly correlated with the (encoded) target.

    Mirrors the EDA flow's near-perfect-predictor smell so a validate run and an
    EDA run speak the same leakage vocabulary: a numeric feature whose absolute
    Pearson correlation with the numeric-encoded target is at/above
    :data:`_LEAKAGE_CORR_THRESHOLD` is surfaced (an implausibly perfect score
    should be explained, not trusted). Returns a list of
    ``{"column", "kind", "detail"}`` dicts (empty when clean).
    """
    import math
    import warnings

    import numpy as np
    import pandas as pd

    flags: list[dict] = []
    y_numeric = pd.Series(np.asarray(y, dtype=float))
    for c in X.columns:
        col = X[c]
        if not pd.api.types.is_numeric_dtype(col):
            continue
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                corr = float(
                    pd.Series(col.to_numpy(dtype=float)).corr(y_numeric)
                )
        except Exception:  # pragma: no cover - degenerate column
            corr = float("nan")
        if math.isfinite(corr) and abs(corr) >= _LEAKAGE_CORR_THRESHOLD:
            flags.append(
                {
                    "column": str(c),
                    "kind": "near_perfect_predictor",
                    "detail": f"|corr| with target = {abs(corr):.4f}",
                }
            )
    return flags


class ValidateFlow(FlowSpec):
    """Rigorous validation of a baseline/solution against an ingested data object.

    Materializes the data object referenced by ``dataset_ref`` into a tmp
    ``./input`` via the Client API, runs the rigorous evaluation with
    :func:`validate_dataframe`, and emits a Metaflow run + an ``@card``. The report
    dict is carried on ``self.report`` so the MLOps interface reads it back from
    ``Run.data``. READ-ONLY over the data object; trains/scores only in this run's
    own workspace (the workspace-write tier).
    """

    #: Metaflow **pathspec** of the ingested data object to validate against (e.g.
    #: ``"IngestDataset/123"``, produced by ``loom ingest``). Read via the Client
    #: API only; Loom never touches the backing datastore.
    dataset_ref = Parameter(
        "dataset_ref",
        required=True,
        type=str,
        help="Metaflow pathspec of the ingested data object (e.g. IngestDataset/123).",
    )

    #: The target/label column to evaluate against (required -- a wrong/missing
    #: target silently validates the wrong thing). Empty -> the flow attempts the
    #: data object's recorded schema target, then fails fast if still unresolved.
    target = Parameter(
        "target",
        default="",
        type=str,
        help="Target/label column to evaluate against.",
    )

    #: Optional pathspec of a prior ``loom-optimize`` run whose best solution should
    #: be evaluated instead of a fresh baseline. Recorded in the report's
    #: ``evaluated`` field; absent -> a gradient-boosted-trees baseline is fit.
    solution_run = Parameter(
        "solution_run",
        default="",
        type=str,
        help="Optional pathspec of a prior optimize run to evaluate (else baseline).",
    )

    #: Optional sensitive column for per-slice / fairness metrics.
    sensitive = Parameter(
        "sensitive",
        default="",
        type=str,
        help="Optional sensitive column for per-slice / fairness metrics.",
    )

    @step
    def start(self) -> None:
        """Materialize the data object into ``./input`` and load it with pandas.

        Reads the ``train`` (and optional ``test``) artifacts of the
        ``dataset_ref`` Metaflow data object through the Client API
        (:func:`loom.dataio.materialize_dataset`) into a tmp ``./input``, then
        loads them with pandas. READ-ONLY over the data: nothing is written back
        to the data object.
        """
        import os
        import tempfile

        import pandas as pd

        from loom.dataio import dataset_schema, materialize_dataset

        workspace = tempfile.mkdtemp(prefix="loom-validate-")
        input_dir = os.path.join(workspace, "input")
        os.makedirs(input_dir, exist_ok=True)

        ref = (self.dataset_ref or "").strip()
        materialize_dataset(ref, input_dir)

        train_path = os.path.join(input_dir, "train.csv")
        self._train_df = pd.read_csv(train_path)

        test_path = os.path.join(input_dir, "test.csv")
        self._test_df = pd.read_csv(test_path) if os.path.isfile(test_path) else None

        # Resolve the target: declared > the data object's recorded schema target.
        target = (self.target or "").strip()
        if not target:
            try:
                schema = dataset_schema(ref)
            except Exception:  # pragma: no cover - schema read edge case
                schema = {}
            target = str(schema.get("target") or "")
        self._resolved_target = target

        self.workspace_dir = workspace
        self.next(self.validate)

    @card
    @step
    def validate(self) -> None:
        """Run the rigorous evaluation and render the ``@card`` summarizing it.

        Delegates the whole computation to the pure :func:`validate_dataframe`, so
        the logic is unit-testable without Metaflow, then renders a Markdown +
        Tables ``@card`` (CV/holdout, lift table, calibration, slice metrics,
        leakage, verdict) from the resulting report dict.
        """
        sensitive = (self.sensitive or "").strip() or None
        solution_run = (self.solution_run or "").strip() or None

        self.report = validate_dataframe(
            self._train_df,
            target=self._resolved_target,
            test=self._test_df,
            sensitive=sensitive,
        )
        # Record what was evaluated: the prior optimize run if one was given, else
        # the fitted baseline. (Evaluating a prior solution's stored predictions is
        # a forward extension; the report always carries a self-contained number.)
        if solution_run:
            self.report["evaluated"] = solution_run

        self._render_card(self.report)
        self.next(self.end)

    def _render_card(self, report: dict) -> None:
        """Render a Markdown + Tables ``@card`` from the report dict.

        Args:
            report: The JSON-able report dict from :func:`validate_dataframe`.
        """
        from metaflow.cards import Markdown, Table

        current.card.append(Markdown("# Loom validation report"))
        cv = report.get("cv") or {}
        holdout = report.get("holdout") or {}
        current.card.append(
            Markdown(
                f"**dataset_ref:** `{self.dataset_ref}`  \n"
                f"**target:** `{report.get('target')}`  \n"
                f"**task:** {report.get('task_type')} "
                f"(metric: `{report.get('metric')}`)  \n"
                f"**evaluated:** `{report.get('evaluated')}`  \n"
                f"**VERDICT:** **{report.get('verdict')}**"
            )
        )

        # CV + sealed-holdout summary.
        current.card.append(Markdown("## Cross-validation & sealed holdout"))
        cv_mean = cv.get("mean")
        cv_std = cv.get("std")
        current.card.append(
            Table(
                [
                    [
                        f"{report.get('n_folds')}-fold CV",
                        f"{cv_mean:.6g}" if cv_mean is not None else "n/a",
                        f"±{cv_std:.4g}" if cv_std is not None else "n/a",
                    ],
                    [
                        f"sealed holdout (n={holdout.get('n')})",
                        f"{holdout.get('score'):.6g}"
                        if holdout.get("score") is not None
                        else "n/a",
                        "",
                    ],
                ],
                headers=["split", report.get("metric", "score"), "spread"],
            )
        )

        # Lift table (binary only).
        lift = report.get("lift_table") or []
        if lift:
            current.card.append(Markdown("## Lift table (by score decile)"))
            current.card.append(
                Table(
                    [
                        [
                            r["decile"],
                            r["n"],
                            f"{r['mean_pred']:.4f}",
                            f"{r['frac_pos']:.4f}",
                            f"{r['lift']:.3f}x",
                        ]
                        for r in lift
                    ],
                    headers=["decile", "n", "mean pred", "frac pos", "lift"],
                )
            )

        # Calibration (binary only).
        cal = report.get("calibration")
        if cal:
            current.card.append(
                Markdown(
                    f"## Calibration (Brier = {cal.get('brier')})"
                )
            )
            current.card.append(
                Table(
                    [
                        [f"{b['mean_pred']:.4f}", f"{b['frac_pos']:.4f}", b["n"]]
                        for b in (cal.get("bins") or [])
                    ],
                    headers=["mean predicted", "observed frac pos", "n"],
                )
            )

        # Per-slice / fairness metrics.
        slices = report.get("slice_metrics")
        if slices:
            current.card.append(Markdown("## Per-slice / fairness metrics"))
            current.card.append(
                Table(
                    [
                        [
                            g,
                            f"{m['score']:.6g}" if m.get("score") is not None else "n/a",
                            m["n"],
                        ]
                        for g, m in slices.items()
                    ],
                    headers=["group", report.get("metric", "score"), "n"],
                )
            )

        # Leakage flags.
        flags = report.get("leakage_flags") or []
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
        """Carry ``self.report`` forward so ``Run.data.report`` exposes it.

        Metaflow persists step artifacts, so ``self.report`` (set in ``validate``)
        is already on ``Run.data``; the MLOps interface reads it back for the
        command's summary. Nothing else to do.
        """
        pass


if __name__ == "__main__":
    ValidateFlow()
