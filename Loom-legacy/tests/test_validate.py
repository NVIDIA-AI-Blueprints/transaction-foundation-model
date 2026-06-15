"""Tests for the validate verb: the pure validation logic + CLI arg-parsing.

The validation *logic* is factored out of :class:`flows.validate.ValidateFlow` into
the module-level pure function :func:`flows.validate.validate_dataframe` (and its
helpers), so it is unit-testable on a small in-memory DataFrame with **no Metaflow
involved**. These tests pin:

* task-type inference (binary / multiclass / regression);
* the sealed holdout being distinct from the CV folds (no overlap);
* a sane CV mean and a holdout score on a learnable signal;
* probability calibration (Brier + an equal-frequency curve) for a binary target;
* per-slice / fairness metrics when a sensitive column is given;
* the leakage gate flagging a near-perfect predictor and the verdict gating
  (``REVIEW`` on leakage, else ``PASS``) -- the executable self-test for the
  composition exit gate.

``pandas`` + ``scikit-learn`` are required for the validation tests
(``importorskip``). The CLI arg-parse tests are pure-Python (no pandas/sklearn):
they only exercise the argparse wiring for ``loom validate``.
"""

from __future__ import annotations

import json

import pytest

from loom.cli import _build_parser


# ---------------------------------------------------------------------------
# Pure validation logic (no Metaflow).
# ---------------------------------------------------------------------------

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")


def _binary_frame(n: int = 300, seed: int = 0) -> "pd.DataFrame":
    """A small learnable binary-classification frame with a sensitive column."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logit = 1.5 * x1 - 0.8 * x2
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)
    g = rng.choice(["a", "b"], size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "g": g, "target": y})


def test_validate_requires_a_target() -> None:
    """An empty/missing target is refused (a wrong target validates the wrong thing)."""
    from flows.validate import validate_dataframe

    df = _binary_frame(n=40)
    with pytest.raises(ValueError):
        validate_dataframe(df, target="")
    with pytest.raises(ValueError):
        validate_dataframe(df, target="nonexistent_col")


def test_validate_infers_binary_task_and_metric() -> None:
    """A two-class target is validated as binary with ROC AUC."""
    from flows.validate import validate_dataframe

    rep = validate_dataframe(_binary_frame(), target="target")
    assert rep["task_type"] == "binary"
    assert rep["metric"] == "roc_auc"
    assert rep["target"] == "target"
    assert rep["n_rows"] == 300


def test_validate_infers_regression_task() -> None:
    """A continuous numeric target is validated as regression (RMSE, no calibration)."""
    from flows.validate import validate_dataframe

    rng = np.random.default_rng(1)
    n = 300
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    df = pd.DataFrame({"a": a, "b": b, "target": 3.0 * a + b + rng.normal(size=n)})
    rep = validate_dataframe(df, target="target")
    assert rep["task_type"] == "regression"
    assert rep["metric"] == "rmse"
    # Calibration / lift are classification-only.
    assert rep["calibration"] is None
    assert rep["lift_table"] is None


def test_validate_infers_multiclass_task() -> None:
    """A 3+ class non-numeric target is validated as multiclass with accuracy."""
    from flows.validate import validate_dataframe

    rng = np.random.default_rng(2)
    n = 180
    x = rng.normal(size=n)
    y = pd.Series(rng.integers(0, 3, n)).map({0: "low", 1: "mid", 2: "high"})
    df = pd.DataFrame({"x": x, "noise": rng.normal(size=n), "target": y})
    rep = validate_dataframe(df, target="target")
    assert rep["task_type"] == "multiclass"
    assert rep["metric"] == "accuracy"


def test_validate_holdout_is_disjoint_from_cv_folds() -> None:
    """The sealed holdout indices never overlap the development (CV) indices."""
    from flows.validate import _holdout_split

    n = 200
    y = (np.arange(n) % 2)
    dev_idx, hold_idx = _holdout_split(n, 0.2, y, random_state=0)
    # Disjoint and exhaustive: dev + holdout partition all rows, no overlap.
    assert set(dev_idx).isdisjoint(set(hold_idx))
    assert sorted([*dev_idx, *hold_idx]) == list(range(n))
    # The holdout is ~20% of the rows.
    assert abs(len(hold_idx) - int(0.2 * n)) <= 1


def test_validate_reports_cv_and_holdout_scores() -> None:
    """CV reports per-fold scores + mean/std and the sealed holdout scores once."""
    from flows.validate import validate_dataframe

    rep = validate_dataframe(_binary_frame(), target="target")
    cv = rep["cv"]
    assert len(cv["scores"]) == rep["n_folds"]
    assert cv["mean"] is not None and cv["std"] is not None
    # A learnable signal -> better-than-chance AUC on the held-out folds.
    assert cv["mean"] > 0.6
    # The sealed holdout was scored on its own rows.
    assert rep["holdout"]["n"] > 0
    assert rep["holdout"]["score"] is not None


def test_validate_calibration_for_binary_target() -> None:
    """A binary target yields a Brier score + an equal-frequency reliability curve."""
    from flows.validate import validate_dataframe

    rep = validate_dataframe(_binary_frame(), target="target")
    cal = rep["calibration"]
    assert cal is not None
    assert cal["brier"] is not None and 0.0 <= cal["brier"] <= 1.0
    assert cal["bins"], "expected calibration bins"
    # Each bin carries a mean predicted prob, an observed positive frac, and a count.
    for b in cal["bins"]:
        assert set(b) == {"mean_pred", "frac_pos", "n"}
        assert b["n"] >= 1


def test_validate_lift_table_present_for_binary() -> None:
    """A binary target yields a decile lift table, top decile lift >= 1 on signal."""
    from flows.validate import validate_dataframe

    rep = validate_dataframe(_binary_frame(), target="target")
    lift = rep["lift_table"]
    assert lift and len(lift) <= 10
    # The highest-scored decile should out-perform the base rate on a real signal.
    assert lift[0]["lift"] >= 1.0


def test_validate_slice_metrics_when_sensitive_given() -> None:
    """A sensitive column yields a per-group holdout metric for each of its values."""
    from flows.validate import validate_dataframe

    rep = validate_dataframe(_binary_frame(), target="target", sensitive="g")
    slices = rep["slice_metrics"]
    assert slices is not None
    assert set(slices) == {"a", "b"}
    for group, m in slices.items():
        assert set(m) == {"score", "n"}
        assert m["n"] >= 1


def test_validate_no_slice_metrics_without_sensitive() -> None:
    """Without a sensitive column, no per-slice metrics are produced."""
    from flows.validate import validate_dataframe

    rep = validate_dataframe(_binary_frame(), target="target")
    assert rep["slice_metrics"] is None


def test_validate_flags_leakage_and_sets_review_verdict() -> None:
    """A near-perfect predictor is flagged and the verdict is REVIEW (the exit gate).

    This is the executable self-test for the composition gate: a sub-trustworthy
    (leaky) validation must NOT silently report PASS -- it must surface the leak
    and gate to REVIEW so a downstream deploy is blocked.
    """
    from flows.validate import validate_dataframe

    df = _binary_frame()
    rng = np.random.default_rng(7)
    # A feature that is essentially the target (a deterministic leak).
    df["leak"] = df["target"].to_numpy(dtype=float) + rng.normal(scale=1e-4, size=len(df))
    rep = validate_dataframe(df, target="target")
    assert rep["leakage"] is True
    kinds = {(f["column"], f["kind"]) for f in rep["leakage_flags"]}
    assert ("leak", "near_perfect_predictor") in kinds
    assert rep["verdict"] == "REVIEW"


def test_validate_clean_data_passes() -> None:
    """Clean data (no leakage) gates to PASS."""
    from flows.validate import validate_dataframe

    rep = validate_dataframe(_binary_frame(), target="target")
    assert rep["leakage"] is False
    assert rep["verdict"] == "PASS"


def test_validate_records_solution_run_in_evaluated() -> None:
    """A given solution_run pathspec is recorded in the report's ``evaluated`` field."""
    from flows.validate import validate_dataframe

    # The pure function defaults to the baseline; the flow overrides ``evaluated``
    # to the solution_run, so here we just confirm the baseline default.
    rep = validate_dataframe(_binary_frame(), target="target")
    assert rep["evaluated"] == "baseline"


def test_validate_report_is_json_able() -> None:
    """The whole report round-trips through JSON (suitable for a RunResult summary)."""
    from flows.validate import validate_dataframe

    rep = validate_dataframe(_binary_frame(), target="target", sensitive="g")
    assert json.loads(json.dumps(rep)) == rep


# ---------------------------------------------------------------------------
# CLI arg-parsing (pure-Python, no pandas/sklearn/Metaflow).
# ---------------------------------------------------------------------------


def test_cli_validate_parses_all_flags() -> None:
    """`loom validate` parses dataset/target/solution/sensitive into the handler."""
    from loom.cli import _cmd_validate

    parser = _build_parser()
    args = parser.parse_args(
        [
            "validate",
            "--dataset",
            "IngestDataset/123",
            "--target",
            "y",
            "--solution",
            "EvalCandidate/3",
            "--sensitive",
            "g",
        ]
    )
    assert args.command == "validate"
    assert args.dataset == "IngestDataset/123"
    assert args.target == "y"
    assert args.solution == "EvalCandidate/3"
    assert args.sensitive == "g"
    assert args.func is _cmd_validate


def test_cli_validate_optional_flags_default_none() -> None:
    """target / solution / sensitive default to None."""
    parser = _build_parser()
    args = parser.parse_args(["validate", "--dataset", "IngestDataset/9"])
    assert args.target is None
    assert args.solution is None
    assert args.sensitive is None


def test_cli_validate_requires_dataset() -> None:
    """`loom validate` without --dataset is a parse error (required argument)."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["validate", "--target", "y"])
