"""Tests for the pipeline verb: the pure stage-orchestration logic + CLI arg-parsing.

The cross-stage gating *logic* is factored out of :class:`flows.pipeline.PipelineFlow`
into the module-level pure function :func:`flows.pipeline.orchestrate_stages`, so the
stage-gate ordering is unit-testable on **stub stage results** with no Metaflow,
pandas, or sklearn involved. These tests pin:

* the happy path (all four stages ran, clean profile) -> PASS;
* leakage handled (profile flagged + dropped) still allows features and PASSes;
* leakage NOT droppable BLOCKS features and FAILs the run;
* a missing/failed earlier stage blocks every later stage (gate ordering);
* a sub-threshold (higher-is-better) validate metric is downgraded to FAIL;
* a regression (lower-is-better) verdict carries through un-thresholded;
* the gate-decision trail is ordered ``profile -> features -> optimize -> validate``.

These are pure-Python (no pandas/sklearn/Metaflow). The CLI arg-parse tests exercise
only the argparse wiring for ``loom pipeline``.
"""

from __future__ import annotations

import pytest

from loom.cli import _build_parser


# ---------------------------------------------------------------------------
# Pure stage-orchestration logic (no Metaflow).
# ---------------------------------------------------------------------------


def _all_ran(verdict: str = "PASS", metric: float = 0.9) -> dict:
    """Stub stage results for a run where all four stages ran on clean data."""
    return {
        "profile": {"leakage": False, "dropped_columns": [], "verdict": "PASS"},
        "features": {"fingerprint": "sha256:abc", "dataset_ref": "PF/x", "verdict": "BUILT"},
        "optimize": {"candidate": "baseline-on-features", "baseline_ok": True, "verdict": "PASS"},
        "validate": {
            "verdict": verdict,
            "metric": metric,
            "higher_is_better": True,
            "task_type": "binary",
        },
    }


def test_orchestrate_happy_path_passes() -> None:
    """All four stages ran on clean, above-threshold data -> PASS, no failed stage."""
    from flows.pipeline import orchestrate_stages

    out = orchestrate_stages(_all_ran())
    assert out["verdict"] == "PASS"
    assert out["failed_stage"] is None
    assert all(s["status"] == "ran" for s in out["stages"].values())
    assert all(s["gate_passed"] for s in out["stages"].values())


def test_orchestrate_handled_leakage_still_passes() -> None:
    """Profile flagged leakage but dropped the columns -> features allowed, PASS."""
    from flows.pipeline import orchestrate_stages

    results = _all_ran()
    results["profile"] = {
        "leakage": True,
        "dropped_columns": ["leaky"],
        "verdict": "REVIEW",
    }
    out = orchestrate_stages(results)
    assert out["leakage"] is True  # leakage was present and handled
    assert out["stages"]["features"]["status"] == "ran"
    assert out["stages"]["features"]["gate_passed"] is True
    assert out["verdict"] == "PASS"


def test_orchestrate_undroppable_leakage_blocks_features() -> None:
    """Profile flagged leakage with NO columns dropped -> features BLOCKED, FAIL."""
    from flows.pipeline import orchestrate_stages

    results = _all_ran()
    results["profile"] = {"leakage": True, "dropped_columns": [], "verdict": "REVIEW"}
    out = orchestrate_stages(results)
    assert out["stages"]["features"]["status"] == "blocked"
    assert out["stages"]["features"]["gate_passed"] is False
    assert out["failed_stage"] == "features"
    assert out["verdict"] == "FAIL"


def test_orchestrate_blocks_later_stages_when_features_missing() -> None:
    """A missing features stage blocks optimize + validate (gate ordering)."""
    from flows.pipeline import orchestrate_stages

    results = _all_ran()
    results.pop("features")  # features did not run / produced nothing
    out = orchestrate_stages(results)
    # features 'passed' its gate but did not run -> the chain stops there.
    assert out["stages"]["features"]["status"] == "skipped"
    assert out["stages"]["optimize"]["status"] == "blocked"
    assert out["stages"]["validate"]["status"] == "blocked"
    assert out["verdict"] == "FAIL"


def test_orchestrate_sub_threshold_metric_downgraded_to_fail() -> None:
    """A higher-is-better holdout below the threshold is downgraded to FAIL."""
    from flows.pipeline import orchestrate_stages

    out = orchestrate_stages(_all_ran(verdict="PASS", metric=0.40), threshold=0.5)
    assert out["verdict"] == "FAIL"


def test_orchestrate_regression_verdict_not_thresholded() -> None:
    """A lower-is-better (regression) metric is NOT thresholded; its verdict carries through."""
    from flows.pipeline import orchestrate_stages

    results = _all_ran()
    results["validate"] = {
        "verdict": "PASS",
        "metric": 0.01,  # tiny RMSE -> would FAIL a naive higher-is-better threshold
        "higher_is_better": False,
        "task_type": "regression",
    }
    out = orchestrate_stages(results, threshold=0.5)
    assert out["verdict"] == "PASS"


def test_orchestrate_gate_decisions_are_ordered() -> None:
    """The gate-decision trail is in the declared lifecycle order."""
    from flows.pipeline import orchestrate_stages

    out = orchestrate_stages(_all_ran())
    stages = [d["stage"] for d in out["gate_decisions"]]
    assert stages == ["profile", "features", "optimize", "validate"]


def test_orchestrate_empty_results_fail() -> None:
    """No stage results at all -> FAIL (validate never ran)."""
    from flows.pipeline import orchestrate_stages

    out = orchestrate_stages({})
    assert out["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# CLI arg-parsing (pure-Python, no pandas/sklearn/Metaflow).
# ---------------------------------------------------------------------------


def test_cli_pipeline_parses_all_flags() -> None:
    """`loom pipeline` parses dataset/goal/target into the handler."""
    from loom.cli import _cmd_pipeline

    parser = _build_parser()
    args = parser.parse_args(
        [
            "pipeline",
            "--dataset",
            "IngestDataset/123",
            "--goal",
            "predict churn",
            "--target",
            "y",
        ]
    )
    assert args.command == "pipeline"
    assert args.dataset == "IngestDataset/123"
    assert args.goal == "predict churn"
    assert args.target == "y"
    assert args.func is _cmd_pipeline


def test_cli_pipeline_target_defaults_none() -> None:
    """target defaults to None (inferred from the data object schema)."""
    parser = _build_parser()
    args = parser.parse_args(
        ["pipeline", "--dataset", "IngestDataset/9", "--goal", "g"]
    )
    assert args.target is None


def test_cli_pipeline_requires_dataset_and_goal() -> None:
    """`loom pipeline` requires both --dataset and --goal."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["pipeline", "--dataset", "IngestDataset/1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["pipeline", "--goal", "g"])
