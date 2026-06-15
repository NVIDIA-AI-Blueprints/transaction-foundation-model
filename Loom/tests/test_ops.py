"""Tests for the ops verb: the pure monitoring logic + CLI arg-parsing.

The monitoring *logic* is factored out of :class:`flows.ops.OpsFlow` into the
module-level pure functions :func:`flows.ops.summarize_ops` and
:func:`flows.ops.compute_drift`, so they are unit-testable on small in-memory run
dicts / DataFrames with **no Metaflow involved**. These tests pin:

* run-health rollups -- run/success/failure counts, success rate, the most-recent
  run, recency ordering, the scored leaderboard, and the HEALTHY/DEGRADED/EMPTY
  status;
* the drift smell test -- ``STABLE`` on identical frames, ``DRIFT`` on a shifted
  numeric mean, a null-rate shift, and schema add/remove drift;
* both summaries round-tripping through JSON for the run summary.

``pandas`` is required for the drift tests (``importorskip``). ``summarize_ops`` is
pure-Python (dicts only). The CLI arg-parse tests are pure-Python (no
pandas/Metaflow): they only exercise the argparse wiring for ``loom ops``.
"""

from __future__ import annotations

import json

import pytest

from loom.cli import _build_parser
from flows.ops import summarize_ops


# ---------------------------------------------------------------------------
# Pure run-health summary logic (no Metaflow, no pandas).
# ---------------------------------------------------------------------------


def _runs() -> list[dict]:
    """Three runs: two successful (one scored best) + a failed one, with timestamps."""
    return [
        {
            "pathspec": "ValidateFlow/1",
            "flow": "ValidateFlow",
            "successful": True,
            "metric": 0.80,
            "created_at": "2026-06-01T00:00:00",
            "finished_at": "2026-06-01T00:01:00",
        },
        {
            "pathspec": "ValidateFlow/2",
            "flow": "ValidateFlow",
            "successful": True,
            "metric": 0.91,
            "created_at": "2026-06-02T00:00:00",
            "finished_at": "2026-06-02T00:01:00",
        },
        {
            "pathspec": "ValidateFlow/3",
            "flow": "ValidateFlow",
            "successful": False,
            "metric": None,
            "created_at": "2026-06-03T00:00:00",
            "finished_at": "2026-06-03T00:01:00",
        },
    ]


def test_summarize_ops_counts_and_success_rate() -> None:
    """Run/success/failure counts and the success rate are computed from the runs."""
    health = summarize_ops(_runs(), flow_name="ValidateFlow")
    assert health["flow_name"] == "ValidateFlow"
    assert health["n_runs"] == 3
    assert health["n_successful"] == 2
    assert health["n_failed"] == 1
    assert health["success_rate"] == pytest.approx(2 / 3)


def test_summarize_ops_status_degraded_when_latest_failed() -> None:
    """The most-recent run failing degrades run health to DEGRADED."""
    health = summarize_ops(_runs(), flow_name="ValidateFlow")
    # The latest run (ValidateFlow/3, 2026-06-03) failed.
    assert health["last_run"]["pathspec"] == "ValidateFlow/3"
    assert health["status"] == "DEGRADED"


def test_summarize_ops_status_healthy_when_latest_succeeded() -> None:
    """When the most-recent run succeeded, run health is HEALTHY."""
    runs = _runs()[:2]  # drop the failed (latest) run
    health = summarize_ops(runs, flow_name="ValidateFlow")
    assert health["last_run"]["pathspec"] == "ValidateFlow/2"
    assert health["status"] == "HEALTHY"


def test_summarize_ops_empty() -> None:
    """No runs -> EMPTY status, no success rate, no last run."""
    health = summarize_ops([], flow_name="ValidateFlow")
    assert health["status"] == "EMPTY"
    assert health["n_runs"] == 0
    assert health["success_rate"] is None
    assert health["last_run"] is None


def test_summarize_ops_leaderboard_orders_scored_best_first() -> None:
    """The leaderboard puts the highest-metric scored run first; unscored runs drop off."""
    health = summarize_ops(_runs(), experiment_id="exp1")
    lb = health["leaderboard"]
    assert [r["pathspec"] for r in lb] == ["ValidateFlow/2", "ValidateFlow/1"]
    assert health["experiment_id"] == "exp1"


def test_summarize_ops_recent_is_most_recent_first() -> None:
    """``recent`` is ordered most-recent first by finished/created timestamp."""
    health = summarize_ops(_runs(), flow_name="ValidateFlow")
    recent = [r["pathspec"] for r in health["recent"]]
    assert recent == ["ValidateFlow/3", "ValidateFlow/2", "ValidateFlow/1"]


def test_summarize_ops_is_json_able() -> None:
    """The run-health summary round-trips through JSON (suitable for a summary)."""
    health = summarize_ops(_runs(), flow_name="ValidateFlow")
    assert json.loads(json.dumps(health)) == health


# ---------------------------------------------------------------------------
# Pure drift logic (pandas).
# ---------------------------------------------------------------------------

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def _ref_frame(n: int = 200, seed: int = 0) -> "pd.DataFrame":
    """A reference data-object frame: two numerics + a categorical."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(loc=0.0, scale=1.0, size=n),
            "b": rng.normal(loc=10.0, scale=2.0, size=n),
            "cat": rng.choice(["x", "y"], size=n),
        }
    )


def test_compute_drift_identical_frames_is_stable() -> None:
    """Identical frames report no drift (STABLE), no flags, no schema change."""
    from flows.ops import compute_drift

    ref = _ref_frame()
    drift = compute_drift(ref.copy(), ref.copy())
    assert drift["status"] == "STABLE"
    assert drift["drift"] is False
    assert drift["drift_flags"] == []
    assert drift["added"] == [] and drift["removed"] == []
    assert drift["n_shared_columns"] == 3


def test_compute_drift_flags_numeric_mean_shift() -> None:
    """A clearly shifted numeric mean is flagged as drift (DRIFT)."""
    from flows.ops import compute_drift

    ref = _ref_frame()
    cur = ref.copy()
    # Shift column 'a' mean far beyond the relative threshold.
    cur["a"] = cur["a"] + 5.0
    drift = compute_drift(cur, ref)
    assert drift["status"] == "DRIFT"
    assert drift["drift"] is True
    kinds = {(f["column"], f["kind"]) for f in drift["drift_flags"]}
    assert ("a", "mean_shift") in kinds


def test_compute_drift_flags_null_rate_shift() -> None:
    """A column that started arriving heavily null is flagged as null-rate drift."""
    from flows.ops import compute_drift

    ref = _ref_frame()
    cur = ref.copy()
    # Make ~half of column 'b' null in the current frame.
    cur.loc[cur.index[: len(cur) // 2], "b"] = np.nan
    drift = compute_drift(cur, ref)
    kinds = {(f["column"], f["kind"]) for f in drift["drift_flags"]}
    assert ("b", "null_rate_shift") in kinds
    assert drift["status"] == "DRIFT"


def test_compute_drift_reports_schema_add_remove() -> None:
    """A column present in only one frame is surfaced under added / removed."""
    from flows.ops import compute_drift

    ref = _ref_frame()
    cur = ref.copy()
    cur["new_col"] = 1.0
    cur = cur.drop(columns=["cat"])
    drift = compute_drift(cur, ref)
    assert "new_col" in drift["added"]
    assert "cat" in drift["removed"]
    assert drift["drift"] is True


def test_compute_drift_is_json_able() -> None:
    """The drift summary round-trips through JSON (suitable for a RunResult summary)."""
    from flows.ops import compute_drift

    ref = _ref_frame()
    cur = ref.copy()
    cur["a"] = cur["a"] + 5.0
    drift = compute_drift(cur, ref)
    assert json.loads(json.dumps(drift)) == drift


# ---------------------------------------------------------------------------
# CLI arg-parsing (pure-Python, no pandas/Metaflow).
# ---------------------------------------------------------------------------


def test_cli_ops_parses_flow() -> None:
    """`loom ops --flow NAME` parses into the ops handler."""
    from loom.cli import _cmd_ops

    parser = _build_parser()
    args = parser.parse_args(["ops", "--flow", "ValidateFlow"])
    assert args.command == "ops"
    assert args.flow_name == "ValidateFlow"
    assert args.func is _cmd_ops


def test_cli_ops_parses_experiment() -> None:
    """`loom ops --experiment ID` parses the experiment id."""
    parser = _build_parser()
    args = parser.parse_args(["ops", "--experiment", "exp1"])
    assert args.experiment == "exp1"


def test_cli_ops_parses_drift_pair() -> None:
    """`loom ops --dataset ... --reference ...` parses the drift pair."""
    parser = _build_parser()
    args = parser.parse_args(
        ["ops", "--dataset", "IngestDataset/9", "--reference", "IngestDataset/1"]
    )
    assert args.dataset == "IngestDataset/9"
    assert args.reference == "IngestDataset/1"


def test_cli_ops_inputs_default_none() -> None:
    """With no flags given, all ops inputs default to None (the handler refuses)."""
    parser = _build_parser()
    args = parser.parse_args(["ops"])
    assert args.flow_name is None
    assert args.experiment is None
    assert args.dataset is None
    assert args.reference is None
