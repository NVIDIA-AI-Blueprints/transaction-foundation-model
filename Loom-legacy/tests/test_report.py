"""Tests for the report verb: the pure assembly logic + Client-API gather + CLI.

The report *assembly* is factored out of :class:`flows.report.ReportFlow` into the
module-level pure function :func:`flows.report.assemble_report`, so it is
unit-testable on small in-memory run dicts with **no Metaflow involved**. The
Client-API gather (:func:`flows.report.gather_runs_by_pathspecs`) is tested against
a **fake metaflow module** injected via ``sys.modules`` -- so the suite needs no
live cluster. These tests pin:

* assembly from a mocked Client API (run dicts in, structured report out): the run
  count, success count, best metric + best run, the metric spread, the
  leaderboard ordering, and the ``OK``/``EMPTY`` verdict;
* the compaction of learnings rows into the report's lineage section;
* the pathspec gather projecting a fake ``metaflow.Run`` into a report dict.

The CLI arg-parse tests are pure-Python (no Metaflow): they exercise the argparse
wiring for ``loom report`` (the mutually-exclusive --experiment / --runs).
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from loom.cli import _build_parser


# ---------------------------------------------------------------------------
# Pure assembly logic (no Metaflow).
# ---------------------------------------------------------------------------


def _runs() -> list[dict]:
    """Three runs: two successful (one the clear best) + one failed/unscored."""
    return [
        {
            "pathspec": "EvalCandidate/1",
            "flow": "EvalCandidate",
            "successful": True,
            "metric": 0.91,
            "tags": ["loom_experiment:exp1"],
        },
        {
            "pathspec": "EvalCandidate/2",
            "flow": "EvalCandidate",
            "successful": True,
            "metric": 0.95,
            "tags": ["loom_experiment:exp1"],
        },
        {
            "pathspec": "EvalCandidate/3",
            "flow": "EvalCandidate",
            "successful": False,
            "metric": None,
            "tags": ["loom_experiment:exp1"],
        },
    ]


def test_assemble_counts_and_best() -> None:
    """Run/success counts, best metric, and best run are computed from the runs."""
    from flows.report import assemble_report

    rep = assemble_report("exp1", _runs())
    assert rep["experiment_id"] == "exp1"
    assert rep["n_runs"] == 3
    assert rep["n_successful"] == 2
    assert rep["best_metric"] == 0.95
    assert rep["best_run"] == "EvalCandidate/2"
    assert rep["verdict"] == "OK"


def test_assemble_metric_spread() -> None:
    """The metric spread reports min/mean/max over the scored runs."""
    from flows.report import assemble_report

    rep = assemble_report("exp1", _runs())
    spread = rep["metric_spread"]
    assert spread == {"min": 0.91, "max": 0.95, "mean": 0.93}


def test_assemble_leaderboard_orders_scored_best_first() -> None:
    """The leaderboard puts the highest-metric run first and trails the unscored one."""
    from flows.report import assemble_report

    rep = assemble_report("exp1", _runs())
    lb = rep["leaderboard"]
    assert lb[0]["pathspec"] == "EvalCandidate/2"  # best metric first
    assert lb[1]["pathspec"] == "EvalCandidate/1"
    # The unscored/failed run trails.
    assert lb[-1]["pathspec"] == "EvalCandidate/3"


def test_assemble_empty_experiment_is_empty_verdict() -> None:
    """An experiment with no runs yields the EMPTY verdict and no best metric."""
    from flows.report import assemble_report

    rep = assemble_report("nope", [])
    assert rep["n_runs"] == 0
    assert rep["n_successful"] == 0
    assert rep["best_metric"] is None
    assert rep["best_run"] is None
    assert rep["metric_spread"] is None
    assert rep["verdict"] == "EMPTY"


def test_assemble_compacts_learnings_lineage() -> None:
    """Learnings rows are compacted to command/success/metric/data_ref/artifacts."""
    from flows.report import assemble_report

    learnings = [
        {
            "command": "optimize",
            "task": {"experiment_id": "exp1", "data_ref": "IngestDataset/1"},
            "outcome": {"best_metric": 0.95, "submission_ok": True},
            "artifacts": ["EvalCandidate/2", "journal.json"],
            "success": True,
        }
    ]
    rep = assemble_report("exp1", _runs(), learnings)
    assert len(rep["learnings"]) == 1
    row = rep["learnings"][0]
    assert row["command"] == "optimize"
    assert row["success"] is True
    assert row["best_metric"] == 0.95
    assert row["data_ref"] == "IngestDataset/1"
    assert row["artifacts"] == ["EvalCandidate/2", "journal.json"]


def test_assemble_report_is_json_able() -> None:
    """The whole report round-trips through JSON (suitable for a RunResult summary)."""
    from flows.report import assemble_report

    rep = assemble_report("exp1", _runs())
    assert json.loads(json.dumps(rep)) == rep


# ---------------------------------------------------------------------------
# Client-API gather against a fake metaflow module (no live cluster).
# ---------------------------------------------------------------------------


class _FakeData:
    """A ``Run.data`` artifact proxy: attribute access yields artifacts."""

    def __init__(self, **artifacts: object) -> None:
        for key, value in artifacts.items():
            setattr(self, key, value)


class _FakeRun:
    """A finished ``metaflow.Run`` stand-in for the gather projection."""

    def __init__(self, pathspec, successful, data=None, tags=None) -> None:
        self.pathspec = pathspec
        self.successful = successful
        self.data = data
        self.tags = tags or []
        self.created_at = "2026-06-09T00:00:00"
        self.finished_at = "2026-06-09T00:01:00"


@pytest.fixture
def fake_metaflow(monkeypatch):
    """Install a fake ``metaflow`` exposing ``Run`` + ``namespace`` for the gather.

    Returns a ``register(pathspec, run)`` setter; ``Run(pathspec)`` then resolves to
    the registered fake run (or raises for an unknown pathspec).
    """
    registry: dict = {}

    mf = types.ModuleType("metaflow")

    def _Run(pathspec):
        if pathspec not in registry:
            raise ValueError(f"no such run: {pathspec}")
        return registry[pathspec]

    def _namespace(_ns):
        return None

    mf.Run = _Run  # type: ignore[attr-defined]
    mf.namespace = _namespace  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metaflow", mf)

    def register(pathspec: str, run: _FakeRun) -> None:
        registry[pathspec] = run

    return register


def test_gather_by_pathspecs_projects_runs(fake_metaflow) -> None:
    """gather_runs_by_pathspecs projects fake runs into report dicts via the Client API."""
    from flows.report import gather_runs_by_pathspecs

    fake_metaflow(
        "EvalCandidate/2",
        _FakeRun(
            "EvalCandidate/2",
            successful=True,
            data=_FakeData(best_metric=0.95),
            tags=["loom_experiment:exp1"],
        ),
    )
    fake_metaflow(
        "EvalCandidate/3",
        _FakeRun("EvalCandidate/3", successful=False, data=None),
    )

    rows = gather_runs_by_pathspecs(
        ["EvalCandidate/2", "EvalCandidate/3", "EvalCandidate/missing"]
    )
    # The missing pathspec is skipped; the two resolvable runs are projected.
    assert [r["pathspec"] for r in rows] == ["EvalCandidate/2", "EvalCandidate/3"]
    assert rows[0]["metric"] == 0.95
    assert rows[0]["successful"] is True
    assert rows[0]["flow"] == "EvalCandidate"
    assert rows[1]["metric"] is None
    assert rows[1]["successful"] is False


def test_gather_reads_validate_holdout_as_metric(fake_metaflow) -> None:
    """A ValidateFlow run's holdout score is read off the report artifact as the metric."""
    from flows.report import gather_runs_by_pathspecs

    fake_metaflow(
        "ValidateFlow/7",
        _FakeRun(
            "ValidateFlow/7",
            successful=True,
            data=_FakeData(report={"holdout": {"score": 0.82, "n": 60}}),
        ),
    )
    rows = gather_runs_by_pathspecs(["ValidateFlow/7"])
    assert rows[0]["metric"] == 0.82
    assert rows[0]["flow"] == "ValidateFlow"


# ---------------------------------------------------------------------------
# CLI arg-parsing (pure-Python, no Metaflow).
# ---------------------------------------------------------------------------


def test_cli_report_parses_experiment() -> None:
    """`loom report --experiment ID` parses into the report handler."""
    from loom.cli import _cmd_report

    parser = _build_parser()
    args = parser.parse_args(["report", "--experiment", "exp1"])
    assert args.command == "report"
    assert args.experiment == "exp1"
    assert args.runs is None
    assert args.func is _cmd_report


def test_cli_report_parses_runs() -> None:
    """`loom report --runs a,b` parses the pathspec list."""
    parser = _build_parser()
    args = parser.parse_args(["report", "--runs", "EvalCandidate/1,EvalCandidate/2"])
    assert args.runs == "EvalCandidate/1,EvalCandidate/2"
    assert args.experiment is None


def test_cli_report_requires_one_source() -> None:
    """`loom report` needs exactly one of --experiment / --runs."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["report"])  # neither given
    with pytest.raises(SystemExit):
        parser.parse_args(["report", "--experiment", "e", "--runs", "r"])  # both
