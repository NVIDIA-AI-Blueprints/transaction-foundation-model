"""Tests for the lifecycle-flow seam: ``ExecutionProvider.run_flow`` + RunResult.

The MLOps interface grows a ``run_flow(flow_path, parameters, tags) -> RunResult``
seam so the lifecycle commands (EDA, connect, validate, ...) run their flows
*through Loom's interface*, never touching Metaflow directly. This module pins:

* the :class:`~loom.types.RunResult` dataclass shape (the locked contract);
* the ABC default (``NotImplementedError`` with the documented message);
* the ``metaflow`` provider's ``run_flow`` reconstructing a RunResult from a
  finished run, with ``metaflow.Runner`` and the card client **mocked** so the
  test needs no live cluster;
* the ``local`` provider's ``run_flow`` raising a clear, actionable
  NotImplementedError pointing at ``--mlops metaflow``.

These tests are pure-Python: ``metaflow`` is injected as a fake module via
``sys.modules`` for the Runner path, so the suite stays green without a real
Metaflow install.
"""

from __future__ import annotations

import dataclasses
import sys
import types

import pytest

from loom.config import LoomConfig
from loom.types import RunResult


# ---------------------------------------------------------------------------
# RunResult shape (the locked contract).
# ---------------------------------------------------------------------------


def test_run_result_fields_and_defaults() -> None:
    """RunResult carries the five documented fields with the documented defaults."""
    fields = {f.name for f in dataclasses.fields(RunResult)}
    assert fields == {"pathspec", "successful", "card_path", "summary", "error"}

    # Minimal construction: only pathspec + successful are required.
    r = RunResult(pathspec="EdaFlow/1", successful=True)
    assert r.pathspec == "EdaFlow/1"
    assert r.successful is True
    assert r.card_path is None
    assert r.summary == {}
    assert r.error is None

    # summary default is a fresh dict per instance (no shared mutable default).
    r2 = RunResult(pathspec=None, successful=False)
    r.summary["k"] = "v"
    assert r2.summary == {}


# ---------------------------------------------------------------------------
# ABC default: run_flow is not implemented on the base port.
# ---------------------------------------------------------------------------


def test_abc_run_flow_default_raises_not_implemented() -> None:
    """The ExecutionProvider ABC's run_flow default raises with the spec message."""
    from loom.providers import ExecutionProvider
    from loom.types import ExecutionResult

    class _Dummy(ExecutionProvider):
        name = "dummy"

        def execute(self, code: str, reset_session: bool = True) -> ExecutionResult:
            return ExecutionResult(term_out=[], exec_time=0.0, exc_type=None)

    with pytest.raises(NotImplementedError) as excinfo:
        _Dummy().run_flow("flows/eda.py", {"dataset_ref": "IngestDataset/1"})
    assert "does not run lifecycle flows" in str(excinfo.value)


# ---------------------------------------------------------------------------
# local provider: run_flow is not supported, with an actionable message.
# ---------------------------------------------------------------------------


def test_local_run_flow_raises_pointing_at_metaflow() -> None:
    """The local provider's run_flow tells the user to use --mlops metaflow."""
    from loom.registry import get_execution

    provider = get_execution("local")(LoomConfig(mlops_provider="local"))
    with pytest.raises(NotImplementedError) as excinfo:
        provider.run_flow("flows/eda.py", {"dataset_ref": "IngestDataset/1"})

    msg = str(excinfo.value)
    assert "lifecycle flows" in msg
    assert "--mlops metaflow" in msg


# ---------------------------------------------------------------------------
# metaflow provider: run_flow via a mocked Runner + card client.
# ---------------------------------------------------------------------------


class _FakeData:
    """A ``Run.data`` artifact proxy: attribute access yields artifacts."""

    def __init__(self, **artifacts: object) -> None:
        for key, value in artifacts.items():
            setattr(self, key, value)


class _FakeStep:
    """A ``metaflow.Step`` stand-in exposing a single ``.task``."""

    def __init__(self, task: object) -> None:
        self.task = task


class _FakeRun:
    """A finished ``metaflow.Run`` stand-in.

    Iterating a run yields its steps (the card-locating helper does
    ``list(run)``); ``.successful``/``.pathspec``/``.data`` are the Client-API
    surface ``run_flow`` reads.
    """

    def __init__(self, pathspec, successful, data, steps=None) -> None:
        self.pathspec = pathspec
        self.successful = successful
        self.data = data
        self._steps = steps or []

    def __iter__(self):
        return iter(self._steps)


class _ExecutingRun:
    """What ``Runner.run(...)`` returns: an object exposing ``.run``."""

    def __init__(self, run: _FakeRun) -> None:
        self.run = run


class _FakeRunner:
    """A context-manager stand-in for ``metaflow.Runner``.

    Records the flow path it was constructed with and the kwargs ``run`` was
    called with, and returns a pre-seeded :class:`_ExecutingRun`.
    """

    last_init: dict = {}
    last_run_kwargs: dict = {}
    executing: _ExecutingRun | None = None

    def __init__(self, flow_path, show_output=False, **kwargs) -> None:
        _FakeRunner.last_init = {
            "flow_path": flow_path,
            "show_output": show_output,
            **kwargs,
        }

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def run(self, **kwargs):
        _FakeRunner.last_run_kwargs = dict(kwargs)
        return _FakeRunner.executing


class _FakeCard:
    """A ``metaflow`` Card stand-in exposing a datastore ``.path``."""

    def __init__(self, path: str) -> None:
        self.path = path


@pytest.fixture
def fake_metaflow(monkeypatch):
    """Install a fake ``metaflow`` module + ``metaflow.cards`` for run_flow.

    Returns a ``configure(run, cards)`` setter the test uses to choose the run
    ``Runner.run`` resolves to and the cards ``get_cards(task)`` returns.
    """
    holder: dict = {"cards": []}

    mf = types.ModuleType("metaflow")
    mf.Runner = _FakeRunner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metaflow", mf)

    cards_mod = types.ModuleType("metaflow.cards")

    def _get_cards(task, **kwargs):
        return list(holder["cards"])

    cards_mod.get_cards = _get_cards  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metaflow.cards", cards_mod)

    def configure(run: _FakeRun, cards=None) -> None:
        _FakeRunner.executing = _ExecutingRun(run)
        holder["cards"] = cards or []

    return configure


def _provider() -> object:
    from loom.providers.metaflow_exec import MetaflowExecutionProvider

    return MetaflowExecutionProvider(LoomConfig(mlops_provider="metaflow"))


def test_metaflow_run_flow_success_builds_runresult(fake_metaflow) -> None:
    """A successful run yields pathspec, summary (from profile), and the card path."""
    task = object()
    run = _FakeRun(
        pathspec="EdaFlow/42",
        successful=True,
        data=_FakeData(profile={"nrows": 10, "leakage": False}),
        steps=[_FakeStep(task)],
    )
    fake_metaflow(run, cards=[_FakeCard("EdaFlow/42/profile/1/card.html")])

    result = _provider().run_flow(
        "flows/eda.py",
        {"dataset_ref": "IngestDataset/1", "target": None},
        tags=["loom_command:eda"],
    )

    assert isinstance(result, RunResult)
    assert result.successful is True
    assert result.pathspec == "EdaFlow/42"
    assert result.summary == {"nrows": 10, "leakage": False}
    assert result.card_path == "EdaFlow/42/profile/1/card.html"
    assert result.error is None

    # None-valued parameters are dropped; tags pass through to Runner.run.
    assert _FakeRunner.last_run_kwargs["dataset_ref"] == "IngestDataset/1"
    assert "target" not in _FakeRunner.last_run_kwargs
    assert _FakeRunner.last_run_kwargs["tags"] == ["loom_command:eda"]


def test_metaflow_run_flow_failed_run_sets_error(fake_metaflow) -> None:
    """An unsuccessful run is reported as not-successful with an error string."""
    run = _FakeRun(pathspec="EdaFlow/7", successful=False, data=None, steps=[])
    fake_metaflow(run)

    result = _provider().run_flow("flows/eda.py", {"dataset_ref": "IngestDataset/1"})

    assert result.successful is False
    assert result.pathspec == "EdaFlow/7"
    assert result.error is not None
    assert "did not complete successfully" in result.error
    assert result.summary == {}


def test_metaflow_run_flow_success_without_card(fake_metaflow) -> None:
    """A successful run with no card still succeeds; card_path is None."""
    run = _FakeRun(
        pathspec="EdaFlow/9",
        successful=True,
        data=_FakeData(profile={"nrows": 3}),
        steps=[],  # no steps -> no cards located
    )
    fake_metaflow(run, cards=[])

    result = _provider().run_flow("flows/eda.py", {"dataset_ref": "IngestDataset/2"})

    assert result.successful is True
    assert result.card_path is None
    assert result.summary == {"nrows": 3}


def test_metaflow_run_flow_runner_failure_is_degraded(monkeypatch) -> None:
    """If the Runner cannot start, run_flow returns a degraded RunResult (no raise)."""

    class _BoomRunner:
        def __init__(self, *a, **k):
            raise RuntimeError("metadata service unreachable")

    mf = types.ModuleType("metaflow")
    mf.Runner = _BoomRunner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metaflow", mf)

    result = _provider().run_flow("flows/eda.py", {"dataset_ref": "IngestDataset/1"})

    assert isinstance(result, RunResult)
    assert result.successful is False
    assert result.pathspec is None
    assert "failed to run flow" in result.error
