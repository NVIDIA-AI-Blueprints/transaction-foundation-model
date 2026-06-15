"""Learnings tests: command-level rollout persistence and the IP boundary.

These are pure-Python: :class:`loom.learnings.Learnings` writes/reads JSONL using
only the standard library + Loom core, so this module runs without AIDE or
Metaflow installed.

Three behaviours under test, mirroring the corpus contract one level up (one row
per *run* rather than one per *node*):

* a :class:`~loom.learnings.LearningRecord` (with its nested ``task``/``outcome``)
  round-trips through ``record`` -> JSONL -> ``all`` losslessly, and a missing
  ``ts`` is stamped on write;
* the IP boundary: :meth:`Learnings.general` returns only the ``"general"`` rows
  (the slice a cross-tenant skill-optimizer may train on) and excludes any row
  tagged with a specific ``owned_by`` tenant; and
* the backing ``learnings_path`` is anchored ABSOLUTE at config load, so the
  rollout survives a provider ``chdir`` into an ephemeral workspace.
"""

from __future__ import annotations

import dataclasses
import json
import os

import pytest

from loom.config import LoomConfig
from loom.learnings import GENERAL, Learnings, LearningRecord, Outcome, TaskSpec


def _make_record(
    *,
    command: str = "loom-aide",
    owned_by: str = GENERAL,
    tenant: str = "default",
    experiment_id: str = "exp-learn",
    best_metric: float | None = 0.91,
    submission_ok: bool = True,
    ts: float = 0.0,
) -> LearningRecord:
    """Build a populated :class:`LearningRecord` with sensible defaults.

    Keyword-only so each test reads as a small, explicit override of the parts it
    actually cares about (``owned_by`` for the boundary tests, ``command`` to
    keep rows distinguishable).
    """
    return LearningRecord(
        command=command,
        task=TaskSpec(
            data_ref="IngestDataset/123",
            goal="predict churn",
            metric="maximize ROC AUC",
            experiment_id=experiment_id,
        ),
        inputs={
            "search_provider": "aide",
            "mlops_provider": "metaflow",
            "budget": {"steps": 10, "num_drafts": 3},
        },
        outcome=Outcome(
            best_metric=best_metric,
            submission_ok=submission_ok,
            node_count=7,
        ),
        artifacts=["journal.json", "tree.html"],
        success=submission_ok,
        model="claude-sonnet-4-5",
        tenant=tenant,
        owned_by=owned_by,
        reflection="draft beat the baseline; improve step stalled",
        ts=ts,
    )


@pytest.fixture
def learnings(tmp_path) -> Learnings:
    """A learnings store backed by a fresh JSONL file under ``tmp_path``."""
    cfg = LoomConfig(learnings_path=str(tmp_path / "learnings" / "rollouts.jsonl"))
    return Learnings(cfg)


def test_record_persists_and_roundtrips(learnings) -> None:
    """A recorded rollout reads back field-for-field identical.

    The only field the store may mutate is ``ts`` (stamped when the caller leaves
    it ``0.0``); every other field, including the nested ``task``/``outcome``
    dataclasses, must survive the JSONL trip.
    """
    rec = _make_record(command="loom-aide", ts=1718000000.0)
    learnings.record(rec)

    read_back = learnings.all()
    assert len(read_back) == 1
    assert read_back[0] == rec
    # Nested dataclasses rehydrate, not bare dicts.
    assert isinstance(read_back[0].task, TaskSpec)
    assert isinstance(read_back[0].outcome, Outcome)
    assert dataclasses.asdict(read_back[0]) == dataclasses.asdict(rec)


def test_record_appends_not_overwrites(learnings) -> None:
    """Successive ``record`` calls append; order is preserved."""
    learnings.record(_make_record(command="loom-a", ts=1.0))
    learnings.record(_make_record(command="loom-b", ts=2.0))
    learnings.record(_make_record(command="loom-c", ts=3.0))

    commands = [r.command for r in learnings.all()]
    assert commands == ["loom-a", "loom-b", "loom-c"]


def test_record_stamps_missing_timestamp(learnings) -> None:
    """A record with the default ``ts == 0.0`` gets a real timestamp on write."""
    learnings.record(_make_record(ts=0.0))

    (only,) = learnings.all()
    assert only.ts > 0.0


def test_record_writes_one_json_object_per_line(learnings) -> None:
    """The backing file is newline-delimited JSON (one object per line)."""
    learnings.record(_make_record(command="loom-a", ts=1.0))
    learnings.record(_make_record(command="loom-b", ts=2.0))

    with open(learnings.path, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    assert len(lines) == 2
    for ln in lines:
        obj = json.loads(ln)  # each line parses independently
        assert obj["task"]["experiment_id"] == "exp-learn"
        assert "best_metric" in obj["outcome"]


def test_general_excludes_tenant_owned(learnings) -> None:
    """``general()`` returns only ``owned_by == "general"`` records.

    The IP boundary, mirroring the corpus: rollouts owned by a specific tenant
    must never appear in the cross-tenant ("general") slice the skill-optimizer
    may train on.
    """
    learnings.record(_make_record(command="g0", owned_by=GENERAL))
    learnings.record(_make_record(command="t0", owned_by="acme", tenant="acme"))
    learnings.record(_make_record(command="g1", owned_by=GENERAL))
    learnings.record(_make_record(command="t1", owned_by="globex", tenant="globex"))

    # all() sees everything; general() sees only the unowned slice.
    assert {r.command for r in learnings.all()} == {"g0", "t0", "g1", "t1"}

    general_cmds = {r.command for r in learnings.general()}
    assert general_cmds == {"g0", "g1"}
    assert all(r.owned_by == GENERAL for r in learnings.general())


def test_general_empty_when_no_general_records(learnings) -> None:
    """If every rollout is tenant-owned, the general slice is empty."""
    learnings.record(_make_record(command="t0", owned_by="acme", tenant="acme"))
    learnings.record(_make_record(command="t1", owned_by="acme", tenant="acme"))

    assert learnings.all()  # records do exist
    assert learnings.general() == []


def test_trajectory_id_is_additive_and_roundtrips(learnings) -> None:
    """The telemetry trajectory_id is an additive, defaulted field that roundtrips.

    A pre-telemetry row carries ``trajectory_id == None`` (the default), and a row
    stamped with a trajectory id reads back unchanged -- the surgical, backward-
    compatible link from the rollout to its telemetry trajectory.
    """
    # Default: absent -> None, and a legacy JSONL line without the key still reads.
    plain = _make_record(command="legacy", ts=1.0)
    assert plain.trajectory_id is None
    learnings.record(plain)

    stamped = dataclasses.replace(
        _make_record(command="stamped", ts=2.0), trajectory_id="loom-exp-7"
    )
    learnings.record(stamped)

    read_back = {r.command: r for r in learnings.all()}
    assert read_back["legacy"].trajectory_id is None
    assert read_back["stamped"].trajectory_id == "loom-exp-7"


def test_all_empty_before_any_write(tmp_path) -> None:
    """Reading a store whose file does not exist yet yields no records."""
    cfg = LoomConfig(learnings_path=str(tmp_path / "missing" / "rollouts.jsonl"))
    fresh = Learnings(cfg)
    assert fresh.all() == []
    assert fresh.general() == []


def test_load_anchors_relative_learnings_path_absolute(monkeypatch, tmp_path) -> None:
    """A relative learnings_path is resolved to an absolute path at load time."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOOM_LEARNINGS_PATH", "learnings/rollouts.jsonl")
    cfg = LoomConfig.load()
    assert os.path.isabs(cfg.learnings_path)
    assert os.path.realpath(cfg.learnings_path) == os.path.realpath(
        str(tmp_path / "learnings" / "rollouts.jsonl")
    )


def test_default_learnings_path_anchored_absolute(monkeypatch, tmp_path) -> None:
    """The default learnings_path is anchored absolute at the launch cwd."""
    monkeypatch.chdir(tmp_path)
    cfg = LoomConfig.load(env={})
    assert os.path.isabs(cfg.learnings_path)
    assert os.path.realpath(cfg.learnings_path) == os.path.realpath(
        str(tmp_path / "learnings" / "rollouts.jsonl")
    )


def test_learnings_record_survives_chdir(monkeypatch, tmp_path) -> None:
    """A rollout written after a chdir lands at the anchored path, not the cwd.

    Simulates the execution provider chdir'ing into an ephemeral workspace: the
    record must persist at the load-time absolute path and must NOT be written
    under the workspace (where teardown would delete it).
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOOM_LEARNINGS_PATH", "learnings/rollouts.jsonl")
    cfg = LoomConfig.load()
    anchored = cfg.learnings_path

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)  # mimic LocalExecutionProvider.setup()
    Learnings(cfg).record(_make_record(command="loom-aide", ts=1.0))
    monkeypatch.chdir(tmp_path)

    assert os.path.isfile(anchored), "learnings must persist at the anchored path"
    assert not (workspace / "learnings" / "rollouts.jsonl").exists(), (
        "learnings must not be written under the ephemeral workspace"
    )
    assert len(Learnings(cfg).all()) == 1
