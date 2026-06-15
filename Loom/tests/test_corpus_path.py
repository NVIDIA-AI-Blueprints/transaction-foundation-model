"""Regression tests: the flywheel corpus path is anchored absolute at load time.

A relative ``corpus_path`` used to be written into (and deleted with) the
execution provider's ephemeral ``chdir`` workspace, so node records were lost and
the post-run leaderboard read found nothing. :meth:`LoomConfig.load` now resolves
a relative ``corpus_path`` to an absolute path at the launch cwd, before any
provider ``chdir``, so records persist regardless of later directory changes.
"""

from __future__ import annotations

import os

from loom.config import LoomConfig
from loom.corpus import Corpus
from loom.types import NodeRecord


def _minimal_record(experiment_id: str) -> NodeRecord:
    """Build a minimal valid NodeRecord for corpus round-trip tests."""
    return NodeRecord(
        experiment_id=experiment_id,
        node_id="n1",
        parent_id=None,
        stage="draft",
        code="print(1)",
        term_out=["ok"],
        exc_type=None,
        metric=0.5,
        judge_summary="fine",
        model="m",
        tokens=None,
    )


def test_load_anchors_relative_corpus_path_absolute(monkeypatch, tmp_path):
    """A relative corpus_path is resolved to an absolute path at load time."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOOM_CORPUS_PATH", "corpus/nodes.jsonl")
    cfg = LoomConfig.load()
    assert os.path.isabs(cfg.corpus_path)
    assert os.path.realpath(cfg.corpus_path) == os.path.realpath(
        str(tmp_path / "corpus" / "nodes.jsonl")
    )


def test_corpus_record_survives_chdir(monkeypatch, tmp_path):
    """A record written after a chdir lands at the anchored path, not the cwd.

    Simulates the execution provider chdir'ing into an ephemeral workspace: the
    record must persist at the load-time absolute path and must NOT be written
    under the workspace (where teardown would delete it).
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOOM_CORPUS_PATH", "corpus/nodes.jsonl")
    cfg = LoomConfig.load()
    anchored = cfg.corpus_path

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)  # mimic LocalExecutionProvider.setup()
    Corpus(cfg).record(_minimal_record("exp-1"))
    monkeypatch.chdir(tmp_path)

    assert os.path.isfile(anchored), "corpus must persist at the anchored absolute path"
    assert not (workspace / "corpus" / "nodes.jsonl").exists(), (
        "corpus must not be written under the ephemeral workspace"
    )
    # And it reads back for the leaderboard.
    assert len(Corpus(cfg).for_experiment("exp-1")) == 1
