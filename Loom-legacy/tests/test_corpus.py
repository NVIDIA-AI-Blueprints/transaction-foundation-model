"""Corpus tests: NodeRecord persistence and the cross-tenant IP boundary.

These are pure-Python: :class:`loom.corpus.Corpus` writes/reads JSONL using only
the standard library + Loom core, so this module runs without AIDE or Metaflow
installed.

The two behaviours under test are the contract's invariants:

* a :class:`~loom.types.NodeRecord` round-trips through ``record`` -> JSONL ->
  ``all`` losslessly, and a missing ``ts`` is stamped on write; and
* the IP boundary: :meth:`Corpus.general` returns only the ``"general"`` records
  (the slice a cross-tenant moat model may train on) and excludes any record
  tagged with a specific ``owned_by`` tenant.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from loom.config import LoomConfig
from loom.corpus import GENERAL, Corpus
from loom.types import NodeRecord


def _make_record(
    *,
    node_id: str,
    owned_by: str = GENERAL,
    tenant: str = "default",
    experiment_id: str = "exp-corpus",
    parent_id: str | None = None,
    metric: float | None = 0.9,
    ts: float = 0.0,
) -> NodeRecord:
    """Build a populated :class:`NodeRecord` with sensible defaults.

    Keyword-only so each test reads as a small, explicit override of the parts
    it actually cares about (``owned_by`` for the boundary tests, ``node_id`` to
    keep rows distinguishable).
    """
    return NodeRecord(
        experiment_id=experiment_id,
        node_id=node_id,
        parent_id=parent_id,
        stage="draft",
        code="print('hi')",
        term_out=["hi\n", "Execution time: 1 second."],
        exc_type=None,
        metric=metric,
        judge_summary="looks fine",
        model="claude-sonnet-4-5",
        tokens=123,
        tenant=tenant,
        owned_by=owned_by,
        ts=ts,
    )


@pytest.fixture
def corpus(tmp_path) -> Corpus:
    """A corpus backed by a fresh JSONL file under ``tmp_path``."""
    cfg = LoomConfig(corpus_path=str(tmp_path / "corpus" / "nodes.jsonl"))
    return Corpus(cfg)


def test_record_persists_and_roundtrips(corpus) -> None:
    """A recorded node is read back field-for-field identical.

    The only field the corpus is allowed to mutate is ``ts`` (stamped when the
    caller leaves it ``0.0``); every other field must survive the JSONL trip.
    """
    rec = _make_record(node_id="n0", ts=1718000000.0)
    corpus.record(rec)

    read_back = corpus.all()
    assert len(read_back) == 1
    assert read_back[0] == rec
    # Round-trips exactly (including the explicit ts).
    assert dataclasses.asdict(read_back[0]) == dataclasses.asdict(rec)


def test_record_appends_not_overwrites(corpus) -> None:
    """Successive ``record`` calls append; order is preserved."""
    corpus.record(_make_record(node_id="n0", ts=1.0))
    corpus.record(_make_record(node_id="n1", ts=2.0))
    corpus.record(_make_record(node_id="n2", ts=3.0))

    ids = [r.node_id for r in corpus.all()]
    assert ids == ["n0", "n1", "n2"]


def test_record_stamps_missing_timestamp(corpus) -> None:
    """A record with the default ``ts == 0.0`` gets a real timestamp on write."""
    corpus.record(_make_record(node_id="n0", ts=0.0))

    (only,) = corpus.all()
    assert only.ts > 0.0


def test_record_writes_one_json_object_per_line(corpus) -> None:
    """The backing file is newline-delimited JSON (one object per line)."""
    corpus.record(_make_record(node_id="n0", ts=1.0))
    corpus.record(_make_record(node_id="n1", ts=2.0))

    with open(corpus.path, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    assert len(lines) == 2
    for ln in lines:
        obj = json.loads(ln)  # each line parses independently
        assert obj["experiment_id"] == "exp-corpus"


def test_general_excludes_tenant_owned(corpus) -> None:
    """``general()`` returns only ``owned_by == "general"`` records.

    This is the IP boundary: records owned by a specific tenant are tagged and
    must never appear in the cross-tenant ("general") slice.
    """
    corpus.record(_make_record(node_id="g0", owned_by=GENERAL))
    corpus.record(_make_record(node_id="t0", owned_by="acme", tenant="acme"))
    corpus.record(_make_record(node_id="g1", owned_by=GENERAL))
    corpus.record(_make_record(node_id="t1", owned_by="globex", tenant="globex"))

    # all() sees everything; general() sees only the unowned slice.
    assert {r.node_id for r in corpus.all()} == {"g0", "t0", "g1", "t1"}

    general_ids = {r.node_id for r in corpus.general()}
    assert general_ids == {"g0", "g1"}
    assert all(r.owned_by == GENERAL for r in corpus.general())


def test_general_empty_when_no_general_records(corpus) -> None:
    """If every record is tenant-owned, the general slice is empty."""
    corpus.record(_make_record(node_id="t0", owned_by="acme", tenant="acme"))
    corpus.record(_make_record(node_id="t1", owned_by="acme", tenant="acme"))

    assert corpus.all()  # records do exist
    assert corpus.general() == []


def test_all_empty_before_any_write(tmp_path) -> None:
    """Reading a corpus whose file does not exist yet yields no records."""
    cfg = LoomConfig(corpus_path=str(tmp_path / "missing" / "nodes.jsonl"))
    fresh = Corpus(cfg)
    assert fresh.all() == []
    assert fresh.general() == []
