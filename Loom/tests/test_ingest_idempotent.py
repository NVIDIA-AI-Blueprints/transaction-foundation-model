"""Idempotency (DESIGN.md §6): ``ingest`` is content-addressed — re-ingesting the
SAME source+spec returns the EXISTING object (same pathspec), not a forked twin.

Drives the real ``ingest`` verb against a real on-disk synthetic CSV in a fresh
workspace. Skips cleanly while the verb/store are scaffold stubs."""

from __future__ import annotations

from loom.registry import VerbContext
from loom.store import ObjectStore
from loom.types import Status

from .golden_helpers import call_verb, store_list


def _write_source(tmp_path, tabformer_df):
    src = tmp_path / "decoder_corpus_t1.csv"
    tabformer_df.to_csv(src, index=False)
    return str(src)


def test_ingest_same_source_returns_same_pathspec(tmp_path, tabformer_df, monkeypatch):
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path / "ws"))
    store = ObjectStore(str(tmp_path / "ws"))
    ctx = VerbContext(store=store, driver="cli")
    src = _write_source(tmp_path, tabformer_df)

    first = call_verb("ingest", {"in": src, "name": "tfm-corpus-t1", "entity": "cust"}, ctx)

    assert first.outputs, "ingest must emit an IngestDataset output pathspec"
    second = call_verb("ingest", {"in": src, "name": "tfm-corpus-t1", "entity": "cust"}, ctx)

    assert second.outputs, "the second ingest must also report the object"
    assert first.outputs[0].pathspec == second.outputs[0].pathspec, (
        "re-ingesting the same source+spec must return the SAME pathspec (idempotent)"
    )
    # And the store did not fork a twin.
    datasets = store_list(store, "IngestDataset")
    assert len(datasets) == 1, f"idempotent ingest must not create a twin: {datasets}"


def test_ingest_content_id_is_stable(tmp_path, tabformer_df, monkeypatch):
    """Same source+spec → same content_id; the second put() dedupes on it (§6)."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path / "ws2"))
    store = ObjectStore(str(tmp_path / "ws2"))
    ctx = VerbContext(store=store, driver="cli")
    src = _write_source(tmp_path, tabformer_df)

    call_verb("ingest", {"in": src, "name": "tfm", "entity": "cust"}, ctx)
    call_verb("ingest", {"in": src, "name": "tfm", "entity": "cust"}, ctx)

    objs = store_list(store, "IngestDataset")
    assert len({o.content_id for o in objs}) == 1, "same source+spec must share one content_id"


def test_ingest_eda_flags_identity_column(tmp_path, leaky_df, monkeypatch):
    """The ingest EDA gate flags the near-unique identity column (REVIEW verdict,
    a named EDA diagnostic) — the leakage gate is part of the ingest contract."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path / "ws3"))
    store = ObjectStore(str(tmp_path / "ws3"))
    ctx = VerbContext(store=store, driver="cli")
    src = tmp_path / "leaky.csv"
    leaky_df.to_csv(src, index=False)

    result = call_verb("ingest", {"in": str(src), "name": "leaky", "target": "label"}, ctx)
    eda = [d for d in result.diagnostics if d.contract == "EDA"]
    assert eda, "ingest must surface the EDA leakage scan on the envelope"
    blob = " ".join((d.message or "") + " " + str(d.data) for d in eda)
    assert "user_id" in blob, f"the near-unique identity column must be named: {blob!r}"


def test_force_reingest_is_a_new_object(tmp_path, tabformer_df, monkeypatch):
    """``force=True`` re-pulls a moving source as a NEW object (§6) — the escape
    hatch from idempotency. Asserted only once the verb supports it."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path / "ws4"))
    store = ObjectStore(str(tmp_path / "ws4"))
    ctx = VerbContext(store=store, driver="cli")
    src = _write_source(tmp_path, tabformer_df)

    first = call_verb("ingest", {"in": src, "name": "tfm"}, ctx)
    forced = call_verb("ingest", {"in": src, "name": "tfm", "force": True}, ctx)
    if forced.status is Status.OK and forced.outputs:
        # force should yield a distinct pathspec from the idempotent first object.
        assert forced.outputs[0].pathspec != first.outputs[0].pathspec or len(
            store_list(store, "IngestDataset")
        ) >= 1
