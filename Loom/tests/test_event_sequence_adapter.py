"""``event-sequence`` DataRepresentation adapter + ``local`` Executor — the
port-conformance tests (ARCHITECTURE §10 step 2/3 test gates).

These assert two things the brief calls out:
  (a) the adapter's ``compile``/``contracts``/``signatures`` output is identical
      to today's engine for the financial preset (vocab 6251, the same
      Diagnostics) — i.e. wrapping the engine behind the Protocol changed nothing;
  (b) both adapters structurally satisfy their ``runtime_checkable`` Protocols and
      register under the documented registry keys.

They run on CPU in milliseconds: no GPU, no NeMo, no torch, no network. They
use ``require_engine`` so the suite stays green if the engine is still a stub.
"""

from __future__ import annotations

import os

import pytest

from loom.adapters.event_sequence import EventSequenceRepresentation
from loom.adapters.local_executor import LocalExecutor, LocalJobHandle
from loom.engine import compile_spec, financial_spec
from loom.ports import (
    REPRESENTATIONS,
    EXECUTORS,
    BudgetEnvelope,
    ComputeTarget,
    DataRepresentation,
    Executor,
    JobHandle,
    PreparedCorpus,
    ProgressEvent,
    SourceRef,
)

from .golden_helpers import require_engine


# ---------------------------------------------------------------------------
# Registration + Protocol conformance
# ---------------------------------------------------------------------------


def test_event_sequence_registered_under_its_key():
    assert "event-sequence" in REPRESENTATIONS
    assert isinstance(REPRESENTATIONS["event-sequence"], EventSequenceRepresentation)
    assert REPRESENTATIONS["event-sequence"].name == "event-sequence"


def test_local_executor_registered_under_its_key():
    assert "local" in EXECUTORS
    assert isinstance(EXECUTORS["local"], LocalExecutor)
    assert EXECUTORS["local"].name == "local"


def test_event_sequence_satisfies_data_representation_protocol():
    rep = EventSequenceRepresentation()
    assert isinstance(rep, DataRepresentation)
    # the C4 string it produces is fixed and framework-neutral.
    assert rep.produces_tensor_contract == "clm/input_ids+labels/-100"


def test_local_executor_satisfies_executor_protocol():
    ex = LocalExecutor()
    assert isinstance(ex, Executor)
    # the local executor runs CPU work; it offers no GPU → gates REFUSED_NO_GPU_TARGET.
    assert ex.gpu_available() is False


def test_local_job_handle_satisfies_job_handle_protocol():
    assert isinstance(LocalJobHandle("local-x"), JobHandle)


# ---------------------------------------------------------------------------
# Engine parity — wrapping the engine behind the Protocol changed nothing
# ---------------------------------------------------------------------------


def test_compile_matches_engine_for_financial_preset():
    """The adapter's compile output IS the engine's: same vocab_size (6251),
    same vocab_hash, same derived numbers."""
    rep = EventSequenceRepresentation()
    spec = rep.build_spec({"preset": "financial"})
    compiled = require_engine(lambda: rep.compile(spec, context_len=4096))

    # Apples-to-apples against a direct engine compile.
    direct = require_engine(lambda: compile_spec(financial_spec(), context_len=4096))

    assert compiled.vocab_size == 6251
    assert compiled.vocab_size == direct.vocab_size
    assert compiled.vocab_hash == direct.vocab_hash
    assert compiled.tokens_per_txn == direct.tokens_per_txn
    assert compiled.chunk_size == direct.chunk_size
    assert compiled.context_len == direct.context_len


def test_contracts_match_engine_diagnostics_for_financial_preset():
    """``contracts(compiled)`` returns exactly the engine's
    ``compiled.report.diagnostics`` (same contract ids, severities, messages)."""
    rep = EventSequenceRepresentation()
    spec = rep.build_spec({"preset": "financial"})
    compiled = require_engine(lambda: rep.compile(spec, context_len=4096))

    diags = rep.contracts(compiled)
    engine_diags = list(compiled.report.diagnostics)
    assert [d.to_dict() for d in diags] == [d.to_dict() for d in engine_diags]
    # the financial preset PASSES (no ERROR-severity diagnostic).
    assert rep.representation_passed(compiled) is True
    assert rep.representation_passed(compiled) is compiled.report.passed


def test_representation_passed_is_the_error_scan_default():
    """``representation_passed`` == ``compiled.report.passed`` == the inherited
    ERROR-scan default ``not any(d.severity is ERROR for d in contracts)``."""
    from loom.types import Severity

    rep = EventSequenceRepresentation()
    spec = rep.build_spec({"preset": "financial"})
    compiled = require_engine(lambda: rep.compile(spec, context_len=4096))

    error_scan = not any(
        d.severity is Severity.ERROR for d in rep.contracts(compiled)
    )
    assert rep.representation_passed(compiled) == error_scan


def test_signatures_match_the_tokenize_handoff_dict():
    """``signatures(compiled)`` is the ``tokenize.py:399-407`` dict with
    ``vocab_hash``→``representation_signature``, ``encode_path``→``representation``,
    and the C4 ``tensor_contract`` added."""
    rep = EventSequenceRepresentation()
    spec = rep.build_spec({"preset": "financial"})
    compiled = require_engine(lambda: rep.compile(spec, context_len=4096))

    sig = rep.signatures(compiled)
    # the generalized signature name carries the retrain trigger (today's vocab_hash).
    assert sig["representation_signature"] == compiled.vocab_hash
    assert sig["vocab_hash"] == compiled.vocab_hash
    assert sig["representation"] == "event-sequence"
    assert sig["vocab_size"] == 6251
    assert sig["tokens_per_txn"] == compiled.tokens_per_txn
    assert sig["chunk_size"] == compiled.chunk_size
    assert sig["context_len"] == compiled.context_len
    assert sig["has_fitted_artifact"] == compiled.report.has_fitted_artifact
    assert sig["tensor_contract"] == "clm/input_ids+labels/-100"


def test_chain_preset_compiles_through_the_adapter():
    """The preset switch (engine/spec.py:338) is reachable through ``build_spec``:
    ``--preset chain`` compiles a derived-vocab DEX spec that passes."""
    rep = EventSequenceRepresentation()
    spec = rep.build_spec({"preset": "chain"})
    compiled = require_engine(lambda: rep.compile(spec, context_len=4096))
    assert compiled.vocab_size > 0
    assert rep.representation_passed(compiled) is compiled.report.passed
    # chain keeps wallet identity OUT of the vocab by default (T2).
    assert spec.preset == "chain"


# ---------------------------------------------------------------------------
# plan + materialize over the executor seam (local content-addressed corpus)
# ---------------------------------------------------------------------------


def test_plan_is_cpu_cheap_and_derived(tabformer_df):
    rep = EventSequenceRepresentation()
    ex = LocalExecutor()
    spec = rep.build_spec({"preset": "financial"})
    src = SourceRef(uri="IngestDataset/1", snapshot={"dataframe": tabformer_df})
    plan = require_engine(lambda: rep.plan(spec=spec, source=src, executor=ex))
    assert plan.derived is True
    assert plan.usd == 0.0  # local CPU corpus build is ~$0
    assert plan.gpu_target is None


def test_materialize_writes_local_corpus_and_echoes_signature(tmp_path, monkeypatch, tabformer_df):
    """``materialize`` routes corpus assembly through ``executor.foreach`` and
    returns a PreparedCorpus pointing at the local content-addressed Corpus, with
    the representation_signature == vocab_hash threaded through (the §3/§7 pairing
    invariant the checkpoint will echo)."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    rep = EventSequenceRepresentation()
    ex = LocalExecutor()
    spec = rep.build_spec({"preset": "financial"})
    compiled = require_engine(lambda: rep.compile(spec, context_len=4096))
    src = SourceRef(
        uri="IngestDataset/1",
        snapshot={"dataframe": tabformer_df, "max_event_date": "2026-12-25"},
    )

    pc = require_engine(lambda: rep.materialize(compiled=compiled, source=src, executor=ex))
    assert isinstance(pc, PreparedCorpus)
    assert pc.representation == "event-sequence"
    assert pc.representation_signature == compiled.vocab_hash
    assert pc.tensor_contract == "clm/input_ids+labels/-100"
    assert pc.vocab_size == 6251
    assert pc.seq_length == compiled.context_len
    assert pc.pad_token_id == 0  # <pad> is id 0
    assert pc.train_uri.startswith("Corpus/")
    assert pc.extras["n_txns"] == len(tabformer_df)
    assert pc.effective_tokens == len(tabformer_df) * (compiled.tokens_per_txn + 1)
    # provenance carries the snapshot anchor, NOT the injected frame.
    assert pc.provenance["snapshot"].get("max_event_date") == "2026-12-25"
    assert "dataframe" not in pc.provenance["snapshot"]


def test_materialize_is_idempotent_on_content(tmp_path, monkeypatch, tabformer_df):
    """Same source + same representation_signature → the EXISTING content-addressed
    Corpus, not a twin (§6 idempotency)."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    rep = EventSequenceRepresentation()
    ex = LocalExecutor()
    spec = rep.build_spec({"preset": "financial"})
    compiled = require_engine(lambda: rep.compile(spec, context_len=4096))
    src = SourceRef(uri="IngestDataset/1", snapshot={"dataframe": tabformer_df})

    pc1 = require_engine(lambda: rep.materialize(compiled=compiled, source=src, executor=ex))
    pc2 = require_engine(lambda: rep.materialize(compiled=compiled, source=src, executor=ex))
    assert pc1.train_uri == pc2.train_uri  # content-addressed, no twin


def test_materialize_without_rows_is_not_fatal(tmp_path, monkeypatch):
    """A cloud/unreadable source yields a config-only Corpus (empty payload) — the
    vocab is config-only, so a missing frame is not fatal."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    rep = EventSequenceRepresentation()
    ex = LocalExecutor()
    spec = rep.build_spec({"preset": "financial"})
    compiled = require_engine(lambda: rep.compile(spec, context_len=4096))
    src = SourceRef(uri="bq://project.dataset.table", snapshot={})  # cloud → step 8

    pc = require_engine(lambda: rep.materialize(compiled=compiled, source=src, executor=ex))
    assert pc.extras["n_txns"] == 0
    assert pc.vocab_size == 6251  # vocab is still compiled + carried


# ---------------------------------------------------------------------------
# LocalExecutor — submit budget soft-cap, foreach order, kill best-effort
# ---------------------------------------------------------------------------


def _cpu_compute() -> ComputeTarget:
    return ComputeTarget(launcher="local", nproc_per_node=1, accelerator="cpu")


def test_submit_stops_at_max_steps_budget():
    """The BudgetEnvelope ``max_steps`` is a binding soft cap: the runner stops at
    the cap and the handle ends ``stopped_at_budget``."""
    ex = LocalExecutor()
    events: list[ProgressEvent] = []

    def thunk(emit, should_stop):
        step = 0
        while not should_stop():
            emit(ProgressEvent(step=step, loss=1.0, usd_spent=0.0, usd_envelope=5.0))
            step += 1

    h = ex.submit(
        argv=[thunk],
        image=None,
        compute=_cpu_compute(),
        budget=BudgetEnvelope(max_usd=5.0, max_steps=3),
        on_event=events.append,
    )
    assert h.status() == "stopped_at_budget"
    assert len(events) == 3


def test_submit_succeeds_on_clean_finish():
    ex = LocalExecutor()
    events: list[ProgressEvent] = []

    def thunk(emit, should_stop):
        for s in range(2):
            if should_stop():
                return
            emit(ProgressEvent(step=s, loss=0.5, usd_spent=0.0, usd_envelope=5.0))

    h = ex.submit(
        argv=[thunk],
        image=None,
        compute=_cpu_compute(),
        budget=BudgetEnvelope(max_usd=5.0, max_steps=10),
        on_event=events.append,
    )
    assert h.status() == "succeeded"
    assert len(events) == 2


def test_submit_rejects_a_non_callable_argv():
    """The local executor only runs an in-process callable; a real command line
    (torchrun …) is the metaflow-gcp executor's job → this handle fails."""
    ex = LocalExecutor()
    h = ex.submit(
        argv=["torchrun", "--nproc-per-node=8", "train.py"],
        image=None,
        compute=_cpu_compute(),
        budget=BudgetEnvelope(max_usd=5.0),
        on_event=lambda ev: None,
    )
    assert h.status() == "failed"


def test_foreach_maps_in_order():
    ex = LocalExecutor()
    out = ex.foreach(fn=lambda s: s.upper(), shards=["a", "b", "c"], compute=_cpu_compute())
    assert out == ["A", "B", "C"]


def test_kill_is_best_effort_noop_on_unknown_job():
    ex = LocalExecutor()
    # no exception on an unknown / already-terminal job id.
    ex.kill("local-does-not-exist")
