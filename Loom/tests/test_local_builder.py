"""Local CPU-rehearsal ModelBuilder tests (ARCHITECTURE §9, §10 step 4).

The ``local`` PPMI+SVD builder is the GPU-free CI oracle for the model-builder
port. These tests assert the C5 checkpoint round-trips, the representation↔
checkpoint signature pairing holds, the launch-and-track event stream carries
the step-0 ``loss≈ln(vocab)`` canary, the cost plan is a derived ~$0 CPU plan,
``supports()`` rejects a corpus with no token stream, and the adapter pulls in
none of torch/transformers/NeMo/RAPIDS (the torch-free invariant)."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

import loom.adapters.local_builder as local_builder  # noqa: F401 - self-registers
from loom.ports import (
    MODEL_BUILDERS,
    BudgetEnvelope,
    CheckpointRef,
    ComputeTarget,
    ModelBuilder,
    ModelSpec,
    Objective,
    PreparedCorpus,
    TrainingHandle,
)


# ---------------------------------------------------------------------------
# Fixtures — a tiny in-memory corpus in both supported token-stream shapes.
# ---------------------------------------------------------------------------


def _model() -> ModelSpec:
    return ModelSpec(
        family="decoder-clm",
        arch={"_target_": "transformers.LlamaConfig", "hidden_size": 32, "vocab_size": 12},
    )


def _objective() -> Objective:
    return Objective(kind="next-token", requires_tensor_contract="clm/input_ids+labels/-100")


def _compute() -> ComputeTarget:
    return ComputeTarget(launcher="local", accelerator="cpu", gpu_target=None)


def _budget() -> BudgetEnvelope:
    return BudgetEnvelope(max_usd=1.0, max_wall_clock_min=5, max_steps=2)


def _corpus_token_lines() -> PreparedCorpus:
    """A corpus carrying ``token_lines`` (the representation-neutral id shape)."""
    # vocab ids 0..11; a couple of repeating co-occurrence patterns.
    lines = [
        [1, 5, 6, 7, 8, 2],
        [1, 5, 6, 9, 10, 2],
        [1, 7, 8, 5, 6, 2],
        [1, 9, 10, 11, 5, 2],
    ]
    return PreparedCorpus(
        representation="event-sequence",
        representation_signature="sig-abc123",
        tensor_contract="clm/input_ids+labels/-100",
        train_uri="mem://train",
        val_uri=None,
        test_uri=None,
        manifest_uri="mem://manifest",
        seq_length=16,
        pad_token_id=0,
        vocab_size=12,
        effective_tokens=sum(len(l) for l in lines),
        extras={"token_lines": lines},
    )


def _corpus_string_lines() -> PreparedCorpus:
    """A corpus carrying the event-sequence ``corpus_lines`` + ``vocab`` shape."""
    vocab = {"<bos>": 1, "<eos>": 2, "A": 3, "B": 4, "C": 5}
    corpus_lines = ["<bos> A B C <eos>", "<bos> A C B <eos>", "<bos> B A C <eos>"]
    return PreparedCorpus(
        representation="event-sequence",
        representation_signature="sig-strings",
        tensor_contract="clm/input_ids+labels/-100",
        train_uri="mem://train",
        val_uri=None,
        test_uri=None,
        manifest_uri="mem://manifest",
        seq_length=16,
        pad_token_id=0,
        vocab_size=len(vocab) + 1,  # +1 so max id < vocab_size
        effective_tokens=0,
        extras={"corpus_lines": corpus_lines, "vocab": vocab},
    )


# ---------------------------------------------------------------------------
# Registration + protocol conformance.
# ---------------------------------------------------------------------------


def test_local_builder_registered_and_conforms():
    assert "local" in MODEL_BUILDERS, "local builder did not self-register"
    b = MODEL_BUILDERS["local"]
    assert b.name == "local"
    assert isinstance(b, ModelBuilder)  # runtime_checkable structural check


def test_adapter_imports_nothing_forbidden():
    """The torch-free invariant: importing the adapter must pull in none of
    torch/transformers/NeMo/RAPIDS."""
    forbidden = ("torch", "transformers", "nemo", "nemo_automodel", "cudf", "rapids", "cupy")
    loaded = [
        m for m in sys.modules
        if any(m == f or m.startswith(f + ".") for f in forbidden)
    ]
    assert loaded == [], f"forbidden modules imported: {loaded}"


# ---------------------------------------------------------------------------
# supports() — accept any token stream, reject the empty corpus.
# ---------------------------------------------------------------------------


def test_supports_accepts_token_lines():
    b = MODEL_BUILDERS["local"]
    cap = b.supports(model=_model(), objective=_objective(), corpus=_corpus_token_lines())
    assert cap.supported is True
    assert cap.reason is None


def test_supports_accepts_string_lines():
    b = MODEL_BUILDERS["local"]
    cap = b.supports(model=_model(), objective=_objective(), corpus=_corpus_string_lines())
    assert cap.supported is True


def test_supports_rejects_no_token_stream():
    b = MODEL_BUILDERS["local"]
    empty = PreparedCorpus(
        representation="x",
        representation_signature="s",
        tensor_contract="any",
        train_uri="m",
        val_uri=None,
        test_uri=None,
        manifest_uri="m",
        seq_length=8,
        pad_token_id=0,
        vocab_size=0,
        effective_tokens=0,
        extras={},
    )
    cap = b.supports(model=_model(), objective=_objective(), corpus=empty)
    assert cap.supported is False
    assert cap.reason and "token stream" in cap.reason


# ---------------------------------------------------------------------------
# plan() — a derived ~$0 CPU plan.
# ---------------------------------------------------------------------------


def test_plan_is_derived_cpu_zero_cost():
    b = MODEL_BUILDERS["local"]
    corpus = _corpus_token_lines()
    plan = b.plan(
        corpus=corpus,
        model=_model(),
        objective=_objective(),
        compute=_compute(),
        budget=_budget(),
        executor=None,
    )
    assert plan.derived is True
    assert plan.usd is not None and plan.usd < 0.01  # ~$0 CPU
    assert plan.params == corpus.vocab_size * b.dim  # embedding table size
    assert plan.tokens == corpus.effective_tokens
    assert plan.gpu_target is None  # CPU rehearsal needs no GPU target
    assert plan.envelope["max_usd"] == _budget().max_usd


# ---------------------------------------------------------------------------
# launch() — the full rehearsal: handle, events, C5 round-trip, signature echo.
# ---------------------------------------------------------------------------


def test_launch_produces_checkpoint_and_events():
    b = MODEL_BUILDERS["local"]
    corpus = _corpus_token_lines()
    handle = b.launch(
        corpus=corpus,
        model=_model(),
        objective=_objective(),
        compute=_compute(),
        budget=_budget(),
        executor=None,
    )
    assert isinstance(handle, TrainingHandle)  # runtime_checkable structural check
    assert handle.status() == "succeeded"

    events = list(handle.stream_events())
    assert len(events) >= 2, "expected a step-0 canary + at least one more event"
    # The step-0 loss canary lives in the first event's note (ARCHITECTURE §3).
    assert events[0].step == 0
    assert events[0].note and "loss≈ln(vocab)" in events[0].note
    assert events[0].note.endswith("(rehearsal)")
    # Budget is untouched (CPU ~$0) and the envelope is carried on every event.
    assert all(e.usd_spent == 0.0 for e in events)
    assert all(e.usd_envelope == _budget().max_usd for e in events)

    ckpt = handle.result()
    assert isinstance(ckpt, CheckpointRef)
    assert ckpt.fmt == "hf-safetensors-consolidated"


def test_c5_safetensors_roundtrips():
    """C5 for the local builder: a VALID consolidated safetensors that
    ``safetensors.numpy.load_file`` re-reads (ARCHITECTURE §10 step 4)."""
    from safetensors.numpy import load_file

    b = MODEL_BUILDERS["local"]
    corpus = _corpus_token_lines()
    ckpt = b.launch(
        corpus=corpus,
        model=_model(),
        objective=_objective(),
        compute=_compute(),
        budget=_budget(),
        executor=None,
    ).result()

    st_path = os.path.join(ckpt.uri, "model.safetensors")
    assert os.path.exists(st_path), "consolidated safetensors not written"
    reloaded = load_file(st_path)
    assert "embeddings" in reloaded
    emb = reloaded["embeddings"]
    assert emb.shape == (corpus.vocab_size, b.dim)
    assert emb.dtype == np.float32

    # A minimal, valid config.json sits beside it (the consolidated dir).
    config_path = os.path.join(ckpt.uri, "config.json")
    assert os.path.exists(config_path)
    config = json.loads(open(config_path, encoding="utf-8").read())
    assert config["vocab_size"] == corpus.vocab_size
    assert config["hidden_size"] == b.dim


def test_representation_signature_is_echoed():
    """The harness-level pairing invariant: CheckpointRef.representation_signature
    is echoed verbatim from the corpus (ARCHITECTURE §3, §7)."""
    b = MODEL_BUILDERS["local"]
    corpus = _corpus_token_lines()
    ckpt = b.launch(
        corpus=corpus,
        model=_model(),
        objective=_objective(),
        compute=_compute(),
        budget=_budget(),
        executor=None,
    ).result()
    assert ckpt.representation_signature == corpus.representation_signature
    assert ckpt.model_signature and ckpt.model_signature.startswith("msig-")
    assert ckpt.model_signature != ckpt.representation_signature


def test_string_line_corpus_also_builds():
    """The event-sequence corpus.json shape (string lines + vocab) trains too."""
    from safetensors.numpy import load_file

    b = MODEL_BUILDERS["local"]
    corpus = _corpus_string_lines()
    handle = b.launch(
        corpus=corpus,
        model=_model(),
        objective=_objective(),
        compute=_compute(),
        budget=_budget(),
        executor=None,
    )
    ckpt = handle.result()
    assert ckpt.representation_signature == corpus.representation_signature
    emb = load_file(os.path.join(ckpt.uri, "model.safetensors"))["embeddings"]
    assert emb.shape == (corpus.vocab_size, b.dim)


def test_launch_runs_through_executor_seam():
    """launch() flows through executor.foreach when one is provided (the §6 seam)."""
    calls = {"foreach": 0}

    class _StubExecutor:
        name = "local-stub"

        def gpu_available(self) -> bool:
            return False

        def submit(self, **kwargs):  # pragma: no cover - unused here
            raise NotImplementedError

        def foreach(self, *, fn, shards, compute):
            calls["foreach"] += 1
            return [fn(s) for s in shards]

        def kill(self, job_id: str) -> None:  # pragma: no cover
            return None

    b = MODEL_BUILDERS["local"]
    ckpt = b.launch(
        corpus=_corpus_token_lines(),
        model=_model(),
        objective=_objective(),
        compute=_compute(),
        budget=_budget(),
        executor=_StubExecutor(),
    ).result()
    assert calls["foreach"] == 1, "launch did not route the fit through executor.foreach"
    assert ckpt.metrics.get("ran_via_executor") is True
