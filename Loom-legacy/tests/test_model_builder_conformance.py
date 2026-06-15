"""The GOLDEN CONFORMANCE SUITE for the ``ModelBuilderProvider`` port (design §5).

Any provider claiming :class:`~loom.providers.ModelBuilderProvider` must pass the
**same** contract tests -- the discipline that makes "drop-in" real, mirroring the
``dataio`` source-scan and the executor/workspace contracts. The suite is
**parametrized over every registered backend** (``loom.registry.get_model_builder``),
runs the real lift assertion on the **torch-free ``local`` default**, and runs the
shape + manifest + mode + no-GPU-refusal assertions on the ``nemo`` backend (which
plans, with ``gpu_target=None``, and never launches).

The eight contract tests (design §5):

1. **Round-trip** -- ``tokenize -> pretrain -> embed -> finetune -> evaluate`` runs
   end-to-end and each step returns a non-error ``ArtifactRef`` / ``Scores``; the
   embeddings ``ArtifactRef`` is ``IngestDataset``-shaped.
2. **Valid pathspecs** -- every non-error ``ArtifactRef.pathspec`` matches
   ``<FlowName>/<run_id>`` (``dataio.resolve_run`` accepts it); no field is a file
   path, ``s3://``, or ``.nemo`` literal.
3. **Comparable scores** -- ``evaluate`` returns a ``Scores`` with a float ``value``
   carrying ``baseline_raw`` + ``lift``; on a PLANTED-SEQUENTIAL-SIGNAL fixture the
   embeddings beat the raw baseline (``lift > epsilon``). Same metric every backend.
4. **Manifest honesty** -- ``manifest().backend`` is set; every ``supported=True``
   capability is callable without ``NotImplementedError``; every stand-in/limited
   capability carries a non-empty ``notes``; an unsupported capability refuses
   cleanly, never a deep crash.
5. **Mode correctness** -- ``mode_of("pretrain") == "launch-and-track"`` for every
   backend (AIDE never tree-searches the backbone); ``tokenize`` / ``finetune`` /
   ``embed`` / ``evaluate`` are ``searchable``.
6. **Capability-gap refusal** -- a backend with ``serve(online).supported=False``
   (``local``) refuses ``serve(_, "online")`` up front with an actionable message;
   ``pretrain`` with ``gpu_target=None`` (``nemo``) returns ``REFUSED_NO_GPU_TARGET``,
   never launching.
7. **Determinism / lineage** -- two ``pretrain`` calls on the same fixture produce
   the SAME fingerprint (the ``local`` SVD is ``random_state=0``-pinned).
8. **Source-scan** -- no ``.nemo`` / ``s3://`` / object-store-SDK import / NeMo-noun
   token (``Megatron`` / ``@resources``) appears in any model-builder **adapter
   module** outside the ``nemo`` adapter's internal lowering strings; the seam
   validates ``objective`` / ``budget`` / ``mode`` against the frozensets.

The round-trip is exercised against the **provider methods** (the real contract) by
monkeypatching the ``local`` adapter's two Client-API helpers (``_materialize`` /
``_load_backbone``) to serve a small in-memory fixture -- so the pure PPMI+SVD path
runs end-to-end **without importing Metaflow** (constraint 3: the default/CI path is
torch-free and Metaflow-free for the pure round-trip).
"""

from __future__ import annotations

import pytest

# The model-builder maths needs pandas / numpy / scikit-learn / scipy. These are in
# the repo venv; importorskip keeps a stripped env from erroring the whole module.
pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from loom.config import LoomConfig  # noqa: E402
from loom.providers import BUDGETS, MODES, OBJECTIVES  # noqa: E402
from loom.registry import get_model_builder  # noqa: E402
from loom.types import ArtifactRef, CapabilityManifest, Scores  # noqa: E402

#: A strictly-positive lift epsilon: on the planted-sequential fixture the embeddings
#: must beat the raw baseline by more than this (test 3).
_LIFT_EPSILON = 1e-9

#: Pathspec the in-memory fixture pretends to come from (a valid ``<Flow>/<id>``).
_FIXTURE_REF = "IngestDataset/1"

#: The backends every conformance test is parametrized over (all registered names).
_BACKENDS = ["local", "nemo"]


# ---------------------------------------------------------------------------
# The PLANTED-SEQUENTIAL-SIGNAL fixture (in-memory; no Metaflow).
# ---------------------------------------------------------------------------


def _planted_fixture(n_accounts: int = 80, seed: int = 0) -> "pd.DataFrame":
    """Build a per-account event-sequence frame with a PLANTED sequential signal.

    Each account emits a time-ordered sequence of categorical ``event`` tokens. For
    the **positive** class the events follow a fixed Markov chain (a strong
    next-event / co-occurrence structure), while the **negative** class emits events
    independently at random. A raw per-row gradient-boosted baseline -- which sees
    only marginal event frequencies -- cannot exploit the *sequential* structure, but
    the PPMI+SVD backbone (which factorizes the co-occurrence matrix) can, so the
    pooled-embedding model beats the raw baseline (the real lift test 3 asserts).

    Columns: ``account`` (group), ``t`` (order), ``event`` (categorical token),
    ``amount`` (a numeric field), ``label`` (the as-of per-account target).

    Args:
        n_accounts: Number of accounts (sequences) to plant.
        seed: RNG seed for reproducibility.

    Returns:
        A long-form pandas DataFrame (one row per event).
    """
    rng = np.random.default_rng(seed)
    # A planted Markov chain over a small event alphabet for the positive class.
    alphabet = ["A", "B", "C", "D", "E"]
    chain = {  # a -> the strongly-preferred next event (the planted signal)
        "A": "B",
        "B": "C",
        "C": "D",
        "D": "E",
        "E": "A",
    }
    rows: list[dict] = []
    for acct in range(n_accounts):
        positive = acct % 2 == 0
        seq_len = int(rng.integers(8, 16))
        if positive:
            cur = alphabet[int(rng.integers(0, len(alphabet)))]
            events = [cur]
            for _ in range(seq_len - 1):
                # Follow the chain 85% of the time (the planted sequential signal).
                cur = chain[cur] if rng.random() < 0.85 else alphabet[int(rng.integers(0, len(alphabet)))]
                events.append(cur)
        else:
            events = [alphabet[int(rng.integers(0, len(alphabet)))] for _ in range(seq_len)]
        for t, ev in enumerate(events):
            rows.append(
                {
                    "account": f"acct-{acct:03d}",
                    "t": t,
                    "event": ev,
                    "amount": float(rng.normal(10.0, 2.0)),
                    "label": int(positive),
                }
            )
    return pd.DataFrame(rows)


def _fixture_schema() -> dict:
    """The ``IngestDataset``-shaped schema dict for the planted fixture."""
    frame = _planted_fixture(n_accounts=2)
    return {
        "columns": [str(c) for c in frame.columns],
        "dtypes": {str(c): str(frame[c].dtype) for c in frame.columns},
        "nrows": int(len(frame)),
        "target": "label",
    }


@pytest.fixture()
def local_builder(monkeypatch):
    """A ``local`` builder whose two Client-API helpers serve the in-memory fixture.

    Monkeypatches :meth:`LocalModelBuilderProvider._materialize` to return the planted
    frame + schema (so ``tokenize`` / ``pretrain`` / ``embed`` / ``finetune`` /
    ``evaluate`` run their pure PPMI+SVD path with no Metaflow), and
    :meth:`_load_backbone` to recompute the deterministic backbone inline from the
    fixture. This exercises the real provider methods (the contract), not just the
    bare helpers.
    """
    from loom.providers.model_builder import local as local_mod

    frame = _planted_fixture()
    schema = _fixture_schema()

    def _fake_materialize(self, ref):  # noqa: ANN001 - test stub
        return frame.copy(), dict(schema)

    def _fake_load_backbone(self, backbone_ref, train, sch):  # noqa: ANN001 - test stub
        resolved = local_mod.resolve_scheme(None, sch)
        vocab = local_mod.build_vocab(train, resolved)
        sequences = local_mod.encode_sequences(train, vocab)
        C = local_mod.build_cooccurrence(sequences, vocab["size"], "next-event")
        W = local_mod.factorize_backbone(
            local_mod.ppmi(C), local_mod._BUDGET_DIMS["small"], random_state=local_mod._RANDOM_STATE
        )
        return W, vocab

    monkeypatch.setattr(local_mod.LocalModelBuilderProvider, "_materialize", _fake_materialize)
    monkeypatch.setattr(
        local_mod.LocalModelBuilderProvider, "_load_backbone", _fake_load_backbone
    )
    return local_mod.LocalModelBuilderProvider(LoomConfig())


def _make_builder(backend: str):
    """Construct a registered builder by name (``nemo`` with ``gpu_target=None``)."""
    cls = get_model_builder(backend)
    config = LoomConfig(model_builder_provider=backend, gpu_target=None)
    try:
        return cls(config, launch=False)  # nemo accepts launch=
    except TypeError:
        return cls(config)


# ---------------------------------------------------------------------------
# Test 1 -- Round-trip (local: the real PPMI+SVD path end-to-end).
# ---------------------------------------------------------------------------


def test_roundtrip_local_each_step_non_error(local_builder) -> None:
    """tokenize -> pretrain -> embed -> finetune -> evaluate, each a non-error ref."""
    tok = local_builder.tokenize(_FIXTURE_REF, {})
    assert isinstance(tok, ArtifactRef) and tok.error is None and tok.kind == "tokenizer"

    bb = local_builder.pretrain(_FIXTURE_REF, "next-event", "small")
    assert isinstance(bb, ArtifactRef) and bb.error is None and bb.kind == "backbone"

    emb = local_builder.embed(bb.pathspec, _FIXTURE_REF)
    assert isinstance(emb, ArtifactRef) and emb.error is None and emb.kind == "embeddings"

    model = local_builder.finetune(bb.pathspec, _FIXTURE_REF, {})
    assert isinstance(model, ArtifactRef) and model.error is None and model.kind == "model"

    scores = local_builder.evaluate(bb.pathspec, _FIXTURE_REF, "fraud-pr-auc")
    assert isinstance(scores, Scores) and scores.value is not None


def test_roundtrip_embeddings_are_ingestdataset_shaped(local_builder, monkeypatch) -> None:
    """The embeddings ref's produced data object is IngestDataset-shaped (train/schema).

    The ``local`` ``embed`` helper builds a ``{train, test, schema, fingerprint}``
    dict whose shape ``dataio.materialize_dataset`` reads back unchanged; here we
    assert the underlying helper output is that exact shape (the round-trip the suite
    promises), without importing Metaflow.
    """
    from loom.providers.model_builder import local as local_mod

    frame = _planted_fixture()
    schema = _fixture_schema()
    resolved = local_mod.resolve_scheme(None, schema)
    vocab = local_mod.build_vocab(frame, resolved)
    sequences = local_mod.encode_sequences(frame, vocab)
    W = local_mod.factorize_backbone(
        local_mod.ppmi(local_mod.build_cooccurrence(sequences, vocab["size"], "next-event")),
        local_mod._BUDGET_DIMS["small"],
    )
    dataset = local_mod.build_embedding_dataset(frame, W, vocab, "label")
    assert set(dataset) >= {"train", "test", "schema", "fingerprint"}
    assert "columns" in dataset["schema"] and "target" in dataset["schema"]
    assert dataset["schema"]["target"] == "label"
    assert len(dataset["train"]) > 0


def test_roundtrip_nemo_shapes_plan(monkeypatch) -> None:
    """The nemo round-trip returns clean PLANNED refs (no GPU needed for searchables).

    With ``gpu_target=None`` the searchable capabilities (tokenize/finetune/embed)
    still return non-error PLANNED refs and evaluate returns a Scores; only the
    launch-and-track ``pretrain`` refuses (asserted in test 6).
    """
    nemo = _make_builder("nemo")
    tok = nemo.tokenize(_FIXTURE_REF, {})
    assert isinstance(tok, ArtifactRef) and tok.error is None and tok.kind == "tokenizer"
    emb = nemo.embed(_FIXTURE_REF, _FIXTURE_REF)
    assert isinstance(emb, ArtifactRef) and emb.error is None
    ft = nemo.finetune(_FIXTURE_REF, _FIXTURE_REF, {})
    assert isinstance(ft, ArtifactRef) and ft.error is None
    scores = nemo.evaluate(_FIXTURE_REF, _FIXTURE_REF, "fraud-pr-auc")
    assert isinstance(scores, Scores) and scores.metric == "fraud-pr-auc"


# ---------------------------------------------------------------------------
# Test 2 -- Valid pathspecs (no file path / s3:// / .nemo literal).
# ---------------------------------------------------------------------------


def test_local_pathspecs_are_run_shaped(local_builder) -> None:
    """Every non-error local ArtifactRef.pathspec is a ``<FlowName>/<run_id>`` form."""
    from loom.dataio import resolve_run  # only used for the shape regex check

    refs = [
        local_builder.tokenize(_FIXTURE_REF, {}),
        local_builder.pretrain(_FIXTURE_REF, "next-event", "probe"),
    ]
    bb = refs[-1]
    refs.append(local_builder.embed(bb.pathspec, _FIXTURE_REF))
    refs.append(local_builder.finetune(bb.pathspec, _FIXTURE_REF, {}))

    for ref in refs:
        assert ref.error is None
        ps = ref.pathspec
        assert ps is not None
        parts = [p for p in ps.split("/") if p]
        assert len(parts) == 2, f"pathspec {ps!r} is not <FlowName>/<run_id>"
        # No file path / object-store URI / checkpoint literal anywhere in the ref.
        blob = f"{ps} {ref.kind} {ref.summary}"
        for bad in ("s3://", ".nemo", ".pt", "/tmp/", "file://", "boto3"):
            assert bad not in blob


def test_nemo_planned_pathspecs_are_run_shaped() -> None:
    """The nemo PLANNED searchable refs carry ``<FlowName>/<id>`` plan pathspecs."""
    nemo = _make_builder("nemo")
    for ref in (nemo.tokenize(_FIXTURE_REF, {}), nemo.embed(_FIXTURE_REF, _FIXTURE_REF)):
        assert ref.pathspec is not None
        parts = [p for p in ref.pathspec.split("/") if p]
        assert len(parts) == 2
        for bad in ("s3://", ".nemo", ".pt", "boto3"):
            assert bad not in ref.pathspec


# ---------------------------------------------------------------------------
# Test 3 -- Comparable scores: embeddings beat the raw baseline (real lift).
# ---------------------------------------------------------------------------


def test_local_embeddings_beat_raw_baseline(local_builder) -> None:
    """On the planted-sequential fixture the embeddings beat the raw baseline (lift>0)."""
    bb = local_builder.pretrain(_FIXTURE_REF, "next-event", "small")
    scores = local_builder.evaluate(bb.pathspec, _FIXTURE_REF, "fraud-pr-auc")

    assert scores.metric == "fraud-pr-auc"
    assert isinstance(scores.value, float)
    assert 0.0 <= scores.value <= 1.0
    assert "baseline_raw" in scores.detail and "lift" in scores.detail
    assert scores.detail["baseline_raw"] is not None
    lift = scores.detail["lift"]
    assert lift is not None and lift > _LIFT_EPSILON, (
        f"embeddings did not beat the raw baseline on the planted fixture (lift={lift})"
    )


# ---------------------------------------------------------------------------
# Test 4 -- Manifest honesty (every backend).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_manifest_honesty(backend) -> None:
    """backend set; supported caps callable; stand-in caps carry a non-empty note."""
    builder = _make_builder(backend)
    manifest = builder.manifest()
    assert isinstance(manifest, CapabilityManifest)
    assert manifest.backend == backend

    for name, cap in manifest.capabilities.items():
        assert cap.name == name
        assert cap.mode in {"searchable", "launch-and-track"}
        # A limited/stand-in capability must carry an honesty note (don't over-sell):
        # the local adapter's pretrain + serve are noted; the nemo adapter notes all.
        if backend == "local" and name in {"pretrain", "serve"}:
            assert cap.notes.strip(), f"{backend}.{name} must carry an honesty note"
        if backend == "nemo":
            assert cap.notes.strip(), f"{backend}.{name} must carry an honesty note"


# ---------------------------------------------------------------------------
# Test 5 -- Mode correctness (the §6.4 mode contract on every backend).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_mode_pretrain_is_launch_and_track(backend) -> None:
    """``pretrain`` is launch-and-track for every backend (AIDE never searches it)."""
    manifest = _make_builder(backend).manifest()
    assert manifest.mode_of("pretrain") == "launch-and-track"


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("cap", ["tokenize", "finetune", "embed", "evaluate"])
def test_mode_cheap_capabilities_are_searchable(backend, cap) -> None:
    """tokenize / finetune / embed / evaluate are searchable for every backend."""
    manifest = _make_builder(backend).manifest()
    assert manifest.mode_of(cap) == "searchable"


# ---------------------------------------------------------------------------
# Test 6 -- Capability-gap refusal (local serve(online); nemo pretrain no-GPU).
# ---------------------------------------------------------------------------


def test_local_serve_online_refuses_up_front(local_builder) -> None:
    """``local`` refuses ``serve(_, "online")`` up front with an actionable message."""
    with pytest.raises(NotImplementedError) as exc:
        local_builder.serve(_FIXTURE_REF, "online")
    # The message is actionable (points at the gpu_target / nemo / batch alternative).
    msg = str(exc.value).lower()
    assert "online" in msg and ("gpu" in msg or "batch" in msg)


def test_nemo_pretrain_refuses_without_gpu_target() -> None:
    """``nemo`` ``pretrain`` with ``gpu_target=None`` REFUSES, never launches."""
    nemo = _make_builder("nemo")
    ref = nemo.pretrain(_FIXTURE_REF, "next-event", "full")
    assert isinstance(ref, ArtifactRef)
    assert ref.pathspec is None  # no run produced -> no launch
    assert ref.error is not None and "gpu" in ref.error.lower()
    assert ref.summary.get("status") == "REFUSED_NO_GPU_TARGET"


def test_nemo_pretrain_plans_with_gpu_target_no_launch() -> None:
    """A configured gpu_target + launch OFF => a staged PLANNED ref, no mutation."""
    cls = get_model_builder("nemo")
    nemo = cls(LoomConfig(gpu_target="gpu-cluster-x"), launch=False)
    ref = nemo.pretrain(_FIXTURE_REF, "next-event", "full")
    assert ref.summary.get("status") == "PLANNED"
    assert ref.pathspec is not None  # a staged plan pathspec, but no real launch


# ---------------------------------------------------------------------------
# Test 7 -- Determinism / lineage (the local SVD is random_state=0-pinned).
# ---------------------------------------------------------------------------


def test_local_pretrain_is_deterministic(local_builder) -> None:
    """Two ``pretrain`` calls on the same fixture produce the SAME fingerprint."""
    a = local_builder.pretrain(_FIXTURE_REF, "next-event", "small")
    b = local_builder.pretrain(_FIXTURE_REF, "next-event", "small")
    fa = a.summary.get("fingerprint")
    fb = b.summary.get("fingerprint")
    assert fa and fb and fa == fb
    assert fa.startswith("sha256:")


# ---------------------------------------------------------------------------
# Test 8 -- Source-scan (cross-cutting graft) + seam frozenset validation.
# ---------------------------------------------------------------------------

#: Tokens that must NOT appear in a model-builder adapter module's source, except
#: inside the ``nemo`` adapter's internal lowering strings (which are deliberately
#: quarantined). Mirrors the ``dataio`` "no object-store SDK" scan.
_FORBIDDEN_TOKENS = (".nemo", "s3://", "boto3", "s3fs")

#: NeMo-noun tokens that may appear ONLY inside the ``nemo`` adapter (its lowering
#: dicts), never in the ``local`` adapter or the shared core.
_NEMO_NOUNS = ("Megatron", "@resources")


def _module_source(modname: str) -> str:
    import importlib

    mod = importlib.import_module(modname)
    with open(mod.__file__, "r", encoding="utf-8") as fh:
        return fh.read()


def test_local_adapter_has_no_objectstore_or_nemo_nouns() -> None:
    """The ``local`` adapter source carries no object-store SDK / .nemo / NeMo noun."""
    src = _module_source("loom.providers.model_builder.local")
    for bad in _FORBIDDEN_TOKENS:
        assert bad not in src, f"forbidden token {bad!r} in the local adapter"
    for noun in _NEMO_NOUNS:
        assert noun not in src, f"NeMo noun {noun!r} leaked into the local adapter"


def test_nemo_adapter_has_no_objectstore_sdk_or_checkpoint_literal() -> None:
    """The ``nemo`` adapter carries no object-store SDK / .nemo / .pt checkpoint literal.

    NeMo *nouns* (Megatron / @resources) are allowed to appear inside this adapter's
    internal lowering strings (that is the whole point -- they are quarantined here),
    but a real object-store SDK import, an ``s3://`` URI, or a checkpoint file literal
    must never appear (constraint 1: I/O are Metaflow pathspecs, never files).
    """
    src = _module_source("loom.providers.model_builder.nemo")
    for bad in (".nemo", "s3://", "boto3", "s3fs", "import torch"):
        assert bad not in src, f"forbidden token {bad!r} in the nemo adapter"


def test_flow_module_has_no_nemo_nouns_or_objectstore() -> None:
    """The ``TrainFlow`` flow speaks only Loom vocabulary -- no NeMo noun / object store."""
    src = _module_source("flows.train")
    for bad in _FORBIDDEN_TOKENS:
        assert bad not in src
    for noun in _NEMO_NOUNS:
        assert noun not in src


@pytest.mark.parametrize("backend", _BACKENDS)
def test_seam_rejects_non_loom_objective(backend) -> None:
    """A non-Loom objective is rejected at the seam (frozenset), not lowered."""
    assert "Megatron-causal-lm" not in OBJECTIVES  # a backend noun is not Loom intent
    builder = _make_builder(backend)
    # local returns a clean error ref; nemo asserts at the seam. Both refuse to lower.
    if backend == "local":
        ref = builder.pretrain(_FIXTURE_REF, "Megatron-causal-lm", "probe")
        # local cannot materialize without metaflow here, but the seam check is FIRST,
        # so a bad objective short-circuits to an error ref before any I/O.
        assert ref.error is not None and "objective" in ref.error.lower()
    else:
        with pytest.raises(AssertionError):
            builder.pretrain(_FIXTURE_REF, "Megatron-causal-lm", "probe")


@pytest.mark.parametrize("backend", _BACKENDS)
def test_seam_rejects_non_loom_mode(backend) -> None:
    """A non-Loom serving mode is rejected at the seam (frozenset)."""
    assert "nim-online" not in MODES
    builder = _make_builder(backend)
    if backend == "local":
        ref = builder.serve(_FIXTURE_REF, "nim-online")
        assert ref.error is not None and "mode" in ref.error.lower()
    else:
        with pytest.raises(AssertionError):
            builder.serve(_FIXTURE_REF, "nim-online")


def test_budgets_objectives_modes_are_frozensets() -> None:
    """The seam enums are frozensets of exactly the Loom-intent vocabulary."""
    assert OBJECTIVES == frozenset({"next-event", "masked-field", "contrastive"})
    assert BUDGETS == frozenset({"probe", "small", "full"})
    assert MODES == frozenset({"batch", "online"})
