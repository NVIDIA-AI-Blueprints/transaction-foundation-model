"""Tests for the ``/loom-train`` verb: the model-builder lift + the no-GPU refusal.

These are the two load-bearing self-tests the ``loom-train`` skill's exit gate names
(design §6.3), plus the ``loom train`` CLI arg-parse wiring:

* :func:`test_train_local_roundtrip_gives_lift` -- the torch-free ``local`` adapter
  builds a backbone and embeddings end-to-end and the embeddings BEAT the raw
  baseline on a planted-sequential fixture (the real CPU lift; the ``/loom-train``
  exit gate's "the produced ArtifactRef has a valid pathspec + a measurable gain");
* :func:`test_train_nemo_refuses_without_gpu_target` -- the ``nemo`` adapter, with
  ``gpu_target=None``, returns a clean ``REFUSED_NO_GPU_TARGET`` ``ArtifactRef`` and
  **never launches** (the constraint-4 gate the exit gate asserts);
* the CLI arg-parse tests pin the ``loom train`` argparse wiring -- notably that
  ``--launch`` is OFF by default (the heavy GPU launch is opt-in).

The lift + refusal tests are pure (no Metaflow): the ``local`` adapter's two
Client-API helpers are monkeypatched to serve a small in-memory fixture, so the
PPMI+SVD path runs end-to-end on the torch-free default; the ``nemo`` refusal needs
no data at all (it refuses before any I/O). The CLI tests only exercise argparse.
"""

from __future__ import annotations

import pytest

from loom.cli import _build_parser, _cmd_train

# The lift test needs the data-science stack; importorskip keeps a stripped env from
# erroring the whole module (the CLI arg-parse tests below stay pure-Python).
pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from loom.config import LoomConfig  # noqa: E402
from loom.registry import get_model_builder  # noqa: E402
from loom.types import ArtifactRef, Scores  # noqa: E402

_FIXTURE_REF = "IngestDataset/1"


# ---------------------------------------------------------------------------
# A small planted-sequential-signal fixture (shared shape with the conformance
# suite): positive accounts follow a Markov chain, negatives are random -- so the
# pooled PPMI+SVD embedding beats a raw per-row baseline.
# ---------------------------------------------------------------------------


def _planted_fixture(n_accounts: int = 80, seed: int = 0) -> "pd.DataFrame":
    """Per-account event sequences with a planted next-event signal (see conftest doc)."""
    rng = np.random.default_rng(seed)
    alphabet = ["A", "B", "C", "D", "E"]
    chain = {"A": "B", "B": "C", "C": "D", "D": "E", "E": "A"}
    rows: list[dict] = []
    for acct in range(n_accounts):
        positive = acct % 2 == 0
        seq_len = int(rng.integers(8, 16))
        if positive:
            cur = alphabet[int(rng.integers(0, len(alphabet)))]
            events = [cur]
            for _ in range(seq_len - 1):
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
    frame = _planted_fixture(n_accounts=2)
    return {
        "columns": [str(c) for c in frame.columns],
        "dtypes": {str(c): str(frame[c].dtype) for c in frame.columns},
        "nrows": int(len(frame)),
        "target": "label",
    }


@pytest.fixture()
def local_builder(monkeypatch):
    """A ``local`` builder whose Client-API helpers serve the in-memory fixture."""
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
    monkeypatch.setattr(local_mod.LocalModelBuilderProvider, "_load_backbone", _fake_load_backbone)
    return local_mod.LocalModelBuilderProvider(LoomConfig())


# ---------------------------------------------------------------------------
# The two exit-gate self-tests the loom-train SKILL.md names.
# ---------------------------------------------------------------------------


def test_train_local_roundtrip_gives_lift(local_builder) -> None:
    """The local adapter builds a backbone whose embeddings BEAT the raw baseline.

    The exit-gate self-test for ``/loom-train`` on the torch-free default path: a
    pretrain produces a valid backbone ``ArtifactRef`` (a run-shaped pathspec, no
    error), and ``evaluate`` on the planted-sequential fixture returns a measurable
    POSITIVE lift over the raw baseline -- the produced model is genuinely better,
    not a no-op stand-in.
    """
    backbone = local_builder.pretrain(_FIXTURE_REF, "next-event", "small")
    assert isinstance(backbone, ArtifactRef)
    assert backbone.error is None
    assert backbone.kind == "backbone"
    # The backbone ref is a valid <FlowName>/<run_id> pathspec, never a file/URI.
    assert backbone.pathspec is not None
    assert len([p for p in backbone.pathspec.split("/") if p]) == 2

    scores = local_builder.evaluate(backbone.pathspec, _FIXTURE_REF, "fraud-pr-auc")
    assert isinstance(scores, Scores)
    assert scores.metric == "fraud-pr-auc"
    assert isinstance(scores.value, float) and 0.0 <= scores.value <= 1.0

    lift = scores.detail.get("lift")
    baseline = scores.detail.get("baseline_raw")
    assert baseline is not None
    assert lift is not None and lift > 0.0, (
        f"embeddings did not beat the raw baseline (value={scores.value}, "
        f"baseline={baseline}, lift={lift})"
    )


def test_train_nemo_refuses_without_gpu_target() -> None:
    """The nemo adapter refuses to launch with no GPU target (constraint 4).

    The exit-gate self-test for the gated launch path: with ``gpu_target=None`` and
    even at ``budget="full"``, ``pretrain`` returns a clean ``REFUSED_NO_GPU_TARGET``
    ``ArtifactRef`` -- ``pathspec is None`` (no run produced => no launch), an
    actionable error message, and the refusal status in the summary. The gate refuses
    rather than launching GPU work.
    """
    cls = get_model_builder("nemo")
    nemo = cls(LoomConfig(model_builder_provider="nemo", gpu_target=None), launch=False)

    ref = nemo.pretrain(_FIXTURE_REF, "next-event", "full")
    assert isinstance(ref, ArtifactRef)
    assert ref.pathspec is None  # nothing launched, nothing produced
    assert ref.error is not None and "gpu" in ref.error.lower()
    assert ref.summary.get("status") == "REFUSED_NO_GPU_TARGET"

    # Even when --launch is requested, no GPU target still refuses (never launches).
    nemo_launch = cls(LoomConfig(gpu_target=None), launch=True)
    ref2 = nemo_launch.pretrain(_FIXTURE_REF, "next-event", "full")
    assert ref2.pathspec is None
    assert ref2.summary.get("status") == "REFUSED_NO_GPU_TARGET"


def test_train_nemo_surfaces_cost_at_the_gate() -> None:
    """The refusal still surfaces the PHYSICS (GPU-hours / $) at the gate (constraint 5).

    The abstraction hides vocabulary, not physics: even when refusing, the plan in the
    summary carries the cost estimate (``budget="full"`` => 8 GPU x 12 h => 96
    GPU-hours), so a caller sees the true cost the launch would incur.
    """
    cls = get_model_builder("nemo")
    nemo = cls(LoomConfig(gpu_target=None), launch=False)
    ref = nemo.pretrain(_FIXTURE_REF, "next-event", "full")
    cost = (ref.summary.get("plan") or {}).get("cost") or ref.summary.get("cost") or {}
    assert cost.get("gpu_count") == 8
    assert cost.get("gpu_hours") == 96.0
    assert cost.get("est_usd") and cost["est_usd"] > 0


# ---------------------------------------------------------------------------
# CLI arg-parse wiring for ``loom train`` (pure argparse; no flow run).
# ---------------------------------------------------------------------------


def test_cli_train_parses_all_flags() -> None:
    """``loom train`` parses every flag and routes to ``_cmd_train``."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            "train",
            "--dataset",
            "IngestDataset/123",
            "--objective",
            "masked-field",
            "--budget",
            "full",
            "--capability",
            "pretrain",
            "--backbone",
            "TrainFlow/7",
            "--metric",
            "fraud-pr-auc",
            "--launch",
        ]
    )
    assert args.command == "train"
    assert args.dataset == "IngestDataset/123"
    assert args.objective == "masked-field"
    assert args.budget == "full"
    assert args.capability == "pretrain"
    assert args.backbone_ref == "TrainFlow/7"
    assert args.metric == "fraud-pr-auc"
    assert args.launch is True
    assert args.func is _cmd_train


def test_cli_train_launch_off_by_default() -> None:
    """``--launch`` is OFF by default -- the real heavy GPU launch is opt-in."""
    parser = _build_parser()
    args = parser.parse_args(["train", "--dataset", "IngestDataset/9"])
    assert args.launch is False


def test_cli_train_optional_flags_default_none() -> None:
    """The optional objective/budget/capability/backbone/metric flags default to None."""
    parser = _build_parser()
    args = parser.parse_args(["train", "--dataset", "IngestDataset/9"])
    assert args.objective is None
    assert args.budget is None
    assert args.capability is None
    assert args.backbone_ref is None
    assert args.metric is None


def test_cli_train_requires_dataset() -> None:
    """``loom train`` requires ``--dataset`` (argparse exits non-zero without it)."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--objective", "next-event"])
