"""``pretrain`` — the gated, budget-enveloped model-builder launch verb
(ARCHITECTURE §3/§4/§10-step6).

Acceptance per the build brief:
  * first call ``pretrain --in <corpus> --model-builder local`` → ``status=PLAN``
    + a ``confirm_token``, and NO Checkpoint is written;
  * second call with the token → ``status=OK`` / ``PASS`` + a Checkpoint object
    whose ``signatures`` carry the echoed ``representation_signature``;
  * an AGENT-originated launch (``dispatch`` with ``--launch``) →
    ``REFUSED_AGENT_CANNOT_LAUNCH``.

All CPU, no GPU, no NeMo: the ``local`` PPMI+SVD rehearsal builder is the oracle.
"""

from __future__ import annotations

import os

import pandas as pd

import loom.verbs  # noqa: F401 - registers prepare/tokenize/ingest/baseline/pretrain
from loom.registry import REGISTRY, VerbContext
from loom.store import ObjectStore
from loom.tools import dispatch
from loom.types import Status, Verdict


def _make_corpus(store: ObjectStore, df: pd.DataFrame) -> str:
    ctx = VerbContext(store=store, driver="cli", interactive=True, extras={"dataframe": df})
    res = REGISTRY["tokenize"].fn({"preset": "chain", "context_len": 256}, ctx)
    assert res.outputs, f"tokenize did not write a Corpus: {res.status}"
    return str(res.outputs[0])


def test_pretrain_registers_expensive_launch_and_track():
    """The verb auto-gates the agent the instant it registers (HARD INVARIANT #5)."""
    from loom.tools import tool_schema
    from loom.types import CapabilityMode, Tier

    v = REGISTRY["pretrain"]
    assert v.tier is Tier.EXPENSIVE
    assert v.capability_mode is CapabilityMode.LAUNCH_AND_TRACK
    assert tool_schema(v)["_loom"]["disable_model_invocation"] is True


def test_first_call_plans_and_writes_no_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    store = ObjectStore(str(tmp_path))
    corpus = _make_corpus(store, _dex())

    ctx = VerbContext(store=store, driver="cli", interactive=True)
    res = REGISTRY["pretrain"].fn({"in": corpus, "model_builder": "local"}, ctx)

    assert res.status is Status.PLAN
    assert res.confirm_token, "a PLAN must carry a single-use confirm_token"
    assert res.outputs == [], "no Checkpoint may be written on the PLAN call"
    assert store.list("Checkpoint") == [], "the store must hold no Checkpoint after a PLAN"
    assert res.cost_plan is not None and res.cost_plan.derived is True


def test_second_call_with_token_launches_and_echoes_signature(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    store = ObjectStore(str(tmp_path))
    corpus = _make_corpus(store, _dex())
    corpus_sig = store.get(corpus).signatures.get("vocab_hash")

    plan = REGISTRY["pretrain"].fn(
        {"in": corpus, "model_builder": "local"},
        VerbContext(store=store, driver="cli", interactive=True),
    )
    token = plan.confirm_token

    res = REGISTRY["pretrain"].fn(
        {"in": corpus, "model_builder": "local", "launch": True, "confirm_token": token},
        VerbContext(store=store, driver="cli", interactive=True),
    )
    assert res.status is Status.OK
    assert res.verdict is Verdict.PASS
    assert len(res.outputs) == 1
    ckpt = store.get(str(res.outputs[0]))
    assert ckpt.kind == "Checkpoint"
    # The checkpoint↔representation pairing invariant (HARD INVARIANT #7).
    assert "representation_signature" in ckpt.signatures
    assert ckpt.signatures["representation_signature"] == corpus_sig
    assert "model_signature" in ckpt.signatures
    assert ckpt.signatures.get("fmt") == "hf-safetensors-consolidated"


def test_checkpoint_safetensors_round_trips(tmp_path, monkeypatch):
    """C5 for the LOCAL builder: a valid consolidated safetensors that re-reads."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    store = ObjectStore(str(tmp_path))
    corpus = _make_corpus(store, _dex())
    plan = REGISTRY["pretrain"].fn(
        {"in": corpus, "model_builder": "local"},
        VerbContext(store=store, driver="cli", interactive=True),
    )
    res = REGISTRY["pretrain"].fn(
        {"in": corpus, "model_builder": "local", "launch": True, "confirm_token": plan.confirm_token},
        VerbContext(store=store, driver="cli", interactive=True),
    )
    ckpt = store.get(str(res.outputs[0]))
    uri = ckpt.extras["checkpoint"]["uri"]

    from safetensors.numpy import load_file

    mat = load_file(os.path.join(uri, "model.safetensors"))
    assert mat, "the consolidated safetensors must re-read with a real tensor"
    assert os.path.exists(os.path.join(uri, "config.json"))


def test_agent_launch_is_refused(tmp_path, monkeypatch):
    """An agent may PLAN but structurally cannot mint a launch (HARD INVARIANT #5)."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    store = ObjectStore(str(tmp_path))
    corpus = _make_corpus(store, _dex())

    res = dispatch(
        "loom.pretrain",
        {"in": corpus, "model_builder": "local", "launch": True, "confirm_token": "anything"},
    )
    assert res.status is Status.REFUSED_AGENT_CANNOT_LAUNCH
    assert res.verdict is Verdict.FAIL
    assert store.list("Checkpoint") == []


def test_agent_can_still_plan(tmp_path, monkeypatch):
    """The agent face still gets a PLAN (it just can't launch)."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    store = ObjectStore(str(tmp_path))
    corpus = _make_corpus(store, _dex())
    res = dispatch("loom.pretrain", {"in": corpus, "model_builder": "local"})
    assert res.status is Status.PLAN
    assert res.confirm_token


def test_spend_cap_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    store = ObjectStore(str(tmp_path))
    corpus = _make_corpus(store, _dex())
    # A vanishingly small cap below the derived (nonzero) CPU estimate is refused.
    res = REGISTRY["pretrain"].fn(
        {"in": corpus, "model_builder": "local", "max_usd": 1e-15},
        VerbContext(store=store, driver="cli", interactive=True),
    )
    assert res.status is Status.REFUSED_SPEND_CAP


def test_confirm_token_is_single_use(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    store = ObjectStore(str(tmp_path))
    corpus = _make_corpus(store, _dex())
    plan = REGISTRY["pretrain"].fn(
        {"in": corpus, "model_builder": "local"},
        VerbContext(store=store, driver="cli", interactive=True),
    )
    tok = plan.confirm_token
    first = REGISTRY["pretrain"].fn(
        {"in": corpus, "model_builder": "local", "launch": True, "confirm_token": tok},
        VerbContext(store=store, driver="cli", interactive=True),
    )
    replay = REGISTRY["pretrain"].fn(
        {"in": corpus, "model_builder": "local", "launch": True, "confirm_token": tok},
        VerbContext(store=store, driver="cli", interactive=True),
    )
    assert first.status is Status.OK
    # A consumed (burned) token cannot launch again — it falls back to a fresh PLAN.
    assert replay.status is Status.PLAN


def test_missing_corpus_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    store = ObjectStore(str(tmp_path))
    res = REGISTRY["pretrain"].fn(
        {"in": "Corpus/9999", "model_builder": "local"},
        VerbContext(store=store, driver="cli", interactive=True),
    )
    assert res.status is Status.REFUSED_CONTRACT


# A module-level df builder so each test gets a fresh frame without the fixture
# plumbing (the fixture is also available for explicit use).
def _dex() -> pd.DataFrame:
    rows = [
        ("0xa1", "2026-06-01 00:00:00", "DEXETH", "BUY", "WETH", 120.0),
        ("0xa1", "2026-06-01 00:05:00", "DEXETH", "SELL", "WETH", 118.5),
        ("0xa1", "2026-06-02 12:00:00", "DEXBASE", "BUY", "USDC", 5000.0),
        ("0xb2", "2026-06-01 09:00:00", "DEXSOL", "BUY", "SOL", 42.0),
        ("0xb2", "2026-06-03 21:30:00", "DEXSOL", "SELL", "SOL", 40.0),
    ]
    df = pd.DataFrame(rows, columns=["wallet", "timestamp", "venue", "side", "item", "size_usd"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
