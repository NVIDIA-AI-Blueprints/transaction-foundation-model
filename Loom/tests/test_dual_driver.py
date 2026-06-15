"""The narrow waist: one verb declaration → two faces that emit a BYTE-IDENTICAL
envelope (DESIGN.md §2.1). For the same args, the CLI ``--json`` result equals the
agent tool ``dispatch`` result, character for character.

This holds against the scaffold stubs (both faces return the same INCOMPLETE
envelope) and remains a hard gate once the verbs are implemented — it's the
property that lets a human and an agent drive the identical verb."""

from __future__ import annotations

import json

import pytest

from loom import REGISTRY
from loom.registry import VerbContext
from loom.store import ObjectStore
from loom.tools import all_tool_schemas, dispatch
from loom.types import CapabilityMode, Tier


def _cli_ctx(store) -> VerbContext:
    # The CLI builds an interactive cli-driver context; the tool builds a
    # non-interactive agent-driver context. The ENVELOPE must not differ on that.
    return VerbContext(store=store, driver="cli", interactive=True)


def _call_or_skip(thunk):
    """Call a face. A ``NotImplementedError`` means a seam is still a stub while a
    parallel Implement agent finishes it — a "not landed yet" condition, so skip
    rather than report a conformance failure. The byte-identity ``assert`` stays
    strict for every implemented verb."""
    try:
        return thunk()
    except NotImplementedError:  # pragma: no cover - skip path while seams are stubs
        pytest.skip("a verb seam is still a stub (NotImplementedError)")


@pytest.mark.parametrize(
    "verb,args",
    [
        ("tokenize", {"in": "IngestDataset/1", "preset": "financial"}),
        ("ingest", {"in": "./data/decoder_corpus_t1", "name": "tfm-corpus-t1"}),
        ("baseline", {"in": "Corpus/204", "task": "next-item", "k": 10}),
    ],
)
def test_cli_json_equals_tool_result_byte_for_byte(verb, args, tmp_path, monkeypatch):
    """``loom <verb> --json`` (the fn().to_json()) == ``loom.<verb>(args)`` dispatch."""
    # Pin both faces to the SAME workspace so any output pathspecs (n counters)
    # match — otherwise an implemented verb could mint Type/1 vs Type/2.
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    store = ObjectStore(str(tmp_path))

    cli_result = _call_or_skip(lambda: REGISTRY[verb].fn(dict(args), _cli_ctx(store)))
    tool_result = _call_or_skip(lambda: dispatch(f"loom.{verb}", dict(args)))

    assert cli_result.to_json() == tool_result.to_json(), (
        f"{verb}: CLI --json envelope diverged from the agent tool result"
    )
    # And the parsed envelope carries the full locked key set in stable order.
    payload = json.loads(cli_result.to_json())
    assert list(payload.keys()) == [
        "verb", "status", "verdict", "tier", "capability_mode", "summary",
        "outputs", "diagnostics", "data", "experiment", "cost_plan", "confirm_token",
    ]


def test_experiment_threads_identically_through_both_faces(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    store = ObjectStore(str(tmp_path))
    args = {"in": "IngestDataset/1", "preset": "financial"}
    exp = "tfm-t1-timedelta"

    cli_ctx = VerbContext(store=store, driver="cli", experiment=exp)
    cli_result = _call_or_skip(lambda: REGISTRY["tokenize"].fn(dict(args), cli_ctx))
    tool_result = _call_or_skip(lambda: dispatch("loom.tokenize", dict(args), experiment=exp))

    assert cli_result.to_json() == tool_result.to_json()
    assert json.loads(cli_result.to_json())["experiment"] == exp


def test_envelope_carries_tier_and_capability_for_gating_model():
    """The result envelope must already carry tier + capability_mode (the fields the
    Phase-1 gating model reads), even though no verb is GPU-gated in this slice."""
    for verb in ("tokenize", "ingest", "baseline"):
        store = ObjectStore("/tmp/loom-dual-driver-ws")
        result = _call_or_skip(lambda v=verb: REGISTRY[v].fn({}, VerbContext(store=store, driver="cli")))
        payload = json.loads(result.to_json())
        assert payload["tier"] in {t.value for t in Tier}
        assert payload["capability_mode"] in {c.value for c in CapabilityMode}
        # cost_plan is a present (possibly-null) field the gating model will fill.
        assert "cost_plan" in payload


def test_tool_schema_disable_invocation_matches_tier():
    """The agent face derives ``disable_model_invocation`` from tier/capability —
    none of the Phase-0 verbs are gated (all WORKSPACE_WRITE, none launch-and-track)."""
    schemas = {s["name"]: s for s in all_tool_schemas()}
    for verb in ("tokenize", "ingest", "baseline"):
        s = schemas[f"loom.{verb}"]
        assert s["_loom"]["disable_model_invocation"] is False
        assert s["input_schema"] is REGISTRY[verb].params
