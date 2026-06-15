"""Scaffold smoke tests — lock the public wiring the Implement agents build on.

These assert the narrow waist is wired (verbs self-register), the dual-driver
envelope round-trips, pathspecs parse, the CLI builds, and the engine API names
exist as the locked contract. They do NOT exercise the (stubbed) verb bodies."""

from __future__ import annotations

import json

import loom
from loom import REGISTRY, DataObjectRef, Status, Tier, Verdict, VerbResult
from loom.cli import build_parser
from loom.tools import all_tool_schemas, dispatch


def test_three_verbs_registered():
    for name in ("tokenize", "ingest", "baseline"):
        assert name in REGISTRY, f"{name} did not self-register"
    assert REGISTRY["tokenize"].tier is Tier.WORKSPACE_WRITE


def test_pathspec_roundtrip():
    ref = DataObjectRef.parse("Corpus/204")
    assert ref.kind == "Corpus" and ref.n == 204
    assert ref.pathspec == "Corpus/204"
    assert str(ref) == "Corpus/204"


def test_envelope_json_is_stable_and_complete():
    r = VerbResult(
        verb="tokenize",
        status=Status.OK,
        verdict=Verdict.PASS,
        tier=Tier.WORKSPACE_WRITE,
        capability_mode=loom.CapabilityMode.NONE,
        outputs=[DataObjectRef("Corpus", 204)],
        summary="ok",
    )
    payload = json.loads(r.to_json())
    for key in ("verb", "status", "verdict", "tier", "capability_mode",
                "summary", "outputs", "diagnostics", "data", "experiment",
                "cost_plan", "confirm_token"):
        assert key in payload, f"envelope missing {key}"
    assert payload["outputs"] == ["Corpus/204"]


def test_cli_parser_builds_with_all_verbs():
    parser = build_parser()
    # argparse exposes subcommands via the _subparsers action; just ensure it builds
    # and --help would list our verbs (smoke: parsing a known verb works).
    ns = parser.parse_args(["tokenize", "IngestDataset/1"])
    assert ns.verb == "tokenize"


def test_cli_json_matches_tool_result():
    """The --json envelope must equal the agent tool result (byte-identical, §2.1)."""
    cli_result = REGISTRY["tokenize"].fn({"in": "IngestDataset/1"},
                                         _dummy_ctx())
    tool_result = dispatch("loom.tokenize", {"in": "IngestDataset/1"})
    assert cli_result.to_json() == tool_result.to_json()


def test_tool_schemas_emitted_for_every_verb():
    schemas = all_tool_schemas()
    names = {s["name"] for s in schemas}
    assert {"loom.tokenize", "loom.ingest", "loom.baseline"} <= names
    for s in schemas:
        assert "input_schema" in s and "_loom" in s


def test_engine_api_names_exist():
    from loom import engine
    for name in ("compile_spec", "financial_spec", "chain_spec", "TokenizerSpec",
                 "FieldStep", "FixedVocab", "Hash", "MappingRange", "MappingDirect",
                 "MappingPassthrough", "TimeDelta", "CompiledTokenizer",
                 "ContractReport", "SPECIAL_TOKENS"):
        assert hasattr(engine, name), f"engine API missing {name}"
    assert engine.SPECIAL_TOKENS == ("<pad>", "<bos>", "<eos>", "<sep>", "<unk>")


def _dummy_ctx():
    from loom.registry import VerbContext
    from loom.store import ObjectStore
    return VerbContext(store=ObjectStore("/tmp/loom-test-ws"), driver="cli")
