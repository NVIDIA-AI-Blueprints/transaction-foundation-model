"""Tests for the machine-readable CLI contract (loom-on-pi §7).

These guard the stable boundary the Pi-based agentic CLI consumes:

  1. ``--json`` on a lifecycle verb emits exactly ONE JSON object to stdout with
     the typed envelope fields (verb/status/VERDICT/pathspec/card_path/summary/
     gate/error), reusing the verb's EXISTING typed ``summary`` dict, and keeps
     the SAME exit code as the prose path.
  2. The prose (non-``--json``) path is unchanged: human text on stdout, no JSON.
  3. ``loom verbs --json`` emits the MANIFEST (one entry per lifecycle verb, each
     with name/summary/required/optional/tier/disable_model_invocation); the four
     irreversible verbs (deploy/train/collab/skillopt) carry
     ``disable_model_invocation == True`` AND ``tier == "irreversible"``, and a
     read-only verb (eda) carries ``tier == "read-only"``.
  4. Arg-parse coverage for the ``--json`` flag and the ``verbs`` subcommand.

The tests are HEADLESS -- no Metaflow, no cluster, no LLM. ``loom doctor`` is a
pure read-only verb whose ``--json`` path runs end-to-end without a datastore, so
it exercises the one-object / exit-code / prose-on-stderr contract live; ``loom
eda`` is exercised against a MOCKED execution provider so the RunResult typed
summary reuse is asserted without a real flow.
"""

from __future__ import annotations

import io
import json
import sys
from unittest import mock

import pytest

from loom import cli
from loom.types import RunResult


# ---------------------------------------------------------------------------
# Helpers: drive a verb handler with captured stdout/stderr (no Metaflow).
# ---------------------------------------------------------------------------


def _run_handler(argv: list[str]) -> tuple[int, str, str]:
    """Parse ``argv`` and invoke its handler, capturing (exit_code, stdout, stderr).

    Uses the real parser + the dispatched handler so the ``--json`` redirect (the
    ``_verb_prose`` contextmanager that sends prose to stderr) is exercised exactly
    as in production.
    """
    parser = cli._build_parser()
    args = parser.parse_args(argv)
    out, err = io.StringIO(), io.StringIO()
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        code = args.func(args)
    finally:
        sys.stdout, sys.stderr = real_out, real_err
    return code, out.getvalue(), err.getvalue()


# A representative typed EDA summary, exactly as flows.eda.EdaFlow builds it.
_FAKE_EDA_SUMMARY = {
    "nrows": 50000,
    "ncols": 31,
    "target": "y",
    "target_inferred": True,
    "leakage_flags": [{"column": "id", "kind": "id-leak", "detail": "row index"}],
}


class _FakeExecution:
    """A stand-in ExecutionProvider whose ``run_flow`` returns a fixed RunResult."""

    _result = RunResult(
        pathspec="EdaFlow/12/end/45",
        successful=True,
        card_path="/tmp/card.html",
        summary=_FAKE_EDA_SUMMARY,
        error=None,
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def run_flow(self, *args: object, **kwargs: object) -> RunResult:
        return self._result


@pytest.fixture
def mocked_eda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch out the execution provider + learnings so `loom eda` runs headless."""
    monkeypatch.setattr(cli, "get_execution", lambda name: _FakeExecution)
    monkeypatch.setattr(cli, "_record_eda_learning", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# (1) --json on a read-only verb: ONE object, typed fields, same exit code.
# ---------------------------------------------------------------------------


def test_eda_json_emits_one_typed_object(mocked_eda: None) -> None:
    """`loom eda --json` emits ONE JSON object reusing the typed summary."""
    code, stdout, stderr = _run_handler(
        ["eda", "--dataset", "IngestDataset/1", "--json"]
    )

    # Exactly one JSON object on stdout (a tool can JSON.parse the whole stream).
    assert stdout.strip().count("\n") == 0
    obj = json.loads(stdout)

    # The typed envelope fields the contract mandates.
    assert obj["verb"] == "eda"
    assert obj["status"] == "ok"
    assert obj["pathspec"] == "EdaFlow/12/end/45"
    assert obj["card_path"] == "/tmp/card.html"
    assert obj["error"] is None
    # eda has no VERDICT line and does not gate.
    assert obj["VERDICT"] is None
    assert obj["gate"] is None

    # The summary is the EXISTING typed dict, reused verbatim (not recomputed).
    assert obj["summary"] == _FAKE_EDA_SUMMARY
    assert obj["summary"]["leakage_flags"][0]["kind"] == "id-leak"

    # Human prose is routed to STDERR, never onto the JSON stdout.
    assert "Loom EDA complete." in stderr
    assert code == 0


def test_eda_json_preserves_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--json` keeps the EXACT exit code of the prose path (1 on a failed run)."""
    failed = RunResult(
        pathspec="EdaFlow/13", successful=False, summary={}, error="flow failed"
    )

    class _FailExec:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def run_flow(self, *a: object, **k: object) -> RunResult:
            return failed

    monkeypatch.setattr(cli, "get_execution", lambda name: _FailExec)
    monkeypatch.setattr(cli, "_record_eda_learning", lambda *a, **k: None)

    json_code, json_out, _ = _run_handler(
        ["eda", "--dataset", "IngestDataset/1", "--json"]
    )
    prose_code, _, _ = _run_handler(["eda", "--dataset", "IngestDataset/1"])

    # Same non-zero exit on both surfaces; the JSON marks status=error.
    assert json_code == prose_code == 1
    assert json.loads(json_out)["status"] == "error"
    assert json.loads(json_out)["error"] == "flow failed"


def test_eda_prose_path_unchanged_without_json(mocked_eda: None) -> None:
    """Without `--json` the prose path is unchanged: human text on stdout, no JSON."""
    code, stdout, _ = _run_handler(["eda", "--dataset", "IngestDataset/1"])
    assert code == 0
    assert stdout.startswith("Profiling data object")
    assert "Loom EDA complete." in stdout
    # The prose path must NOT emit a JSON object on stdout.
    with pytest.raises(json.JSONDecodeError):
        json.loads(stdout)


def test_doctor_json_is_one_object_live() -> None:
    """`loom doctor --json` (a pure read-only verb) emits ONE object, prose on stderr.

    doctor needs no Metaflow/cluster, so this exercises the one-object / exit-code
    / prose-on-stderr contract live (not mocked).
    """
    json_code, stdout, stderr = _run_handler(["doctor", "--json"])
    prose_code, prose_out, _ = _run_handler(["doctor"])

    # ONE JSON object on stdout.
    assert stdout.strip().count("\n") == 0
    obj = json.loads(stdout)
    assert obj["verb"] == "doctor"
    assert obj["status"] in ("ok", "error")
    assert obj["VERDICT"] in ("PASS", "FAIL")
    assert "checks" in obj["summary"]

    # The exit code is IDENTICAL to the prose path, and prose moved to stderr.
    assert json_code == prose_code
    assert stderr.startswith("Loom doctor")
    # The prose path prints the same banner to stdout (unchanged behavior).
    assert prose_out.startswith("Loom doctor")


# ---------------------------------------------------------------------------
# (3) loom verbs --json: the MANIFEST contract.
# ---------------------------------------------------------------------------

_MANIFEST_KEYS = {
    "name",
    "summary",
    "required",
    "optional",
    "tier",
    "disable_model_invocation",
}
_IRREVERSIBLE_DMI_VERBS = {"deploy", "train", "collab", "skillopt"}
_TIERS = {"read-only", "workspace-write", "expensive", "irreversible"}


def test_verbs_json_emits_full_manifest() -> None:
    """`loom verbs --json` emits ONE JSON array; every entry has the six fields."""
    code, stdout, _ = _run_handler(["verbs", "--json"])
    assert code == 0

    manifest = json.loads(stdout)
    assert isinstance(manifest, list)
    assert len(manifest) == 15  # the 15 flat lifecycle verbs

    for entry in manifest:
        assert set(entry.keys()) == _MANIFEST_KEYS
        assert isinstance(entry["name"], str) and entry["name"]
        assert isinstance(entry["summary"], str) and entry["summary"]
        assert isinstance(entry["required"], list)
        assert isinstance(entry["optional"], list)
        assert entry["tier"] in _TIERS
        assert isinstance(entry["disable_model_invocation"], bool)


def test_verbs_json_irreversible_and_dmi_are_the_same_four() -> None:
    """The four irreversible verbs are exactly the four disable_model_invocation verbs."""
    _, stdout, _ = _run_handler(["verbs", "--json"])
    manifest = {v["name"]: v for v in json.loads(stdout)}

    dmi = {n for n, v in manifest.items() if v["disable_model_invocation"]}
    irreversible = {n for n, v in manifest.items() if v["tier"] == "irreversible"}

    assert dmi == _IRREVERSIBLE_DMI_VERBS
    assert irreversible == _IRREVERSIBLE_DMI_VERBS
    # Each of the four carries BOTH the flag and the irreversible tier.
    for name in _IRREVERSIBLE_DMI_VERBS:
        assert manifest[name]["disable_model_invocation"] is True
        assert manifest[name]["tier"] == "irreversible"


def test_verbs_json_read_only_verb_tier() -> None:
    """A read-only verb (eda) carries tier == 'read-only' and is model-invocable."""
    _, stdout, _ = _run_handler(["verbs", "--json"])
    manifest = {v["name"]: v for v in json.loads(stdout)}

    assert manifest["eda"]["tier"] == "read-only"
    assert manifest["eda"]["disable_model_invocation"] is False
    # eda's required noun is its dataset; --json/--help are NOT in the arg lists.
    assert "dataset" in manifest["eda"]["required"]
    assert "json" not in manifest["eda"]["optional"]
    assert "help" not in manifest["eda"]["optional"]


def test_verbs_human_table_without_json() -> None:
    """`loom verbs` (no --json) prints a human table, not JSON."""
    code, stdout, _ = _run_handler(["verbs"])
    assert code == 0
    assert stdout.startswith("Loom verbs (15):")
    # The table marks the disable-model-invocation verbs and shows tiers.
    assert "[no-model-invoke]" in stdout
    assert "read-only" in stdout
    with pytest.raises(json.JSONDecodeError):
        json.loads(stdout)


# ---------------------------------------------------------------------------
# (4) Arg-parse coverage for --json + the verbs subcommand.
# ---------------------------------------------------------------------------


def test_json_flag_parses_on_every_lifecycle_verb() -> None:
    """`--json` parses to args.json on a representative verb of each kind."""
    parser = cli._build_parser()
    cases = {
        "eda": ["eda", "--dataset", "X", "--json"],
        "datasets": ["datasets", "--json"],
        "doctor": ["doctor", "--json"],
        "ingest": ["ingest", "--source", "X", "--json"],
        "deploy": ["deploy", "--validate", "ValidateFlow/1", "--json"],
    }
    for verb, argv in cases.items():
        args = parser.parse_args(argv)
        assert args.command == verb
        assert args.json is True


def test_json_defaults_off_when_absent() -> None:
    """`--json` defaults OFF so the prose path is the unchanged default."""
    parser = cli._build_parser()
    args = parser.parse_args(["eda", "--dataset", "X"])
    assert args.json is False


def test_verbs_subcommand_parses() -> None:
    """`loom verbs` and `loom verbs --json` both parse to the verbs handler."""
    parser = cli._build_parser()

    bare = parser.parse_args(["verbs"])
    assert bare.command == "verbs"
    assert bare.func is cli._cmd_verbs
    assert bare.json is False

    with_json = parser.parse_args(["verbs", "--json"])
    assert with_json.json is True


def test_verbs_carries_root_parser_for_introspection() -> None:
    """`loom verbs` is wired with the root parser it introspects for the manifest."""
    parser = cli._build_parser()
    args = parser.parse_args(["verbs"])
    # _build_parser pins the root parser onto the namespace so the manifest can be
    # built by introspecting the live subparsers.
    assert getattr(args, "_root_parser", None) is not None
