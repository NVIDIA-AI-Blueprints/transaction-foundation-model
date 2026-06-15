"""Tests for the collab verb: the pure bundle logic + CLI arg-parsing.

The bundle *logic* is factored out of :class:`flows.collab.CollabFlow` into the
module-level pure functions :func:`flows.collab.build_bundle` and
:func:`flows.collab.sanitize_bundle`, so they are unit-testable on a small in-memory
report dict with **no Metaflow involved**. These tests pin:

* bundle assembly -- the sanitized payload + the lineage manifest (source ref +
  card path + fingerprint + commit);
* that the off-box SEND is OFF by default (``build_bundle`` never sets ``sent``;
  the default verdict is ``BUILT`` not ``SENT``);
* sanitization -- secret-looking keys are redacted, long strings are truncated, and
  raw non-JSON objects are reduced to a short repr (no raw rows / secrets ride
  off-box);
* the bundle round-tripping through JSON.

The bundle logic is pure-Python (dicts only) -- no pandas/Metaflow needed. The CLI
arg-parse tests are pure-Python too: they only exercise the argparse wiring for
``loom collab`` (the mutually-exclusive --run / --experiment, --send off by default).
"""

from __future__ import annotations

import json

import pytest

from loom.cli import _build_parser
from flows.collab import build_bundle, sanitize_bundle


# ---------------------------------------------------------------------------
# Pure bundle assembly + sanitization (no Metaflow).
# ---------------------------------------------------------------------------


def _report() -> dict:
    """A small validate-style report dict (the model-card payload)."""
    return {
        "target": "y",
        "task_type": "binary",
        "metric": "roc_auc",
        "holdout": {"score": 0.91, "n": 60},
        "verdict": "PASS",
        "leakage": False,
    }


def test_build_bundle_assembles_payload_and_lineage() -> None:
    """The bundle carries the sanitized payload + a lineage manifest."""
    bundle = build_bundle(
        "ValidateFlow/12",
        _report(),
        card_path="/cards/abc",
        fingerprint="deadbeef",
        commit="abc123",
    )
    assert bundle["source_ref"] == "ValidateFlow/12"
    assert bundle["payload"]["verdict"] == "PASS"
    lineage = bundle["lineage"]
    assert lineage == {
        "source_ref": "ValidateFlow/12",
        "card_path": "/cards/abc",
        "fingerprint": "deadbeef",
        "commit": "abc123",
    }


def test_build_bundle_send_off_by_default() -> None:
    """With ``send`` unset, the bundle is build-only: not sent, verdict BUILT."""
    bundle = build_bundle("ValidateFlow/12", _report())
    assert bundle["send"] is False
    assert bundle["sent"] is False
    assert bundle["verdict"] == "BUILT"
    # No sink resolved means none configured; build-only never sends regardless.
    assert bundle["sink"] is None


def test_build_bundle_send_requested_records_intent_not_sent() -> None:
    """``send=True`` records the SEND_REQUESTED intent but never performs the send itself."""
    bundle = build_bundle(
        "ValidateFlow/12", _report(), send=True, sink="https://example.test/hook"
    )
    assert bundle["send"] is True
    # The pure function records intent + the would-send sink, but never sets sent.
    assert bundle["sent"] is False
    assert bundle["verdict"] == "SEND_REQUESTED"
    assert bundle["sink"] == "https://example.test/hook"


def test_build_bundle_empty_report_yields_empty_payload() -> None:
    """A missing/None report yields an empty (but well-formed) payload."""
    bundle = build_bundle("ValidateFlow/12", None)
    assert bundle["payload"] == {}
    assert bundle["verdict"] == "BUILT"


def test_sanitize_redacts_secret_keys() -> None:
    """Secret-looking keys are redacted, not carried off-box."""
    dirty = {
        "metric": 0.9,
        "api_key": "sk-very-secret",
        "nested": {"password": "hunter2", "ok": 1},
        "authorization": "Bearer xyz",
    }
    clean = sanitize_bundle(dirty)
    assert clean["metric"] == 0.9
    assert clean["api_key"] == "<redacted>"
    assert clean["authorization"] == "<redacted>"
    assert clean["nested"]["password"] == "<redacted>"
    assert clean["nested"]["ok"] == 1
    # No secret value survives anywhere in the sanitized structure.
    assert "sk-very-secret" not in json.dumps(clean)
    assert "hunter2" not in json.dumps(clean)


def test_sanitize_truncates_long_strings() -> None:
    """A very long string (e.g. a pasted log / raw-row dump) is truncated."""
    blob = "x" * 10000
    clean = sanitize_bundle({"notes": blob})
    assert len(clean["notes"]) < len(blob)
    assert len(clean["notes"]) <= 2000


def test_sanitize_reduces_non_json_objects_to_repr() -> None:
    """A non-JSON object (a fake DataFrame) is reduced to a short repr, not raw rows."""

    class _FakeFrame:
        def __repr__(self) -> str:  # pragma: no cover - exercised via sanitize
            return "<FakeFrame rows=1000000>"

    clean = sanitize_bundle({"df": _FakeFrame(), "n": 3})
    assert isinstance(clean["df"], str)
    assert clean["df"].startswith("<FakeFrame")
    assert clean["n"] == 3


def test_build_bundle_is_json_able() -> None:
    """The whole bundle round-trips through JSON (suitable for a RunResult summary)."""
    bundle = build_bundle(
        "ValidateFlow/12", _report(), card_path="/cards/abc", fingerprint="fp"
    )
    assert json.loads(json.dumps(bundle)) == bundle


# ---------------------------------------------------------------------------
# CLI arg-parsing (pure-Python, no pandas/Metaflow).
# ---------------------------------------------------------------------------


def test_cli_collab_parses_run_and_send() -> None:
    """`loom collab --run ... --send` parses into the collab handler."""
    from loom.cli import _cmd_collab

    parser = _build_parser()
    args = parser.parse_args(["collab", "--run", "ValidateFlow/12", "--send"])
    assert args.command == "collab"
    assert args.run_pathspec == "ValidateFlow/12"
    assert args.send is True
    assert args.func is _cmd_collab


def test_cli_collab_send_off_by_default() -> None:
    """--send is a store_true flag that defaults to False (build-only)."""
    parser = _build_parser()
    args = parser.parse_args(["collab", "--run", "ValidateFlow/12"])
    assert args.send is False


def test_cli_collab_parses_experiment() -> None:
    """`loom collab --experiment ID` parses the experiment id."""
    parser = _build_parser()
    args = parser.parse_args(["collab", "--experiment", "exp1"])
    assert args.experiment == "exp1"
    assert args.run_pathspec is None


def test_cli_collab_requires_one_source() -> None:
    """`loom collab` requires exactly one of --run / --experiment (mutually exclusive)."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["collab"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["collab", "--run", "ValidateFlow/12", "--experiment", "exp1"]
        )
