"""Telemetry core tests: events, attributes, the OTel no-op, and CLI arg-parse.

These exercise the always-available, dependency-light telemetry engine (no
OpenTelemetry SDK, no network):

* :func:`loom.telemetry.log_event` -- standard attributes are merged in, a
  MONOTONIC ``event.sequence`` orders events within a process, and content is
  REDACTED BY DEFAULT (un-redacted only when ``LOOM_LOG_CONTENT`` is set);
  ``log_event`` is a safe no-op (returns ``None``) when ``LOOM_TELEMETRY`` is off.
* :func:`loom.telemetry.bootstrap_otel` -- degrades cleanly to a no-op with an
  actionable message when the SDK is absent (which it is in the venv); it never
  raises and never makes the SDK a hard import.
* ``loom telemetry status|export|trace`` -- the CLI arg-parse wiring.

All file I/O is confined to ``tmp_path`` via a :class:`LoomConfig` whose
``telemetry_path`` points there, so the tests never touch the repo's corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loom.config import LoomConfig
from loom.telemetry import (
    bootstrap_otel,
    log_event,
    read_events,
    telemetry_attributes,
)
from loom.telemetry.events import redact


def _config(tmp_path: Path) -> LoomConfig:
    """A config whose telemetry path is isolated under ``tmp_path``."""
    return LoomConfig(
        telemetry_path=str(tmp_path / "telemetry" / "events.jsonl"),
        owned_by="general",
        tenant="default",
    )


@pytest.fixture
def telemetry_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable telemetry capture; content stays redacted by default."""
    monkeypatch.setenv("LOOM_TELEMETRY", "1")
    monkeypatch.delenv("LOOM_LOG_CONTENT", raising=False)


# ---------------------------------------------------------------------------
# log_event: standard attrs + the monotonic sequence + redacted-by-default.
# ---------------------------------------------------------------------------


def test_log_event_noop_when_telemetry_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With LOOM_TELEMETRY unset, log_event returns None and writes nothing."""
    monkeypatch.delenv("LOOM_TELEMETRY", raising=False)
    cfg = _config(tmp_path)

    event = log_event("llm_request", "traj-1", cfg, content="secret prompt")

    assert event is None
    assert not Path(cfg.telemetry_path).exists()


def test_log_event_writes_standard_attrs_and_monotonic_sequence(
    tmp_path: Path, telemetry_on: None
) -> None:
    """log_event merges the standard attrs and stamps a MONOTONIC event.sequence."""
    cfg = _config(tmp_path)

    log_event("trajectory.start", "traj-1", cfg, attrs={"verb": "aide"})
    log_event("llm_request", "traj-1", cfg, attrs={"model": "claude-x"})
    log_event("trajectory.end", "traj-1", cfg)

    rows = read_events(cfg)
    assert len(rows) == 3

    # Standard attributes are hoisted to the top level of every row.
    for row in rows:
        assert row["owned_by"] == "general"
        assert row["tenant"] == "default"
        assert row["service.name"] == "loom"
        assert row["trajectory_id"] == "traj-1"
        assert "event.name" in row and "event.ts" in row

    # The sequence is strictly increasing in file order (the ordering guarantee
    # assemble_trajectory relies on). It is process-monotonic, so we only assert
    # monotonicity, not specific values.
    seqs = [row["event.sequence"] for row in rows]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == 3
    assert seqs[0] < seqs[1] < seqs[2]

    # The event names round-trip in order.
    assert [r["event.name"] for r in rows] == [
        "trajectory.start",
        "llm_request",
        "trajectory.end",
    ]


def test_log_event_redacts_content_by_default(
    tmp_path: Path, telemetry_on: None
) -> None:
    """Content is REDACTED BY DEFAULT to the typed sentinel (prompt hygiene)."""
    cfg = _config(tmp_path)

    log_event(
        "llm_request",
        "traj-1",
        cfg,
        content="the raw user prompt with PII",
        content_kind="prompt",
    )

    row = read_events(cfg)[0]
    assert row["content"] == "<REDACTED:prompt>"
    assert row["content.kind"] == "prompt"
    # The raw bytes never reached disk.
    assert "PII" not in json.dumps(row)


def test_log_event_logs_content_only_with_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With LOOM_LOG_CONTENT set, the raw content passes through verbatim."""
    monkeypatch.setenv("LOOM_TELEMETRY", "1")
    monkeypatch.setenv("LOOM_LOG_CONTENT", "1")
    cfg = _config(tmp_path)

    log_event(
        "llm_request",
        "traj-1",
        cfg,
        content="raw prompt text",
        content_kind="prompt",
    )

    row = read_events(cfg)[0]
    assert row["content"] == "raw prompt text"


def test_redact_helper_is_default_off() -> None:
    """The redact() primitive returns the sentinel unless content logging is on."""
    env_off: dict[str, str] = {}
    assert redact("hello", "prompt", env_off) == "<REDACTED:prompt>"
    assert redact(None, "prompt", env_off) is None
    assert redact("hello", "prompt", {"LOOM_LOG_CONTENT": "1"}) == "hello"


def test_telemetry_attributes_carry_no_secrets_and_gate_cardinality() -> None:
    """Standard attrs carry the IP boundary, never secrets; version is off by default."""
    cfg = LoomConfig(owned_by="general", tenant="acme")
    # Session id on (default), version off (default).
    attrs = telemetry_attributes(cfg, run_id="exp-1", env={})
    assert attrs["owned_by"] == "general"
    assert attrs["tenant"] == "acme"
    assert attrs["run.id"] == "exp-1"
    assert "session.id" in attrs
    assert "app.version" not in attrs
    # No key-ish material anywhere.
    assert "key" not in json.dumps(attrs).lower()

    # Toggling the cardinality knobs flips the optional dimensions.
    attrs2 = telemetry_attributes(
        cfg,
        env={
            "LOOM_TELEMETRY_INCLUDE_SESSION_ID": "0",
            "LOOM_TELEMETRY_INCLUDE_VERSION": "1",
        },
    )
    assert "session.id" not in attrs2
    assert "app.version" in attrs2


# ---------------------------------------------------------------------------
# OTel bootstrap: degrades cleanly when the SDK is absent (no hard dep).
# ---------------------------------------------------------------------------


def test_bootstrap_otel_disabled_is_clean_noop() -> None:
    """With telemetry off / no exporter requested, bootstrap is a clean no-op."""
    boot = bootstrap_otel(env={})
    assert boot.enabled is False
    assert boot.available is False
    assert boot.exporters == []
    assert "disabled" in boot.message.lower()


def test_bootstrap_otel_requested_but_sdk_absent_degrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requested-but-absent SDK => available=False + an actionable pip message; never raises.

    The SDK is not installed in the venv, so requesting an exporter must NOT raise
    and must surface a pip hint while keeping the local JSONL capture working.
    """
    env = {
        "LOOM_TELEMETRY": "1",
        "OTEL_LOGS_EXPORTER": "otlp",
        "OTEL_METRICS_EXPORTER": "console",
    }
    boot = bootstrap_otel(env=env)
    assert boot.enabled is True
    assert boot.available is False  # SDK absent in the venv
    assert "pip install" in boot.message
    assert "opentelemetry" in boot.message


def test_import_loom_telemetry_without_opentelemetry() -> None:
    """`import loom.telemetry` must work with the OpenTelemetry SDK absent."""
    import importlib

    mod = importlib.import_module("loom.telemetry")
    assert hasattr(mod, "log_event")
    assert hasattr(mod, "assemble_trajectory")
    assert hasattr(mod, "build_distillation_dataset")


# ---------------------------------------------------------------------------
# CLI arg-parse: the `loom telemetry status|export|trace` wiring.
# ---------------------------------------------------------------------------


def test_cli_parses_telemetry_status() -> None:
    """`loom telemetry status` parses to the status handler."""
    from loom.cli import _build_parser, _cmd_telemetry_status

    args = _build_parser().parse_args(["telemetry", "status"])
    assert args.func is _cmd_telemetry_status


def test_cli_parses_telemetry_export_flags() -> None:
    """`loom telemetry export` parses --owned-by / --out / --with-content."""
    from loom.cli import _build_parser, _cmd_telemetry_export

    args = _build_parser().parse_args(
        ["telemetry", "export", "--owned-by", "general", "--out", "ds.jsonl",
         "--with-content"]
    )
    assert args.func is _cmd_telemetry_export
    assert args.owned_by == "general"
    assert args.out == "ds.jsonl"
    assert args.with_content is True


def test_cli_export_defaults_redacted_general() -> None:
    """The export defaults are the safe ones: general-only + redacted."""
    from loom.cli import _build_parser

    args = _build_parser().parse_args(["telemetry", "export"])
    assert args.owned_by == "general"
    assert args.with_content is False


def test_cli_parses_telemetry_trace_requires_id() -> None:
    """`loom telemetry trace` requires --trajectory and parses it through."""
    from loom.cli import _build_parser, _cmd_telemetry_trace

    parser = _build_parser()
    args = parser.parse_args(["telemetry", "trace", "--trajectory", "loom-abc"])
    assert args.func is _cmd_telemetry_trace
    assert args.trajectory == "loom-abc"

    with pytest.raises(SystemExit):
        parser.parse_args(["telemetry", "trace"])  # missing --trajectory
