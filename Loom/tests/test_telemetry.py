"""Telemetry core tests: events, attributes, the OTel no-op, and CLI arg-parse.

These exercise the always-available, dependency-light telemetry engine (no
OpenTelemetry SDK, no network):

* :func:`loom.telemetry.log_event` -- standard attributes are merged in, a
  MONOTONIC ``event.sequence`` orders events within a process, and content is
  REDACTED BY DEFAULT (un-redacted only when ``LOOM_LOG_CONTENT`` is set);
  ``log_event`` is a safe no-op (returns ``None``) when ``LOOM_TELEMETRY`` is off.
* :func:`loom.telemetry.bootstrap_ops_telemetry` (alias :func:`bootstrap_otel`)
  -- the OPTIONAL ops-only mirror, a SEPARATE opt-in gated by
  ``LOOM_TELEMETRY_OTEL_OPS`` (capture via ``LOOM_TELEMETRY`` does NOT enable it);
  it degrades cleanly to a no-op with an actionable message when the SDK is
  absent (which it is in the venv); it never raises and never makes the SDK a
  hard import.
* the COMPLETE corpus sink -- every logged event lands (no sampling/drop) with
  the ops mirror OFF, so capture has no dependence on opentelemetry.
* ``loom telemetry status|export|trace`` -- the CLI arg-parse wiring, including
  ``export --to-dataset`` (the versioned-Metaflow-data-object sink).

All file I/O is confined to ``tmp_path`` via a :class:`LoomConfig` whose
``telemetry_path`` points there, so the tests never touch the repo's corpus.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from loom.config import LoomConfig
from loom.telemetry import (
    bootstrap_ops_telemetry,
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
# Completeness: the corpus captures EVERY event with the ops mirror OFF.
# ---------------------------------------------------------------------------


def test_corpus_capture_is_complete_no_sampling_no_otel(
    tmp_path: Path, telemetry_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The corpus is COMPLETE: every logged event lands -- no sampling/drop, no OTel.

    With the ops mirror explicitly OFF (LOOM_TELEMETRY_OTEL_OPS unset) and the
    OpenTelemetry SDK absent from the venv, logging many events must persist EVERY
    one of them in order. This is the training-corpus completeness guarantee: the
    append-only JSONL never samples, never batch-drops, and never depends on otel.
    """
    monkeypatch.delenv("LOOM_TELEMETRY_OTEL_OPS", raising=False)
    cfg = _config(tmp_path)

    n = 500
    for i in range(n):
        log_event("llm_request", "traj-complete", cfg, attrs={"i": i})

    rows = read_events(cfg)
    # EVERY event landed (no sampling / no overflow drop).
    assert len(rows) == n
    # In order, none missing -- a faithful, lossless record.
    assert [r["i"] for r in rows] == list(range(n))


# ---------------------------------------------------------------------------
# Ops mirror: a SEPARATE opt-in; LOOM_TELEMETRY alone does NOT start it.
# ---------------------------------------------------------------------------


def test_bootstrap_ops_disabled_is_clean_noop() -> None:
    """With the ops mirror off / no exporter requested, bootstrap is a clean no-op."""
    boot = bootstrap_ops_telemetry(env={})
    assert boot.enabled is False
    assert boot.available is False
    assert boot.exporters == []
    assert "disabled" in boot.message.lower()


def test_capture_alone_does_not_start_the_ops_mirror() -> None:
    """LOOM_TELEMETRY (capture) alone must NOT enable the OTel ops mirror.

    The two planes are decoupled: enabling the complete corpus capture must not
    imply routing it to a sampling observability backend. The ops mirror requires
    its OWN explicit LOOM_TELEMETRY_OTEL_OPS opt-in in addition to an exporter.
    """
    # Capture on + an exporter set, but the ops opt-in is NOT present.
    env = {
        "LOOM_TELEMETRY": "1",
        "OTEL_LOGS_EXPORTER": "otlp",
        "OTEL_METRICS_EXPORTER": "console",
    }
    boot = bootstrap_ops_telemetry(env=env)
    assert boot.enabled is False  # capture alone did not start the ops mirror
    assert boot.exporters == []
    # The alias points at the same gate.
    assert bootstrap_otel(env=env).enabled is False


def test_bootstrap_ops_requested_but_sdk_absent_degrades() -> None:
    """The ops opt-in + an exporter, SDK absent => available=False + a pip hint; never raises.

    The SDK is not installed in the venv, so requesting the ops mirror must NOT
    raise and must surface a pip hint while keeping the complete JSONL corpus
    working. The ops mirror is its OWN opt-in (LOOM_TELEMETRY_OTEL_OPS).
    """
    env = {
        "LOOM_TELEMETRY_OTEL_OPS": "1",
        "OTEL_LOGS_EXPORTER": "otlp",
        "OTEL_METRICS_EXPORTER": "console",
    }
    boot = bootstrap_ops_telemetry(env=env)
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
    """The export defaults are the safe ones: general-only + redacted + no data object."""
    from loom.cli import _build_parser

    args = _build_parser().parse_args(["telemetry", "export"])
    assert args.owned_by == "general"
    assert args.with_content is False
    assert args.to_dataset is None  # --to-dataset is opt-in


def test_cli_parses_telemetry_export_to_dataset() -> None:
    """`loom telemetry export --to-dataset NAME` parses the data-object sink flag."""
    from loom.cli import _build_parser

    args = _build_parser().parse_args(
        ["telemetry", "export", "--to-dataset", "loom-ds-1"]
    )
    assert args.to_dataset == "loom-ds-1"


def test_export_default_out_still_works_without_data_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --to-dataset, export writes ONLY the --out JSONL (no ingest seam hit).

    The existing file export must keep working and must NOT touch the Metaflow
    ingest seam when --to-dataset is absent.
    """
    import loom.cli as cli

    out_path = tmp_path / "ds.jsonl"
    cfg = LoomConfig(
        telemetry_path=str(tmp_path / "telemetry" / "events.jsonl"),
        trajectories_path=str(tmp_path / "telemetry" / "trajectories.jsonl"),
        owned_by="general",
        tenant="default",
    )
    monkeypatch.setattr(cli, "_build_config", lambda args: cfg)
    monkeypatch.setattr(cli, "_telemetry_collect_trajectories", lambda config: [])

    # If the ingest seam is reached without --to-dataset, fail loudly.
    def _no_ingest(*a, **k):  # pragma: no cover - asserts it is NOT called
        raise AssertionError("the ingest seam must not be hit without --to-dataset")

    monkeypatch.setattr(cli, "_ingest_source", _no_ingest)

    args = cli._build_parser().parse_args(
        ["telemetry", "export", "--out", str(out_path)]
    )
    rc = cli._cmd_telemetry_export(args)

    assert rc == 0
    assert out_path.exists()  # the --out file export still produced


def test_export_to_dataset_ingests_versioned_data_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`export --to-dataset` routes the corpus through the IngestDataset seam -> a pathspec.

    Mocks the ingest/dataio seam (:func:`loom.cli._ingest_source`) to a data-object
    pathspec and asserts the corpus is staged + handed to it (NOT to any
    observability endpoint), and that the --out JSONL is still produced.
    """
    import loom.cli as cli
    from loom.telemetry.distill import DistillExample

    out_path = tmp_path / "ds.jsonl"
    cfg = LoomConfig(
        mlops_provider="metaflow",
        telemetry_path=str(tmp_path / "telemetry" / "events.jsonl"),
        trajectories_path=str(tmp_path / "telemetry" / "trajectories.jsonl"),
        owned_by="general",
        tenant="default",
    )
    monkeypatch.setattr(cli, "_build_config", lambda args: cfg)

    # A general-only example with nested fields, to prove the lossless staging.
    example = DistillExample(
        trajectory_id="traj-1",
        verb="aide",
        context=[{"role": "system", "content": "Loom verb: aide"}],
        teacher_output="<REDACTED:output>",
        tools_trajectory=[{"index": 0, "kind": "llm_call"}],
        reward=1.0,
        weight=1.0,
        owned_by="general",
    )

    # Stub trajectory assembly + the distill build to one example.
    monkeypatch.setattr(cli, "_telemetry_collect_trajectories", lambda config: [object()])
    import loom.telemetry as tele

    monkeypatch.setattr(tele, "build_distillation_dataset", lambda *a, **k: [example])

    captured: dict[str, object] = {}

    def _fake_ingest(source: str, name: str, config) -> tuple:
        captured["source"] = source
        captured["name"] = name
        # The staged CSV the seam would ingest must exist + contain the corpus.
        assert os.path.isfile(os.path.join(source, "train.csv"))
        return "IngestDataset/4242", None

    monkeypatch.setattr(cli, "_ingest_source", _fake_ingest)

    args = cli._build_parser().parse_args(
        ["telemetry", "export", "--out", str(out_path), "--to-dataset", "loom-ds-1"]
    )
    rc = cli._cmd_telemetry_export(args)

    assert rc == 0
    assert captured["name"] == "loom-ds-1"
    assert out_path.exists()  # the --out file export is still produced


def test_export_to_dataset_requires_metaflow_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--to-dataset` is guarded: a non-metaflow MLOps provider is refused cleanly."""
    import loom.cli as cli
    from loom.telemetry.distill import DistillExample

    cfg = LoomConfig(
        mlops_provider="local",
        telemetry_path=str(tmp_path / "telemetry" / "events.jsonl"),
        trajectories_path=str(tmp_path / "telemetry" / "trajectories.jsonl"),
        owned_by="general",
        tenant="default",
    )
    monkeypatch.setattr(cli, "_build_config", lambda args: cfg)
    monkeypatch.setattr(cli, "_telemetry_collect_trajectories", lambda config: [object()])
    import loom.telemetry as tele

    monkeypatch.setattr(
        tele,
        "build_distillation_dataset",
        lambda *a, **k: [DistillExample(trajectory_id="t", verb="aide", owned_by="general")],
    )

    args = cli._build_parser().parse_args(
        ["telemetry", "export", "--out", str(tmp_path / "ds.jsonl"),
         "--to-dataset", "loom-ds-1"]
    )
    rc = cli._cmd_telemetry_export(args)
    assert rc == 2  # guarded like the other lifecycle paths


def test_cli_parses_telemetry_trace_requires_id() -> None:
    """`loom telemetry trace` requires --trajectory and parses it through."""
    from loom.cli import _build_parser, _cmd_telemetry_trace

    parser = _build_parser()
    args = parser.parse_args(["telemetry", "trace", "--trajectory", "loom-abc"])
    assert args.func is _cmd_telemetry_trace
    assert args.trajectory == "loom-abc"

    with pytest.raises(SystemExit):
        parser.parse_args(["telemetry", "trace"])  # missing --trajectory
