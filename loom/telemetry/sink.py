"""Telemetry sinks: the JSONL store + the OPTIONAL, lazy OTel bootstrap.

Two sinks, both deliberately decoupled from the OpenTelemetry SDK so that
``import loom.telemetry`` works in any environment (the SDK is **never** a hard
dependency):

* :func:`append_event` / :func:`append_trajectory` -- the always-available local
  JSONL sink (dir-created, flushed, abspath-anchored), matching
  :meth:`loom.corpus.Corpus.record` / :func:`loom.proxy.server.log_call`. This is
  what actually persists the trajectory corpus the distillation export reads.

* :func:`bootstrap_otel` -- the OPTIONAL bridge to an external OTel collector,
  modeled on Claude Code's ``instrumentation.ts`` ``bootstrapTelemetry``: gated by
  ``LOOM_TELEMETRY`` + ``OTEL_*_EXPORTER``, it LAZILY imports the OpenTelemetry
  SDK (MeterProvider + LoggerProvider) only when actually invoked, supports the
  ``console`` and ``otlp`` exporters, and degrades to a clean no-op with an
  ACTIONABLE message when the SDK is not installed. It is never imported at module
  load and never required for the core capture path.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping


def _append_jsonl(path: str, row: Mapping[str, Any]) -> None:
    """Append one JSON object as a line to ``path`` (dir-created, flushed).

    The shared write primitive for both telemetry sinks, identical in discipline
    to :meth:`loom.corpus.Corpus.record`: the parent dir is created lazily and the
    line is flushed so a crash mid-run still leaves a valid prefix of whole lines.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


def append_event(path: str, row: Mapping[str, Any]) -> None:
    """Append one telemetry event row to the events JSONL at ``path``."""
    _append_jsonl(path, row)


def append_trajectory(path: str, row: Mapping[str, Any]) -> None:
    """Append one assembled-trajectory row to the trajectories JSONL at ``path``."""
    _append_jsonl(path, row)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read every JSON object from a JSONL file, tolerating blank lines.

    A missing file yields an empty list. Used by the readers in
    :mod:`loom.telemetry.events` and the status/trace CLI.

    Args:
        path: The JSONL file path.

    Returns:
        The rows as plain dicts, in file order.
    """
    if not os.path.isfile(path):
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@dataclass
class OtelBootstrap:
    """The outcome of an :func:`bootstrap_otel` attempt (a status report).

    Always returned (never raised), so a caller can log/inspect the outcome and
    keep running whether or not the SDK was present. The local JSONL sink is
    unaffected by this entirely.

    Attributes:
        enabled: Whether telemetry export was requested (``LOOM_TELEMETRY`` +
            an ``OTEL_*_EXPORTER`` set).
        available: Whether the OpenTelemetry SDK was importable.
        exporters: The exporter protocols that were wired (e.g.
            ``["console"]`` / ``["otlp"]``), empty when none.
        message: A human-readable, actionable status line.
        meter_provider: The constructed OTel ``MeterProvider`` (or ``None``).
        logger_provider: The constructed OTel ``LoggerProvider`` (or ``None``).
    """

    enabled: bool
    available: bool
    exporters: list[str]
    message: str
    meter_provider: Any = None
    logger_provider: Any = None


def _parse_exporters(value: str | None) -> list[str]:
    """Parse an ``OTEL_*_EXPORTER`` env value into a clean protocol list.

    Mirrors CC's ``parseExporterTypes``: comma-split, trimmed, ``"none"`` dropped
    (per the OTel spec ``none`` means "no auto-configured exporter").
    """
    return [
        t.strip()
        for t in (value or "").split(",")
        if t.strip() and t.strip() != "none"
    ]


def _is_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def bootstrap_otel(
    config: LoomConfig | None = None,
    env: Mapping[str, str] | None = None,
) -> OtelBootstrap:
    """Optionally bring up an OpenTelemetry export bridge -- LAZY + no hard dep.

    The Loom analogue of CC's ``bootstrapTelemetry`` + ``initializeTelemetry``,
    pared to the engine essentials:

    1. **Gate.** Does nothing unless ``LOOM_TELEMETRY`` is truthy AND at least one
       of ``OTEL_METRICS_EXPORTER`` / ``OTEL_LOGS_EXPORTER`` names an exporter.
    2. **Lazy import.** Only then does it attempt ``import opentelemetry`` (+ the
       SDK). The import lives inside this function, never at module load, so
       ``import loom.telemetry`` works with the SDK absent.
    3. **No-op on absence.** If the SDK is not installed it returns a report with
       ``available=False`` and an ACTIONABLE message (how to install / that the
       local JSONL capture is unaffected) -- it never raises.
    4. **Exporters.** Supports ``console`` and ``otlp`` (the OTLP exporter package
       is itself lazily imported per protocol, as CC does).

    Args:
        config: The active configuration (unused beyond future resource attrs;
            accepted for symmetry / forward-compat).
        env: Environment mapping (defaults to ``os.environ``).

    Returns:
        An :class:`OtelBootstrap` status report -- always; never raises.
    """
    e: Mapping[str, str] = env if env is not None else os.environ

    metrics_exporters = _parse_exporters(e.get("OTEL_METRICS_EXPORTER"))
    logs_exporters = _parse_exporters(e.get("OTEL_LOGS_EXPORTER"))
    requested = sorted(set(metrics_exporters) | set(logs_exporters))

    if not _is_truthy(e.get("LOOM_TELEMETRY")) or not requested:
        return OtelBootstrap(
            enabled=False,
            available=False,
            exporters=[],
            message=(
                "OTel export disabled (set LOOM_TELEMETRY=1 and an "
                "OTEL_METRICS_EXPORTER / OTEL_LOGS_EXPORTER to enable). Local "
                "JSONL telemetry capture is unaffected."
            ),
        )

    # (2) Lazy import -- the ONLY place the optional SDK is touched.
    try:
        from opentelemetry.sdk._logs import LoggerProvider  # type: ignore
        from opentelemetry.sdk.metrics import MeterProvider  # type: ignore
        from opentelemetry.sdk.resources import Resource  # type: ignore
    except Exception:  # noqa: BLE001 - SDK absent => clean no-op, never raise
        return OtelBootstrap(
            enabled=True,
            available=False,
            exporters=[],
            message=(
                "OTel export was requested (LOOM_TELEMETRY + OTEL_*_EXPORTER) but "
                "the OpenTelemetry SDK is not installed. Install it to enable the "
                "external exporter:\n"
                "  pip install opentelemetry-sdk opentelemetry-exporter-otlp\n"
                "Loom's local JSONL telemetry capture continues unaffected."
            ),
        )

    resource = Resource.create({"service.name": "loom"})
    wired: list[str] = []
    meter_provider: Any = None
    logger_provider: Any = None

    # (4) Metrics readers -- console + otlp (each exporter package imported lazily).
    if metrics_exporters:
        try:
            from opentelemetry.sdk.metrics.export import (  # type: ignore
                PeriodicExportingMetricReader,
            )

            readers = []
            for proto in metrics_exporters:
                if proto == "console":
                    from opentelemetry.sdk.metrics.export import (  # type: ignore
                        ConsoleMetricExporter,
                    )

                    readers.append(
                        PeriodicExportingMetricReader(ConsoleMetricExporter())
                    )
                    wired.append("metrics:console")
                elif proto == "otlp":
                    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (  # type: ignore
                        OTLPMetricExporter,
                    )

                    readers.append(
                        PeriodicExportingMetricReader(OTLPMetricExporter())
                    )
                    wired.append("metrics:otlp")
            if readers:
                meter_provider = MeterProvider(resource=resource, metric_readers=readers)
        except Exception:  # noqa: BLE001 - a missing exporter extra degrades cleanly
            pass

    # Logs (events) -- console + otlp.
    if logs_exporters:
        try:
            from opentelemetry.sdk._logs.export import (  # type: ignore
                BatchLogRecordProcessor,
            )

            logger_provider = LoggerProvider(resource=resource)
            for proto in logs_exporters:
                if proto == "console":
                    from opentelemetry.sdk._logs.export import (  # type: ignore
                        ConsoleLogExporter,
                    )

                    logger_provider.add_log_record_processor(
                        BatchLogRecordProcessor(ConsoleLogExporter())
                    )
                    wired.append("logs:console")
                elif proto == "otlp":
                    from opentelemetry.exporter.otlp.proto.http._log_exporter import (  # type: ignore
                        OTLPLogExporter,
                    )

                    logger_provider.add_log_record_processor(
                        BatchLogRecordProcessor(OTLPLogExporter())
                    )
                    wired.append("logs:otlp")
        except Exception:  # noqa: BLE001 - degrade cleanly
            logger_provider = None

    return OtelBootstrap(
        enabled=True,
        available=True,
        exporters=wired,
        message=(
            f"OTel export active (exporters: {', '.join(wired) or 'none wired'}). "
            "Loom's local JSONL telemetry capture also continues."
        ),
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )


# Late import to avoid a cycle at module top (config imports nothing from here).
from loom.config import LoomConfig  # noqa: E402


__all__ = [
    "append_event",
    "append_trajectory",
    "read_jsonl",
    "bootstrap_otel",
    "OtelBootstrap",
]
