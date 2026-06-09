"""The CANONICAL training-data sink (append-only JSONL) + an OPTIONAL ops mirror.

THE CORPUS IS A TRANSCRIPT, NOT OBSERVABILITY
=============================================
This module's append-only JSONL functions (:func:`append_event` /
:func:`append_trajectory`) are the **canonical training-data sink** for
distilling LOOM-DS-1. Its purpose is collecting a **COMPLETE** dataset, not
operational observability. The completeness guarantees are load-bearing:

* **Append-only.** Every call appends one whole, flushed line; nothing is ever
  rewritten or compacted. A crash mid-run leaves a valid prefix of whole lines.
* **EVERY event / trajectory, in full.** There is **NO sampling**, **NO
  batch-with-drop / overflow queue**, **NO retention TTL / expiry**, and **NO
  aggregation / rollup**. Each logged signal lands on disk verbatim. This is what
  makes the store a faithful training corpus rather than a lossy metrics feed.

This is modeled on the way Claude Code keeps the COMPLETE record: the append-only
**session transcript JSONL** (a write stream opened in append mode -- every
message, never sampled), NOT on its telemetry/analytics plane. Even a
*first-party* analytics pipeline samples (a per-event sample rate, a random draw
below it keeps the event and the rest are dropped) and batches with a bounded
queue (overflow drops); OTel metrics additionally aggregate. None of those
lossy behaviors are acceptable for a corpus we distill a model from, so the
corpus sink deliberately copies the transcript discipline and nothing else.

THE TWO PLANES ARE DECOUPLED
============================
* :func:`append_event` / :func:`append_trajectory` -- the always-available,
  dependency-light corpus sink (dir-created, flushed, abspath-anchored), matching
  :meth:`loom.corpus.Corpus.record` / :func:`loom.proxy.server.log_call`. This is
  what actually persists the COMPLETE trajectory corpus the distillation export
  reads. Enabled by ``LOOM_TELEMETRY`` (the capture signal).

* :func:`bootstrap_ops_telemetry` (back-compat alias :func:`bootstrap_otel`) --
  an **OPTIONAL OPS-MONITORING mirror ONLY**, NOT the corpus. It is a SEPARATE,
  explicit opt-in: it requires ``LOOM_TELEMETRY_OTEL_OPS`` **in addition to** an
  ``OTEL_*_EXPORTER``, so enabling capture (``LOOM_TELEMETRY``) never implies the
  ops mirror. ⚠ Observability/metrics backends **SAMPLE, AGGREGATE, and EXPIRE**
  data; they MUST NOT carry the training corpus. This bridge exists only to feed
  ops dashboards, lazily imports the OpenTelemetry SDK (MeterProvider +
  LoggerProvider) when invoked, supports the ``console`` and ``otlp`` exporters,
  and degrades to a clean no-op with an ACTIONABLE message when the SDK is
  absent. It is never imported at module load and never required for capture.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping


def _append_jsonl(path: str, row: Mapping[str, Any]) -> None:
    """Append one JSON object as a line to ``path`` (dir-created, flushed).

    The shared write primitive for the canonical corpus sink, identical in
    discipline to :meth:`loom.corpus.Corpus.record` and to a transcript write
    stream opened in append mode: the parent dir is created lazily and the line is
    flushed so a crash mid-run still leaves a valid prefix of whole lines.

    COMPLETENESS: this is a pure append of the WHOLE row -- no sampling, no
    bounded queue / overflow drop, no TTL, no aggregation. Every call lands one
    line on disk, so the corpus is a lossless record of every signal.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


def append_event(path: str, row: Mapping[str, Any]) -> None:
    """Append one event row to the COMPLETE, append-only corpus at ``path``.

    The canonical training-data sink: every event is captured in full, never
    sampled or dropped (see the module docstring's completeness guarantees).
    """
    _append_jsonl(path, row)


def append_trajectory(path: str, row: Mapping[str, Any]) -> None:
    """Append one assembled-trajectory row to the COMPLETE corpus at ``path``.

    The canonical training-data sink: every trajectory is captured in full, never
    sampled or dropped (see the module docstring's completeness guarantees).
    """
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
    """The outcome of a :func:`bootstrap_ops_telemetry` attempt (a status report).

    Always returned (never raised), so a caller can log/inspect the outcome and
    keep running whether or not the SDK was present. The OPS mirror this reports
    on is NOT the corpus: the complete append-only JSONL corpus is unaffected by
    this entirely and never flows through any wired exporter.

    Attributes:
        enabled: Whether the OPS mirror was requested -- i.e. the explicit
            ``LOOM_TELEMETRY_OTEL_OPS`` opt-in AND an ``OTEL_*_EXPORTER`` set.
            (``LOOM_TELEMETRY`` capture alone does NOT enable it.)
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


def bootstrap_ops_telemetry(
    config: LoomConfig | None = None,
    env: Mapping[str, str] | None = None,
) -> OtelBootstrap:
    """Optionally bring up the OPS-MONITORING OTel mirror -- LAZY + no hard dep.

    ⚠ This is **NOT the training corpus.** Observability/metrics backends SAMPLE,
    AGGREGATE, and EXPIRE data, so they must never carry the complete corpus we
    distill LOOM-DS-1 from. This bridge exists ONLY to feed ops dashboards; the
    canonical corpus is the append-only JSONL written by :func:`append_event` /
    :func:`append_trajectory`, which is unaffected by anything here.

    The Loom analogue of CC's ops ``instrumentation.ts`` bootstrap, pared to the
    engine essentials:

    1. **Separate, explicit gate.** Does nothing unless the ops mirror is
       explicitly opted into via ``LOOM_TELEMETRY_OTEL_OPS`` AND at least one of
       ``OTEL_METRICS_EXPORTER`` / ``OTEL_LOGS_EXPORTER`` names an exporter. The
       capture signal ``LOOM_TELEMETRY`` does **NOT** enable this -- corpus
       capture never implies the ops mirror, and the two planes stay decoupled.
    2. **Lazy import.** Only then does it attempt ``import opentelemetry`` (+ the
       SDK). The import lives inside this function, never at module load, so
       ``import loom.telemetry`` works with the SDK absent.
    3. **No-op on absence.** If the SDK is not installed it returns a report with
       ``available=False`` and an ACTIONABLE message (how to install / that the
       complete JSONL corpus is unaffected) -- it never raises.
    4. **Exporters.** Supports ``console`` and ``otlp`` (the OTLP exporter package
       is itself lazily imported per protocol).

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

    # SEPARATE OPS GATE: the ops mirror is its own explicit opt-in
    # (LOOM_TELEMETRY_OTEL_OPS), distinct from the LOOM_TELEMETRY capture signal,
    # so enabling the corpus never implies routing it to a sampling backend.
    if not _is_truthy(e.get("LOOM_TELEMETRY_OTEL_OPS")) or not requested:
        return OtelBootstrap(
            enabled=False,
            available=False,
            exporters=[],
            message=(
                "OTel ops mirror disabled (set LOOM_TELEMETRY_OTEL_OPS=1 and an "
                "OTEL_METRICS_EXPORTER / OTEL_LOGS_EXPORTER to enable the "
                "ops-dashboards mirror). This is ops-only and NOT the training "
                "corpus; the complete append-only JSONL corpus is unaffected."
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
                "The OTel ops mirror was requested (LOOM_TELEMETRY_OTEL_OPS + "
                "OTEL_*_EXPORTER) but the OpenTelemetry SDK is not installed. "
                "Install it to enable the ops-dashboards exporter:\n"
                "  pip install opentelemetry-sdk opentelemetry-exporter-otlp\n"
                "This mirror is ops-only and NOT the corpus; Loom's complete "
                "append-only JSONL corpus continues unaffected."
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
            f"OTel ops mirror active (exporters: {', '.join(wired) or 'none wired'}). "
            "This is the ops-only mirror, NOT the corpus -- it samples/aggregates; "
            "Loom's complete append-only JSONL corpus is captured separately."
        ),
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )


# Back-compat alias. The function was historically named ``bootstrap_otel``; it
# is now ``bootstrap_ops_telemetry`` to make clear it is the OPS mirror, not the
# corpus. The old name is kept so existing callers/imports keep working.
bootstrap_otel = bootstrap_ops_telemetry


# Late import to avoid a cycle at module top (config imports nothing from here).
from loom.config import LoomConfig  # noqa: E402


__all__ = [
    "append_event",
    "append_trajectory",
    "read_jsonl",
    "bootstrap_ops_telemetry",
    "bootstrap_otel",
    "OtelBootstrap",
]
