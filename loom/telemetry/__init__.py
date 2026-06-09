"""Loom telemetry: COMPLETE training-data collection for LOOM-DS-1.

This package is **training-data collection, NOT observability**. Its purpose is
assembling a COMPLETE corpus for distilling LOOM-DS-1 -- every trajectory in full
-- so it is modeled on Claude Code's append-only **session transcript** (the
complete, unsampled record), NOT on a telemetry/analytics/metrics pipeline. It
does NOT re-log what Loom already captures -- the per-node corpus
(:mod:`loom.corpus`), the command-level rollouts (:mod:`loom.learnings`), and the
proxy LLM I/O (``loom.proxy.server``). Instead it ADDS the missing piece:
**trajectory correlation**. Every scattered signal is stamped with one stable
``trajectory_id`` so :func:`assemble_trajectory` can stitch a full agent
trajectory, and :func:`build_distillation_dataset` can turn those into the
LOOM-DS-1 SFT/teacher corpus.

Hard invariants:

* **COMPLETE, not sampled.** The append-only JSONL corpus
  (:func:`append_event` / :func:`append_trajectory`) captures EVERY event /
  trajectory in full -- NO sampling, NO batch-with-drop, NO TTL, NO aggregation.
* **OTel is the OPTIONAL OPS mirror ONLY.** The OpenTelemetry SDK is never
  imported at module load and never a hard dependency -- ``import loom.telemetry``
  works with the SDK absent. The only place it is touched is inside
  :func:`bootstrap_ops_telemetry` (back-compat alias :func:`bootstrap_otel`),
  lazily, gated by a SEPARATE ``LOOM_TELEMETRY_OTEL_OPS`` opt-in +
  ``OTEL_*_EXPORTER`` (capture via ``LOOM_TELEMETRY`` does NOT enable it). ⚠ It
  is ops-only and MUST NOT carry the training corpus (observability backends
  sample + aggregate + expire); it degrades to a clean no-op with an actionable
  message when the SDK is missing.
* **Prompt hygiene.** Content is REDACTED BY DEFAULT (``"<REDACTED:{kind}>"``)
  unless ``LOOM_LOG_CONTENT`` is set; only schema/preview/metric values enter
  telemetry, never raw rows; secrets are never logged.
* **IP boundary.** The distillation export trains ONLY on ``owned_by=="general"``,
  reusing the learnings/corpus discipline.

Public surface (what the wire/test layers import from here):

* :func:`telemetry_attributes` -- the standard attribute dict (cardinality
  toggles, no secrets).
* :class:`TelemetryEvent`, :func:`log_event`, :func:`read_events` -- the
  redacted, sequenced, trajectory-correlated event row + its JSONL sink.
* :class:`TrajectoryRecord`, :func:`start_trajectory`, :func:`end_trajectory`,
  :func:`assemble_trajectory` -- the interaction-root model + the JOIN.
* :class:`DistillExample`, :func:`build_distillation_dataset` -- the bridge to
  LOOM-DS-1 (general-only, redacted by default).
* :func:`append_event`, :func:`append_trajectory`, :func:`read_jsonl` -- the
  COMPLETE, append-only corpus sink.
* :func:`bootstrap_ops_telemetry` (alias :func:`bootstrap_otel`) -- the OPTIONAL
  ops-only OTel mirror (NOT the corpus).
"""

from __future__ import annotations

from loom.telemetry.attributes import session_id, telemetry_attributes
from loom.telemetry.distill import (
    GENERAL,
    DistillExample,
    build_distillation_dataset,
)
from loom.telemetry.events import (
    TelemetryEvent,
    content_logging_enabled,
    log_event,
    read_events,
    redact,
    telemetry_enabled,
)
from loom.telemetry.sink import (
    OtelBootstrap,
    append_event,
    append_trajectory,
    bootstrap_otel,
    bootstrap_ops_telemetry,
    read_jsonl,
)
from loom.telemetry.trajectory import (
    TrajectoryOutcome,
    TrajectoryRecord,
    TrajectoryStep,
    assemble_trajectory,
    end_trajectory,
    start_trajectory,
)

__all__ = [
    # attributes
    "telemetry_attributes",
    "session_id",
    # events
    "TelemetryEvent",
    "log_event",
    "read_events",
    "redact",
    "telemetry_enabled",
    "content_logging_enabled",
    # trajectory
    "TrajectoryRecord",
    "TrajectoryStep",
    "TrajectoryOutcome",
    "start_trajectory",
    "end_trajectory",
    "assemble_trajectory",
    # distill
    "DistillExample",
    "build_distillation_dataset",
    "GENERAL",
    # corpus sink + the optional ops-only OTel mirror
    "bootstrap_ops_telemetry",
    "bootstrap_otel",
    "OtelBootstrap",
    "append_event",
    "append_trajectory",
    "read_jsonl",
]
