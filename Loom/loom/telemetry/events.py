"""Telemetry events: the redacted, sequenced, trajectory-correlated event row.

Modeled on Claude Code's ``events.ts`` (``logOTelEvent`` + ``redactIfDisabled``):
every event row carries the standard attributes (:func:`telemetry_attributes`)
plus ``event.name``, an ISO ``event.ts``, and a **monotonic** ``event.sequence``
for ordering within a process -- and, crucially for Loom, a ``trajectory_id`` so
the distillation layer can stitch a full agent trajectory from the scattered
signals.

PROMPT HYGIENE (a hard invariant): bulk data NEVER enters telemetry. Any
``content`` (a prompt, a model output, a tool observation) is **REDACTED BY
DEFAULT** to ``"<REDACTED:{kind}>"`` -- the CC ``redactIfDisabled`` discipline --
and only passes through verbatim when ``LOOM_LOG_CONTENT`` is explicitly set. The
attribute dict carries only schema/preview/metric values, never raw rows; secrets
are never logged (the inputs carry none).

Telemetry is OFF unless ``LOOM_TELEMETRY`` is set (read at the point of use):
:func:`log_event` is a safe no-op when telemetry is disabled, so callers can emit
unconditionally without guarding every call site.

Pure-ish + testable: the only side effect is appending one JSONL row to
``config.telemetry_path`` (via :mod:`loom.telemetry.sink`); no network, no OTel.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from loom.config import LoomConfig
from loom.telemetry.attributes import telemetry_attributes
from loom.telemetry.sink import append_event

# Monotonically increasing counter for ordering events within a process, mirroring
# CC's module-level ``eventSequence``. ``itertools.count`` is atomic enough for the
# single-threaded controller path and keeps the counter out of any global object.
_SEQUENCE = itertools.count()


def _is_truthy(value: str | None) -> bool:
    """Return whether an env string spells a truthy value."""
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def telemetry_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether telemetry capture is enabled (``LOOM_TELEMETRY`` truthy).

    Read at the point of use (not baked onto the config) so a process can toggle
    capture without rebuilding config, matching CC's ``isTelemetryEnabled``.
    """
    e: Mapping[str, str] = env if env is not None else os.environ
    return _is_truthy(e.get("LOOM_TELEMETRY"))


def content_logging_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether raw content may be logged (``LOOM_LOG_CONTENT`` truthy).

    The CC ``isUserPromptLoggingEnabled`` analogue. **False by default**, so
    prompts/outputs are redacted unless an operator opts in explicitly.
    """
    e: Mapping[str, str] = env if env is not None else os.environ
    return _is_truthy(e.get("LOOM_LOG_CONTENT"))


def redact(
    content: str | None,
    content_kind: str | None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Redact ``content`` unless content logging is enabled.

    Mirrors CC's ``redactIfDisabled``: returns the content verbatim only when
    ``LOOM_LOG_CONTENT`` is set, otherwise the typed sentinel
    ``"<REDACTED:{kind}>"`` (so the schema/shape is preserved without the bytes).
    A ``None`` content stays ``None``.

    Args:
        content: The raw content (prompt/output/observation), or ``None``.
        content_kind: A short kind tag (e.g. ``"prompt"``, ``"output"``) baked
            into the redaction sentinel.
        env: Environment mapping (defaults to ``os.environ``).

    Returns:
        The content, the typed redaction sentinel, or ``None``.
    """
    if content is None:
        return None
    if content_logging_enabled(env):
        return content
    return f"<REDACTED:{content_kind or 'content'}>"


@dataclass
class TelemetryEvent:
    """One telemetry event row -- the JSONL shape :func:`log_event` appends.

    A flat, JSON-serializable record: the standard attributes are merged in at the
    top level (so a downstream OTel exporter sees them as span/log attributes),
    alongside the event name, ISO timestamp, monotonic sequence, trajectory id,
    any extra attrs, and the (redacted-by-default) content.

    Attributes:
        name: The event name (e.g. ``"trajectory.start"``, ``"llm_request"``).
        ts: ISO-8601 UTC timestamp of when the event was emitted.
        sequence: Monotonic per-process ordering counter (CC ``event.sequence``).
        trajectory_id: The trajectory this event belongs to -- the join key the
            distillation layer stitches a full trajectory on.
        attributes: The standard telemetry attributes + any caller ``attrs``
            (schema/preview/metric values only; never raw rows or secrets).
        content: The redacted-by-default content sentinel (or the raw content when
            ``LOOM_LOG_CONTENT`` is set), or ``None`` when the event carried none.
        content_kind: The kind tag for ``content`` (e.g. ``"prompt"``).
    """

    name: str
    ts: str
    sequence: int
    trajectory_id: str
    attributes: dict[str, Any] = field(default_factory=dict)
    content: str | None = None
    content_kind: str | None = None

    def to_row(self) -> dict[str, Any]:
        """Flatten to the JSONL row shape (standard attrs hoisted to top level)."""
        row: dict[str, Any] = {
            "event.name": self.name,
            "event.ts": self.ts,
            "event.sequence": self.sequence,
            "trajectory_id": self.trajectory_id,
        }
        # Hoist standard + caller attributes to the top level (CC attaches them as
        # event attributes), but never let them clobber the reserved keys above.
        for key, value in self.attributes.items():
            if key not in row:
                row[key] = value
        if self.content is not None:
            row["content"] = self.content
            row["content.kind"] = self.content_kind
        return row


def log_event(
    name: str,
    trajectory_id: str,
    config: LoomConfig,
    *,
    attrs: Mapping[str, Any] | None = None,
    content: str | None = None,
    content_kind: str | None = None,
    run_id: str | None = None,
    env: Mapping[str, str] | None = None,
) -> TelemetryEvent | None:
    """Build and append one telemetry event row; a safe no-op when disabled.

    Assembles a :class:`TelemetryEvent` from the standard attributes
    (:func:`telemetry_attributes`) + ``event.name`` + an ISO ``event.ts`` + the
    next monotonic ``event.sequence`` + ``trajectory_id`` + any caller ``attrs``,
    redacts ``content`` by default (CC ``redactIfDisabled``), and appends it as
    one JSONL row to ``config.telemetry_path``.

    When ``LOOM_TELEMETRY`` is not set the call is a no-op returning ``None`` --
    so callers (the controller, the proxy) can emit unconditionally without
    guarding each site. The sequence counter still advances only when an event is
    actually emitted, keeping the on-disk sequence dense.

    Args:
        name: The event name.
        trajectory_id: The trajectory join key.
        config: The active Loom configuration (telemetry path + std attributes).
        attrs: Extra low-cardinality attributes (schema/preview/metric only).
        content: Optional content payload (redacted by default).
        content_kind: Kind tag for ``content`` (baked into the redaction sentinel).
        run_id: Optional run/experiment id stamped as ``run.id``.
        env: Environment mapping (defaults to ``os.environ``).

    Returns:
        The appended :class:`TelemetryEvent`, or ``None`` when telemetry is off.
    """
    if not telemetry_enabled(env):
        return None

    attributes = telemetry_attributes(config, run_id=run_id, env=env)
    if attrs:
        # Caller attrs win over standard ones only for non-reserved keys; this is
        # the place to pass schema/preview/metric dimensions, never raw rows.
        attributes.update(dict(attrs))

    event = TelemetryEvent(
        name=name,
        ts=datetime.now(timezone.utc).isoformat(),
        sequence=next(_SEQUENCE),
        trajectory_id=trajectory_id,
        attributes=attributes,
        content=redact(content, content_kind, env),
        content_kind=content_kind if content is not None else None,
    )

    append_event(config.telemetry_path, event.to_row())
    return event


def read_events(config: LoomConfig) -> list[dict[str, Any]]:
    """Read every telemetry event row from ``config.telemetry_path``, in order.

    A thin reader used by the status/trace CLI and the assembly layer. Missing
    file => empty list; blank lines are skipped.

    Args:
        config: The active configuration (only ``telemetry_path`` is read).

    Returns:
        The event rows as plain dicts, in file order.
    """
    from loom.telemetry.sink import read_jsonl

    return read_jsonl(config.telemetry_path)


__all__ = [
    "TelemetryEvent",
    "log_event",
    "read_events",
    "redact",
    "telemetry_enabled",
    "content_logging_enabled",
]
