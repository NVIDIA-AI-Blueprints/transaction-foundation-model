"""Trajectories: the interaction-root model + the JOIN that stitches a full one.

This is the crux of the telemetry layer, modeled on Claude Code's
``sessionTracing.ts``: every user request -> agent response cycle is a single
ROOT **trajectory** (CC's "interaction" span), under which the correlated
operation steps (LLM requests, tool/exec, observations) are grouped. CC stitches
them via the active OTel context; Loom stitches them by a stable
``trajectory_id`` written onto every scattered signal -- the telemetry events
(:mod:`loom.telemetry.events`), the proxy LLM call rows
(``loom.proxy.server.proxy_calls.jsonl``), and the command-level rollout
(``loom.learnings``). :func:`assemble_trajectory` is the pure JOIN that
re-materializes one ordered trajectory from those three sources.

Two thin lifecycle helpers mirror CC's ``startInteractionSpan`` /
``endInteractionSpan``:

* :func:`start_trajectory` -- emit a ``trajectory.start`` event and return the
  ``trajectory_id`` (defaulting to a generated id, or pinned to an
  ``experiment_id`` so the rollout JOINs cleanly);
* :func:`end_trajectory` -- emit a ``trajectory.end`` event with the outcome + a
  ``trajectory.duration_ms`` (CC's ``interaction.duration_ms``).

:func:`assemble_trajectory` itself is **pure** -- it takes already-read lists and
returns a :class:`TrajectoryRecord`, so it is trivially testable without any I/O.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from loom.config import LoomConfig
from loom.telemetry.events import log_event

# Monotonic per-process interaction counter, mirroring CC's ``interactionSequence``
# (ordering of trajectories within a process). Independent of the per-event
# sequence in events.py.
import itertools as _itertools

_INTERACTION_SEQUENCE = _itertools.count(1)


@dataclass
class TrajectoryStep:
    """One ordered step inside an assembled trajectory.

    A step pairs an LLM request/response with the tool/exec it drove and the
    observation that came back -- the unit a distillation example is built from.
    Content fields are whatever the source signals carried (already redacted at
    emit time unless ``LOOM_LOG_CONTENT`` was set), never re-fetched raw rows.

    Attributes:
        index: 0-based position of this step within the trajectory.
        kind: The step kind (e.g. ``"llm_request"``, ``"tool"``, ``"exec"``).
        llm_request: The proxy call row's request projection (model, messages
            shape) or ``None``.
        llm_response: The proxy call row's response projection (text/usage/status)
            or ``None``.
        observation: The tool/exec observation (e.g. a node's metric/term-out
            preview) or ``None``.
        attributes: Small low-cardinality attributes for the step (model, latency,
            status, metric) -- never raw rows.
    """

    index: int
    kind: str
    llm_request: dict[str, Any] | None = None
    llm_response: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryOutcome:
    """The terminal outcome of a trajectory (the reward signal for distillation).

    Projected from the command-level rollout (:class:`loom.learnings.LearningRecord`)
    so it carries the same metric/verdict the SkillOpt flywheel uses.

    Attributes:
        metric: The best validation metric, or ``None``.
        verdict: A short verdict string (e.g. ``"PASS"``/``"FAIL"``) when known.
        success: Whether the trajectory produced a usable solution.
        reward: A scalar reward derived for distillation (see
            :func:`_derive_reward`): the metric when present, else 1.0/0.0 on
            success.
    """

    metric: float | None = None
    verdict: str | None = None
    success: bool = False
    reward: float | None = None


@dataclass
class TrajectoryRecord:
    """A fully assembled trajectory -- the interaction-root JOIN, ready to distill.

    The ordered training example CC's tracing would render as one interaction span
    with child operation spans, here re-stitched from Loom's three signals by
    ``trajectory_id``: an input context -> an ordered list of steps -> a terminal
    outcome, all tagged with the IP-boundary ``owned_by`` so the distillation
    export can filter.

    Attributes:
        trajectory_id: The join key (the experiment id when pinned).
        verb: The Loom verb/command this trajectory exercised (e.g. ``"aide"``).
        task: A compact task/context projection (goal/metric/data_ref) -- the
            distillation input context.
        steps: The ordered operation steps (see :class:`TrajectoryStep`).
        outcome: The terminal outcome / reward (see :class:`TrajectoryOutcome`).
        owned_by: The IP owner; ``"general"`` => usable by the cross-tenant
            LOOM-DS-1 distillation set, any other value tags it tenant-owned.
        tenant: The multi-tenant tag.
        started_ts: Epoch seconds the trajectory started (from the start event).
        duration_ms: Wall-clock duration in ms (from the end event), or ``None``.
        event_count: How many telemetry events were correlated.
    """

    trajectory_id: str
    verb: str
    task: dict[str, Any] = field(default_factory=dict)
    steps: list[TrajectoryStep] = field(default_factory=list)
    outcome: TrajectoryOutcome = field(default_factory=TrajectoryOutcome)
    owned_by: str = "general"
    tenant: str = "default"
    started_ts: float | None = None
    duration_ms: float | None = None
    event_count: int = 0


def start_trajectory(
    verb: str,
    task: Mapping[str, Any] | None,
    config: LoomConfig,
    *,
    trajectory_id: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Open a trajectory: emit ``trajectory.start`` and return its id.

    The CC ``startInteractionSpan`` analogue. The ``trajectory_id`` defaults to a
    generated id, but a caller (the controller) should pin it to the run's
    ``experiment_id`` so the rollout JOINs cleanly in :func:`assemble_trajectory`.
    The start event carries the interaction sequence + the (redacted-by-default)
    task context as low-cardinality attributes; emitting is a no-op when telemetry
    is off, but the id is still returned so the caller can stamp it everywhere.

    Args:
        verb: The Loom verb/command (e.g. the search provider name).
        task: A compact task context (goal/metric/data_ref); preview only.
        config: The active configuration.
        trajectory_id: Explicit id to use (else a generated ``loom-traj-<uuid>``).
        env: Environment mapping (defaults to ``os.environ``).

    Returns:
        The trajectory id to stamp onto every correlated signal.
    """
    tid = trajectory_id or f"loom-traj-{uuid.uuid4().hex}"
    task_attrs = _task_attrs(task)
    log_event(
        "trajectory.start",
        tid,
        config,
        attrs={
            "verb": verb,
            "interaction.sequence": next(_INTERACTION_SEQUENCE),
            "trajectory.started_ts": time.time(),
            **task_attrs,
        },
        run_id=tid,
        env=env,
    )
    return tid


def end_trajectory(
    trajectory_id: str,
    outcome: Mapping[str, Any] | None,
    config: LoomConfig,
    *,
    started_ts: float | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Close a trajectory: emit ``trajectory.end`` with the outcome + duration.

    The CC ``endInteractionSpan`` analogue (which sets ``interaction.duration_ms``).
    The outcome (metric/verdict/success) is recorded as low-cardinality attributes;
    a no-op when telemetry is off.

    Args:
        trajectory_id: The id returned by :func:`start_trajectory`.
        outcome: A compact outcome mapping (``metric``/``verdict``/``success``).
        config: The active configuration.
        started_ts: The start epoch seconds (to compute ``trajectory.duration_ms``).
        env: Environment mapping (defaults to ``os.environ``).
    """
    attrs: dict[str, Any] = {}
    if outcome:
        for key in ("metric", "verdict", "success"):
            if key in outcome and outcome[key] is not None:
                attrs[f"outcome.{key}"] = outcome[key]
    if started_ts is not None:
        attrs["trajectory.duration_ms"] = round((time.time() - started_ts) * 1000.0, 3)
    log_event(
        "trajectory.end",
        trajectory_id,
        config,
        attrs=attrs,
        run_id=trajectory_id,
        env=env,
    )


def _task_attrs(task: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project a task mapping into small preview attributes (never raw rows)."""
    if not task:
        return {}
    out: dict[str, Any] = {}
    for key in ("goal", "metric", "data_ref", "experiment_id"):
        if task.get(key) is not None:
            out[f"task.{key}"] = task[key]
    return out


def _derive_reward(
    metric: float | None, verdict: str | None, success: bool
) -> float | None:
    """Derive a scalar distillation reward from metric/verdict/success.

    Preference order, so a continuous signal beats a boolean: the validation
    ``metric`` when present; else a verdict mapped to {PASS:1.0, FAIL:0.0}; else
    ``1.0``/``0.0`` from ``success``.
    """
    if metric is not None:
        return float(metric)
    if verdict is not None:
        v = verdict.strip().upper()
        if v in {"PASS", "OK", "ALLOWED", "PROMOTED"}:
            return 1.0
        if v in {"FAIL", "BLOCKED", "REJECTED"}:
            return 0.0
    return 1.0 if success else 0.0


def assemble_trajectory(
    trajectory_id: str,
    events: Sequence[Mapping[str, Any]],
    proxy_calls: Sequence[Mapping[str, Any]],
    rollout: Mapping[str, Any] | None,
    *,
    verb: str | None = None,
) -> TrajectoryRecord:
    """JOIN the three signals into one ordered :class:`TrajectoryRecord` -- PURE.

    The interaction-root reconstruction (CC's interaction span + child operation
    spans, re-stitched offline). All three inputs are already-read lists, so this
    function does NO I/O and is trivially testable:

    1. **Events.** Filter ``events`` to those whose ``trajectory_id`` matches; read
       the ``trajectory.start`` for the verb/task/started_ts and the
       ``trajectory.end`` for the duration; turn every other ``llm_request`` /
       ``tool`` / ``exec`` event into an ordered step (sorted by ``event.sequence``
       so the on-disk monotonic order is the trajectory order).
    2. **Proxy calls.** Attach each proxy LLM call row tagged with this
       ``trajectory_id`` as a step's request/response projection (model, messages
       shape, response text/usage/status) -- the LLM I/O for the step.
    3. **Rollout.** Project the matching command-level rollout (joined on
       ``trajectory_id`` else its ``task.experiment_id``) into the terminal
       :class:`TrajectoryOutcome`, deriving the scalar ``reward``.

    Args:
        trajectory_id: The trajectory to assemble.
        events: Telemetry event rows (any trajectory; filtered here).
        proxy_calls: Proxy LLM call rows (any trajectory; filtered here).
        rollout: The matching command-level rollout dict, or ``None``.
        verb: Optional verb override (else read from the start event / rollout).

    Returns:
        The assembled :class:`TrajectoryRecord` (steps ordered, outcome derived).
    """
    mine = [e for e in events if e.get("trajectory_id") == trajectory_id]
    mine.sort(key=lambda e: e.get("event.sequence", 0))

    start = next((e for e in mine if e.get("event.name") == "trajectory.start"), None)
    end = next((e for e in mine if e.get("event.name") == "trajectory.end"), None)

    resolved_verb = verb
    task: dict[str, Any] = {}
    owned_by = "general"
    tenant = "default"
    started_ts: float | None = None

    if start is not None:
        resolved_verb = resolved_verb or start.get("verb")
        owned_by = start.get("owned_by", owned_by)
        tenant = start.get("tenant", tenant)
        started_ts = start.get("trajectory.started_ts")
        for key in ("goal", "metric", "data_ref", "experiment_id"):
            if f"task.{key}" in start:
                task[key] = start[f"task.{key}"]

    duration_ms: float | None = None
    if end is not None:
        duration_ms = end.get("trajectory.duration_ms")

    # (1) Ordered operation steps from the non-lifecycle events.
    steps: list[TrajectoryStep] = []
    for ev in mine:
        name = ev.get("event.name")
        if name in (None, "trajectory.start", "trajectory.end"):
            continue
        step_attrs = {
            k: v
            for k, v in ev.items()
            if k in ("model", "latency_ms", "status", "metric", "tool", "stage")
        }
        steps.append(
            TrajectoryStep(
                index=len(steps),
                kind=str(name),
                llm_request=(
                    {"content": ev.get("content"), "content.kind": ev.get("content.kind")}
                    if ev.get("content") is not None
                    else None
                ),
                attributes=step_attrs,
            )
        )

    # (2) Attach the proxy LLM calls tagged with this trajectory, in file order.
    for call in proxy_calls:
        if call.get("trajectory_id") != trajectory_id:
            continue
        steps.append(
            TrajectoryStep(
                index=len(steps),
                kind="llm_call",
                llm_request={
                    "model": call.get("model"),
                    "system": call.get("system"),
                    "messages": call.get("messages"),
                },
                llm_response={
                    "response_text": call.get("response_text"),
                    "usage": call.get("usage"),
                    "status": call.get("status"),
                },
                attributes={
                    k: v
                    for k, v in (
                        ("model", call.get("model")),
                        ("latency_ms", call.get("latency_ms")),
                        ("status", call.get("status")),
                        ("skill", call.get("skill")),
                    )
                    if v is not None
                },
            )
        )
        if call.get("owned_by") is not None:
            owned_by = call["owned_by"]
        if call.get("tenant") is not None:
            tenant = call["tenant"]

    # (3) Project the rollout into the terminal outcome + reward.
    outcome = TrajectoryOutcome()
    if rollout is not None:
        roll_outcome = rollout.get("outcome") or {}
        metric = roll_outcome.get("best_metric")
        success = bool(rollout.get("success", roll_outcome.get("submission_ok", False)))
        verdict = rollout.get("verdict") or (rollout.get("inputs") or {}).get("verdict")
        outcome = TrajectoryOutcome(
            metric=metric,
            verdict=verdict,
            success=success,
            reward=_derive_reward(metric, verdict, success),
        )
        resolved_verb = resolved_verb or rollout.get("command")
        if rollout.get("owned_by") is not None:
            owned_by = rollout["owned_by"]
        if rollout.get("tenant") is not None:
            tenant = rollout["tenant"]
        if not task:
            roll_task = rollout.get("task") or {}
            for src, dst in (
                ("goal", "goal"),
                ("metric", "metric"),
                ("data_ref", "data_ref"),
                ("experiment_id", "experiment_id"),
            ):
                if roll_task.get(src) is not None:
                    task[dst] = roll_task[src]

    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        verb=resolved_verb or "unknown",
        task=task,
        steps=steps,
        outcome=outcome,
        owned_by=owned_by,
        tenant=tenant,
        started_ts=started_ts,
        duration_ms=duration_ms,
        event_count=len(mine),
    )


__all__ = [
    "TrajectoryRecord",
    "TrajectoryStep",
    "TrajectoryOutcome",
    "start_trajectory",
    "end_trajectory",
    "assemble_trajectory",
]
