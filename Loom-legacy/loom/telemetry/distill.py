"""Distillation: assembled trajectories -> the LOOM-DS-1 SFT/teacher format.

The bridge from the telemetry trajectory corpus to LOOM-DS-1, Loom's own
data-science model. :func:`build_distillation_dataset` turns assembled
:class:`~loom.telemetry.trajectory.TrajectoryRecord` objects into the
supervised-fine-tuning / teacher-distillation examples a training job consumes:
a ``messages``/``context`` input, the ``teacher_output`` the agent produced, the
``tools_trajectory`` it executed, and a scalar ``reward`` derived from the
outcome (metric/verdict/success), with a ``weight`` for reward-weighted SFT.

Two hard invariants are enforced here, reusing the exact discipline of
:mod:`loom.learnings` / :mod:`loom.corpus`:

* **IP BOUNDARY.** Training is allowed ONLY on ``owned_by == "general"``
  trajectories. A tenant-owned trajectory is dropped (the self-test that keeps
  tenant-confidential trajectories out of the cross-tenant LOOM-DS-1 set).
* **PROMPT HYGIENE.** Content is REDACTED BY DEFAULT: the input/output text is
  emitted as the typed redaction sentinel unless ``with_content=True`` is passed
  explicitly. Even with content, only what the upstream signals carried is used --
  never re-fetched raw rows or secrets.

Pure + testable: takes assembled trajectories, returns a list of dataclasses; no
I/O, no OTel, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from loom.telemetry.trajectory import TrajectoryRecord, TrajectoryStep

# The IP-boundary sentinel, identical to :data:`loom.learnings.GENERAL` /
# :data:`loom.corpus.GENERAL` so all four stores share one vocabulary.
GENERAL = "general"

# Redaction sentinel for a step/example field whose content was withheld. Mirrors
# the ``"<REDACTED:{kind}>"`` shape :mod:`loom.telemetry.events` writes at emit
# time, applied here at distill time when ``with_content`` is off.
def _redacted(kind: str) -> str:
    return f"<REDACTED:{kind}>"


@dataclass
class DistillExample:
    """One LOOM-DS-1 SFT/teacher-distillation example.

    The training-ready projection of one assembled trajectory: an input context
    (messages), the teacher output, the tool trajectory, and the reward/weight
    that turn it into a reward-weighted SFT target.

    Attributes:
        trajectory_id: The source trajectory's id (provenance / dedup key).
        verb: The Loom verb the trajectory exercised.
        context: The input context as chat ``messages`` (system + user turns).
            Content is redacted unless ``with_content`` was set.
        teacher_output: The teacher (agent) output to imitate -- the assistant
            turn. Redacted unless ``with_content`` was set.
        tools_trajectory: The ordered tool/exec/LLM-call steps the agent ran, as
            small JSON-able dicts (model/status/metric, plus content when allowed).
        reward: The scalar reward from the outcome (metric/verdict/success).
        weight: A non-negative training weight derived from the reward (clamped to
            ``[0, 1]``) for reward-weighted SFT.
        owned_by: Always ``"general"`` for an emitted example (the IP boundary);
            carried for auditability.
        metric: The raw outcome metric, for reference.
        success: Whether the trajectory succeeded.
    """

    trajectory_id: str
    verb: str
    context: list[dict[str, Any]] = field(default_factory=list)
    teacher_output: str | None = None
    tools_trajectory: list[dict[str, Any]] = field(default_factory=list)
    reward: float | None = None
    weight: float = 0.0
    owned_by: str = GENERAL
    metric: float | None = None
    success: bool = False


def _clamp_weight(reward: float | None) -> float:
    """Map a reward to a non-negative training weight in ``[0, 1]``.

    A plain clamp: ``None`` -> 0.0; a reward below 0 -> 0.0; above 1 -> 1.0;
    otherwise the reward itself. Keeps reward-weighted SFT well-behaved for both
    boolean (0/1) and continuous (a 0..1 metric) reward signals.
    """
    if reward is None:
        return 0.0
    if reward < 0.0:
        return 0.0
    if reward > 1.0:
        return 1.0
    return float(reward)


def _context_messages(
    trajectory: TrajectoryRecord, with_content: bool
) -> list[dict[str, Any]]:
    """Build the input ``messages`` from the trajectory's task context.

    A system turn naming the verb + a user turn carrying the task goal/metric. The
    goal/metric are treated as content: redacted to a sentinel unless
    ``with_content`` is set (they can embed task-specific detail).
    """
    goal = trajectory.task.get("goal")
    metric = trajectory.task.get("metric")
    if not with_content:
        goal = _redacted("task.goal") if goal is not None else None
        metric = _redacted("task.metric") if metric is not None else None

    user_parts: list[str] = []
    if goal is not None:
        user_parts.append(f"Goal: {goal}")
    if metric is not None:
        user_parts.append(f"Metric: {metric}")

    return [
        {"role": "system", "content": f"Loom verb: {trajectory.verb}"},
        {"role": "user", "content": "\n".join(user_parts) or _redacted("task")},
    ]


def _step_to_tool(step: TrajectoryStep, with_content: bool) -> dict[str, Any]:
    """Project one trajectory step into a JSON-able tools_trajectory entry."""
    entry: dict[str, Any] = {"index": step.index, "kind": step.kind}
    # Carry the small low-cardinality attributes (model/status/metric) verbatim.
    if step.attributes:
        entry["attributes"] = dict(step.attributes)

    if step.llm_response is not None:
        text = step.llm_response.get("response_text")
        entry["response"] = {
            "text": (text if with_content else _redacted("output"))
            if text is not None
            else None,
            "status": step.llm_response.get("status"),
            "usage": step.llm_response.get("usage"),
        }
    if step.observation is not None:
        entry["observation"] = step.observation if with_content else _redacted("observation")
    return entry


def _teacher_output(trajectory: TrajectoryRecord, with_content: bool) -> str | None:
    """Pick the teacher output to imitate: the last LLM response text.

    Walks the steps backward for the last ``llm_response`` carrying text. Redacted
    to a sentinel unless ``with_content`` is set.
    """
    for step in reversed(trajectory.steps):
        if step.llm_response is not None:
            text = step.llm_response.get("response_text")
            if text is not None:
                return text if with_content else _redacted("output")
    return None


def build_distillation_dataset(
    trajectories: Iterable[TrajectoryRecord],
    *,
    owned_by_filter: str = GENERAL,
    with_content: bool = False,
) -> list[DistillExample]:
    """Build the LOOM-DS-1 dataset from assembled trajectories -- general-only.

    The bridge telemetry -> LOOM-DS-1. Each kept trajectory becomes one
    :class:`DistillExample` (context -> teacher_output + tools_trajectory +
    reward/weight). Two invariants are enforced:

    * **IP boundary.** Only trajectories whose ``owned_by == owned_by_filter``
      (default ``"general"``) are included; a tenant-owned trajectory is dropped.
      This is the self-test that keeps tenant data out of the cross-tenant set.
    * **Redaction.** Content (goal/metric/outputs/observations) is redacted to the
      typed sentinel unless ``with_content=True`` is passed explicitly.

    Args:
        trajectories: Assembled trajectories to distill.
        owned_by_filter: The IP-owner tag a trajectory must carry to be included
            (default ``"general"`` -- the only safe default for the moat model).
        with_content: When ``True``, emit raw content instead of the redaction
            sentinel (off by default; the operator must opt in).

    Returns:
        The list of :class:`DistillExample` (general-only, redacted by default).
    """
    examples: list[DistillExample] = []
    for traj in trajectories:
        # IP BOUNDARY: never train on a tenant-owned trajectory.
        if traj.owned_by != owned_by_filter:
            continue

        examples.append(
            DistillExample(
                trajectory_id=traj.trajectory_id,
                verb=traj.verb,
                context=_context_messages(traj, with_content),
                teacher_output=_teacher_output(traj, with_content),
                tools_trajectory=[
                    _step_to_tool(s, with_content) for s in traj.steps
                ],
                reward=traj.outcome.reward,
                weight=_clamp_weight(traj.outcome.reward),
                owned_by=traj.owned_by,
                metric=traj.outcome.metric,
                success=traj.outcome.success,
            )
        )
    return examples


__all__ = ["DistillExample", "build_distillation_dataset", "GENERAL"]
