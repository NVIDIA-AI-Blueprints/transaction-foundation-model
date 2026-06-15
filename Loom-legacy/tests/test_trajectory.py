"""Trajectory assembly tests: the JOIN that stitches a full trajectory.

:func:`loom.telemetry.assemble_trajectory` is the crux of the telemetry layer --
the interaction-root reconstruction (CC's interaction span + child operation
spans) re-stitched offline by ``trajectory_id``. It is **pure**: given
already-read lists of telemetry events, proxy LLM call rows, and the
command-level rollout, it returns one ordered :class:`TrajectoryRecord`. These
tests build those three inputs by hand and assert the JOIN:

* events filtered to the trajectory and ORDERED by ``event.sequence``;
* the proxy LLM call appended as an ``llm_call`` step (request + response);
* the rollout projected into the terminal outcome, with ``reward`` derived from
  the metric.
"""

from __future__ import annotations

from loom.telemetry import assemble_trajectory
from loom.telemetry.trajectory import _derive_reward


def _events(trajectory_id: str) -> list[dict]:
    """A start -> llm_request -> end event sequence for one trajectory.

    The sequence numbers are intentionally interleaved with a foreign trajectory's
    events and presented out of file order so the test proves the sort-by-sequence
    behavior, not incidental list order.
    """
    return [
        # A foreign trajectory's event that must be filtered out.
        {"event.name": "llm_request", "event.sequence": 1, "trajectory_id": "other"},
        {
            "event.name": "trajectory.end",
            "event.sequence": 5,
            "trajectory_id": trajectory_id,
            "trajectory.duration_ms": 1234.0,
        },
        {
            "event.name": "trajectory.start",
            "event.sequence": 2,
            "trajectory_id": trajectory_id,
            "verb": "aide",
            "owned_by": "general",
            "tenant": "default",
            "trajectory.started_ts": 1000.0,
            "task.goal": "<REDACTED:task.goal>",
            "task.metric": "<REDACTED:task.metric>",
            "task.experiment_id": trajectory_id,
        },
        {
            "event.name": "llm_request",
            "event.sequence": 3,
            "trajectory_id": trajectory_id,
            "model": "claude-x",
        },
    ]


def _proxy_calls(trajectory_id: str) -> list[dict]:
    """One matching proxy LLM call row + one foreign row that must be excluded."""
    return [
        {
            "model": "claude-x",
            "system": [{"type": "text", "text": "sys"}],
            "messages": [{"role": "user", "content": "<REDACTED>"}],
            "response_text": "the model answer",
            "usage": {"input_tokens": 10, "output_tokens": 3},
            "status": 200,
            "latency_ms": 42.0,
            "skill": "loom-aide",
            "owned_by": "general",
            "tenant": "default",
            "trajectory_id": trajectory_id,
        },
        {
            "model": "claude-x",
            "response_text": "foreign",
            "status": 200,
            "trajectory_id": "other",
        },
    ]


def _rollout(trajectory_id: str, *, metric: float | None, success: bool) -> dict:
    """A command-level rollout dict (the LearningRecord.asdict() shape)."""
    return {
        "command": "loom-aide",
        "task": {
            "data_ref": "IngestDataset/1",
            "goal": "win the metric",
            "metric": "auc",
            "experiment_id": trajectory_id,
        },
        "outcome": {"best_metric": metric, "submission_ok": success, "node_count": 4},
        "success": success,
        "owned_by": "general",
        "tenant": "default",
        "trajectory_id": trajectory_id,
    }


def test_assemble_joins_events_proxy_and_rollout_in_order() -> None:
    """assemble_trajectory JOINs the three signals into one ordered record."""
    tid = "loom-exp-1"
    traj = assemble_trajectory(
        tid,
        _events(tid),
        _proxy_calls(tid),
        _rollout(tid, metric=0.91, success=True),
    )

    # The start event yields verb / owned_by / started_ts; the end event the duration.
    assert traj.trajectory_id == tid
    assert traj.verb == "aide"
    assert traj.owned_by == "general"
    assert traj.tenant == "default"
    assert traj.started_ts == 1000.0
    assert traj.duration_ms == 1234.0

    # Only this trajectory's three events were correlated (the foreign one dropped).
    assert traj.event_count == 3

    # Steps: the non-lifecycle event (llm_request) THEN the proxy llm_call, in order.
    kinds = [s.kind for s in traj.steps]
    assert kinds == ["llm_request", "llm_call"]
    assert [s.index for s in traj.steps] == [0, 1]

    # The proxy call became a full request/response projection.
    llm_call = traj.steps[1]
    assert llm_call.llm_request["model"] == "claude-x"
    assert llm_call.llm_response["response_text"] == "the model answer"
    assert llm_call.llm_response["usage"] == {"input_tokens": 10, "output_tokens": 3}
    assert llm_call.attributes["status"] == 200
    assert llm_call.attributes["latency_ms"] == 42.0

    # The rollout projected into the terminal outcome; reward = the metric.
    assert traj.outcome.metric == 0.91
    assert traj.outcome.success is True
    assert traj.outcome.reward == 0.91


def test_assemble_orders_steps_by_event_sequence() -> None:
    """Events out of file order are sorted by event.sequence before stepping."""
    tid = "t"
    events = [
        {"event.name": "trajectory.start", "event.sequence": 10, "trajectory_id": tid,
         "verb": "aide"},
        {"event.name": "tool", "event.sequence": 30, "trajectory_id": tid,
         "tool": "second"},
        {"event.name": "llm_request", "event.sequence": 20, "trajectory_id": tid,
         "model": "m"},
    ]
    traj = assemble_trajectory(tid, events, [], None)
    # Step order follows the monotonic sequence (20 before 30), not list order.
    assert [s.kind for s in traj.steps] == ["llm_request", "tool"]


def test_assemble_falls_back_to_experiment_id_join() -> None:
    """With no telemetry events, the rollout alone still assembles by its id."""
    tid = "loom-exp-2"
    traj = assemble_trajectory(
        tid, [], [], _rollout(tid, metric=None, success=False)
    )
    assert traj.event_count == 0
    assert traj.verb == "loom-aide"  # read from the rollout's command
    assert traj.outcome.success is False
    assert traj.outcome.reward == 0.0  # no metric, not successful -> 0.0
    # The task context falls back from the rollout when no start event carried it.
    assert traj.task.get("goal") == "win the metric"


def test_derive_reward_prefers_metric_then_verdict_then_success() -> None:
    """Reward preference: metric (continuous) > verdict map > success boolean."""
    assert _derive_reward(0.73, "FAIL", False) == 0.73  # metric wins
    assert _derive_reward(None, "PASS", False) == 1.0  # verdict map
    assert _derive_reward(None, "FAIL", True) == 0.0  # verdict map
    assert _derive_reward(None, None, True) == 1.0  # success boolean
    assert _derive_reward(None, None, False) == 0.0
