"""Distillation tests: the IP boundary + redaction + the SFT example shape.

:func:`loom.telemetry.build_distillation_dataset` is the bridge from assembled
trajectories to the LOOM-DS-1 SFT/teacher corpus. Two hard invariants are the
centerpiece of these tests:

* **IP BOUNDARY (the self-test).** Training is allowed ONLY on
  ``owned_by == "general"`` trajectories; a tenant-owned trajectory is EXCLUDED
  from the dataset -- the discipline that keeps tenant-confidential traces out of
  the cross-tenant model.
* **PROMPT HYGIENE.** Content (goal/metric/outputs) is REDACTED BY DEFAULT to the
  typed sentinels; raw text appears only with ``with_content=True``.

Plus the example *shape*: context messages, the teacher output, the tool
trajectory, and the reward/weight.
"""

from __future__ import annotations

from loom.telemetry import build_distillation_dataset
from loom.telemetry.trajectory import (
    TrajectoryOutcome,
    TrajectoryRecord,
    TrajectoryStep,
)


def _trajectory(
    trajectory_id: str,
    *,
    owned_by: str,
    metric: float | None = 0.9,
    success: bool = True,
) -> TrajectoryRecord:
    """A small assembled trajectory with one llm_call step and a real outcome."""
    step = TrajectoryStep(
        index=0,
        kind="llm_call",
        llm_request={"model": "claude-x", "messages": [{"role": "user"}]},
        llm_response={"response_text": "the teacher answer", "status": 200},
        attributes={"model": "claude-x", "status": 200},
    )
    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        verb="aide",
        task={"goal": "win the metric", "metric": "auc"},
        steps=[step],
        outcome=TrajectoryOutcome(
            metric=metric,
            success=success,
            reward=metric if metric is not None else (1.0 if success else 0.0),
        ),
        owned_by=owned_by,
        tenant="default" if owned_by == "general" else owned_by,
    )


def test_ip_boundary_excludes_tenant_owned_trajectory() -> None:
    """The IP-boundary self-test: a tenant-owned trajectory is EXCLUDED."""
    general = _trajectory("g1", owned_by="general")
    tenant = _trajectory("t1", owned_by="tala")  # tenant-owned -> must be dropped

    examples = build_distillation_dataset([general, tenant])

    # Only the general trajectory survives the IP boundary.
    assert len(examples) == 1
    assert examples[0].trajectory_id == "g1"
    assert examples[0].owned_by == "general"
    # The tenant id never appears anywhere in the emitted dataset.
    ids = {ex.trajectory_id for ex in examples}
    assert "t1" not in ids


def test_owned_by_filter_is_explicit_and_overridable() -> None:
    """The filter is explicit: changing it lets a specific owner through (only it)."""
    general = _trajectory("g1", owned_by="general")
    tenant = _trajectory("t1", owned_by="tala")

    examples = build_distillation_dataset([general, tenant], owned_by_filter="tala")
    assert [ex.trajectory_id for ex in examples] == ["t1"]


def test_content_redacted_by_default() -> None:
    """Content is REDACTED BY DEFAULT: goal/metric/teacher_output are sentinels."""
    examples = build_distillation_dataset([_trajectory("g1", owned_by="general")])
    ex = examples[0]

    # The context messages carry the verb (low-cardinality) but redact the task text.
    system_msg, user_msg = ex.context
    assert system_msg["role"] == "system"
    assert "aide" in system_msg["content"]
    assert "<REDACTED:task.goal>" in user_msg["content"]
    assert "<REDACTED:task.metric>" in user_msg["content"]
    # The raw goal text never appears.
    assert "win the metric" not in user_msg["content"]

    # The teacher output (the last LLM response) is redacted too.
    assert ex.teacher_output == "<REDACTED:output>"
    # And the tool trajectory's response text.
    tool0 = ex.tools_trajectory[0]
    assert tool0["response"]["text"] == "<REDACTED:output>"
    assert tool0["response"]["status"] == 200


def test_with_content_opt_in_unredacts() -> None:
    """With with_content=True the raw text passes through (the explicit opt-in)."""
    examples = build_distillation_dataset(
        [_trajectory("g1", owned_by="general")], with_content=True
    )
    ex = examples[0]
    user_msg = ex.context[1]
    assert "win the metric" in user_msg["content"]
    assert ex.teacher_output == "the teacher answer"


def test_sft_example_shape_and_reward_weight() -> None:
    """The example shape: context/teacher_output/tools_trajectory/reward/weight."""
    examples = build_distillation_dataset([_trajectory("g1", owned_by="general")])
    ex = examples[0]

    assert ex.verb == "aide"
    assert isinstance(ex.context, list) and len(ex.context) == 2
    assert isinstance(ex.tools_trajectory, list) and len(ex.tools_trajectory) == 1
    assert ex.tools_trajectory[0]["kind"] == "llm_call"
    # Reward comes from the outcome metric; weight is the reward clamped to [0, 1].
    assert ex.reward == 0.9
    assert ex.weight == 0.9
    assert ex.metric == 0.9
    assert ex.success is True


def test_weight_clamps_out_of_range_reward() -> None:
    """The training weight clamps to [0, 1] for both boolean and continuous rewards."""
    high = _trajectory("g1", owned_by="general", metric=2.5)  # reward 2.5 -> weight 1.0
    fail = _trajectory("g2", owned_by="general", metric=None, success=False)  # 0.0

    examples = build_distillation_dataset([high, fail])
    by_id = {ex.trajectory_id: ex for ex in examples}
    assert by_id["g1"].weight == 1.0
    assert by_id["g2"].weight == 0.0
