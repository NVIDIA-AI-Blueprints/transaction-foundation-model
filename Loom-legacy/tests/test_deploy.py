"""Tests for the deploy verb: the cross-verb exit gate + CLI arg-parsing.

The deploy *gate* logic is factored out of :class:`flows.deploy.DeployFlow` into the
module-level pure functions :func:`flows.deploy.deploy_gate` and
:func:`flows.deploy.build_deploy_plan`, so the gate is unit-testable on a small
in-memory validate summary with **no Metaflow involved**. These tests pin the
load-bearing invariant of the whole verb:

* **the exit-gate self-test** -- a known-bad (sub-threshold / REVIEW / leaky)
  validate summary BLOCKS the deploy (the gate fails closed, never open);
* a clean ``VERDICT==PASS`` validate summary ALLOWS it;
* a missing validate report BLOCKS (you cannot deploy what was never validated);
* an optional holdout floor BLOCKS a too-low score;
* the plan/manifest status reflects the safety posture (BLOCKED / PLANNED / APPLIED)
  and the default ``apply=False`` never reaches APPLIED.

The deploy gate is pure Python (no pandas/sklearn/Metaflow), so these run
everywhere. The CLI arg-parse tests only exercise the argparse wiring for
``loom deploy``.
"""

from __future__ import annotations

import json

import pytest

from loom.cli import _build_parser


# ---------------------------------------------------------------------------
# The exit gate (pure; no Metaflow). The deploy.py gate is plain Python.
# ---------------------------------------------------------------------------


def _pass_summary(holdout: float = 0.82) -> dict:
    """A clean validate summary that should gate to ALLOW."""
    return {
        "verdict": "PASS",
        "leakage": False,
        "metric": "roc_auc",
        "target": "y",
        "dataset_ref": "IngestDataset/1",
        "holdout": {"score": holdout, "n": 60},
        "cv": {"mean": holdout - 0.02, "std": 0.01},
    }


def test_deploy_gate_blocks_subthreshold_validate() -> None:
    """A REVIEW (leaky) validate summary BLOCKS the deploy -- the exit-gate self-test.

    This is the executable self-test for the cross-verb composition gate: a
    sub-trustworthy validation must NOT silently let a deploy through. Feed a
    known-bad (leakage -> REVIEW) summary and assert the gate refuses (fails closed).
    """
    from flows.deploy import deploy_gate

    bad = _pass_summary()
    bad["verdict"] = "REVIEW"
    bad["leakage"] = True
    gate = deploy_gate(bad)
    assert gate["allow"] is False
    assert gate["decision"] == "BLOCK"
    assert gate["reasons"], "a blocked gate must explain why"


def test_deploy_gate_allows_clean_pass() -> None:
    """A clean VERDICT==PASS validate summary ALLOWS the deploy."""
    from flows.deploy import deploy_gate

    gate = deploy_gate(_pass_summary())
    assert gate["allow"] is True
    assert gate["decision"] == "ALLOW"
    assert gate["reasons"] == []


def test_deploy_gate_blocks_missing_report() -> None:
    """No validate report at all BLOCKS (cannot deploy what was never validated)."""
    from flows.deploy import deploy_gate

    gate = deploy_gate(None)
    assert gate["allow"] is False
    assert gate["decision"] == "BLOCK"
    assert gate["reasons"]


def test_deploy_gate_blocks_fail_verdict() -> None:
    """A FAIL verdict BLOCKS even without leakage."""
    from flows.deploy import deploy_gate

    bad = _pass_summary()
    bad["verdict"] = "FAIL"
    assert deploy_gate(bad)["allow"] is False


def test_deploy_gate_blocks_below_holdout_floor() -> None:
    """A PASS whose holdout is below an explicit floor BLOCKS."""
    from flows.deploy import deploy_gate

    gate = deploy_gate(_pass_summary(holdout=0.55), min_holdout=0.8)
    assert gate["allow"] is False
    assert any("floor" in r for r in gate["reasons"])
    # The same summary clears a lower floor.
    assert deploy_gate(_pass_summary(holdout=0.55), min_holdout=0.5)["allow"] is True


def test_deploy_gate_blocks_pass_without_holdout() -> None:
    """A PASS report carrying no sealed-holdout score BLOCKS."""
    from flows.deploy import deploy_gate

    bad = _pass_summary()
    bad["holdout"] = {"score": None, "n": 0}
    assert deploy_gate(bad)["allow"] is False


def test_deploy_gate_blocks_pass_with_leakage() -> None:
    """Defence in depth: leakage BLOCKS even if the verdict slipped to PASS."""
    from flows.deploy import deploy_gate

    bad = _pass_summary()
    bad["leakage"] = True  # verdict still PASS
    assert deploy_gate(bad)["allow"] is False


# ---------------------------------------------------------------------------
# The plan/manifest (pure): status reflects the safety posture.
# ---------------------------------------------------------------------------


def test_deploy_plan_blocked_when_gate_blocks() -> None:
    """A blocked gate yields a BLOCKED manifest regardless of --apply (never APPLIED)."""
    from flows.deploy import build_deploy_plan, deploy_gate

    gate = deploy_gate(None)  # blocked
    plan = build_deploy_plan("ValidateFlow/9", None, gate, apply=True, target="t")
    assert plan["status"] == "BLOCKED"
    assert plan["verdict"] == "BLOCKED"
    assert plan["manifest"]["status"] == "BLOCKED"


def test_deploy_plan_staged_by_default() -> None:
    """The default (apply=False) on an allowed gate is a STAGED plan, no mutation."""
    from flows.deploy import build_deploy_plan, deploy_gate

    gate = deploy_gate(_pass_summary())
    plan = build_deploy_plan(
        "ValidateFlow/9", _pass_summary(), gate, apply=False, target="staged-registry"
    )
    assert plan["status"] == "PLANNED"
    assert plan["verdict"] == "STAGED"
    assert plan["apply"] is False


def test_deploy_plan_applied_only_when_allowed_and_apply() -> None:
    """An APPLIED manifest requires BOTH gate allow AND apply=True."""
    from flows.deploy import build_deploy_plan, deploy_gate

    gate = deploy_gate(_pass_summary())
    plan = build_deploy_plan(
        "ValidateFlow/9", _pass_summary(), gate, apply=True, target="t"
    )
    assert plan["status"] == "APPLIED"
    assert plan["verdict"] == "DEPLOYED"
    # Lineage points back to exactly what was validated.
    assert plan["manifest"]["lineage"]["validate_run"] == "ValidateFlow/9"


def test_deploy_plan_is_json_able() -> None:
    """The whole plan round-trips through JSON (suitable for a RunResult summary)."""
    from flows.deploy import build_deploy_plan, deploy_gate

    gate = deploy_gate(_pass_summary())
    plan = build_deploy_plan("ValidateFlow/9", _pass_summary(), gate, apply=False)
    assert json.loads(json.dumps(plan)) == plan


# ---------------------------------------------------------------------------
# CLI arg-parsing (pure-Python; no Metaflow).
# ---------------------------------------------------------------------------


def test_cli_deploy_parses_validate_and_apply() -> None:
    """`loom deploy --validate RUN --apply` parses into the handler."""
    from loom.cli import _cmd_deploy

    parser = _build_parser()
    args = parser.parse_args(["deploy", "--validate", "ValidateFlow/12", "--apply"])
    assert args.command == "deploy"
    assert args.validate_run == "ValidateFlow/12"
    assert args.apply is True
    assert args.func is _cmd_deploy


def test_cli_deploy_apply_off_by_default() -> None:
    """--apply is OFF by default (the irreversible action is opt-in)."""
    parser = _build_parser()
    args = parser.parse_args(["deploy", "--solution", "EvalCandidate/3"])
    assert args.apply is False
    assert args.solution_run == "EvalCandidate/3"


def test_cli_deploy_requires_one_source() -> None:
    """`loom deploy` requires exactly one of --validate / --solution."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["deploy"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["deploy", "--validate", "ValidateFlow/1", "--solution", "EvalCandidate/2"]
        )
