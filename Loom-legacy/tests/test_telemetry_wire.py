"""Telemetry wire tests: the LIGHT, ADDITIVE integration into the live code path.

The telemetry-core tests (``test_telemetry`` / ``test_trajectory`` /
``test_distill``) exercise the pure engine. These pin the two integration seams
that stamp the ``trajectory_id`` onto Loom's existing capture so the pure JOIN
later has something to correlate -- the surgical, backward-compatible edits the
wire brief calls out:

* **The controller rollup.** :func:`loom.controller._learning_from` pins the
  rollout's ``trajectory_id`` to the run's ``experiment_id`` (and degrades to
  ``None`` when none is supplied), so the command-level rollout JOINs the
  trajectory's events + proxy calls in :func:`assemble_trajectory`.
* **The proxy side-channel.** :func:`loom.proxy.server._emit_llm_request_event`
  emits a small, REDACTED ``llm_request`` telemetry event tagged with the
  trajectory id -- correlating the proxied LLM call WITHOUT re-logging the bulk
  request/response (those stay in ``proxy_calls``). It is a no-op when telemetry
  is off and reflects the call's IP-boundary tags.

Both seams are additive: the rollout field defaults ``None`` and the emission is
gated by ``LOOM_TELEMETRY`` (off => nothing written), so neither perturbs the
existing capture or the suite when telemetry is disabled.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.config import LoomConfig
from loom.telemetry import read_events


# ---------------------------------------------------------------------------
# The controller rollup stamps the trajectory id (pinned to experiment_id).
# ---------------------------------------------------------------------------


def test_controller_rollup_stamps_trajectory_id() -> None:
    """``_learning_from`` pins the rollout's trajectory_id to the experiment id.

    Exercised purely (no flow / no providers): the rollout row carries the JOIN
    key so the distillation layer can stitch the rollout to its events + proxy
    calls, and the key equals the experiment id the controller pinned the
    trajectory to.
    """
    from loom.controller import _learning_from
    from loom.types import SearchResult, Task

    task = Task(
        data_dir="", goal="g", eval="auc", experiment_id="loom-exp-9", tenant="acme"
    )
    result = SearchResult(best_code="print(1)", best_metric=0.8, node_count=3)

    rec = _learning_from(
        result, task, LoomConfig(owned_by="general", tenant="acme"), "loom-exp-9"
    )
    assert rec.trajectory_id == "loom-exp-9"
    assert rec.task.experiment_id == "loom-exp-9"
    assert rec.success is True


def test_controller_rollup_trajectory_id_is_additive_default_none() -> None:
    """No trajectory id supplied => the field defaults None (backward-compatible)."""
    from loom.controller import _learning_from
    from loom.types import SearchResult, Task

    task = Task(data_dir="", goal="g", eval="auc", experiment_id="exp", tenant="default")
    result = SearchResult(best_code=None, best_metric=None, node_count=0)

    rec = _learning_from(result, task, LoomConfig())
    assert rec.trajectory_id is None
    # The pre-telemetry rollout shape is otherwise unchanged (still a clean record).
    assert rec.success is False


# ---------------------------------------------------------------------------
# The proxy side-channel: a redacted, trajectory-tagged llm_request event.
# ---------------------------------------------------------------------------


def test_proxy_emit_llm_request_is_noop_when_telemetry_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With LOOM_TELEMETRY off, the proxy emits nothing (the gated no-op)."""
    from loom.proxy.server import _emit_llm_request_event

    monkeypatch.delenv("LOOM_TELEMETRY", raising=False)
    telemetry_file = tmp_path / "telemetry" / "events.jsonl"
    monkeypatch.setenv("LOOM_TELEMETRY_PATH", str(telemetry_file))

    _emit_llm_request_event("loom-exp-9", "claude-x", "loom-run", "acme", "general")

    assert not telemetry_file.exists()


def test_proxy_emit_llm_request_event_is_tagged_and_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With telemetry on, the proxy emits one trajectory-tagged llm_request event.

    The event correlates the proxied call (model + skill + IP-boundary tags) and
    carries NO bulk content -- the request/response bytes live only in the
    proxy_calls row. The owned_by/tenant reflect the call's tags so the
    distillation export filters this event on the same boundary.
    """
    from loom.proxy.server import _emit_llm_request_event

    monkeypatch.setenv("LOOM_TELEMETRY", "1")
    monkeypatch.delenv("LOOM_LOG_CONTENT", raising=False)
    telemetry_file = tmp_path / "telemetry" / "events.jsonl"
    monkeypatch.setenv("LOOM_TELEMETRY_PATH", str(telemetry_file))

    _emit_llm_request_event("loom-exp-9", "claude-x", "loom-run", "acme", "general")

    cfg = LoomConfig.load()
    rows = read_events(cfg)
    assert len(rows) == 1
    row = rows[0]
    assert row["event.name"] == "llm_request"
    assert row["trajectory_id"] == "loom-exp-9"
    assert row["model"] == "claude-x"
    assert row["skill"] == "loom-run"
    # The IP-boundary tags reflect the call (the export filters on owned_by).
    assert row["owned_by"] == "general"
    assert row["tenant"] == "acme"
    # No bulk content rides this side-channel event.
    assert "content" not in row


def test_proxy_emit_llm_request_reflects_tenant_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tenant-owned call emits an event tagged tenant-owned (excluded downstream)."""
    from loom.proxy.server import _emit_llm_request_event

    monkeypatch.setenv("LOOM_TELEMETRY", "1")
    telemetry_file = tmp_path / "telemetry" / "events.jsonl"
    monkeypatch.setenv("LOOM_TELEMETRY_PATH", str(telemetry_file))

    _emit_llm_request_event("t-1", "claude-x", None, "acme", "acme-corp")

    rows = read_events(LoomConfig.load())
    assert rows[0]["owned_by"] == "acme-corp"
    assert rows[0]["tenant"] == "acme"


# ---------------------------------------------------------------------------
# The `loom telemetry` CLI handlers, driven end-to-end against a seeded corpus.
# ---------------------------------------------------------------------------


def _seed_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner: str = "general") -> LoomConfig:
    """Seed an isolated telemetry corpus (events + proxy call + rollout) under tmp.

    Points every capture path at ``tmp_path`` via the env (so the CLI's
    ``_build_config`` -> ``LoomConfig.load`` reads them), emits a full
    start/llm_request/end trajectory pinned to ``exp-1``, writes a matching
    proxy_calls row, and records the command-level rollout. Returns the config.
    """
    import json

    from loom.learnings import Learnings, LearningRecord, Outcome, TaskSpec
    from loom.telemetry import end_trajectory, log_event, start_trajectory

    base = tmp_path / "telemetry"
    base.mkdir(parents=True, exist_ok=True)
    events = base / "events.jsonl"
    trajs = base / "trajectories.jsonl"
    proxy = base / "proxy_calls.jsonl"
    learn = base / "rollouts.jsonl"

    monkeypatch.setenv("LOOM_TELEMETRY", "1")
    monkeypatch.delenv("LOOM_LOG_CONTENT", raising=False)
    monkeypatch.setenv("LOOM_TELEMETRY_PATH", str(events))
    monkeypatch.setenv("LOOM_TRAJECTORIES_PATH", str(trajs))
    monkeypatch.setenv("LOOM_PROXY_LOG_PATH", str(proxy))
    monkeypatch.setenv("LOOM_LEARNINGS_PATH", str(learn))
    monkeypatch.setenv("LOOM_OWNED_BY", owner)
    monkeypatch.setenv("LOOM_TENANT", "acme")

    cfg = LoomConfig.load()
    tid = start_trajectory(
        "aide", {"goal": "g", "metric": "auc", "experiment_id": "exp-1"}, cfg, trajectory_id="exp-1"
    )
    log_event("llm_request", tid, cfg, attrs={"model": "m"}, content="secret", content_kind="prompt")
    end_trajectory(tid, {"metric": 0.8, "success": True}, cfg, started_ts=0.0)

    proxy.write_text(
        json.dumps({"trajectory_id": "exp-1", "model": "m", "response_text": "ANS", "status": 200}) + "\n",
        encoding="utf-8",
    )
    Learnings(cfg).record(
        LearningRecord(
            command="loom-aide",
            task=TaskSpec(data_ref=None, goal="g", metric="auc", experiment_id="exp-1"),
            inputs={},
            outcome=Outcome(best_metric=0.8, submission_ok=True),
            success=True,
            owned_by=owner,
            tenant="acme",
            trajectory_id="exp-1",
        )
    )
    return cfg


def _ns(**kw: object):
    """A minimal argparse-like namespace carrying the config-build flags + extras."""
    import argparse

    base = dict(
        config=None, mlops=None, search=None, steps=None,
        model_provider=None, code_provider=None, feedback_provider=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_cli_status_reports_counts_and_ip_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`loom telemetry status` summarizes the corpus (counts, IP split, OTel off)."""
    from loom.cli import _cmd_telemetry_status

    _seed_corpus(tmp_path, monkeypatch)
    rc = _cmd_telemetry_status(_ns())
    out = capsys.readouterr().out

    assert rc == 0
    assert "events           : 3" in out
    assert "trajectories     : 1" in out
    assert "1 general" in out  # the IP-boundary split
    assert "capture enabled  : on" in out
    # The OTel SDK is absent in the venv -> reported as such, never crashing.
    assert "sdk-absent" in out


def test_cli_export_writes_general_only_redacted_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`loom telemetry export` writes a general-only, redacted LOOM-DS-1 JSONL."""
    import json

    from loom.cli import _cmd_telemetry_export

    _seed_corpus(tmp_path, monkeypatch)
    out_path = tmp_path / "ds.jsonl"
    rc = _cmd_telemetry_export(_ns(owned_by="general", out=str(out_path), with_content=False))
    assert rc == 0

    lines = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    ex = lines[0]
    assert ex["trajectory_id"] == "exp-1"
    assert ex["owned_by"] == "general"
    assert ex["reward"] == 0.8
    # Redacted by default: no raw prompt/output text on disk.
    blob = json.dumps(ex)
    assert "<REDACTED:" in blob
    assert "secret" not in blob


def test_cli_export_excludes_tenant_owned_ip_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE IP-BOUNDARY CLI SELF-TEST: a tenant-owned corpus exports ZERO general rows.

    Seeding the corpus as tenant-owned (owned_by != general) must produce an empty
    general export -- the same fail-closed boundary the pure distill test pins,
    proven here through the actual CLI handler.
    """
    import json

    from loom.cli import _cmd_telemetry_export

    _seed_corpus(tmp_path, monkeypatch, owner="acme-corp")
    out_path = tmp_path / "ds.jsonl"
    rc = _cmd_telemetry_export(_ns(owned_by="general", out=str(out_path), with_content=False))
    assert rc == 0

    lines = [l for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert lines == []  # the tenant-owned trajectory is EXCLUDED from the general set


def test_cli_trace_shows_one_assembled_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`loom telemetry trace --trajectory exp-1` prints the assembled trajectory."""
    from loom.cli import _cmd_telemetry_trace

    _seed_corpus(tmp_path, monkeypatch)
    rc = _cmd_telemetry_trace(_ns(trajectory="exp-1", with_content=False))
    out = capsys.readouterr().out

    assert rc == 0
    assert "Trajectory exp-1" in out
    assert "verb        : aide" in out
    assert "reward=0.8" in out
    # Content stays redacted in the trace view by default.
    assert "<REDACTED:output>" in out


def test_cli_trace_unknown_id_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`loom telemetry trace` on an unknown id returns 1 (no crash, actionable)."""
    from loom.cli import _cmd_telemetry_trace

    _seed_corpus(tmp_path, monkeypatch)
    rc = _cmd_telemetry_trace(_ns(trajectory="does-not-exist", with_content=False))
    assert rc == 1
