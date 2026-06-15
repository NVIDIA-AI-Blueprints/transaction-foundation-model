"""Tests for ``loom.ui.render`` -- the PURE verb-summary -> Rich renderer layer.

Each renderer takes the small typed summary a Loom verb already produces (a dict,
or a :class:`~loom.types.RunResult` / :class:`~loom.types.SearchResult` it reads
``.summary`` off) and returns a Rich renderable (a Table or a Panel). They are
pure -- no I/O, no global state -- and tolerant of missing keys, so the contract
these tests pin is:

* every renderer renders to a HEADLESS console (``Console(file=StringIO())``)
  WITHOUT raising;
* the rendered text contains the load-bearing field labels / values for that
  verb (the pathspec, the VERDICT, the cost line, ...);
* the verdict-colored panels (validate / deploy / train) carry the right
  PASS/REVIEW/FAIL · ALLOW/BLOCK · BUILT/PLANNED/REFUSED border style;
* a renderer tolerates BOTH a plain summary dict and a RunResult/SearchResult,
  and tolerates an empty / missing-key summary;
* the :func:`~loom.ui.render.render_summary` dispatcher routes the bespoke verbs
  to their renderer and everything else to the generic one;
* :func:`~loom.ui.theme.banner` renders the ASCII logo + tagline + version +
  providers.

Everything is headless (no TTY): the console is built over a ``StringIO`` via the
shared :func:`loom.ui.theme.get_console` factory, exactly the way the REPL builds
its console in tests.
"""

from __future__ import annotations

from io import StringIO

from rich.panel import Panel
from rich.table import Table

from loom.types import RunResult, SearchResult
from loom.ui import render, theme


# ---------------------------------------------------------------------------
# Headless console helper + a renderer-to-text helper.
# ---------------------------------------------------------------------------


def _console() -> theme.Console:
    """Build a fixed-width, terminal-forced Loom console over a buffer."""
    return theme.get_console(file=StringIO(), force_terminal=False, width=120)


def _text(renderable) -> str:
    """Render a Rich renderable to plain text via a headless console."""
    console = _console()
    console.print(renderable)
    return console.file.getvalue()


# ---------------------------------------------------------------------------
# datasets -> a Table.
# ---------------------------------------------------------------------------


def test_render_datasets_table_has_rows_and_headers() -> None:
    rows = [
        {"pathspec": "IngestDataset/1", "name": "fraud", "nrows": 1000, "ncols": 12, "target": "is_fraud"},
        # the alt key names (rows/cols) the renderer also accepts:
        {"pathspec": "IngestDataset/2", "name": "churn", "rows": 500, "cols": 8, "target": "churned"},
    ]
    table = render.render_datasets(rows)
    assert isinstance(table, Table)
    out = _text(table)
    for needle in ("IngestDataset/1", "fraud", "1000", "is_fraud", "IngestDataset/2", "churn", "pathspec", "target"):
        assert needle in out, needle


def test_render_datasets_empty_is_well_formed() -> None:
    # An empty listing must still render a valid (empty) table without raising.
    out = _text(render.render_datasets([]))
    assert "pathspec" in out  # the header row still renders


# ---------------------------------------------------------------------------
# eda -> a profile Panel.
# ---------------------------------------------------------------------------


def test_render_eda_profile_panel_with_leakage() -> None:
    summary = {
        "nrows": 1000,
        "ncols": 12,
        "target": "y",
        "target_inferred": True,
        "target_balance": {"0": 800, "1": 200},
        "leakage_flags": [{"column": "id", "kind": "id-like", "detail": "unique per row"}],
    }
    panel = render.render_eda(summary)
    assert isinstance(panel, Panel)
    out = _text(panel)
    assert "EDA profile" in out
    assert "1000" in out and "12" in out
    assert "(inferred)" in out
    # the leakage flag surfaces with its column + kind
    assert "id" in out and "id-like" in out
    assert "LEAKAGE" in out


def test_render_eda_reads_runresult_dot_summary() -> None:
    # render_eda must accept a RunResult and read .summary off it.
    result = RunResult(pathspec="EdaFlow/3", successful=True, summary={"nrows": 5, "ncols": 2})
    out = _text(render.render_eda(result))
    assert "EDA profile" in out and "5" in out


def test_render_eda_tolerates_empty_summary() -> None:
    out = _text(render.render_eda({}))
    assert "EDA profile" in out  # no raise, panel still titled


# ---------------------------------------------------------------------------
# leaderboard -> a Table (both row shapes).
# ---------------------------------------------------------------------------


def test_render_leaderboard_handles_search_and_metaflow_shapes() -> None:
    rows = [
        {"metric": 0.91, "node_id": "n1", "stage": "improve"},  # search-native
        {"run_id": "EvalCandidate/3", "submission_ok": True, "exec_time": 2.5},  # metaflow run
    ]
    table = render.render_leaderboard(rows, title="Leaderboard")
    assert isinstance(table, Table)
    out = _text(table)
    assert "Leaderboard" in out
    assert "n1" in out and "improve" in out
    assert "EvalCandidate/3" in out and "submission" in out


def test_render_leaderboard_respects_limit() -> None:
    rows = [{"metric": float(i), "node_id": f"n{i}"} for i in range(20)]
    out = _text(render.render_leaderboard(rows, limit=3))
    assert "n0" in out and "n2" in out
    assert "n5" not in out  # truncated by the limit


# ---------------------------------------------------------------------------
# validate -> a VERDICT Panel colored by PASS/REVIEW/FAIL.
# ---------------------------------------------------------------------------


def test_render_validate_pass_panel() -> None:
    summary = {
        "metric": "roc_auc",
        "verdict": "PASS",
        "target": "y",
        "task_type": "binary",
        "cv": {"mean": 0.9, "std": 0.01},
        "n_folds": 5,
        "holdout": {"score": 0.88, "n": 200},
        "calibration": {"brier": 0.1},
    }
    panel = render.render_validate(summary)
    assert isinstance(panel, Panel)
    assert panel.border_style == "loom.pass"
    out = _text(panel)
    assert "PASS" in out
    assert "holdout" in out and "0.88" in out
    assert "Brier" in out


def test_render_validate_fail_border() -> None:
    panel = render.render_validate({"verdict": "FAIL", "target": "y", "task_type": "binary"})
    assert panel.border_style == "loom.fail"
    assert "FAIL" in _text(panel)


def test_render_validate_review_border() -> None:
    panel = render.render_validate({"verdict": "REVIEW", "target": "y", "task_type": "binary"})
    assert panel.border_style == "loom.review"


# ---------------------------------------------------------------------------
# deploy -> a GATE Panel (ALLOW/BLOCK).
# ---------------------------------------------------------------------------


def test_render_deploy_block_panel() -> None:
    summary = {
        "target": "registry",
        "apply": False,
        "gate": {"decision": "BLOCK", "allow": False, "verdict": "REVIEW", "reasons": ["validate not PASS"]},
        "verdict": "BLOCK",
    }
    panel = render.render_deploy(summary)
    assert panel.border_style == "loom.block"
    out = _text(panel)
    assert "BLOCK" in out
    assert "validate not PASS" in out  # the gate reason surfaces
    assert "OFF" in out  # apply off -> staged plan only


def test_render_deploy_allow_panel() -> None:
    summary = {
        "target": "registry",
        "apply": True,
        "gate": {"decision": "ALLOW", "allow": True, "verdict": "PASS"},
        "applied_detail": {"entry": "registry://fraud@1"},
        "verdict": "ALLOW",
    }
    panel = render.render_deploy(summary)
    assert panel.border_style == "loom.allow"
    out = _text(panel)
    assert "ALLOW" in out and "registry://fraud@1" in out


# ---------------------------------------------------------------------------
# train -> a cost / STATUS Panel (BUILT/PLANNED/REFUSED).
# ---------------------------------------------------------------------------


def test_render_train_planned_panel_shows_cost() -> None:
    summary = {
        "backend": "local",
        "model_builder_provider": "local",
        "capability": "pretrain",
        "capability_mode": "launch-and-track",
        "objective": "next-event",
        "budget": "probe",
        "cost": {"headline": "~2 GPU-h ~$4"},
        "launch": False,
        "launch_posture": "plan",
        "status": "PLANNED",
    }
    panel = render.render_train(summary)
    assert panel.border_style == "loom.review"  # PLANNED -> amber
    out = _text(panel)
    assert "PLANNED" in out
    assert "~2 GPU-h ~$4" in out
    assert "pretrain" in out


def test_render_train_built_and_refused_borders() -> None:
    built = render.render_train({"status": "BUILT", "artifact_pathspec": "TrainFlow/9", "artifact_kind": "backbone"})
    assert built.border_style == "loom.pass"
    assert "TrainFlow/9" in _text(built)

    refused = render.render_train({"status": "REFUSED_NO_GPU_TARGET", "error": "no gpu_target set"})
    assert refused.border_style == "loom.fail"
    assert "no gpu_target set" in _text(refused)


# ---------------------------------------------------------------------------
# telemetry status -> the corpus Panel.
# ---------------------------------------------------------------------------


def test_render_telemetry_status_panel() -> None:
    summary = {
        "events": 10,
        "trajectories": 3,
        "general": 2,
        "tenant_owned": 1,
        "events_path": "/x/events.jsonl",
        "capture_enabled": True,
        "content_logging": False,
    }
    panel = render.render_telemetry_status(summary)
    assert isinstance(panel, Panel)
    out = _text(panel)
    assert "Telemetry corpus" in out
    assert "/x/events.jsonl" in out
    assert "general" in out and "tenant-owned" in out
    assert "redacted" in out  # content off -> redacted note


# ---------------------------------------------------------------------------
# generic RunResult / SearchResult renderer + the dispatcher.
# ---------------------------------------------------------------------------


def test_render_run_result_searchresult_shape() -> None:
    result = SearchResult(best_code="x = 1", best_metric=0.9, journal_path="/j", tree_path="/t", node_count=4)
    panel = render.render_run_result(result)
    out = _text(panel)
    assert "best metric" in out and "0.9" in out
    assert "nodes" in out and "4" in out
    assert "line(s)" in out  # best_code -> a line count, never inlined bytes


def test_render_run_result_runresult_shape_with_pathspec_and_card() -> None:
    result = RunResult(pathspec="OpsFlow/2", successful=True, card_path="card://x", summary={"status": "HEALTHY"})
    out = _text(render.render_run_result(result))
    assert "OpsFlow/2" in out
    assert "card://x" in out
    assert "successful" in out


def test_render_run_result_failed_shows_error() -> None:
    result = RunResult(pathspec=None, successful=False, error="flow failed to start")
    out = _text(render.render_run_result(result))
    assert "FAILED" in out and "flow failed to start" in out


def test_render_summary_dispatches_bespoke_and_generic() -> None:
    # A mapped verb -> its bespoke renderer (validate -> a verdict-colored panel).
    validate_panel = render.render_summary("validate", {"verdict": "PASS", "target": "y", "task_type": "binary"})
    assert isinstance(validate_panel, Panel)
    assert validate_panel.border_style == "loom.pass"

    # An unmapped verb -> the generic run-result renderer, titled "<verb> result".
    generic = render.render_summary("ops", RunResult(pathspec="OpsFlow/2", successful=True, summary={"status": "HEALTHY"}))
    out = _text(generic)
    assert "ops result" in out and "OpsFlow/2" in out


# ---------------------------------------------------------------------------
# banner() -- the ASCII logo + tagline + version + providers.
# ---------------------------------------------------------------------------


class _FakeConfig:
    """A minimal duck-typed config carrying only the provider/model names."""

    search_provider = "aide"
    mlops_provider = "metaflow"
    model_builder_provider = "local"
    code_provider = "anthropic-api"
    feedback_provider = "anthropic-api"


def test_banner_renders_logo_tagline_version_providers() -> None:
    console = _console()
    returned = theme.banner(_FakeConfig(), console=console)
    assert returned is console  # banner returns the console it printed on
    out = console.file.getvalue()
    # the ASCII LOOM block logo (first row of the bundled glyph art)
    assert theme.LOOM_ASCII_LOGO[0].strip()[:2] in out or "█" in out
    assert theme.TAGLINE in out  # "an agentic CLI for data science"
    assert "v" in out  # the version line
    # the active providers summary
    assert "aide" in out and "metaflow" in out and "anthropic-api" in out


def test_banner_tolerates_partial_config() -> None:
    # A config missing provider attrs must degrade to "?" rather than raise.
    class _Bare:
        pass

    console = _console()
    theme.banner(_Bare(), console=console)
    out = console.file.getvalue()
    assert theme.TAGLINE in out
    assert "?" in out  # missing provider names degrade to a placeholder


# ---------------------------------------------------------------------------
# theme helpers render cleanly to a headless console.
# ---------------------------------------------------------------------------


def test_theme_helpers_print_without_error() -> None:
    console = _console()
    theme.info(console, "an info line")
    theme.success(console, "a success line")
    theme.warning(console, "a warning line")
    theme.error(console, "an error line")
    theme.section(console, "A section")
    theme.panel(console, "A title", "a body")
    out = console.file.getvalue()
    for needle in ("an info line", "a success line", "a warning line", "an error line", "A section", "A title", "a body"):
        assert needle in out, needle
