"""CLI leaderboard rendering tests: both execution-provider row shapes.

The CLI's :func:`loom.cli._print_result` prints a short leaderboard from whatever
the active execution provider's ``runs()`` returns, and those rows are shaped
differently per provider:

* the ``local`` provider (Loom corpus) emits search-native rows with ``metric`` /
  ``node_id`` / ``stage``;
* the ``metaflow`` provider emits Metaflow run rows with ``run_id`` (a pathspec) /
  ``submission_ok`` / ``exec_time`` / ``exc_type`` and **no** scored metric.

These tests are pure-Python (no AIDE, Metaflow, or any LLM): they call the
formatting helper / printer directly on hand-built rows. The regression guard is
that a Metaflow row must NOT render as ``metric=n/a node=?`` (the old bug) and a
local row must keep rendering ``metric=...  node=...``.
"""

from __future__ import annotations

import pytest

from loom.cli import _format_leaderboard_row, _judge_preflight, _print_result
from loom.config import LoomConfig
from loom.types import SearchResult


@pytest.fixture
def clean_env_for_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear OpenRouter/OpenAI env so the judge pre-flight is test-controlled."""
    for name in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENROUTER_HTTP_REFERER",
        "OPENROUTER_X_TITLE",
    ):
        monkeypatch.delenv(name, raising=False)


# A representative row from each provider's ``runs()``.
_LOCAL_ROW = {
    "node_id": "n3",
    "parent_id": "n1",
    "stage": "improve",
    "metric": 0.873,
    "exc_type": None,
    "judge_summary": "looks good",
    "model": "claude-sonnet-4-5",
    "ts": 1700000000.0,
}
_METAFLOW_ROW = {
    "run_id": "EvalCandidate/42",
    "pathspec": "EvalCandidate/42",
    "experiment_id": "loom-abc",
    "successful": True,
    "submission_ok": True,
    "exc_type": None,
    "exec_time": 12.5,
    "created_at": "2026-06-09",
    "finished_at": "2026-06-09",
}


def test_local_row_renders_metric_and_node() -> None:
    """A local/corpus row keeps the search-native ``metric=...  node=...`` shape."""
    line = _format_leaderboard_row(1, _LOCAL_ROW)
    assert "metric=0.873" in line
    assert "node=n3" in line
    assert "[improve]" in line
    # It must NOT fall through to the Metaflow run shape.
    assert "run=" not in line
    assert "submission=" not in line


def test_metaflow_row_renders_pathspec_and_submission() -> None:
    """A Metaflow row renders pathspec + submission + exec_time, never metric=n/a."""
    line = _format_leaderboard_row(1, _METAFLOW_ROW)
    assert "run=EvalCandidate/42" in line
    assert "submission=ok" in line
    assert "exec_time=12.5s" in line
    # The regression: it must NOT render as the empty search shape.
    assert "metric=n/a" not in line
    assert "node=?" not in line


def test_metaflow_row_without_submission_or_metric_is_graceful() -> None:
    """A failed Metaflow run (no submission, exc set) still renders cleanly."""
    row = {
        "run_id": "EvalCandidate/7",
        "submission_ok": False,
        "exec_time": 0.0,
        "exc_type": "ValueError",
    }
    line = _format_leaderboard_row(2, row)
    assert "run=EvalCandidate/7" in line
    assert "submission=no" in line
    assert "exc=ValueError" in line
    assert "metric=n/a" not in line


def test_metaflow_row_falls_back_to_pathspec_key() -> None:
    """When only ``pathspec`` (not ``run_id``) is present, it is still rendered."""
    row = {"pathspec": "EvalCandidate/9", "submission_ok": True}
    line = _format_leaderboard_row(1, row)
    assert "run=EvalCandidate/9" in line


def test_print_result_renders_mixed_shapes(capsys) -> None:
    """_print_result handles a metaflow leaderboard end-to-end without the old bug."""
    result = SearchResult(
        best_metric=0.9,
        node_count=3,
        best_code="print('x')",
        journal_path=None,
        tree_path=None,
    )
    _print_result(result, [_METAFLOW_ROW])
    out = capsys.readouterr().out
    assert "Leaderboard (top 1):" in out
    assert "run=EvalCandidate/42" in out
    assert "metric=n/a  node=?" not in out


def test_print_result_renders_local_shape(capsys) -> None:
    """_print_result keeps the local leaderboard's metric/node rendering."""
    result = SearchResult(
        best_metric=0.873,
        node_count=3,
        best_code=None,
        journal_path=None,
        tree_path=None,
    )
    _print_result(result, [_LOCAL_ROW])
    out = capsys.readouterr().out
    assert "Leaderboard (top 1):" in out
    assert "metric=0.873" in out
    assert "node=n3" in out


# ---------------------------------------------------------------------------
# Judge pre-flight: the OpenRouter slug guard is wired into the CLI gate.
# ---------------------------------------------------------------------------


def test_judge_preflight_blocks_bare_openrouter_feedback_slug(
    monkeypatch, clean_env_for_judge
) -> None:
    """The CLI judge pre-flight fails fast (shape message) for a bare OR slug."""
    cfg = LoomConfig(
        code_provider="openrouter",
        feedback_provider="openrouter",
        code_model="anthropic/claude-sonnet-4.5",
        feedback_model="claude-sonnet-4-5",  # bare reserved slug -> not routable
    )
    msg = _judge_preflight(cfg)
    assert msg is not None
    assert "provider/model" in msg
    assert "openrouter" in msg


def test_judge_preflight_passes_good_openrouter_feedback_slug(
    monkeypatch, clean_env_for_judge
) -> None:
    """A proper, tool-capable provider/model feedback slug passes the CLI gate."""
    import loom.providers.model.openrouter as orouter

    monkeypatch.setattr(orouter, "feedback_slug_supports_tools", lambda _slug: True)
    cfg = LoomConfig(
        code_provider="openrouter",
        feedback_provider="openrouter",
        code_model="anthropic/claude-sonnet-4.5",
        feedback_model="anthropic/claude-sonnet-4.5",
    )
    assert _judge_preflight(cfg) is None
