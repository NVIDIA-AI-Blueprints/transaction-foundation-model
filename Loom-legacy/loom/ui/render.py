"""PURE renderers: a verb's typed summary -> a Rich renderable.

Each function here takes the small, JSON-able **typed summary** a Loom verb
already produces (the ``summary`` dict on a :class:`~loom.types.RunResult`, or a
:class:`~loom.types.SearchResult`) and returns a Rich renderable -- a
:class:`~rich.table.Table` or a :class:`~rich.panel.Panel` -- ready to hand to a
console. They are **pure** (no I/O, no console, no global state) and tolerant of
missing keys, so they render cleanly to a headless ``StringIO`` console and are
trivially unit-testable.

The summary key names mirror exactly what :mod:`loom.cli`'s ``_print_*_summary``
helpers read, so the rendered output is faithful to each verb:

* :func:`render_datasets` -- ``loom datasets`` rows (pathspec · name · rows ·
  cols · target) -> a Table.
* :func:`render_eda` -- the EDA profile (shape · missingness · target balance ·
  leakage) -> a Panel.
* :func:`render_leaderboard` -- a search/report leaderboard (rank · metric ·
  node · op) -> a Table.
* :func:`render_validate` -- the validation VERDICT (CV · sealed holdout ·
  calibration · leakage), the border colored by PASS/REVIEW/FAIL -> a Panel.
* :func:`render_deploy` -- the deploy GATE (ALLOW/BLOCK · apply on/off) -> a
  Panel.
* :func:`render_train` -- the train cost/STATUS (budget · GPU · $ ·
  BUILT/PLANNED/REFUSED) -> a Panel.
* :func:`render_telemetry_status` -- the corpus summary -> a Panel.
* :func:`render_run_result` -- a GENERIC :class:`~loom.types.RunResult` /
  :class:`~loom.types.SearchResult` renderer (pathspec + @card path + the
  summary key/values).

A thin :func:`render_summary` dispatcher maps a verb name + summary to the right
function, which the REPL uses after dispatching a verb.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence

from rich.table import Table
from rich.text import Text

from loom.ui.theme import make_panel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rich.console import RenderableType

    from loom.types import RunResult, SearchResult


# ---------------------------------------------------------------------------
# Small shared helpers (pure).
# ---------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    """Render a scalar for display, tolerating ``None`` and floats."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _verdict_style(verdict: Optional[str]) -> str:
    """Map a VERDICT/status string to a Loom theme style name.

    PASS/ALLOW/BUILT-like -> green; REVIEW/ATTENTION/PLAN-like -> amber;
    FAIL/BLOCK/REFUSED-like -> rose; anything else -> the neutral border.
    """
    text = (verdict or "").strip().upper()
    if text in {"PASS", "ALLOW", "ALLOWED", "BUILT", "OK", "HEALTHY", "GREEN"}:
        return "loom.pass"
    if text in {"REVIEW", "ATTENTION", "WARN", "PLANNED", "PLAN", "AMBER"}:
        return "loom.review"
    if text in {
        "FAIL",
        "BLOCK",
        "BLOCKED",
        "DENY",
        "DENIED",
        "REFUSED",
        "REFUSED_NO_GPU_TARGET",
        "RED",
    }:
        return "loom.fail"
    return "loom.border"


def _kv_table() -> Table:
    """Build a borderless two-column key/value table (the panel-body workhorse)."""
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="left", style="loom.ash", no_wrap=True)
    table.add_column(justify="left", style="loom.ink")
    return table


def _kv(table: Table, label: str, value: Any, *, value_style: Optional[str] = None) -> None:
    """Append one ``label : value`` row to a key/value grid."""
    rendered = _fmt(value)
    cell: "RenderableType" = (
        Text(rendered, style=value_style) if value_style else rendered
    )
    table.add_row(f"{label} :", cell)


def _coerce_summary(summary_or_result: Any) -> Mapping[str, Any]:
    """Return the summary mapping from either a dict or a RunResult-like object."""
    if isinstance(summary_or_result, Mapping):
        return summary_or_result
    got = getattr(summary_or_result, "summary", None)
    return got if isinstance(got, Mapping) else {}


# ---------------------------------------------------------------------------
# datasets -> a Table.
# ---------------------------------------------------------------------------

def render_datasets(rows: Sequence[Mapping[str, Any]]) -> Table:
    """Render the ``loom datasets`` listing as a Table.

    Args:
        rows: One mapping per ingested data object. Recognized keys (all
            optional): ``pathspec`` / ``name`` / ``nrows`` (or ``rows``) /
            ``ncols`` (or ``cols``) / ``target``.

    Returns:
        A Rich :class:`~rich.table.Table` (pathspec · name · rows · cols ·
        target). An empty input still yields a well-formed, empty table.
    """
    table = Table(
        title=Text("Ingested data objects", style="loom.title"),
        title_justify="left",
        border_style="loom.border",
        header_style="loom.section",
        expand=False,
    )
    table.add_column("pathspec", style="loom.ink", no_wrap=True)
    table.add_column("name", style="loom.ink")
    table.add_column("rows", justify="right", style="loom.stone")
    table.add_column("cols", justify="right", style="loom.stone")
    table.add_column("target", style="loom.ash")

    for row in rows:
        table.add_row(
            _fmt(row.get("pathspec")),
            _fmt(row.get("name") or "-"),
            _fmt(row.get("nrows", row.get("rows"))),
            _fmt(row.get("ncols", row.get("cols"))),
            _fmt(row.get("target")),
        )
    return table


# ---------------------------------------------------------------------------
# eda -> a profile Panel.
# ---------------------------------------------------------------------------

def render_eda(summary: Any) -> "RenderableType":
    """Render an EDA profile summary as a Panel.

    Reads the keys ``loom eda`` records (see ``_print_eda_summary``): ``nrows`` /
    ``ncols`` / ``target`` / ``target_inferred`` / ``target_balance`` /
    ``leakage_flags``.

    Args:
        summary: The EDA summary dict (or a RunResult whose ``.summary`` is one).

    Returns:
        A Rich Panel titled "EDA profile".
    """
    s = _coerce_summary(summary)
    body = _kv_table()
    _kv(body, "rows x cols", f"{_fmt(s.get('nrows'))} x {_fmt(s.get('ncols'))}")
    target = s.get("target")
    inferred = " (inferred)" if s.get("target_inferred") else ""
    _kv(body, "target", f"{target if target is not None else 'none'}{inferred}")

    missing = s.get("missingness") or s.get("missing")
    if missing is not None:
        _kv(body, "missingness", missing)

    balance = s.get("target_balance")
    if balance:
        shown = list(balance.items())[:5]
        pretty = ", ".join(f"{k}={v}" for k, v in shown)
        more = "" if len(balance) <= 5 else f" (+{len(balance) - 5} more)"
        _kv(body, "balance", f"{pretty}{more}")

    flags = s.get("leakage_flags") or []
    if flags:
        _kv(body, "LEAKAGE", f"{len(flags)} flag(s) -- review before features",
            value_style="loom.warning")
        for flag in flags[:10]:
            body.add_row(
                "",
                Text(
                    f"- {flag.get('column')} [{flag.get('kind')}]: "
                    f"{flag.get('detail')}",
                    style="loom.stone",
                ),
            )
    else:
        _kv(body, "leakage", "none detected", value_style="loom.success")

    return make_panel("EDA profile", body)


# ---------------------------------------------------------------------------
# search / report leaderboard -> a Table.
# ---------------------------------------------------------------------------

def render_leaderboard(
    rows: Iterable[Mapping[str, Any]],
    *,
    limit: int = 10,
    title: str = "Leaderboard",
) -> Table:
    """Render a leaderboard (search nodes or report runs) as a ranked Table.

    Tolerates both row shapes the CLI handles (see ``_format_leaderboard_row``):
    the search-native shape (``metric`` / ``node_id`` / ``stage``) and the
    Metaflow run shape (``run_id`` / ``pathspec`` / ``submission_ok`` /
    ``exec_time``), plus the report shape (``pathspec`` + ``metric``).

    Args:
        rows: An iterable of run/node dicts.
        limit: Max rows to display.
        title: The table title.

    Returns:
        A Rich Table (rank · metric · node · op).
    """
    table = Table(
        title=Text(title, style="loom.title"),
        title_justify="left",
        border_style="loom.border",
        header_style="loom.section",
        expand=False,
    )
    table.add_column("rank", justify="right", style="loom.stone", no_wrap=True)
    table.add_column("metric", justify="right", style="loom.ink")
    table.add_column("node", style="loom.ink", no_wrap=True)
    table.add_column("op", style="loom.ash")

    for rank, row in enumerate(list(rows)[:limit], start=1):
        if "metric" in row or "node_id" in row or "stage" in row:
            metric = _fmt(row.get("metric"))
            node = _fmt(row.get("node_id", row.get("id", row.get("pathspec", "?"))))
            op = _fmt(row.get("stage") or row.get("op") or "")
        else:
            # Metaflow run shape: no scored metric; surface the execution outcome.
            metric = "n/a"
            node = _fmt(row.get("run_id", row.get("pathspec", "?")))
            sub = "ok" if row.get("submission_ok") else "no"
            et = row.get("exec_time")
            op = f"submission={sub}" + (
                f" {float(et):.6g}s" if isinstance(et, (int, float)) else ""
            )
        table.add_row(str(rank), metric, node, op)
    return table


# ---------------------------------------------------------------------------
# validate -> a VERDICT Panel (colored by PASS/REVIEW/FAIL).
# ---------------------------------------------------------------------------

def render_validate(summary: Any) -> "RenderableType":
    """Render a validation summary as a VERDICT panel colored by the verdict.

    Reads the keys ``loom validate`` records (see ``_print_validate_summary``):
    ``metric`` / ``verdict`` / ``target`` / ``task_type`` / ``cv`` (mean/std) /
    ``n_folds`` / ``holdout`` (score/n) / ``calibration`` (brier) /
    ``slice_metrics`` / ``leakage_flags``.

    Args:
        summary: The validate summary dict (or a RunResult).

    Returns:
        A Rich Panel whose border is green/amber/rose per PASS/REVIEW/FAIL.
    """
    s = _coerce_summary(summary)
    metric = s.get("metric", "score")
    verdict = s.get("verdict", "?")
    style = _verdict_style(verdict)

    body = _kv_table()
    _kv(body, "target", f"{s.get('target')} ({s.get('task_type')})")

    cv = s.get("cv") or {}
    cv_mean = cv.get("mean")
    if cv_mean is not None:
        std = cv.get("std")
        folds = s.get("n_folds")
        cv_line = _fmt(cv_mean)
        if std is not None:
            cv_line += f" +/- {_fmt(std)}"
        if folds is not None:
            cv_line += f" ({folds}-fold)"
        _kv(body, f"CV {metric}", cv_line)

    holdout = s.get("holdout") or {}
    if holdout.get("score") is not None:
        _kv(body, "holdout", f"{_fmt(holdout.get('score'))} (sealed, n={holdout.get('n')})")

    cal = s.get("calibration")
    if cal and cal.get("brier") is not None:
        _kv(body, "calibration", f"Brier={_fmt(cal.get('brier'))}")

    slices = s.get("slice_metrics")
    if slices:
        shown = ", ".join(
            f"{g}={_fmt(m.get('score'))}" for g, m in list(slices.items())[:5]
        )
        _kv(body, "fairness", shown)

    flags = s.get("leakage_flags") or []
    if flags:
        _kv(body, "LEAKAGE", f"{len(flags)} flag(s) -- explain before trusting",
            value_style="loom.warning")
        for flag in flags[:10]:
            body.add_row(
                "",
                Text(
                    f"- {flag.get('column')} [{flag.get('kind')}]: "
                    f"{flag.get('detail')}",
                    style="loom.stone",
                ),
            )

    _kv(body, "VERDICT", verdict, value_style=style)
    return make_panel("Validation verdict", body, border_style=style)


# ---------------------------------------------------------------------------
# deploy -> a GATE Panel (ALLOW/BLOCK, apply on/off).
# ---------------------------------------------------------------------------

def render_deploy(summary: Any) -> "RenderableType":
    """Render a deploy summary as a GATE panel.

    Reads the keys ``loom deploy`` records (see ``_print_deploy_summary``):
    ``target`` / ``apply`` / ``gate`` (decision/verdict/reasons) /
    ``applied_detail`` / ``verdict``.

    Args:
        summary: The deploy summary dict (or a RunResult).

    Returns:
        A Rich Panel whose border is green when the gate ALLOWs, rose when it
        BLOCKs.
    """
    s = _coerce_summary(summary)
    gate = s.get("gate") or {}
    decision = gate.get("decision", "?")
    allow = bool(gate.get("allow")) or str(decision).strip().upper() in {"ALLOW", "ALLOWED"}
    style = "loom.allow" if allow else "loom.block"
    apply = bool(s.get("apply"))

    body = _kv_table()
    _kv(body, "target", s.get("target"))
    _kv(
        body,
        "apply",
        f"{apply} (real external action {'ON' if apply else 'OFF -- staged plan only'})",
    )
    _kv(body, "upstream", f"validate VERDICT={gate.get('verdict')}")
    _kv(body, "GATE", decision, value_style=style)

    reasons = gate.get("reasons") or []
    if reasons:
        body.add_row("blocked by :", Text(""))
        for reason in reasons[:10]:
            body.add_row("", Text(f"- {reason}", style="loom.stone"))

    applied = s.get("applied_detail")
    if applied and applied.get("entry"):
        _kv(body, "registered", applied.get("entry"))

    _kv(body, "VERDICT", s.get("verdict", "?"), value_style=_verdict_style(s.get("verdict")))
    return make_panel("Deploy gate", body, border_style=style)


# ---------------------------------------------------------------------------
# train -> a cost / STATUS Panel.
# ---------------------------------------------------------------------------

def render_train(summary: Any) -> "RenderableType":
    """Render a train summary as a cost / STATUS panel.

    Reads the keys ``loom train`` records (see ``_print_train_summary``):
    ``backend`` / ``model_builder_provider`` / ``capability`` /
    ``capability_mode`` / ``objective`` / ``budget`` / ``cost`` (headline) /
    ``launch`` / ``launch_posture`` / ``gpu_target`` / ``artifact_pathspec`` /
    ``artifact_kind`` / ``fingerprint`` / ``error`` / ``status``.

    Args:
        summary: The train summary dict (or a RunResult).

    Returns:
        A Rich Panel whose border tracks the STATUS (BUILT green / PLANNED amber
        / REFUSED rose).
    """
    s = _coerce_summary(summary)
    status = s.get("status", "?")
    style = _verdict_style(status)

    body = _kv_table()
    _kv(body, "backend", f"{s.get('backend')} (provider {s.get('model_builder_provider')})")
    _kv(body, "capability", f"{s.get('capability')} (mode {s.get('capability_mode')})")
    _kv(body, "objective", f"{s.get('objective')} (budget {s.get('budget')})")

    cost = s.get("cost") or {}
    if cost.get("headline"):
        _kv(body, "cost (gate)", cost.get("headline"), value_style="loom.warning")

    launch = bool(s.get("launch"))
    _kv(
        body,
        "launch",
        f"{launch} (real GPU launch {'ON' if launch else 'OFF -- plan only'}; "
        f"posture {s.get('launch_posture')})",
    )
    if s.get("gpu_target") is not None:
        _kv(body, "gpu_target", s.get("gpu_target"))

    _kv(body, "artifact", f"{s.get('artifact_pathspec') or 'none'} ({s.get('artifact_kind')})")
    if s.get("fingerprint"):
        _kv(body, "fingerprint", s.get("fingerprint"))
    if s.get("error"):
        _kv(body, "refused", s.get("error"), value_style="loom.error")

    _kv(body, "STATUS", status, value_style=style)
    return make_panel("Train status", body, border_style=style)


# ---------------------------------------------------------------------------
# telemetry status -> the corpus Panel.
# ---------------------------------------------------------------------------

def render_telemetry_status(summary: Any) -> "RenderableType":
    """Render a telemetry-status summary as the corpus panel.

    Reads (all optional): ``events_path`` / ``trajectories_path`` /
    ``proxy_calls_path`` / ``learnings_path`` / ``events`` / ``trajectories`` /
    ``general`` / ``tenant_owned`` / ``capture_enabled`` / ``content_logging`` /
    ``otel``.

    Args:
        summary: The telemetry-status summary dict.

    Returns:
        A Rich Panel titled "Telemetry corpus".
    """
    s = _coerce_summary(summary)
    body = _kv_table()
    _kv(body, "events path", s.get("events_path"))
    _kv(body, "trajectories path", s.get("trajectories_path"))
    _kv(body, "proxy calls path", s.get("proxy_calls_path"))
    _kv(body, "learnings path", s.get("learnings_path"))
    _kv(body, "events", s.get("events"))
    _kv(body, "trajectories", s.get("trajectories"))
    _kv(
        body,
        "IP boundary",
        f"{_fmt(s.get('general'))} general / {_fmt(s.get('tenant_owned'))} tenant-owned",
    )
    if s.get("capture_enabled") is not None:
        _kv(body, "capture", "on" if s.get("capture_enabled") else "off")
    if s.get("content_logging") is not None:
        _kv(body, "content", "on" if s.get("content_logging") else "off (redacted)")
    if s.get("otel") is not None:
        _kv(body, "OTel ops mirror", s.get("otel"))
    return make_panel("Telemetry corpus", body)


# ---------------------------------------------------------------------------
# GENERIC RunResult / SearchResult renderer.
# ---------------------------------------------------------------------------

def render_run_result(
    result: Any,
    *,
    pathspec: Optional[str] = None,
    title: str = "Run result",
) -> "RenderableType":
    """Render any :class:`~loom.types.RunResult` / :class:`~loom.types.SearchResult`.

    The generic fallback the REPL uses for a verb without a bespoke renderer: it
    shows the run pathspec, the ``@card`` path, the success flag, any error, and
    a flat dump of the summary key/values (for a SearchResult, the best metric /
    node count / journal / tree paths).

    Args:
        result: A RunResult-like object (``pathspec`` / ``successful`` /
            ``card_path`` / ``summary`` / ``error``), a SearchResult-like object
            (``best_metric`` / ``node_count`` / ``journal_path`` / ``tree_path``
            / ``best_code``), or a plain summary mapping.
        pathspec: An optional pathspec to show (e.g. the input dataset_ref) when
            the result itself carries none.
        title: The panel title.

    Returns:
        A Rich Panel.
    """
    body = _kv_table()

    # SearchResult shape (the `loom run` output).
    if hasattr(result, "best_metric") and not hasattr(result, "successful"):
        _kv(body, "best metric", getattr(result, "best_metric", None))
        _kv(body, "nodes", getattr(result, "node_count", None))
        _kv(body, "journal", getattr(result, "journal_path", None) or "n/a")
        _kv(body, "tree", getattr(result, "tree_path", None) or "n/a")
        code = getattr(result, "best_code", None)
        if code:
            _kv(body, "best code", f"{code.count(chr(10)) + 1} line(s)")
        else:
            _kv(body, "best code", "none produced")
        return make_panel(title, body)

    # RunResult shape (the lifecycle-verb output).
    run_pathspec = getattr(result, "pathspec", None) or pathspec
    if run_pathspec is not None:
        _kv(body, "run", run_pathspec)
    card = getattr(result, "card_path", None)
    if card is not None:
        _kv(body, "card", card or "n/a")
    if hasattr(result, "successful"):
        ok = bool(getattr(result, "successful"))
        _kv(body, "status", "successful" if ok else "FAILED",
            value_style="loom.success" if ok else "loom.error")
    err = getattr(result, "error", None)
    if err:
        _kv(body, "error", err, value_style="loom.error")

    summary = _coerce_summary(result)
    # Reserved labels already shown above from the RunResult fields; avoid a
    # duplicate row when the summary repeats them.
    _reserved = {"run", "card", "error"}
    if hasattr(result, "successful"):
        _reserved.add("status")
    for key, value in summary.items():
        if key in _reserved:
            continue
        # Skip large/nested structures in the generic dump; show scalars only.
        if isinstance(value, (str, int, float, bool)) or value is None:
            _kv(body, key, value)

    style = "loom.border"
    verdict = summary.get("verdict") or summary.get("status")
    if verdict:
        style = _verdict_style(verdict)
    return make_panel(title, body, border_style=style)


# ---------------------------------------------------------------------------
# Dispatcher.
# ---------------------------------------------------------------------------

#: Verb name -> the bespoke renderer for that verb's summary. The REPL looks a
#: verb up here after dispatching it; an unlisted verb falls back to the generic
#: :func:`render_run_result`.
_VERB_RENDERERS = {
    "eda": render_eda,
    "validate": render_validate,
    "deploy": render_deploy,
    "train": render_train,
}


def render_summary(verb: str, summary: Any) -> "RenderableType":
    """Dispatch a verb name + its summary to the right renderer.

    Args:
        verb: The verb token (e.g. ``"eda"``, ``"validate"``).
        summary: The verb's typed summary dict (or a RunResult).

    Returns:
        The Rich renderable from the matching renderer, or the generic
        :func:`render_run_result` for an unmapped verb.
    """
    renderer = _VERB_RENDERERS.get(verb)
    if renderer is not None:
        return renderer(summary)
    return render_run_result(summary, title=f"{verb} result")


__all__ = [
    "render_datasets",
    "render_eda",
    "render_leaderboard",
    "render_validate",
    "render_deploy",
    "render_train",
    "render_telemetry_status",
    "render_run_result",
    "render_summary",
]
