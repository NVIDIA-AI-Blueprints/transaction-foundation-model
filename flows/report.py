"""Loom's read-only experiment-report Metaflow flow.

This module defines the single static ``FlowSpec`` -- :class:`ReportFlow` -- that
the ``loom report`` command runs (via the Metaflow MLOps interface) to **assemble**
a structured analysis / model-card for an experiment: its runs, their metrics, and
their lineage. Report is the **read-only tier** of the approval matrix (design-spec
§3): it trains nothing and writes nothing back; it only reads finished runs through
the Metaflow Client API and emits a Metaflow run + an ``@card``. It NEVER prompts.

The input is either an ``experiment_id`` (the stable tag ``loom-optimize`` groups a
run's candidates under) or an explicit comma list of ``run_pathspecs``. Either way
the flow gathers the matching runs -- their pathspec, success, tags, the metric the
run carried, and the learnings rows that reference them -- through the Client API
only; Loom never touches the underlying datastore (local or S3/minio) directly.

Flow shape::

    start --> assemble --> end

* ``start``    -- resolve the experiment's runs (by ``experiment_id`` tag or by the
                  explicit ``run_pathspecs`` list) into a list of small run dicts,
                  via the Metaflow Client API + the learnings corpus. READ-ONLY.
* ``assemble`` -- assemble the structured report via the pure, unit-testable
                  :func:`assemble_report`, store it on ``self.report``, and render
                  an ``@card`` (overview, leaderboard, per-run lineage).
* ``end``      -- carry ``self.report`` forward so ``Run.data.report`` exposes it to
                  the MLOps interface's Client-API read.

The assembly *logic* is factored into the module-level pure function
:func:`assemble_report` so it is unit-testable on small in-memory run dicts with no
Metaflow involved (the test mocks the Client-API gather and feeds the dicts
straight in). The flow step is a thin wrapper that gathers + calls it. The
*narrative prose* of a report is the SKILL's job; this flow assembles the
structured data + the card only.

Only standard Metaflow APIs are used (``FlowSpec``, ``@step``, ``Parameter``,
``@card``, and the ``current.card`` append API). ``metaflow`` / ``loom`` are
imported *inside* the steps so the flow file parses even where they are not yet
importable until the Runner subprocess sets up the environment.
"""

from __future__ import annotations

from typing import Any

from metaflow import FlowSpec, Parameter, card, current, step

#: How many leaderboard rows to surface in the report card (best first).
_LEADERBOARD_LIMIT = 25


def assemble_report(
    experiment_id: str | None,
    runs: list[dict],
    learnings: list[dict] | None = None,
) -> dict:
    """Assemble an experiment's runs + metrics + lineage into a report dict (pure).

    This is the unit-testable core of :class:`ReportFlow`: given already-gathered
    run dicts (each from the Client API) and optional learnings rows, it builds the
    whole structured model-card with no Metaflow involved. Domain-neutral -- it
    reports only what the runs and learnings actually carry, never a vertical or a
    column meaning.

    Each input run dict is expected to carry (best-effort; missing keys tolerated):
    ``pathspec``, ``successful`` (bool), ``metric`` (float or ``None``), ``flow``
    (the flow name), ``created_at``, ``finished_at``, and ``tags`` (list). Each
    learnings row is the JSON shape :class:`loom.learnings.LearningRecord` persists
    (``command`` / ``task`` / ``outcome`` / ``artifacts`` / ``success`` / ...).

    Args:
        experiment_id: The experiment the report is for (``None`` when the report
            was assembled from an explicit pathspec list rather than a tag).
        runs: The gathered run dicts (see above).
        learnings: Optional learnings rows referencing these runs.

    Returns:
        A JSON-able report dict with keys: ``experiment_id``, ``n_runs``,
        ``n_successful``, ``best_metric``, ``best_run`` (pathspec of the best run or
        ``None``), ``metric_spread`` (``{"min", "max", "mean"}`` over scored runs or
        ``None``), ``leaderboard`` (run dicts sorted best-first, capped), ``runs``
        (every run dict, in input order), ``learnings`` (compact rows: command /
        success / best_metric / data_ref / artifacts), and ``verdict``
        (``"OK"`` when at least one successful run exists, else ``"EMPTY"``).
    """
    runs = list(runs or [])
    learnings = list(learnings or [])

    n_runs = len(runs)
    successful = [r for r in runs if bool(r.get("successful"))]
    n_successful = len(successful)

    # Scored runs: those carrying a numeric metric (regardless of success flag, but
    # a metric usually implies a completed run).
    scored = [
        r for r in runs if isinstance(r.get("metric"), (int, float))
    ]
    metrics = [float(r["metric"]) for r in scored]

    # Leaderboard: scored runs best-first (higher metric first -- the report does
    # not know the optimization direction, so it surfaces the spread either way and
    # the SKILL narrates direction). Unscored runs trail, ordered by success.
    leaderboard = sorted(
        runs,
        key=lambda r: (
            not isinstance(r.get("metric"), (int, float)),  # scored first
            -(float(r["metric"]) if isinstance(r.get("metric"), (int, float)) else 0.0),
            not bool(r.get("successful")),
        ),
    )[:_LEADERBOARD_LIMIT]

    best_metric = max(metrics) if metrics else None
    best_run = None
    if metrics:
        best = max(scored, key=lambda r: float(r["metric"]))
        best_run = best.get("pathspec")

    metric_spread = None
    if metrics:
        metric_spread = {
            "min": round(min(metrics), 6),
            "max": round(max(metrics), 6),
            "mean": round(sum(metrics) / len(metrics), 6),
        }

    compact_learnings = [_compact_learning(row) for row in learnings]

    return {
        "experiment_id": experiment_id,
        "n_runs": n_runs,
        "n_successful": n_successful,
        "best_metric": round(best_metric, 6) if best_metric is not None else None,
        "best_run": best_run,
        "metric_spread": metric_spread,
        "leaderboard": leaderboard,
        "runs": runs,
        "learnings": compact_learnings,
        "verdict": "OK" if n_successful else "EMPTY",
    }


def _compact_learning(row: dict) -> dict:
    """Project a learnings row to the small fields the report card surfaces.

    Reads only references + scalars (never raw rows or secrets, of which the
    learnings schema carries none): the command, the success flag, the best
    metric, the data reference, and the artifact pathspecs.
    """
    task = row.get("task") or {}
    outcome = row.get("outcome") or {}
    return {
        "command": row.get("command"),
        "success": bool(row.get("success")),
        "best_metric": outcome.get("best_metric"),
        "data_ref": task.get("data_ref"),
        "experiment_id": task.get("experiment_id"),
        "artifacts": list(row.get("artifacts") or []),
    }


def gather_runs_by_experiment(
    experiment_id: str,
    flow_names: list[str] | None = None,
) -> list[dict]:
    """Gather an experiment's runs into report dicts via the Metaflow Client API.

    Reads runs tagged ``loom_experiment:<experiment_id>`` across the candidate /
    lifecycle flows through the Client API only -- never touching the datastore.
    Each run becomes a small dict (pathspec / success / metric / flow / timestamps
    / tags). Best-effort: an unreadable run is skipped; a missing flow yields no
    rows. ``metaflow`` is imported lazily so importing this module never requires
    it.

    Args:
        experiment_id: The experiment to read runs for.
        flow_names: Flow names to scan (defaults to the candidate-eval flow plus the
            lifecycle flows that tag an experiment).

    Returns:
        A list of run dicts (unordered; :func:`assemble_report` ranks them).
    """
    from metaflow import Flow, namespace

    try:
        namespace(None)
    except Exception:  # pragma: no cover - namespace API edge case
        pass

    names = flow_names or ["EvalCandidate", "ValidateFlow"]
    tag = f"loom_experiment:{experiment_id}"
    out: list[dict] = []
    for flow_name in names:
        try:
            flow = Flow(flow_name)
            matching = flow.runs(tag)
        except Exception:  # noqa: BLE001 - no such flow yet / metadata down
            continue
        for run in matching:
            row = _run_to_dict(run, flow_name)
            if row is not None:
                out.append(row)
    return out


def gather_runs_by_pathspecs(run_pathspecs: list[str]) -> list[dict]:
    """Gather explicit run pathspecs into report dicts via the Client API.

    Resolves each ``"<FlowName>/<run_id>"`` pathspec to a ``metaflow.Run`` and
    projects it to a small report dict, skipping any that cannot be read. Used by
    ``loom report --runs <pathspec,...>``. ``metaflow`` is imported lazily.

    Args:
        run_pathspecs: Run pathspecs to read.

    Returns:
        A list of run dicts (input order preserved for resolvable runs).
    """
    from metaflow import Run, namespace

    try:
        namespace(None)
    except Exception:  # pragma: no cover - namespace API edge case
        pass

    out: list[dict] = []
    for spec in run_pathspecs:
        spec = (spec or "").strip()
        if not spec:
            continue
        try:
            run = Run(spec)
        except Exception:  # noqa: BLE001 - bad/unresolvable pathspec
            continue
        flow_name = spec.split("/")[0]
        row = _run_to_dict(run, flow_name)
        if row is not None:
            out.append(row)
    return out


def _run_to_dict(run: Any, flow_name: str) -> dict | None:
    """Project a ``metaflow.Run`` to a small report dict via the Client API.

    Reads only the Client-API surface (``pathspec`` / ``successful`` / ``tags`` /
    timestamps and a ``metric``/``best_metric``/``report`` artifact off ``.data``).
    Best-effort: any failure yields ``None`` so one unreadable run never breaks the
    gather.
    """
    try:
        pathspec = getattr(run, "pathspec", None) or str(run)
        successful = bool(run.successful)
    except Exception:  # noqa: BLE001 - metadata read edge case
        return None

    metric = None
    try:
        data = run.data if successful else None
        if data is not None:
            for name in ("best_metric", "metric"):
                value = getattr(data, name, None)
                if isinstance(value, (int, float)):
                    metric = float(value)
                    break
            if metric is None:
                report = getattr(data, "report", None)
                if isinstance(report, dict):
                    holdout = report.get("holdout") or {}
                    score = holdout.get("score")
                    if isinstance(score, (int, float)):
                        metric = float(score)
    except Exception:  # noqa: BLE001 - artifact read edge case
        metric = None

    try:
        tags = sorted(str(t) for t in (getattr(run, "tags", []) or []))
    except Exception:  # pragma: no cover - tags read edge case
        tags = []

    return {
        "pathspec": pathspec,
        "flow": flow_name,
        "successful": successful,
        "metric": metric,
        "tags": tags,
        "created_at": str(getattr(run, "created_at", "")),
        "finished_at": str(getattr(run, "finished_at", "")),
    }


class ReportFlow(FlowSpec):
    """Read-only assembly of an experiment's runs + metrics + lineage.

    Gathers the experiment's runs (by ``experiment_id`` tag or an explicit
    ``run_pathspecs`` list) through the Client API + the learnings corpus, assembles
    a structured report with :func:`assemble_report`, and emits a Metaflow run + an
    ``@card``. The report dict is carried on ``self.report`` so the MLOps interface
    reads it back from ``Run.data``. READ-ONLY: trains nothing, writes nothing back.
    """

    #: Experiment id whose runs to report on (the ``loom_experiment:<id>`` tag).
    #: One of ``experiment_id`` / ``run_pathspecs`` is required.
    experiment_id = Parameter(
        "experiment_id",
        default="",
        type=str,
        help="Experiment id to report on (the loom_experiment tag).",
    )

    #: Comma-separated explicit run pathspecs to report on (alternative to
    #: ``experiment_id``), e.g. ``"EvalCandidate/1,EvalCandidate/2"``.
    run_pathspecs = Parameter(
        "run_pathspecs",
        default="",
        type=str,
        help="Comma-separated run pathspecs to report on (alt to experiment_id).",
    )

    @step
    def start(self) -> None:
        """Resolve the experiment's runs + learnings rows via the Client API.

        Gathers runs by ``experiment_id`` tag or the explicit ``run_pathspecs``
        list (Client API only), and the learnings rows that reference this
        experiment (best-effort, from the corpus). READ-ONLY.
        """
        experiment_id = (self.experiment_id or "").strip() or None
        pathspecs = [
            s.strip() for s in (self.run_pathspecs or "").split(",") if s.strip()
        ]

        if pathspecs:
            self._runs = gather_runs_by_pathspecs(pathspecs)
        elif experiment_id:
            self._runs = gather_runs_by_experiment(experiment_id)
        else:
            self._runs = []

        # Learnings rows referencing this experiment (best-effort, sanitized; the
        # learnings schema carries no secrets). Read through the public corpus API.
        self._learnings = self._gather_learnings(experiment_id)
        self._experiment_id = experiment_id
        self.next(self.assemble)

    @staticmethod
    def _gather_learnings(experiment_id: str | None) -> list[dict]:
        """Read learnings rows referencing ``experiment_id`` (best-effort).

        Uses :class:`loom.learnings.Learnings` (the public corpus API) and filters
        to rows whose task experiment id matches. Any failure yields ``[]`` so a
        missing corpus never fails the report.
        """
        if not experiment_id:
            return []
        try:
            import dataclasses

            from loom.config import LoomConfig
            from loom.learnings import Learnings

            store = Learnings(LoomConfig.load())
            rows = []
            for rec in store.all():
                raw = dataclasses.asdict(rec)
                task = raw.get("task") or {}
                if task.get("experiment_id") == experiment_id:
                    rows.append(raw)
            return rows
        except Exception:  # noqa: BLE001 - corpus optional / unreadable
            return []

    @card
    @step
    def assemble(self) -> None:
        """Assemble the structured report and render the ``@card``.

        Delegates the whole assembly to the pure :func:`assemble_report`, so the
        logic is unit-testable without Metaflow, then renders a Markdown + Tables
        ``@card`` (overview, leaderboard, per-run lineage) from the report dict.
        """
        self.report = assemble_report(
            self._experiment_id, self._runs, self._learnings
        )
        self._render_card(self.report)
        self.next(self.end)

    def _render_card(self, report: dict) -> None:
        """Render a Markdown + Tables ``@card`` from the report dict.

        Args:
            report: The JSON-able report dict from :func:`assemble_report`.
        """
        from metaflow.cards import Markdown, Table

        current.card.append(Markdown("# Loom experiment report"))
        spread = report.get("metric_spread") or {}
        current.card.append(
            Markdown(
                f"**experiment:** `{report.get('experiment_id')}`  \n"
                f"**runs:** {report.get('n_runs')} "
                f"({report.get('n_successful')} successful)  \n"
                f"**best metric:** "
                f"{report.get('best_metric') if report.get('best_metric') is not None else 'n/a'}"
                f" (`{report.get('best_run')}`)  \n"
                f"**VERDICT:** **{report.get('verdict')}**"
            )
        )

        if spread:
            current.card.append(
                Markdown(
                    f"Metric spread — min `{spread.get('min')}`, "
                    f"mean `{spread.get('mean')}`, max `{spread.get('max')}`."
                )
            )

        # Leaderboard.
        leaderboard = report.get("leaderboard") or []
        if leaderboard:
            current.card.append(Markdown("## Leaderboard"))
            current.card.append(
                Table(
                    [
                        [
                            i,
                            r.get("pathspec", "?"),
                            r.get("flow", ""),
                            f"{r['metric']:.6g}"
                            if isinstance(r.get("metric"), (int, float))
                            else "n/a",
                            "ok" if r.get("successful") else "no",
                        ]
                        for i, r in enumerate(leaderboard, start=1)
                    ],
                    headers=["#", "run", "flow", "metric", "success"],
                )
            )
        else:
            current.card.append(
                Markdown("_No runs found for this experiment._")
            )

        # Lineage (learnings rows).
        learnings = report.get("learnings") or []
        if learnings:
            current.card.append(Markdown("## Lineage (learnings rows)"))
            current.card.append(
                Table(
                    [
                        [
                            row.get("command", ""),
                            "ok" if row.get("success") else "no",
                            f"{row['best_metric']:.6g}"
                            if isinstance(row.get("best_metric"), (int, float))
                            else "n/a",
                            row.get("data_ref", ""),
                            ", ".join(row.get("artifacts") or []) or "—",
                        ]
                        for row in learnings
                    ],
                    headers=["command", "ok", "metric", "data_ref", "artifacts"],
                )
            )

    @step
    def end(self) -> None:
        """Carry ``self.report`` forward so ``Run.data.report`` exposes it.

        Metaflow persists step artifacts, so ``self.report`` (set in ``assemble``)
        is already on ``Run.data``; the MLOps interface reads it back for the
        command's summary. Nothing else to do.
        """
        pass


if __name__ == "__main__":
    ReportFlow()
