"""Loom's read-only operations / monitoring Metaflow flow.

This module defines the single static ``FlowSpec`` -- :class:`OpsFlow` -- that the
``loom ops`` command runs (via the Metaflow MLOps interface) to **monitor** Loom's
own runs and data objects. Ops is the **read-only tier** of the approval matrix
(design-spec §3; ``CONVENTIONS.md`` §1): it trains nothing, writes nothing back, and
**never prompts** -- it only reads finished runs and data-object schemas through the
Metaflow Client API and emits a Metaflow run + an ``@card``.

What it monitors (any/all, depending on the input given):

* **run health** -- recent runs of a flow (or experiment), with success/failure
  counts and the most recent outcomes;
* **the leaderboard** -- the ranked runs for an experiment (the same shape the
  ``metaflow`` provider's ``runs()`` returns);
* **schedule / run health** -- the recency + success rate that a scheduled flow's
  health is read from;
* a simple **DRIFT check** -- compare a data object's schema / summary stats to a
  *reference* data object and surface columns whose distribution moved.

The input is one of: a ``flow_name`` (run health), an ``experiment`` id
(leaderboard + health), or a ``dataset_ref`` + ``reference`` pair (drift). Data is
read through the Metaflow **Client API** only; Loom never touches the underlying
datastore (local or S3/minio) directly.

Flow shape::

    start --> observe --> end

* ``start``   -- gather the requested observations via the Client API (recent runs
                 for a flow/experiment; the two data objects' schemas/stats for a
                 drift check). READ-ONLY.
* ``observe`` -- summarize run health with the pure :func:`summarize_ops` and (when
                 a drift pair was given) compute drift with the pure
                 :func:`compute_drift`, store the typed summary on ``self.summary``,
                 and render an ``@card`` (run health, leaderboard, drift table).
* ``end``     -- carry ``self.summary`` forward so ``Run.data.summary`` exposes it.

The monitoring *logic* is factored into the module-level pure functions
:func:`summarize_ops` and :func:`compute_drift` so they are unit-testable on small
in-memory run dicts / DataFrames with no Metaflow involved. The flow steps are thin
wrappers that gather + call them.

Only standard Metaflow APIs are used (``FlowSpec``, ``@step``, ``Parameter``,
``@card``, and the ``current.card`` append API). ``metaflow`` / ``pandas`` /
``loom`` are imported *inside* the steps/functions so the flow file parses even
where they are not yet importable until the Runner subprocess sets up the
environment.
"""

from __future__ import annotations

from typing import Any

from metaflow import FlowSpec, Parameter, card, current, step

#: How many recent runs to surface in the run-health table (most recent first).
_RECENT_LIMIT = 25

#: Relative-mean-shift threshold at/above which a numeric column is flagged as
#: drifted (``|mean_ref - mean_cur| / (|mean_ref| + eps)``). Deliberately a simple,
#: dependency-free smell test -- a proper KS/PSI check is a forward extension; this
#: catches an obvious distribution move without importing scipy.
_DRIFT_MEAN_SHIFT_THRESHOLD = 0.25

#: Absolute null-rate shift (fraction, 0..1) at/above which a column is flagged as
#: drifted on missingness (e.g. a feed that started arriving 30% null).
_DRIFT_NULL_SHIFT_THRESHOLD = 0.2


def summarize_ops(
    runs: list[dict],
    flow_name: str | None = None,
    experiment_id: str | None = None,
) -> dict:
    """Summarize run / schedule health from gathered run dicts (pure function).

    This is the unit-testable core of :class:`OpsFlow` for the run-health view:
    given already-gathered run dicts (each from the Client API) it computes the
    health rollup with no Metaflow involved. Domain-neutral -- it reports only what
    the runs carry (success flags, timestamps, metrics), never a vertical.

    Each input run dict is read best-effort for: ``pathspec``, ``successful``
    (bool), ``flow``, ``created_at`` / ``finished_at``, ``exc_type``, and ``metric``.

    Args:
        runs: The gathered run dicts.
        flow_name: The flow these runs belong to (echoed into the summary), or
            ``None`` when gathered by experiment.
        experiment_id: The experiment these runs belong to (echoed), or ``None``.

    Returns:
        A JSON-able health dict with keys: ``flow_name``, ``experiment_id``,
        ``n_runs``, ``n_successful``, ``n_failed``, ``success_rate`` (0..1 or
        ``None`` when no runs), ``last_run`` (the most recent run dict or ``None``),
        ``recent`` (the most recent runs, capped), ``leaderboard`` (scored runs
        best-first), and ``status`` (``"HEALTHY"`` when there are runs and the most
        recent succeeded, ``"DEGRADED"`` when the most recent failed, ``"EMPTY"``
        when there are no runs).
    """
    runs = [r for r in (runs or []) if isinstance(r, dict)]
    n_runs = len(runs)
    n_successful = sum(1 for r in runs if bool(r.get("successful")))
    n_failed = n_runs - n_successful
    success_rate = (n_successful / n_runs) if n_runs else None

    # Most recent first by finished/created timestamp (string-sortable ISO is the
    # common case; fall back to the input order's reverse for unparseable stamps).
    def _stamp(r: dict) -> str:
        return str(r.get("finished_at") or r.get("created_at") or "")

    recent_sorted = sorted(runs, key=_stamp, reverse=True)
    recent = recent_sorted[:_RECENT_LIMIT]
    last_run = recent_sorted[0] if recent_sorted else None

    # Leaderboard: scored runs best-first (higher metric first -- ops does not know
    # the optimization direction, so it surfaces order and the card narrates).
    scored = [r for r in runs if isinstance(r.get("metric"), (int, float))]
    leaderboard = sorted(
        scored, key=lambda r: -float(r["metric"])
    )[:_RECENT_LIMIT]

    if not n_runs:
        status = "EMPTY"
    elif last_run is not None and not bool(last_run.get("successful")):
        status = "DEGRADED"
    else:
        status = "HEALTHY"

    return {
        "flow_name": flow_name,
        "experiment_id": experiment_id,
        "n_runs": n_runs,
        "n_successful": n_successful,
        "n_failed": n_failed,
        "success_rate": round(success_rate, 6) if success_rate is not None else None,
        "last_run": last_run,
        "recent": recent,
        "leaderboard": leaderboard,
        "status": status,
    }


def compute_drift(
    current_df: Any,
    reference_df: Any,
    mean_shift_threshold: float = _DRIFT_MEAN_SHIFT_THRESHOLD,
    null_shift_threshold: float = _DRIFT_NULL_SHIFT_THRESHOLD,
) -> dict:
    """Compare a current data object to a reference and flag drifted columns (pure).

    This is the unit-testable core of :class:`OpsFlow`'s drift view: given two pandas
    DataFrames (the current data object and a reference) it computes a simple,
    dependency-free distribution-shift smell test with no Metaflow involved. It is
    domain-neutral -- it compares only the columns the two frames share and reports
    only what their summary stats say, never a vertical.

    The checks per shared column:

    * **schema drift** -- a column present in one frame but absent from the other is
      surfaced under ``added`` / ``removed``;
    * **mean shift** (numeric) -- relative change in the column mean
      ``|mean_ref - mean_cur| / (|mean_ref| + eps)`` at/above ``mean_shift_threshold``;
    * **null-rate shift** -- absolute change in the fraction of missing values
      at/above ``null_shift_threshold`` (a feed that started arriving null).

    Args:
        current_df: The current data object's DataFrame.
        reference_df: The reference (baseline) DataFrame to compare against.
        mean_shift_threshold: Relative mean-shift at/above which a numeric column is
            flagged.
        null_shift_threshold: Absolute null-rate shift at/above which a column is
            flagged.

    Returns:
        A JSON-able drift dict with keys: ``n_shared_columns``, ``added`` (columns
        only in current), ``removed`` (columns only in reference), ``drift_flags``
        (list of ``{"column", "kind", "detail", "ref", "cur"}`` dicts), ``drift``
        (bool, any flag/schema change present), and ``status`` (``"DRIFT"`` when
        drift is present, else ``"STABLE"``).
    """
    import math

    import pandas as pd

    cur_cols = [str(c) for c in current_df.columns]
    ref_cols = [str(c) for c in reference_df.columns]
    cur_set, ref_set = set(cur_cols), set(ref_cols)

    added = sorted(cur_set - ref_set)
    removed = sorted(ref_set - cur_set)
    shared = [c for c in cur_cols if c in ref_set]

    eps = 1e-12
    flags: list[dict] = []

    for col in shared:
        cur = current_df[col]
        ref = reference_df[col]

        # Null-rate shift (applies to every dtype).
        cur_null = float(cur.isna().mean()) if len(cur) else 0.0
        ref_null = float(ref.isna().mean()) if len(ref) else 0.0
        if abs(cur_null - ref_null) >= float(null_shift_threshold):
            flags.append(
                {
                    "column": col,
                    "kind": "null_rate_shift",
                    "detail": f"null rate {ref_null:.3f} -> {cur_null:.3f}",
                    "ref": round(ref_null, 6),
                    "cur": round(cur_null, 6),
                }
            )
            continue  # one flag per column is enough

        # Mean shift (numeric columns only).
        if pd.api.types.is_numeric_dtype(cur) and pd.api.types.is_numeric_dtype(ref):
            cur_mean = cur.dropna()
            ref_mean = ref.dropna()
            if not cur_mean.empty and not ref_mean.empty:
                cm = float(cur_mean.mean())
                rm = float(ref_mean.mean())
                if math.isfinite(cm) and math.isfinite(rm):
                    rel = abs(rm - cm) / (abs(rm) + eps)
                    if rel >= float(mean_shift_threshold):
                        flags.append(
                            {
                                "column": col,
                                "kind": "mean_shift",
                                "detail": (
                                    f"mean {rm:.6g} -> {cm:.6g} "
                                    f"(rel {rel:.3f})"
                                ),
                                "ref": round(rm, 6),
                                "cur": round(cm, 6),
                            }
                        )

    drift = bool(flags or added or removed)
    return {
        "n_shared_columns": len(shared),
        "added": added,
        "removed": removed,
        "drift_flags": flags,
        "drift": drift,
        "status": "DRIFT" if drift else "STABLE",
    }


# ---------------------------------------------------------------------------
# Client-API gather helpers (no datastore access; importable, lazily use Metaflow).
# ---------------------------------------------------------------------------


def gather_flow_runs(flow_name: str, limit: int = 50) -> list[dict]:
    """Gather a flow's recent runs into ops dicts via the Metaflow Client API.

    Reads the named flow's runs through the Client API only -- never touching the
    datastore. Each run becomes a small dict (pathspec / success / flow / timestamps
    / exc_type / metric). Best-effort: an unreadable run is skipped; a missing flow
    yields no rows. ``metaflow`` is imported lazily.

    Args:
        flow_name: The flow whose runs to read (e.g. ``"ValidateFlow"``).
        limit: Maximum number of (most-recent) runs to read.

    Returns:
        A list of run dicts (unordered; :func:`summarize_ops` ranks/recency-sorts).
    """
    from metaflow import Flow, namespace

    try:
        namespace(None)
    except Exception:  # pragma: no cover - namespace API edge case
        pass

    out: list[dict] = []
    try:
        flow = Flow(flow_name)
        runs = flow.runs()
    except Exception:  # noqa: BLE001 - no such flow yet / metadata down
        return out
    for i, run in enumerate(runs):
        if i >= int(limit):
            break
        row = _run_to_ops_dict(run, flow_name)
        if row is not None:
            out.append(row)
    return out


def gather_experiment_runs(
    experiment_id: str, flow_names: list[str] | None = None
) -> list[dict]:
    """Gather an experiment's runs into ops dicts via the Metaflow Client API.

    Reads runs tagged ``loom_experiment:<experiment_id>`` across the candidate /
    lifecycle flows through the Client API only. Mirrors the report flow's gather
    shape so ops and report speak the same run-dict vocabulary. ``metaflow`` is
    imported lazily.

    Args:
        experiment_id: The experiment to read runs for.
        flow_names: Flow names to scan (defaults to the candidate + validate flows).

    Returns:
        A list of run dicts (unordered).
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
            row = _run_to_ops_dict(run, flow_name)
            if row is not None:
                out.append(row)
    return out


def _run_to_ops_dict(run: Any, flow_name: str) -> dict | None:
    """Project a ``metaflow.Run`` to a small ops dict via the Client API.

    Reads only the Client-API surface (``pathspec`` / ``successful`` / timestamps /
    ``exc_type`` and a ``metric``/``best_metric`` artifact off ``.data``).
    Best-effort: any failure yields ``None`` so one unreadable run never breaks the
    gather.
    """
    try:
        pathspec = getattr(run, "pathspec", None) or str(run)
        successful = bool(run.successful)
    except Exception:  # noqa: BLE001 - metadata read edge case
        return None

    metric = None
    exc_type = None
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
        else:
            exc_type = "RunFailed"
    except Exception:  # noqa: BLE001 - artifact read edge case
        metric = None

    return {
        "pathspec": pathspec,
        "flow": flow_name,
        "successful": successful,
        "metric": metric,
        "exc_type": exc_type,
        "created_at": str(getattr(run, "created_at", "")),
        "finished_at": str(getattr(run, "finished_at", "")),
    }


class OpsFlow(FlowSpec):
    """Read-only monitoring of Loom's runs and data objects, emitted as a ``@card``.

    Gathers the requested observations (a flow's recent runs, an experiment's
    leaderboard, or a drift comparison of two data objects) through the Client API,
    summarizes run health with :func:`summarize_ops` and drift with
    :func:`compute_drift`, and emits a Metaflow run + an ``@card``. The typed summary
    is carried on ``self.summary`` so the MLOps interface reads it back from
    ``Run.data``. READ-ONLY: trains nothing, writes nothing back, never prompts.
    """

    #: Flow name whose recent run health to read (e.g. ``ValidateFlow``). One of
    #: ``flow_name`` / ``experiment`` / (``dataset_ref`` + ``reference``) is given.
    flow_name = Parameter(
        "flow_name",
        default="",
        type=str,
        help="Flow name whose recent run health to read (e.g. ValidateFlow).",
    )

    #: Experiment id whose runs + leaderboard to read (the loom_experiment tag).
    experiment = Parameter(
        "experiment",
        default="",
        type=str,
        help="Experiment id whose runs + leaderboard to read.",
    )

    #: Data object pathspec to drift-check (the current data object).
    dataset_ref = Parameter(
        "dataset_ref",
        default="",
        type=str,
        help="Data object pathspec to drift-check against --reference.",
    )

    #: Reference data object pathspec to compare the current one against (drift).
    reference = Parameter(
        "reference",
        default="",
        type=str,
        help="Reference data object pathspec for the drift comparison.",
    )

    @step
    def start(self) -> None:
        """Gather the requested observations via the Client API. READ-ONLY.

        Resolves which monitoring view was requested -- a flow's run health, an
        experiment's leaderboard, and/or a drift comparison of two data objects --
        and gathers the inputs through the Client API only (recent run dicts; the
        two data objects' frames for drift). Nothing is written outside this run.
        """
        import os
        import tempfile

        import pandas as pd

        from loom.dataio import materialize_dataset

        flow_name = (self.flow_name or "").strip() or None
        experiment_id = (self.experiment or "").strip() or None
        dataset_ref = (self.dataset_ref or "").strip() or None
        reference = (self.reference or "").strip() or None

        self._flow_name = flow_name
        self._experiment_id = experiment_id
        self._dataset_ref = dataset_ref
        self._reference = reference

        # Run-health / leaderboard gather (flow or experiment).
        if experiment_id:
            self._runs = gather_experiment_runs(experiment_id)
        elif flow_name:
            self._runs = gather_flow_runs(flow_name)
        else:
            self._runs = []

        # Drift gather: materialize both data objects' train frames via the Client
        # API. Only when BOTH a current and a reference ref were given.
        self._cur_df = None
        self._ref_df = None
        if dataset_ref and reference:
            workspace = tempfile.mkdtemp(prefix="loom-ops-")
            cur_dir = os.path.join(workspace, "current")
            ref_dir = os.path.join(workspace, "reference")
            os.makedirs(cur_dir, exist_ok=True)
            os.makedirs(ref_dir, exist_ok=True)
            materialize_dataset(dataset_ref, cur_dir)
            materialize_dataset(reference, ref_dir)
            self._cur_df = pd.read_csv(os.path.join(cur_dir, "train.csv"))
            self._ref_df = pd.read_csv(os.path.join(ref_dir, "train.csv"))
            self.workspace_dir = workspace

        self.next(self.observe)

    @card
    @step
    def observe(self) -> None:
        """Summarize run health + (when requested) drift, and render the ``@card``.

        Delegates the run-health rollup to the pure :func:`summarize_ops` and the
        drift comparison to the pure :func:`compute_drift`, so both are
        unit-testable without Metaflow, then renders a Markdown + Tables ``@card``
        (run health, leaderboard, drift table) and stores the typed summary on
        ``self.summary``.
        """
        health = summarize_ops(
            self._runs,
            flow_name=self._flow_name,
            experiment_id=self._experiment_id,
        )

        drift = None
        if self._cur_df is not None and self._ref_df is not None:
            drift = compute_drift(self._cur_df, self._ref_df)

        # Top-level status: a DEGRADED run health or detected DRIFT degrades the
        # overall ops status; otherwise it follows the run-health status.
        statuses = [health.get("status")]
        if drift is not None:
            statuses.append(drift.get("status"))
        if "DEGRADED" in statuses or "DRIFT" in statuses:
            overall = "ATTENTION"
        elif health.get("status") == "EMPTY" and drift is None:
            overall = "EMPTY"
        else:
            overall = "OK"

        self.summary = {
            "health": health,
            "drift": drift,
            "dataset_ref": self._dataset_ref,
            "reference": self._reference,
            "status": overall,
        }
        self._render_card(self.summary)
        self.next(self.end)

    def _render_card(self, summary: dict) -> None:
        """Render a Markdown + Tables ``@card`` (run health, leaderboard, drift)."""
        from metaflow.cards import Markdown, Table

        health = summary.get("health") or {}
        drift = summary.get("drift")

        current.card.append(Markdown("# Loom ops / monitoring"))
        current.card.append(
            Markdown(
                f"**flow:** `{health.get('flow_name')}`  \n"
                f"**experiment:** `{health.get('experiment_id')}`  \n"
                f"**runs:** {health.get('n_runs')} "
                f"({health.get('n_successful')} ok, {health.get('n_failed')} failed)  \n"
                f"**success rate:** "
                f"{health.get('success_rate') if health.get('success_rate') is not None else 'n/a'}  \n"
                f"**run health:** {health.get('status')}  \n"
                f"**VERDICT:** **{summary.get('status')}**"
            )
        )

        # Recent runs (most recent first).
        recent = health.get("recent") or []
        if recent:
            current.card.append(Markdown("## Recent runs"))
            current.card.append(
                Table(
                    [
                        [
                            r.get("pathspec", "?"),
                            r.get("flow", ""),
                            "ok" if r.get("successful") else "FAILED",
                            f"{r['metric']:.6g}"
                            if isinstance(r.get("metric"), (int, float))
                            else "n/a",
                            r.get("finished_at") or r.get("created_at") or "",
                        ]
                        for r in recent
                    ],
                    headers=["run", "flow", "status", "metric", "finished"],
                )
            )

        # Leaderboard (scored runs best-first).
        leaderboard = health.get("leaderboard") or []
        if leaderboard:
            current.card.append(Markdown("## Leaderboard"))
            current.card.append(
                Table(
                    [
                        [
                            i,
                            r.get("pathspec", "?"),
                            f"{r['metric']:.6g}"
                            if isinstance(r.get("metric"), (int, float))
                            else "n/a",
                        ]
                        for i, r in enumerate(leaderboard, start=1)
                    ],
                    headers=["#", "run", "metric"],
                )
            )

        # Drift table.
        if drift is not None:
            current.card.append(
                Markdown(f"## Drift vs. reference — **{drift.get('status')}**")
            )
            current.card.append(
                Markdown(
                    f"shared columns: {drift.get('n_shared_columns')}  \n"
                    f"added: {', '.join(drift.get('added') or []) or '—'}  \n"
                    f"removed: {', '.join(drift.get('removed') or []) or '—'}"
                )
            )
            flags = drift.get("drift_flags") or []
            if flags:
                current.card.append(
                    Table(
                        [
                            [f["column"], f["kind"], f["detail"]]
                            for f in flags
                        ],
                        headers=["column", "kind", "detail"],
                    )
                )
            else:
                current.card.append(Markdown("_No per-column drift smells detected._"))

    @step
    def end(self) -> None:
        """Carry ``self.summary`` forward so ``Run.data.summary`` exposes it.

        Metaflow persists step artifacts, so ``self.summary`` (set in ``observe``)
        is already on ``Run.data``; the MLOps interface reads it back for the
        command's summary. Nothing else to do.
        """
        pass


if __name__ == "__main__":
    OpsFlow()
