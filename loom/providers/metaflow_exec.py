"""Metaflow execution provider -- the default MLOps "muscle".

:class:`MetaflowExecutionProvider` is the default :class:`~loom.providers.ExecutionProvider`
(registry name ``"metaflow"``). Each :meth:`execute` call runs the single static
:class:`flows.eval_candidate.EvalCandidate` flow as a subprocess via
``metaflow.Runner``, passing the candidate solution as **data** (an
``IncludeFile``) and the task context as ``Parameter``s. After the run finishes
it reconstructs a :class:`loom.types.ExecutionResult` from ``Run.data`` so the
provider is field-identical to AIDE's interpreter output and can be used directly
as an AIDE exec callback (via the callable ``__call__`` seam on the base class).

Design points enforced here:

* **ONE static flow, candidates are data.** We never generate a flow per
  candidate; we write the candidate to a temp file and hand it to the static
  flow as an ``IncludeFile`` value.
* **Input is a Metaflow data object.** The task's input is an ingested Metaflow
  **artifact** referenced by ``task.dataset_ref`` (a pathspec). The flow reads it
  only through the Metaflow Client API (``loom.dataio``); this provider never
  touches the underlying datastore (local or S3/minio) directly. Run ``loom
  ingest`` to produce a ``dataset_ref``.
* **BYO endpoint / perimeter.** The Metaflow ``profile`` comes from
  ``config.metaflow_profile`` (env ``METAFLOW_PROFILE``), so a tenant points
  Loom at *their* Metaflow metadata service and datastore and their data never
  leaves their perimeter.
* **Lazy imports.** ``metaflow`` is imported *inside* methods, so this module
  (and therefore ``import loom.providers``) loads even when Metaflow is not
  installed -- the guarded import in ``loom/providers/__init__.py`` relies on
  this not raising at import time.
* **No secrets on the object.** Only the (non-secret) profile name is read from
  config; all key material / endpoints come from the environment that the
  Runner subprocess inherits.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Optional

from loom.config import LoomConfig
from loom.providers import ExecutionProvider
from loom.registry import register_execution
from loom.types import ExecutionResult, RunResult, Task

#: Flow name as seen by the Metaflow Client API (matches the FlowSpec subclass
#: name in ``flows/eval_candidate.py``).
_FLOW_NAME = "EvalCandidate"

#: Default per-candidate execution timeout (seconds) passed to the flow. Loom's
#: ``BudgetConfig`` budgets *search steps*, not per-run wall-clock, so the
#: per-candidate ceiling is a provider constant (the flow can be re-run with a
#: different value by overriding this if a tenant needs longer-running cells).
_CANDIDATE_TIMEOUT_S = 3600


@register_execution("metaflow")
class MetaflowExecutionProvider(ExecutionProvider):
    """Run candidate code through the static ``EvalCandidate`` Metaflow flow.

    The provider is constructed from a :class:`~loom.config.LoomConfig` (the
    uniform provider constructor the controller uses). It holds no task state
    until :meth:`setup` is called and never stores secret material.

    Attributes:
        name: Registry name (``"metaflow"``).
        config: The active Loom configuration (profile, tenant, ownership).
    """

    name = "metaflow"

    def __init__(self, config: LoomConfig) -> None:
        """Initialise the provider from configuration.

        Args:
            config: Active Loom configuration. The Metaflow profile is taken
                from ``config.metaflow_profile`` (``None`` -> Metaflow default
                profile); tenant/ownership tag the run.
        """
        self.config = config
        self._task: Optional[Task] = None
        self._dataset_ref: Optional[str] = None
        #: Tags applied to every run for Client-API leaderboard filtering.
        self._tags: list[str] = []

    # -- lifecycle ---------------------------------------------------------

    def setup(self, task: Task) -> None:
        """Record the task's dataset reference and prepare run tags.

        The Metaflow provider's input is a **Metaflow data object** referenced by
        ``task.dataset_ref`` (a pathspec like ``"IngestDataset/123"`` produced by
        ``loom ingest``). The flow's ``start`` step materializes it into
        ``./input`` through the Metaflow Client API (``loom.dataio``); here we
        only carry the reference and compute the tags used to group/rank this
        experiment's runs. We do not resolve any local path or touch a datastore
        -- the datastore (local or S3/minio) is Metaflow's concern.

        Args:
            task: The task to evaluate candidates for. ``task.dataset_ref`` is
                required for this provider; a missing reference is surfaced as a
                clear error result by :meth:`execute` (instructing the user to
                run ``loom ingest`` first).
        """
        self._task = task
        self._dataset_ref = (task.dataset_ref or "").strip() if task.dataset_ref else ""
        self._tags = [
            f"loom_experiment:{task.experiment_id}",
            f"loom_tenant:{task.tenant}",
            f"loom_owned_by:{self.config.owned_by}",
        ]

    def teardown(self) -> None:
        """Release per-task state. Workspace dirs are owned by the flow runs."""
        self._task = None
        self._dataset_ref = None
        self._tags = []

    # -- execution ---------------------------------------------------------

    def execute(self, code: str, reset_session: bool = True) -> ExecutionResult:
        """Evaluate ``code`` by running the static flow and read back the result.

        Writes ``code`` to a temporary file, runs
        :class:`flows.eval_candidate.EvalCandidate` via ``metaflow.Runner``
        (blocking until completion) with the candidate as an ``IncludeFile`` and
        the task context as parameters, then reconstructs a
        :class:`~loom.types.ExecutionResult` from the run's ``end`` data.

        Args:
            code: Candidate Python source to evaluate.
            reset_session: Accepted for AIDE exec-callback parity. Each flow run
                is already a fresh, process-isolated workspace, so this provider
                is stateless across calls and the flag has no effect.

        Returns:
            The execution result reconstructed from ``Run.data``. If the flow
            cannot be started or produced no usable data, a degraded
            ``ExecutionResult`` with ``exc_type="MetaflowRunError"`` is returned
            so the search loop keeps making progress rather than crashing.
        """
        # Lazy import: keeps the module importable without Metaflow installed.
        from metaflow import Runner

        from flows import EVAL_CANDIDATE_FLOW_PATH

        task = self._task
        goal = task.goal if task else ""
        eval_desc = task.eval if task else ""
        dataset_ref = self._dataset_ref or ""

        # The Metaflow provider's input IS the Metaflow data object. Without a
        # dataset_ref there is nothing to evaluate against, so fail with an
        # actionable message rather than running the flow against empty ./input.
        if not dataset_ref:
            return self._error_result(
                "no dataset_ref set for the metaflow provider: ingest your data "
                "first with `loom ingest --source <path>` and pass the printed "
                "pathspec via `loom run --dataset <pathspec> --mlops metaflow`."
            )

        # Write the candidate to a temp file: it enters the static flow as DATA
        # (IncludeFile), never as a generated flow.
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="loom-candidate-",
            delete=False,
            encoding="utf-8",
        )
        try:
            tmp.write(code)
            tmp.flush()
            tmp.close()
            candidate_path = tmp.name

            runner_kwargs: dict[str, Any] = {}
            # Profile from config (BYO endpoint); None -> Metaflow default.
            if self.config.metaflow_profile:
                runner_kwargs["profile"] = self.config.metaflow_profile

            try:
                with Runner(
                    EVAL_CANDIDATE_FLOW_PATH,
                    show_output=False,
                    **runner_kwargs,
                ) as runner:
                    executing = runner.run(
                        candidate_code=candidate_path,
                        goal=goal,
                        eval=eval_desc,
                        timeout=_CANDIDATE_TIMEOUT_S,
                        dataset_ref=dataset_ref,
                        tags=list(self._tags),
                    )
                    run = executing.run
            except Exception as exc:  # pragma: no cover - infra failure path
                return self._error_result(f"failed to run EvalCandidate flow: {exc}")

            return self._result_from_run(run)
        finally:
            # Best-effort cleanup of the candidate temp file.
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    # -- lifecycle flows ---------------------------------------------------

    def run_flow(
        self,
        flow_path: str,
        parameters: dict,
        tags: Optional[list[str]] = None,
    ) -> RunResult:
        """Run a Loom lifecycle flow via ``metaflow.Runner`` and read back a RunResult.

        Mirrors :meth:`execute` (the candidate path) but for a whole lifecycle
        command's static ``FlowSpec``: we *write nothing*, run the named flow with
        the given ``parameters`` and ``tags`` under ``config.metaflow_profile``,
        block until it finishes, then reconstruct a :class:`~loom.types.RunResult`
        from the run -- its pathspec, success, the located ``@card`` reference, a
        small summary dict of the run's output artifacts, and an ``error`` string
        on failure. The data the flow reads is a Metaflow data object referenced
        by a ``dataset_ref`` parameter (read via the Client API in the flow);
        this provider never touches the datastore directly.

        Args:
            flow_path: Filesystem path to the static flow file (e.g.
                ``flows.EDA_FLOW_PATH``).
            parameters: Flow ``Parameter`` values to pass (e.g.
                ``{"dataset_ref": "IngestDataset/1", "target": "label"}``). ``None``
                values are dropped so the flow's own defaults apply.
            tags: Optional run tags for Client-API filtering/leaderboards.

        Returns:
            The :class:`~loom.types.RunResult` for the produced run. If the flow
            cannot be started, a degraded result with ``successful=False`` and an
            actionable ``error`` is returned rather than raising, so a calling
            command surfaces the failure cleanly.
        """
        # Lazy import: keeps the module importable without Metaflow installed.
        from metaflow import Runner

        runner_kwargs: dict[str, Any] = {}
        # Profile from config (BYO endpoint); None -> Metaflow default.
        if self.config.metaflow_profile:
            runner_kwargs["profile"] = self.config.metaflow_profile

        # Drop None-valued parameters so the flow's declared defaults apply, and
        # pass tags through under Metaflow's run() ``tags`` kwarg.
        run_kwargs: dict[str, Any] = {
            k: v for k, v in (parameters or {}).items() if v is not None
        }
        if tags:
            run_kwargs["tags"] = list(tags)

        try:
            with Runner(
                flow_path,
                show_output=False,
                **runner_kwargs,
            ) as runner:
                executing = runner.run(**run_kwargs)
                run = executing.run
        except Exception as exc:  # pragma: no cover - infra failure path
            return self._run_error_result(
                f"failed to run flow {flow_path!r}: {type(exc).__name__}: {exc}"
            )

        return self._run_result_from_run(run)

    # -- leaderboard -------------------------------------------------------

    def runs(self, experiment_id: str) -> list[dict]:
        """Return a ranked leaderboard of past runs for ``experiment_id``.

        Uses the Metaflow Client API to find runs tagged for this experiment and
        builds one dict per run with its outcome. Runs that produced a usable
        submission and completed without an exception rank highest; ties break
        on faster ``exec_time``.

        Args:
            experiment_id: The experiment to read runs for.

        Returns:
            A list of run dicts (best first). Empty if Metaflow is unavailable
            or no matching runs exist.
        """
        try:
            from metaflow import Flow, get_metadata, metadata  # noqa: F401
        except Exception:  # pragma: no cover - Metaflow not installed
            return []

        # Point the Client API at the configured profile's metadata service.
        # (Runner uses a profile env; the Client reads METAFLOW_PROFILE, which
        # the provider's environment already carries when a profile is set.)
        if self.config.metaflow_profile:
            os.environ.setdefault("METAFLOW_PROFILE", self.config.metaflow_profile)

        tag = f"loom_experiment:{experiment_id}"
        leaderboard: list[dict] = []
        try:
            flow = Flow(_FLOW_NAME)
            matching = flow.runs(tag)
        except Exception:  # pragma: no cover - no such flow yet / metadata down
            return []

        for run in matching:
            try:
                ok = bool(run.successful)
                data = run.data if ok else None
                exc_type = getattr(data, "exc_type", None) if data else "RunFailed"
                exec_time = float(getattr(data, "exec_time", 0.0) or 0.0) if data else 0.0
                submission_ok = bool(getattr(data, "submission_ok", False)) if data else False
                leaderboard.append(
                    {
                        "run_id": run.pathspec,
                        # ``pathspec`` is the canonical Metaflow identifier; we
                        # alias it alongside ``run_id`` so the CLI leaderboard
                        # renderer (which prefers a ``pathspec``/``run_id`` for
                        # the Metaflow row shape) finds it under either key.
                        "pathspec": run.pathspec,
                        "experiment_id": experiment_id,
                        "successful": ok,
                        "submission_ok": submission_ok,
                        "exc_type": exc_type,
                        "exec_time": exec_time,
                        "created_at": str(getattr(run, "created_at", "")),
                        "finished_at": str(getattr(run, "finished_at", "")),
                    }
                )
            except Exception:  # pragma: no cover - skip unreadable runs
                continue

        # Rank: usable submission & no exception first, then faster exec_time.
        leaderboard.sort(
            key=lambda r: (
                not (r["submission_ok"] and r["exc_type"] is None),
                r["exec_time"],
            )
        )
        return leaderboard

    # -- internals ---------------------------------------------------------

    def _result_from_run(self, run: Any) -> ExecutionResult:
        """Reconstruct an :class:`ExecutionResult` from a finished ``Run``.

        Reads the five execution fields the flow's ``end`` step carries via
        ``Run.data``. Missing fields are filled with neutral defaults so the
        returned object always satisfies the contract.

        Args:
            run: A ``metaflow.Run`` (or ``ExecutingRun.run``) that has finished.

        Returns:
            The reconstructed execution result.
        """
        try:
            data = run.data
        except Exception as exc:  # pragma: no cover - datastore read failure
            return self._error_result(f"could not read Run.data: {exc}")

        if data is None:
            return self._error_result("EvalCandidate produced no end data")

        term_out = list(getattr(data, "term_out", []) or [])
        exec_time = float(getattr(data, "exec_time", 0.0) or 0.0)
        exc_type = getattr(data, "exc_type", None)
        exc_info = getattr(data, "exc_info", None)
        exc_stack = getattr(data, "exc_stack", None)

        return ExecutionResult(
            term_out=term_out,
            exec_time=exec_time,
            exc_type=exc_type,
            exc_info=exc_info,
            exc_stack=exc_stack,
        )

    def _run_result_from_run(self, run: Any) -> RunResult:
        """Reconstruct a :class:`RunResult` from a finished lifecycle-flow ``Run``.

        Reads the run's ``successful`` flag and ``pathspec``, locates the
        ``@card`` rendered by the flow (best-effort, via the Client API), and
        builds a small JSON-able summary of the run's output artifacts from
        ``Run.data`` -- preferring a ``profile``/``summary`` artifact when the
        flow carries one. Bulk output stays in Metaflow; only a small summary is
        inlined.

        Args:
            run: A ``metaflow.Run`` (or ``ExecutingRun.run``) that has finished.

        Returns:
            The reconstructed :class:`RunResult`.
        """
        pathspec = getattr(run, "pathspec", None)
        try:
            successful = bool(run.successful)
        except Exception:  # pragma: no cover - metadata read edge case
            successful = False

        if not successful:
            return RunResult(
                pathspec=pathspec,
                successful=False,
                card_path=None,
                summary={},
                error=f"flow run {pathspec or '<unknown>'} did not complete successfully",
            )

        try:
            data = run.data
        except Exception as exc:  # pragma: no cover - datastore read failure
            return RunResult(
                pathspec=pathspec,
                successful=True,
                card_path=None,
                summary={},
                error=f"could not read Run.data: {exc}",
            )

        summary = self._build_summary(data)
        card_path = self._locate_card_path(run)

        return RunResult(
            pathspec=pathspec,
            successful=True,
            card_path=card_path,
            summary=summary,
            error=None,
        )

    @staticmethod
    def _build_summary(data: Any) -> dict:
        """Build a small, JSON-able summary dict from a finished run's ``data``.

        Prefers a flow-provided ``profile`` (the EDA flow's profile dict) or a
        ``summary`` artifact; otherwise returns an empty dict. The summary is the
        small derived context a command narrates -- bulk artifacts stay in
        Metaflow and are referenced by pathspec, never inlined here.

        Args:
            data: The run's ``Run.data`` artifact proxy (or ``None``).

        Returns:
            A JSON-able dict (empty when the run carries no summary artifact).
        """
        if data is None:  # pragma: no cover - a successful run has data
            return {}
        for name in ("profile", "summary"):
            value = getattr(data, name, None)
            if isinstance(value, dict):
                return dict(value)
        return {}

    @staticmethod
    def _locate_card_path(run: Any) -> Optional[str]:
        """Locate the ``@card`` rendered by the run, via the Client API.

        Walks the run's steps for the first task carrying a ``@card`` and returns
        that card's datastore ``path`` (the canonical card reference). Best-effort:
        any failure (no card, Metaflow card plugin unavailable) yields ``None`` so
        a missing card never fails the command.

        Args:
            run: A finished ``metaflow.Run``.

        Returns:
            The first located card's ``path``, or ``None``.
        """
        try:
            from metaflow.cards import get_cards
        except Exception:  # pragma: no cover - card plugin unavailable
            return None

        try:
            steps = list(run)
        except Exception:  # pragma: no cover - metadata read edge case
            return None

        for step in steps:
            try:
                task = step.task
            except Exception:  # pragma: no cover - empty/foreach step
                continue
            if task is None:
                continue
            try:
                cards = get_cards(task)
            except Exception:  # pragma: no cover - no cards for this task
                continue
            for card in cards:
                path = getattr(card, "path", None)
                if path:
                    return str(path)
        return None

    @staticmethod
    def _run_error_result(message: str) -> RunResult:
        """Build a degraded :class:`RunResult` describing a flow-run failure.

        Used when a lifecycle flow cannot be started, so a calling command sees a
        clean failed result (with an actionable ``error``) rather than an
        exception bubbling out of the MLOps interface. Mirrors
        :meth:`_error_result` for the candidate path.

        Args:
            message: Human-readable description of what went wrong.

        Returns:
            A :class:`RunResult` with ``successful=False`` and ``error`` set.
        """
        return RunResult(
            pathspec=None,
            successful=False,
            card_path=None,
            summary={},
            error=message,
        )

    @staticmethod
    def _error_result(message: str) -> ExecutionResult:
        """Build a degraded :class:`ExecutionResult` describing an infra error.

        Used when the flow cannot be started or its data cannot be read, so the
        search loop sees a normal (if failed) execution rather than an
        exception bubbling out of the exec callback.

        Args:
            message: Human-readable description of what went wrong.

        Returns:
            An execution result with ``exc_type="MetaflowRunError"``.
        """
        return ExecutionResult(
            term_out=[message],
            exec_time=0.0,
            exc_type="MetaflowRunError",
            exc_info={"msg": message},
            exc_stack=None,
        )


__all__ = ["MetaflowExecutionProvider"]
