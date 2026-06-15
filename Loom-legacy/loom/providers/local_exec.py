"""Local execution provider (``"local"``) -- the Metaflow-free dev path.

This adapter implements the :class:`~loom.providers.ExecutionProvider` port by
running candidate code in-process via the vendored interpreter
(:func:`loom.providers._interpreter.run_code`). It needs neither Metaflow nor
AIDE installed, which makes it the fast default for development and tests: a
search provider (e.g. ``aide``) can drive a full propose -> execute -> score
loop entirely on the local machine.

Workspace layout matches AIDE's (see ``aide/utils/config.py``): a per-task
directory containing ``./input`` (populated from ``task.data_dir``) and an empty
``./working`` (where the candidate is expected to drop ``submission.csv``). The
provider ``chdir``\\ s into that workspace in :meth:`setup` so candidate code can
use the relative ``./input`` / ``./working`` paths it is prompted to expect, and
restores the original cwd in :meth:`teardown`.

The provider is *callable* (``__call__`` -> :meth:`execute`, inherited from the
port), so it can be handed straight to a search provider as the exec callback.

No secrets are read or written here; ``runs`` reads only the local JSONL corpus.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from loom.config import LoomConfig
from loom.providers import ExecutionProvider
from loom.providers import _interpreter
from loom.registry import register_execution
from loom.types import ExecutionResult, RunResult, Task


@register_execution("local")
class LocalExecutionProvider(ExecutionProvider):
    """Run candidate code locally, in a per-task workspace, via the vendored REPL.

    Instantiated by the controller as ``LocalExecutionProvider(config)``. The
    timeout for each execution is derived from the search-step budget if not
    otherwise constrained; the vendored interpreter default is used as a
    fallback.

    Attributes:
        name: Registry name, ``"local"``.
        config: The active :class:`~loom.config.LoomConfig`.
        timeout: Per-execution wall-clock timeout in seconds.
        workspace_dir: The staged workspace directory (set in :meth:`setup`).
    """

    name = "local"

    def __init__(
        self, config: LoomConfig, timeout: int = _interpreter.DEFAULT_TIMEOUT
    ) -> None:
        """Initialize the provider from a Loom config.

        Args:
            config: The active configuration. Only non-secret fields
                (``corpus_path``, ``tenant``) are read.
            timeout: Per-execution timeout in seconds (defaults to the vendored
                interpreter's default).
        """
        self.config = config
        self.timeout = timeout
        self.workspace_dir: Optional[Path] = None
        self._task: Optional[Task] = None
        self._prev_cwd: Optional[str] = None
        self._owns_workspace = False

    def setup(self, task: Task) -> None:
        """Stage a workspace for ``task`` and chdir into it.

        Creates a fresh temporary workspace directory containing ``./input``
        (a copy of ``task.data_dir`` when it exists, otherwise an empty dir) and
        an empty ``./working``, then changes the process working directory into
        it so candidate code can reference ``./input`` / ``./working`` relative
        paths exactly as AIDE prompts it to.

        Args:
            task: The task whose data directory seeds ``./input``.
        """
        self._task = task
        self._prev_cwd = os.getcwd()

        workspace = Path(tempfile.mkdtemp(prefix=f"loom-local-{task.experiment_id}-"))
        self._owns_workspace = True
        self.workspace_dir = workspace

        input_dir = workspace / "input"
        working_dir = workspace / "working"
        working_dir.mkdir(parents=True, exist_ok=True)

        # Populate ./input from the task data directory. Copy (not symlink) so a
        # candidate cannot accidentally mutate the original dataset, matching
        # AIDE's ``copy_data: True`` default.
        data_dir = Path(task.data_dir) if task.data_dir else None
        if data_dir is not None and data_dir.exists():
            shutil.copytree(data_dir, input_dir, dirs_exist_ok=True)
        else:
            input_dir.mkdir(parents=True, exist_ok=True)

        os.chdir(str(workspace))

    def execute(self, code: str, reset_session: bool = True) -> ExecutionResult:
        """Execute ``code`` in the staged workspace via the vendored interpreter.

        Args:
            code: Python source to run.
            reset_session: Accepted for the :class:`ExecCallback` signature; the
                local provider always runs each snippet in a fresh child process
                (there is no persistent session to reuse), so this is advisory.

        Returns:
            A :class:`loom.types.ExecutionResult` with AIDE's five fields.

        Raises:
            RuntimeError: If called before :meth:`setup` staged a workspace.
        """
        if self.workspace_dir is None:
            raise RuntimeError(
                "LocalExecutionProvider.execute called before setup(); no "
                "workspace has been staged."
            )
        return _interpreter.run_code(
            code,
            working_dir=self.workspace_dir,
            timeout=self.timeout,
        )

    def run_flow(
        self,
        flow_path: str,
        parameters: dict,
        tags: Optional[list[str]] = None,
    ) -> RunResult:
        """Not supported: lifecycle flows need the Metaflow MLOps provider.

        The ``local`` provider is the Metaflow-free dev path for AIDE
        *candidate* execution (:meth:`execute`); it has no Metaflow run store, so
        it cannot produce the **Metaflow run + @card** every lifecycle command
        mandates. Lifecycle commands (EDA, connect, validate, ...) must run on the
        ``metaflow`` MLOps provider.

        Args:
            flow_path: Ignored (the local provider runs no flows).
            parameters: Ignored.
            tags: Ignored.

        Raises:
            NotImplementedError: Always, with guidance to use ``--mlops metaflow``.
        """
        raise NotImplementedError(
            "the 'local' MLOps provider cannot run lifecycle flows (it is the "
            "Metaflow-free AIDE candidate-exec dev path). Lifecycle commands "
            "produce a Metaflow run + @card, so run them on the metaflow MLOps "
            "provider: pass `--mlops metaflow`."
        )

    def teardown(self) -> None:
        """Restore the original cwd and remove the staged workspace."""
        if self._prev_cwd is not None:
            try:
                os.chdir(self._prev_cwd)
            except OSError:
                pass
            self._prev_cwd = None

        if self._owns_workspace and self.workspace_dir is not None:
            shutil.rmtree(self.workspace_dir, ignore_errors=True)
        self.workspace_dir = None
        self._owns_workspace = False

    def runs(self, experiment_id: str) -> list[dict]:
        """Return a leaderboard of recorded nodes for ``experiment_id``.

        The local provider has no external run store, so it reads the Loom JSONL
        corpus (the same one node records are appended to) and returns the
        finished nodes for this experiment ranked best-first. A node "ran
        successfully" when it has no exception and a numeric metric. Because the
        corpus does not record the optimization direction, results are ordered
        by descending metric (a stable, deterministic default); callers that
        know the direction can re-sort.

        Args:
            experiment_id: The experiment to read runs for.

        Returns:
            A list of ranked run dicts, each with ``node_id``, ``parent_id``,
            ``stage``, ``metric``, ``exc_type``, ``judge_summary``, ``model``,
            and ``ts``. Empty if the corpus is missing/empty.
        """
        # Lazy import to keep this module dependency-light and avoid a circular
        # import at package-load time (corpus imports config/types only).
        from loom.corpus import Corpus

        records = Corpus(self.config).for_experiment(experiment_id)

        leaderboard = [
            {
                "node_id": rec.node_id,
                "parent_id": rec.parent_id,
                "stage": rec.stage,
                "metric": rec.metric,
                "exc_type": rec.exc_type,
                "judge_summary": rec.judge_summary,
                "model": rec.model,
                "ts": rec.ts,
            }
            for rec in records
        ]

        # Rank: successful (numeric metric, no exception) first, by descending
        # metric; everything else after, by recency.
        def _sort_key(row: dict) -> tuple:
            ok = row["exc_type"] is None and isinstance(row["metric"], (int, float))
            metric = row["metric"] if isinstance(row["metric"], (int, float)) else float("-inf")
            return (0 if ok else 1, -metric if ok else 0.0, -row["ts"])

        leaderboard.sort(key=_sort_key)
        return leaderboard


__all__ = ["LocalExecutionProvider"]
