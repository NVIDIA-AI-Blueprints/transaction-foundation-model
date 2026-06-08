"""AIDE search provider (``"aide"``) -- the default "brain".

This adapter implements the :class:`~loom.providers.SearchProvider` port by
*driving* AIDE's agent loop. It does **not** edit or fork AIDE; it imports the
real ``aide`` package (pinned by SHA ``40dcf28`` in ``pyproject.toml``) and
calls its public surface exactly as ``aide.run`` / ``aide.Experiment`` do:

* build an OmegaConf config via ``aide.utils.config._load_cfg(use_cli_args=False)``,
  patch ``data_dir`` / ``goal`` / ``eval`` and the agent/search/model knobs from
  the :class:`~loom.types.Task` and budget, then ``prep_cfg`` it;
* ``load_task_desc(cfg)`` and ``prep_agent_workspace(cfg)`` to materialize the
  ``./input`` + ``./working`` workspace;
* construct ``Journal()`` and ``Agent(task_desc, cfg, journal)`` and run
  ``agent.step(exec_callback=...)`` for ``steps`` iterations, ``save_run`` after
  each step, converting the newest journal node into a
  :class:`~loom.types.NodeRecord` and emitting it via ``on_node``;
* return the best node as a :class:`~loom.types.SearchResult`.

The execution seam: AIDE's ``ExecCallbackType`` is
``Callable[[str, bool], aide.interpreter.ExecutionResult]``. A Loom
:class:`~loom.providers.ExecutionProvider` returns a
:class:`loom.types.ExecutionResult` (field-identical to AIDE's), so this module
wraps the provided ``execute`` in a tiny bridge that converts the Loom result
into AIDE's type with ``aide.interpreter.ExecutionResult(**asdict(r))``.

Model routing: ``cfg.agent.code.model`` / ``cfg.agent.feedback.model`` are set
from :class:`~loom.config.LoomConfig`. AIDE's own backend (see
``aide/backend/__init__.py``) selects the provider from the model name and from
``OPENAI_BASE_URL`` (e.g. an NVIDIA NIM endpoint), reading the API key from the
environment. This module never reads or stores key material.

AIDE is an *optional* dependency: this module only imports it inside methods, so
``import loom.providers`` (and the registration of ``aide``) succeeds even when
AIDE is not installed; the import error only surfaces when a run is attempted.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Optional

from loom.config import BudgetConfig, LoomConfig
from loom.providers import ExecCallback, OnNode, SearchProvider
from loom.registry import register_search
from loom.types import ExecutionResult, NodeRecord, SearchResult, Task


@register_search("aide")
class AideSearchProvider(SearchProvider):
    """Drive AIDE's tree-search agent as a Loom search provider.

    Instantiated by the controller as ``AideSearchProvider(config)``. Each
    :meth:`run` builds a fresh AIDE config/journal/agent, runs the budgeted
    number of steps against the supplied exec callback, mirrors finished nodes
    into the corpus via ``on_node``, and returns the best solution found.

    Attributes:
        name: Registry name, ``"aide"``.
        config: The active :class:`~loom.config.LoomConfig` (model routing,
            budget defaults, tenant/ownership tags).
    """

    name = "aide"

    def __init__(self, config: LoomConfig) -> None:
        """Initialize the provider from a Loom config.

        Args:
            config: The active configuration. Model names and budget defaults
                are read; secret material is never read here (AIDE's backend
                consumes keys/endpoints from the environment).
        """
        self.config = config

    # ------------------------------------------------------------------ run --

    def run(
        self,
        task: Task,
        execute: ExecCallback,
        on_node: Optional[OnNode] = None,
        budget: Optional[object] = None,
    ) -> SearchResult:
        """Run AIDE's agent loop over ``task`` and return the best solution.

        Args:
            task: The task to solve (``data_dir`` / ``goal`` / ``eval`` feed the
                AIDE config; ``experiment_id`` / ``tenant`` tag corpus records).
            execute: The exec callback (typically the chosen
                :class:`~loom.providers.ExecutionProvider`). Wrapped so AIDE
                receives its own ``ExecutionResult`` type.
            on_node: Optional sink invoked with a :class:`NodeRecord` for each
                node as it finishes (e.g. ``Corpus.record``).
            budget: Optional :class:`~loom.config.BudgetConfig` (falls back to
                ``self.config.budget``).

        Returns:
            A :class:`SearchResult` describing the best node AIDE produced.

        Raises:
            ImportError: If the ``aide`` package is not installed.
        """
        # Import AIDE lazily so this module imports without AIDE present.
        import aide  # noqa: F401  (ensures the package is importable)
        from aide.agent import Agent
        from aide.journal import Journal, Node
        from aide.utils.config import (
            _load_cfg,
            load_task_desc,
            prep_agent_workspace,
            prep_cfg,
            save_run,
        )

        bud = self._resolve_budget(budget)

        # 1) Build + patch + prep the AIDE config (no CLI args).
        cfg = self._build_cfg(_load_cfg, prep_cfg, task, bud)

        # 2) Materialize task description + workspace (./input + ./working).
        task_desc = load_task_desc(cfg)
        prep_agent_workspace(cfg)

        # 3) Journal + Agent, and the loom<->aide exec bridge.
        journal = Journal()
        agent = Agent(task_desc=task_desc, cfg=cfg, journal=journal)
        bridge = self._make_bridge(execute)

        # 4) Drive the loop: step, persist, then mirror the newest node out.
        steps = int(getattr(cfg.agent, "steps", bud.steps))
        emitted = 0
        for _ in range(steps):
            agent.step(exec_callback=bridge)
            save_run(cfg, journal)

            # The agent appends exactly one node per step; mirror any newly
            # finished node(s) (defensively handle >1 in case of future changes).
            while emitted < len(journal):
                node: Node = journal[emitted]
                emitted += 1
                if on_node is not None:
                    on_node(self._node_to_record(node, task, cfg))

        # 5) Best node -> SearchResult.
        best = journal.get_best_node(only_good=False)
        best_code = best.code if best is not None else None
        best_metric = self._metric_value(best)

        log_dir = self._as_str(getattr(cfg, "log_dir", None))
        journal_path = self._join(log_dir, "journal.json")
        tree_path = self._join(log_dir, "tree_plot.html")

        return SearchResult(
            best_code=best_code,
            best_metric=best_metric,
            journal_path=journal_path,
            tree_path=tree_path,
            node_count=len(journal),
        )

    # -------------------------------------------------------------- helpers --

    def _resolve_budget(self, budget: Optional[object]) -> BudgetConfig:
        """Return the effective :class:`BudgetConfig` for this run.

        Args:
            budget: A caller-supplied budget (used if it is a
                :class:`BudgetConfig`), otherwise the config's budget.

        Returns:
            The budget to apply.
        """
        if isinstance(budget, BudgetConfig):
            return budget
        return self.config.budget

    def _build_cfg(
        self,
        load_cfg_fn: Any,
        prep_cfg_fn: Any,
        task: Task,
        budget: BudgetConfig,
    ) -> Any:
        """Build and prepare the AIDE OmegaConf config for ``task``.

        Patches the data/goal/eval fields and the agent/search/model knobs from
        the task, budget, and :class:`LoomConfig`, then runs AIDE's ``prep_cfg``
        (which resolves paths, names the experiment, and validates the schema).

        Args:
            load_cfg_fn: AIDE's ``_load_cfg`` callable.
            prep_cfg_fn: AIDE's ``prep_cfg`` callable.
            task: The task whose fields seed the config.
            budget: The effective search budget.

        Returns:
            The prepared AIDE ``Config`` (an OmegaConf object).
        """
        cfg = load_cfg_fn(use_cli_args=False)

        # Task-driven fields.
        cfg.data_dir = task.data_dir
        cfg.goal = task.goal
        cfg.eval = task.eval

        # Search budget.
        cfg.agent.steps = budget.steps
        cfg.agent.search.num_drafts = budget.num_drafts
        cfg.agent.search.debug_prob = budget.debug_prob
        cfg.agent.search.max_debug_depth = budget.max_debug_depth

        # Model routing (names only; AIDE's backend resolves provider + reads
        # keys/endpoints from the environment, e.g. OPENAI_BASE_URL for NIM).
        cfg.agent.code.model = self.config.code_model
        cfg.agent.feedback.model = self.config.feedback_model

        return prep_cfg_fn(cfg)

    @staticmethod
    def _make_bridge(execute: ExecCallback) -> Any:
        """Wrap a Loom exec callback so AIDE receives its own ExecutionResult.

        AIDE calls ``exec_callback(code, reset_session)`` and expects an
        ``aide.interpreter.ExecutionResult``. The Loom ``execute`` returns a
        field-identical :class:`loom.types.ExecutionResult`, so we convert it
        with a straight field-for-field copy.

        Args:
            execute: The Loom exec callback (an ExecutionProvider or any
                ``(code, reset_session) -> loom ExecutionResult`` callable).

        Returns:
            A callable matching AIDE's ``ExecCallbackType``.
        """
        from aide.interpreter import ExecutionResult as AideExecutionResult

        def bridge(code: str, reset_session: bool = True) -> Any:
            result: ExecutionResult = execute(code, reset_session)
            # Field-identical dataclasses: convert by name. asdict deep-copies
            # the nested lists/dicts, which is fine for a one-way handoff.
            return AideExecutionResult(**dataclasses.asdict(result))

        return bridge

    def _node_to_record(self, node: Any, task: Task, cfg: Any) -> NodeRecord:
        """Convert an AIDE journal :class:`Node` into a :class:`NodeRecord`.

        Pulls the node's code, raw terminal output, exception type, metric, and
        judge summary, and tags the record with the experiment/tenant/owner from
        the task and config. No secret material is touched.

        Args:
            node: The AIDE journal node to convert.
            task: The task being solved (supplies ``experiment_id`` / ``tenant``).
            cfg: The prepared AIDE config (supplies the code model name).

        Returns:
            A populated :class:`NodeRecord` ready to append to the corpus.
        """
        parent = getattr(node, "parent", None)
        parent_id = getattr(parent, "id", None) if parent is not None else None

        # ``_term_out`` is the raw list[str]; ``term_out`` is a trimmed joined
        # string. The corpus stores the raw lines.
        term_out = getattr(node, "_term_out", None) or []

        model = None
        try:
            model = cfg.agent.code.model
        except Exception:  # noqa: BLE001 - config shape is advisory here
            model = self.config.code_model

        return NodeRecord(
            experiment_id=task.experiment_id,
            node_id=str(getattr(node, "id", "")),
            parent_id=str(parent_id) if parent_id is not None else None,
            stage=self._stage_name(node),
            code=getattr(node, "code", "") or "",
            term_out=list(term_out),
            exc_type=getattr(node, "exc_type", None),
            metric=self._metric_value(node),
            judge_summary=getattr(node, "analysis", None),
            model=model,
            tokens=None,
            tenant=task.tenant,
            owned_by=self.config.owned_by,
            ts=time.time(),
        )

    @staticmethod
    def _stage_name(node: Any) -> str:
        """Return the node's search stage (``draft`` / ``debug`` / ``improve``).

        Uses AIDE's ``Node.stage_name`` property when available, falling back to
        inferring from the parent so the record is always populated.

        Args:
            node: The AIDE journal node.

        Returns:
            The stage label.
        """
        try:
            return str(node.stage_name)
        except Exception:  # noqa: BLE001 - fall back to a parent-based guess
            parent = getattr(node, "parent", None)
            if parent is None:
                return "draft"
            return "debug" if getattr(parent, "is_buggy", False) else "improve"

    @staticmethod
    def _metric_value(node: Any) -> Optional[float]:
        """Extract a node's numeric validation metric, if any.

        AIDE stores the metric as a ``MetricValue`` whose ``.value`` is ``None``
        for a buggy/worst node. Returns the float value or ``None``.

        Args:
            node: The AIDE journal node (or ``None``).

        Returns:
            The metric as a float, or ``None`` when unavailable.
        """
        if node is None:
            return None
        metric = getattr(node, "metric", None)
        if metric is None:
            return None
        value = getattr(metric, "value", None)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_str(value: Any) -> Optional[str]:
        """Return ``str(value)`` or ``None`` if ``value`` is ``None``."""
        return None if value is None else str(value)

    @staticmethod
    def _join(directory: Optional[str], name: str) -> Optional[str]:
        """Join ``name`` onto ``directory`` (or ``None`` if no directory).

        Args:
            directory: The base directory string, or ``None``.
            name: The filename to append.

        Returns:
            The joined path, or ``None`` if ``directory`` is ``None``.
        """
        if directory is None:
            return None
        import os

        return os.path.join(directory, name)


__all__ = ["AideSearchProvider"]
