"""The Loom controller: wires registry-resolved providers and runs a task.

This module is the single orchestration seam between configuration and the two
ports. Given a :class:`~loom.types.Task` and a :class:`~loom.config.LoomConfig`
it:

1. resolves the search ("brain") and execution ("muscle") provider *classes*
   from :mod:`loom.registry` by their configured names;
2. instantiates each provider from the config, plus a
   :class:`~loom.corpus.Corpus`;
3. lets the execution provider stage its workspace (``setup``);
4. hands the execution provider to the search provider as the exec callback and
   the corpus's ``record`` as the per-node sink, then runs the search;
5. tears the execution provider down (always, even on failure);
6. appends ONE command-level :class:`~loom.learnings.LearningRecord` to the
   learnings flywheel (the moat fuel, shaped as a SkillOpt rollout) and returns
   the :class:`~loom.types.SearchResult`.

The two captures are complementary, not redundant: the corpus records one row
per *search node* (the substrate), while the learnings flywheel records one row
per *run* (the command-level rollup the skill-optimizer consumes).

The controller never imports a concrete adapter directly -- that indirection is
exactly what makes Loom pluggable. It is also dependency-light: it imports only
stdlib + Loom core, so the heavy optional dependencies (AIDE, Metaflow, ...) are
only pulled in when a provider that needs them is actually resolved.
"""

from __future__ import annotations

import dataclasses
import time

from loom.config import LoomConfig
from loom.corpus import Corpus
from loom.learnings import LearningRecord, Learnings, Outcome, TaskSpec
from loom.providers import ExecutionProvider, SearchProvider
from loom.registry import get_execution, get_search
from loom.telemetry import end_trajectory, start_trajectory
from loom.types import SearchResult, Task


def run_loom(task: Task, config: LoomConfig) -> SearchResult:
    """Run a single task end-to-end through the configured providers.

    Resolves the execution and search providers named by ``config`` from the
    registry, instantiates them (along with the corpus), stages the workspace,
    runs the search loop with the execution provider as the exec callback and
    ``Corpus.record`` as the per-node sink, appends one command-level
    :class:`~loom.learnings.LearningRecord` to the flywheel, and returns the best
    solution.

    The execution provider's :meth:`~loom.providers.ExecutionProvider.teardown`
    is invoked in a ``finally`` block so workspace resources are released even if
    the search raises.

    Args:
        task: The task to solve. Its ``tenant`` is reconciled with the config's
            tenant below so corpus records are tagged consistently.
        config: The active configuration selecting providers, models, budget,
            corpus path, and tenant/ownership. No secret material is read from
            it; adapters consume keys/endpoints directly from the environment.

    Returns:
        The :class:`~loom.types.SearchResult` describing the best solution found.
    """
    # Resolve provider *classes* by their configured names. The registry raises
    # an informative KeyError (listing available names) if a name is unknown.
    exec_cls = get_execution(config.mlops_provider)
    search_cls = get_search(config.search_provider)

    # Instantiate from config (the uniform provider constructor signature) plus
    # the corpus. The corpus's ``record`` method is the per-node sink.
    execution: ExecutionProvider = exec_cls(config)
    search: SearchProvider = search_cls(config)
    corpus = Corpus(config)
    learnings = Learnings(config)

    # Open the telemetry trajectory (the CC interaction-root span analogue) at the
    # top of the request->response cycle, PINNED to the task's experiment_id so the
    # command-level rollout JOINs cleanly in :func:`assemble_trajectory`. This is a
    # no-op (still returning the id) when LOOM_TELEMETRY is off, so it is always
    # safe to call. The id is stamped onto the rollout below to correlate the three
    # scattered signals (events + proxy_calls + the rollout).
    trajectory_id = start_trajectory(
        config.search_provider,
        {
            "goal": task.goal,
            "metric": task.eval,
            "data_ref": task.dataset_ref or task.data_dir or None,
            "experiment_id": task.experiment_id,
        },
        config,
        trajectory_id=task.experiment_id,
    )
    started_ts = time.time()

    # Stage the workspace (./input populated, empty ./working, cwd set), run the
    # search with the execution provider as the exec callback, then always tear
    # down so workspace resources are released even on error.
    execution.setup(task)
    try:
        result = search.run(
            task,
            execute=execution,
            on_node=corpus.record,
            budget=config.budget,
        )
    finally:
        execution.teardown()

    # Command-level rollup: append ONE LearningRecord per run (the moat fuel,
    # shaped as a SkillOpt rollout input). This is in addition to -- not a
    # replacement for -- the per-node corpus capture wired above. Built purely
    # from the SearchResult + task + config; no secret material is read.
    learnings.record(_learning_from(result, task, config, trajectory_id))

    # Close the trajectory (the CC endInteractionSpan analogue): emit
    # trajectory.end + trajectory.duration_ms with the run's outcome. A no-op when
    # telemetry is off.
    end_trajectory(
        trajectory_id,
        {
            "metric": result.best_metric,
            "success": result.best_code is not None,
        },
        config,
        started_ts=started_ts,
    )

    return result


def _learning_from(
    result: SearchResult,
    task: Task,
    config: LoomConfig,
    trajectory_id: str | None = None,
) -> LearningRecord:
    """Roll up one run into a command-level :class:`LearningRecord`.

    Projects the :class:`~loom.types.SearchResult`, the :class:`~loom.types.Task`
    spec, and the :class:`~loom.config.LoomConfig` knobs into a single SkillOpt
    rollout row. ``submission_ok`` / ``success`` are true exactly when a best
    solution was produced; artifact *references* (journal/tree paths) are
    recorded but never inlined bytes.

    Args:
        result: The search outcome to summarize.
        task: The task the run was executed against.
        config: The active configuration (provider/budget/ownership knobs).
        trajectory_id: The telemetry trajectory join key (the experiment id) so the
            rollout correlates with its telemetry events + proxy LLM calls. Purely
            additive / defaulted; ``None`` leaves the field absent for non-telemetry
            callers.

    Returns:
        The command-level rollout record to append to the flywheel.
    """
    submission_ok = result.best_code is not None
    artifacts = [p for p in (result.journal_path, result.tree_path) if p]

    return LearningRecord(
        command=f"loom-{config.search_provider}",
        task=TaskSpec(
            data_ref=task.dataset_ref or task.data_dir or None,
            goal=task.goal,
            metric=task.eval,
            experiment_id=task.experiment_id,
        ),
        inputs={
            "search_provider": config.search_provider,
            "mlops_provider": config.mlops_provider,
            "budget": dataclasses.asdict(config.budget),
        },
        outcome=Outcome(
            best_metric=result.best_metric,
            submission_ok=submission_ok,
            node_count=result.node_count,
        ),
        artifacts=artifacts,
        success=submission_ok,
        model=config.code_model,
        tenant=task.tenant,
        owned_by=config.owned_by,
        trajectory_id=trajectory_id,
    )


__all__ = ["run_loom"]
