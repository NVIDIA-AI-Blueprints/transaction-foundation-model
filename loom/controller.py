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
5. tears the execution provider down (always, even on failure) and returns the
   :class:`~loom.types.SearchResult`.

The controller never imports a concrete adapter directly -- that indirection is
exactly what makes Loom pluggable. It is also dependency-light: it imports only
stdlib + Loom core, so the heavy optional dependencies (AIDE, Metaflow, ...) are
only pulled in when a provider that needs them is actually resolved.
"""

from __future__ import annotations

from loom.config import LoomConfig
from loom.corpus import Corpus
from loom.providers import ExecutionProvider, SearchProvider
from loom.registry import get_execution, get_search
from loom.types import SearchResult, Task


def run_loom(task: Task, config: LoomConfig) -> SearchResult:
    """Run a single task end-to-end through the configured providers.

    Resolves the execution and search providers named by ``config`` from the
    registry, instantiates them (along with the corpus), stages the workspace,
    runs the search loop with the execution provider as the exec callback and
    ``Corpus.record`` as the per-node sink, and returns the best solution.

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

    return result


__all__ = ["run_loom"]
