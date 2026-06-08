"""Provider interfaces (the two Loom seams) and built-in adapter registration.

Loom is ports-and-adapters: this package defines the abstract *ports* that the
controller drives, while concrete *adapters* (AIDE, Metaflow, local) live in
sibling modules and register themselves with :mod:`loom.registry`.

Two provider kinds in v0.1:

* :class:`SearchProvider` -- the "brain" that proposes/scores candidate code.
  Default adapter name ``"aide"``.
* :class:`ExecutionProvider` -- the MLOps "muscle" that runs candidate code and
  returns an :class:`loom.types.ExecutionResult`. Default adapter name
  ``"metaflow"``; ``"local"`` is a Metaflow-free dev path.

The crucial seam: an :class:`ExecutionProvider` is itself *callable* with the
AIDE ``ExecCallbackType`` signature ``(code, reset_session) -> ExecutionResult``
(``__call__`` is aliased to :meth:`ExecutionProvider.execute`), so any execution
provider can be handed straight to a search provider as its exec callback.

This module imports only stdlib + ``loom`` core at the top. The built-in
adapters are imported at the *bottom*, each inside its own ``try/except`` so a
missing optional dependency (AIDE, Metaflow, ...) cannot break ``import
loom.providers`` or, transitively, ``loom`` core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from loom.types import ExecutionResult, NodeRecord, SearchResult, Task

# The exec-callback seam, matching AIDE's ``ExecCallbackType``
# (aide/agent.py): ``Callable[[str, bool], ExecutionResult]`` i.e.
# ``(code, reset_session) -> ExecutionResult``.
ExecCallback = Callable[[str, bool], ExecutionResult]

# Type of the per-node sink a search provider calls as each node finishes.
OnNode = Callable[[NodeRecord], None]


class ExecutionProvider(ABC):
    """Port: runs candidate code and returns an :class:`ExecutionResult`.

    The MLOps "muscle". Implementations stage a workspace (``./input`` populated
    from the task data, an empty ``./working``, and an appropriate current
    working directory), execute code, and optionally expose a leaderboard.

    An execution provider is *callable*: ``provider(code, reset_session)`` is an
    alias for :meth:`execute`, so it satisfies AIDE's ``ExecCallbackType`` and
    can be passed directly to a :class:`SearchProvider`.

    Attributes:
        name: The registry name of this provider (e.g. ``"metaflow"``).
    """

    name: str = "execution"

    @abstractmethod
    def execute(self, code: str, reset_session: bool = True) -> ExecutionResult:
        """Execute ``code`` and return its :class:`ExecutionResult`.

        Args:
            code: The Python source to run inside the prepared workspace.
            reset_session: Whether to start a fresh execution session before
                running (AIDE semantics: ``True`` for an isolated run).

        Returns:
            The execution result, field-identical to AIDE's ExecutionResult.
        """
        raise NotImplementedError

    def __call__(self, code: str, reset_session: bool = True) -> ExecutionResult:
        """Exec-callback seam: an ExecutionProvider *is* an ``ExecCallback``.

        Delegates to :meth:`execute`, so the provider satisfies AIDE's
        ``ExecCallbackType`` signature ``(code, reset_session) -> ExecutionResult``
        and can be passed directly to a :class:`SearchProvider`.
        """
        return self.execute(code, reset_session)

    def setup(self, task: Task) -> None:
        """Prepare the workspace for ``task``.

        Implementations populate ``./input`` from ``task.data_dir``, create an
        empty ``./working`` directory, and set the current working directory
        appropriately. The default is a no-op.

        Args:
            task: The task whose data/workspace to stage.
        """
        return None

    def teardown(self) -> None:
        """Release any resources acquired in :meth:`setup`. Default no-op."""
        return None

    def runs(self, experiment_id: str) -> list[dict]:
        """Return a leaderboard of runs for ``experiment_id``.

        Args:
            experiment_id: The experiment to read runs for.

        Returns:
            A list of run dicts (ranked by the provider). Default ``[]``.
        """
        return []


class SearchProvider(ABC):
    """Port: proposes, executes, scores, and records candidate solutions.

    The "brain". A search provider runs its own loop -- propose code, execute it
    via the supplied :class:`ExecCallback`, score the result, and record each
    finished node through ``on_node`` -- ultimately returning the best solution.

    Attributes:
        name: The registry name of this provider (e.g. ``"aide"``).
    """

    name: str = "search"

    @abstractmethod
    def run(
        self,
        task: Task,
        execute: ExecCallback,
        on_node: Optional[OnNode] = None,
        budget: Optional[object] = None,
    ) -> SearchResult:
        """Run the search loop for ``task`` and return the best solution.

        Args:
            task: The task to solve.
            execute: The exec callback (typically the chosen
                :class:`ExecutionProvider`) used to run proposed code.
            on_node: Optional sink invoked with a :class:`NodeRecord` as each
                node finishes (e.g. ``Corpus.record``).
            budget: Provider-specific budget object (e.g. a
                :class:`loom.config.BudgetConfig`) constraining the search.

        Returns:
            The :class:`SearchResult` describing the best solution found.
        """
        raise NotImplementedError


__all__ = [
    "ExecCallback",
    "OnNode",
    "ExecutionProvider",
    "SearchProvider",
]


# ---------------------------------------------------------------------------
# Built-in adapter registration.
#
# Each import is guarded independently: a missing optional dependency for one
# adapter must NOT prevent the others (or core) from importing. Importing each
# module triggers its ``@register_*`` decorator side effects.
# ---------------------------------------------------------------------------

try:  # local execution provider ("local") -- dev path, no Metaflow.
    from . import local_exec  # noqa: F401
except Exception:  # pragma: no cover - optional dependency guard
    pass

try:  # AIDE search provider ("aide").
    from . import aide_search  # noqa: F401
except Exception:  # pragma: no cover - optional dependency guard
    pass

try:  # Metaflow execution provider ("metaflow").
    from . import metaflow_exec  # noqa: F401
except Exception:  # pragma: no cover - optional dependency guard
    pass

try:  # Model providers (the LLM-backend port); each adapter self-registers.
    from . import model  # noqa: F401
except Exception:  # pragma: no cover - optional dependency guard
    pass
