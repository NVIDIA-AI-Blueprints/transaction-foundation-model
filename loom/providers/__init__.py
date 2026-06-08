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

from loom.types import (
    ArtifactRef,
    CapabilityManifest,
    ExecutionResult,
    NodeRecord,
    RunResult,
    Scores,
    SearchResult,
    Task,
)

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

    def run_flow(
        self,
        flow_path: str,
        parameters: dict,
        tags: list[str] | None = None,
    ) -> RunResult:
        """Run a Loom **lifecycle flow** through the MLOps interface.

        Where :meth:`execute` runs a single *candidate snippet* (the AIDE search
        "muscle"), ``run_flow`` runs a whole lifecycle command's static
        ``FlowSpec`` (EDA, connect, validate, ...) and returns a
        :class:`~loom.types.RunResult` describing the produced Metaflow run +
        ``@card`` -- the mandated artifact of every ``/loom-*`` command
        (design-spec §3). This is the seam the lifecycle skills speak to so they
        run flows through Loom's *interface* rather than touching Metaflow
        directly; the default MLOps implementation is Metaflow and is swappable.

        Args:
            flow_path: Filesystem path to the static flow file to run (e.g.
                ``flows.EDA_FLOW_PATH``), resolved by the provider's runner.
            parameters: Flow ``Parameter`` values to pass to the run (e.g.
                ``{"dataset_ref": "IngestDataset/1", "target": "label"}``).
            tags: Optional run tags for Client-API filtering/leaderboards (e.g.
                ``["loom_command:eda", "loom_tenant:default"]``).

        Returns:
            The :class:`~loom.types.RunResult` for the produced run.

        Raises:
            NotImplementedError: For providers that do not run lifecycle flows
                (the default). Only an MLOps provider that can run a Metaflow
                ``FlowSpec`` (e.g. ``"metaflow"``) implements this.
        """
        raise NotImplementedError(
            "this MLOps provider does not run lifecycle flows"
        )

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


# ---------------------------------------------------------------------------
# The model-builder port (the third heavy backend, a fourth port).
#
# The abstraction boundary is drawn in Loom DS-intent vocabulary: the adapter
# is a *compiler* that lowers intent -> backend config. A backend noun (Megatron,
# ``.nemo``, ``@resources(gpu=8)``) lives inside the adapter and nowhere else.
# These module-level frozensets are the seam: a NeMo noun is rejected here, in
# Loom vocabulary, before any adapter sees it (constraint 2). Enums stay
# ``str``-typed at the boundary (the repo passes provider names/roles as plain
# strings everywhere) but are validated against these sets at the seam.
# ---------------------------------------------------------------------------

OBJECTIVES = frozenset({"next-event", "masked-field", "contrastive"})
BUDGETS = frozenset({"probe", "small", "full"})
MODES = frozenset({"batch", "online"})


class ModelBuilderProvider(ABC):
    """Port: builds/serves models stated in Loom DS-intent vocabulary.

    The training/serving "model builder" — the third heavy backend, a fourth
    port exactly parallel to :class:`SearchProvider` and
    :class:`ExecutionProvider`. I/O are Metaflow artifact pathspecs (via
    :mod:`loom.dataio` / the Client API), never ``.nemo`` files or raw object
    storage. Each capability declares a ``mode`` in :meth:`manifest` so the
    AIDE-vs-builder split (``/loom-train`` launch-and-track vs ``/loom-optimize``
    searchable) is a *provider fact*, not user knowledge.

    Mirrors the :class:`ExecutionProvider` style: a concrete ``ABC`` with a
    ``name`` class attribute and exactly **one** :func:`abstractmethod`
    (:meth:`manifest`). The six capabilities are plain methods that
    ``raise NotImplementedError`` by default (the ``run_flow`` optional-by-raise
    pattern): a backend overrides only the capabilities it supports and declares
    the rest unsupported in its manifest, so a capability gap is refused up front
    rather than failing deep in a job.

    Instantiated uniformly as ``Provider(config)`` (like the other ports), so the
    flow/CLI can resolve a builder from :class:`~loom.config.LoomConfig`.

    Attributes:
        name: The registry name of this provider (e.g. ``"local"``, ``"nemo"``).
    """

    name: str = "model_builder"

    @abstractmethod
    def manifest(self) -> CapabilityManifest:
        """Return the :class:`~loom.types.CapabilityManifest` for this backend.

        The negotiation contract the up-front refusal reads: which capabilities
        are supported, the AIDE-search mode of each, and the honesty notes for
        stand-in/limited capabilities. The only required method.
        """
        raise NotImplementedError

    def tokenize(self, sequences_ref: str, scheme: dict) -> ArtifactRef:
        """Build a tokenizer/vocab over the sequences. Mode: ``searchable``.

        Args:
            sequences_ref: Pathspec of the sequences data object.
            scheme: The tokenization scheme (fields, min_count, max_vocab,
                n_buckets) AIDE may tree-search.

        Returns:
            An :class:`~loom.types.ArtifactRef` of ``kind="tokenizer"``.
        """
        raise NotImplementedError

    def pretrain(self, sequences_ref: str, objective: str, budget: str) -> ArtifactRef:
        """Pretrain a backbone from the sequences. Mode: ``launch-and-track``.

        Args:
            sequences_ref: Pathspec of the sequences data object.
            objective: A Loom objective validated against :data:`OBJECTIVES`.
            budget: A budget validated against :data:`BUDGETS` (physics —
                surfaced at the gate, never faked away).

        Returns:
            An :class:`~loom.types.ArtifactRef` of ``kind="backbone"``.
        """
        raise NotImplementedError

    def finetune(self, backbone_ref: str, task_ref: str, recipe: dict) -> ArtifactRef:
        """Fit a cheap head on a frozen backbone. Mode: ``searchable``.

        Args:
            backbone_ref: Pathspec of the frozen backbone.
            task_ref: Pathspec of the task data object.
            recipe: The head recipe (estimator/regularization/pooled features)
                AIDE may tree-search.

        Returns:
            An :class:`~loom.types.ArtifactRef` of ``kind="model"``.
        """
        raise NotImplementedError

    def embed(self, backbone_ref: str, data_ref: str) -> ArtifactRef:
        """Embed data through a frozen backbone. Mode: ``searchable``.

        Args:
            backbone_ref: Pathspec of the frozen backbone.
            data_ref: Pathspec of the data object to embed.

        Returns:
            An :class:`~loom.types.ArtifactRef` of ``kind="embeddings"`` whose
            pathspec is an ``IngestDataset``-shaped, first-class ``dataset_ref``.
        """
        raise NotImplementedError

    def evaluate(self, model_ref: str, holdout_ref: str, metric: str) -> Scores:
        """Score a model on a sealed holdout. Mode: ``searchable``.

        Args:
            model_ref: Pathspec of the fitted model.
            holdout_ref: Pathspec of the sealed temporal holdout data object.
            metric: The metric name (e.g. ``"fraud-pr-auc"``).

        Returns:
            A :class:`~loom.types.Scores` carrying the scalar + comparability
            detail (baseline + lift).
        """
        raise NotImplementedError

    def serve(self, model_ref: str, mode: str) -> ArtifactRef:
        """Serve a model. Mode declared per backend in the manifest.

        ``EndpointRef`` collapses to ``ArtifactRef(kind="endpoint")`` in v0.1
        (no live endpoint object until NIM online serving lands).

        Args:
            model_ref: Pathspec of the model to serve.
            mode: A serving mode validated against :data:`MODES`.

        Returns:
            An :class:`~loom.types.ArtifactRef` of ``kind="endpoint"``.
        """
        raise NotImplementedError


__all__ = [
    "ExecCallback",
    "OnNode",
    "ExecutionProvider",
    "SearchProvider",
    "ModelBuilderProvider",
    "OBJECTIVES",
    "BUDGETS",
    "MODES",
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

try:  # Model-builder providers (the training/serving port); each adapter self-registers.
    from . import model_builder  # noqa: F401
except Exception:  # pragma: no cover - optional dependency guard
    pass
