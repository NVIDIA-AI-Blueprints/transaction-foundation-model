"""Core data types for Loom.

These types are the *locked contract* every other Loom module depends on, so
this module is deliberately dependency-light: it imports only from the Python
standard library and must remain importable in any environment.

The most important invariant is :class:`ExecutionResult`, which is **field
identical** to AIDE's ``aide.interpreter.ExecutionResult``. Field parity is what
lets any Loom :class:`~loom.providers.ExecutionProvider` be used directly as an
AIDE ``exec_callback`` (after a trivial type-shim conversion in the AIDE
adapter). The contract reference is::

    /tmp/ds-research/aideml/aide/interpreter.py  (lines 26-37)

If AIDE's ExecutionResult ever changes shape, this dataclass MUST be updated to
match it verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExecutionResult:
    """Result of executing a code snippet.

    Field-identical to ``aide.interpreter.ExecutionResult`` so that a Loom
    execution provider's output can be converted into AIDE's type with a
    straight field-for-field copy (``aide.interpreter.ExecutionResult(
    **dataclasses.asdict(result))``) and vice versa.

    Attributes:
        term_out: Captured stdout/stderr lines from the run, in order. The
            interpreter appends a final ``"Execution time: ..."`` (or
            ``"TimeoutError: ..."``) line.
        exec_time: Wall-clock execution time in seconds.
        exc_type: Class name of the raised exception, or ``None`` if the code
            ran to completion without raising. A timeout is reported as the
            string ``"TimeoutError"``.
        exc_info: Mapping of selected exception attributes (``args``, ``name``,
            ``msg``, ``obj``) when available; ``None`` otherwise.
        exc_stack: Extracted traceback frames as ``(filename, lineno, name,
            line)`` tuples; ``None`` if there was no exception.
    """

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None


@dataclass
class Task:
    """A single experiment to run through Loom.

    Attributes:
        data_dir: Path to the directory holding the task's input data. An
            execution provider stages this into the workspace's ``./input``.
        goal: Natural-language description of what the solution should achieve.
        eval: Natural-language description of how a solution is evaluated
            (the validation metric the search provider should optimize).
        experiment_id: Stable identifier used to group runs/leaderboard entries
            and to key corpus records.
        tenant: Logical tenant the task belongs to (multi-tenant boundary).
            Defaults to ``"default"``.
    """

    data_dir: str
    goal: str
    eval: str
    experiment_id: str
    tenant: str = "default"


@dataclass
class SearchResult:
    """Outcome of a search provider's run over a task.

    Attributes:
        best_code: Source code of the best solution found, or ``None`` if no
            viable solution was produced.
        best_metric: Validation metric of the best solution, or ``None``.
        journal_path: Filesystem path to the persisted search journal/log.
        tree_path: Filesystem path to the persisted search tree (e.g. an HTML
            visualization), if any.
        node_count: Total number of search nodes explored.
    """

    best_code: str | None
    best_metric: float | None
    journal_path: str | None = None
    tree_path: str | None = None
    node_count: int = 0


@dataclass
class NodeRecord:
    """One finished search node, persisted by the corpus.

    Emitted by a search provider via the ``on_node`` callback as each node
    finishes, and appended (as JSONL) by :class:`loom.corpus.Corpus`. The
    ``owned_by`` field is the IP boundary: records owned by a specific tenant
    are tagged and excluded from the cross-tenant "general" corpus.

    Attributes:
        experiment_id: Experiment this node belongs to.
        node_id: Identifier of this node within the search tree.
        parent_id: Identifier of the parent node, or ``None`` for a draft/root.
        stage: Search stage that produced the node (e.g. ``"draft"``,
            ``"improve"``, ``"debug"``).
        code: Source code evaluated at this node.
        term_out: Captured terminal output lines from executing the code.
        exc_type: Exception class name if the run raised, else ``None``.
        metric: Validation metric reported for this node, or ``None``.
        judge_summary: Short natural-language review/summary of the run.
        model: Identifier of the model that generated the code.
        tokens: Token count attributed to generating this node.
        tenant: Tenant the node belongs to.
        owned_by: IP owner of the record; ``"general"`` means it may be used by
            a cross-tenant moat model, any other value tags it as tenant-owned.
        ts: Epoch timestamp (seconds) when the record was created.
    """

    experiment_id: str
    node_id: str
    parent_id: str | None
    stage: str
    code: str
    term_out: list[str]
    exc_type: str | None
    metric: float | None
    judge_summary: str | None
    model: str | None
    tokens: int | None
    tenant: str = "default"
    owned_by: str = "general"
    ts: float = field(default=0.0)


__all__ = [
    "ExecutionResult",
    "Task",
    "SearchResult",
    "NodeRecord",
]
