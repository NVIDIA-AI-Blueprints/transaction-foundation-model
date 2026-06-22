"""``local`` — the executor port adapter (ARCHITECTURE §2.3, §9).

The cheapest executor: it runs everything **in-process, single box**, with no
GPU, no Metaflow, no subprocess fan-out. It is the intermediate adapter that
exercises the scale-out seam first — ``DataRepresentation.materialize()`` calls
``executor.foreach(...)`` and ``ModelBuilder.launch()`` calls
``executor.submit(...)``, and both flow through the same budget/kill contract
(ARCHITECTURE §2.3). The ``metaflow-gcp`` per-shard GPU fan-out (step 7-8) slots
in behind this identical Protocol with no caller change — the seam test.

Structurally satisfies :class:`loom.ports.Executor` (``runtime_checkable``):
``name``, ``gpu_available()``, ``submit()``, ``foreach()``, ``kill()``.

Scope honesty (ARCHITECTURE §6): this is the in-process ``foreach`` over a
single box. ``gpu_available()`` returns ``False`` — the local executor runs CPU
work only, so any GPU-requiring builder must be refused upstream
(``REFUSED_NO_GPU_TARGET``). ``submit()`` runs a Python callable passed in
``argv[0]`` if one is provided (the local-builder path, where the "argv" is a
thunk, not a ``torchrun`` command line) and honors the :class:`BudgetEnvelope`
``max_steps``/``max_wall_clock_min`` as a SOFT cap, emitting a
``stopped_at_budget`` status when it trips.

This module imports nothing from NeMo/torch/RAPIDS/BigQuery.
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Callable, Literal, Optional

from ..ports import (
    BudgetEnvelope,
    ComputeTarget,
    ProgressEvent,
    register_executor,
)

_JobStatus = Literal[
    "pending", "running", "succeeded", "failed", "killed", "stopped_at_budget"
]

# Process-wide job counter — the local stand-in for a run-id allocator (§9).
_JOB_COUNTER = itertools.count(1)


class LocalJobHandle:
    """A :class:`loom.ports.JobHandle` over an in-process unit of work.

    Because ``submit()`` runs synchronously in-process, a returned handle is
    already terminal (``succeeded`` / ``failed`` / ``stopped_at_budget``) unless
    it was cancelled before it ran. ``cancel()`` flips a flag the runner checks at
    each step (cooperative); a handle whose work already finished is a no-op."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self._status: _JobStatus = "pending"
        self._cancelled = False

    def status(self) -> _JobStatus:
        return self._status

    def cancel(self) -> None:
        self._cancelled = True
        if self._status in ("pending", "running"):
            self._status = "killed"


class LocalExecutor:
    """In-process, single-box executor (no GPU, no subprocess)."""

    name: str = "local"

    def __init__(self) -> None:
        # Live handles by id so ``kill(job_id)`` can target them best-effort.
        self._jobs: dict[str, LocalJobHandle] = {}

    def gpu_available(self) -> bool:
        """``False`` — the local executor runs CPU work in-process. A GPU-requiring
        model-builder is refused upstream with ``REFUSED_NO_GPU_TARGET``; the CPU
        local-builder (PPMI+SVD) runs fine here."""
        return False

    def submit(
        self,
        *,
        argv: list[str],
        image: Optional[str],
        compute: ComputeTarget,
        budget: BudgetEnvelope,
        on_event: Callable[[ProgressEvent], None],
    ) -> LocalJobHandle:
        """Run a unit of work in-process under the budget envelope.

        In this slice ``argv`` carries a single callable thunk (the local-builder's
        training step generator) rather than a ``torchrun`` command line: the
        convention is ``argv == [thunk]`` where ``thunk(emit, should_stop) -> None``
        drives the loop, calling ``emit(ProgressEvent)`` per step and checking
        ``should_stop()`` for the cooperative budget/kill cap. The
        ``metaflow-gcp`` executor will instead spawn ``torchrun argv`` — same
        Protocol, same ``on_event`` wiring, same kill contract.

        Budget enforcement is a SOFT cap: when ``max_steps`` or
        ``max_wall_clock_min`` trips, the runner is asked to stop and the handle
        ends ``stopped_at_budget``; a clean finish ends ``succeeded``.
        """
        handle = LocalJobHandle(job_id=f"local-{next(_JOB_COUNTER)}")
        self._jobs[handle.job_id] = handle

        if not argv:
            handle._status = "failed"
            return handle

        thunk = argv[0]
        if not callable(thunk):
            # A real command line (torchrun …) is the metaflow-gcp executor's job;
            # the local executor only knows how to run an in-process callable.
            handle._status = "failed"
            return handle

        start = time.monotonic()
        budget_tripped = {"hit": False}
        step_counter = itertools.count(0)

        def _wall_clock_min() -> float:
            return (time.monotonic() - start) / 60.0

        def should_stop() -> bool:
            if handle._cancelled:
                return True
            n = next(step_counter)  # peek the *next* step index
            if budget.max_steps is not None and n >= int(budget.max_steps):
                budget_tripped["hit"] = True
                return True
            if (
                budget.max_wall_clock_min is not None
                and _wall_clock_min() >= float(budget.max_wall_clock_min)
            ):
                budget_tripped["hit"] = True
                return True
            return False

        def emit(ev: ProgressEvent) -> None:
            if on_event is not None:
                on_event(ev)

        handle._status = "running"
        try:
            thunk(emit, should_stop)
        except Exception:  # pragma: no cover - the builder surfaces its own failure
            handle._status = "failed"
            self._jobs.pop(handle.job_id, None)
            raise

        if handle._cancelled:
            handle._status = "killed"
        elif budget_tripped["hit"]:
            handle._status = "stopped_at_budget"
        else:
            handle._status = "succeeded"
        self._jobs.pop(handle.job_id, None)
        return handle

    def foreach(
        self, *, fn: Callable[[str], str], shards: list[str], compute: ComputeTarget
    ) -> list[str]:
        """Map ``fn`` over ``shards`` in-process, in order (the Tier-B fan-out
        seam, ARCHITECTURE §6 — one task per shard, never the whole corpus in RAM
        at the executor level). ``metaflow-gcp`` runs the same ``fn`` as a per-shard
        on-demand GPU worker; this executor just calls it locally."""
        return [fn(shard) for shard in shards]

    def kill(self, job_id: str) -> None:
        """Best-effort hard-kill of a live job (the binding-envelope kill, §2.3).

        In-process work is synchronous, so a job is usually already terminal by the
        time ``kill`` could be called from the same thread; this flips any still-live
        handle to ``killed`` for the out-of-thread / future-async case."""
        handle = self._jobs.get(job_id)
        if handle is not None:
            handle.cancel()


# ARCHITECTURE §2.4 / §10 step 3: one-line registration under the registry key.
register_executor(LocalExecutor())

__all__ = ["LocalExecutor", "LocalJobHandle"]
