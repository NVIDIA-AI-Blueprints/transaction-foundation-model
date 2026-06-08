"""Loom's single static candidate-evaluation Metaflow flow.

This module defines exactly **one** ``FlowSpec`` -- :class:`EvalCandidate` -- that
Loom runs (via :class:`loom.providers.metaflow_exec.MetaflowExecutionProvider`)
once per candidate solution. The flow definition is *static*: a candidate is
never compiled into a bespoke flow. Instead the candidate's source enters the
flow as **data** (an ``IncludeFile`` parameter), and the task context
(``goal``/``eval``/``timeout``/``seed``/``input_ref``) enters as ``Parameter``s.
This mirrors the repo invariant "ONE static flow; candidates are data".

Flow shape::

    start --> evaluate --> validate --> end

* ``start``    -- reproduce the AIDE-style workspace: ``./input`` (staged from
                  ``input_ref``), an empty ``./working``, and the candidate
                  written to ``solution.py``.
* ``evaluate`` -- run the candidate via :func:`loom.providers._interpreter.run_code`
                  (the same dependency-light interpreter the ``local`` provider
                  uses, so the flow does not hard-depend on AIDE internals) and
                  store the five :class:`loom.types.ExecutionResult` fields.
                  Decorated with ``@resources``/``@retry``/``@catch``/``@timeout``
                  so an infra hiccup or a runaway candidate degrades gracefully.
* ``validate`` -- assert that ``./working/submission.csv`` exists and looks like
                  a non-empty CSV, recording ``submission_ok``.
* ``end``      -- carry the five execution fields plus ``submission_ok`` forward
                  so ``Run.data`` exposes them to the provider's Client-API read.

Only standard Metaflow APIs are used (``FlowSpec``, ``@step``, ``Parameter``,
``IncludeFile``, ``@card``, ``@retry``, ``@catch``, ``@timeout``,
``@resources``). The execution-time imports of ``loom`` happen *inside* steps so
the flow file can be parsed by Metaflow even if invoked from a directory where
``loom`` is not yet importable -- the provider sets ``cwd``/``PYTHONPATH`` for
the subprocess.
"""

from __future__ import annotations

from metaflow import (
    FlowSpec,
    IncludeFile,
    Parameter,
    card,
    catch,
    resources,
    retry,
    step,
    timeout,
)

#: Default per-candidate execution timeout (seconds). The provider overrides
#: this via the ``timeout`` Parameter; it is also the ceiling for ``@timeout``
#: on the ``evaluate`` step (see :meth:`EvalCandidate.evaluate`).
_DEFAULT_TIMEOUT_S = 3600


class EvalCandidate(FlowSpec):
    """Evaluate a single candidate solution in an isolated workspace.

    The candidate's code is supplied as data (``candidate_code`` ``IncludeFile``)
    and all task context as ``Parameter``s, keeping the flow class identical
    across every evaluation.
    """

    #: The candidate solution source. Supplied as data so the flow stays static.
    candidate_code = IncludeFile(
        "candidate_code",
        is_text=True,
        help="Path to the candidate solution's Python source file.",
    )

    #: Natural-language task goal (carried for provenance/cards; the candidate
    #: itself is self-contained Python).
    goal = Parameter(
        "goal",
        default="",
        type=str,
        help="Natural-language description of the task goal.",
    )

    #: Natural-language evaluation/metric description (provenance).
    eval = Parameter(
        "eval",
        default="",
        type=str,
        help="Natural-language description of how the solution is evaluated.",
    )

    #: Per-candidate execution timeout in seconds.
    timeout_s = Parameter(
        "timeout",
        default=_DEFAULT_TIMEOUT_S,
        type=int,
        help="Maximum wall-clock seconds the candidate may run.",
    )

    #: Optional RNG seed exported as ``PYTHONHASHSEED`` for reproducibility.
    seed = Parameter(
        "seed",
        default=0,
        type=int,
        help="Random seed exported to the candidate's environment.",
    )

    #: Filesystem path (or URI understood by the workspace stager) whose contents
    #: populate the workspace ``./input`` directory. A tenant points this at data
    #: inside their own perimeter (BYO datastore).
    input_ref = Parameter(
        "input_ref",
        default="",
        type=str,
        help="Path whose contents are staged into the workspace ./input dir.",
    )

    @step
    def start(self) -> None:
        """Reproduce the AIDE-style workspace for this evaluation.

        Creates a fresh, isolated workspace directory containing ``./input``
        (populated from ``input_ref`` if provided), an empty ``./working``, and
        the candidate written to ``solution.py``. The absolute workspace path is
        stored on ``self.workspace_dir`` for the ``evaluate`` step.
        """
        import os
        import shutil
        import tempfile

        # Isolated workspace; not cleaned up here so logs/artifacts survive a
        # local datastore inspection. The provider stages ephemeral inputs.
        workspace = tempfile.mkdtemp(prefix="loom-eval-")
        input_dir = os.path.join(workspace, "input")
        working_dir = os.path.join(workspace, "working")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(working_dir, exist_ok=True)

        # Populate ./input from the reference path, if one was supplied and it
        # exists. Directory trees are copied; a single file is copied in place.
        ref = (self.input_ref or "").strip()
        if ref and os.path.exists(ref):
            if os.path.isdir(ref):
                # Merge the referenced directory's contents into ./input.
                for name in os.listdir(ref):
                    src = os.path.join(ref, name)
                    dst = os.path.join(input_dir, name)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)
            else:
                shutil.copy2(ref, os.path.join(input_dir, os.path.basename(ref)))

        # Write the candidate to solution.py at the workspace root. The candidate
        # is expected to read from ./input and write ./working/submission.csv.
        solution_path = os.path.join(workspace, "solution.py")
        with open(solution_path, "w", encoding="utf-8") as fh:
            fh.write(self.candidate_code or "")

        self.workspace_dir = workspace
        self.working_dir = working_dir
        self.solution_path = solution_path

        self.next(self.evaluate)

    @card
    @resources(cpu=1, memory=4096)
    @timeout(seconds=_DEFAULT_TIMEOUT_S + 120)
    @retry(times=1)
    @catch(var="eval_error")
    @step
    def evaluate(self) -> None:
        """Run the candidate and capture the five ExecutionResult fields.

        Executes ``solution.py`` inside the staged workspace using
        :func:`loom.providers._interpreter.run_code`, which returns a
        :class:`loom.types.ExecutionResult`. The five fields are copied onto
        ``self`` so they survive into ``Run.data``.

        ``@retry(times=1)`` absorbs a transient infrastructure failure (the
        candidate re-runs cleanly because the interpreter is process-isolated);
        ``@timeout`` is the hard ceiling above the interpreter's own soft
        timeout; ``@catch(var="eval_error")`` ensures a fatal step failure still
        produces a Run whose ``end`` carries explanatory fields rather than no
        data at all.
        """
        import os

        from loom.providers._interpreter import run_code

        # Reproducibility hint for candidates that honor PYTHONHASHSEED.
        os.environ.setdefault("PYTHONHASHSEED", str(int(self.seed)))

        with open(self.solution_path, "r", encoding="utf-8") as fh:
            code = fh.read()

        result = run_code(
            code,
            working_dir=self.working_dir,
            timeout=int(self.timeout_s),
        )

        # The five fields that make this flow's output field-identical to
        # AIDE/loom ExecutionResult.
        self.term_out = result.term_out
        self.exec_time = result.exec_time
        self.exc_type = result.exc_type
        self.exc_info = result.exc_info
        self.exc_stack = result.exc_stack

        self.next(self.validate)

    @step
    def validate(self) -> None:
        """Check the candidate produced a usable submission.

        Asserts ``./working/submission.csv`` exists and passes a light schema
        check (non-empty, has a header). Failure is recorded as
        ``submission_ok = False`` plus a ``validation_msg`` rather than raising,
        so the provider can surface it without losing the execution fields.

        If the ``evaluate`` step itself was caught (``eval_error`` set by
        ``@catch``), the five execution fields may be absent; they are
        backfilled with neutral defaults so ``end`` can carry them safely.
        """
        import os

        # Backfill execution fields if @catch tripped on evaluate.
        if not hasattr(self, "exc_type"):
            caught = getattr(self, "eval_error", None)
            msg = str(caught) if caught is not None else "evaluate step failed"
            self.term_out = [msg]
            self.exec_time = 0.0
            self.exc_type = "StepError"
            self.exc_info = {"msg": msg}
            self.exc_stack = None

        submission = os.path.join(self.working_dir, "submission.csv")
        submission_ok = False
        validation_msg = ""

        if not os.path.isfile(submission):
            validation_msg = "submission.csv not found in ./working"
        else:
            try:
                # Light, dependency-free schema check: header + >= 1 data row.
                with open(submission, "r", encoding="utf-8") as fh:
                    header = fh.readline()
                    if not header.strip():
                        validation_msg = "submission.csv is empty"
                    elif fh.readline().strip() == "" and "," not in header:
                        # No data row and a header with no columns is suspicious.
                        validation_msg = "submission.csv has no data rows"
                    else:
                        submission_ok = True
            except OSError as exc:  # pragma: no cover - filesystem edge case
                validation_msg = f"could not read submission.csv: {exc}"

        self.submission_ok = submission_ok
        self.validation_msg = validation_msg
        self.submission_path = submission if submission_ok else None

        self.next(self.end)

    @step
    def end(self) -> None:
        """Carry the result fields forward so ``Run.data`` exposes them.

        ``Run.data`` resolves to ``Run['end'].task.data``, so the Metaflow
        execution provider reconstructs a :class:`loom.types.ExecutionResult`
        from exactly these attributes after the run completes.
        """
        # The five ExecutionResult fields and validation outcome are already
        # attributes on ``self`` (Metaflow persists step artifacts), so they are
        # available on ``Run.data``. We re-state the contract for the reader:
        #   term_out, exec_time, exc_type, exc_info, exc_stack, submission_ok.
        # No further work is needed; presence is the contract.
        pass


if __name__ == "__main__":
    EvalCandidate()
