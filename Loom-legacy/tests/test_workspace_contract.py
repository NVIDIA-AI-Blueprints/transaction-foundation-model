"""Gating test: a known-good solution must produce ``./working/submission.csv``.

This is the second contract gate. The Metaflow evaluation flow
(``flows/eval_candidate.py``) reproduces a workspace (``./input`` populated, an
empty ``./working``, the candidate written to ``solution.py``), executes the
candidate, then *validates* that ``./working/submission.csv`` exists. That
workspace shape mirrors AIDE's (``aide/utils/config.py`` ->
``prep_agent_workspace``: ``input/`` + ``working/`` under the workspace dir), so
both the local and Metaflow execution paths agree on where a submission lands.

Running the full Metaflow ``Runner`` subprocess inside the test suite is
impractical (it shells out, needs a metadata/datastore backend, etc.), so per
the build contract we exercise a *lighter* but faithful check of the same
workspace-prep + validate logic:

* prepare the exact ``input/`` + ``working/`` workspace the flow uses;
* run a known-good candidate through the dependency-light vendored interpreter
  (the same ``run_code`` the flow's ``evaluate`` step calls) when it is
  available, else execute the candidate directly in the workspace; and
* assert the validate-step invariant: ``working/submission.csv`` exists and has
  a sane header.

``metaflow`` itself is imported via :func:`pytest.importorskip` only for the
checks that genuinely need it (that the *one static flow* exists and exposes the
candidate-as-data parameters), so the workspace/validate logic still runs
without Metaflow installed.
"""

from __future__ import annotations

import csv
import os
import runpy
from pathlib import Path

import pytest

# A known-good candidate solution: writes the required submission into ./working
# (relative to the workspace cwd) and prints a marker. This is the shape the
# flow's evaluate step runs and the validate step then checks.
GOOD_SOLUTION = (
    "import os\n"
    "import csv\n"
    "print('candidate running')\n"
    "os.makedirs('working', exist_ok=True)\n"
    "with open(os.path.join('working', 'submission.csv'), 'w', newline='') as f:\n"
    "    w = csv.writer(f)\n"
    "    w.writerow(['id', 'prediction'])\n"
    "    w.writerow([0, 1])\n"
    "    w.writerow([1, 0])\n"
)


@pytest.fixture
def workspace(tmp_path) -> Path:
    """Reproduce the flow's workspace: populated ``input/`` + empty ``working/``.

    Mirrors ``flows.eval_candidate.EvalCandidate.start`` (and AIDE's
    ``prep_agent_workspace``): ``input/`` carries the task data, ``working/`` is
    created empty for the candidate to write its submission into.
    """
    ws = tmp_path / "ws"
    (ws / "input").mkdir(parents=True)
    (ws / "working").mkdir(parents=True)
    # A token input file, so input/ is non-empty exactly like a staged data_dir.
    (ws / "input" / "train.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    return ws


@pytest.fixture
def restore_cwd():
    """Restore the process working directory after the test chdirs."""
    original = os.getcwd()
    try:
        yield
    finally:
        os.chdir(original)


def _validate_submission(working_dir: Path) -> bool:
    """Re-implement the flow's ``validate`` invariant: submission.csv is present.

    The flow's validate step asserts ``./working/submission.csv`` exists and has
    a basic schema. We assert the same here and additionally sanity-check the
    header, returning the boolean the flow would store as ``self.submission_ok``.
    """
    submission = working_dir / "submission.csv"
    if not submission.is_file():
        return False
    with open(submission, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    # Non-empty with a header row -> the flow considers it a valid submission.
    return len(rows) >= 1 and len(rows[0]) >= 1


def test_good_solution_writes_submission_via_interpreter(workspace, restore_cwd) -> None:
    """The vendored interpreter runs the candidate and produces a submission.

    Uses ``loom.providers._interpreter.run_code`` -- the *same* function the
    Metaflow flow's evaluate step and the local provider both call -- so neither
    path hard-depends on AIDE internals. If that module is not present yet (a
    separate build slice), fall back to executing the candidate directly so this
    gate still asserts the workspace/validate contract.
    """
    working = workspace / "working"

    run_code = None
    try:
        from loom.providers._interpreter import run_code  # type: ignore
    except Exception:
        run_code = None

    if run_code is not None:
        result = run_code(GOOD_SOLUTION, str(workspace), 60)
        # Field-identical to AIDE's ExecutionResult: a clean run has no exc.
        assert result.exc_type is None, result.term_out
        assert any("candidate running" in line for line in result.term_out)
    else:  # pragma: no cover - exercised only before the interpreter slice lands
        os.chdir(workspace)
        exec(compile(GOOD_SOLUTION, "solution.py", "exec"), {"__name__": "__main__"})

    # The validate-step invariant: ./working/submission.csv exists with a header.
    assert _validate_submission(working) is True


def test_validate_fails_without_submission(workspace) -> None:
    """A solution that writes nothing fails the validate-step invariant.

    The negative case: an empty ``working/`` means no submission, so the flow's
    ``submission_ok`` would be ``False``.
    """
    assert _validate_submission(workspace / "working") is False


def test_validate_logic_against_a_handwritten_submission(workspace) -> None:
    """The validate helper accepts a well-formed submission written by hand.

    Decouples the validate invariant from any code-execution path so the gate
    still pins the schema check even where execution is unavailable.
    """
    working = workspace / "working"
    with open(working / "submission.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "prediction"])
        w.writerow([0, 1])
    assert _validate_submission(working) is True


def test_static_flow_exists_and_takes_candidate_as_data() -> None:
    """There is ONE static flow and the candidate enters it as *data*.

    Loads the static ``EvalCandidate`` flow file and asserts the contract shape:
    a single ``FlowSpec`` subclass with the canonical ``start -> evaluate ->
    validate -> end`` steps, whose candidate arrives via an ``IncludeFile``
    parameter (never by generating a new flow per candidate).

    Needs Metaflow to import the flow module (it imports ``metaflow`` at top
    level), so it is skipped when Metaflow is absent. Skipped too if the flow
    file has not been written yet (a separate build slice).
    """
    pytest.importorskip("metaflow")

    try:
        from flows import EVAL_CANDIDATE_FLOW_PATH
    except Exception:
        pytest.skip("flows package not importable yet")

    if not os.path.isfile(EVAL_CANDIDATE_FLOW_PATH):
        pytest.skip("eval_candidate flow not written yet")

    from metaflow import FlowSpec, IncludeFile

    # Import the flow file by path without running it. Metaflow flow files guard
    # their CLI entrypoint under ``if __name__ == '__main__'``; running under a
    # different module name keeps that from executing.
    namespace = runpy.run_path(EVAL_CANDIDATE_FLOW_PATH, run_name="loom_flow_under_test")

    flow_classes = [
        obj
        for obj in namespace.values()
        if isinstance(obj, type)
        and issubclass(obj, FlowSpec)
        and obj is not FlowSpec
    ]
    # Exactly one static flow class.
    assert len(flow_classes) == 1, [c.__name__ for c in flow_classes]
    flow_cls = flow_classes[0]
    assert flow_cls.__name__ == "EvalCandidate"

    # The canonical step set is present as methods on the flow.
    for step_name in ("start", "evaluate", "validate", "end"):
        assert callable(getattr(flow_cls, step_name, None)), step_name

    # The candidate enters as data: an IncludeFile parameter named candidate_code.
    candidate_params = [
        name
        for name, val in vars(flow_cls).items()
        if isinstance(val, IncludeFile)
    ]
    assert "candidate_code" in candidate_params, candidate_params


def test_flow_runs_end_to_end_via_runner() -> None:
    """Optional full-flow smoke test, skipped unless explicitly enabled.

    Running the real Metaflow ``Runner`` shells out to a subprocess and needs a
    configured metadata/datastore backend, so it is impractical in ordinary CI
    and is gated behind ``LOOM_RUN_METAFLOW_FLOW=1``. When enabled it drives the
    one static flow with the known-good candidate as data and asserts the run
    reports ``submission_ok`` true.
    """
    if os.environ.get("LOOM_RUN_METAFLOW_FLOW") != "1":
        pytest.skip("set LOOM_RUN_METAFLOW_FLOW=1 to run the full Metaflow flow")

    pytest.importorskip("metaflow")
    from flows import EVAL_CANDIDATE_FLOW_PATH

    if not os.path.isfile(EVAL_CANDIDATE_FLOW_PATH):
        pytest.skip("eval_candidate flow not written yet")

    from metaflow import Runner

    with Runner(EVAL_CANDIDATE_FLOW_PATH).run() as running:  # type: ignore[call-arg]
        assert running.status == "successful", running.status
        assert running.run.data.submission_ok is True
