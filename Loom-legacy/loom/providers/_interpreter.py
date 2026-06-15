"""Dependency-light Python interpreter, vendored from AIDE.

This is a faithful, dependency-light port of AIDE's
``aide/interpreter.py`` (reference SHA ``40dcf28``). It executes a code
snippet in an isolated child process, captures stdout/stderr, enforces a
wall-clock timeout, and summarizes any raised exception into the exact shape
AIDE produces -- but it returns a :class:`loom.types.ExecutionResult` (which is
field-identical to ``aide.interpreter.ExecutionResult``) instead of the AIDE
type, and it drops AIDE's third-party conveniences (``humanize`` for the timing
line, ``shutup`` for warning muting, ``dataclasses_json`` on the result).

Why vendor it? Both the ``local`` execution provider and the Metaflow
``evaluate`` step must run candidate code and emit the same five-field result
*without* hard-depending on AIDE internals (AIDE is an optional dependency, only
required by the ``aide`` search provider). Vendoring keeps execution providers
importable and runnable in environments where AIDE is not installed.

Fidelity contract: :func:`exception_summary` is copied field-for-field from
AIDE so that ``exc_info`` / ``exc_stack`` / the filtered traceback and the
appended ``"Execution time:"`` / ``"TimeoutError:"`` terminal line are
byte-compatible with what AIDE's interpreter would have produced. The only
deliberate change is the timing-line wording: AIDE uses ``humanize`` to render a
human-friendly duration; to stay dependency-light we render the duration with a
tiny local helper (:func:`_naturaldelta`) that mirrors ``humanize.naturaldelta``
closely enough for log parity without the dependency.
"""

from __future__ import annotations

import os
import queue
import signal
import sys
import time
import traceback
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional

from loom.types import ExecutionResult

# Default execution timeout (seconds), matching AIDE's interpreter default.
DEFAULT_TIMEOUT = 3600

# Default name of the file the candidate code is written to inside the child
# process, matching AIDE's ``agent_file_name`` default.
DEFAULT_AGENT_FILE_NAME = "runfile.py"

# Marker the child puts on the output queue to signal end of captured output.
_EOF_MARKER = "<|EOF|>"


def _naturaldelta(seconds: float) -> str:
    """Render a duration the way ``humanize.naturaldelta`` would, dependency-free.

    AIDE appends a human-friendly duration to ``term_out`` via
    ``humanize.naturaldelta``. To avoid taking a hard dependency on
    ``humanize`` we approximate it: sub-minute durations are rendered as a whole
    number of seconds (``"a second"`` / ``"N seconds"``), and longer durations
    fall back to minutes/hours. This is only used for the trailing log line, so
    exact wording parity is not load-bearing for correctness.

    Args:
        seconds: Duration in seconds.

    Returns:
        A short human-readable string such as ``"3 seconds"`` or ``"2 minutes"``.
    """
    secs = int(round(seconds))
    if secs <= 1:
        return "a second"
    if secs < 60:
        return f"{secs} seconds"
    minutes = secs // 60
    if minutes < 60:
        return "a minute" if minutes == 1 else f"{minutes} minutes"
    hours = minutes // 60
    return "an hour" if hours == 1 else f"{hours} hours"


def exception_summary(
    e: BaseException,
    working_dir: Path,
    exec_file_name: str,
    format_tb_ipython: bool,
) -> tuple[str, str, dict, list[tuple]]:
    """Summarize an exception and its stack trace.

    Copied field-for-field from ``aide.interpreter.exception_summary`` so the
    ``exc_info`` mapping, the ``exc_stack`` frame tuples, and the filtered
    traceback string match AIDE exactly. Specifically:

    * the traceback is rendered with the standard Python REPL formatting (or
      IPython's ``VerboseTB`` when ``format_tb_ipython`` is set);
    * traceback lines containing ``"aide/"`` or ``"importlib"`` are filtered out
      (kept verbatim from AIDE so traceback text stays identical);
    * the absolute path ``working_dir / exec_file_name`` is replaced with just
      ``exec_file_name`` to strip the workspace directory;
    * ``exc_info`` collects ``args`` plus any of ``name`` / ``msg`` / ``obj``
      attributes the exception exposes;
    * ``exc_stack`` is a list of ``(filename, lineno, name, line)`` tuples.

    Args:
        e: The caught exception instance.
        working_dir: The child process working directory (used to strip the
            absolute path of the executed file from the traceback).
        exec_file_name: The basename the candidate code was written to.
        format_tb_ipython: Whether to format the traceback using IPython's
            ``VerboseTB`` instead of the standard REPL format.

    Returns:
        A ``(tb_str, exc_type_name, exc_info, exc_stack)`` tuple, matching the
        AIDE function's return shape.
    """
    if format_tb_ipython:
        import IPython.core.ultratb

        # tb_offset = 1 to skip parts of the stack trace in weflow code
        tb = IPython.core.ultratb.VerboseTB(tb_offset=1, color_scheme="NoColor")
        tb_str = str(tb.text(*sys.exc_info()))
    else:
        tb_lines = traceback.format_exception(e)
        # skip parts of stack trace in weflow code
        tb_str = "".join(
            [
                line
                for line in tb_lines
                if "aide/" not in line and "importlib" not in line
            ]
        )

    # replace whole path to file with just filename (to remove agent workspace dir)
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class _RedirectQueue:
    """File-like object that forwards writes to a multiprocessing queue.

    Mirrors AIDE's ``RedirectQueue``: the child process rebinds ``sys.stdout``
    and ``sys.stderr`` to an instance of this so all printed output is streamed
    back to the parent via ``result_outq``.
    """

    def __init__(self, q: Queue, timeout: int = 5) -> None:
        """Initialize the redirect.

        Args:
            q: The multiprocessing queue to forward writes to.
            timeout: Seconds to wait when the queue is full before dropping a
                write (a full queue is logged-and-dropped, matching AIDE).
        """
        self.queue = q
        self.timeout = timeout

    def write(self, msg: str) -> None:
        """Forward ``msg`` to the queue, dropping it on a full-queue timeout."""
        try:
            self.queue.put(msg, timeout=self.timeout)
        except queue.Full:
            # Match AIDE behavior: drop the write rather than block forever.
            pass

    def flush(self) -> None:
        """No-op flush (the queue is the sink)."""
        pass


class Interpreter:
    """Standalone Python REPL with a wall-clock execution limit.

    A dependency-light port of ``aide.interpreter.Interpreter``. Each
    :meth:`run` (with ``reset_session=True``) spawns a fresh child process,
    sends it the code, waits for completion or timeout, collects captured
    output, and returns a :class:`loom.types.ExecutionResult`.

    Attributes:
        working_dir: Resolved workspace directory the child ``chdir``\\ s into.
        timeout: Per-execution timeout in seconds (``None`` disables it).
        format_tb_ipython: Whether to format tracebacks via IPython.
        agent_file_name: Basename the candidate code is written to.
    """

    def __init__(
        self,
        working_dir: "Path | str",
        timeout: int = DEFAULT_TIMEOUT,
        format_tb_ipython: bool = False,
        agent_file_name: str = DEFAULT_AGENT_FILE_NAME,
    ) -> None:
        """Initialize the interpreter.

        Args:
            working_dir: Working directory of the run; must already exist.
            timeout: Timeout for each code execution step (seconds).
            format_tb_ipython: Use IPython traceback formatting if ``True``.
            agent_file_name: Name for the candidate's code file.
        """
        # this really needs to be a path, otherwise causes issues that don't raise exc
        self.working_dir = Path(working_dir).resolve()
        assert (
            self.working_dir.exists()
        ), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.agent_file_name = agent_file_name
        self.process: Optional[Process] = None

    def child_proc_setup(self, result_outq: Queue) -> None:
        """Set up the child process: chdir, sys.path, and stdout/stderr capture.

        Args:
            result_outq: Queue that captured stdout/stderr is streamed to.
        """
        os.chdir(str(self.working_dir))

        # This seems to only be necessary because we're exec'ing code from a
        # string; a .py file should be able to import modules from cwd anyway.
        sys.path.append(str(self.working_dir))

        # capture stdout and stderr
        sys.stdout = sys.stderr = _RedirectQueue(result_outq)  # type: ignore[assignment]

    def _run_session(
        self, code_inq: Queue, result_outq: Queue, event_outq: Queue
    ) -> None:
        """Child-process loop: receive code, exec it, report state + output.

        Args:
            code_inq: Queue the parent sends code strings on.
            result_outq: Queue captured output is streamed back on.
            event_outq: Queue lifecycle events (``state:ready`` /
                ``state:finished``) are reported on.
        """
        self.child_proc_setup(result_outq)

        global_scope: dict = {}
        while True:
            code = code_inq.get()
            os.chdir(str(self.working_dir))
            with open(self.agent_file_name, "w") as f:
                f.write(code)

            event_outq.put(("state:ready",))
            try:
                exec(compile(code, self.agent_file_name, "exec"), global_scope)
            except BaseException as e:  # noqa: BLE001 - we must catch everything
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                    e,
                    self.working_dir,
                    self.agent_file_name,
                    self.format_tb_ipython,
                )
                result_outq.put(tb_str)
                if e_cls_name == "KeyboardInterrupt":
                    e_cls_name = "TimeoutError"

                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))

            # remove the file after execution (otherwise it might be included in
            # the data preview)
            try:
                os.remove(self.agent_file_name)
            except OSError:
                pass

            # put EOF marker to indicate that we're done
            result_outq.put(_EOF_MARKER)

    def create_process(self) -> None:
        """Spawn the child process and its three communication queues."""
        # We use three queues to communicate with the child process:
        # - code_inq: send code to child to execute
        # - result_outq: receive stdout/stderr from child
        # - event_outq: receive events from child (state:ready, state:finished)
        self.code_inq, self.result_outq, self.event_outq = Queue(), Queue(), Queue()
        self.process = Process(
            target=self._run_session,
            args=(self.code_inq, self.result_outq, self.event_outq),
        )
        self.process.start()

    def cleanup_session(self) -> None:
        """Terminate and clean up the child process, escalating if needed."""
        if self.process is None:
            return
        try:
            # Reduce grace period from 2 seconds to 0.5
            self.process.terminate()
            self.process.join(timeout=0.5)

            if self.process.exitcode is None:
                self.process.kill()
                self.process.join(timeout=0.5)

                if self.process.exitcode is None and self.process.pid is not None:
                    os.kill(self.process.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001 - cleanup must never raise
            pass
        finally:
            if self.process is not None:
                self.process.close()
                self.process = None

    def run(self, code: str, reset_session: bool = True) -> ExecutionResult:
        """Execute ``code`` in a child process and return its result.

        Faithful to AIDE's ``Interpreter.run``: spawns (or reuses) a child,
        sends the code, waits for ``state:ready`` then ``state:finished``,
        enforces the timeout (escalating SIGINT -> kill), drains the output
        queue up to the EOF marker, and appends the trailing
        ``"Execution time: ..."`` (or ``"TimeoutError: ..."``) line.

        Args:
            code: Python source to execute.
            reset_session: Whether to start a fresh child before running. Must
                be ``True`` on the first call.

        Returns:
            A :class:`loom.types.ExecutionResult` with the five AIDE fields.
        """
        if reset_session:
            if self.process is not None:
                # terminate and clean up previous process
                self.cleanup_session()
            self.create_process()
        else:
            # reset_session needs to be True on first exec
            assert self.process is not None

        assert self.process is not None and self.process.is_alive()

        self.code_inq.put(code)

        # wait for child to actually start execution (don't interrupt setup)
        try:
            state = self.event_outq.get(timeout=10)
        except queue.Empty:
            raise RuntimeError("REPL child process failed to start execution") from None
        assert state[0] == "state:ready", state
        start_time = time.time()

        # this flag indicates the child has exceeded the time limit and an
        # interrupt was sent; if the child dies without it set, it's unexpected.
        child_in_overtime = False
        exec_time = 0.0

        while True:
            try:
                # check if the child is done
                state = self.event_outq.get(timeout=1)  # wait for state:finished
                assert state[0] == "state:finished", state
                exec_time = time.time() - start_time
                break
            except queue.Empty:
                # haven't heard back -> check the child is still alive (assuming
                # the overtime interrupt wasn't sent yet)
                if not child_in_overtime and not self.process.is_alive():
                    raise RuntimeError("REPL child process died unexpectedly") from None

                # child is alive and still executing -> check if we should sigint
                if self.timeout is None:
                    continue
                running_time = time.time() - start_time
                if running_time > self.timeout:
                    os.kill(self.process.pid, signal.SIGINT)  # type: ignore[arg-type]
                    child_in_overtime = True

                    # terminate if we're overtime by more than 5 seconds
                    if running_time > self.timeout + 5:
                        self.cleanup_session()
                        state = (None, "TimeoutError", {}, [])
                        exec_time = float(self.timeout)
                        break

        output: list[str] = []
        # read all stdout/stderr from child up to the EOF marker; waiting until
        # the queue is empty is not enough since the feeder thread in the child
        # might still be adding to the queue.
        start_collect = time.time()
        while not self.result_outq.empty() or not output or output[-1] != _EOF_MARKER:
            try:
                # Add 5-second timeout for output collection
                if time.time() - start_collect > 5:
                    break
                output.append(self.result_outq.get(timeout=1))
            except queue.Empty:
                continue
        if output and output[-1] == _EOF_MARKER:
            output.pop()  # remove the EOF marker

        e_cls_name, exc_info, exc_stack = state[1:]

        if e_cls_name == "TimeoutError":
            output.append(
                f"TimeoutError: Execution exceeded the time limit of "
                f"{_naturaldelta(self.timeout)}"
            )
        else:
            output.append(
                f"Execution time: {_naturaldelta(exec_time)} seconds "
                f"(time limit is {_naturaldelta(self.timeout)})."
            )
        return ExecutionResult(output, exec_time, e_cls_name, exc_info, exc_stack)


def run_code(
    code: str,
    working_dir: "Path | str",
    timeout: int = DEFAULT_TIMEOUT,
    format_tb_ipython: bool = False,
    agent_file_name: str = DEFAULT_AGENT_FILE_NAME,
) -> ExecutionResult:
    """Execute ``code`` once in ``working_dir`` and return its result.

    Convenience one-shot wrapper around :class:`Interpreter` for callers that
    do not need to keep a session alive between executions (the ``local``
    execution provider and the Metaflow ``evaluate`` step). Spawns a fresh
    child, runs the code with ``reset_session=True``, and always cleans up the
    child process before returning.

    Args:
        code: Python source to execute.
        working_dir: Directory to run inside (must contain ``./input`` and
            ``./working`` per the AIDE workspace layout; must already exist).
        timeout: Per-execution timeout in seconds.
        format_tb_ipython: Use IPython traceback formatting if ``True``.
        agent_file_name: Name for the candidate's code file.

    Returns:
        A :class:`loom.types.ExecutionResult` with the five AIDE fields.
    """
    interpreter = Interpreter(
        working_dir,
        timeout=timeout,
        format_tb_ipython=format_tb_ipython,
        agent_file_name=agent_file_name,
    )
    try:
        return interpreter.run(code, reset_session=True)
    finally:
        interpreter.cleanup_session()


__all__ = [
    "ExecutionResult",
    "Interpreter",
    "exception_summary",
    "run_code",
    "DEFAULT_TIMEOUT",
    "DEFAULT_AGENT_FILE_NAME",
]
