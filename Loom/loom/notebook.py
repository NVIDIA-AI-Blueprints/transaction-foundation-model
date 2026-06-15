"""``loom notebook`` — a GPU-backed remote Jupyter in the NeMo container, on Modal.

The fresh-Mac gap: the documented DS path for GPU-native repos is "remote Jupyter
inside the NeMo container on a Linux/NVIDIA host" — painful to stand up on an
Apple-Silicon laptop (host prep, Docker flags, Git LFS, port forwarding). Loom is
NOT a notebook IDE/host, but it already provisions exactly that container on an
on-demand Modal H100 for *training* (see
:mod:`loom.providers.model_builder._modal_launcher`). This verb reuses that same
seam to launch **JupyterLab** in the **same NeMo image** on a Modal GPU and forward
it back to the laptop — so a data scientist gets the documented remote-notebook
environment with one command, and never touches a GPU host. It is just another
remote-compute *launcher*, consistent with Loom's "laptop is the control plane;
heavy compute is ephemeral and remote" architecture.

Dependency posture mirrors the training launcher: ``modal`` is an OPTIONAL,
lazily-imported dependency (the submission-spec build is a pure function needing no
Modal and no account, fully unit-testable on a CPU box), and the container image is
read from the environment at the point of use (``LOOM_NEMO_IMAGE``). No secret
material is read or logged here; the Modal token lives in Modal's own config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Reuse the exact Modal+NeMo seam the training launcher already established: same
# container image resolution, same GPU class, same app-name parsing, same lazy
# import gate. A notebook is a different *use* of that one remote-compute seam.
from loom.providers.model_builder._modal_launcher import (
    MODAL_GPU,
    _require_modal,
    _resolve_image,
    parse_modal_target,
)

#: The port JupyterLab binds inside the container (forwarded to the laptop).
DEFAULT_NOTEBOOK_PORT = 8888

#: Default wall-clock ceiling (seconds) for an interactive notebook session,
#: overridable via ``LOOM_NOTEBOOK_TIMEOUT``. Interactive, so generous (4h); the
#: burst still disappears when the session ends or the ceiling hits.
_DEFAULT_TIMEOUT_SECONDS = 4 * 3600

#: The datastore env vars forwarded into the remote notebook so it can read Loom
#: data objects through the Metaflow Client API exactly as the laptop would — the
#: same set ``loom doctor`` checks. Forwarded as values, never logged.
_DATASTORE_ENV_VARS = (
    "METAFLOW_DEFAULT_DATASTORE",
    "METAFLOW_DATASTORE_SYSROOT_S3",
    "METAFLOW_S3_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "METAFLOW_DEFAULT_METADATA",
    "METAFLOW_SERVICE_URL",
    "METAFLOW_USER",
)


def routes_to_modal(gpu_target: str) -> bool:
    """Whether a target routes to the Modal launcher (``modal`` / ``modal://app``)."""
    return (gpu_target or "").strip().startswith("modal")


@dataclass(frozen=True)
class NotebookSubmission:
    """A pure, JSON-able description of the remote notebook to launch — PURE.

    Built by :func:`build_notebook_submission` with no Modal installed and no
    account, so the routing decision (app, GPU, image, port, timeout, whether the
    datastore env is forwarded) is fully unit-testable. :func:`launch_notebook`
    consumes it; nothing here touches the network.

    Attributes:
        app_name: The Modal app the notebook function registers under.
        gpu: The GPU class requested (``"H100"``).
        image: The NeMo NGC container reference (env-overridable).
        port: The port JupyterLab binds inside the container.
        timeout_seconds: Wall-clock ceiling for the interactive session.
        mount_datastore: Whether to forward the datastore env into the container so
            the notebook can read Loom data objects via the Metaflow Client API.
        gpu_target: The original routing string (``"modal"`` / ``"modal://app"``).
    """

    app_name: str
    gpu: str
    image: str
    port: int
    timeout_seconds: int
    mount_datastore: bool
    gpu_target: str = "modal"


def _resolve_timeout() -> int:
    """Wall-clock ceiling from ``LOOM_NOTEBOOK_TIMEOUT`` (seconds) or the default."""
    raw = (os.environ.get("LOOM_NOTEBOOK_TIMEOUT") or "").strip()
    try:
        return int(raw) if raw else _DEFAULT_TIMEOUT_SECONDS
    except ValueError:
        return _DEFAULT_TIMEOUT_SECONDS


def build_notebook_submission(
    gpu_target: str,
    *,
    mount_datastore: bool = True,
    port: int = DEFAULT_NOTEBOOK_PORT,
) -> NotebookSubmission:
    """Build the remote-notebook submission spec — PURE (no Modal, no account).

    Args:
        gpu_target: The Modal routing string (``"modal"`` / ``"modal://<app>"``).
        mount_datastore: Forward the datastore env into the container.
        port: The JupyterLab port inside the container.

    Returns:
        The :class:`NotebookSubmission` describing the GPU notebook to launch.
    """
    return NotebookSubmission(
        app_name=parse_modal_target(gpu_target),
        gpu=MODAL_GPU,
        image=_resolve_image(),
        port=port,
        timeout_seconds=_resolve_timeout(),
        mount_datastore=mount_datastore,
        gpu_target=(gpu_target or "modal"),
    )


def _datastore_env() -> dict:
    """The subset of the live datastore env to forward (only the set vars)."""
    return {
        k: os.environ[k] for k in _DATASTORE_ENV_VARS if (os.environ.get(k) or "").strip()
    }


def launch_notebook(submission: NotebookSubmission) -> int:
    """Launch JupyterLab in the NeMo container on a Modal GPU and forward it back.

    The laptop is the control plane: it declares a Modal app + a GPU function over
    the NeMo NGC image that starts JupyterLab and opens a public tunnel
    (``modal.forward``) to it, prints the URL+token, and blocks for the session.
    The GPU burst is ephemeral — it disappears when you stop the notebook or the
    timeout hits. Mirrors :func:`loom.providers.model_builder._modal_launcher.launch_on_modal`.

    Args:
        submission: The pure spec from :func:`build_notebook_submission`.

    Returns:
        Process exit code (0 once the remote session ends).

    Raises:
        RuntimeError: If ``modal`` is not installed (actionable message, reused
            from the training launcher's gate).
    """
    import secrets

    modal = _require_modal()
    token = secrets.token_urlsafe(16)
    datastore_env = _datastore_env() if submission.mount_datastore else {}

    app = modal.App(submission.app_name)
    # JupyterLab is present in recent NeMo NGC images, but pip-install is idempotent
    # and keeps this robust across image pins.
    image = modal.Image.from_registry(submission.image).pip_install("jupyterlab")

    @app.function(
        gpu=submission.gpu,
        image=image,
        timeout=submission.timeout_seconds,
        serialized=True,  # _serve is nested (not global scope); required by modal>=1.x
    )
    def _serve(env: dict, tok: str) -> None:  # pragma: no cover - runs remotely on the GPU
        # Inside the NeMo container on the GPU: export the forwarded datastore env
        # (so the Metaflow Client API works), open a public tunnel to the Jupyter
        # port, print the URL, and run JupyterLab in the foreground for the session.
        import os as _os
        import subprocess

        _os.environ.update(env)
        with modal.forward(submission.port) as tunnel:
            url = f"{tunnel.url}/lab?token={tok}"
            # Machine-parseable marker so the detached parent (spawn_notebook) can
            # capture the URL from this process's log, plus a human line.
            print(f"LOOM_NOTEBOOK_URL={url}", flush=True)
            print(f"\n  Open your GPU notebook:  {url}\n", flush=True)
            subprocess.run(
                [
                    "jupyter", "lab",
                    "--ip=0.0.0.0", f"--port={submission.port}",
                    "--no-browser", "--allow-root",
                    f"--ServerApp.token={tok}",
                ],
                check=False,
            )

    print(
        f"Serving a {submission.gpu} notebook in {submission.image} on Modal app "
        f"'{submission.app_name}' (up to {submission.timeout_seconds // 3600}h)."
    )
    with app.run():
        _serve.remote(datastore_env, token)
    return 0


# --- Background lifecycle: spin up (detached), report URL, stop ----------------
# `loom notebook --launch` must be NON-BLOCKING so the agent can spin a notebook up
# and tell the user its URL. So it backgrounds the blocking server above (run via
# the hidden `loom notebook --serve`) in a detached session, captures the URL from
# its log, and returns — the GPU notebook keeps running until `--stop`.

_URL_MARKER = "LOOM_NOTEBOOK_URL="


def _state_dir() -> Path:
    """Per-user state dir for background notebook logs + pids (~/.loom/notebooks)."""
    d = Path(os.path.expanduser("~/.loom/notebooks"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_url(log: Path) -> Optional[str]:
    """Read the captured notebook URL from a server log, or None if not up yet."""
    try:
        for line in log.read_text(errors="ignore").splitlines():
            if line.startswith(_URL_MARKER):
                return line[len(_URL_MARKER):].strip()
    except OSError:
        pass
    return None


def spawn_notebook(submission: NotebookSubmission, wait_seconds: int = 45) -> dict:
    """Spin the notebook up in a DETACHED background process and return its URL.

    Non-blocking: backgrounds ``loom notebook --serve`` (the blocking server) in a
    new session so the GPU notebook outlives this call, then polls its log for the
    URL. Returns ``LAUNCHED`` + url when the tunnel is up, or ``LAUNCHING`` (the
    first launch pulls the container, minutes) with the URL to be fetched via
    :func:`notebook_status`. ``--stop`` tears it down. Returns a JSON-able dict.
    """
    import subprocess
    import sys
    import time

    sd = _state_dir()
    app = submission.app_name
    log = sd / f"{app}.log"
    pidf = sd / f"{app}.pid"

    argv = [sys.executable, "-m", "loom", "notebook", "--serve", "--gpu", submission.gpu_target]
    if not submission.mount_datastore:
        argv.append("--no-datastore")

    logf = open(log, "w")  # noqa: SIM115 - handed to the detached child; closed on its exit
    proc = subprocess.Popen(
        argv, stdout=logf, stderr=subprocess.STDOUT,
        start_new_session=True, env=dict(os.environ),
    )
    pidf.write_text(str(proc.pid))

    deadline = time.time() + max(0, wait_seconds)
    url = _read_url(log)
    while url is None and time.time() < deadline:
        if proc.poll() is not None:  # the server exited early -> a real failure
            return {
                "status": "ERROR", "url": None, "app_name": app, "log": str(log),
                "message": f"the launch process exited early (code {proc.returncode}); see {log}",
            }
        time.sleep(2)
        url = _read_url(log)

    if url:
        return {"status": "LAUNCHED", "url": url, "app_name": app, "pid": proc.pid, "log": str(log)}
    return {
        "status": "LAUNCHING", "url": None, "app_name": app, "pid": proc.pid, "log": str(log),
        "message": "provisioning — the first launch pulls the container (~minutes). "
                   "Ask for the status (or `loom notebook --status`) for the URL.",
    }


def notebook_status(app_name: str) -> dict:
    """Report a background notebook's status + URL (LAUNCHED / LAUNCHING / STOPPED)."""
    sd = _state_dir()
    log = sd / f"{app_name}.log"
    pidf = sd / f"{app_name}.pid"
    url = _read_url(log)
    alive = False
    if pidf.exists():
        try:
            os.kill(int(pidf.read_text().strip()), 0)
            alive = True
        except (ValueError, ProcessLookupError, PermissionError):
            alive = False
    if alive and url:
        return {"status": "LAUNCHED", "url": url, "app_name": app_name}
    if alive:
        return {"status": "LAUNCHING", "url": None, "app_name": app_name,
                "message": "still provisioning; check again shortly."}
    return {"status": "STOPPED", "url": None, "app_name": app_name,
            "message": "no running notebook for this app."}


def stop_notebook(app_name: str) -> dict:
    """Tear down a background notebook (stops the GPU + billing) by its app name."""
    import signal

    sd = _state_dir()
    pidf = sd / f"{app_name}.pid"
    if not pidf.exists():
        return {"status": "STOPPED", "app_name": app_name,
                "message": "no running notebook for this app."}
    try:
        pid = int(pidf.read_text().strip())
        # The server runs in its own session (start_new_session); kill the group so
        # app.run() exits and Modal releases the GPU.
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ValueError, ProcessLookupError, PermissionError):
        pass
    finally:
        pidf.unlink(missing_ok=True)
    return {"status": "STOPPED", "app_name": app_name, "message": "notebook stopped (GPU released)."}


__all__ = [
    "DEFAULT_NOTEBOOK_PORT",
    "NotebookSubmission",
    "routes_to_modal",
    "build_notebook_submission",
    "launch_notebook",
    "spawn_notebook",
    "notebook_status",
    "stop_notebook",
]
