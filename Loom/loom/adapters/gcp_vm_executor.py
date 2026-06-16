"""``gcp-vm`` — the executor port adapter: a SINGLE GCP GPU VM (ARCHITECTURE §10
step 7; Anub's decision: SINGLE-VM, NOT the Metaflow fan-out).

This executor submits a training ``torchrun`` argv to ONE named GCP GPU VM using
the REAL mechanism already in the TFM repo:

  * ``scripts/gcp-gpu-up.sh``        — Terraform-apply / start the single VM
    (``infra/gcp-notebook/`` + ``gcp-lib.sh`` env: GCP_INSTANCE, GCP_ZONE,
    GCP_MACHINE_TYPE, NOTEBOOK_IMAGE=nvcr.io/nvidia/nemo:25.09.01, …).
  * ``scripts/gcp-sync-workspace.sh`` — tar + ``gcloud compute ssh`` the workspace
    onto the VM at ``$REMOTE_WORKSPACE`` (so ``src/clm_data.py`` + the rendered
    YAML are present for the file-path ``_target_``).
  * the NeMo-container exec pattern from ``.github/workflows/ci.yml`` — ``docker
    run -d --gpus all <NeMo image> sleep infinity`` to keep PID 1 alive, then
    ``docker exec`` the actual ``torchrun`` inside the container.
  * ``scripts/gcp-gpu-down.sh``      — stop the VM (the binding-envelope hard-kill
    / idle teardown).

The single graft (ARCHITECTURE §2.3): ``ModelBuilder.launch()`` calls
``executor.submit(argv=…)`` and the corpus build calls ``executor.foreach(…)`` —
both flow through the same budget/kill contract, so the gated-launch behavior is
identical to the ``local`` executor. This adapter just knows *where* the work
runs (one VM) and *how* it is killed (``gcloud compute instances stop``).

================================  HARD CONSTRAINT #4  ==========================
DRY-RUNNABLE, NO GPU SPEND. Construct every gcloud/ssh/docker command WITHOUT
executing it. ``dry_run=True`` (the default, and forced by ``LOOM_DRY_RUN`` /
absence of ``LOOM_GCP_LIVE``) makes ``submit``/``foreach``/``kill`` build the
exact command lines and RETURN them on the handle (``handle.commands``) without a
single subprocess. Verification is STRUCTURAL: assert the argv/commands, never run
them. A live run requires an explicit opt-in AND ``gcloud`` present; this build +
verify never brings up a VM, runs a model turn, or spends a cent.
===============================================================================

This module imports nothing from NeMo/torch/RAPIDS/BigQuery — it is pure
subprocess-command construction + the loom ports/types.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
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

# The TFM repo root (this package lives at ``<repo>/Loom/loom/adapters/``). The
# gcp-*.sh scripts the executor wraps live under ``<repo>/scripts/``.
_TFM_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _TFM_ROOT / "scripts"

# The NeMo container the recipe runs in (matches gcp-lib.sh NOTEBOOK_IMAGE +
# ci.yml NEMO_IMAGE). Overridable via LOOM_NEMO_IMAGE / ComputeTarget.image.
_DEFAULT_NEMO_IMAGE = "nvcr.io/nvidia/nemo:25.09.01"
# The container name the recipe runs under (mirrors ci.yml CONTAINER_NAME).
_CONTAINER_NAME = "loom-nemo-train"
# The remote workspace the sync lands in (matches gcp-lib.sh REMOTE_WORKSPACE).
_REMOTE_WORKSPACE = "/mnt/tfm/workspace"
# Where the synced workspace is mounted INSIDE the NeMo container. The executor's
# ``docker run -v $REMOTE_WORKSPACE:/workspace`` puts the tar-synced tree here, so
# an in-container ``/workspace/<rel>`` path is the host file ``$REMOTE_WORKSPACE/<rel>``
# (the host↔container mount geometry, grounding §5). ``fetch_progress`` reads the
# HOST path off the VM; ``container_path`` is the inverse map the nemo builder uses
# to bake in-container paths into the torchrun argv.
_CONTAINER_WORKSPACE = "/workspace"


def _is_live() -> bool:
    """Live execution requires an EXPLICIT opt-in. Default is dry-run (no spend)."""
    if os.environ.get("LOOM_DRY_RUN"):
        return False
    return bool(os.environ.get("LOOM_GCP_LIVE"))


@dataclass
class GcpVmJobHandle:
    """A :class:`loom.ports.JobHandle` over a single-VM training job.

    On a DRY-RUN the job is born terminal (``succeeded``) and carries the exact
    command sequence it WOULD have run on ``.commands`` for structural assertion.
    On a live run ``.commands`` are the issued command lines and ``status()``
    reflects the supervised ``docker exec`` outcome."""

    job_id: str
    commands: list[list[str]] = field(default_factory=list)
    _status: _JobStatus = "pending"
    dry_run: bool = True

    def status(self) -> _JobStatus:
        return self._status

    def cancel(self) -> None:
        if self._status in ("pending", "running"):
            self._status = "killed"


class GcpVmExecutor:
    """Single GCP GPU VM executor (``name="gcp-vm"``).

    Structurally satisfies :class:`loom.ports.Executor`: ``name``,
    ``gpu_available()``, ``submit()``, ``foreach()``, ``kill()``. Plus two
    nemo-builder-facing extras: ``dry_run`` (read by the builder to mark the
    handle) and ``checkpoint_uri(run_dir, ckpt_dir)`` (the durable artifact uri the
    consolidated safetensors is fetched from)."""

    name: str = "gcp-vm"

    def __init__(self, *, dry_run: Optional[bool] = None) -> None:
        self.dry_run = (not _is_live()) if dry_run is None else dry_run
        self._jobs: dict[str, GcpVmJobHandle] = {}
        # Env knobs (read once; gcp-lib.sh owns the real defaults on the VM side).
        self._instance = os.environ.get("GCP_INSTANCE", "tfm-gpu-notebook")
        self._zone = os.environ.get("GCP_ZONE", "us-central1-f")
        self._project = os.environ.get("GCP_PROJECT", "")
        self._bucket = os.environ.get("GCP_BUCKET", "")
        self._remote_ws = os.environ.get("REMOTE_WORKSPACE", _REMOTE_WORKSPACE)

    # -- gpu_available() -------------------------------------------------

    def gpu_available(self) -> bool:
        """Gate for ``REFUSED_NO_GPU_TARGET``. A single GPU VM is the whole point of
        this executor, so it ADVERTISES GPU availability — the verb still refuses if
        no ``gpu_target`` (the machine type) is selected, but the executor itself is
        GPU-capable (unlike the ``local`` in-process executor, which returns False).

        On a DRY-RUN we never probe the cloud; we report capability so the plan +
        argv path is exercised. A live executor could additionally verify the VM is
        reachable, but that is a live-only concern outside this no-spend build."""
        return True

    # -- submit() --------------------------------------------------------

    def submit(
        self,
        *,
        argv: list[str],
        image: Optional[str],
        compute: ComputeTarget,
        budget: BudgetEnvelope,
        on_event: Callable[[ProgressEvent], None],
    ) -> GcpVmJobHandle:
        """Submit the ``torchrun`` ``argv`` to the single VM (ARCHITECTURE §5.1/§10).

        Builds — and on a live run issues — the exact command sequence:

          1. ``gcp-gpu-up.sh``         (Terraform-apply/start the VM)
          2. ``gcp-sync-workspace.sh`` (tar + ssh the workspace incl. the rendered YAML)
          3. ``docker run -d --gpus all --name <c> <NeMo image> sleep infinity``
             (PID-1-alive pattern from ci.yml), wrapped through ``gcloud compute ssh``
          4. ``docker exec -w <remote_ws> <c> <torchrun argv>`` (the recipe)

        The binding ``BudgetEnvelope`` (``max_wall_clock_min``) is passed to the live
        ``docker exec`` as a ``timeout`` wrapper — the orchestrator hard-kill
        (``kill()`` → ``gcp-gpu-down.sh``) is the binding-envelope enforcement.

        DRY-RUN: returns the handle with ``.commands`` populated and ``status() ==
        "succeeded"``; NO subprocess runs, NO VM is touched, NO cent is spent."""
        job_id = f"gcp-vm-{int(time.time() * 1000)}"
        image = image or os.environ.get("LOOM_NEMO_IMAGE") or _DEFAULT_NEMO_IMAGE
        commands = self._build_submit_commands(
            argv=argv, image=image, compute=compute, budget=budget
        )
        handle = GcpVmJobHandle(job_id=job_id, commands=commands, dry_run=self.dry_run)
        self._jobs[job_id] = handle

        if self.dry_run:
            # STRUCTURAL ONLY: the commands are constructed, not executed.
            handle._status = "succeeded"
            return handle

        # --- live path (explicit opt-in only; never exercised in this build) ----
        handle._status = "running"
        try:
            for cmd in commands:  # pragma: no cover - live-only, no-spend build never enters here
                self._run(cmd)
            handle._status = "succeeded"
        except subprocess.CalledProcessError:  # pragma: no cover - live-only
            handle._status = "failed"
        return handle

    def _build_submit_commands(
        self,
        *,
        argv: list[str],
        image: str,
        compute: ComputeTarget,
        budget: BudgetEnvelope,
    ) -> list[list[str]]:
        """Construct the exact command sequence (the heart of the dry-run)."""
        up = ["bash", str(_SCRIPTS / "gcp-gpu-up.sh")]
        sync = ["bash", str(_SCRIPTS / "gcp-sync-workspace.sh")]

        # The container start: ci.yml's PID-1-alive pattern, wrapped in `gcloud ... ssh`.
        # NVIDIA NGC images ship an ENTRYPOINT that does CUDA init; `sleep infinity`
        # keeps PID 1 up for the subsequent `docker exec`.
        docker_run = (
            f"sudo docker rm -f {shlex.quote(_CONTAINER_NAME)} >/dev/null 2>&1 || true; "
            f"sudo docker run -d --name {shlex.quote(_CONTAINER_NAME)} "
            f"--gpus all --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=8g "
            f"-v {shlex.quote(self._remote_ws)}:/workspace -w /workspace "
            f"{shlex.quote(image)} sleep infinity"
        )
        start_container = self._ssh_wrap(docker_run)

        # The recipe exec: `docker exec` the torchrun argv inside the container, from
        # the workspace CWD (so the `src/clm_data.py:...` file-path _target_ resolves
        # and the rendered YAML path — under the synced workspace — is found).
        torchrun = " ".join(shlex.quote(a) for a in argv)
        recipe = "cd /workspace && " + torchrun
        if budget.max_wall_clock_min:
            # The binding wall-clock cap as a hard `timeout` around the exec (the
            # orchestrator-side kill() is the authoritative envelope enforcement).
            secs = int(budget.max_wall_clock_min) * 60
            inner = f"timeout {secs} bash -lc {shlex.quote(recipe)}"
        else:
            inner = f"bash -lc {shlex.quote(recipe)}"
        docker_exec = f"sudo docker exec -w /workspace {shlex.quote(_CONTAINER_NAME)} {inner}"
        run_recipe = self._ssh_wrap(docker_exec)

        return [up, sync, start_container, run_recipe]

    def _ssh_wrap(self, remote_cmd: str) -> list[str]:
        """Wrap a remote shell command in ``gcloud compute ssh`` to the single VM —
        the same invocation shape as ``gcp-lib.sh:gcloud_compute_ssh``."""
        cmd = [
            "gcloud", "compute", "ssh", self._instance,
            "--zone", self._zone,
        ]
        if self._project:
            cmd += ["--project", self._project]
        if os.environ.get("GCP_TUNNEL_THROUGH_IAP") == "1":
            cmd += ["--tunnel-through-iap"]
        cmd += ["--command", remote_cmd]
        return cmd

    # -- foreach() -------------------------------------------------------

    def foreach(
        self, *, fn: Callable[[str], str], shards: list[str], compute: ComputeTarget
    ) -> list[str]:
        """The Tier-B fan-out seam (ARCHITECTURE §6). For the SINGLE-VM executor this
        is sequential on the one box (NOT the Metaflow per-shard GPU fan-out, which
        is explicitly out of scope per Anub's decision): map ``fn`` over ``shards``
        in order. A representation's ``materialize`` uses this; it never holds the
        whole corpus in RAM at the executor level (one shard per call)."""
        return [fn(shard) for shard in shards]

    # -- kill() ----------------------------------------------------------

    def kill(self, job_id: str) -> None:
        """The binding-envelope hard-kill (ARCHITECTURE §2.3): stop the VM via
        ``gcp-gpu-down.sh`` (which ``gcloud compute instances stop``s it — disk +
        bucket preserved). On a DRY-RUN this records the teardown command on the
        handle without issuing it."""
        handle = self._jobs.get(job_id)
        down = ["bash", str(_SCRIPTS / "gcp-gpu-down.sh")]
        # Also kill the in-container process first (best-effort) so a partial
        # checkpoint is consolidated before the VM stops.
        stop_exec = self._ssh_wrap(
            f"sudo docker exec {shlex.quote(_CONTAINER_NAME)} pkill -f torchrun || true"
        )
        if handle is not None:
            handle.commands.append(stop_exec)
            handle.commands.append(down)
            handle.cancel()
        if self.dry_run:
            return
        try:  # pragma: no cover - live-only, no-spend build never enters here
            self._run(stop_exec)
            self._run(down)
        except subprocess.CalledProcessError:
            pass

    # -- nemo-builder-facing extra: the durable checkpoint uri -----------

    def checkpoint_uri(self, run_dir: str, ckpt_dir: str) -> str:
        """The durable uri the consolidated safetensors is fetched from (read by
        ``nemo_builder``). Prefers the GCS artifact bucket (``gcp-sync-models.sh``
        syncs the VM checkpoint dir there); falls back to a ``vm:`` uri naming the
        VM path so the verb records a REAL location (never a tempdir, the §10
        step-1-6 follow-up fix)."""
        rel = str(ckpt_dir).strip("/")
        if self._bucket:
            return f"gs://{self._bucket}/loom-checkpoints/{Path(run_dir).name}/{rel}"
        return f"vm://{self._instance}/{self._remote_ws.strip('/')}/{rel}"

    # -- nemo-builder-facing extra: control-plane → in-container path map -

    def container_path(self, host_path: str) -> str:
        """Translate a control-plane path → its IN-CONTAINER location (CWD
        ``/workspace``) — the duck-typed hook the nemo builder probes when baking
        paths into the ``torchrun`` argv (mirrors ``checkpoint_uri``). The executor
        tar-syncs the workspace root to ``$REMOTE_WORKSPACE`` and ``docker run -v
        $REMOTE_WORKSPACE:/workspace``, so making the prefix-swap executor-authoritative
        here keeps the mount geometry in ONE place (the executor that owns the mount),
        rather than relying on the builder's default heuristic.

        A cloud URI (``gs://…``) or a path already under ``/workspace`` /
        ``$REMOTE_WORKSPACE`` passes through (the latter remapped to its in-container
        prefix); a control-plane path under the synced workspace root is prefix-swapped
        to ``/workspace/<rel>``; anything else is returned unchanged so a live run
        surfaces the real error rather than silently mangling an unrelated path."""
        if not host_path:
            return host_path
        if "://" in host_path:
            return host_path
        norm = host_path.replace(os.sep, "/")
        # Already in-container.
        if norm == _CONTAINER_WORKSPACE or norm.startswith(_CONTAINER_WORKSPACE + "/"):
            return norm
        # A host-mount path ($REMOTE_WORKSPACE/<rel>) → /workspace/<rel>.
        rws = self._remote_ws.rstrip("/")
        if norm == rws or norm.startswith(rws + "/"):
            rel = norm[len(rws):].lstrip("/")
            return _CONTAINER_WORKSPACE + ("/" + rel if rel else "")
        # A control-plane path under the synced workspace root → /workspace/<rel>.
        for root in self._workspace_roots():
            r = root.rstrip("/")
            if not r:
                continue
            if norm == r or norm.startswith(r + "/"):
                rel = norm[len(r):].lstrip("/")
                return _CONTAINER_WORKSPACE + ("/" + rel if rel else "")
        return host_path

    @staticmethod
    def _workspace_roots() -> list[str]:
        """Control-plane roots of the tree the executor tar-syncs onto the VM, in
        precedence order (``LOOM_WORKSPACE`` then the TFM repo root). Mirrors the
        builder-side helper so the two agree on which prefix maps to ``/workspace``."""
        roots: list[str] = []
        ws = os.environ.get("LOOM_WORKSPACE")
        if ws:
            roots.append(os.path.abspath(ws))
        roots.append(str(_TFM_ROOT))
        seen: set[str] = set()
        out: list[str] = []
        for r in roots:
            if r and r not in seen:
                seen.add(r)
                out.append(r)
        return out

    # -- nemo-builder-facing extra: pull the JSONL back from the VM ------

    def _host_path(self, remote: str) -> str:
        """Map a path the launcher wrote IN the container to its HOST location on
        the VM. The launcher's ``--loom-progress-jsonl`` is the in-container
        ``/workspace/<rel>`` path; the backing file on the VM is
        ``$REMOTE_WORKSPACE/<rel>`` (the ``-v $REMOTE_WORKSPACE:/workspace`` mount,
        grounding §5). Reading the host path lets ``fetch_progress`` use a plain
        ``cat`` over ssh — no second ``docker exec``. A non-``/workspace`` path
        (already a host/durable path like ``/mnt/tfm/artifacts/…``) is read as-is."""
        if not remote:
            return remote
        norm = remote.replace(os.sep, "/")
        rws = self._remote_ws.rstrip("/")
        if norm == _CONTAINER_WORKSPACE:
            return rws
        if norm.startswith(_CONTAINER_WORKSPACE + "/"):
            rel = norm[len(_CONTAINER_WORKSPACE) + 1:]
            return rws + "/" + rel
        return norm

    def fetch_progress(self, remote: str, local: str) -> None:
        """Pull the launcher's progress JSONL back from the VM → the control-plane
        read path (the duck-typed hook :class:`~loom.adapters.nemo_builder.NeMoTrainingHandle`
        calls before each read; grounding finding #3). Without it the net-new JSONL
        the launcher writes ON THE VM is unreachable from the control plane, so the
        launch-and-track widget feed, the per-step ``usd_spent`` telemetry, and
        ``result()``'s ``final_loss``/``step0_canary`` are all silently empty on a
        live single-VM run.

        Mechanism (grounding §5): ``gcloud compute ssh <inst> --zone --project
        [--tunnel-through-iap] --command 'cat <host_path>'`` (the same ssh primitive
        as ``gcp-lib.sh:gcloud_compute_ssh``), redirected into the local control-plane
        file — ``remote`` is the in-container ``/workspace/<rel>`` path, mapped to its
        backing host file ``$REMOTE_WORKSPACE/<rel>`` first.

        DRY-RUN: a NO-OP (no ssh, no spend) — the structural path never has a VM file
        to pull, so reading degrades to whatever the control-plane file already holds.
        This keeps HARD CONSTRAINT #4 (the no-spend build never touches the VM)."""
        if self.dry_run or not remote or not local:
            return
        host = self._host_path(remote)
        cmd = self._ssh_wrap(f"cat {shlex.quote(host)}")
        # pragma: no cover - live-only, the no-spend build never reaches here.
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)  # pragma: no cover
        with open(local, "w", encoding="utf-8") as fh:  # pragma: no cover - live-only
            subprocess.run(cmd, check=True, stdout=fh)

    # -- live subprocess runner (live-only) ------------------------------

    @staticmethod
    def _run(cmd: list[str]) -> None:  # pragma: no cover - live-only, no-spend build
        """Run a command, raising on non-zero. NEVER reached on a dry-run."""
        subprocess.run(cmd, check=True)


# ARCHITECTURE §2.4 / §10 step 7: one-line registration under the registry key.
register_executor(GcpVmExecutor())

__all__ = ["GcpVmExecutor", "GcpVmJobHandle"]
