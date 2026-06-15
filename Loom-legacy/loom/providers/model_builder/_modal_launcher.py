"""On-demand H100 GPU launcher for the ``nemo`` model-builder, via Modal.

This is the **real** ``_launch_and_track`` body behind ``LOOM_GPU_TARGET=modal``:
the path the ``nemo`` model-builder adapter
(``loom/providers/model_builder/nemo.py``) takes once the launch gate ALLOWS
(``launch`` requested **and** a ``gpu_target``
set). It submits the already-lowered NeMo training plan to an on-demand H100 on
Modal, tracks it, and hands the produced checkpoint back as a **Metaflow
artifact pathspec** -- never a raw file path or an object-store URI.

The shape (the ``nemo-testing.md`` design)
------------------------------------------
The Air (the laptop) is the **control plane**: it builds the submission spec and
calls ``.remote()``; the H100 burst lives on Modal and disappears when the job
finishes. Data crosses the boundary as **Metaflow artifacts** in both
directions -- the lowered plan references the input sequences by *pathspec*, and
the trained checkpoint comes back snapshotted as a Metaflow artifact pathspec
(the snapshot itself is the ``nemo`` adapter's job; see
:meth:`NemoModelBuilderProvider._snapshot_checkpoint`). This module never reads
or writes the datastore directly: it stays inside Loom's "the backbone IS the
pathspec" invariant.

Dependency posture (optional + lazy)
------------------------------------
``modal`` is **not** a Loom dependency. It is imported **lazily, inside the
function that actually submits**, and gated with a clean, actionable error when
absent ("pip install modal" + ``modal token set``), so importing this module --
and therefore ``loom`` -- works on the CPU-only conformance box with no Modal
installed and no Modal account. The submission-spec construction
(:func:`build_modal_submission`) is factored out as a **pure function** that
needs neither Modal installed nor an account, so it is fully unit-testable in
the default/CI environment.

Image / app naming
------------------
The training container and the Modal app name are read from the **environment at
the point of use** (never config fields, never committed): ``LOOM_NEMO_IMAGE``
selects the NeMo NGC container (default :data:`DEFAULT_NEMO_IMAGE`), and the
Modal app name comes from the ``modal://<app>`` form of the target (default
:data:`DEFAULT_MODAL_APP`). No secret material is read here or logged; Modal's
own token lives in the Modal config / environment, opaque to Loom.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

#: Default NeMo NGC training container (overridable via ``LOOM_NEMO_IMAGE``). A
#: pinned, documented NGC tag -- the one backend-container noun this module
#: names, deliberately confined here (the ``nemo`` adapter's lowering owns the
#: rest of the backend vocabulary). Bump the pin as the team validates newer
#: NeMo releases on H100.
DEFAULT_NEMO_IMAGE = "nvcr.io/nvidia/nemo:24.07"

#: Default Modal app name when the target is the bare ``"modal"`` (no ``//app``).
DEFAULT_MODAL_APP = "loom-nemo-pretrain"

#: The GPU class the on-demand burst requests. The spec headline is H100.
MODAL_GPU = "H100"

#: A generous default wall-clock ceiling (seconds) for the remote training job,
#: scaled up from the plan's estimated hours so a real ``budget="full"`` run is
#: not killed mid-flight. A routing constant, not a billing contract.
_TIMEOUT_SECONDS_FLOOR = 3600


@dataclass(frozen=True)
class ModalSubmission:
    """A pure, JSON-able description of *what* would be submitted to Modal.

    Built by :func:`build_modal_submission` with **no** Modal installed and no
    account, so the routing decision (which app, which container, which GPU, the
    lowered training config, the timeout) is fully unit-testable. The actual
    ``.remote()`` call (:func:`launch_on_modal`) consumes this spec; nothing here
    touches the network or a datastore.

    Attributes:
        app_name: The Modal app name the function is registered under (parsed
            from ``modal://<app>`` or the default).
        gpu: The GPU class requested (``"H100"``).
        image: The NeMo NGC training container reference (env-overridable).
        timeout_seconds: Wall-clock ceiling for the remote job.
        training_config: The lowered NeMo training config (the ``nemo`` adapter's
            plan ``backend``/``recipe``/``resources``/``tokenizer`` block). Carried
            verbatim; this module does not re-interpret backend nouns.
        gpu_target: The original Loom routing string (``"modal"`` /
            ``"modal://<app>"``) -- echoed for lineage.
    """

    app_name: str
    gpu: str
    image: str
    timeout_seconds: int
    training_config: dict = field(default_factory=dict)
    gpu_target: str = "modal"


@dataclass(frozen=True)
class ModalLaunchResult:
    """The outcome the launcher returns to the ``nemo`` adapter.

    Intentionally backend-flavored only in its ``checkpoint_handle`` (an opaque
    location string the remote job produced); the ``nemo`` adapter snapshots that
    into a Metaflow artifact and surfaces a **pathspec**, so nothing downstream
    ever sees the raw handle. No object-store URI is constructed here.

    Attributes:
        checkpoint_handle: An opaque location string the remote training job
            reported for the produced checkpoint (e.g. a Modal volume path). The
            ``nemo`` adapter turns this into a Metaflow artifact pathspec; callers
            never read it directly.
        metrics: A small JSON-able dict of final training metrics the job
            reported (e.g. ``{"final_loss": ..}``).
        gpu: The GPU class the job ran on (``"H100"``).
        app_name: The Modal app the job ran under.
    """

    checkpoint_handle: str
    metrics: dict = field(default_factory=dict)
    gpu: str = MODAL_GPU
    app_name: str = DEFAULT_MODAL_APP


def parse_modal_target(gpu_target: str) -> str:
    """Parse a ``modal`` / ``modal://<app>`` target into a Modal app name.

    The ``nemo`` adapter has already decided the target routes to Modal; this
    helper extracts the app name. The bare ``"modal"`` maps to
    :data:`DEFAULT_MODAL_APP`; ``"modal://my-app"`` maps to ``"my-app"``.

    Args:
        gpu_target: The routing string, ``"modal"`` or ``"modal://<app>"``.

    Returns:
        The Modal app name to register the training function under.
    """
    target = (gpu_target or "").strip()
    prefix = "modal://"
    if target.startswith(prefix):
        app = target[len(prefix):].strip().strip("/")
        return app or DEFAULT_MODAL_APP
    return DEFAULT_MODAL_APP


def _resolve_image() -> str:
    """Resolve the NeMo NGC training container from the environment.

    Read ``LOOM_NEMO_IMAGE`` at the point of use (never a config field, never
    committed); fall back to the documented pinned default. No secret material.
    """
    return (os.environ.get("LOOM_NEMO_IMAGE") or "").strip() or DEFAULT_NEMO_IMAGE


def _resolve_timeout(plan: dict) -> int:
    """Derive a wall-clock ceiling (seconds) from the plan's cost estimate.

    Uses the ``cost.wall_clock_hours`` physics the ``nemo`` adapter already
    surfaced at the gate, with a generous 2x headroom and a floor, so a real
    ``budget="full"`` run is not killed mid-flight. A routing estimate.
    """
    cost = plan.get("cost") or {}
    try:
        hours = float(cost.get("wall_clock_hours") or 0.0)
    except (TypeError, ValueError):
        hours = 0.0
    return max(_TIMEOUT_SECONDS_FLOOR, int(hours * 2 * 3600))


def build_modal_submission(plan: dict, gpu_target: str) -> ModalSubmission:
    """Build the Modal submission spec from a lowered NeMo plan -- PURE.

    The testable core: given the ``nemo`` adapter's already-lowered, already-costed
    launch ``plan`` and the Modal ``gpu_target``, construct the full
    :class:`ModalSubmission` (app, GPU=H100, the NeMo NGC container, the lowered
    training config, the timeout) **without importing Modal**. No network, no
    datastore, no account -- so the routing decision is unit-tested in the
    default/CI environment exactly like the rest of the lowering.

    The training config carried to the remote job is the plan's backend block
    (``backend``/``recipe``/``resources``/``tokenizer`` + the Loom-intent echo);
    this module does not re-interpret those backend nouns, it only ships them.

    Args:
        plan: The lowered, costed launch plan
            (:meth:`NemoModelBuilderProvider._lower_to_nemo_config` output).
        gpu_target: The Modal routing string (``"modal"`` / ``"modal://<app>"``).

    Returns:
        The :class:`ModalSubmission` describing the H100 training job.
    """
    app_name = parse_modal_target(gpu_target)
    # The training config is exactly the plan's backend lowering plus the intent
    # echo and cost -- shipped verbatim to the remote container, never re-lowered.
    training_config = {
        "intent": plan.get("intent", {}),
        "backend": plan.get("backend"),
        "recipe": plan.get("recipe"),
        "resources": plan.get("resources"),
        "tokenizer": plan.get("tokenizer"),
        "cost": plan.get("cost", {}),
    }
    return ModalSubmission(
        app_name=app_name,
        gpu=MODAL_GPU,
        image=_resolve_image(),
        timeout_seconds=_resolve_timeout(plan),
        training_config=training_config,
        gpu_target=(gpu_target or "modal"),
    )


def _require_modal() -> Any:
    """Import ``modal`` lazily, with a clean actionable error when it is absent.

    ``modal`` is an optional heavy dependency, not a Loom requirement, so it is
    imported only here -- at the point of an actual launch. When it is missing,
    raise a :class:`RuntimeError` that tells the operator exactly how to enable
    the Modal launch target (install + authenticate), rather than a bare
    ``ModuleNotFoundError`` deep in a job.

    Returns:
        The imported ``modal`` module.

    Raises:
        RuntimeError: If ``modal`` is not installed.
    """
    try:
        import modal  # type: ignore  # noqa: PLC0415  (lazy by design)
    except ImportError as exc:  # pragma: no cover - exercised via stubbing in tests
        raise RuntimeError(
            "the 'modal' Modal launch target is selected (LOOM_GPU_TARGET=modal) "
            "but the 'modal' package is not installed. Install it and authenticate:"
            "\n    pip install modal"
            "\n    modal token set  # (or: modal setup)"
            "\nThen re-run with --launch. Or use `model-builder local` for the "
            "CPU stand-in that needs no GPU target."
        ) from exc
    return modal


def launch_on_modal(plan: dict, gpu_target: str) -> ModalLaunchResult:
    """Submit the lowered NeMo plan to an on-demand H100 on Modal and track it.

    The real launch the ``nemo`` adapter calls once the gate ALLOWS. Steps:

    1. Build the pure :class:`ModalSubmission` spec (:func:`build_modal_submission`).
    2. Lazily import ``modal`` (clean refusal if absent -- :func:`_require_modal`).
    3. Define a Modal app + an ``H100`` function over the NeMo NGC container that
       runs the lowered training config, and invoke it with ``.remote()`` (the
       Air stays the control plane; the H100 burst is ephemeral).
    4. Return a :class:`ModalLaunchResult` with the produced checkpoint's opaque
       handle + metrics. The ``nemo`` adapter snapshots that handle into a
       **Metaflow artifact** and hands back a *pathspec* -- no raw file/URI leaks.

    Args:
        plan: The lowered, costed launch plan.
        gpu_target: The Modal routing string (``"modal"`` / ``"modal://<app>"``).

    Returns:
        A :class:`ModalLaunchResult` (checkpoint handle + metrics).

    Raises:
        RuntimeError: If ``modal`` is not installed (actionable message).
    """
    submission = build_modal_submission(plan, gpu_target)
    modal = _require_modal()

    # Control plane (the Air): declare the app + the H100 function over the NeMo
    # NGC container, then call it remotely. The heavy GPU burst is Modal's; this
    # process only submits and collects. Secrets (the Modal token) live in
    # Modal's own config/env -- never read or logged here.
    app = modal.App(submission.app_name)
    image = modal.Image.from_registry(submission.image)

    @app.function(
        gpu=submission.gpu,
        image=image,
        timeout=submission.timeout_seconds,
    )
    def _train(training_config: dict) -> dict:  # pragma: no cover - runs remotely on H100
        # Body executes inside the NeMo NGC container on the H100. It runs the
        # lowered training config and returns the produced checkpoint's location
        # plus final metrics. (The concrete NeMo invocation lives in the
        # container's entrypoint; Loom ships the config and collects the result.)
        from loom.providers.model_builder._nemo_entrypoint import run_training

        return run_training(training_config)

    with app.run():
        result = _train.remote(submission.training_config)

    result = result or {}
    return ModalLaunchResult(
        checkpoint_handle=str(result.get("checkpoint_handle", "")),
        metrics=dict(result.get("metrics", {})),
        gpu=submission.gpu,
        app_name=submission.app_name,
    )


__all__ = [
    "DEFAULT_NEMO_IMAGE",
    "DEFAULT_MODAL_APP",
    "MODAL_GPU",
    "ModalSubmission",
    "ModalLaunchResult",
    "parse_modal_target",
    "build_modal_submission",
    "launch_on_modal",
]
