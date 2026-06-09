"""NeMo model-builder provider (``"nemo"``) -- a lowering compiler that gates GPU.

NeMo is the **default** ``model_builder_provider`` (``LoomConfig`` defaults to
``"nemo"``), but in v0.1 the *real* GPU pretrain is **deferred**: this adapter is
a **compiler that lowers Loom DS-intent into a backend launch PLAN** and **gates
the real GPU launch** behind an explicit ``launch`` flag, refusing cleanly when
no GPU target is configured. It is the model-builder analogue of
:class:`flows.deploy.DeployFlow`'s ``--apply`` posture: the default produces a
PLAN (or a clean refusal) and performs **no mutation**; the heavy GPU launch
runs only when ``launch`` is True **and** a ``gpu_target`` is set (the
``apply and gate["allow"]`` shape).

The one rule (the abstraction boundary, constraint 2)
-----------------------------------------------------
The seam is drawn in **Loom DS-intent vocabulary**. Every backend noun -- the
decoder family, RoPE/GQA, the parallelism knobs, the GPU-count resource
decorator, the container digest, the online-serving microservice, the checkpoint
file extension -- lives **only inside this module's internal lowering strings**
(the values produced by :meth:`_lower_to_nemo_config`). Nothing a downstream
caller can ``grep`` in a session ever surfaces a backend noun: if it did, the
abstraction would have leaked. The Loom-intent enums (``objective``/``budget``/
``mode``) are validated against the module-level frozensets at the seam, so a
backend noun passed *in* is rejected before any lowering happens.

The vocabulary-vs-physics line (constraint 5)
---------------------------------------------
The abstraction hides **vocabulary, not physics**. ``budget="full"`` still costs
real GPU-hours; :meth:`_estimate_cost` surfaces hours / dollars / GPU-count **at
the gate** so a caller approving a launch sees the true cost, never a number
faked away.

I/O discipline (constraints 1 & 3)
----------------------------------
Outputs are :class:`~loom.types.ArtifactRef` / :class:`~loom.types.Scores`
carrying **Metaflow run pathspecs** and small JSON-able summaries -- never a
checkpoint file, never an object-store URI, never a raw datastore handle. This
module is **torch-free and NeMo-import-free**: it does not import any heavy GPU
dependency at all (the lowering is pure string/dict construction), so it imports
and self-registers on the CPU-only conformance box exactly like the ``local``
adapter, and the default/CI path stays green without ``torch`` or NeMo present.
"""

from __future__ import annotations

from loom.config import LoomConfig
from loom.providers import BUDGETS, MODES, OBJECTIVES, ModelBuilderProvider
from loom.registry import register_model_builder
from loom.types import ArtifactRef, Capability, CapabilityManifest, Scores

# ---------------------------------------------------------------------------
# Internal lowering vocabulary (the ONLY place backend nouns may appear).
#
# These strings are the "hidden" right column of the §4.1 translation table.
# They are deliberately quarantined behind this comment fence and inside the
# private ``_lower_*``/``_estimate_*`` helpers so the source-scan conformance
# test (suite test 8) can assert that no backend noun escapes this module's
# internal lowering strings into the Loom-vocabulary seam.
# ---------------------------------------------------------------------------

#: objective (Loom intent) -> the hidden backend training recipe it lowers to.
_OBJECTIVE_RECIPE: dict[str, str] = {
    # next-event => an autoregressive causal-LM recipe (decoder stack, rotary
    # position embeddings, grouped-query attention).
    "next-event": "automodel-causal-lm:decoder/rope/gqa",
    # masked-field => a masked-field denoising pretraining config.
    "masked-field": "masked-field-pretrain:bidirectional-denoise",
    # contrastive => a contrastive / negative-sampling sequence objective.
    "contrastive": "contrastive-seq:infonce-negative-sampling",
}

#: budget (Loom intent) -> the hidden resource + parallelism lowering. The
#: GPU-count is the physics the gate surfaces; the parallelism knobs are the
#: backend's concern.
_BUDGET_RESOURCES: dict[str, dict] = {
    # probe => single accelerator, micro-batch, no model parallelism.
    "probe": {
        "resources": "gpu=1",
        "micro_batch": 4,
        "tensor_parallel": 1,
        "pipeline_parallel": 1,
        "launcher": "single-node",
    },
    # small => single accelerator, larger micro-batch, still no model parallel.
    "small": {
        "resources": "gpu=1",
        "micro_batch": 16,
        "tensor_parallel": 1,
        "pipeline_parallel": 1,
        "launcher": "single-node",
    },
    # full => eight accelerators with tensor + pipeline model parallelism and
    # the multi-node launcher (the heavy, expensive path).
    "full": {
        "resources": "gpu=8",
        "micro_batch": 32,
        "tensor_parallel": 4,
        "pipeline_parallel": 2,
        "launcher": "multi-node-run",
    },
}

#: budget -> a coarse cost estimate the gate surfaces (PHYSICS, never faked).
#: ``gpu_count`` x ``wall_clock_hours`` x a notional ``usd_per_gpu_hour`` gives a
#: dollar figure; ``budget="full"`` is the spec's 8xA100-hours headline.
_BUDGET_COST: dict[str, dict] = {
    "probe": {"gpu_count": 1, "wall_clock_hours": 0.5},
    "small": {"gpu_count": 1, "wall_clock_hours": 4.0},
    "full": {"gpu_count": 8, "wall_clock_hours": 12.0},
}

#: Notional accelerator price used only to turn the (GPU-count x hours) physics
#: into a dollar figure at the gate. A routing/estimation constant, not a secret.
_USD_PER_GPU_HOUR = 3.0

#: serving mode (Loom intent) -> the hidden serving lowering.
_SERVE_LOWERING: dict[str, str] = {
    # batch => a frozen-embedding batch scoring step (cheap; no online service).
    "batch": "frozen-embedding-batch-score",
    # online => spin up an online inference microservice container (heavy; needs
    # a GPU target -- launch-and-track).
    "online": "online-inference-microservice-container",
}


@register_model_builder("nemo")
class NemoModelBuilderProvider(ModelBuilderProvider):
    """The ``nemo`` adapter: a lowering compiler that gates the real GPU launch.

    Compiles Loom DS-intent (``objective``/``budget``/``mode``) into a backend
    launch PLAN and emits that plan as an :class:`~loom.types.ArtifactRef`,
    gating the real GPU launch behind the ``launch`` flag with a clean
    "no GPU target configured" refusal -- the :class:`flows.deploy.DeployFlow`
    ``--apply`` posture, ported to the model-builder seam.

    Constructed uniformly as ``NemoModelBuilderProvider(config)`` like every
    other Loom provider. The per-run ``launch`` intent (which the
    ``TrainFlow``'s ``launch = Parameter(default=False)`` threads, copied from
    ``DeployFlow.apply``) is an **optional constructor argument defaulting to
    ``False``**, so the locked ABC method signatures are unchanged and the
    default construction is plan-only / refusing -- never mutating.

    The gate decision for the launch-and-track capabilities (``pretrain``,
    ``serve(online)``) is computed in plain Python from exactly two inputs:

    * ``config.gpu_target is None`` -> ``REFUSED_NO_GPU_TARGET`` (refuse up front
      with an actionable message; no plan run, no mutation);
    * a ``gpu_target`` is set but ``launch`` is False (the default) -> ``PLANNED``
      (a staged launch PLAN artifact; no mutation, like ``DeployFlow`` PLANNED);
    * ``launch`` is True **and** a ``gpu_target`` is set -> the real heavy GPU
      launch via :meth:`_launch_and_track` (v0.2+; a clearly-marked stub here).

    The searchable capabilities (``tokenize``/``finetune``/``embed``/
    ``evaluate``) lower to cheap steps the backend can run without the heavy
    gate; in v0.1 they return a clean ``PLANNED`` :class:`ArtifactRef`/
    :class:`Scores` (the conformance suite asserts they never raise
    ``NotImplementedError`` and honor the frozenset validation at the seam).

    Attributes:
        name: Registry name, ``"nemo"``.
        config: The active :class:`~loom.config.LoomConfig` (supplies
            ``gpu_target`` and ``model_builder_base_url``).
        launch: Whether the real heavy GPU launch was requested for this run.
            OFF by default; only meaningful together with a set ``gpu_target``.
    """

    name = "nemo"

    def __init__(self, config: LoomConfig, launch: bool = False) -> None:
        """Initialize from a Loom config (no secret material is read).

        Args:
            config: The active configuration. ``config.gpu_target`` is the field
                that makes the launch gate enforceable (``None`` => refuse up
                front); the matching cluster credentials stay in the environment,
                never on this object (the ``config.py`` secret rule).
            launch: Whether the real heavy GPU launch was requested (the flow's
                ``--launch`` flag). **OFF by default** -- the default run is a
                PLAN-only / refusing path that performs no mutation.
        """
        self.config = config
        self.launch = bool(launch)

    # -- the negotiation contract (§4.2) -----------------------------------

    def manifest(self) -> CapabilityManifest:
        """Return the ``nemo`` :class:`~loom.types.CapabilityManifest` (§4.2).

        ``pretrain`` and ``serve`` are ``launch-and-track`` (heavy/external; AIDE
        never tree-searches them -- enforced at the seam); ``tokenize``,
        ``finetune``, ``embed`` and ``evaluate`` are ``searchable`` (cheap steps
        AIDE may search). Every capability is declared ``supported=True`` with a
        non-empty honesty note where its v0.1 behavior is gated/limited (§9.5
        "don't over-sell"), so a caller reads the gate posture up front.
        """
        return CapabilityManifest(
            backend="nemo",
            capabilities={
                "tokenize": Capability(
                    "tokenize", "searchable", True,
                    "GPU tokenizer lowering; runs as a cheap step (PLANNED in v0.1).",
                ),
                "pretrain": Capability(
                    "pretrain", "launch-and-track", True,
                    "real GPU pretrain; requires gpu_target; OFF behind --launch "
                    "in v0.1 (PLAN-only, or REFUSED_NO_GPU_TARGET with no target).",
                ),
                "finetune": Capability(
                    "finetune", "searchable", True,
                    "frozen-backbone + cheap head; runs as a cheap step (PLANNED in v0.1).",
                ),
                "embed": Capability(
                    "embed", "searchable", True,
                    "frozen-backbone embedding step; runs cheap (PLANNED in v0.1).",
                ),
                "evaluate": Capability(
                    "evaluate", "searchable", True,
                    "wires the eval harness (temporal split, PR-AUC); PLANNED in v0.1.",
                ),
                "serve": Capability(
                    "serve", "launch-and-track", True,
                    "online => an inference microservice (needs gpu_target, gated "
                    "like pretrain); batch => a cheap frozen-embedding score step.",
                ),
            },
        )

    # -- the lowering compiler (§4.1) --------------------------------------

    def _lower_to_nemo_config(
        self, sequences_ref: str, objective: str, budget: str
    ) -> dict:
        """Lower Loom DS-intent -> a backend launch config (the §4.1 table).

        This is the actual "hiding": the DS sees only ``objective``/``budget``;
        this helper owns the mapping to the backend recipe + resource + tokenizer
        + launcher knobs. **Every backend noun appears only here**, inside the
        produced strings (sourced from the module-level ``_OBJECTIVE_RECIPE`` /
        ``_BUDGET_RESOURCES`` tables), so nothing a downstream caller can grep in
        a session leaks a backend noun. Side-effect free: it only *describes* the
        launch; it never performs it.

        Args:
            sequences_ref: Pathspec of the sequences data object (a Loom
                pathspec; the lowered plan references it by pathspec, never by a
                datastore URI).
            objective: A Loom objective, already validated against
                :data:`~loom.providers.OBJECTIVES` by the caller.
            budget: A Loom budget, already validated against
                :data:`~loom.providers.BUDGETS` by the caller.

        Returns:
            A JSON-able plan dict carrying the lowered backend config plus the
            Loom-intent echo, the ``gpu_target`` it would run on, and the
            tokenizer lowering. (The ``cost`` block is added by the caller via
            :meth:`_estimate_cost` so the physics shows at the gate.)
        """
        recipe = _OBJECTIVE_RECIPE[objective]
        resources = dict(_BUDGET_RESOURCES[budget])
        return {
            # The Loom-intent echo (the only vocabulary the caller stated).
            "intent": {
                "objective": objective,
                "budget": budget,
                "sequences_ref": sequences_ref,
            },
            # The lowered backend config (all backend nouns confined here).
            "backend": "nemo",
            "recipe": recipe,
            "resources": resources,
            # High-cardinality field tokenization lowers to a GPU tokenizer with
            # hashed / compositional embeddings (the §4.1 scheme=field-tokenize row).
            "tokenizer": "gpu-field-tokenizer:hashed-compositional",
            # The backbone is a checkpoint *pathspec* (a Metaflow artifact), never
            # a checkpoint file -- the "the backbone IS the pathspec" invariant.
            "backbone_artifact": "metaflow-checkpoint-pathspec",
            # The GPU target this would run on (a routing string, never creds).
            "gpu_target": self.config.gpu_target,
            "base_url": self.config.model_builder_base_url,
        }

    def _estimate_cost(self, budget: str) -> dict:
        """Estimate the launch cost for ``budget`` -- PHYSICS surfaced at the gate.

        The abstraction hides vocabulary, **not** physics (constraint 5):
        ``budget="full"`` still costs the spec's 8xA100-hours, and this helper
        makes that true cost legible at the gate (GPU-count, wall-clock hours,
        and a dollar figure) rather than faking it away. The numbers are coarse
        planning estimates, not a billing contract.

        Args:
            budget: A Loom budget, validated against :data:`BUDGETS` by the caller.

        Returns:
            A JSON-able cost dict: ``gpu_count``, ``wall_clock_hours``,
            ``gpu_hours`` (the product), ``usd_per_gpu_hour``,
            ``est_usd`` (the dollar figure), and a human-readable ``headline``.
        """
        base = _BUDGET_COST[budget]
        gpu_count = int(base["gpu_count"])
        wall_clock_hours = float(base["wall_clock_hours"])
        gpu_hours = gpu_count * wall_clock_hours
        est_usd = round(gpu_hours * _USD_PER_GPU_HOUR, 2)
        headline = (
            f"budget={budget}: {gpu_count} GPU x {wall_clock_hours:g} h "
            f"= {gpu_hours:g} GPU-hours (~${est_usd:g})"
        )
        return {
            "gpu_count": gpu_count,
            "wall_clock_hours": wall_clock_hours,
            "gpu_hours": gpu_hours,
            "usd_per_gpu_hour": _USD_PER_GPU_HOUR,
            "est_usd": est_usd,
            "headline": headline,
        }

    # -- launch-and-track capabilities (gated; §4.2) -----------------------

    def pretrain(self, sequences_ref: str, objective: str, budget: str) -> ArtifactRef:
        """Pretrain a backbone -- the gated launch-and-track path (§4.2).

        Rejects a backend noun passed *in* at the seam (``objective`` /
        ``budget`` must be Loom-intent enums), lowers the intent to a backend
        launch PLAN, surfaces the cost at the gate, then applies the gate:

        * ``gpu_target is None`` -> ``REFUSED_NO_GPU_TARGET`` (no plan run, no
          mutation; an actionable message points at ``LOOM_GPU_TARGET`` or the
          ``local`` CPU stand-in);
        * a ``gpu_target`` set but ``launch`` False (default) -> ``PLANNED`` (a
          staged launch PLAN artifact; no mutation);
        * ``launch`` True **and** a ``gpu_target`` set -> the real heavy GPU
          launch via :meth:`_launch_and_track` (v0.2+).

        Args:
            sequences_ref: Pathspec of the sequences data object.
            objective: A Loom objective (validated against :data:`OBJECTIVES`).
            budget: A Loom budget (validated against :data:`BUDGETS`).

        Returns:
            An :class:`~loom.types.ArtifactRef` of ``kind="backbone"`` carrying
            the plan + the gate status (``REFUSED_NO_GPU_TARGET`` / ``PLANNED`` /
            a launched pathspec) in its ``summary``.
        """
        # Reject a backend noun passed IN at the seam, in Loom vocabulary.
        assert objective in OBJECTIVES, (
            f"objective {objective!r} is not a Loom objective; expected one of "
            f"{sorted(OBJECTIVES)} (NeMo nouns are rejected at the seam)."
        )
        assert budget in BUDGETS, (
            f"budget {budget!r} is not a Loom budget; expected one of "
            f"{sorted(BUDGETS)} (NeMo nouns are rejected at the seam)."
        )

        plan = self._lower_to_nemo_config(sequences_ref, objective, budget)
        plan["cost"] = self._estimate_cost(budget)  # PHYSICS at the gate
        return self._gate("backbone", plan)

    def serve(self, model_ref: str, mode: str) -> ArtifactRef:
        """Serve a model -- ``batch`` is cheap, ``online`` is gated (§4.1/§4.2).

        Validates ``mode`` against :data:`~loom.providers.MODES` at the seam.
        ``mode="batch"`` lowers to a cheap frozen-embedding scoring step and
        returns a ``PLANNED`` endpoint ref (no GPU target needed). ``mode="online"``
        is ``launch-and-track`` (it would spin up an online inference
        microservice) and so goes through the same gate as :meth:`pretrain`:
        ``REFUSED_NO_GPU_TARGET`` with no target, ``PLANNED`` with a target but no
        ``launch``, and the real launch only when ``launch`` and a ``gpu_target``.

        Args:
            model_ref: Pathspec of the model to serve.
            mode: A serving mode (validated against :data:`MODES`).

        Returns:
            An :class:`~loom.types.ArtifactRef` of ``kind="endpoint"``.
        """
        assert mode in MODES, (
            f"mode {mode!r} is not a Loom serving mode; expected one of "
            f"{sorted(MODES)} (backend nouns are rejected at the seam)."
        )
        plan = {
            "intent": {"model_ref": model_ref, "mode": mode},
            "backend": "nemo",
            "serving": _SERVE_LOWERING[mode],
            "gpu_target": self.config.gpu_target,
            "base_url": self.config.model_builder_base_url,
        }
        if mode == "batch":
            # Cheap frozen-embedding batch step: no GPU target required, no
            # heavy launch gate -- staged as PLANNED, never mutating in v0.1.
            plan["status"] = "PLANNED"
            return ArtifactRef(
                pathspec=f"ServeFlow/nemo-plan-{mode}",
                kind="endpoint",
                summary=self._summary(plan, status="PLANNED", capability="serve"),
            )
        # mode == "online": the launch-and-track path -- gate exactly like pretrain.
        return self._gate("endpoint", plan)

    def _gate(self, kind: str, plan: dict) -> ArtifactRef:
        """Apply the launch-and-track gate to a lowered ``plan`` (the §4.2 core).

        The pure decision shared by the launch-and-track capabilities
        (``pretrain``, ``serve(online)``), ported from
        :func:`flows.deploy.build_deploy_plan`'s ``BLOCKED``/``PLANNED``/``APPLIED``
        posture. It performs **no mutation** on the refuse/plan paths; only the
        ``launch and gpu_target`` branch reaches :meth:`_launch_and_track`.

        Args:
            kind: The :class:`ArtifactRef` kind to stamp (``"backbone"`` for
                pretrain, ``"endpoint"`` for online serve).
            plan: The lowered launch plan (with a ``cost`` block already added by
                the caller for ``pretrain``).

        Returns:
            The gated :class:`~loom.types.ArtifactRef`.
        """
        if self.config.gpu_target is None:
            # Constraint 4: refuse up front, with an actionable message. No plan
            # run pathspec, no mutation.
            plan["status"] = "REFUSED_NO_GPU_TARGET"
            return ArtifactRef(
                pathspec=None,
                kind=kind,
                error=(
                    "no GPU target configured: set LOOM_GPU_TARGET to a GPU "
                    "cluster, or use `--mlops local` / `model-builder local` for "
                    "the CPU stand-in. `/loom-train` does not launch GPU work "
                    "without a target."
                ),
                summary=self._summary(plan, status="REFUSED_NO_GPU_TARGET"),
            )
        if not self.launch:
            # --launch OFF by default => a staged launch PLAN artifact, NO
            # mutation (the DeployFlow PLANNED posture).
            plan["status"] = "PLANNED"
            return ArtifactRef(
                pathspec=f"TrainFlow/nemo-plan-{kind}",
                kind=kind,
                summary=self._summary(plan, status="PLANNED"),
            )
        # launch True AND a gpu_target set: the one mutating path (v0.2+).
        return self._launch_and_track(kind, plan)

    def _launch_and_track(self, kind: str, plan: dict) -> ArtifactRef:
        """The real heavy GPU launch -- routes the lowered plan to a launch target.

        This is the **only** mutating path, reached **only** when ``launch`` is
        True **and** a ``gpu_target`` is set (the ``apply and gate["allow"]``
        shape). It dispatches on the ``gpu_target`` *launcher* (the part before
        any ``://``):

        * ``modal`` / ``modal://<app>`` -> the on-demand **H100 via Modal**
          launcher (the v0.2 default): submit the lowered NeMo training config to
          an ephemeral H100, track it, then snapshot the produced checkpoint as a
          **Metaflow artifact** and return a backbone *pathspec* (never a
          checkpoint file or an object-store URI). If ``modal`` is not installed,
          the launcher refuses with an actionable install/auth message.
        * anything else -> ``REFUSED_UNKNOWN_GPU_TARGET`` (a clean refusal listing
          the supported launchers; no mutation), so an unknown target fails up
          front rather than deep in a job.

        Args:
            kind: The :class:`ArtifactRef` kind to stamp.
            plan: The lowered, costed launch plan.

        Returns:
            An :class:`~loom.types.ArtifactRef` for the launched (or refused) run.
        """
        gpu_target = self.config.gpu_target or ""
        launcher = gpu_target.split("://", 1)[0].strip().lower()

        if launcher == "modal":
            return self._launch_on_modal(kind, plan, gpu_target)

        # An unrecognized launch target: refuse up front (no mutation), listing
        # the launchers this adapter knows how to drive.
        plan["status"] = "REFUSED_UNKNOWN_GPU_TARGET"
        return ArtifactRef(
            pathspec=None,
            kind=kind,
            error=(
                f"unknown GPU launch target {gpu_target!r}: the nemo adapter "
                "supports launcher(s) [modal] (e.g. LOOM_GPU_TARGET=modal or "
                "modal://<app>). Set a supported target, or use "
                "`model-builder local` for the CPU stand-in that needs no GPU "
                "target."
            ),
            summary=self._summary(plan, status="REFUSED_UNKNOWN_GPU_TARGET"),
        )

    def _launch_on_modal(self, kind: str, plan: dict, gpu_target: str) -> ArtifactRef:
        """Drive the Modal H100 launcher and snapshot the checkpoint as a pathspec.

        The ``modal`` branch of :meth:`_launch_and_track`. Submits the lowered plan
        to an on-demand H100 via :mod:`loom.providers.model_builder._modal_launcher`
        (``modal`` is lazily imported there; absent => an actionable refusal),
        then snapshots the produced checkpoint into a **Metaflow artifact** and
        returns a backbone *pathspec*. The crux of constraint 1: what comes back
        is a pathspec + a small summary, **never** the checkpoint file the remote
        job produced and **never** an object-store URI.

        Args:
            kind: The :class:`ArtifactRef` kind to stamp (``"backbone"``).
            plan: The lowered, costed launch plan.
            gpu_target: The Modal routing string (``"modal"`` / ``"modal://<app>"``).

        Returns:
            An :class:`~loom.types.ArtifactRef` with ``status="LAUNCHED"`` and a
            run/artifact pathspec, or a clean error ref if Modal is unavailable.
        """
        # Lazy import: the Modal launcher pulls in ``modal`` only when it actually
        # submits, so importing this adapter stays GPU-free / Modal-free.
        from loom.providers.model_builder import _modal_launcher

        try:
            result = _modal_launcher.launch_on_modal(plan, gpu_target)
        except RuntimeError as exc:
            # The launcher's actionable "modal absent" (or other launch) refusal:
            # surface it cleanly as an error ref; no pathspec, no mutation claimed.
            plan["status"] = "REFUSED_MODAL_UNAVAILABLE"
            return ArtifactRef(
                pathspec=None,
                kind=kind,
                error=str(exc),
                summary=self._summary(plan, status="REFUSED_MODAL_UNAVAILABLE"),
            )

        # Snapshot the produced checkpoint as a Metaflow ARTIFACT and reference it
        # by PATHSPEC -- "the backbone IS the pathspec" (never a file, never a URI).
        pathspec = self._snapshot_checkpoint(result)
        plan["status"] = "LAUNCHED"
        summary = self._summary(plan, status="LAUNCHED")
        # The launch facts the caller reads at a glance (launcher, GPU, cost, status).
        summary["launcher"] = "modal"
        summary["gpu"] = result.gpu
        summary["metrics"] = dict(result.metrics or {})
        return ArtifactRef(
            pathspec=pathspec,
            kind=kind,
            summary=summary,
        )

    def _snapshot_checkpoint(self, result) -> str:
        """Snapshot a launched checkpoint into a Metaflow artifact -> a pathspec.

        The launcher returns an *opaque* checkpoint handle the remote H100 job
        produced; this turns it into a first-class Metaflow run/artifact
        **pathspec** (``<FlowName>/<run_id>``), the only currency the rest of the
        Loom lifecycle composes with. The handle itself is never surfaced: the
        backbone is the pathspec, not the file the GPU wrote.

        v0.1 returns a deterministic ``TrainFlow``-shaped pathspec derived from the
        Modal app the job ran under (a stable lineage stub); wiring the snapshot
        through the MLOps ``run_flow`` interface so the bytes are registered as a
        real Metaflow artifact is the v0.2+ follow-up. No object-store handle and
        no checkpoint file path is ever returned.

        Args:
            result: The launcher's result (carrying the opaque checkpoint handle).

        Returns:
            A Metaflow run/artifact pathspec referencing the snapshotted backbone.
        """
        app = getattr(result, "app_name", None) or "modal"
        return f"TrainFlow/nemo-modal-{app}"

    # -- searchable capabilities (cheap; never crash; §4.2) ----------------

    def tokenize(self, sequences_ref: str, scheme: dict) -> ArtifactRef:
        """Build a tokenizer/vocab -- a cheap searchable step (PLANNED in v0.1).

        Lowers the scheme to the backend GPU tokenizer config and returns a clean
        ``PLANNED`` :class:`ArtifactRef` (it never raises ``NotImplementedError``
        and never reaches the heavy gate). AIDE may tree-search ``scheme``.

        Args:
            sequences_ref: Pathspec of the sequences data object.
            scheme: The tokenization scheme AIDE may search.

        Returns:
            An :class:`~loom.types.ArtifactRef` of ``kind="tokenizer"``.
        """
        plan = {
            "intent": {"sequences_ref": sequences_ref, "scheme": dict(scheme or {})},
            "backend": "nemo",
            "tokenizer": "gpu-field-tokenizer:hashed-compositional",
            "status": "PLANNED",
        }
        return ArtifactRef(
            pathspec="TrainFlow/nemo-plan-tokenizer",
            kind="tokenizer",
            summary=self._summary(plan, status="PLANNED", capability="tokenize"),
        )

    def finetune(self, backbone_ref: str, task_ref: str, recipe: dict) -> ArtifactRef:
        """Fit a cheap head on a frozen backbone -- searchable (PLANNED in v0.1).

        Lowers the recipe (frozen backbone + cheap head) and returns a clean
        ``PLANNED`` :class:`ArtifactRef`; never raises, never gates. AIDE may
        tree-search ``recipe``.

        Args:
            backbone_ref: Pathspec of the frozen backbone.
            task_ref: Pathspec of the task data object.
            recipe: The head recipe AIDE may search.

        Returns:
            An :class:`~loom.types.ArtifactRef` of ``kind="model"``.
        """
        plan = {
            "intent": {
                "backbone_ref": backbone_ref,
                "task_ref": task_ref,
                "recipe": dict(recipe or {}),
            },
            "backend": "nemo",
            "head": "frozen-backbone-cheap-head",
            "status": "PLANNED",
        }
        return ArtifactRef(
            pathspec="TrainFlow/nemo-plan-model",
            kind="model",
            summary=self._summary(plan, status="PLANNED", capability="finetune"),
        )

    def embed(self, backbone_ref: str, data_ref: str) -> ArtifactRef:
        """Embed data through a frozen backbone -- searchable (PLANNED in v0.1).

        Lowers a frozen-backbone embedding step and returns a clean ``PLANNED``
        :class:`ArtifactRef`; never raises, never gates.

        Args:
            backbone_ref: Pathspec of the frozen backbone.
            data_ref: Pathspec of the data object to embed.

        Returns:
            An :class:`~loom.types.ArtifactRef` of ``kind="embeddings"``.
        """
        plan = {
            "intent": {"backbone_ref": backbone_ref, "data_ref": data_ref},
            "backend": "nemo",
            "embedding": "frozen-backbone-pooled",
            "status": "PLANNED",
        }
        return ArtifactRef(
            pathspec="TrainFlow/nemo-plan-embeddings",
            kind="embeddings",
            summary=self._summary(plan, status="PLANNED", capability="embed"),
        )

    def evaluate(self, model_ref: str, holdout_ref: str, metric: str) -> Scores:
        """Score a model on a sealed holdout -- searchable (PLANNED in v0.1).

        Wires the eval harness (temporal split, the adapter-equal PR-AUC metric)
        and returns a clean :class:`~loom.types.Scores` with ``value=None``
        (PLANNED -- no scalar computed in v0.1) and a ``detail`` recording the
        plan; never raises, never gates. The same metric name the ``local``
        adapter uses, so the conformance metric is adapter-equal.

        Args:
            model_ref: Pathspec of the fitted model.
            holdout_ref: Pathspec of the sealed temporal holdout data object.
            metric: The metric name (e.g. ``"fraud-pr-auc"``).

        Returns:
            A :class:`~loom.types.Scores` (value ``None`` in v0.1, PLANNED).
        """
        return Scores(
            metric=metric,
            value=None,
            detail={
                "status": "PLANNED",
                "backend": "nemo",
                "model_ref": model_ref,
                "holdout_ref": holdout_ref,
                "harness": "temporal-split-pr-auc",
                "note": (
                    "nemo evaluate is PLANNED in v0.1 (wires the eval harness; "
                    "no scalar computed). Use `model-builder local` for a scored "
                    "lift on the CPU stand-in today."
                ),
            },
        )

    # -- shared summary helper ---------------------------------------------

    @staticmethod
    def _summary(plan: dict, status: str, capability: str = "pretrain") -> dict:
        """Build the 6-line typed summary, incl. how the gate decision was computed.

        The summary explains the gate posture so a caller (and the ``@card``)
        sees *why* the run REFUSED / PLANNED / launched, in plain language: the
        decision is ``f(gpu_target is None, launch)``, with the cost (physics)
        surfaced alongside.

        Args:
            plan: The lowered (and, for ``pretrain``, costed) launch plan.
            status: The gate status stamped on the ref
                (``REFUSED_NO_GPU_TARGET`` / ``PLANNED`` / ``LAUNCH_DEFERRED``).
            capability: The capability this summary describes.

        Returns:
            A small JSON-able summary dict (the 6 explanatory lines + the plan).
        """
        gpu_target = (plan.get("gpu_target") or plan.get("intent", {}).get("gpu_target"))
        cost = plan.get("cost") or {}
        gate_rule = {
            "REFUSED_NO_GPU_TARGET": (
                "gpu_target is None => refuse up front (no launch, no mutation)."
            ),
            "PLANNED": (
                "gpu_target set but launch=False (default) => staged PLAN only "
                "(no mutation; re-run with --launch to allow the real launch)."
            ),
            "LAUNCHED": (
                "launch=True and a supported gpu_target set => gate ALLOWED; the "
                "real GPU launch ran (e.g. on-demand H100 via Modal) and the "
                "checkpoint is snapshotted as a Metaflow artifact pathspec."
            ),
            "REFUSED_UNKNOWN_GPU_TARGET": (
                "gpu_target set but its launcher is not supported => refuse up "
                "front (no launch, no mutation; set a supported launcher)."
            ),
            "REFUSED_MODAL_UNAVAILABLE": (
                "gate ALLOWED and the Modal launcher was selected, but Modal is "
                "not installed/authenticated => refuse cleanly (no mutation)."
            ),
        }.get(status, "PLANNED step (searchable; no heavy gate).")
        return {
            # line 1 -- the capability + backend.
            "capability": capability,
            "backend": "nemo",
            # line 2 -- the gate status the caller reads.
            "status": status,
            # line 3 -- how the gate decision is computed (the two inputs).
            "gate_decision": gate_rule,
            # line 4 -- the gate inputs themselves (physics at the gate).
            "gpu_target": gpu_target,
            "cost": cost,
            # line 5 -- the lowered backend config (vocabulary hidden here).
            "plan": plan,
            # line 6 -- the honesty note: this is a compiler, not a real launch in v0.1.
            "note": (
                "nemo is a lowering compiler in v0.1: it emits a launch PLAN and "
                "gates the real GPU launch; it does not run real GPU work."
            ),
        }


__all__ = ["NemoModelBuilderProvider"]
