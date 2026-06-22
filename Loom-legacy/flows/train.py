"""Loom's model-training Metaflow flow -- the ModelBuilderProvider seam + the gate.

This module defines the single static ``FlowSpec`` -- :class:`TrainFlow` -- that the
``loom train`` command runs (via the Metaflow MLOps interface) to **build a model**
stated in Loom DS-intent vocabulary (``objective`` / ``budget`` / ``backbone`` /
``metric``). It is the flow side of the third heavy backend, the
:class:`~loom.providers.ModelBuilderProvider` port: the *training backend*
(``local`` PPMI+SVD stand-in / ``nemo`` lowering compiler) is resolved **inside**
the ``build`` step from ``config.model_builder_provider``, while the whole flow runs
*through* the MLOps provider's ``run_flow`` seam -- two orthogonal, config-only swap
axes (design §6.1). The skill speaks the interface; it never names NeMo.

Train is the **EXPENSIVE / MUTATE, ALWAYS-GATE** tier of the approval matrix
(design §6.3): the ``pretrain`` capability is manifest-typed ``launch-and-track``
(AIDE never tree-searches it), and the real heavy GPU launch is **OFF by default**
behind an explicit ``launch`` flag that mirrors :class:`flows.deploy.DeployFlow`'s
``--apply`` posture. With ``launch=False`` (the default) the ``nemo`` adapter
returns a staged ``PLANNED`` :class:`~loom.types.ArtifactRef`; with ``gpu_target``
unset it returns the clean ``REFUSED_NO_GPU_TARGET`` refusal; the torch-free
``local`` adapter actually produces the backbone / embeddings artifacts on CPU. The
abstraction hides **vocabulary, not physics**: the cost PLAN (hours / $ / GPU-count
for the chosen ``budget``) is surfaced at the GATE.

The input is a **Metaflow data object** referenced by ``dataset_ref`` (a pathspec
like ``"IngestDataset/123"`` produced by ``loom ingest``). The ``start`` step reads
its recorded schema through the Metaflow **Client API** only
(:func:`loom.dataio.dataset_schema`); the builder's capability methods materialize
the data the same way. Loom never touches the underlying datastore -- that is
Metaflow's concern. Outputs are :class:`~loom.types.ArtifactRef` carrying Metaflow
run **pathspecs**, never on-disk checkpoint files or object-store URIs.

Flow shape::

    start --> plan --> build --> end

* ``start`` -- validate ``objective`` / ``budget`` / ``capability`` against the
               module-level frozensets (refuse a backend noun at the seam, early),
               and read the ``dataset_ref`` data object's recorded schema via the
               Client API. READ-ONLY over the data object.
* ``plan`` (``@card``) -- resolve ``builder = get_model_builder(provider)(config)``,
               read its ``manifest()``, **assert the chosen capability's mode is
               ``launch-and-track``** (a ``searchable`` capability is redirected to
               ``/loom-optimize``), build the cost PLAN (physics at the gate), and
               render the GATE ``@card`` exactly as ``DeployFlow._render_card``
               renders the apply/gate posture.
* ``build`` (the heavy step) -- call the resolved capability through the builder
               (``pretrain`` / ``tokenize`` / ``finetune`` / ``embed``); persist
               ``self.artifact_ref`` + a small ``self.summary`` the MLOps interface
               reads back, and -- when the capability emits a produced data object
               (the ``local`` ``embed`` path) -- write its ``train`` / ``test`` /
               ``schema`` / ``fingerprint`` artifacts on ``self`` (the
               ``FeaturesFlow`` write-a-new-data-object pattern), so the produced
               run's pathspec is itself a first-class ``dataset_ref``.
* ``end`` -- carry ``self.summary`` forward so ``Run.data.summary`` exposes it.

Only standard Metaflow APIs are used (``FlowSpec``, ``@step``, ``Parameter``,
``@card``, and the ``current.card`` append API). ``loom`` (and any numeric library)
is imported *inside* the steps so the flow file parses even where those are not yet
importable until the Runner subprocess sets up the environment.
"""

from __future__ import annotations

from typing import Any

from metaflow import FlowSpec, Parameter, card, current, step

#: The two AIDE-search modes a capability may declare. ``/loom-train`` only invokes
#: ``launch-and-track`` capabilities (``pretrain``; ``serve(online)`` for ``nemo``);
#: a ``searchable`` capability (``tokenize`` / ``finetune`` / ``embed`` /
#: ``evaluate``) is redirected to ``/loom-optimize`` (the §6.4 mode split). This is a
#: provider fact read from ``manifest().mode_of(capability)``, never user knowledge.
_LAUNCH_AND_TRACK = "launch-and-track"


class TrainFlow(FlowSpec):
    """Build a model through the ModelBuilderProvider seam, gated like a deploy.

    Reads the ``dataset_ref`` data object's schema via the Client API, resolves the
    configured model-builder backend inside the ``build`` step, computes the cost
    PLAN + GATE posture (rendered as an ``@card``), invokes the chosen
    ``launch-and-track`` capability, and emits a Metaflow run + an ``@card``. The
    typed summary is carried on ``self.summary`` so the MLOps interface reads it back
    from ``Run.data``; when the capability emits a produced data object (the ``local``
    ``embed`` path) its ``train`` / ``test`` / ``schema`` / ``fingerprint`` artifacts
    are written on ``self`` so the produced pathspec is a first-class ``dataset_ref``.

    EXPENSIVE / MUTATE tier -- always gated; the real heavy GPU launch is OFF by
    default behind ``launch`` (the ``DeployFlow.apply`` posture) and refuses cleanly
    when ``gpu_target`` is unset. The skill sets ``disable-model-invocation: true``.
    """

    #: Metaflow **pathspec** of the sequences/ingested data object to build from
    #: (e.g. ``"IngestDataset/123"``, produced by ``loom ingest`` / ``loom
    #: features``). Read via the Client API only; Loom never touches the datastore.
    dataset_ref = Parameter(
        "dataset_ref",
        required=True,
        type=str,
        help="Metaflow pathspec of the sequences data object (e.g. IngestDataset/123).",
    )

    #: The capability to invoke. ``/loom-train`` only invokes ``launch-and-track``
    #: capabilities (``pretrain``; ``serve`` for ``nemo``); a ``searchable``
    #: capability is refused with "use `/loom-optimize`" (the §6.4 mode split).
    capability = Parameter(
        "capability",
        default="pretrain",
        type=str,
        help="Capability to build: pretrain | tokenize | finetune | embed | serve.",
    )

    #: The pretraining objective (validated against :data:`loom.providers.OBJECTIVES`
    #: at the seam; a backend noun is rejected here, in Loom vocabulary).
    objective = Parameter(
        "objective",
        default="next-event",
        type=str,
        help="Pretraining objective: next-event | masked-field | contrastive.",
    )

    #: The training budget (validated against :data:`loom.providers.BUDGETS`). The
    #: budget is PHYSICS -- the cost PLAN (hours / $ / GPU-count) is surfaced at the
    #: gate from it, never faked away.
    budget = Parameter(
        "budget",
        default="probe",
        type=str,
        help="Training budget: probe | small | full (physics, surfaced at the gate).",
    )

    #: Pathspec of a frozen backbone to build on, for ``finetune`` / ``embed``
    #: (e.g. a prior ``TrainFlow`` pretrain run). Empty for ``pretrain`` / ``tokenize``.
    backbone_ref = Parameter(
        "backbone_ref",
        default="",
        type=str,
        help="Pathspec of a frozen backbone for finetune/embed (e.g. TrainFlow/12).",
    )

    #: The evaluation metric the build is steered toward (recorded in the summary;
    #: the adapter-equal ``fraud-pr-auc`` is the default the ``evaluate`` capability
    #: wires the same way across backends).
    metric = Parameter(
        "metric",
        default="fraud-pr-auc",
        type=str,
        help="Evaluation metric (e.g. fraud-pr-auc).",
    )

    #: Whether to perform the real heavy GPU launch. **OFF by default** -- the
    #: default run produces a PLAN (``local`` actually builds on CPU; ``nemo`` stages
    #: a ``PLANNED`` plan, or ``REFUSED_NO_GPU_TARGET`` with no target). The external
    #: launch runs ONLY when ``launch`` is True AND a ``gpu_target`` is set (the
    #: ``DeployFlow.apply`` posture, ported to the model-builder seam).
    launch = Parameter(
        "launch",
        default=False,
        type=bool,
        help="Perform the real heavy GPU launch (OFF by default; needs a gpu_target).",
    )

    @step
    def start(self) -> None:
        """Validate the intent at the seam and read the data object's schema.

        Validates ``objective`` / ``budget`` / ``capability`` against the
        module-level frozensets (:data:`loom.providers.OBJECTIVES` /
        :data:`~loom.providers.BUDGETS`) -- a backend noun is rejected here, in Loom
        vocabulary, before any builder sees it (constraint 2) -- and reads the
        ``dataset_ref`` data object's recorded schema through the Client API
        (:func:`loom.dataio.dataset_schema`). READ-ONLY over the data object; raises
        early with an actionable message rather than failing deep in the build.
        """
        from loom.dataio import dataset_schema
        from loom.providers import BUDGETS, OBJECTIVES

        ref = (self.dataset_ref or "").strip()
        if not ref:
            raise ValueError(
                "dataset_ref is empty; expected a Metaflow pathspec like "
                "'IngestDataset/123' (run `loom ingest` to produce one)."
            )

        capability = (self.capability or "").strip()
        objective = (self.objective or "").strip()
        budget = (self.budget or "").strip()

        # Seam validation in Loom vocabulary (refuse a NeMo noun up front). The
        # capability itself is checked against the resolved manifest in ``plan``.
        if objective not in OBJECTIVES:
            raise ValueError(
                f"objective {objective!r} is not a Loom objective; expected one of "
                f"{sorted(OBJECTIVES)} (backend nouns are rejected at the seam)."
            )
        if budget not in BUDGETS:
            raise ValueError(
                f"budget {budget!r} is not a Loom budget; expected one of "
                f"{sorted(BUDGETS)} (backend nouns are rejected at the seam)."
            )

        # Read the data object's recorded schema (read-only; the builder's capability
        # methods materialize the rows the same way through the Client API).
        try:
            schema = dataset_schema(ref)
        except Exception:  # pragma: no cover - schema read edge case
            schema = {}
        self._schema = dict(schema or {})
        self._resolved_capability = capability
        self.next(self.plan)

    @card
    @step
    def plan(self) -> None:
        """Resolve the builder, assert the mode, build the cost PLAN, render the GATE.

        Resolves ``builder = get_model_builder(config.model_builder_provider)(config)``
        from the active :class:`~loom.config.LoomConfig`, reads its ``manifest()``,
        and **asserts the chosen capability's mode is ``launch-and-track``** -- a
        ``searchable`` capability is redirected to ``/loom-optimize`` (the §6.4 mode
        split; AIDE never tree-searches a launch-and-track capability). Builds the
        cost PLAN (hours / $ / GPU-count for the budget -- physics at the gate) and
        renders the GATE ``@card`` (plan, gate decision, lineage), exactly as
        :meth:`flows.deploy.DeployFlow._render_card` renders the apply/gate posture.
        Stores the plan context on ``self`` for the ``build`` step.
        """
        from loom.config import LoomConfig
        from loom.registry import get_model_builder

        config = LoomConfig.load()
        self._model_builder_provider = config.model_builder_provider
        self._gpu_target = config.gpu_target

        capability = self._resolved_capability

        # Resolve the builder + its manifest (the negotiation contract).
        builder = get_model_builder(config.model_builder_provider)(config)
        manifest = builder.manifest()
        self._backend = manifest.backend

        supported = manifest.supports(capability)
        mode = manifest.mode_of(capability)
        self._capability_mode = mode

        # The §6.4 mode gate: /loom-train only invokes launch-and-track capabilities.
        # A searchable capability is redirected to /loom-optimize; an unknown/
        # unsupported capability is refused up front (never failing deep in a job).
        if mode is None or not supported:
            self._mode_gate = {
                "allow": False,
                "reason": (
                    f"capability {capability!r} is not a supported launch-and-track "
                    f"capability of backend {manifest.backend!r}; the {manifest.backend!r} "
                    f"manifest declares {sorted(manifest.capabilities)}."
                ),
            }
        elif mode != _LAUNCH_AND_TRACK:
            self._mode_gate = {
                "allow": False,
                "reason": (
                    f"capability {capability!r} is mode {mode!r} (searchable), not "
                    f"{_LAUNCH_AND_TRACK!r}; /loom-train only invokes launch-and-track "
                    "capabilities -- use `/loom-optimize` to tree-search this cheap "
                    "scalar (head / tokenization / embed / evaluate)."
                ),
            }
        else:
            self._mode_gate = {"allow": True, "reason": ""}

        # The cost PLAN -- physics surfaced at the gate (hours / $ / GPU-count for the
        # chosen budget). Prefer the backend's own estimator when it exposes one
        # (the nemo adapter's _estimate_cost); otherwise a neutral CPU-stand-in cost.
        self._cost = self._estimate_cost(builder, self.budget)

        # The launch posture (the DeployFlow apply/gate vocabulary): the real heavy
        # launch is OFF unless ``launch`` AND a ``gpu_target`` is set. The actual
        # gate decision per capability is the adapter's (nemo refuses with no target);
        # this is the up-front posture the card narrates.
        launch = bool(self.launch)
        if self._gpu_target is None and launch:
            launch_posture = "REQUESTED_NO_GPU_TARGET"
        elif launch:
            launch_posture = "LAUNCH_REQUESTED"
        else:
            launch_posture = "PLAN_ONLY"
        self._launch_posture = launch_posture

        self._plan = {
            "dataset_ref": (self.dataset_ref or "").strip(),
            "backend": manifest.backend,
            "model_builder_provider": config.model_builder_provider,
            "capability": capability,
            "capability_mode": mode,
            "objective": (self.objective or "").strip(),
            "budget": (self.budget or "").strip(),
            "backbone_ref": (self.backbone_ref or "").strip() or None,
            "metric": (self.metric or "").strip(),
            "launch": launch,
            "launch_posture": launch_posture,
            "gpu_target": self._gpu_target,
            "mode_gate": self._mode_gate,
            "cost": self._cost,
        }

        self._render_card(self._plan)
        self.next(self.build)

    @staticmethod
    def _estimate_cost(builder: Any, budget: str) -> dict:
        """Build the cost PLAN -- physics at the gate (hours / $ / GPU-count).

        Prefers the backend's own ``_estimate_cost(budget)`` when it exposes one
        (the ``nemo`` adapter, which owns the real GPU-hour physics for the gate);
        otherwise returns a neutral CPU-stand-in cost so the ``@card`` always carries
        a legible cost line (the ``local`` adapter is cheap CPU work, surfaced
        honestly rather than faked). The abstraction hides vocabulary, not physics.

        Args:
            builder: The resolved model-builder provider.
            budget: The Loom budget (validated at the seam).

        Returns:
            A JSON-able cost dict with at least ``gpu_count`` / ``wall_clock_hours``
            / ``gpu_hours`` / ``est_usd`` / ``headline``.
        """
        estimator = getattr(builder, "_estimate_cost", None)
        if callable(estimator):
            try:
                cost = estimator(budget)
                if isinstance(cost, dict):
                    return cost
            except Exception:  # pragma: no cover - fall back to the neutral cost
                pass
        # Neutral CPU stand-in cost: the ``local`` adapter runs sub-2s on one CPU,
        # no GPU and no dollar cost. Surfaced honestly (not faked) so the gate card
        # still shows the physics line for every backend.
        return {
            "gpu_count": 0,
            "wall_clock_hours": 0.0,
            "gpu_hours": 0.0,
            "usd_per_gpu_hour": 0.0,
            "est_usd": 0.0,
            "headline": (
                f"budget={budget}: CPU stand-in, 0 GPU "
                "(local PPMI+SVD; sub-2s, no GPU-hours)"
            ),
        }

    @step
    def build(self) -> None:
        """Invoke the chosen capability; persist the typed ref + summary + data object.

        Resolves the builder (threading ``launch`` to the adapter when it accepts
        it -- the ``nemo`` adapter's ``launch`` constructor arg, OFF by default) and,
        when the mode gate allowed, calls the chosen ``launch-and-track`` capability:
        ``pretrain(dataset_ref, objective, budget)`` (or ``serve(model_ref, mode)``).
        With ``launch=False`` the ``nemo`` adapter returns a ``PLANNED``
        :class:`~loom.types.ArtifactRef`; with ``gpu_target`` unset it returns the
        ``REFUSED_NO_GPU_TARGET`` ref; the ``local`` adapter actually produces the
        backbone (and, for ``embed``, an ``IngestDataset``-shaped data object).

        Persists ``self.artifact_ref`` (a JSON-able dict) + a small ``self.summary``
        the MLOps interface reads back. When the capability emits a produced data
        object, its ``train`` / ``test`` / ``schema`` / ``fingerprint`` artifacts are
        written on ``self`` (the ``FeaturesFlow`` pattern) so the produced run's
        pathspec is a first-class ``dataset_ref``; ``self.backbone`` /
        ``self.backbone_vocab`` are written for a ``pretrain`` so a downstream
        ``embed``/``finetune`` can load the frozen backbone via the Client API.
        """
        from loom.config import LoomConfig
        from loom.registry import get_model_builder

        config = LoomConfig.load()
        builder = self._make_builder(get_model_builder, config)

        capability = self._resolved_capability
        dataset_ref = (self.dataset_ref or "").strip()
        backbone_ref = (self.backbone_ref or "").strip() or dataset_ref
        objective = (self.objective or "").strip()
        budget = (self.budget or "").strip()
        metric = (self.metric or "").strip()

        # The mode gate (from ``plan``): /loom-train refuses a non-launch-and-track
        # capability up front rather than invoking it (the §6.4 split).
        if not self._mode_gate.get("allow"):
            artifact_ref = {
                "pathspec": None,
                "kind": self._kind_for(capability),
                "summary": {"status": "REFUSED_NOT_LAUNCH_AND_TRACK"},
                "error": self._mode_gate.get("reason"),
            }
            status = "REFUSED_NOT_LAUNCH_AND_TRACK"
            self.artifact_ref = artifact_ref
            self.summary = self._build_summary(artifact_ref, status, builder)
            self.next(self.end)
            return

        # Invoke the resolved launch-and-track capability through the builder. The
        # adapter owns the gate (nemo: REFUSED_NO_GPU_TARGET / PLANNED / launch);
        # the local adapter actually builds. A capability that emits a produced data
        # object writes its artifacts on ``self`` below.
        produced = None
        try:
            if capability == "pretrain":
                ref = builder.pretrain(dataset_ref, objective, budget)
                produced = self._maybe_persist_backbone(builder, dataset_ref, ref)
            elif capability == "serve":
                # serve is launch-and-track for the nemo backend (online => a gated
                # microservice). Default mode "online" exercises the gate; a backend
                # that only serves batch declares serve searchable and is refused
                # by the mode gate above.
                ref = builder.serve(backbone_ref, "online")
            elif capability == "tokenize":
                ref = builder.tokenize(dataset_ref, {})
            elif capability == "finetune":
                ref = builder.finetune(backbone_ref, dataset_ref, {})
            elif capability == "embed":
                ref = builder.embed(backbone_ref, dataset_ref)
                produced = self._maybe_persist_dataset(builder, backbone_ref, dataset_ref)
            else:  # pragma: no cover - the mode gate already refused unknowns
                ref = builder.pretrain(dataset_ref, objective, budget)
        except NotImplementedError as exc:
            # A capability-gap refusal at call time (e.g. serve(online) on local):
            # record it as a clean errored ref rather than crashing the run.
            artifact_ref = {
                "pathspec": None,
                "kind": self._kind_for(capability),
                "summary": {"status": "REFUSED_UNSUPPORTED"},
                "error": str(exc),
            }
            self.artifact_ref = artifact_ref
            self.summary = self._build_summary(
                artifact_ref, "REFUSED_UNSUPPORTED", builder
            )
            self.next(self.end)
            return

        # Stamp the real produced-run pathspec onto a non-error ref (the standalone
        # adapter echoes the source run id; inside this flow the produced data object
        # IS this run, so its pathspec is the RUN-level ``<FlowName>/<run_id>`` --
        # NOT ``current.pathspec``, which is the task-level four-part pathspec the
        # Client API's ``resolve_run`` would reject).
        artifact_ref = self._ref_to_dict(ref)
        if produced is not None and artifact_ref.get("pathspec"):
            artifact_ref["pathspec"] = f"{current.flow_name}/{current.run_id}"

        status = self._status_of(artifact_ref)
        self.artifact_ref = artifact_ref
        self.summary = self._build_summary(artifact_ref, status, builder)
        self._render_build_card(artifact_ref, status)
        self.next(self.end)

    def _make_builder(self, get_model_builder: Any, config: Any) -> Any:
        """Construct the model-builder, threading ``launch`` when the adapter accepts it.

        Every Loom provider is constructed ``Provider(config)``; the ``nemo`` adapter
        additionally accepts ``launch=`` (the per-run ``--launch`` intent the
        ``DeployFlow.apply``-style flag threads). The ``local`` adapter does not. We
        pass ``launch`` only when the constructor accepts it so both adapters stay
        uniform at the seam and the locked ABC signatures are unchanged.

        Args:
            get_model_builder: The registry getter (imported in the step).
            config: The active :class:`~loom.config.LoomConfig`.

        Returns:
            The constructed model-builder provider.
        """
        import inspect

        cls = get_model_builder(config.model_builder_provider)
        try:
            params = inspect.signature(cls.__init__).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtin __init__
            params = {}
        if "launch" in params:
            return cls(config, launch=bool(self.launch))
        return cls(config)

    def _maybe_persist_backbone(
        self, builder: Any, sequences_ref: str, ref: Any
    ) -> Any:
        """Write a pretrain's backbone artifacts on ``self`` for downstream loading.

        For a non-error ``local`` ``pretrain`` ref, recompute the deterministic
        backbone (``W`` + vocab) the adapter's pure helpers produce and write
        ``self.backbone`` / ``self.backbone_vocab`` / ``self.fingerprint`` so a later
        ``embed`` / ``finetune`` can load the frozen backbone via the Client API
        (``LocalModelBuilderProvider._load_backbone`` reads exactly those artifacts).
        Best-effort and adapter-shaped: a backend with no such helpers (the ``nemo``
        PLAN path) simply writes nothing and the produced ref stays a plan pathspec.

        Args:
            builder: The resolved builder.
            sequences_ref: The sequences data object pathspec.
            ref: The :class:`~loom.types.ArtifactRef` the ``pretrain`` returned.

        Returns:
            A truthy marker when backbone artifacts were written (so the caller
            stamps ``current.pathspec``), else ``None``.
        """
        if getattr(ref, "error", None) or getattr(ref, "pathspec", None) is None:
            return None
        # Only the torch-free ``local`` adapter exposes the pure backbone helpers; a
        # PLAN-only backend (nemo) carries no real backbone matrix in v0.1.
        try:
            from loom.providers.model_builder import local as _local
        except Exception:  # pragma: no cover - local adapter absent
            return None
        if not isinstance(builder, _local.LocalModelBuilderProvider):
            return None

        train, schema = builder._materialize(sequences_ref)
        resolved = _local.resolve_scheme(None, schema)
        vocab = _local.build_vocab(train, resolved)
        sequences = _local.encode_sequences(train, vocab)
        budget = (self.budget or "").strip()
        dim = _local._BUDGET_DIMS.get(budget, _local._BUDGET_DIMS["probe"])
        C = _local.build_cooccurrence(sequences, vocab["size"], (self.objective or "").strip())
        W = _local.factorize_backbone(_local.ppmi(C), dim, random_state=_local._RANDOM_STATE)

        self.backbone = W
        self.backbone_vocab = vocab
        self.fingerprint = _local.backbone_fingerprint(W, vocab)
        return True

    def _maybe_persist_dataset(
        self, builder: Any, backbone_ref: str, data_ref: str
    ) -> Any:
        """Write an embed's produced data object (``train``/``test``/``schema``) on ``self``.

        For the ``local`` ``embed`` capability, recompute the ``IngestDataset``-shaped
        embedding frame the adapter's pure helpers produce and write
        ``self.train`` / ``self.test`` / ``self.schema`` / ``self.fingerprint`` on
        ``self`` -- the ``FeaturesFlow`` write-a-new-data-object pattern -- so this
        run's pathspec round-trips through :func:`loom.dataio.materialize_dataset`
        unchanged and is a first-class ``dataset_ref`` any ``/loom-*`` verb consumes.
        Best-effort: a PLAN-only backend (nemo) writes nothing and the produced ref
        stays a plan pathspec.

        Args:
            builder: The resolved builder.
            backbone_ref: The frozen backbone pathspec to embed through.
            data_ref: The data object pathspec to embed.

        Returns:
            A truthy marker when data-object artifacts were written, else ``None``.
        """
        try:
            from loom.providers.model_builder import local as _local
        except Exception:  # pragma: no cover - local adapter absent
            return None
        if not isinstance(builder, _local.LocalModelBuilderProvider):
            return None

        train, schema = builder._materialize(data_ref)
        W, vocab = builder._load_backbone(backbone_ref, train, schema)
        target = schema.get("target")
        dataset = _local.build_embedding_dataset(train, W, vocab, target)

        self.train = dataset["train"]
        self.test = dataset["test"]
        self.schema = dataset["schema"]
        self.fingerprint = dataset["fingerprint"]
        self.dataset_name = f"embeddings:{data_ref}"
        return True

    @staticmethod
    def _ref_to_dict(ref: Any) -> dict:
        """Convert an :class:`~loom.types.ArtifactRef` to a JSON-able dict."""
        return {
            "pathspec": getattr(ref, "pathspec", None),
            "kind": getattr(ref, "kind", None),
            "summary": dict(getattr(ref, "summary", {}) or {}),
            "error": getattr(ref, "error", None),
        }

    @staticmethod
    def _status_of(artifact_ref: dict) -> str:
        """Derive the headline status line for a produced artifact ref.

        Prefers the adapter's own ``summary["status"]`` (the ``nemo`` gate's
        ``PLANNED`` / ``REFUSED_NO_GPU_TARGET`` / ``LAUNCH_DEFERRED``); falls back to
        ``ERROR`` when the ref carries an error with no status, else ``BUILT`` (the
        ``local`` adapter actually produced an artifact).
        """
        summary = artifact_ref.get("summary") or {}
        status = summary.get("status")
        if status:
            return str(status)
        if artifact_ref.get("error"):
            return "ERROR"
        return "BUILT"

    @staticmethod
    def _kind_for(capability: str) -> str:
        """The :class:`ArtifactRef` kind a capability produces (for refusal refs)."""
        return {
            "pretrain": "backbone",
            "tokenize": "tokenizer",
            "finetune": "model",
            "embed": "embeddings",
            "serve": "endpoint",
        }.get(capability, "backbone")

    def _build_summary(self, artifact_ref: dict, status: str, builder: Any) -> dict:
        """Build the small, JSON-able typed summary the MLOps interface reads back.

        Carries only references + small derived scalars (the produced pathspec, the
        gate status, the backend, the cost line) -- never raw rows or secrets. The
        ``status`` is the headline line a command narrates (PLAN / PLANNED /
        REFUSED_NO_GPU_TARGET / BUILT).

        Args:
            artifact_ref: The produced artifact-ref dict.
            status: The headline status line.
            builder: The resolved builder (for the backend name).

        Returns:
            The summary dict (read back via the ``summary`` artifact name).
        """
        ref_summary = artifact_ref.get("summary") or {}
        return {
            "dataset_ref": (self.dataset_ref or "").strip(),
            "backend": getattr(self, "_backend", getattr(builder, "name", "?")),
            "model_builder_provider": getattr(self, "_model_builder_provider", None),
            "capability": self._resolved_capability,
            "capability_mode": getattr(self, "_capability_mode", None),
            "objective": (self.objective or "").strip(),
            "budget": (self.budget or "").strip(),
            "metric": (self.metric or "").strip(),
            "launch": bool(self.launch),
            "launch_posture": getattr(self, "_launch_posture", None),
            "gpu_target": getattr(self, "_gpu_target", None),
            "cost": getattr(self, "_cost", {}),
            "artifact_ref": artifact_ref,
            "artifact_pathspec": artifact_ref.get("pathspec"),
            "artifact_kind": artifact_ref.get("kind"),
            "fingerprint": ref_summary.get("fingerprint")
            or getattr(self, "fingerprint", None),
            "error": artifact_ref.get("error"),
            "status": status,
            # The headline VERDICT line downstream/CI reads (parallels DeployFlow):
            "verdict": status,
        }

    def _render_card(self, plan: dict) -> None:
        """Render the GATE ``@card`` (plan, gate decision, cost, lineage).

        Mirrors :meth:`flows.deploy.DeployFlow._render_card`: a Markdown header, the
        cost PLAN (physics at the gate), the mode/launch GATE decision, and the
        lineage. The card is rendered in ``plan`` (before the heavy ``build`` step)
        so the gate posture is legible whether or not the build then refuses.

        Args:
            plan: The plan dict assembled in :meth:`plan`.
        """
        from metaflow.cards import Markdown, Table

        cost = plan.get("cost") or {}
        mode_gate = plan.get("mode_gate") or {}

        current.card.append(Markdown("# Loom training plan"))
        current.card.append(
            Markdown(
                f"**dataset_ref:** `{plan.get('dataset_ref')}`  \n"
                f"**backend:** `{plan.get('backend')}` "
                f"(model_builder_provider `{plan.get('model_builder_provider')}`)  \n"
                f"**capability:** `{plan.get('capability')}` "
                f"(mode `{plan.get('capability_mode')}`)  \n"
                f"**objective / budget:** `{plan.get('objective')}` / "
                f"`{plan.get('budget')}`  \n"
                f"**launch:** {plan.get('launch')} "
                f"(real GPU launch {'ON' if plan.get('launch') else 'OFF — plan only'}; "
                f"posture `{plan.get('launch_posture')}`)  \n"
                f"**gpu_target:** `{plan.get('gpu_target')}`"
            )
        )

        # Cost PLAN -- physics surfaced at the gate (hours / $ / GPU-count).
        current.card.append(Markdown("## Cost plan (physics at the gate)"))
        current.card.append(
            Table(
                [
                    [
                        cost.get("gpu_count", "n/a"),
                        f"{cost.get('wall_clock_hours', 'n/a')}",
                        f"{cost.get('gpu_hours', 'n/a')}",
                        f"${cost.get('est_usd', 'n/a')}",
                    ]
                ],
                headers=["GPU count", "wall-clock h", "GPU-hours", "est. $"],
            )
        )
        current.card.append(Markdown(f"_{cost.get('headline', '')}_"))

        # Mode / launch GATE decision (the centerpiece: why train will/won't invoke).
        current.card.append(Markdown("## Gate decision"))
        if mode_gate.get("allow"):
            current.card.append(
                Markdown(
                    "_Mode GATE ALLOWED: the capability is `launch-and-track` "
                    "(AIDE never tree-searches it). The real GPU launch is OFF "
                    "unless `--launch` AND a `gpu_target` is set; the backend "
                    "adapter computes the launch gate (a missing GPU target refuses "
                    "cleanly)._"
                )
            )
        else:
            current.card.append(
                Table(
                    [[1, mode_gate.get("reason") or "refused"]],
                    headers=["#", "mode-gate blocking reason"],
                )
            )

        # Lineage (what this build traces back to).
        current.card.append(Markdown("## Lineage"))
        current.card.append(
            Table(
                [
                    ["dataset_ref", plan.get("dataset_ref") or "n/a"],
                    ["backbone_ref", plan.get("backbone_ref") or "n/a"],
                    ["backend", plan.get("backend") or "n/a"],
                    ["capability", plan.get("capability") or "n/a"],
                    ["metric", plan.get("metric") or "n/a"],
                ],
                headers=["field", "value"],
            )
        )

    def _render_build_card(self, artifact_ref: dict, status: str) -> None:
        """Append the produced-artifact outcome to the ``@card`` (post-build).

        Note the ``@card`` decorator is on the ``plan`` step (where the gate posture
        is rendered); the ``build`` step's outcome is recorded on ``self.summary``
        (the MLOps interface reads it back). This helper is a no-op safeguard kept
        for symmetry with ``DeployFlow``'s applied-detail narration and is only
        exercised when ``build`` is itself carded by a future change.

        Args:
            artifact_ref: The produced artifact-ref dict.
            status: The headline status line.
        """
        # ``build`` is not a @card step (the gate card is rendered in ``plan``), so
        # there is no ``current.card`` to append to here. The produced-artifact
        # outcome lives on ``self.summary`` / ``self.artifact_ref`` for the MLOps
        # interface's Client-API read. Intentionally a no-op.
        return None

    @step
    def end(self) -> None:
        """Carry ``self.summary`` forward so ``Run.data.summary`` exposes it.

        Metaflow persists step artifacts, so ``self.summary`` (set in ``build``) is
        already on ``Run.data``; the MLOps interface reads it back for the command's
        summary (and any produced data object's ``train`` / ``schema`` /
        ``fingerprint`` artifacts round-trip through the Client API). Nothing else
        to do.
        """
        pass


if __name__ == "__main__":
    TrainFlow()
