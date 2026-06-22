"""Loom ports — the two adapter Protocols + the executor seam (ARCHITECTURE §2).

Loom is a generic ports-and-adapters foundation-model training harness. The
*harness* is the constant (the verb spine, the typed result envelope, the
lineage store, the gated/budget-enveloped launch, the telemetry capture, the
checkpoint↔representation signature). The *variable* is three ports a campaign
spec selects:

- a **data-representation port** (raw events → corpus),
- a **model-builder port** (corpus → checkpoint),
- a thin **executor port** (where heavy work runs).

An "FM kind" is a ``(data-representation adapter, model-builder adapter,
objective)`` triple; **TFM is the bundled default triple** ``event-sequence`` +
``nemo`` + ``next-token``.

This module is **pure types** — frozen dataclasses (the inter-stage value
objects) plus ``typing.Protocol`` interfaces — and three module-global
registries mirroring the proven ``registry.py:REGISTRY`` pattern. It carries
**no behavior**.

Placement & blast radius. The value-objects live here, NOT appended to the
LOCKED ``loom/types.py`` — that keeps the envelope module's blast radius zero.
This module imports ONLY ``CostPlan`` and ``Diagnostic`` from ``loom.types``
(the existing envelope value-objects it reuses); ``CostPlan`` is GPU-shaped
already and is never redefined here.

Hard rule (ARCHITECTURE §0/§2): **nothing in this module names or imports NeMo,
nemo_automodel, torch, transformers, FSDP2, torchrun, safetensors, RAPIDS/cudf,
or BigQuery.** Those are adapter-local realizations behind these Protocols.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal, Optional, Protocol, runtime_checkable

from .types import CostPlan, Diagnostic  # reuse the EXISTING envelope value-objects

# A content hash; the retrain trigger. Generalizes today's ``vocab_hash`` — any
# representation exposes a stable ``representation_signature`` of this type, and
# the model-builder echoes it onto the checkpoint (the harness-level pairing
# invariant, §3/§7).
Signature = str


# ---------------------------------------------------------------------------
# 2.0 Harness-level value objects — the inter-stage artifact contracts.
#
# SourceRef, PreparedCorpus, CheckpointRef, BudgetEnvelope, Signature are
# *harness-level*: they are the narrow waist BETWEEN the ports. CostPlan is
# reused from types.py (its GPU fields are already shaped), never redefined.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceRef:
    """Raw input to ``prepare`` — cloud-resident, NEVER local (§6, I1)."""

    uri: str  # bq://project.dataset.table  |  gs://bucket/prefix/
    # {max_event_date | date_prefix_range} — the provenance anchor.
    snapshot: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedCorpus:
    """The ``prepare → pretrain`` handoff. OPAQUE to the model-builder, which
    interprets nothing beyond ``tensor_contract`` (§9)."""

    representation: str  # registry name, e.g. "event-sequence"
    # generalizes today's vocab_hash (the retrain trigger).
    representation_signature: Signature
    # "clm/input_ids+labels/-100" (C4) | "mlm/..." | "vision/patches".
    tensor_contract: str
    train_uri: str  # gs://.../train/shard-*.arrow  (NOT an in-RAM list — I1)
    val_uri: Optional[str]
    test_uri: Optional[str]
    manifest_uri: str  # counts, snapshot range, split definition
    seq_length: int
    pad_token_id: int
    # None for continuous representations; feeds model arch (C1 → arch).
    vocab_size: Optional[int]
    # measured NON-redundant tokens → feeds the D/N param-scaling gate.
    effective_tokens: int
    # snapshot range / MAX(event_date) — separate from the port-local C2.
    provenance: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    """Architecture, framework-neutral."""

    # "decoder-clm" | "encoder-mlm" | "vision-mae"  (picks the objective family).
    family: str
    # {"_target_": "transformers.LlamaConfig", "vocab_size": ..., ...}
    arch: dict
    init: str = "from_scratch"  # | "from_pretrained:<hub_id_or_uri>"


@dataclass(frozen=True)
class Objective:
    """The third leg of an FM kind — a plain record, NOT a port."""

    kind: str  # "next-token" | "masked-lm" | "contrastive"
    requires_tensor_contract: str  # the C4 string the corpus MUST satisfy
    # advisory; adapter validates support (NeMo CLM ⇒ MaskedCrossEntropy).
    loss_target: Optional[str] = None


@dataclass(frozen=True)
class ComputeTarget:
    """Consumed by the executor; NOT itself a port."""

    launcher: str  # "torchrun" | "local" | "metaflow-gcp" | "modal"
    nproc_per_node: int = 1
    nnodes: int = 1
    accelerator: Literal["gpu", "cpu"] = "gpu"
    # echoes LOOM_GPU_TARGET; None ⇒ REFUSED_NO_GPU_TARGET.
    gpu_target: Optional[str] = None
    image: Optional[str] = None  # echoes LOOM_NEMO_IMAGE
    # {dp,tp,cp,sequence_parallel} — adapter maps to its own knobs.
    parallelism: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BudgetEnvelope:
    """The BINDING approval object — the orchestrator hard-kills at these."""

    max_usd: float
    max_wall_clock_min: Optional[int] = None
    max_steps: Optional[int] = None


@dataclass(frozen=True)
class Capability:
    """``supports()`` return — declare/reject an FM-kind combo before any spend."""

    supported: bool
    # why a combo is rejected → surfaced as a REFUSED_CONTRACT diagnostic.
    reason: Optional[str] = None


@dataclass(frozen=True)
class CheckpointRef:
    """The ``pretrain → embed`` handoff."""

    uri: str  # consolidated HF safetensors dir
    fmt: str  # "hf-safetensors-consolidated"  (C5)
    # ECHOED verbatim from PreparedCorpus → the harness-level pairing invariant.
    representation_signature: Signature
    # hash of {arch, objective, code_sha} — the replay anchor.
    model_signature: Signature
    # final/last loss, step, val, step0_canary.
    metrics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ProgressEvent:
    """What feeds the Pi widgets (the launch-and-track feed + budget telemetry)."""

    step: int
    loss: Optional[float]
    usd_spent: float
    usd_envelope: float
    gpu_pct: Optional[float] = None
    wall_clock_min: float = 0.0
    phase: str = "train"  # "warmup" | "train" | "val" | "consolidate"
    note: Optional[str] = None  # e.g. step-0 canary "loss≈ln(vocab)"


# ---------------------------------------------------------------------------
# 2.1 The data-representation port.
# ---------------------------------------------------------------------------


@runtime_checkable
class DataRepresentation(Protocol):
    """raw events → corpus. Adapter #1 is ``event-sequence`` (tokenized CLM).

    Every representation exposes a stable ``representation_signature`` (today's
    ``vocab_hash``) and declares its ``tensor_contract``. ``contracts()`` returns
    ``Diagnostic(contract=…)`` cards under its OWN names (the free-string
    ``Diagnostic.contract`` is the universal extension hook). ``materialize()`` is
    the only method that touches scale — it runs the harness-owned executor and
    returns a ``PreparedCorpus`` pointing at sharded object storage, never an
    in-RAM list (§6, I1)."""

    name: str  # registry key; "event-sequence" is adapter #1
    # the C4 string it emits, e.g. "clm/input_ids+labels/-100".
    produces_tensor_contract: str

    def build_spec(self, args: dict) -> Any:  # was tokenize._build_spec
        ...

    # was engine.compile_spec (data-free, deterministic; emits the signature).
    def compile(self, spec: Any, *, context_len: int) -> Any:
        ...

    # PORT-LOCAL contract cards (event-sequence: C1/C2/C3).
    def contracts(self, compiled: Any) -> list[Diagnostic]:
        ...

    def representation_passed(self, compiled: Any) -> bool:
        """The contract-NAME-AGNOSTIC corpus write-gate (the §8 fix).

        Returns ``True`` iff the compiled object carries NO ERROR-severity
        Diagnostic, regardless of contract string. The default any adapter
        inherits is literally::

            not any(d.severity is Severity.ERROR for d in self.contracts(compiled))

        The event-sequence adapter satisfies it for free (``compiled.report.passed``
        is already ``False`` on any ERROR). ``prepare`` refuses to persist a Corpus
        iff ``not representation_passed(compiled)`` — it never names "C1"/"C3" nor
        reads tokenizer-shaped attributes."""
        ...

    # CPU-cheap, or a BigQuery-bytes estimate. NO materialization.
    def plan(self, *, spec: Any, source: SourceRef, executor: "Executor") -> CostPlan:
        ...

    # the I1 scale-out (§6) — runs executor.foreach and writes SHARDS.
    def materialize(
        self, *, compiled: Any, source: SourceRef, executor: "Executor"
    ) -> PreparedCorpus:
        ...

    # the harness handoff dict (today: tokenize.py:399-407).
    def signatures(self, compiled: Any) -> dict:
        ...


# ---------------------------------------------------------------------------
# 2.2 The model-builder port.
# ---------------------------------------------------------------------------


@runtime_checkable
class TrainingHandle(Protocol):
    """The launch-and-track capability surface returned by ``ModelBuilder.launch``."""

    job_id: str

    # → loom top --json widget feed (the net-new progress source, §5.1).
    def stream_events(self) -> Iterator[ProgressEvent]:
        ...

    def status(
        self,
    ) -> Literal[
        "pending", "running", "succeeded", "failed", "killed", "stopped_at_budget"
    ]:
        ...

    def cancel(self) -> None:
        ...

    def result(self) -> CheckpointRef:  # blocks until terminal
        ...


@runtime_checkable
class ModelBuilder(Protocol):
    """corpus → checkpoint. ``nemo`` (#1, FSDP2/torchrun/HF-safetensors) and
    ``local`` (#2, CPU rehearsal — PPMI+SVD, torch-free, the CI oracle)."""

    name: str  # "nemo" (#1) | "local" (CPU rehearsal, #2)

    def supports(
        self, *, model: ModelSpec, objective: Objective, corpus: PreparedCorpus
    ) -> Capability:  # declare/reject an FM-kind combo BEFORE spend
        ...

    def plan(
        self,
        *,
        corpus: PreparedCorpus,
        model: ModelSpec,
        objective: Objective,
        compute: ComputeTarget,
        budget: BudgetEnvelope,
        executor: "Executor",
    ) -> CostPlan:  # NO launch
        ...

    def launch(
        self,
        *,
        corpus: PreparedCorpus,
        model: ModelSpec,
        objective: Objective,
        compute: ComputeTarget,
        budget: BudgetEnvelope,
        executor: "Executor",
    ) -> TrainingHandle:
        ...


# ---------------------------------------------------------------------------
# 2.3 The executor port — the scale-out seam.
#
# The single graft that makes the GPU launch and the corpus-build fan-out share
# ONE swappable seam: the executor is the only place that knows how work is
# submitted, fanned out, and killed. ``ModelBuilder.launch()`` calls
# ``executor.submit(...)``; ``DataRepresentation.materialize()`` calls
# ``executor.foreach(...)`` — both flow through the same budget/kill contract.
# ---------------------------------------------------------------------------


@runtime_checkable
class JobHandle(Protocol):
    """A submitted unit of work — the budget-killable handle ``submit()`` returns
    and ``kill()`` targets. The thin process/job surface beneath a
    ``TrainingHandle``."""

    job_id: str

    def status(
        self,
    ) -> Literal[
        "pending", "running", "succeeded", "failed", "killed", "stopped_at_budget"
    ]:
        ...

    def cancel(self) -> None:
        ...


@runtime_checkable
class Executor(Protocol):
    """where heavy work runs. ``local`` (in-process foreach, single box),
    ``metaflow-gcp`` (on-demand GPU fan-out), ``modal``."""

    name: str  # "local" | "metaflow-gcp" | "modal"

    def gpu_available(self) -> bool:  # gates REFUSED_NO_GPU_TARGET
        ...

    def submit(
        self,
        *,
        argv: list[str],
        image: Optional[str],
        compute: ComputeTarget,
        budget: BudgetEnvelope,
        on_event: Callable[[ProgressEvent], None],
    ) -> JobHandle:  # runs e.g. `torchrun … train_decoder_model.py`
        ...

    # the Tier-B fan-out (§6): one task per shard, never the whole corpus in RAM.
    def foreach(
        self, *, fn: Callable[[str], str], shards: list[str], compute: ComputeTarget
    ) -> list[str]:
        ...

    def kill(self, job_id: str) -> None:  # the binding-envelope hard-kill
        ...


# ---------------------------------------------------------------------------
# 2.4 The registries — three module-globals mirroring registry.py:REGISTRY.
# ---------------------------------------------------------------------------

REPRESENTATIONS: dict[str, DataRepresentation] = {}
MODEL_BUILDERS: dict[str, ModelBuilder] = {}
EXECUTORS: dict[str, Executor] = {}


def register_representation(r: DataRepresentation) -> DataRepresentation:
    """Register a data-representation adapter under its ``name``."""
    REPRESENTATIONS[r.name] = r
    return r


def register_model_builder(b: ModelBuilder) -> ModelBuilder:
    """Register a model-builder adapter under its ``name``."""
    MODEL_BUILDERS[b.name] = b
    return b


def register_executor(e: Executor) -> Executor:
    """Register an executor adapter under its ``name``."""
    EXECUTORS[e.name] = e
    return e


__all__ = [
    "Signature",
    # value objects
    "SourceRef",
    "PreparedCorpus",
    "ModelSpec",
    "Objective",
    "ComputeTarget",
    "BudgetEnvelope",
    "Capability",
    "CheckpointRef",
    "ProgressEvent",
    # protocols
    "DataRepresentation",
    "ModelBuilder",
    "TrainingHandle",
    "JobHandle",
    "Executor",
    # registries + register fns
    "REPRESENTATIONS",
    "MODEL_BUILDERS",
    "EXECUTORS",
    "register_representation",
    "register_model_builder",
    "register_executor",
]
