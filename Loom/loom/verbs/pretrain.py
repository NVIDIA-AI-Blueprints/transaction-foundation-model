"""``pretrain`` — the GATED, budget-enveloped model-builder launch verb
(ARCHITECTURE §3/§4/§10-step6).

This is the harness-owned planner that resolves a campaign triple
``(data-representation, model-builder, objective)`` to a derived cost PLAN, runs
the gated-launch handshake, and — only on a human-confirmed second call — drives
``ModelBuilder.launch()`` through an ``Executor`` to produce a portable
``CheckpointRef``, persisted as a ``Checkpoint`` :class:`~loom.store.DataObject`.

It is registered ``tier=EXPENSIVE, capability_mode=LAUNCH_AND_TRACK`` — the moment
it registers, ``tools.py:tool_schema`` auto-sets ``disable_model_invocation=true``
for the agent face (the launch verb cannot be fired structurally by the LLM).

The flow (ARCHITECTURE §4 resolution + §3 obligations):

  1. Resolve the input ``Corpus/<n>`` → a :class:`~loom.ports.PreparedCorpus`
     (from its ``signatures`` + payload — the prepare→pretrain handoff).
  2. Look up ``MODEL_BUILDERS[builder]`` (default per-triple; this slice supports
     ``"local"``) and ``EXECUTORS[target]`` (default ``"local"``).
  3. ``builder.supports(...)`` → on False, ``REFUSED_CONTRACT`` carrying the
     reason as a :class:`~loom.types.Diagnostic`.
  4. ``builder.plan(...)`` → a derived :class:`~loom.types.CostPlan`. The first
     call (and EVERY agent-originated call) returns ``status=PLAN`` + a
     single-use ``confirm_token`` over the plan-hash and STOPS — no launch.
  5. An agent-originated launch ATTEMPT (``--launch`` from ``driver="agent"``) is
     refused ``REFUSED_AGENT_CANNOT_LAUNCH`` — the agent can plan but cannot mint
     a launch.
  6. A human-confirmed call (``--launch`` + a valid ``--confirm-token``, or an
     interactive CLI confirm) → ``builder.launch(...)`` via the executor, consume
     the ``ProgressEvent`` stream, write a ``Checkpoint`` DataObject whose
     ``signatures`` carry the ECHOED ``representation_signature`` + the builder's
     ``model_signature`` + the CostPlan actuals, and return ``status=OK`` /
     ``verdict=PASS`` with the ``CheckpointRef``.

Refusals honored: ``REFUSED_NO_GPU_TARGET`` only when the chosen builder/target
REQUIRES a GPU (the ``local`` builder runs on CPU and proceeds); ``REFUSED_SPEND_CAP``
when the derived ``usd`` exceeds the budget cap; ``REFUSED_AGENT_CANNOT_LAUNCH``
for an agent launch attempt; ``REFUSED_CONTRACT`` for an unsupported FM-kind combo.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from ..ports import (
    EXECUTORS,
    MODEL_BUILDERS,
    BudgetEnvelope,
    CheckpointRef,
    ComputeTarget,
    ModelSpec,
    Objective,
    PreparedCorpus,
    ProgressEvent,
)
from ..registry import VerbContext, register
from ..tools import make_confirm_token, validate_confirm_token
from ..types import (
    CapabilityMode,
    CostPlan,
    Diagnostic,
    Severity,
    Status,
    Tier,
    Verdict,
    VerbResult,
)

# The C4 tensor-contract string the next-token CLM objective requires; the
# representation produces it, the builder+objective consume it — the narrow waist.
_CLM_TENSOR_CONTRACT = "clm/input_ids+labels/-100"

# Default per-triple model-builder. The TFM default is "nemo"; THIS slice supports
# only the CPU-rehearsal "local" builder (NeMo is step 7), so resolution falls
# back to "local" when "nemo" is selected but unregistered.
_DEFAULT_BUILDER = "local"
_DEFAULT_EXECUTOR = "local"
_DEFAULT_OBJECTIVE = "next-token"

PRETRAIN_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "in": {"type": "string", "description": "input Corpus/<n> pathspec (the PreparedCorpus)"},
        "model_builder": {"type": "string", "enum": ["local", "nemo"],
                          "description": "model-builder port adapter (default per triple; this slice: local)"},
        "objective": {"type": "string",
                      "description": "training objective (default next-token)"},
        "metric": {"type": "string", "description": "the metric the run optimizes/reports"},
        "budget": {"type": "string", "description": "budget envelope shorthand (max_usd[:max_steps])"},
        "max_usd": {"type": "number", "description": "binding spend cap (USD) the orchestrator hard-kills at"},
        "max_steps": {"type": "integer", "description": "binding max training steps"},
        "max_wall_clock_min": {"type": "integer", "description": "binding wall-clock cap (minutes)"},
        "launch": {"type": "boolean", "description": "request the actual launch (second-call, human-confirmed)"},
        "confirm_token": {"type": "string", "description": "the PLAN's single-use confirm token (§5.3)"},
        "gpu_target": {"type": "string", "description": "GPU target (env LOOM_GPU_TARGET); only required by GPU builders"},
        "family": {"type": "string", "description": "model family (default decoder-clm)"},
    },
}


# ---------------------------------------------------------------------------
# Adapter loading — the registries are populated by the adapter modules at import.
# Importing them is best-effort: a sibling Implement agent may not have landed the
# module yet, and a test may register a builder/executor directly. We never let an
# adapter import failure break `import loom`.
# ---------------------------------------------------------------------------


def _ensure_adapters_loaded() -> None:
    """Import the local builder + local executor so they self-register, if present."""
    import importlib

    for mod in (
        "loom.adapters.local_builder",
        "loom.adapters.local_executor",
        "loom.adapters.event_sequence",
    ):
        try:
            importlib.import_module(mod)
        except Exception:  # noqa: BLE001 - adapter not landed / optional in this slice
            pass


# ---------------------------------------------------------------------------
# Corpus → PreparedCorpus resolution (the prepare→pretrain handoff, §4).
# ---------------------------------------------------------------------------


def _corpus_to_prepared(obj: Any) -> PreparedCorpus:
    """Reconstruct a :class:`PreparedCorpus` from a stored Corpus DataObject.

    The corpus's ``signatures`` dict is the canonical handoff (today's
    ``vocab_hash``→``representation_signature``, ``encode_path``→``representation``,
    plus ``vocab_size``/``context_len``). Local corpora have no ``gs://`` shards
    yet (that is the §8 streaming/datastore step), so the URIs point at the local
    payload and the manifest is the object pathspec — opaque to the builder, which
    interprets nothing beyond ``tensor_contract``."""
    sig = dict(getattr(obj, "signatures", {}) or {})
    obj_extras = dict(getattr(obj, "extras", {}) or {})

    # Load the heavy payload (the Corpus' corpus.json) so the model-builder has a
    # token stream to rehearse on. The builder reads extras['corpus_lines'] +
    # extras['vocab'] (the event-sequence shape) or extras['token_lines']; we
    # thread them straight off the persisted payload. A missing/unreadable payload
    # is tolerated (the builder's supports() will then refuse with a clear reason).
    payload: dict[str, Any] = {}
    payload_path = getattr(obj, "payload_path", None)
    if payload_path:
        try:
            with open(payload_path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, ValueError):
            payload = {}

    representation = sig.get("encode_path") or sig.get("representation") or obj_extras.get("preset") or "event-sequence"
    representation_signature = (
        sig.get("representation_signature") or sig.get("vocab_hash") or obj.content_id
    )
    vocab_size = sig.get("vocab_size") or payload.get("vocab_size")
    seq_length = int(sig.get("context_len") or sig.get("seq_length") or payload.get("context_len") or 4096)
    n_lines = int(obj_extras.get("n_lines", payload.get("n_lines", 0)) or 0)
    chunk_size = int(sig.get("chunk_size") or payload.get("chunk_size") or 0)
    # effective_tokens: measured non-redundant token budget. For the local
    # rehearsal we derive a cheap, deterministic estimate from the corpus shape
    # (lines × chunk_size), never a hardcoded number.
    effective_tokens = max(1, n_lines * max(1, chunk_size))

    payload_uri = getattr(obj, "payload_path", None) or obj.pathspec

    return PreparedCorpus(
        representation=str(representation),
        representation_signature=str(representation_signature),
        tensor_contract=str(sig.get("tensor_contract") or _CLM_TENSOR_CONTRACT),
        train_uri=str(payload_uri),
        val_uri=None,
        test_uri=None,
        manifest_uri=obj.pathspec,
        seq_length=seq_length,
        pad_token_id=int(sig.get("pad_token_id") or 0),
        vocab_size=int(vocab_size) if vocab_size is not None else None,
        effective_tokens=effective_tokens,
        provenance={"corpus": obj.pathspec, "content_id": obj.content_id},
        # Thread the token stream the model-builder rehearses on (event-sequence
        # shape: corpus_lines + vocab; or pre-id'd token_lines). The builder is
        # OPAQUE to everything else — it only reads these + tensor_contract.
        extras={
            "signatures": sig,
            "corpus_lines": payload.get("corpus_lines"),
            "vocab": payload.get("vocab"),
            "token_lines": payload.get("token_lines"),
        },
    )


def _plan_hash(
    *,
    corpus: PreparedCorpus,
    builder_name: str,
    objective: Objective,
    model: ModelSpec,
    compute: ComputeTarget,
    budget: BudgetEnvelope,
    plan: CostPlan,
) -> str:
    """A stable hash of the compiled plan — the confirm_token is scoped to THIS.

    Any change to the corpus signature, the builder, the objective, the arch, the
    compute, or the budget envelope produces a different plan_hash, invalidating a
    confirm_token minted for a different plan (the gated-launch replay anchor)."""
    payload = {
        "representation_signature": corpus.representation_signature,
        "tensor_contract": corpus.tensor_contract,
        "builder": builder_name,
        "objective": objective.kind,
        "family": model.family,
        "arch": model.arch,
        "compute": {
            "launcher": compute.launcher,
            "accelerator": compute.accelerator,
            "nproc_per_node": compute.nproc_per_node,
            "nnodes": compute.nnodes,
        },
        "budget": {
            "max_usd": budget.max_usd,
            "max_steps": budget.max_steps,
            "max_wall_clock_min": budget.max_wall_clock_min,
        },
        "plan_usd": plan.usd,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Result builders.
# ---------------------------------------------------------------------------


def _refused(
    status: Status,
    summary: str,
    *,
    diagnostics: Optional[list[Diagnostic]] = None,
    cost_plan: Optional[CostPlan] = None,
    experiment: Optional[str] = None,
    data: Optional[dict[str, Any]] = None,
) -> VerbResult:
    return VerbResult(
        verb="pretrain",
        status=status,
        verdict=Verdict.FAIL,
        tier=Tier.EXPENSIVE,
        capability_mode=CapabilityMode.LAUNCH_AND_TRACK,
        summary=summary,
        diagnostics=diagnostics or [],
        data=data or {},
        experiment=experiment,
        cost_plan=cost_plan or CostPlan(),
    )


def _budget_from_args(args: dict[str, Any], plan_usd: Optional[float]) -> BudgetEnvelope:
    """Build the binding :class:`BudgetEnvelope` from the args.

    ``--budget`` is a shorthand ``max_usd[:max_steps]``; explicit ``--max-usd`` /
    ``--max-steps`` / ``--max-wall-clock-min`` override. If no cap is given, the
    envelope defaults to the derived plan usd (so the human still approves a
    binding number, never an open-ended spend)."""
    max_usd: Optional[float] = None
    max_steps: Optional[int] = None
    max_wall: Optional[int] = None

    budget_s = args.get("budget")
    if isinstance(budget_s, str) and budget_s:
        head, _, tail = budget_s.partition(":")
        try:
            max_usd = float(head)
        except ValueError:
            max_usd = None
        if tail:
            try:
                max_steps = int(tail)
            except ValueError:
                max_steps = None

    if args.get("max_usd") is not None:
        max_usd = float(args["max_usd"])
    if args.get("max_steps") is not None:
        max_steps = int(args["max_steps"])
    if args.get("max_wall_clock_min") is not None:
        max_wall = int(args["max_wall_clock_min"])

    if max_usd is None:
        max_usd = float(plan_usd) if plan_usd is not None else 0.0

    return BudgetEnvelope(max_usd=max_usd, max_wall_clock_min=max_wall, max_steps=max_steps)


@register(
    "pretrain",
    summary="plan + (gated) launch a model-builder run over a Corpus → a portable Checkpoint",
    tier=Tier.EXPENSIVE,
    capability_mode=CapabilityMode.LAUNCH_AND_TRACK,
    params=PRETRAIN_PARAMS,
)
def _pretrain(args: dict[str, Any], ctx: VerbContext) -> VerbResult:
    from ..store import DataObject  # local import: store is the v0.2 seam

    experiment = ctx.experiment
    in_spec = args.get("in") or ""

    _ensure_adapters_loaded()

    # --- resolve the input Corpus → PreparedCorpus ----------------------------
    if not in_spec:
        return _refused(
            Status.REFUSED_CONTRACT,
            "pretrain needs an input Corpus/<n> (the PreparedCorpus handoff)",
            diagnostics=[Diagnostic(
                contract="C4", severity=Severity.ERROR,
                message="no --in Corpus/<n> given",
                fix="pass `loom pretrain --in Corpus/<n> --model-builder local`",
            )],
            experiment=experiment,
        )
    try:
        corpus_obj = ctx.store.get(in_spec)
    except (KeyError, ValueError) as exc:
        return _refused(
            Status.REFUSED_CONTRACT,
            f"pretrain could not resolve input {in_spec!r}: {exc}",
            diagnostics=[Diagnostic(
                contract="C4", severity=Severity.ERROR,
                message=f"input {in_spec!r} not found in the store",
                fix="run `loom tokenize`/`loom prepare` to produce a Corpus first",
            )],
            experiment=experiment,
        )
    corpus = _corpus_to_prepared(corpus_obj)

    # --- resolve the FM-kind triple -------------------------------------------
    builder_name = (args.get("model_builder") or _DEFAULT_BUILDER).lower()
    if builder_name not in MODEL_BUILDERS and _DEFAULT_BUILDER in MODEL_BUILDERS:
        # e.g. "nemo" requested but not registered in this no-GPU slice → fall back.
        builder_name = _DEFAULT_BUILDER
    builder = MODEL_BUILDERS.get(builder_name)
    if builder is None:
        return _refused(
            Status.REFUSED_CONTRACT,
            f"pretrain: no model-builder {builder_name!r} registered",
            diagnostics=[Diagnostic(
                contract="C5", severity=Severity.ERROR,
                message=f"MODEL_BUILDERS has no adapter {builder_name!r}",
                fix="this slice supports --model-builder local",
            )],
            experiment=experiment,
        )

    target_name = (args.get("execution") or _DEFAULT_EXECUTOR).lower()
    executor = EXECUTORS.get(target_name) or EXECUTORS.get(_DEFAULT_EXECUTOR)
    if executor is None:
        return _refused(
            Status.REFUSED_CONTRACT,
            f"pretrain: no executor {target_name!r} registered",
            diagnostics=[Diagnostic(
                contract="C5", severity=Severity.ERROR,
                message=f"EXECUTORS has no adapter {target_name!r}",
                fix="this slice provides the local (in-process) executor",
            )],
            experiment=experiment,
        )

    objective = Objective(
        kind=(args.get("objective") or _DEFAULT_OBJECTIVE),
        requires_tensor_contract=_CLM_TENSOR_CONTRACT,
        loss_target=None,
    )
    model = ModelSpec(
        family=(args.get("family") or "decoder-clm"),
        arch={
            "_target_": "transformers.LlamaConfig",
            "vocab_size": corpus.vocab_size,  # C1 → arch (from-corpus, not hardcoded)
            "max_position_embeddings": corpus.seq_length,
        },
        init="from_scratch",
    )

    # gpu_target echoes LOOM_GPU_TARGET; only GPU builders require it.
    import os
    gpu_target = args.get("gpu_target") or os.environ.get("LOOM_GPU_TARGET")
    # The local CPU-rehearsal builder/executor run on CPU; a GPU builder asks the
    # executor whether a GPU is reachable, gating REFUSED_NO_GPU_TARGET.
    requires_gpu = bool(getattr(executor, "gpu_available", lambda: False)()) is False and (
        builder_name not in ("local",) and target_name not in ("local",)
    )
    compute = ComputeTarget(
        launcher=target_name,
        nproc_per_node=1,
        nnodes=1,
        accelerator="cpu" if builder_name == "local" else "gpu",
        gpu_target=gpu_target,
        image=os.environ.get("LOOM_NEMO_IMAGE"),
    )

    # --- 1. validate the FM kind BEFORE any spend (§4) ------------------------
    cap = builder.supports(model=model, objective=objective, corpus=corpus)
    if not cap.supported:
        return _refused(
            Status.REFUSED_CONTRACT,
            f"pretrain REFUSED_CONTRACT: builder {builder_name!r} does not support this FM-kind"
            + (f" — {cap.reason}" if cap.reason else ""),
            diagnostics=[Diagnostic(
                contract="C4", severity=Severity.ERROR,
                message=cap.reason or "unsupported (data-representation, model-builder, objective) combo",
                fix="select a builder/objective compatible with the corpus tensor_contract",
                data={"tensor_contract": corpus.tensor_contract,
                      "requires_tensor_contract": objective.requires_tensor_contract},
            )],
            experiment=experiment,
        )

    # --- GPU gate (only when the chosen builder/target REQUIRES a GPU) --------
    if requires_gpu and not gpu_target:
        return _refused(
            Status.REFUSED_NO_GPU_TARGET,
            f"pretrain REFUSED_NO_GPU_TARGET: builder {builder_name!r}/executor {target_name!r} "
            "requires a GPU but no --gpu-target / LOOM_GPU_TARGET is set",
            diagnostics=[Diagnostic(
                contract="C5", severity=Severity.ERROR,
                message="no GPU target for a GPU-requiring builder",
                fix="set LOOM_GPU_TARGET or pass --gpu-target; or use --model-builder local (CPU)",
            )],
            experiment=experiment,
        )

    # --- 2. the derived cost PLAN (no launch) ---------------------------------
    # A provisional envelope (max_usd from args, else filled from the plan below).
    provisional_budget = _budget_from_args(args, None)
    plan = builder.plan(
        corpus=corpus, model=model, objective=objective,
        compute=compute, budget=provisional_budget, executor=executor,
    )
    # Bind the real envelope now that we know the derived usd.
    budget = _budget_from_args(args, plan.usd)
    plan.envelope = {
        "max_usd": budget.max_usd,
        "max_steps": budget.max_steps,
        "max_wall_clock_min": budget.max_wall_clock_min,
    }
    plan_hash = _plan_hash(
        corpus=corpus, builder_name=builder_name, objective=objective,
        model=model, compute=compute, budget=budget, plan=plan,
    )

    # --- spend-cap gate: a derived usd over the binding cap is refused --------
    if budget.max_usd is not None and plan.usd is not None and plan.usd > budget.max_usd + 1e-9:
        return _refused(
            Status.REFUSED_SPEND_CAP,
            f"pretrain REFUSED_SPEND_CAP: derived ${plan.usd:.2f} exceeds the binding "
            f"cap ${budget.max_usd:.2f}",
            diagnostics=[Diagnostic(
                contract="C5", severity=Severity.ERROR,
                message=f"derived cost ${plan.usd:.4f} > cap ${budget.max_usd:.4f}",
                fix="raise --max-usd or shrink --max-steps / the arch",
            )],
            cost_plan=plan,
            experiment=experiment,
            data={"plan_hash": plan_hash},
        )

    launch_requested = bool(args.get("launch"))
    is_agent = ctx.driver == "agent"

    # --- 5. an AGENT cannot mint a launch (structural refusal) ----------------
    if launch_requested and is_agent:
        return _refused(
            Status.REFUSED_AGENT_CANNOT_LAUNCH,
            "pretrain REFUSED_AGENT_CANNOT_LAUNCH: an agent may PLAN but cannot launch a "
            "money/compute spend; a human confirms with the token.",
            diagnostics=[Diagnostic(
                contract="C5", severity=Severity.ERROR,
                message="agent-originated launch attempt on a launch-and-track verb",
                fix="surface the PLAN + confirm_token to a human; the human launches.",
            )],
            cost_plan=plan,
            experiment=experiment,
            data={"plan_hash": plan_hash},
        )

    # --- 3/4. the gated handshake ---------------------------------------------
    token = ctx.confirm_token or args.get("confirm_token")
    confirmed = launch_requested and validate_confirm_token(token, plan_hash)

    if not confirmed:
        # First call (or no/invalid token, or no --launch): return the PLAN + a
        # fresh single-use confirm_token scoped to this plan_hash, and STOP.
        confirm_token = make_confirm_token(plan_hash)
        return VerbResult(
            verb="pretrain",
            status=Status.PLAN,
            verdict=Verdict.REVIEW,
            tier=Tier.EXPENSIVE,
            capability_mode=CapabilityMode.LAUNCH_AND_TRACK,
            summary=(
                f"PLAN: {builder_name} build over {corpus_obj.pathspec} → "
                f"~${(plan.usd or 0.0):.4f} ({plan.confidence or 'LOW'}), "
                f"envelope max_usd=${budget.max_usd:.2f}"
                + (f", max_steps={budget.max_steps}" if budget.max_steps else "")
                + ". Confirm to launch."
            ),
            outputs=[],
            diagnostics=[],
            data={
                "plan_hash": plan_hash,
                "model_builder": builder_name,
                "executor": target_name,
                "objective": objective.kind,
                "representation_signature": corpus.representation_signature,
                "tensor_contract": corpus.tensor_contract,
                "corpus": corpus_obj.pathspec,
                "requires_gpu": requires_gpu,
            },
            experiment=experiment,
            cost_plan=plan,
            confirm_token=confirm_token,
        )

    # --- 6. CONFIRMED launch — drive the builder through the executor ---------
    handle = builder.launch(
        corpus=corpus, model=model, objective=objective,
        compute=compute, budget=budget, executor=executor,
    )

    # Consume the progress stream (the launch-and-track feed). The local builder
    # streams a couple of events incl. a step-0 canary note; collect them for the
    # checkpoint metrics + the run record.
    events: list[ProgressEvent] = []
    try:
        for ev in handle.stream_events():
            events.append(ev)
    except Exception:  # noqa: BLE001 - a builder may produce no stream; result() is authoritative
        pass

    ckpt: CheckpointRef = handle.result()  # blocks until terminal
    final_status = handle.status()

    # Pairing invariant (§3/§7): the checkpoint MUST echo the corpus signature.
    sig_match = ckpt.representation_signature == corpus.representation_signature
    if not sig_match:
        return _refused(
            Status.REFUSED_CONTRACT,
            "pretrain REFUSED_CONTRACT: checkpoint representation_signature does not "
            "match the corpus (the checkpoint↔representation pairing invariant)",
            diagnostics=[Diagnostic(
                contract="C5", severity=Severity.ERROR,
                message="representation_signature mismatch between corpus and checkpoint",
                fix="rebuild the checkpoint from the same corpus signature",
                data={"corpus_sig": corpus.representation_signature,
                      "checkpoint_sig": ckpt.representation_signature},
            )],
            cost_plan=plan,
            experiment=experiment,
            data={"plan_hash": plan_hash},
        )

    # CostPlan actuals (what the run cost vs the derived estimate).
    last_event = events[-1] if events else None
    cost_actuals = {
        "usd_spent": (last_event.usd_spent if last_event else (plan.usd or 0.0)),
        "steps": (last_event.step if last_event else (budget.max_steps or 0)),
        "wall_clock_min": (last_event.wall_clock_min if last_event else 0.0),
        "derived_usd": plan.usd,
    }

    terminal_verdict = (
        Verdict.PASS if final_status == "succeeded"
        else Verdict.INCOMPLETE if final_status == "stopped_at_budget"
        else Verdict.FAIL
    )

    # --- persist the Checkpoint DataObject ------------------------------------
    ref = ctx.store.new_ref("Checkpoint")
    cobj = DataObject(
        ref=ref,
        kind="Checkpoint",
        content_id=f"{corpus_obj.content_id}:checkpoint:{ckpt.model_signature}",
        parents=[corpus_obj.pathspec],
        producer_verb="pretrain",
        producer_args={
            "in": in_spec,
            "model_builder": builder_name,
            "executor": target_name,
            "objective": objective.kind,
            "max_usd": budget.max_usd,
            "max_steps": budget.max_steps,
        },
        # The cross-port pairing invariant travels WITH the object: embed/evaluate
        # (step 7) assert representation_signature equality before any forward pass.
        signatures={
            "representation_signature": ckpt.representation_signature,
            "model_signature": ckpt.model_signature,
            "fmt": ckpt.fmt,
            "tensor_contract": corpus.tensor_contract,
            "vocab_size": corpus.vocab_size,
        },
        verdict=terminal_verdict,
        status=(Status.OK if final_status == "succeeded" else Status.OK),
        experiment=experiment,
        cost_actuals=cost_actuals,
        envelope=plan.envelope,
        extras={
            "checkpoint": {
                "uri": ckpt.uri,
                "fmt": ckpt.fmt,
                "metrics": ckpt.metrics,
            },
            "job_id": getattr(handle, "job_id", None),
            "final_status": final_status,
            "progress_events": [
                {
                    "step": e.step, "loss": e.loss, "usd_spent": e.usd_spent,
                    "usd_envelope": e.usd_envelope, "phase": e.phase, "note": e.note,
                }
                for e in events
            ],
        },
    )
    stored = ctx.store.put(cobj)

    plan.envelope = plan.envelope  # already bound
    return VerbResult(
        verb="pretrain",
        status=Status.OK,
        verdict=terminal_verdict,
        tier=Tier.EXPENSIVE,
        capability_mode=CapabilityMode.LAUNCH_AND_TRACK,
        summary=(
            f"{stored.pathspec} verdict={terminal_verdict.value} "
            f"builder={builder_name} fmt={ckpt.fmt} "
            f"sig={ckpt.representation_signature[:18]}… "
            f"model_sig={ckpt.model_signature[:12]}… "
            f"usd_spent=${cost_actuals['usd_spent']:.4f}"
        ),
        outputs=[stored.ref],
        diagnostics=[],
        data={
            "pathspec": stored.pathspec,
            "plan_hash": plan_hash,
            "checkpoint_uri": ckpt.uri,
            "fmt": ckpt.fmt,
            "representation_signature": ckpt.representation_signature,
            "model_signature": ckpt.model_signature,
            "final_status": final_status,
            "cost_actuals": cost_actuals,
            "n_events": len(events),
        },
        experiment=experiment,
        cost_plan=plan,
    )
