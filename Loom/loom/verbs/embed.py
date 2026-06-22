"""``embed`` — a LOCAL, UNGATED, CPU/$0 PPMI-SVD embedding of a Corpus.

The first half of the *validate-before-train* loop (`propose → tokenize → embed
→ evaluate → refine`): it turns a tokenized ``Corpus`` into a real
``Embeddings`` artifact a user can SCORE *before* spending a single GPU-hour
(ARCHITECTURE §10 step 7 — the local rehearsal as a validation seam, not a
launch).

Why this is NOT ``pretrain``. ``pretrain`` is the gated, ``EXPENSIVE`` /
``LAUNCH_AND_TRACK`` launch verb: it routes through
:meth:`LocalModelBuilder.launch` (the ``MODEL_BUILDERS`` port), which emits
``ProgressEvent``s, a step-0 canary, a ``CheckpointRef`` round-trip, and the
``executor.foreach`` seam — all the launch-and-track machinery. ``embed`` does
none of that. It is a ``WORKSPACE_WRITE`` / ``SEARCHABLE`` verb (so the agent
can call it freely and the gate is inert) that calls the *pure free functions*
of the local builder directly:

  * :func:`loom.verbs.pretrain._corpus_to_prepared` — the gate-free Corpus →
    ``PreparedCorpus`` loader (reads the stored ``corpus.json`` payload), then
  * :func:`loom.adapters.local_builder._read_token_stream` — pull the
    ``(token_id_lines, vocab_size)`` the fit rehearses on,
  * :func:`loom.adapters.local_builder._fit_ppmi_svd` — the deterministic
    (sklearn ``TruncatedSVD`` ``random_state=0``) co-occurrence → PPMI → SVD
    embedding fit, and
  * :func:`loom.adapters.local_builder._write_consolidated_safetensors` — the
    C5-valid consolidated HF-safetensors (``model.safetensors`` weight
    ``embeddings`` + co-located ``config.json``).

No ``Executor``, no ``BudgetEnvelope``, no ``CheckpointRef``, no
``confirm_token`` — ``cost_plan=CostPlan()`` (a bare $0 CPU plan). The output is
an ``Embeddings/<n>`` object whose row ``i`` is exactly vocab id ``i`` (NO
permutation), addressed deterministically off the corpus content_id so a re-run
returns the existing object via the store's content dedupe.

GUARD (labeled): :func:`_fit_ppmi_svd` builds a *dense* ``O(vocab²)``
co-occurrence matrix. That is fine on a SAMPLE-scoped vocab (the validate loop
is meant to run on a small slice), but do NOT point ``embed`` at a full-corpus
vocab — the memory ceiling is the vocab size, not ``dim``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..adapters import local_builder
from ..registry import VerbContext, register
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
from . import pretrain

# ``dim`` default is 128 (Anub's product decision — the new default; 16 stays a
# valid value, refine can change it). ``window`` defaults to the local builder's
# co-occurrence half-window (the single source of truth, not a re-typed literal).
_DEFAULT_DIM = 128
_DEFAULT_WINDOW = local_builder._WINDOW

EMBED_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "in": {
            "type": "string",
            "description": "input Corpus/<n> pathspec to embed (PPMI-SVD, CPU, ~$0)",
        },
        "dim": {
            "type": "integer",
            "description": "embedding dimensionality (default 128; clamped down for a tiny vocab)",
        },
        "window": {
            "type": "integer",
            "description": "co-occurrence context half-window in tokens (default 2)",
        },
    },
}


def _refused_no_stream(in_spec: str, experiment: Any) -> VerbResult:
    """A CLEAN structural refusal when the Corpus carries no readable token stream
    (C2). Never a stack trace, never an all-zeros matrix masquerading as success —
    ``cost_plan=CostPlan()`` ($0), verdict FAIL, a named C2 diagnostic with the
    exact one-line fix."""
    msg = "Corpus has no token stream to embed"
    # A STRUCTURAL refusal: the signal lives in ``status=REFUSED_CONTRACT`` →
    # ``exit_code==2`` (the spec's structural-refusal code). The verdict is
    # INCOMPLETE (not FAIL): a FAIL *verdict* is reserved for a COMPUTED "refine it"
    # outcome (evaluate's job) and would short-circuit ``exit_code`` to 1. Nothing
    # was scored here — there was simply no stream to embed.
    return VerbResult(
        verb="embed",
        status=Status.REFUSED_CONTRACT,
        verdict=Verdict.INCOMPLETE,
        tier=Tier.WORKSPACE_WRITE,
        capability_mode=CapabilityMode.SEARCHABLE,
        summary=f"embed REFUSED_CONTRACT on {in_spec}: {msg}",
        diagnostics=[
            Diagnostic(
                contract="C2",
                severity=Severity.ERROR,
                message=msg,
                fix="re-run loom tokenize on an IngestDataset with readable rows",
            )
        ],
        experiment=experiment,
        cost_plan=CostPlan(),
    )


def _payload_dir(store: Any, ref: Any) -> str:
    """The on-disk payload dir for the freshly-minted Embeddings object, derived
    from the PUBLIC store layout (``<objects_dir>/<kind>/<n>/payload``) — the same
    convention :meth:`ObjectStore.put` writes payloads under. Reads only public
    attributes (``store.objects_dir``, ``ref.kind/n``); it never edits or calls
    into store internals (HARD CONSTRAINT: zero edits to store.py). Falls back to a
    workspace ``.loom/embeddings`` dir if the store exposes no ``objects_dir`` (a
    non-local store backend). The safetensors + config.json are written here so the
    artifact travels WITH the object."""
    import os

    objects_dir = getattr(store, "objects_dir", None)
    if objects_dir is not None:
        return os.path.join(str(objects_dir), ref.kind, str(ref.n), "payload")
    ws = os.environ.get("LOOM_WORKSPACE") or os.getcwd()
    return os.path.join(ws, ".loom", "embeddings", f"{ref.kind}-{ref.n}")


@register(
    "embed",
    summary="embed a Corpus locally (PPMI-SVD, CPU, ~$0) — validate a tokenizer "
            "before a GPU-hour",
    tier=Tier.WORKSPACE_WRITE,
    capability_mode=CapabilityMode.SEARCHABLE,
    params=EMBED_PARAMS,
)
def _embed(args: dict[str, Any], ctx: VerbContext) -> VerbResult:
    # Local import: the store is the v0.2 seam, and importing DataObject locally
    # avoids the store↔verb import cycle (HARD INVARIANT).
    from ..store import DataObject

    in_spec = args.get("in") or ""
    dim = int(args.get("dim") or _DEFAULT_DIM)
    window = int(args.get("window") or _DEFAULT_WINDOW)
    experiment = ctx.experiment

    if not in_spec:
        return VerbResult(
            verb="embed",
            status=Status.REFUSED_CONTRACT,
            verdict=Verdict.INCOMPLETE,
            tier=Tier.WORKSPACE_WRITE,
            capability_mode=CapabilityMode.SEARCHABLE,
            summary="embed needs an input Corpus/<n>",
            diagnostics=[
                Diagnostic(
                    contract="C2",
                    severity=Severity.ERROR,
                    message="embed needs an input Corpus pathspec",
                    fix="pass `loom embed Corpus/<n>`",
                )
            ],
            experiment=experiment,
            cost_plan=CostPlan(),
        )

    # --- resolve the Corpus ---------------------------------------------------
    try:
        corpus = ctx.store.get(in_spec)
    except (KeyError, ValueError) as exc:
        return VerbResult(
            verb="embed",
            status=Status.REFUSED_CONTRACT,
            verdict=Verdict.INCOMPLETE,
            tier=Tier.WORKSPACE_WRITE,
            capability_mode=CapabilityMode.SEARCHABLE,
            summary=f"embed could not resolve input {in_spec!r}: {exc}",
            diagnostics=[
                Diagnostic(
                    contract="C2",
                    severity=Severity.ERROR,
                    message=f"could not resolve Corpus {in_spec!r}",
                    fix="run loom tokenize to produce a Corpus/<n> first",
                )
            ],
            experiment=experiment,
            cost_plan=CostPlan(),
        )

    # --- load the token stream via the SAME gate-free path pretrain uses ------
    # _corpus_to_prepared loads the corpus.json payload into a PreparedCorpus
    # (extras{corpus_lines, vocab, token_lines}); _read_token_stream pulls the
    # (token_id_lines, vocab_size) the fit rehearses on. SILENTLY None ⇒ no stream.
    prepared = pretrain._corpus_to_prepared(corpus)
    stream = local_builder._read_token_stream(prepared)
    if stream is None:
        return _refused_no_stream(in_spec, experiment)
    lines, vocab_size = stream

    # --- the deterministic PPMI + Truncated-SVD fit (sklearn random_state=0) ---
    # NEVER routed through LocalModelBuilder.launch / MODEL_BUILDERS / the gate —
    # we call the pure free function directly (no ProgressEvents, no canary, no
    # CheckpointRef, no executor.foreach).
    emb, final_objective = local_builder._fit_ppmi_svd(lines, vocab_size, dim)

    # --- mint the Embeddings ref UP FRONT so the safetensors land in the object's
    # OWN payload dir (the artifact travels WITH the object, never a tempdir). The
    # mint is an atomic counter bump; it does NOT write the object — only put() does.
    ref = ctx.store.new_ref("Embeddings")
    payload_dir = _payload_dir(ctx.store, ref)

    # C5: write a VALID consolidated HF-safetensors dir (model.safetensors weight
    # 'embeddings' + co-located config.json) that safetensors.numpy.load_file
    # re-reads. Row i == vocab id i (no permutation).
    local_builder._write_consolidated_safetensors(
        payload_dir,
        emb,
        vocab_size=vocab_size,
        dim=dim,
        arch={"builder": "local-ppmi-svd", "window": window},
    )

    import os

    safetensors_path = os.path.join(payload_dir, "model.safetensors")
    with open(safetensors_path, "rb") as fh:
        payload_bytes = fh.read()

    sigs = dict(getattr(corpus, "signatures", {}) or {})
    representation_signature = sigs.get("vocab_hash") or sigs.get(
        "representation_signature"
    ) or corpus.content_id

    eobj = DataObject(
        ref=ref,
        kind="Embeddings",
        # Deterministic content address: same corpus + same (dim, window) ⇒ the
        # store dedupes to the existing object (a re-run never forks a twin).
        content_id=f"{corpus.content_id}:embed:dim{dim}:win{window}",
        parents=[corpus.pathspec],
        producer_verb="embed",
        producer_args={"in": in_spec, "dim": dim, "window": window},
        # The cross-port pairing invariant travels WITH the object: evaluate
        # asserts representation_signature == the corpus vocab_hash (C5) before any
        # forward pass over the matrix.
        signatures={
            "representation_signature": representation_signature,
            "vocab_size": int(vocab_size),
            "dim": int(dim),
            "window": int(window),
            "final_objective": float(final_objective),
        },
        verdict=Verdict.PASS,
        status=Status.OK,
        experiment=experiment,
        payload_path=safetensors_path,
        created_at=datetime.now(timezone.utc).isoformat(),
        extras={"safetensors_dir": payload_dir},
    )
    # Persist with the model.safetensors bytes as the payload (idempotent on
    # content_id: a prior identical object wins and its safetensors_dir is returned).
    stored = ctx.store.put(eobj, payload=payload_bytes, payload_name="model.safetensors")

    n_tokens = sum(len(l) for l in lines)
    summary = (
        f"{stored.pathspec}  embeddings {vocab_size}×{dim}  "
        f"(window={window}, n_tokens={n_tokens}, final_objective={final_objective:.6f})"
    )

    data_block = {
        "pathspec": stored.pathspec,
        "corpus": corpus.pathspec,
        "safetensors_dir": stored.extras.get("safetensors_dir", payload_dir),
        "vocab_size": int(vocab_size),
        "dim": int(dim),
        "window": int(window),
        "final_objective": float(final_objective),
        "representation_signature": representation_signature,
    }

    return VerbResult(
        verb="embed",
        status=Status.OK,
        verdict=Verdict.PASS,
        tier=Tier.WORKSPACE_WRITE,
        capability_mode=CapabilityMode.SEARCHABLE,
        summary=summary,
        outputs=[stored.ref],
        diagnostics=[
            Diagnostic(
                contract="EMBED",
                severity=Severity.INFO,
                message=(
                    "local PPMI-SVD rehearsal on a SAMPLE-scoped vocab; the dense "
                    "O(vocab²) co-occurrence ceiling is the vocab size, not dim — "
                    "do not point embed at a full-corpus vocab"
                ),
                fix=None,
            )
        ],
        data=data_block,
        experiment=experiment,
        cost_plan=CostPlan(),
    )


__all__ = ["EMBED_PARAMS"]
