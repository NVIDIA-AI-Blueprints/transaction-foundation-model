"""``prepare`` — THE generic data-representation verb (ARCHITECTURE §8 recast).

This is the generalization of the v0.1 ``tokenize`` verb. Where ``tokenize`` was
hard-wired to the tokenizer engine and to the C1/C3 contract names, ``prepare``
looks up a :class:`~loom.ports.DataRepresentation` adapter by name
(``REPRESENTATIONS[args["representation"]]``, default ``"event-sequence"``) and
drives it through its locked Protocol surface only. Everything that was
representation-specific is now a delegation; everything that was harness
plumbing — the ``content_id`` dedupe, ``new_ref``/``put``, the lineage
``signatures`` block, and the ``VerbResult`` assembly — is reused verbatim.

THE ONE GENERALIZED LINE (the §8 fix). The v0.1 verb decided the corpus
write-refusal by *naming contracts and reading tokenizer-shaped attributes*
(``not (compiled.report.injective and compiled.report.dense)`` for C1; a scan
for ``d.contract == "C3"`` for the grammar fail). ``prepare`` replaces both with
a SINGLE contract-name-agnostic gate::

    diags = repr.contracts(compiled)
    if not repr.representation_passed(compiled):      # == any ERROR-severity Diagnostic
        return VerbResult(status=REFUSED_CONTRACT, …, diagnostics=diags)  # no Corpus

The verb never knows the strings ``"C1"``/``"C3"`` nor the ``report.injective/
.dense`` shape — a second representation (encoder-MLM, vision-patches) folds its
own failures into ERROR-severity ``Diagnostic`` cards under its own contract
names and is gated identically, with zero harness edit (ARCHITECTURE §7/§8/§9).
This is behavior-preserving for ``event-sequence``: ``representation_passed`` is
``compiled.report.passed``, which ``ContractReport.add`` already flips to
``False`` on every ERROR-severity card (C1 injectivity/density, C3 grammar) — so
the exact same corpora are refused.

``tokenize`` survives (ARCHITECTURE §4): it is re-registered as a thin binding to
this same implementation with ``representation`` forced to ``"event-sequence"``
and the envelope ``verb`` field pinned to ``"tokenize"`` so ``loom tokenize
--json`` and ``dispatch("loom.tokenize", …)`` stay BYTE-IDENTICAL to v0.1 (the
locked dual-driver invariant). See ``verbs/__init__.py`` for the wiring.

Workspace-write tier: cheap, CPU, no GPU gating — the envelope still carries
tier / capability_mode / cost_plan so the downstream gating model reads the same
shape (the gated launch lives in ``pretrain``, not here).
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd

from ..ports import REPRESENTATIONS, DataRepresentation
from ..registry import VerbContext
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

#: The default representation an unqualified ``prepare`` (and every ``tokenize``)
#: selects — TFM out of the box (ARCHITECTURE §4).
DEFAULT_REPRESENTATION = "event-sequence"


PREPARE_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "representation": {
            "type": "string",
            "description": "data-representation adapter name (default event-sequence)",
        },
        "in": {"type": "string", "description": "input IngestDataset/<n> pathspec"},
        "spec": {"type": "string",
                 "description": "a proposed/edited tokenizer field-map to compile instead of a "
                                "preset: a TokenizerSpec/<n> pathspec (from `loom propose`) OR a "
                                "path to a loom-fieldmap/1 YAML/JSON file. When set, the preset is "
                                "ignored and the custom field-map is compiled + C1/C2/C3-checked."},
        "preset": {"type": "string", "description": "representation preset (event-sequence: financial | chain)",
                   "enum": ["financial", "chain"]},
        "include_time_delta": {"type": "boolean",
                               "description": "add the TDIF time-delta field (T1)"},
        "merchant_hash_size": {"type": "integer", "description": "merchant hash buckets (financial)"},
        "amount_strategy": {"type": "string", "enum": ["fixed", "quantile", "kmeans"],
                            "description": "amount binning; quantile/kmeans is a C2 fitted artifact"},
        "drop_step": {"type": "string", "description": "drop a named step (e.g. cust) — T2"},
        "reorder_step": {"type": "string", "description": "reorder a step (e.g. card:first) — T5"},
        "no_identity_token": {"type": "boolean",
                              "description": "chain: keep wallet identity out of the vocab (T2)"},
        "eval_split": {"type": "string", "description": "temporal | entity-disjoint (C6)"},
        "context_len": {"type": "integer", "description": "model context length (default 4096)"},
        "confirm_token": {"type": "string", "description": "agent second-call confirm token (§5.3)"},
    },
}

#: ``tokenize`` keeps the v0.1 schema verbatim (no ``representation`` flag — it is
#: pinned). This is the exact dict the v0.1 verb registered, so the agent tool's
#: ``input_schema`` for ``loom.tokenize`` is unchanged.
TOKENIZE_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "in": {"type": "string", "description": "input IngestDataset/<n> pathspec"},
        "spec": {"type": "string",
                 "description": "a proposed/edited tokenizer field-map to compile instead of a "
                                "preset: a TokenizerSpec/<n> pathspec (from `loom propose`) OR a "
                                "path to a loom-fieldmap/1 YAML/JSON file. When set, the preset is "
                                "ignored and the custom field-map is compiled + C1/C2/C3-checked."},
        "preset": {"type": "string", "description": "tokenizer preset: financial | chain",
                   "enum": ["financial", "chain"]},
        "include_time_delta": {"type": "boolean",
                               "description": "add the TDIF time-delta field (T1)"},
        "merchant_hash_size": {"type": "integer", "description": "merchant hash buckets (financial)"},
        "amount_strategy": {"type": "string", "enum": ["fixed", "quantile", "kmeans"],
                            "description": "amount binning; quantile/kmeans is a C2 fitted artifact"},
        "drop_step": {"type": "string", "description": "drop a named step (e.g. cust) — T2"},
        "reorder_step": {"type": "string", "description": "reorder a step (e.g. card:first) — T5"},
        "no_identity_token": {"type": "boolean",
                              "description": "chain: keep wallet identity out of the vocab (T2)"},
        "eval_split": {"type": "string", "description": "temporal | entity-disjoint (C6)"},
        "context_len": {"type": "integer", "description": "model context length (default 4096)"},
        "confirm_token": {"type": "string", "description": "agent second-call confirm token (§5.3)"},
    },
}


# ---------------------------------------------------------------------------
# Representation resolution — the registry lookup that replaces the hard-wired
# engine import (ARCHITECTURE §8: the ``spec.preset=="chain"`` switch becomes the
# REPRESENTATIONS lookup).
# ---------------------------------------------------------------------------


#: Adapters self-register on import but are NOT imported by ``import loom`` (the
#: registries start empty). Map the bundled adapter names to their modules so
#: ``prepare`` can best-effort import the one it needs, exactly as ``pretrain``
#: does. Unknown names are never auto-imported.
_BUNDLED_ADAPTER_MODULES = {
    DEFAULT_REPRESENTATION: "loom.adapters.event_sequence",
}


def _resolve_representation(name: str) -> Optional[DataRepresentation]:
    """Return the registered adapter for ``name``, importing the bundled adapter
    module on demand if it has not registered yet.

    The bundled ``event-sequence`` adapter (``loom.adapters.event_sequence``) is a
    pure CPU-engine delegation that imports cleanly with zero banned deps, so the
    on-demand import always lands it. Any unknown name returns ``None`` → a clean
    REFUSED_CONTRACT (no Corpus). There is no engine-backed shim fallback: the
    adapter IS the delegation, so a missing adapter is a refusal, not a silent
    re-implementation (ARCHITECTURE §5.2)."""
    adapter = REPRESENTATIONS.get(name)
    if adapter is not None:
        return adapter
    module = _BUNDLED_ADAPTER_MODULES.get(name)
    if module is not None:
        try:
            __import__(module)
        except Exception:  # pragma: no cover - defensive; bundled adapter imports cleanly
            pass
        adapter = REPRESENTATIONS.get(name)
        if adapter is not None:
            return adapter
    return None


# ---------------------------------------------------------------------------
# Input loading + corpus materialization — representation-agnostic plumbing
# carried over verbatim from the v0.1 verb (the dataset is config-orthogonal for
# the vocab, so a missing/unwired store degrades to a compile-only PASS).
# ---------------------------------------------------------------------------


def _load_rows(
    in_path: Optional[str], ctx: VerbContext
) -> tuple[Optional[pd.DataFrame], Optional[str], list[str]]:
    """Best-effort load of the input dataset's rows for corpus materialization.

    Returns ``(df, dataset_pathspec, parents)``. The vocabulary is config-only, so
    a missing/unreadable dataset is NOT fatal — the Corpus still carries the
    compiled vocab + signature, and corpus-line materialization is skipped. An
    in-memory frame may be injected via ``ctx.extras["dataframe"]`` (test/agent)."""
    injected = ctx.extras.get("dataframe") if ctx.extras else None
    if injected is not None:
        df = injected if isinstance(injected, pd.DataFrame) else pd.DataFrame(injected)
        return df, in_path, ([in_path] if in_path else [])
    if not in_path:
        return None, None, []
    try:
        obj = ctx.store.get(in_path)
    except (KeyError, FileNotFoundError, NotImplementedError, AttributeError):
        return None, None, []
    except Exception:  # pragma: no cover - any store backend error degrades gracefully
        return None, None, []
    parents = [obj.pathspec]
    if obj.payload_path:
        try:
            return pd.read_csv(obj.payload_path), obj.pathspec, parents
        except (FileNotFoundError, OSError, ValueError):
            return None, obj.pathspec, parents
    return None, obj.pathspec, parents


def _resolve_fieldmap_spec(spec_ref: Any, ctx: VerbContext) -> Any:
    """Resolve the ``--spec`` reference to a field-map dict the adapter can compile.

    ``--spec`` accepts two forms (the build brief's `<file-or-pathspec>`):

    * a ``TokenizerSpec/<n>`` pathspec — the proposal object ``loom propose``
      wrote. The editable field-map lives on the object (``extras.proposal.fieldmap``,
      with the YAML mirror as the heavy payload). This is the path that needs
      ``ctx.store``, which the adapter's pure ``_build_spec`` does NOT have — so the
      verb layer (which holds ``ctx``) resolves it here and hands the adapter a
      plain dict, preserving the port boundary (Ground 2 §3d).
    * a filesystem path to a ``loom-fieldmap/1`` YAML/JSON file — a human-edited
      spec. Read + parse it here so the adapter only ever sees a dict/str.

    Anything else (already a dict, or unresolvable) is returned UNCHANGED so the
    adapter can decide — an unknown/garbled spec then flows into the same
    contract-checked refusal path, never a silent-broken corpus (INVARIANT #2).
    This function only runs when ``--spec`` was passed; the preset path never
    touches it, so ``tokenize --preset financial|chain`` stays byte-identical
    (INVARIANT #1)."""
    # Already a parsed field-map (e.g. an in-process/agent call) → pass through.
    if isinstance(spec_ref, dict):
        return spec_ref
    if not isinstance(spec_ref, str) or not spec_ref:
        return spec_ref

    # A TokenizerSpec/<n> pathspec → resolve via the store and lift the field-map.
    looks_like_pathspec = "/" in spec_ref and spec_ref.split("/", 1)[0] == "TokenizerSpec"
    if looks_like_pathspec:
        try:
            obj = ctx.store.get(spec_ref)
        except (KeyError, ValueError, FileNotFoundError, NotImplementedError, AttributeError):
            return spec_ref  # leave it for the adapter to refuse with a named diff
        proposal = (getattr(obj, "extras", {}) or {}).get("proposal") or {}
        fieldmap = proposal.get("fieldmap")
        if isinstance(fieldmap, dict) and fieldmap:
            return fieldmap
        # Fall back to the heavy YAML payload the proposal persisted.
        payload_path = getattr(obj, "payload_path", None)
        if payload_path:
            loaded = _load_fieldmap_file(payload_path)
            if loaded is not None:
                return loaded
        return spec_ref

    # Otherwise treat it as a file path (human-edited YAML/JSON field-map).
    loaded = _load_fieldmap_file(spec_ref)
    return loaded if loaded is not None else spec_ref


def _load_fieldmap_file(path: str) -> Optional[dict[str, Any]]:
    """Parse a ``loom-fieldmap/1`` YAML or JSON file into a dict, or ``None`` on
    any read/parse failure (the caller then leaves the raw ref for the adapter to
    refuse with a named diff rather than crashing the verb)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except (FileNotFoundError, OSError):
        return None
    try:
        import yaml

        data = yaml.safe_load(text)
    except Exception:  # pragma: no cover - YAML is a declared dep; JSON is a subset
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return None
    return data if isinstance(data, dict) else None


def _derived_numbers(spec, compiled) -> dict[str, Any]:
    """The human-facing summary numbers carried on the envelope ``data`` block.

    For the event-sequence representation (``CompiledTokenizer``/``TokenizerSpec``)
    this is EXACTLY the v0.1 ``tokenize.py`` ``derived`` dict — byte-identity is
    preserved. For any other representation whose ``compiled``/``spec`` lacks those
    attributes (a probe, an MLM rep mid-build), it degrades to the attributes that
    exist, so the contract-name-agnostic refusal never requires tokenizer shape."""
    out: dict[str, Any] = {}
    preset = getattr(spec, "preset", None)
    if preset is not None:
        out["preset"] = preset
    for attr in ("vocab_size", "vocab_hash", "tokens_per_txn", "chunk_size", "context_len"):
        if hasattr(compiled, attr):
            out[attr] = getattr(compiled, attr)
    step_names = getattr(spec, "step_names", None)
    if callable(step_names):
        out["step_names"] = step_names()
    report = getattr(compiled, "report", None)
    if report is not None and hasattr(report, "has_fitted_artifact"):
        out["has_fitted_artifact"] = report.has_fitted_artifact
    return out


def _materialize_corpus(compiled, df: pd.DataFrame) -> tuple[list[str], int]:
    """Assemble corpus lines for the event-sequence representation. Delegates to
    the engine's public ``materialize_corpus_lines`` so the verb and the
    conformance tests share one corpus-assembly path."""
    from ..engine import materialize_corpus_lines

    return materialize_corpus_lines(compiled, df)


# ---------------------------------------------------------------------------
# Result envelope builders — reused VERBATIM from the v0.1 verb so the PASS
# envelope is byte-identical. The ``verb`` field is parameterized so the
# ``tokenize`` binding emits ``verb="tokenize"`` (byte-identity) and ``prepare``
# emits ``verb="prepare"``.
# ---------------------------------------------------------------------------


def _compile_only_result(
    verb_name: str,
    compiled,
    diagnostics: list[Diagnostic],
    derived: dict[str, Any],
    parents: list[str],
    experiment: Optional[str],
) -> VerbResult:
    """A deterministic PASS envelope when the store cannot persist (not yet wired)."""
    return VerbResult(
        verb=verb_name,
        status=Status.OK,
        verdict=Verdict.PASS,
        tier=Tier.WORKSPACE_WRITE,
        capability_mode=CapabilityMode.NONE,
        summary=(
            f"compiled OK (no store) verdict=PASS vocab={compiled.vocab_size} "
            f"tokens/txn={compiled.tokens_per_txn} chunk_size={compiled.chunk_size} "
            f"sig={compiled.vocab_hash[:18]}…"
        ),
        outputs=[],
        diagnostics=diagnostics,
        data={**derived, "wrote_corpus": False, "store_unavailable": True,
              "parents": parents},
        experiment=experiment,
        cost_plan=CostPlan(),
    )


def _corpus_result(
    verb_name: str,
    ref,
    compiled,
    diagnostics: list[Diagnostic],
    derived: dict[str, Any],
    content_id: str,
    n_lines: int,
    n_txns: int,
    parents: list[str],
    experiment: Optional[str],
) -> VerbResult:
    """The single PASS envelope for a written/idempotent Corpus (byte-identical
    across the fresh-write and idempotent-hit paths, and across both faces)."""
    summary = (
        f"{ref.pathspec} verdict=PASS vocab={compiled.vocab_size} "
        f"tokens/txn={compiled.tokens_per_txn} chunk_size={compiled.chunk_size} "
        f"sig={compiled.vocab_hash[:18]}…"
    )
    return VerbResult(
        verb=verb_name,
        status=Status.OK,
        verdict=Verdict.PASS,
        tier=Tier.WORKSPACE_WRITE,
        capability_mode=CapabilityMode.NONE,
        summary=summary,
        outputs=[ref],
        diagnostics=diagnostics,
        data={
            **derived,
            "pathspec": ref.pathspec,
            "wrote_corpus": True,
            "content_id": content_id,
            "n_lines": n_lines,
            "n_txns": n_txns,
            "parents": parents,
        },
        experiment=experiment,
        cost_plan=CostPlan(),
    )


def _unknown_representation_result(
    verb_name: str, representation: str, experiment: Optional[str]
) -> VerbResult:
    """No adapter registered under ``representation`` → a clean refusal, no write.

    Surfaced as a ``Diagnostic(contract="REPRESENTATION")`` card (a harness-level
    contract name, not a port-local one) so the agent/human gets a named diff, not
    a stack trace."""
    known = sorted(REPRESENTATIONS.keys()) or [DEFAULT_REPRESENTATION]
    diag = Diagnostic(
        contract="REPRESENTATION",
        severity=Severity.ERROR,
        message=(
            f"unknown data-representation {representation!r} — no adapter is "
            f"registered under that name."
        ),
        fix=f"use one of: {', '.join(known)}.",
        data={"requested": representation, "known": known},
    )
    return VerbResult(
        verb=verb_name,
        status=Status.REFUSED_CONTRACT,
        verdict=Verdict.FAIL,
        tier=Tier.WORKSPACE_WRITE,
        capability_mode=CapabilityMode.NONE,
        summary=(
            f"REFUSED_CONTRACT: unknown representation {representation!r} — no "
            "Corpus written."
        ),
        outputs=[],
        diagnostics=[diag],
        data={"representation": representation, "wrote_corpus": False},
        experiment=experiment,
        cost_plan=CostPlan(),
    )


def _generic_refusal_summary(representation: str, diagnostics: list[Diagnostic]) -> str:
    """The contract-name-agnostic fallback summary — built entirely from the PORT's
    own ERROR-severity diagnostics. The verb does NOT name any contract; whatever
    contract strings appear here were minted by the representation, surfaced
    verbatim. A representation with no C1/C3 is phrased by this exact path under its
    own names. Used whenever the port does NOT supply the optional
    ``refusal_summary`` hook (any stub, or a novel representation)."""
    errors = [d for d in diagnostics if d.severity is Severity.ERROR]
    named = "; ".join(f"{d.contract}: {d.message}" for d in errors) or (
        "a representation contract failed"
    )
    return (
        f"REFUSED_CONTRACT: {representation} representation contract(s) failed "
        f"({named}) — no Corpus written."
    )


def _refused_contract_result(
    verb_name: str,
    representation: str,
    diagnostics: list[Diagnostic],
    derived: dict[str, Any],
    experiment: Optional[str],
    *,
    repr_: Any = None,
    compiled: Any = None,
) -> VerbResult:
    """THE generalized, contract-name-agnostic write-refusal (the §8 fix).

    The DECISION to refuse is made name-agnostically by the caller
    (``not repr_.representation_passed(compiled)``); this builder only RENDERS the
    one-line summary for a refusal that has ALREADY been made. To keep
    ``loom tokenize --json`` byte-identical to v0.1 on the REFUSED envelope
    (invariant #3, not merely the PASS envelope), the renderer first OFFERS the port
    an optional, purely-cosmetic ``refusal_summary(compiled, diagnostics)`` hook: a
    representation may mirror its v0.1 wording verbatim from its OWN already-refused
    cards. The verb never decides anything by it — it falls back to the generic,
    contract-name-agnostic rendering (built from the port's ERROR cards) for any
    port or stub that omits the hook. The write-gate stays name-agnostic; this
    helper never reads ``report.injective/.dense`` nor names "C1"/"C3"."""
    summary: Optional[str] = None
    hook = getattr(repr_, "refusal_summary", None)
    if callable(hook) and compiled is not None:
        try:
            summary = hook(compiled, diagnostics)
        except Exception:  # pragma: no cover - a cosmetic hook never breaks refusal
            summary = None
    if not summary:
        summary = _generic_refusal_summary(representation, diagnostics)
    return VerbResult(
        verb=verb_name,
        status=Status.REFUSED_CONTRACT,
        verdict=Verdict.FAIL,
        tier=Tier.WORKSPACE_WRITE,
        capability_mode=CapabilityMode.NONE,
        summary=summary,
        outputs=[],
        diagnostics=diagnostics,
        data={**derived, "wrote_corpus": False},
        experiment=experiment,
        cost_plan=CostPlan(),
    )


# ---------------------------------------------------------------------------
# The shared implementation. ``prepare`` and the ``tokenize`` binding both route
# here; ``verb_name`` pins the envelope ``verb`` field (byte-identity).
# ---------------------------------------------------------------------------


def _prepare_impl(
    args: dict[str, Any], ctx: VerbContext, *, verb_name: str
) -> VerbResult:
    from ..store import DataObject  # local import: store is the v0.2 seam

    representation = (args.get("representation") or DEFAULT_REPRESENTATION)
    in_path = args.get("in") or None
    context_len = int(args.get("context_len") or 4096)
    experiment = ctx.experiment

    repr_ = _resolve_representation(representation)
    if repr_ is None:
        return _unknown_representation_result(verb_name, representation, experiment)

    # --- NEW additive --spec path: resolve a TokenizerSpec/<n> pathspec (via the
    # store — the adapter's pure build_spec has no ctx) or a YAML/JSON file into a
    # field-map dict the adapter compiles through `spec_from_field_map`. Runs ONLY
    # when --spec was passed; the preset branch below is byte-identical to v0.1
    # otherwise (INVARIANT #1). A bad/unresolvable spec flows into the SAME
    # C1/C2/C3 refusal as any preset spec (INVARIANT #2) — never silently shipped.
    spec_ref = args.get("spec")
    if spec_ref:
        args = dict(args)
        args["spec"] = _resolve_fieldmap_spec(spec_ref, ctx)

    # --- compile the spec (data-free) via the PORT + collect its contract cards.
    spec = repr_.build_spec(dict(args))
    compiled = repr_.compile(spec, context_len=context_len)
    diagnostics = repr_.contracts(compiled)

    # === THE ONE GENERALIZED LINE (ARCHITECTURE §8) ===========================
    # The corpus write-gate, FIRST — before the verb reads any representation-
    # shaped attribute. Contract-NAME-AGNOSTIC: refuse iff the port reports ANY
    # ERROR-severity Diagnostic. The verb never names "C1"/"C3" nor reads
    # report.injective/.dense — it asks the port for its verdict. Behavior-
    # preserving for event-sequence (== compiled.report.passed).
    if not repr_.representation_passed(compiled):
        return _refused_contract_result(
            verb_name, representation, diagnostics,
            _derived_numbers(spec, compiled), experiment,
            repr_=repr_, compiled=compiled,
        )
    # ==========================================================================

    derived = _derived_numbers(spec, compiled)

    # --- best-effort corpus materialization (vocab is config-only either way) -
    df, dataset_pathspec, parents = _load_rows(in_path, ctx)
    corpus_lines: list[str] = []
    n_txns = 0
    if df is not None:
        try:
            corpus_lines, n_txns = _materialize_corpus(compiled, df)
        except Exception as exc:  # pragma: no cover - defensive; vocab still valid
            diagnostics = diagnostics + [
                Diagnostic(
                    contract="C3",
                    severity=Severity.WARNING,
                    message=f"corpus materialization skipped: {exc}",
                    fix="check the dataset columns match the preset's source fields.",
                )
            ]
    else:
        diagnostics = diagnostics + [
            Diagnostic(
                contract="C3",
                severity=Severity.INFO,
                message=(
                    "no input rows materialized (dataset missing/empty) — Corpus "
                    "carries the compiled vocab + signature only."
                ),
                fix="pass an IngestDataset/<n> with readable rows to emit corpus lines.",
            )
        ]

    # --- content address: spec hash over the compiled signature ---------------
    spec_hash = compiled.vocab_hash
    source_fp = dataset_pathspec or (in_path or "<no-input>")

    try:
        content_id = ctx.store.content_id(source_fp, spec_hash)
        existing = ctx.store.find_by_content(content_id)
    except (NotImplementedError, AttributeError):
        return _compile_only_result(
            verb_name, compiled, diagnostics, derived, parents, experiment
        )

    # idempotency (§6): same input+spec → return the EXISTING Corpus, byte-
    # identical to the fresh-write envelope.
    if existing is not None:
        n_lines = int(existing.extras.get("n_lines", len(corpus_lines)))
        n_txns = int(existing.extras.get("n_txns", n_txns))
        return _corpus_result(
            verb_name, existing.ref, compiled, diagnostics, derived, content_id,
            n_lines, n_txns, parents, experiment,
        )

    # --- persist the Corpus (vocab + signature + derived numbers) -------------
    try:
        ref = ctx.store.new_ref("Corpus")
    except (NotImplementedError, AttributeError):
        return _compile_only_result(
            verb_name, compiled, diagnostics, derived, parents, experiment
        )

    # The harness handoff signatures come from the PORT (representation +
    # representation_signature travel with the object — invariant #7), merged with
    # the v0.1 lineage keys so embed/pretrain assert them.
    port_sigs = repr_.signatures(compiled)
    payload = {
        "vocab": compiled.vocab,
        "vocab_hash": compiled.vocab_hash,
        "vocab_size": compiled.vocab_size,
        "tokens_per_txn": compiled.tokens_per_txn,
        "chunk_size": compiled.chunk_size,
        "context_len": compiled.context_len,
        "preset": spec.preset,
        "step_names": spec.step_names(),
        "grammar": {
            "bos": compiled.grammar.bos,
            "eos": compiled.grammar.eos,
            "sep": compiled.grammar.sep,
            "chunk_size": compiled.grammar.chunk_size,
            "tokens_per_txn": compiled.grammar.tokens_per_txn,
            "context_len": compiled.grammar.context_len,
        },
        "fitted_state": compiled.fitted_state,
        "n_lines": len(corpus_lines),
        "n_txns": n_txns,
        "corpus_lines": corpus_lines,
    }
    obj = DataObject(
        ref=ref,
        kind="Corpus",
        content_id=content_id,
        parents=parents,
        producer_verb=verb_name,
        producer_args={
            "representation": representation,
            "in": in_path,
            "preset": spec.preset,
            "include_time_delta": bool(args.get("include_time_delta", False)),
            "merchant_hash_size": args.get("merchant_hash_size"),
            "amount_strategy": getattr(
                getattr(spec, "amount_strategy", None), "value", None
            ),
            "drop_step": args.get("drop_step"),
            "context_len": context_len,
        },
        # The signature travels WITH the object — embed/pretrain assert it. Carries
        # the new harness-level ``representation_signature`` (the §7 pairing
        # invariant) alongside the v0.1 keys.
        signatures={
            **port_sigs,
            "vocab_hash": compiled.vocab_hash,
            "vocab_size": compiled.vocab_size,
            "tokens_per_txn": compiled.tokens_per_txn,
            "chunk_size": compiled.chunk_size,
            "context_len": compiled.context_len,
            "has_fitted_artifact": compiled.report.has_fitted_artifact,
            "encode_path": spec.preset,
        },
        verdict=Verdict.PASS,
        status=Status.OK,
        experiment=experiment,
        extras={
            "representation": representation,
            "preset": spec.preset,
            "step_names": spec.step_names(),
            "n_lines": len(corpus_lines),
            "n_txns": n_txns,
            "contract_report": {
                "passed": compiled.report.passed,
                "injective": compiled.report.injective,
                "dense": compiled.report.dense,
                "deterministic": compiled.report.deterministic,
                "has_fitted_artifact": compiled.report.has_fitted_artifact,
                "diagnostics": [d.to_dict() for d in diagnostics],
            },
        },
    )
    try:
        stored = ctx.store.put(obj, payload=json.dumps(payload), payload_name="corpus.json")
    except (NotImplementedError, AttributeError):
        return _compile_only_result(
            verb_name, compiled, diagnostics, derived, parents, experiment
        )

    return _corpus_result(
        verb_name, stored.ref, compiled, diagnostics, derived, content_id,
        len(corpus_lines), n_txns, parents, experiment,
    )


# ---------------------------------------------------------------------------
# The two registered entry points. Registration happens in verbs/__init__.py so
# the import-order guard there controls landing; these are the bare ``fn``s.
# ---------------------------------------------------------------------------


def prepare_fn(args: dict[str, Any], ctx: VerbContext) -> VerbResult:
    """``loom prepare`` — generic; ``representation`` selects the adapter."""
    return _prepare_impl(args, ctx, verb_name="prepare")


def tokenize_fn(args: dict[str, Any], ctx: VerbContext) -> VerbResult:
    """``loom tokenize`` — the bound alias: ``prepare`` with ``representation``
    pinned to ``event-sequence`` and the envelope ``verb`` field pinned to
    ``tokenize`` so v0.1 byte-identity holds on both faces."""
    bound = dict(args)
    bound["representation"] = DEFAULT_REPRESENTATION
    return _prepare_impl(bound, ctx, verb_name="tokenize")
