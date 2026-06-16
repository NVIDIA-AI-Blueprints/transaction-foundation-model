"""``propose`` — the EDA-grounded field→strategy PROPOSER (the BYO-schema seam).

A first-time user with their OWN tabular schema (``account_id, txn_amount, mcc,
channel, dr_cr, balance, txn_ts``) cannot tokenize through the two hardcoded
presets (``financial`` = TabFormer's exact columns, ``chain`` = DEX). ``propose``
closes that gap: it reads an already-ingested ``IngestDataset/<n>`` (its sniffed
schema, its EDA leakage flags, and the chosen ``entity``/``event``/``target``),
runs the pure field→strategy classifier (``loom.engine.propose.propose_spec`` —
the rules from ``docs/04-data/08-from-raw-data-to-training-run.md`` §1–§3), and
emits a **reviewable, editable tokenizer SPEC** (a declarative field-map) plus a
card that states, per field, WHY each column earned a token — and, critically,
WHY the entity (T2), the target (leakage), and any dropped column are EXCLUDED
from the vocab.

THE FLOW (the build brief):
    loom ingest  (schema sniff + EDA leakage flags — EXISTS)
      → loom propose  (THIS verb: field→strategy proposer → editable SPEC)
      → human reviews / tweaks the spec
      → loom tokenize --spec <TokenizerSpec/n | file>  (compiles THAT field-map,
        contract-checked by C1/C2/C3, → a Corpus)

This verb owns NO classification logic — that is the pure engine function
``engine.propose.propose_spec`` (no NeMo/torch, CPU-only). ``propose`` is the
HARNESS plumbing: resolve the input object, hand the classifier its inputs,
persist the proposed field-map as a content-addressed ``TokenizerSpec/<n>``
data-object (with the editable YAML as its heavy payload so a human can open and
tweak it), and assemble the dual-driver ``VerbResult`` card.

HARD INVARIANT #4 (the core correctness of the flow): the proposal EXCLUDES the
entity column (T2 — identity comes from history, not an embedding) and the target
column (leakage), and **surfaces every consumed EDA flag** as part of the
proposal so the human sees *why* each column was dropped and can override. The
verb renders those exclusions + their originating ``Diagnostic`` cards verbatim.

Workspace-write tier: cheap, CPU, no GPU gating — the proposal is a data-light
compile-time artifact (it reads row cardinalities to size hash buckets / apply
the "earns a token" occupancy gate, but emits no corpus).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

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

PROPOSE_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "in": {
            "type": "string",
            "description": "input IngestDataset/<n> pathspec to propose a tokenizer spec for",
        },
        "entity": {
            "type": "string",
            "description": "override the grouping entity column (the sequence owner; "
                           "EXCLUDED from the vocab, T2). Defaults to the entity pinned at ingest.",
        },
        "event": {
            "type": "string",
            "description": "override the event/timestamp column. Defaults to the one pinned at ingest.",
        },
        "target": {
            "type": "string",
            "description": "override the label column (EXCLUDED from the vocab as leakage). "
                           "Defaults to the target pinned at ingest.",
        },
        "context_len": {
            "type": "integer",
            "description": "model context length used to derive chunk_size (default 4096)",
        },
        "confirm_token": {"type": "string", "description": "agent second-call confirm token (§5.3)"},
    },
}


# ---------------------------------------------------------------------------
# Result-envelope helpers — a clean REFUSED/FAIL envelope (no proposal written)
# kept byte-shape-consistent with the other workspace-write verbs.
# ---------------------------------------------------------------------------


def _refused(message: str, fix: str, experiment: Optional[str]) -> VerbResult:
    """A structural refusal (bad/missing input) surfaced as a named diagnostic —
    no ``TokenizerSpec`` written."""
    return VerbResult(
        verb="propose",
        status=Status.FAIL,
        verdict=Verdict.FAIL,
        tier=Tier.WORKSPACE_WRITE,
        capability_mode=CapabilityMode.NONE,
        summary=f"propose refused: {message}",
        outputs=[],
        diagnostics=[
            Diagnostic(
                contract="PROPOSE",
                severity=Severity.ERROR,
                message=message,
                fix=fix,
            )
        ],
        data={"wrote_spec": False},
        experiment=experiment,
        cost_plan=CostPlan(),
    )


# ---------------------------------------------------------------------------
# Reading the proposal back — the classifier may return a small dataclass OR a
# plain dict; read both flexibly so Task-C is robust to the engine's final shape.
# The CONTRACT (agreed with engine.propose.propose_spec) is a result carrying:
#   .fieldmap        the declarative field-map dict (the editable SPEC)
#   .included        [{name, source, strategy, count, rationale}, …]
#   .excluded        [{column, reason, diagnostic?}, …]   (entity/target/dropped)
#   .vocab_size      hand-counted int
#   .tokens_per_event int
#   .chunk_size      int  (== context_len // (tokens_per_event + 1))
# ---------------------------------------------------------------------------


def _attr(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` off a dataclass-like result OR a plain dict, with a default."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _load_rows(obj: Any) -> Any:
    """Best-effort read of the IngestDataset rows payload (``rows.csv``) as a frame.

    Mirrors ``prepare._load_rows``: the ``DataObject.payload_path`` points at the
    full frame the proposer needs to enumerate the observed categorical values. A
    missing/unreadable payload is NOT fatal — the classifier degrades to an
    ``n_values``-only proposal — so any read error returns ``None``."""
    payload_path = getattr(obj, "payload_path", None)
    if not payload_path:
        return None
    try:
        import pandas as pd

        return pd.read_csv(payload_path)
    except Exception:  # pragma: no cover - defensive; an empty corpus is acceptable
        return None


def _diag_from_flag(flag: Any) -> Optional[Diagnostic]:
    """Rebuild a ``Diagnostic`` card from a serialized EDA flag dict (the form
    ``ingest`` persisted under ``ingest_report.eda_diagnostics``) so the proposal
    surfaces the ORIGINATING leakage flag verbatim (HARD INVARIANT #4). Accepts
    an already-constructed ``Diagnostic`` too."""
    if flag is None:
        return None
    if isinstance(flag, Diagnostic):
        return flag
    if not isinstance(flag, dict):
        return None
    try:
        return Diagnostic(
            contract=flag.get("contract", "EDA"),
            severity=Severity(flag.get("severity", "warning")),
            message=flag.get("message", ""),
            fix=flag.get("fix"),
            data=flag.get("data", {}) or {},
        )
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# The verb.
# ---------------------------------------------------------------------------


@register(
    "propose",
    summary="propose an editable tokenizer spec for a NOVEL schema "
            "(EDA-grounded field→strategy map; entity/target excluded; vocab/chunk estimate)",
    tier=Tier.WORKSPACE_WRITE,
    capability_mode=CapabilityMode.NONE,
    params=PROPOSE_PARAMS,
)
def _propose(args: dict[str, Any], ctx: VerbContext) -> VerbResult:
    from ..store import DataObject  # local import: store is the v0.2 seam
    from ..engine import propose as engine_propose  # the pure classifier (Task A)

    in_spec = args.get("in") or ""
    context_len = int(args.get("context_len") or 4096)
    experiment = ctx.experiment

    if not in_spec:
        return _refused(
            "propose needs an input dataset (IngestDataset/<n>)",
            "ingest the source first, then `loom propose IngestDataset/<n> --experiment <id>`",
            experiment,
        )

    # --- resolve the IngestDataset object (schema + EDA flags + entity/event) --
    try:
        obj = ctx.store.get(in_spec)
    except (KeyError, ValueError) as exc:
        return _refused(
            f"propose could not resolve input {in_spec!r}: {exc}",
            "ingest the dataset first; propose reads an IngestDataset/<n>",
            experiment,
        )

    extras = getattr(obj, "extras", {}) or {}
    schema = extras.get("schema") or {}
    if not schema or not schema.get("columns"):
        return _refused(
            f"propose found no sniffed schema on {in_spec} — not an IngestDataset?",
            "re-ingest the source so the schema sniff + EDA flags are persisted",
            experiment,
        )

    # entity/event/target: the CLI/agent override wins, else what ingest pinned.
    entity = args.get("entity") or extras.get("entity")
    event = args.get("event") or extras.get("event")
    target = args.get("target") or extras.get("target")

    # The EDA leakage flags ingest already computed — the proposer CONSUMES these
    # (it does not re-run the heuristics), so the same flags drive the exclusions
    # the human sees on the card (HARD INVARIANT #4 / the eda.py consumption contract).
    ingest_report = extras.get("ingest_report") or {}
    eda_flags = ingest_report.get("eda_diagnostics") or []

    # The ingested rows frame (the IngestDataset payload = rows.csv). The classifier
    # enumerates the OBSERVED distinct values of every low-card categorical off this
    # frame so the proposed `mapping` fields carry a REAL `values: [...]` list — the
    # schema sniff alone carries only `n_unique` (a count), which would collapse every
    # categorical value to the single default token at tokenize time. Best-effort: a
    # missing/unreadable payload degrades to an n_values-only proposal (still valid;
    # the human fills in the values), never a hard failure.
    rows = _load_rows(obj)

    # --- run the pure classifier (Task A) -------------------------------------
    # ``propose_spec`` is keyword-only and PURE: it sizes hash buckets / applies
    # the "earns a token" occupancy gate from ``schema["n_rows"]`` + per-column
    # ``n_unique`` (the sniffed cardinalities), and reads the rows frame ONLY to
    # enumerate the real categorical value lists (config-only, C2-clean).
    try:
        proposal = engine_propose.propose_spec(
            schema=schema,
            eda_flags=eda_flags,
            entity=entity,
            event=event,
            target=target,
            context_len=context_len,
            rows=rows,
        )
    except Exception as exc:  # pragma: no cover - defensive; classifier is pure
        return _refused(
            f"propose classifier failed on {in_spec}: {exc}",
            "check the ingested schema/entity are consistent; file a bug if it persists",
            experiment,
        )

    # SpecDraft: .fieldmap (the editable dict), .fields (included FieldProposals),
    # .excluded (ExclusionProposals), + the hand-counted estimates.
    fieldmap = _attr(proposal, "fieldmap", {}) or {}
    included = list(_attr(proposal, "fields", []) or [])
    excluded = list(_attr(proposal, "excluded", []) or [])
    vocab_size = _attr(proposal, "vocab_size")
    tokens_per_event = _attr(proposal, "tokens_per_event")
    chunk_size = _attr(proposal, "chunk_size")

    # The field-map must at least pin the grouping/entity so `tokenize --spec` can
    # exclude it from the vocab (T2). Defensive: stamp it from the resolved entity.
    if isinstance(fieldmap, dict):
        fieldmap.setdefault("entity", entity)
        fieldmap.setdefault("event", event)
        if target is not None:
            fieldmap.setdefault("target", target)
        fieldmap.setdefault("context_len", context_len)

    # --- persist the proposed SPEC as a content-addressed TokenizerSpec object -
    # Heavy payload = the editable YAML field-map (so the human can open, tweak,
    # and re-feed it to `tokenize --spec`). The content_id binds the proposal to
    # this exact input + entity/target so re-proposing the same thing is idempotent.
    spec_yaml = _dump_fieldmap_yaml(fieldmap)
    content_id = ctx.store.content_id(
        getattr(obj, "content_id", "") or obj.pathspec,
        f"propose:{entity}:{target}:{context_len}",
    )

    included_block = _serialize_included(included)
    excluded_block = _serialize_excluded(excluded)
    data_block: dict[str, Any] = {
        "input": obj.pathspec,
        "entity": entity,
        "event": event,
        "target": target,
        "fieldmap": fieldmap,
        "included": included_block,
        "excluded": excluded_block,
        "vocab_size": vocab_size,
        "tokens_per_event": tokens_per_event,
        "chunk_size": chunk_size,
        "context_len": context_len,
        "format": "loom-fieldmap/1",
    }

    ref = ctx.store.new_ref("TokenizerSpec")
    sobj = DataObject(
        ref=ref,
        kind="TokenizerSpec",
        content_id=content_id,
        parents=[obj.pathspec],
        producer_verb="propose",
        producer_args={
            "in": in_spec,
            "entity": entity,
            "event": event,
            "target": target,
            "context_len": context_len,
        },
        signatures={
            "vocab_size": vocab_size,
            "tokens_per_event": tokens_per_event,
            "chunk_size": chunk_size,
            "fieldmap_format": "loom-fieldmap/1",
        },
        verdict=Verdict.REVIEW,  # a PROPOSAL — the human reviews/edits before tokenize
        status=Status.OK,
        experiment=experiment,
        created_at=datetime.now(timezone.utc).isoformat(),
        extras={"proposal": data_block},
    )
    stored = ctx.store.put(sobj, payload=spec_yaml, payload_name="fieldmap.yaml")

    # --- the proposal card: included fields + the EXCLUDED columns (with WHY) --
    # The excluded entries carry their originating EDA Diagnostic so the human
    # reads exactly why a column was dropped and can override (HARD INVARIANT #4).
    diagnostics = _proposal_diagnostics(included_block, excluded_block)

    n_incl = len(included_block)
    n_excl = len(excluded_block)
    vocab_str = f"vocab≈{vocab_size}" if vocab_size is not None else "vocab=?"
    chunk_str = f"chunk_size={chunk_size}" if chunk_size is not None else "chunk_size=?"
    tpe_str = (
        f"tokens/event={tokens_per_event}" if tokens_per_event is not None else "tokens/event=?"
    )
    summary = (
        f"{stored.pathspec} verdict=REVIEW  {n_incl} fields tokenized, "
        f"{n_excl} excluded  {vocab_str}  {tpe_str}  {chunk_str}  "
        f"(edit {stored.pathspec} then `loom tokenize --spec {stored.pathspec}`)"
    )

    return VerbResult(
        verb="propose",
        status=Status.OK,
        verdict=Verdict.REVIEW,
        tier=Tier.WORKSPACE_WRITE,
        capability_mode=CapabilityMode.NONE,
        summary=summary,
        outputs=[stored.ref],
        diagnostics=diagnostics,
        data={"pathspec": stored.pathspec, "wrote_spec": True, **data_block},
        experiment=experiment,
        cost_plan=CostPlan(),
    )


# ---------------------------------------------------------------------------
# Card / payload helpers.
# ---------------------------------------------------------------------------


def _dump_fieldmap_yaml(fieldmap: dict[str, Any]) -> str:
    """Serialize the field-map as editable YAML (the heavy payload). Falls back to
    JSON-in-a-string if PyYAML is somehow unavailable so the proposal is never lost.
    A leading ``# loom-fieldmap/1`` banner documents the format for the human."""
    banner = (
        "# loom-fieldmap/1 — Loom tokenizer spec (PROPOSED; review & edit me)\n"
        "# entity is EXCLUDED from the vocab (T2); target is EXCLUDED (leakage).\n"
        "# edit the fields below, then: loom tokenize --spec <this TokenizerSpec/n or this file>\n"
    )
    try:
        import yaml  # PyYAML is in the Phase-0 dep set (numpy/pandas/PyYAML)

        body = yaml.safe_dump(fieldmap, sort_keys=False, default_flow_style=False)
    except Exception:  # pragma: no cover - defensive; YAML is a declared dep
        body = json.dumps(fieldmap, indent=2, default=str)
    return banner + body


def _serialize_included(included: list[Any]) -> list[dict[str, Any]]:
    """Normalize each :class:`~loom.engine.propose.FieldProposal` to a plain dict
    (``name``/``source``/``strategy``/``params``/``token_count``/``rationale``) so
    the persisted proposal + the agent envelope are pure JSON."""
    out: list[dict[str, Any]] = []
    for f in included:
        out.append(
            {
                "name": _attr(f, "name"),
                "source": _attr(f, "source"),
                "strategy": _attr(f, "strategy"),
                "params": _attr(f, "params", {}) or {},
                "token_count": _attr(f, "token_count"),
                "rationale": _attr(f, "rationale") or "",
            }
        )
    return out


def _serialize_excluded(excluded: list[Any]) -> list[dict[str, Any]]:
    """Normalize each :class:`~loom.engine.propose.ExclusionProposal` to a plain
    dict carrying the column, the machine reason code, the human rationale, and the
    serialized originating EDA card (``.eda``, if any) — so the persisted proposal
    + the agent envelope both state the reasoning (HARD INVARIANT #4)."""
    out: list[dict[str, Any]] = []
    for e in excluded:
        col = _attr(e, "name") or _attr(e, "column")
        reason = _attr(e, "reason") or ""
        rationale = _attr(e, "rationale") or ""
        # The originating EDA card travels under ``.eda`` (serialized dict) on the
        # ExclusionProposal; tolerate a live Diagnostic or a ``diagnostic`` alias.
        eda = _attr(e, "eda")
        if eda is None:
            eda = _attr(e, "diagnostic")
        eda_dict: Optional[dict[str, Any]] = None
        if isinstance(eda, Diagnostic):
            eda_dict = eda.to_dict()
        elif isinstance(eda, dict):
            eda_dict = eda
        entry: dict[str, Any] = {"column": col, "reason": reason, "rationale": rationale}
        if eda_dict is not None:
            entry["eda"] = eda_dict
        out.append(entry)
    return out


def _proposal_diagnostics(included: list[Any], excluded: list[Any]) -> list[Diagnostic]:
    """Build the card's Diagnostic list.

    One INFO card per INCLUDED field (the field→strategy rationale), and one
    WARNING/INFO card per EXCLUDED column — reusing the column's ORIGINATING EDA
    flag verbatim when present (the leakage card the human must see, INVARIANT #4),
    else a minted ``PROPOSE`` card carrying the drop reason. The entity/target
    exclusions are always surfaced even when ingest emitted no EDA flag for them."""
    diags: list[Diagnostic] = []

    for f in included:
        name = _attr(f, "name") or _attr(f, "source") or "?"
        source = _attr(f, "source") or name
        strategy = _attr(f, "strategy") or "?"
        count = _attr(f, "token_count")
        rationale = _attr(f, "rationale") or ""
        count_str = f" → {count} tokens" if count is not None else ""
        diags.append(
            Diagnostic(
                contract="PROPOSE",
                severity=Severity.INFO,
                message=f"field {name!r} (col {source!r}) → {strategy}{count_str}: {rationale}",
                fix=None,
                data={
                    "kind": "included",
                    "name": name,
                    "source": source,
                    "strategy": strategy,
                    "token_count": count,
                    "rationale": rationale,
                },
            )
        )

    for e in excluded:
        col = _attr(e, "column") or _attr(e, "name") or "?"
        reason = _attr(e, "reason") or _attr(e, "rationale") or ""
        origin = _diag_from_flag(_attr(e, "eda") or _attr(e, "diagnostic"))
        if origin is not None:
            # Surface the originating EDA flag VERBATIM so the human sees why
            # ingest flagged it — prefixed so it reads as an exclusion decision.
            diags.append(
                Diagnostic(
                    contract=origin.contract,
                    severity=origin.severity,
                    message=f"EXCLUDED {col!r} ({reason}): {origin.message}",
                    fix=origin.fix,
                    data={**(origin.data or {}), "kind": "excluded", "reason": reason},
                )
            )
        else:
            diags.append(
                Diagnostic(
                    contract="PROPOSE",
                    severity=Severity.WARNING,
                    message=f"EXCLUDED {col!r}: {reason}",
                    fix=(
                        f"if you want {col!r} in the vocab, add it to the spec's "
                        f"`fields` and re-run `loom tokenize --spec`"
                    ),
                    data={"kind": "excluded", "column": col, "reason": reason},
                )
            )

    return diags
