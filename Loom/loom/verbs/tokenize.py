"""``tokenize`` — THE contract compiler verb (DESIGN.md item #1).

One declarative tokenizer spec → the engine compiles it (derives ``vocab_size``,
``vocab_hash``, ``tokens_per_txn``, ``chunk_size``), runs C1 (injectivity +
density), C2 (determinism / fitted-artifact), C3 (grammar / chunk derivation),
and surfaces each finding as a named-diff :class:`~loom.types.Diagnostic` card —
not a stack trace. On a C1 collision the verb REFUSES to write a Corpus (status
``REFUSED_CONTRACT`` / verdict ``FAIL``) with the named diagnostic; on PASS it
persists a ``Corpus`` :class:`~loom.store.DataObject` carrying the vocab, the C1
signature (``vocab_hash``) and the derived numbers.

Workspace-write tier: cheap, CPU, no GPU gating — but the envelope already
carries tier / capability_mode / cost_plan so the gating model downstream reads
the same shape. The ``--json`` envelope is byte-identical to the agent-tool
result (the contract-effects live in ``diagnostics`` and ``data``).

TODO(v0.2): the confirm_token PLAN handshake (§5.3) and Metaflow execution
adapter attach here unchanged; this slice commits directly (cheap write).
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd

from ..engine import AmountStrategy, compile_spec
from ..engine.api import CompiledTokenizer, TokenizerSpec
from ..engine.spec import chain_spec, financial_spec
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

TOKENIZE_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "in": {"type": "string", "description": "input IngestDataset/<n> pathspec"},
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


def _build_spec(args: dict[str, Any]) -> TokenizerSpec:
    """Build the requested :class:`TokenizerSpec` from the CLI/agent args.

    Defaults to the ``financial`` preset (the conformance oracle). ``--drop-step``
    removes a named step (T2); ``--include-time-delta`` adds the TDIF field (T1);
    ``--schema chain`` / ``--preset chain`` selects the DEX next-trade spec."""
    preset = (args.get("preset") or args.get("schema") or "financial").lower()
    drop = tuple(s for s in [args.get("drop_step")] if s)

    if preset == "chain":
        # T2: keep wallet identity out of the vocab by default (no_identity_token
        # defaults True for chain; an explicit include flips it).
        no_identity = args.get("no_identity_token", True)
        return chain_spec(
            item_hash_size=int(args.get("merchant_hash_size") or 5000),
            include_identity_token=not no_identity,
            drop_steps=drop,
        )

    amount_strategy = AmountStrategy(args.get("amount_strategy", "fixed"))
    return financial_spec(
        merchant_hash_size=int(args.get("merchant_hash_size") or 2000),
        amount_strategy=amount_strategy,
        include_time_delta=bool(args.get("include_time_delta", False)),
        drop_steps=drop,
    )


def _load_rows(in_path: Optional[str], ctx: VerbContext) -> tuple[Optional[pd.DataFrame], Optional[str], list[str]]:
    """Best-effort load of the input dataset's rows for corpus materialization.

    Returns ``(df, dataset_pathspec, parents)``. The vocabulary is config-only, so
    a missing/unreadable dataset is NOT fatal — the Corpus still carries the
    compiled vocab + signature, and corpus-line materialization is skipped. An
    in-memory frame may be injected via ``ctx.extras["dataframe"]`` (test/agent).
    """
    injected = ctx.extras.get("dataframe") if ctx.extras else None
    if injected is not None:
        df = injected if isinstance(injected, pd.DataFrame) else pd.DataFrame(injected)
        return df, in_path, ([in_path] if in_path else [])
    if not in_path:
        return None, None, []
    # A missing dataset is NOT fatal (vocab is config-only). We also tolerate a
    # store that is not yet wired (NotImplementedError) so tokenize compiles a
    # deterministic, byte-identical envelope on both faces during incremental
    # build (DESIGN.md §2.1).
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


def _materialize_corpus(
    compiled: CompiledTokenizer, df: pd.DataFrame
) -> tuple[list[str], int]:
    """Preprocess rows for the spec's preset, emit per-step token columns, and
    assemble corpus lines (C3 grammar). Returns ``(lines, n_txns)``.

    Delegates to the engine's public ``materialize_corpus_lines`` so the verb and
    the conformance tests share one corpus-assembly path."""
    from ..engine import materialize_corpus_lines

    return materialize_corpus_lines(compiled, df)


def _contract_diagnostics(compiled: CompiledTokenizer) -> list[Diagnostic]:
    """The C1/C2/C3 findings as the envelope's named-diff cards (in spec order)."""
    return list(compiled.report.diagnostics)


def _compile_only_result(
    compiled: CompiledTokenizer,
    diagnostics: list[Diagnostic],
    derived: dict[str, Any],
    parents: list[str],
    experiment: Optional[str],
) -> VerbResult:
    """A deterministic PASS envelope when the store cannot persist (not yet wired).

    The spec compiled cleanly and passed C1/C2/C3, but no Corpus was written —
    so ``outputs`` is empty and ``wrote_corpus`` is False. This keeps the
    ``--json`` envelope byte-identical to the agent-tool result on both faces
    during incremental build (DESIGN.md §2.1); the persisting path takes over
    once the ObjectStore seam is wired."""
    return VerbResult(
        verb="tokenize",
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
    ref,
    compiled: CompiledTokenizer,
    diagnostics: list[Diagnostic],
    derived: dict[str, Any],
    content_id: str,
    n_lines: int,
    n_txns: int,
    parents: list[str],
    experiment: Optional[str],
) -> VerbResult:
    """The single PASS envelope for a written/idempotent Corpus.

    Both the fresh-write path and the idempotent-hit path build the envelope here
    so the result is byte-identical for the same compiled spec regardless of
    whether the object was just minted or already existed (§2.1 dual-driver
    byte-identity / §6 idempotency). The pathspec is content-addressed, so it is
    identical across calls/faces too."""
    summary = (
        f"{ref.pathspec} verdict=PASS vocab={compiled.vocab_size} "
        f"tokens/txn={compiled.tokens_per_txn} chunk_size={compiled.chunk_size} "
        f"sig={compiled.vocab_hash[:18]}…"
    )
    return VerbResult(
        verb="tokenize",
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


@register(
    "tokenize",
    summary="compile a declarative tokenizer spec to a Corpus (C1/C2/C3 checked, <1s, no GPU)",
    tier=Tier.WORKSPACE_WRITE,
    capability_mode=CapabilityMode.NONE,
    params=TOKENIZE_PARAMS,
)
def _tokenize(args: dict[str, Any], ctx: VerbContext) -> VerbResult:
    from ..store import DataObject  # local import: store is the v0.2 seam

    in_path = args.get("in") or None
    context_len = int(args.get("context_len") or 4096)
    experiment = ctx.experiment

    # --- compile the spec (data-free) + run C1/C2/C3 --------------------------
    spec = _build_spec(args)
    compiled = compile_spec(spec, context_len=context_len)
    diagnostics = _contract_diagnostics(compiled)

    derived = {
        "preset": spec.preset,
        "vocab_size": compiled.vocab_size,
        "vocab_hash": compiled.vocab_hash,
        "tokens_per_txn": compiled.tokens_per_txn,
        "chunk_size": compiled.chunk_size,
        "context_len": compiled.context_len,
        "step_names": spec.step_names(),
        "has_fitted_artifact": compiled.report.has_fitted_artifact,
    }

    # --- C1 REFUSAL: a collision/non-dense vocab refuses to write a Corpus ----
    if not (compiled.report.injective and compiled.report.dense):
        return VerbResult(
            verb="tokenize",
            status=Status.REFUSED_CONTRACT,
            verdict=Verdict.FAIL,
            tier=Tier.WORKSPACE_WRITE,
            capability_mode=CapabilityMode.NONE,
            summary=(
                f"REFUSED_CONTRACT: C1 injectivity/density failed for preset "
                f"'{spec.preset}' — no Corpus written (the named diff explains the "
                "collision; reordering shifts every id ⇒ vocab_hash changes ⇒ "
                "retrain required)."
            ),
            outputs=[],
            diagnostics=diagnostics,
            data={**derived, "wrote_corpus": False},
            experiment=experiment,
            cost_plan=CostPlan(),
        )

    # --- C3 grammar hard-fail (chunk < 1 txn): also refuse ---------------------
    grammar_failed = any(
        d.contract == "C3" and d.severity is Severity.ERROR for d in diagnostics
    )
    if grammar_failed:
        return VerbResult(
            verb="tokenize",
            status=Status.REFUSED_CONTRACT,
            verdict=Verdict.FAIL,
            tier=Tier.WORKSPACE_WRITE,
            capability_mode=CapabilityMode.NONE,
            summary=(
                f"REFUSED_CONTRACT: C3 grammar failed — chunk_size="
                f"{compiled.chunk_size} holds less than one transaction; no Corpus written."
            ),
            outputs=[],
            diagnostics=diagnostics,
            data={**derived, "wrote_corpus": False},
            experiment=experiment,
            cost_plan=CostPlan(),
        )

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

    # The store is the v0.2 seam; tolerate it being unwired during incremental
    # build so tokenize still returns a deterministic, byte-identical PASS
    # envelope (compile-only) on both faces (DESIGN.md §2.1).
    try:
        content_id = ctx.store.content_id(source_fp, spec_hash)
        existing = ctx.store.find_by_content(content_id)
    except (NotImplementedError, AttributeError):
        return _compile_only_result(
            compiled, diagnostics, derived, parents, experiment
        )

    # idempotency (§6): same input+spec → return the EXISTING Corpus with an
    # envelope byte-identical to the fresh-write one (§2.1 — two sequential
    # identical calls, or the two driver faces, must not diverge). The pathspec is
    # already identical (content-addressed); we rebuild the same summary/data from
    # the persisted object so the byte-identity invariant holds.
    if existing is not None:
        n_lines = int(existing.extras.get("n_lines", len(corpus_lines)))
        n_txns = int(existing.extras.get("n_txns", n_txns))
        return _corpus_result(
            existing.ref, compiled, diagnostics, derived, content_id,
            n_lines, n_txns, parents, experiment,
        )

    # --- persist the Corpus (vocab + signature + derived numbers) -------------
    try:
        ref = ctx.store.new_ref("Corpus")
    except (NotImplementedError, AttributeError):
        return _compile_only_result(
            compiled, diagnostics, derived, parents, experiment
        )
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
        producer_verb="tokenize",
        producer_args={
            "in": in_path,
            "preset": spec.preset,
            "include_time_delta": bool(args.get("include_time_delta", False)),
            "merchant_hash_size": args.get("merchant_hash_size"),
            "amount_strategy": spec.amount_strategy.value,
            "drop_step": args.get("drop_step"),
            "context_len": context_len,
        },
        # The C1 signature travels WITH the object — embed/pretrain assert it.
        signatures={
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
            compiled, diagnostics, derived, parents, experiment
        )

    return _corpus_result(
        stored.ref, compiled, diagnostics, derived, content_id,
        len(corpus_lines), n_txns, parents, experiment,
    )
