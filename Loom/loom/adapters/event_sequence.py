"""``event-sequence`` — the data-representation port adapter #1 (ARCHITECTURE §5.2).

The existing ``loom/engine/`` package **is** this adapter; this module wraps it
behind the :class:`loom.ports.DataRepresentation` Protocol with near-zero change.
Every port method delegates 1:1 to a function that already exists and is
exported from ``loom/engine/__init__.py`` (and to the ``tokenize`` verb's
``_build_spec`` logic for argument parsing). The delegation table is verbatim
ARCHITECTURE §5.2:

    | Port method                  | Existing implementation                          |
    |------------------------------|--------------------------------------------------|
    | build_spec(args)             | tokenize.py:_build_spec (financial/chain presets) |
    | compile(spec, context_len)   | engine.compile_spec → CompiledTokenizer          |
    | contracts(compiled)          | engine/contracts → Diagnostic(contract="C1"…)    |
    | representation_passed        | compiled.report.passed (False iff any ERROR)     |
    | materialize(...)             | engine.materialize_corpus_lines (local corpus)   |
    | signatures(compiled)         | the dict at tokenize.py:399-407                  |
    | produces_tensor_contract     | "clm/input_ids+labels/-100"                      |

``CompiledTokenizer.vocab_hash`` **is** the ``representation_signature`` (today's
``vocab_hash`` generalized — the retrain trigger). The ``signatures`` handoff
renames ``vocab_hash``→``representation_signature`` and ``encode_path``→
``representation`` so it is framework-neutral on the wire.

Scope (this slice, ARCHITECTURE §10 step 2 + the §6/I1 note): ``materialize()``
produces a :class:`~loom.ports.PreparedCorpus` pointing at the **local
content-addressed corpus** (the existing :class:`~loom.store.ObjectStore` path),
not ``gs://`` sharded Arrow — the BigQuery/RAPIDS Tier-A/Tier-B pushdown and the
cloud datastore land in step 8. The corpus-build fan-out is still routed through
``executor.foreach`` so the scale-out seam is exercised exactly where step 8 will
swap in the GPU per-shard fan-out, with zero change to this adapter's shape.

Hard rule: this module imports nothing from NeMo/torch/transformers/RAPIDS/
BigQuery — it is pure CPU engine delegation + the harness ports/store/types.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd

from ..engine import AmountStrategy, compile_spec, materialize_corpus_lines
from ..engine.api import CompiledTokenizer, TokenizerSpec
from ..engine.spec import chain_spec, financial_spec
from ..ports import (
    Executor,
    PreparedCorpus,
    SourceRef,
    register_representation,
)
from ..store import DataObject, ObjectStore, default_store
from ..types import CostPlan, Diagnostic, Severity, Status, Verdict

# The C4 tensor-contract string this representation emits — the narrow waist the
# model-builder + objective consume but neither owns (ARCHITECTURE §7, C4). The
# corpus is {input_ids, labels} with -100 ignore-index (verified src/clm_data.py).
_TENSOR_CONTRACT = "clm/input_ids+labels/-100"


def _build_spec(args: dict[str, Any]) -> TokenizerSpec:
    """Build the requested :class:`TokenizerSpec` from CLI/agent args.

    This is the exact logic of ``loom/verbs/tokenize.py:_build_spec`` (the §5.2
    delegation target), reproduced here so the adapter owns its spec assembly
    without the verb importing the adapter or vice-versa. Defaults to the
    ``financial`` preset (the conformance oracle); ``--preset chain`` selects the
    DEX next-trade spec; ``--drop-step`` removes a named step (T2)."""
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


class EventSequenceRepresentation:
    """``DataRepresentation`` #1 — tokenized CLM event sequences over the engine.

    Structurally satisfies :class:`loom.ports.DataRepresentation`
    (``runtime_checkable``): it carries ``name`` + ``produces_tensor_contract``
    attributes and the eight port methods, every one of which delegates to the
    already-verified engine. It owns no NeMo/GPU/cloud concern."""

    name: str = "event-sequence"
    produces_tensor_contract: str = _TENSOR_CONTRACT

    # -- the data-free, deterministic spine (delegations only) ---------------

    def build_spec(self, args: dict) -> TokenizerSpec:
        """args → TokenizerSpec. Delegates to the §5.2 ``_build_spec`` logic."""
        return _build_spec(args)

    def compile(self, spec: Any, *, context_len: int) -> CompiledTokenizer:
        """spec → CompiledTokenizer (data-free, deterministic; emits the
        ``vocab_hash`` == ``representation_signature``). Delegates to
        ``engine.compile_spec``."""
        return compile_spec(spec, context_len=context_len)

    def contracts(self, compiled: Any) -> list[Diagnostic]:
        """The port-local C1/C2/C3 cards, in spec order — verbatim the engine's
        ``compiled.report.diagnostics`` (``tokenize.py:_contract_diagnostics``)."""
        return list(compiled.report.diagnostics)

    def representation_passed(self, compiled: Any) -> bool:
        """The contract-NAME-AGNOSTIC corpus write-gate (ARCHITECTURE §2.1/§8).

        Returns ``compiled.report.passed`` — which ``ContractReport.add`` already
        flips to ``False`` on *any* ERROR-severity Diagnostic (``engine/api.py``),
        regardless of contract string. This is exactly the inherited ERROR-scan
        default ``not any(d.severity is ERROR for d in self.contracts(compiled))``;
        the event-sequence adapter satisfies it for free."""
        return bool(compiled.report.passed)

    def refusal_summary(self, compiled: Any, diagnostics: list[Diagnostic]) -> Optional[str]:
        """OPTIONAL port hook (ARCHITECTURE §8 / review-finding): render the one-line
        REFUSED_CONTRACT *summary* for this representation's own ERROR diagnostics so
        ``loom tokenize --json`` stays BYTE-IDENTICAL to v0.1 on the refusal envelope
        (invariant #3), not merely on the PASS envelope.

        This is a pure *message renderer* over already-refused diagnostics — it never
        decides the refusal. ``prepare``'s write-gate stays name-agnostic
        (``not representation_passed``); it only *asks* the port how to phrase a
        refusal it has ALREADY made, and falls back to the generic rendering for any
        port (or stub) that does not implement this hook. The verb therefore still
        never names "C1"/"C3" nor reads ``report.injective/.dense``.

        The two v0.1 ``tokenize.py`` strings, reconstructed verbatim from the port's
        OWN cards (C1 detected by an injectivity/density ERROR, C3 by a grammar
        ERROR), with ``compiled.chunk_size`` interpolated exactly as v0.1 did. Returns
        ``None`` if no recognized ERROR card is present (→ generic fallback)."""
        errors = [d for d in diagnostics if d.severity is Severity.ERROR]
        # v0.1 ordering: C1 (injectivity/density) is checked and refused BEFORE C3.
        if any(d.contract == "C1" for d in errors):
            preset = getattr(getattr(compiled, "spec", None), "preset", None)
            return (
                f"REFUSED_CONTRACT: C1 injectivity/density failed for preset "
                f"'{preset}' — no Corpus written (the named diff explains the "
                "collision; reordering shifts every id ⇒ vocab_hash changes ⇒ "
                "retrain required)."
            )
        if any(d.contract == "C3" for d in errors):
            return (
                f"REFUSED_CONTRACT: C3 grammar failed — chunk_size="
                f"{compiled.chunk_size} holds less than one transaction; no Corpus written."
            )
        return None

    def signatures(self, compiled: Any) -> dict:
        """The harness handoff dict — exactly ``tokenize.py:399-407`` with
        ``vocab_hash``→``representation_signature`` and ``encode_path``→
        ``representation`` (framework-neutral on the wire)."""
        return {
            "representation": self.name,
            "representation_signature": compiled.vocab_hash,
            "vocab_hash": compiled.vocab_hash,  # kept for byte-identity w/ the corpus payload
            "vocab_size": compiled.vocab_size,
            "tokens_per_txn": compiled.tokens_per_txn,
            "chunk_size": compiled.chunk_size,
            "context_len": compiled.context_len,
            "has_fitted_artifact": compiled.report.has_fitted_artifact,
            "tensor_contract": self.produces_tensor_contract,
        }

    # -- cost plan (CPU-cheap; NO materialization) ---------------------------

    def plan(self, *, spec: Any, source: SourceRef, executor: "Executor") -> CostPlan:
        """A CPU-cheap, ~$0 plan for the local materialize. No BigQuery-bytes
        estimate in this slice (Tier-A pushdown is step 8). ``derived=True``,
        ``usd=0.0`` — the corpus build is in-process CPU work."""
        compiled = self.compile(spec, context_len=4096)
        return CostPlan(
            derived=True,
            usd=0.0,
            confidence="high",
            tokens=None,
            seq_len=compiled.context_len,
            gpu_target=None,
            inputs={
                "representation": self.name,
                "source": source.uri,
                "vocab_size": compiled.vocab_size,
                "executor": getattr(executor, "name", None),
            },
        )

    # -- the scale-out (§6) — local corpus via the executor seam -------------

    def materialize(
        self, *, compiled: Any, source: SourceRef, executor: "Executor"
    ) -> PreparedCorpus:
        """Build a corpus and persist it as a local content-addressed Corpus, then
        return a :class:`PreparedCorpus` pointing at it.

        This slice is LOCAL (ARCHITECTURE §6 / §10 step 8 note): the corpus is the
        existing local content-addressed payload, not ``gs://`` sharded Arrow. The
        corpus-line assembly is still routed through ``executor.foreach`` over a
        single in-memory shard so the scale-out seam is exercised exactly where the
        step-8 GPU per-shard fan-out will swap in — with zero change to this
        adapter's shape.

        Rows come from ``source.snapshot["dataframe"]`` (an injected frame, the
        test/agent path) or ``source.uri`` read as a local CSV; a missing/unreadable
        source is NOT fatal (the vocab is config-only) — the PreparedCorpus still
        carries the compiled signature + an empty corpus payload.
        """
        df = self._load_rows(source)

        # Route corpus assembly through the executor seam. The fan-out is over a
        # single shard here (one local frame); step 8 fans this out per-GCS-shard.
        corpus_lines: list[str] = []
        n_txns = 0
        if df is not None:
            store_box: dict[str, pd.DataFrame] = {"shard-0000": df}

            def _build_shard(shard_id: str) -> str:
                frame = store_box[shard_id]
                lines, ntx = materialize_corpus_lines(compiled, frame)
                return json.dumps({"lines": lines, "n_txns": int(ntx)})

            results = executor.foreach(
                fn=_build_shard,
                shards=list(store_box.keys()),
                compute=_cpu_compute(),
            )
            for payload in results:
                rec = json.loads(payload)
                corpus_lines.extend(rec["lines"])
                n_txns += int(rec["n_txns"])

        store = self._store()
        spec = compiled.spec
        sig = self.signatures(compiled)

        # Content address over (source, representation_signature) — same scheme as
        # today's tokenize (source_fingerprint + spec_hash).
        source_fp = source.uri or "<no-input>"
        content_id = store.content_id(source_fp, compiled.vocab_hash)

        existing = store.find_by_content(content_id)
        if existing is not None:
            return self._prepared_from_object(existing, compiled, sig, source)

        ref = store.new_ref("Corpus")
        payload = {
            "vocab": compiled.vocab,
            "vocab_hash": compiled.vocab_hash,
            "vocab_size": compiled.vocab_size,
            "tokens_per_txn": compiled.tokens_per_txn,
            "chunk_size": compiled.chunk_size,
            "context_len": compiled.context_len,
            "preset": spec.preset,
            "step_names": spec.step_names(),
            "n_lines": len(corpus_lines),
            "n_txns": n_txns,
            "corpus_lines": corpus_lines,
        }
        obj = DataObject(
            ref=ref,
            kind="Corpus",
            content_id=content_id,
            parents=[source.uri] if source.uri else [],
            producer_verb="prepare",
            producer_args={"representation": self.name, "source": source.uri},
            signatures=sig,
            verdict=Verdict.PASS,
            status=Status.OK,
            extras={
                "preset": spec.preset,
                "step_names": spec.step_names(),
                "n_lines": len(corpus_lines),
                "n_txns": n_txns,
                "snapshot": _provenance(source),
            },
        )
        stored = store.put(obj, payload=json.dumps(payload), payload_name="corpus.json")
        return self._prepared_from_object(stored, compiled, sig, source)

    # -- helpers -------------------------------------------------------------

    def _prepared_from_object(
        self,
        obj: DataObject,
        compiled: CompiledTokenizer,
        sig: dict,
        source: SourceRef,
    ) -> PreparedCorpus:
        """Assemble the PreparedCorpus from a persisted local Corpus DataObject.

        ``train_uri`` is the local corpus pathspec (``Corpus/<n>``); ``val_uri`` /
        ``test_uri`` are ``None`` in this slice (the C6 temporal split lands with
        the Tier-A pushdown, step 8). ``effective_tokens`` is the measured corpus
        token count (n_txns * (tokens_per_txn + 1) for the per-txn <sep>) — a
        local stand-in for the step-8 measured non-redundant count."""
        n_txns = int(obj.extras.get("n_txns", 0))
        n_lines = int(obj.extras.get("n_lines", 0))
        effective_tokens = n_txns * (compiled.tokens_per_txn + 1)
        return PreparedCorpus(
            representation=self.name,
            representation_signature=compiled.vocab_hash,
            tensor_contract=self.produces_tensor_contract,
            train_uri=obj.pathspec,
            val_uri=None,
            test_uri=None,
            manifest_uri=obj.pathspec,
            seq_length=compiled.context_len,
            pad_token_id=0,  # SPECIAL_TOKENS[0] == "<pad>" at id 0 (engine.api)
            vocab_size=compiled.vocab_size,
            effective_tokens=effective_tokens,
            provenance=_provenance(source),
            extras={
                "n_lines": n_lines,
                "n_txns": n_txns,
                "chunk_size": compiled.chunk_size,
                "tokens_per_txn": compiled.tokens_per_txn,
                "preset": compiled.spec.preset,
                "corpus_pathspec": obj.pathspec,
            },
        )

    @staticmethod
    def _load_rows(source: SourceRef) -> Optional[pd.DataFrame]:
        """Load rows from the source. Prefers an injected frame
        (``source.snapshot["dataframe"]`` — the test/agent path); else reads
        ``source.uri`` as a local CSV. A missing/unreadable source is NOT fatal —
        the vocab is config-only, so ``None`` just means an empty corpus payload."""
        injected = source.snapshot.get("dataframe") if source.snapshot else None
        if injected is not None:
            return injected if isinstance(injected, pd.DataFrame) else pd.DataFrame(injected)
        uri = source.uri or ""
        if not uri or uri.startswith(("bq://", "gs://")):
            # Cloud sources are step-8 (the Tier-A/Tier-B pushdown) — not readable
            # by the local slice; degrade to a config-only (empty corpus) build.
            return None
        try:
            return pd.read_csv(uri)
        except (FileNotFoundError, OSError, ValueError):
            return None

    @staticmethod
    def _store() -> ObjectStore:
        return default_store()


def _provenance(source: SourceRef) -> dict:
    """The provenance anchor (snapshot range / MAX(event_date)) carried with every
    corpus (ARCHITECTURE §7), distilled from ``source.snapshot`` minus the injected
    in-RAM frame (which is data, not provenance)."""
    snap = dict(source.snapshot or {})
    snap.pop("dataframe", None)
    return {"uri": source.uri, "snapshot": snap}


def _cpu_compute():
    """A CPU :class:`~loom.ports.ComputeTarget` for the local corpus fan-out."""
    from ..ports import ComputeTarget

    return ComputeTarget(launcher="local", nproc_per_node=1, accelerator="cpu")


# ARCHITECTURE §2.4 / §10 step 2: one-line registration under the registry key.
register_representation(EventSequenceRepresentation())

__all__ = ["EventSequenceRepresentation"]
