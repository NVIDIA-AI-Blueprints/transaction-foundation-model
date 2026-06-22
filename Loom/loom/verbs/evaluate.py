"""``evaluate`` — the LOCAL, UNGATED, CPU/$0 *validate-before-train* verdict.

The second half of the validate loop (`propose → tokenize → embed → evaluate →
refine`): it SCORES a tokenizer spec's local embeddings against the classical
controls a model must beat, so a user SEES whether their tokenizer is worth a
GPU-hour *before* spending one. It is a ``WORKSPACE_WRITE`` / ``SEARCHABLE`` verb
(``cost_plan=CostPlan()``, never a ``confirm_token``/Executor/CheckpointRef) — the
agent calls it freely and the gate is inert.

It resolves THREE objects: the **Corpus** (vocab + ``contract_report`` +
signatures), its parent **IngestDataset** (``corpus.parents[0]`` → the raw rows for
the hold-out), and an **Embeddings** matrix (default-resolved by lineage + the C5
pairing if ``--embeddings`` is omitted). The hold-out is baseline's IDENTICAL
leave-one-last-out temporal split via the SHARED :mod:`loom.engine.holdout` helper
(NOT a fork), so the strong repeat-last-item control is computed exactly one way.

Three scores, then a STRICT verdict (Anub's product decision):

  * **intrinsic** (read off spec+corpus, NO training): ``oov_unk_rate`` over the
    SAME id'd stream/vocab/drop-rule ``embed`` fit on; a per-id ``occupancy``
    histogram (min/median/max + normalized entropy); ``dead_token_frac`` +
    ``starved_token_count`` measured off ``vocab_size`` (NEVER the embedding dim);
    and the C1/C2/C3 ``contract`` facts read VERBATIM off
    ``Corpus.extras['contract_report']`` (NO recompile).
  * **extrinsic** (weak / high-variance on a small sample — LABELED so): an
    embedding-kNN next-item Prec@k. For each leave-one-last-out target, the
    predictor maps ``prev_item`` raw value → token id via the Corpus vocab/encode
    (NOT a string match — a hashed/bucketed field maps to its bucket token), takes
    the k cosine-nearest item-family vocab ids, maps them back to item space, and
    scores Prec@k against the held-out actual item. The predictor reads ONLY
    history (never the actual item). A held-out item with no vocab token is an
    EXPLICIT OOV miss (a discriminating signal), never a silent drop.
  * **controls**: ``repeat_last_prec1`` + ``popularity_prec_at_k`` — read off the
    resolved Baseline's metrics OR recomputed via the SHARED helper at matching
    k/eval_split (never differently than baseline).

STRICT VERDICT — the intrinsic gate dominates (FAIL on contract-fail / pathological
oov / pathological dead-tokens). Else, with ``knn``=kNN Prec@k, ``rli``=repeat-last
Prec@1, ``pop``=popularity Prec@k: **PASS** iff ``knn > rli AND knn >= pop``;
**FAIL** iff ``knn <= rli`` (cannot beat the strong control → not worth a GPU-hour);
**REVIEW** iff ``knn > rli`` but a yellow flag (``knn < pop``, or elevated-but-not-
pathological dead-tokens / occupancy skew). NO hardcoded goodness threshold — every
comparison is RELATIVE to the reused controls and the data's own occupancy.

A COMPUTED verdict (PASS/REVIEW/FAIL) is ``status=OK`` (a FAIL spec is OK+FAIL, like
baseline carries PASS+OK). ``REFUSED_CONTRACT`` is reserved for STRUCTURAL refusals:
no embeddings resolvable, a C5 representation-signature mismatch, or ``n_eval==0``
(C6). The result mints a ``Scores/<n>`` object (payload ``scores.json``) and a
REFINE handshake: a :class:`~loom.types.Diagnostic` per failing dimension whose
``.fix`` names the EXACT fieldmap.yaml knob, PLUS a structured
``data['refine_plan']`` of ``{field, knob, current, suggested, reason, signal}``.

CPU-only, deterministic, torch-free — reads the matrix via
``safetensors.numpy.load_file`` (numpy API only).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..engine.holdout import (
    build_leave_one_last_out,
    infer_cols,
    load_rows,
    prec_at_k,
)
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

# ``<bos>``/``<eos>``/``<sep>``/``<pad>``/``<unk>`` — the special-token prefixes
# excluded from the item-family kNN candidate space (ids 0-4). Matched by the
# ``<...>`` shape so a custom grammar's specials are excluded too.
_DEFAULT_K = 5
_DEFAULT_EVAL_SPLIT = "temporal"

EVALUATE_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "in": {
            "type": "string",
            "description": "input Corpus/<n> pathspec to evaluate",
        },
        "embeddings": {
            "type": "string",
            "description": (
                "Embeddings/<n> pathspec (optional; default-resolves the most-recent "
                "Embeddings paired to this Corpus by lineage + the C5 vocab_hash)"
            ),
        },
        "baseline": {
            "type": "string",
            "description": (
                "Baseline/<n> pathspec for the controls (optional; default-resolves "
                "for the experiment, else recomputes via the shared hold-out helper)"
            ),
        },
        "k": {
            "type": "integer",
            "description": "Prec@K cutoff for the kNN next-item score + popularity control (default 5)",
        },
        "eval_split": {
            "type": "string",
            "description": "temporal | entity-disjoint (C6); must match the baseline's split",
        },
    },
}


# ---------------------------------------------------------------------------
# Structural-refusal envelope ($0 on a mispaired/missing artifact).
# ---------------------------------------------------------------------------


def _refused(
    msg: str,
    fix: str,
    experiment: Optional[str],
    *,
    contract: str = "C6",
    status: Status = Status.REFUSED_CONTRACT,
    data: Optional[dict[str, Any]] = None,
) -> VerbResult:
    """A CLEAN structural refusal — ``cost_plan=CostPlan()`` ($0), a single named
    diagnostic with the exact one-line fix. Never a stack trace.

    The verdict is ``INCOMPLETE`` (NOT ``FAIL``): nothing was SCORED here — there was
    no embedding to pair (C5), no rows for the hold-out (C6), etc. A FAIL *verdict* is
    reserved for a COMPUTED "refine it" outcome (the strict gate over real scores) and
    would short-circuit ``exit_code`` to 1; a STRUCTURAL refusal must surface its
    signal as ``status=REFUSED_*`` → ``exit_code==2`` (the build spec's contract:
    REFUSED_* exits 2, a computed FAIL exits 1). Matches ``embed._refused_no_stream``."""
    return VerbResult(
        verb="evaluate",
        status=status,
        verdict=Verdict.INCOMPLETE,
        tier=Tier.WORKSPACE_WRITE,
        capability_mode=CapabilityMode.SEARCHABLE,
        summary=msg,
        diagnostics=[Diagnostic(contract=contract, severity=Severity.ERROR, message=msg, fix=fix)],
        data=data or {},
        experiment=experiment,
        cost_plan=CostPlan(),
    )


# ---------------------------------------------------------------------------
# Resolving the Embeddings + Baseline by lineage (with the C5 pairing filter).
# ---------------------------------------------------------------------------


def _resolve_embeddings(
    ctx: VerbContext, corpus: Any, vocab_hash: Any, explicit: Optional[str]
) -> tuple[Optional[Any], Optional[str]]:
    """Resolve the Embeddings object to score. An explicit ``--embeddings`` pathspec
    wins (still C5-checked downstream). Otherwise pick the MOST-RECENT (highest n)
    Embeddings whose ``parents == [corpus.pathspec]`` AND whose
    ``signatures.representation_signature == corpus vocab_hash`` — the C5 pairing
    filter (a stale/mispaired artifact is never silently scored). Returns
    ``(obj, error_message)``; ``obj is None`` ⇒ a clean refusal."""
    if explicit:
        try:
            return ctx.store.get(explicit), None
        except (KeyError, ValueError) as exc:
            return None, f"could not resolve embeddings {explicit!r}: {exc}"

    candidates = [
        e
        for e in ctx.store.list("Embeddings")
        if corpus.pathspec in (e.parents or [])
        and (e.signatures or {}).get("representation_signature") == vocab_hash
    ]
    if not candidates:
        return None, (
            f"no Embeddings paired to {corpus.pathspec} "
            f"(parents+vocab_hash) found in the store"
        )
    candidates.sort(key=lambda e: e.ref.n)
    return candidates[-1], None


def _resolve_baseline(
    ctx: VerbContext, corpus: Any, ingest: Any, experiment: Optional[str], explicit: Optional[str]
) -> Optional[Any]:
    """Resolve a Baseline for the controls. An explicit ``--baseline`` wins. Else the
    most-recent Baseline that matches THIS experiment, or whose parent is the same
    IngestDataset the Corpus descends from (lineage). Returns ``None`` ⇒ recompute the
    controls via the SHARED hold-out helper (never a different code path)."""
    if explicit:
        try:
            return ctx.store.get(explicit)
        except (KeyError, ValueError):
            return None

    ingest_spec = ingest.pathspec if ingest is not None else None
    matches = []
    for b in ctx.store.list("Baseline"):
        by_exp = experiment is not None and b.experiment == experiment
        by_lineage = ingest_spec is not None and ingest_spec in (b.parents or [])
        if by_exp or by_lineage:
            matches.append(b)
    if not matches:
        return None
    matches.sort(key=lambda b: b.ref.n)
    return matches[-1]


# ---------------------------------------------------------------------------
# The token-space ↔ row-space bridge — the subtlest part (RISK in the spec).
# ---------------------------------------------------------------------------


def _reconstruct_compiled(corpus: Any, ctx: VerbContext, ingest: Any):
    """Rebuild the :class:`CompiledTokenizer` the Corpus was tokenized with, so a
    raw item value can be ENCODED to its token (NOT string-matched). Two faithful
    paths, both flowing through the LOCKED ``compile_spec``:

      * a **preset** Corpus (``chain``/``financial``) → the preset factory, with the
        hash-bucket size recovered from the VOCAB's prefix-family count (so the
        reconstruction is byte-exact even though the knob isn't on the Corpus); and
      * a **custom** Corpus → the source ``TokenizerSpec`` field-map, found by lineage
        (same experiment / same IngestDataset parent) and compiled via
        ``spec_from_field_map``.

    Returns the ``CompiledTokenizer`` or ``None`` (then the bridge falls back to a
    verbatim vocab lookup, and an unencodable value becomes an explicit OOV miss)."""
    from ..engine import (
        chain_spec,
        compile_spec,
        financial_spec,
        spec_from_field_map,
    )
    from ..engine.api import AmountStrategy

    pargs = dict(getattr(corpus, "producer_args", {}) or {})
    preset = pargs.get("preset") or (getattr(corpus, "extras", {}) or {}).get("preset")
    context_len = int(pargs.get("context_len") or 4096)
    vocab = ((_load_corpus_payload(corpus) or {}).get("vocab")) or {}

    def _prefix_count(prefix: str) -> int:
        return sum(1 for tok in vocab if str(tok).startswith(prefix + "_"))

    try:
        if preset == "chain":
            # Recover item_hash_size + size_bins from the vocab so the rebuilt Hash
            # buckets EXACTLY match the corpus the embeddings were fit on.
            item_buckets = _prefix_count("ITEM") or 5000
            size_bins = _prefix_count("SIZE") or 8
            drop = (pargs.get("drop_step"),) if pargs.get("drop_step") else ()
            spec = chain_spec(item_hash_size=item_buckets, size_bins=size_bins, drop_steps=drop)
            return compile_spec(spec, context_len=context_len)
        if preset == "financial":
            merch = int(pargs.get("merchant_hash_size") or _prefix_count("MERCH") or 2000)
            amt = pargs.get("amount_strategy")
            amount_strategy = AmountStrategy(amt) if amt else AmountStrategy.FIXED
            drop = (pargs.get("drop_step"),) if pargs.get("drop_step") else ()
            spec = financial_spec(
                merchant_hash_size=merch,
                amount_strategy=amount_strategy,
                include_time_delta=bool(pargs.get("include_time_delta")),
                drop_steps=drop,
            )
            return compile_spec(spec, context_len=context_len)
    except Exception:  # noqa: BLE001 — a bad rebuild degrades to the verbatim fallback
        return None

    # custom: find the source TokenizerSpec by lineage and recompile its field-map.
    fieldmap = _find_fieldmap(ctx, corpus, ingest)
    if isinstance(fieldmap, dict) and fieldmap:
        try:
            return compile_spec(spec_from_field_map(fieldmap, context_len=context_len), context_len=context_len)
        except Exception:  # noqa: BLE001
            return None
    return None


def _find_fieldmap(ctx: VerbContext, corpus: Any, ingest: Any) -> Optional[dict]:
    """Find the field-map the custom Corpus was tokenized with.

    PRIMARY (self-describing): ``tokenize --spec`` records the resolved field-map
    VERBATIM on the Corpus it writes — on ``extras['fieldmap']`` and in the
    ``corpus.json`` payload. Read it straight off the Corpus so the bridge works for
    ANY custom Corpus, including one tokenized from a hand-edited YAML FILE with no
    ``propose``-written ``TokenizerSpec`` in the store (the common authoring path).

    FALLBACK (lineage): a ``TokenizerSpec`` the proposer wrote, matched by experiment
    then by the IngestDataset the proposal was built on — for older Corpora that
    predate the self-describing record."""
    on_extras = (getattr(corpus, "extras", {}) or {}).get("fieldmap")
    if isinstance(on_extras, dict) and on_extras:
        return on_extras
    on_payload = (_load_corpus_payload(corpus) or {}).get("fieldmap")
    if isinstance(on_payload, dict) and on_payload:
        return on_payload

    ingest_spec = ingest.pathspec if ingest is not None else None
    specs = ctx.store.list("TokenizerSpec")
    by_exp = [s for s in specs if corpus.experiment is not None and s.experiment == corpus.experiment]
    by_lineage = [
        s
        for s in specs
        if ingest_spec is not None
        and ((s.extras or {}).get("proposal", {}) or {}).get("input") == ingest_spec
    ]
    for pool in (by_lineage, by_exp, specs):
        if pool:
            pool.sort(key=lambda s: s.ref.n)
            proposal = (pool[-1].extras or {}).get("proposal", {}) or {}
            fm = proposal.get("fieldmap")
            if isinstance(fm, dict) and fm:
                return fm
    return None


def _make_item_encoder(corpus: Any, ctx: VerbContext, ingest: Any, item_col: str, vocab: dict):
    """Return ``encode(raw_value) -> Optional[int]`` mapping a raw item value to its
    vocab id the SAME way ``tokenize`` did, plus the set of item-family vocab ids.

    The item field is the FieldStep whose ``source`` (or ``name``) equals the inferred
    item column. Its ``transform`` (Hash/Mapping/FixedVocab) is a pure 1:1 value→token
    map — the generic field-map / chain / financial preprocess passes a plain item
    source through verbatim, so encoding a single raw value through ``transform`` is
    faithful to the full corpus-materialization path (NOT a verbatim string match: a
    hashed item maps to its bucket token). Unencodable ⇒ ``None`` (an explicit OOV
    miss).

    One source needs the SAME pre-transform the materialize path applies: a continuous
    field tokenized with the ``amount`` strategy has source ``__amt__<col>__<bins>``
    and a ``FixedVocab`` over the bin INDEX — its ``transform`` expects the already-
    binned index, not the raw magnitude. So when the item field is amount-binned we
    reproduce ``preprocess_field_map``'s log-threshold binning here before the
    transform, exactly as ``materialize_corpus_lines`` did — otherwise a raw value
    like ``50.0`` would be mis-read as bin id 50. This is the row-space → token-space
    bridge for a continuous item field (and the knob the 2-bin-amount case exercises)."""
    from ..engine import strategies
    from ..engine.spec import _amount_bin_thresholds, _log_thresholds

    compiled = _reconstruct_compiled(corpus, ctx, ingest)
    item_step = None
    if compiled is not None:
        # Prefer a name match (the field's user-facing name); the source may be a
        # ``__amt__``/``__cal__`` recipe that never equals the raw column name.
        for step in compiled.spec.steps:
            if step.name == item_col:
                item_step = step
                break
        if item_step is None:
            for step in compiled.spec.steps:
                if step.source == item_col:
                    item_step = step
                    break

    # Detect an amount-recipe source so the encode pre-bins the raw magnitude the
    # SAME way the corpus was materialized (the bins the GOOD vs 2-bin specs differ on).
    amt_bins: Optional[int] = None
    if item_step is not None and str(getattr(item_step, "source", "")).startswith("__amt__"):
        try:
            amt_bins = int(str(item_step.source).rpartition("__")[2])
        except (ValueError, TypeError):
            amt_bins = None

    # The item-family vocab ids: the contiguous block whose token strings share the
    # item step's prefix (kNN candidates live ONLY here — never a special/other field).
    item_ids: set[int] = set()
    item_prefix = None
    if item_step is not None:
        item_prefix = getattr(item_step.strategy, "prefix", None)
    if item_prefix:
        item_ids = {
            int(i) for tok, i in vocab.items() if str(tok).startswith(item_prefix + "_")
        }

    def encode(raw_value: Any) -> Optional[int]:
        if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
            return None
        # Faithful encode: run the item step's strategy transform on a 1-value Series.
        if item_step is not None:
            try:
                series = pd.Series([raw_value])
                if amt_bins is not None:
                    # Reproduce the materialize-time log-threshold binning so the
                    # FixedVocab transform sees the bin INDEX, not the raw magnitude.
                    series = _amount_bin_thresholds(series, _log_thresholds(amt_bins))
                tok = strategies.transform(item_step.strategy, series)
                token_str = str(tok.iloc[0])
                if token_str in vocab:
                    return int(vocab[token_str])
            except Exception:  # noqa: BLE001 — fall through to the verbatim lookup
                pass
        # Verbatim fallback (no reconstructable strategy): try a few prefix forms.
        if item_prefix:
            cand = f"{item_prefix}_{raw_value}"
            if cand in vocab:
                return int(vocab[cand])
        sval = str(raw_value)
        if sval in vocab:
            return int(vocab[sval])
        return None  # explicit OOV miss — a discriminating signal, never a silent drop

    # ``amt_bins`` tells the refine handshake the inferred item field is a continuous
    # field tokenized by the amount strategy — so a low-knn REVIEW/FAIL names the
    # amount ``n_bins`` knob (the real cause), not a hash size.
    return encode, item_ids, amt_bins


# ---------------------------------------------------------------------------
# Intrinsic metrics — read off the spec + the stored corpus, NO training.
# ---------------------------------------------------------------------------


def _load_corpus_payload(corpus: Any) -> Optional[dict]:
    """Load the Corpus ``corpus.json`` payload (vocab + corpus_lines), or None."""
    import json

    path = getattr(corpus, "payload_path", None)
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        return loaded if isinstance(loaded, dict) else None
    except (OSError, ValueError):
        return None


def _id_stream(payload: dict, vocab_size: int) -> tuple[list[list[int]], int, int]:
    """Re-derive the EXACT id'd token stream ``embed`` fit on (``_read_token_stream``
    shape-2: ``corpus_lines`` × the ``{token:id}`` vocab, same drop rule — a token
    not in the vocab is dropped). Returns ``(lines, n_in_vocab_tokens, n_total)``."""
    vocab = payload.get("vocab") or {}
    corpus_lines = payload.get("corpus_lines") or []
    lines: list[list[int]] = []
    n_in = 0
    n_total = 0
    for ln in corpus_lines:
        toks = str(ln).split()
        n_total += len(toks)
        ids = [int(vocab[t]) for t in toks if t in vocab]
        n_in += len(ids)
        if ids:
            lines.append(ids)
    return lines, n_in, n_total


def _intrinsic(payload: dict, corpus: Any, vocab_size: int) -> dict[str, Any]:
    """The training-free health read: oov over the embed-fit stream, the per-id
    occupancy histogram (off ``vocab_size``, NOT the embedding dim), dead/starved
    token fractions, and the contract facts read VERBATIM off the corpus."""
    lines, n_in, n_total = _id_stream(payload, vocab_size)
    # oov: the fraction of the corpus token stream that fell outside the vocab (the
    # same stream/vocab/drop-rule embed fit on). 0 when every token is in-vocab.
    oov_rate = round((n_total - n_in) / n_total, 6) if n_total else 0.0

    # Occupancy: per-id token counts over vocab_size bins (specials included — the
    # full id space the embedding table covers). NEVER off the embedding dim.
    counts = np.zeros(int(vocab_size), dtype=np.int64)
    for ids in lines:
        for i in ids:
            if 0 <= i < vocab_size:
                counts[i] += 1
    occ = counts[counts > 0]
    occ_min = int(occ.min()) if occ.size else 0
    occ_med = float(np.median(occ)) if occ.size else 0.0
    occ_max = int(occ.max()) if occ.size else 0

    # Normalized entropy of the occupancy distribution (1.0 = perfectly uniform use,
    # → 0 = a few ids dominate). A scalar skew signal, no hardcoded threshold.
    total = counts.sum()
    if total > 0 and vocab_size > 1:
        p = counts[counts > 0] / total
        entropy = float(-(p * np.log(p)).sum())
        norm_entropy = round(entropy / np.log(vocab_size), 6)
    else:
        norm_entropy = 0.0

    # dead = ids the corpus never used; starved = used but below the 10th percentile
    # of the non-zero occupancy (elevated-but-not-pathological skew signal).
    dead = int((counts == 0).sum())
    dead_token_frac = round(dead / vocab_size, 6) if vocab_size else 0.0
    if occ.size:
        low = np.percentile(occ, 10)
        starved = int((occ <= low).sum())
    else:
        starved = 0

    report = (getattr(corpus, "extras", {}) or {}).get("contract_report", {}) or {}
    contract = {
        "passed": bool(report.get("passed", True)),
        "injective": bool(report.get("injective", True)),
        "dense": bool(report.get("dense", True)),
        "deterministic": bool(report.get("deterministic", True)),
        "has_fitted_artifact": bool(report.get("has_fitted_artifact", False)),
    }

    return {
        "oov_rate": oov_rate,
        "occupancy": {
            "min": occ_min,
            "median": occ_med,
            "max": occ_max,
            "entropy": norm_entropy,
            "n_occupied": int(occ.size),
        },
        "dead_token_frac": dead_token_frac,
        "starved_count": starved,
        "contract": contract,
    }


# ---------------------------------------------------------------------------
# Extrinsic — embedding-kNN next-item Prec@k on the shared hold-out.
# ---------------------------------------------------------------------------


def _knn_prec_at_k(
    targets: list[dict],
    emb: np.ndarray,
    encode,
    item_ids: set[int],
    k: int,
) -> tuple[float, int, int]:
    """Embedding-kNN next-item Prec@k over the leave-one-last-out targets.

    For each target the predictor reads ONLY history (``prev_item``): map it to a
    token id via ``encode`` (the Corpus vocab/encode bridge), take the k cosine-
    nearest item-family vocab ids (excluding the prev id itself), map the held-out
    actual item to its token id, and score Prec@k in TOKEN space. A held-out actual
    with no vocab token is an EXPLICIT miss (never a silent drop). Returns
    ``(prec_at_k, n_eval, n_oov_actual)``."""
    if not item_ids:
        return 0.0, len(targets), 0

    item_id_arr = np.array(sorted(item_ids), dtype=np.int64)
    item_vecs = emb[item_id_arr]
    norms = np.linalg.norm(item_vecs, axis=1)
    norms[norms == 0] = 1.0
    unit = item_vecs / norms[:, None]

    hits = 0.0
    n_oov_actual = 0
    for t in targets:
        actual_id = encode(t["actual_item"])
        if actual_id is None:
            n_oov_actual += 1
            continue  # explicit OOV miss: contributes 0 to the numerator
        prev_id = encode(t["prev_item"])
        if prev_id is None:
            continue  # no history token to query from → 0 (cannot beat repeat-last)
        q = emb[prev_id]
        qn = np.linalg.norm(q)
        if qn == 0:
            continue
        sims = unit @ (q / qn)
        # k nearest item ids EXCLUDING the query id itself (a fair next-item rank).
        order = np.argsort(-sims)
        topk: list[int] = []
        for idx in order:
            cand = int(item_id_arr[idx])
            if cand == prev_id:
                continue
            topk.append(cand)
            if len(topk) >= k:
                break
        hits += prec_at_k(topk, actual_id)

    n_eval = len(targets)
    prec = round(hits / n_eval, 6) if n_eval else 0.0
    return prec, n_eval, n_oov_actual


# ---------------------------------------------------------------------------
# Controls — read off the resolved Baseline, or recompute via the SHARED helper.
# ---------------------------------------------------------------------------


def _controls(
    baseline: Optional[Any],
    targets: list[dict],
    history: pd.DataFrame,
    cols: dict,
    k: int,
) -> tuple[float, float, str]:
    """``(repeat_last_prec1, popularity_prec_at_k, source)``. Read off the resolved
    Baseline's metrics when present (NEVER recomputed differently than baseline);
    otherwise recompute via the SHARED hold-out helper at matching k — the SAME
    code path baseline uses (popularity fit on history only; repeat-last over the
    targets)."""
    if baseline is not None:
        metrics = ((baseline.extras or {}).get("baseline", {}) or {}).get("metrics", {}) or {}
        if not metrics:
            metrics = (baseline.signatures or {}).get("metrics", {}) or {}
        rli = metrics.get("repeat-last-item", {}).get("value")
        pop = metrics.get("popularity", {}).get("value")
        if rli is not None and pop is not None:
            return float(rli), float(pop), baseline.pathspec

    # Recompute via the shared helper (identical to baseline's own computation).
    item_col = cols["item"]
    n_eval = len(targets) or 1
    rli_hits = sum(1.0 for t in targets if t["actual_item"] == t["prev_item"])
    rli = round(rli_hits / n_eval, 6)
    pop_topk = list(history[item_col].value_counts().head(k).index)
    pop_hits = sum(prec_at_k(pop_topk, t["actual_item"]) for t in targets)
    pop = round(pop_hits / n_eval, 6)
    return rli, pop, "recomputed"


# ---------------------------------------------------------------------------
# The REFINE handshake — knob-named fixes + a structured refine_plan.
# ---------------------------------------------------------------------------


def _build_refine_plan(
    intrinsic: dict,
    knn: float,
    rli: float,
    pop: float,
    cols: dict,
    corpus: Any,
    *,
    item_family_present: bool = True,
    item_amt_bins: Optional[int] = None,
) -> tuple[list[dict], list[Diagnostic]]:
    """Build the structured ``refine_plan`` + matching :class:`Diagnostic`s. Each
    failing dimension names the EXACT fieldmap.yaml knob (the canonical mappings):
    high dead-tokens on a Hash field → lower ``merchant_hash_size``/``item_hash_size``
    (or Hash→mapping); ``knn<=rli`` from under-binned amount → raise amount
    ``n_bins`` / switch to quantile; ``knn<=rli`` from a dropped signal field →
    re-include it. NO hardcoded goodness threshold — the triggers are RELATIVE.

    ``item_family_present`` is False when the inferred item column earns NO vocab
    token (no item-family ids) — the DROP-SIGNAL case where the very field we
    predict next-item on was omitted from the spec. A custom field-map drops a
    field by simply not listing it in ``fields[]`` (NOT via the preset-only
    ``drop_step`` knob, which stays null), so an empty item family is the faithful,
    code-path-agnostic signal that the predictive field is gone — and it names the
    inferred item column as the field to re-include.

    ``item_amt_bins`` is set (to the current bin count) when the inferred item field
    is a continuous field tokenized by the amount strategy — then the actionable knob
    for a low-knn REVIEW/FAIL is that field's amount ``n_bins`` (coarse bins collapsed
    the distinct magnitudes the sequence rides on), NOT a hash size. This is the
    2-bin-amount case where the amount field IS the next-item carrier."""
    plan: list[dict] = []
    diags: list[Diagnostic] = []
    item_field = cols.get("item") or "item"
    amount_field = cols.get("amount") or "amount"
    item_is_amount = item_amt_bins is not None
    # When the item field is the amount carrier, fixes name THAT field's n_bins.
    amt_knob_field = item_field if item_is_amount else amount_field
    amt_current = (
        f"{item_amt_bins} bins (too few)" if item_is_amount else "too few (under-binned)"
    )
    # A dropped signal field. Two faithful detections: the preset ``drop_step`` knob
    # (a step dropped from a financial/chain preset), OR — for a custom field-map,
    # which drops a field by OMISSION (drop_step stays null) — the inferred item
    # column having no item-family vocab token at all (nothing to rank next-item on).
    dropped = (getattr(corpus, "producer_args", {}) or {}).get("drop_step")
    if not dropped and not item_family_present:
        dropped = item_field

    contract = intrinsic["contract"]
    dead = intrinsic["dead_token_frac"]
    occ = intrinsic["occupancy"]

    # 1. contract failure — the corpus itself is broken (C1/C2/C3).
    if not contract["passed"]:
        diags.append(
            Diagnostic(
                contract="INTRINSIC",
                severity=Severity.ERROR,
                message="the Corpus failed its tokenizer contract (C1/C2/C3) — embeddings are not trustworthy",
                fix="fix the contract diagnostic on the Corpus, then re-run tokenize → embed → evaluate",
            )
        )
        plan.append(
            {
                "field": "<corpus>",
                "knob": "contract",
                "current": "failed",
                "suggested": "passing",
                "reason": "tokenizer contract C1/C2/C3 not satisfied",
                "signal": "contract.passed=false",
            }
        )

    # 2. high dead-token fraction → the hash blew the vocab up far beyond the data.
    #    Sparse occupancy (dead_token_frac high) on a hashed field is the classic
    #    HASH-EVERYTHING failure: lower the hash size or switch Hash→mapping.
    if dead > 0.5:
        diags.append(
            Diagnostic(
                contract="INTRINSIC",
                severity=Severity.WARNING if dead <= 0.9 else Severity.ERROR,
                message=f"dead_token_frac={dead} — most of the vocab is never used (over-hashed)",
                fix=(
                    f"lower merchant_hash_size/item_hash_size on field {item_field!r}, "
                    f"or switch Hash→mapping, in TokenizerSpec/<n> fieldmap.yaml; "
                    f"re-run tokenize --spec → embed → evaluate"
                ),
            )
        )
        plan.append(
            {
                "field": item_field,
                "knob": "item_hash_size",
                "current": "too large",
                "suggested": "lower (≈ distinct item count) or switch Hash→mapping",
                "reason": "most hash buckets are dead (vocab >> distinct values)",
                "signal": f"dead_token_frac={dead}",
            }
        )

    # 3. knn cannot beat repeat-last — the spec lost the next-item signal.
    if knn <= rli:
        # a dropped signal field is the strongest, most actionable cause.
        if dropped:
            diags.append(
                Diagnostic(
                    contract="EXTRINSIC",
                    severity=Severity.ERROR,
                    message=f"kNN Prec@k ({knn}) ≤ repeat-last ({rli}) and field {dropped!r} was dropped",
                    fix=f"re-include the dropped field {dropped!r} in fieldmap.yaml fields[]; re-run tokenize → embed → evaluate",
                )
            )
            plan.append(
                {
                    "field": dropped,
                    "knob": "fields[]",
                    "current": "dropped",
                    "suggested": "re-include",
                    "reason": "dropping this field removed the next-item signal",
                    "signal": f"knn={knn} <= repeat_last={rli}",
                }
            )
        else:
            # otherwise an under-binned amount is the canonical next-item-signal loss.
            diags.append(
                Diagnostic(
                    contract="EXTRINSIC",
                    severity=Severity.ERROR,
                    message=f"kNN Prec@k ({knn}) ≤ repeat-last ({rli}) — the spec cannot beat the strong control",
                    fix="raise amount n_bins or switch amount_strategy to quantile in fieldmap.yaml; re-run tokenize → embed → evaluate",
                )
            )
            plan.append(
                {
                    "field": amt_knob_field,
                    "knob": "n_bins",
                    "current": amt_current,
                    "suggested": "raise n_bins or switch amount_strategy to quantile",
                    "reason": "coarse amount bins collapse distinct events to one token",
                    "signal": f"knn={knn} <= repeat_last={rli}",
                }
            )

    # 4. yellow flag: beats repeat-last but not popularity, or skewed occupancy.
    elif knn < pop:
        # When the item field is the amount carrier, the actionable knob is its
        # n_bins (coarse bins blurred the magnitudes), not a hash size.
        if item_is_amount:
            plan.append(
                {
                    "field": amt_knob_field,
                    "knob": "n_bins",
                    "current": amt_current,
                    "suggested": "raise n_bins or switch amount_strategy to quantile",
                    "reason": "beats repeat-last but not popularity; coarse amount bins blur the sequence",
                    "signal": f"knn={knn} < popularity={pop}",
                }
            )
        else:
            plan.append(
                {
                    "field": item_field,
                    "knob": "item_hash_size",
                    "current": "review",
                    "suggested": "consider a tighter hash / more bins on the weakest field",
                    "reason": "beats repeat-last but not the popularity control",
                    "signal": f"knn={knn} < popularity={pop}",
                }
            )

    return plan, diags


# ---------------------------------------------------------------------------
# The verb.
# ---------------------------------------------------------------------------


@register(
    "evaluate",
    summary="score a Corpus's local embeddings vs the controls a model must beat "
            "(intrinsic+extrinsic, CPU, ~$0) — the GO/NO-GO before a GPU-hour",
    tier=Tier.WORKSPACE_WRITE,
    capability_mode=CapabilityMode.SEARCHABLE,
    params=EVALUATE_PARAMS,
)
def _evaluate(args: dict[str, Any], ctx: VerbContext) -> VerbResult:
    # Local import: store is the v0.2 seam; importing DataObject locally avoids the
    # store↔verb import cycle (HARD INVARIANT).
    from ..store import DataObject

    in_spec = args.get("in") or ""
    emb_spec = args.get("embeddings") or None
    base_spec = args.get("baseline") or None
    k = int(args.get("k") or _DEFAULT_K)
    eval_split = args.get("eval_split") or _DEFAULT_EVAL_SPLIT
    experiment = ctx.experiment

    if not in_spec:
        return _refused(
            "evaluate needs an input Corpus/<n>",
            "pass `loom evaluate Corpus/<n>` (after loom embed on it)",
            experiment,
            contract="C2",
        )

    # --- resolve the Corpus + its parent IngestDataset (the raw rows) ----------
    try:
        corpus = ctx.store.get(in_spec)
    except (KeyError, ValueError) as exc:
        return _refused(
            f"evaluate could not resolve Corpus {in_spec!r}: {exc}",
            "run loom tokenize to produce a Corpus/<n> first",
            experiment,
            contract="C2",
        )

    vocab_hash = (corpus.signatures or {}).get("vocab_hash")

    ingest = None
    parents = corpus.parents or []
    if parents:
        try:
            ingest = ctx.store.get(parents[0])
        except (KeyError, ValueError):
            ingest = None

    # --- resolve the Embeddings (default by lineage + the C5 pairing filter) ---
    emb_obj, emb_err = _resolve_embeddings(ctx, corpus, vocab_hash, emb_spec)
    if emb_obj is None:
        return _refused(
            f"evaluate found no embeddings to score for {in_spec}: {emb_err}",
            f"run loom embed {in_spec} first",
            experiment,
            status=Status.REFUSED_NO_METRIC,
            contract="C5",
        )

    # --- C5 GUARD FIRST: the embeddings must be paired to THIS corpus's vocab ---
    # ($0 on a mispaired artifact — before any forward pass over the matrix.)
    emb_sig = (emb_obj.signatures or {}).get("representation_signature")
    if emb_sig != vocab_hash:
        return _refused(
            f"C5 mismatch: {emb_obj.pathspec} representation_signature {emb_sig!r} "
            f"!= Corpus {in_spec} vocab_hash {vocab_hash!r}",
            "embed the SAME Corpus you evaluate (the embeddings are paired to another vocab)",
            experiment,
            contract="C5",
            data={"embeddings": emb_obj.pathspec, "corpus": in_spec},
        )

    # --- load the matrix (numpy API only — torch-free) -------------------------
    from safetensors.numpy import load_file

    safetensors_dir = (emb_obj.extras or {}).get("safetensors_dir")
    import os

    st_path = (
        os.path.join(safetensors_dir, "model.safetensors")
        if safetensors_dir
        else emb_obj.payload_path
    )
    try:
        emb = load_file(st_path)["embeddings"]
    except Exception as exc:  # noqa: BLE001
        return _refused(
            f"evaluate could not read the embeddings matrix at {st_path!r}: {exc}",
            f"re-run loom embed {in_spec}",
            experiment,
            contract="C5",
        )
    emb = np.asarray(emb, dtype=np.float64)
    vocab_size = int((emb_obj.signatures or {}).get("vocab_size") or emb.shape[0])

    # --- the raw rows + the SHARED leave-one-last-out hold-out (C6) ------------
    df = load_rows(ingest) if ingest is not None else None
    payload = _load_corpus_payload(corpus) or {}
    vocab = payload.get("vocab") or {}

    if df is None or len(df) == 0:
        return _refused(
            f"evaluate found no rows on the Corpus's parent IngestDataset for {in_spec}",
            "re-ingest the source so the rows payload is persisted, then re-tokenize → embed",
            experiment,
            contract="C6",
        )

    extras = getattr(ingest, "extras", {}) or {}
    cols = infer_cols(df, extras)
    if cols["entity"] is None or cols["item"] is None:
        return _refused(
            f"evaluate needs an entity + item column (entity={cols['entity']!r}, item={cols['item']!r})",
            "pin --entity at ingest time and ensure an item-like column exists",
            experiment,
            contract="C6",
        )

    targets, history, n_eval = build_leave_one_last_out(df, cols)
    if n_eval == 0:
        return _refused(
            f"evaluate has no held-out targets on {in_spec}: every entity has <2 events",
            "ingest a dataset with multi-event entities (sequences)",
            experiment,
            contract="C6",
        )

    # --- intrinsic (training-free) + extrinsic (kNN) + controls ----------------
    intrinsic = _intrinsic(payload, corpus, vocab_size)
    encode, item_ids, item_amt_bins = _make_item_encoder(corpus, ctx, ingest, cols["item"], vocab)
    knn, _n, n_oov_actual = _knn_prec_at_k(targets, emb, encode, item_ids, k)

    baseline = _resolve_baseline(ctx, corpus, ingest, experiment, base_spec)
    rli, pop, control_src = _controls(baseline, targets, history, cols, k)

    final_objective = (emb_obj.signatures or {}).get("final_objective")

    # --- STRICT VERDICT (Anub) -------------------------------------------------
    # INTRINSIC GATE dominates: contract-fail / pathological oov / pathological dead.
    contract_passed = intrinsic["contract"]["passed"]
    oov_pathological = intrinsic["oov_rate"] >= 0.5
    dead_pathological = intrinsic["dead_token_frac"] >= 0.9
    occupancy_skew = (
        intrinsic["occupancy"]["entropy"] < 0.25 and intrinsic["occupancy"]["n_occupied"] > 1
    )
    dead_warn = intrinsic["dead_token_frac"] > 0.5  # elevated-but-not-pathological

    if (not contract_passed) or oov_pathological or dead_pathological:
        verdict = Verdict.FAIL
    elif knn <= rli:
        # STRICT: cannot beat the strong control → not worth a GPU-hour.
        verdict = Verdict.FAIL
    elif knn > rli and knn >= pop and not (dead_warn or occupancy_skew):
        verdict = Verdict.PASS
    else:
        # knn > rli but a yellow flag (knn < pop, or elevated dead / occupancy skew).
        verdict = Verdict.REVIEW

    beats_repeat_last = bool(knn > rli)

    refine_plan, refine_diags = _build_refine_plan(
        intrinsic, knn, rli, pop, cols, corpus,
        item_family_present=bool(item_ids),
        item_amt_bins=item_amt_bins,
    )

    # --- assemble the scores.json payload + persist a Scores/<n> object --------
    scores = {
        "intrinsic": {
            "oov_rate": intrinsic["oov_rate"],
            "occupancy": {
                "min": intrinsic["occupancy"]["min"],
                "median": intrinsic["occupancy"]["median"],
                "max": intrinsic["occupancy"]["max"],
                "entropy": intrinsic["occupancy"]["entropy"],
            },
            "dead_token_frac": intrinsic["dead_token_frac"],
            "starved_count": intrinsic["starved_count"],
            "contract": intrinsic["contract"],
        },
        "extrinsic": {
            "knn_prec_at_k": knn,
            "final_objective": final_objective,
            "n_oov_actual": n_oov_actual,
        },
        "controls": {
            "repeat_last_prec1": rli,
            "popularity_prec_at_k": pop,
            "beats_repeat_last": beats_repeat_last,
        },
        "refine_plan": refine_plan,
    }

    import json

    ref = ctx.store.new_ref("Scores")
    sobj = DataObject(
        ref=ref,
        kind="Scores",
        content_id=f"{corpus.content_id}:evaluate:{emb_obj.content_id}:{eval_split}:k{k}",
        parents=[
            corpus.pathspec,
            emb_obj.pathspec,
        ]
        + ([ingest.pathspec] if ingest is not None else [])
        + ([baseline.pathspec] if baseline is not None else []),
        producer_verb="evaluate",
        producer_args={"in": in_spec, "embeddings": emb_obj.pathspec, "k": k, "eval_split": eval_split},
        signatures={
            "verdict": verdict.value,
            "knn_prec_at_k": knn,
            "repeat_last_prec1": rli,
            "popularity_prec_at_k": pop,
            "oov_rate": intrinsic["oov_rate"],
            "dead_token_frac": intrinsic["dead_token_frac"],
            "n_eval": n_eval,
            "eval_split": eval_split,
            "k": k,
        },
        verdict=verdict,
        status=Status.OK,
        experiment=experiment,
        created_at=datetime.now(timezone.utc).isoformat(),
        extras={"scores": scores},
    )
    stored = ctx.store.put(sobj, payload=json.dumps(scores), payload_name="scores.json")

    # --- the readable one-liner + the dual-driver envelope ---------------------
    summary = (
        f"{stored.pathspec}  verdict={verdict.value}  knn@{k}={knn} "
        f"vs repeat-last={rli} popularity@{k}={pop}  "
        f"oov={intrinsic['oov_rate']} dead={intrinsic['dead_token_frac']}  (n={n_eval})"
    )

    data_block = {
        "pathspec": stored.pathspec,
        "corpus": corpus.pathspec,
        "embeddings": emb_obj.pathspec,
        "ingest": ingest.pathspec if ingest is not None else None,
        "baseline": baseline.pathspec if baseline is not None else None,
        "control_source": control_src,
        "k": k,
        "eval_split": eval_split,
        "n_eval": n_eval,
        "intrinsic": scores["intrinsic"],
        "extrinsic": scores["extrinsic"],
        "controls": scores["controls"],
        "refine_plan": refine_plan,
    }

    return VerbResult(
        verb="evaluate",
        status=Status.OK,
        verdict=verdict,
        tier=Tier.WORKSPACE_WRITE,
        capability_mode=CapabilityMode.SEARCHABLE,
        summary=summary,
        outputs=[stored.ref],
        diagnostics=refine_diags,
        data=data_block,
        experiment=experiment,
        cost_plan=CostPlan(),
    )


__all__ = ["EVALUATE_PARAMS"]
