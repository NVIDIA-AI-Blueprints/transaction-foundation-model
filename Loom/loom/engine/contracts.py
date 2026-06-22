"""The contract compiler + C1/C2/C3 checks (DESIGN.md §0, §7.2).

This is where a :class:`~loom.engine.api.TokenizerSpec` becomes a
:class:`~loom.engine.api.CompiledTokenizer`: specials at ids 0-4, then each
step's contiguous block at a running offset using **0-based local indices**
(``id = offset + local_index``). Contracts are surfaced as named-diff
:class:`~loom.types.Diagnostic` cards, never stack traces:

  * **C1 (injectivity + density):** every ``(step, value)`` maps to a UNIQUE,
    contiguous id; the vocab is dense ``0..vocab_size-1``. On a collision the
    compiler emits a FAIL diagnostic naming the two colliding tokens + ids + a
    fix, and the verb layer refuses to write a Corpus.
  * **C2 (determinism):** the vocab is built from config alone. A fitted-artifact
    strategy (quantile/kmeans amount) is flagged WARNING and its state must be
    persisted (``has_fitted_artifact``).
  * **C3 (grammar):** ``chunk_size = context_len // (tokens_per_txn + 1)`` is
    derived and announced; a corpus line is ``<bos> txn (<sep> txn)* <eos>``.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Optional

import pandas as pd

from ..types import Diagnostic, Severity
from . import strategies
from .api import (
    SPECIAL_TOKENS,
    AmountStrategy,
    ChunkGrammar,
    CompiledTokenizer,
    ContractReport,
    FieldStep,
    TokenizerSpec,
)


# ---------------------------------------------------------------------------
# Vocab layout — the bug fix: 0-based local indices at a running offset.
# ---------------------------------------------------------------------------


def _layout_vocab(
    spec: TokenizerSpec, report: ContractReport
) -> tuple[dict[str, int], dict[int, str]]:
    """Lay out specials (0-4) then each step's block at a running offset.

    Returns ``(vocab, id_to_token)``. On a collision (two distinct tokens
    assigned the same id) C1 fails with a named diagnostic; on a dead/skipped id
    (a gap that would make the vocab non-dense) C1 also fails. The reference bug
    (FixedVocab keyed ids by raw value, so MONTH min_val=1 left a dead id 2167
    and collided MONTH_12 with CARD_0 at 2179) cannot occur here because ids use
    0-based local indices — but C1 still actively checks for ANY overlap so a
    hand-built overlapping spec is refused."""
    vocab: dict[str, int] = {}
    id_to_token: dict[int, str] = {}

    def _assign(token: str, tid: int, owner: str) -> None:
        if tid in id_to_token and id_to_token[tid] != token:
            _c1_collision(report, id_to_token[tid], token, tid, owner)
        if token in vocab and vocab[token] != tid:
            _c1_collision(report, token, token, tid, owner, dup_token=True)
        vocab[token] = tid
        id_to_token[tid] = token

    # Specials occupy ids 0..4 in the locked order.
    offset = 0
    for tok in SPECIAL_TOKENS:
        _assign(tok, offset, "specials")
        offset += 1

    # Each field step lays a contiguous block at the running offset.
    for step in spec.steps:
        tokens = strategies.build_vocab(step.strategy)
        for local_index, token in enumerate(tokens):
            _assign(token, offset + local_index, step.name)
        offset += len(tokens)

    return vocab, id_to_token


def _c1_collision(
    report: ContractReport,
    token_a: str,
    token_b: str,
    tid: int,
    owner: str,
    *,
    dup_token: bool = False,
) -> None:
    report.injective = False
    if dup_token:
        msg = (
            f"C1 injectivity FAIL: token {token_a!r} is assigned to more than "
            f"one id (last in step {owner!r})."
        )
        fix = (
            "two steps emit the same token string; rename the prefix on one "
            "step so every (step,value) maps to a unique token."
        )
        data = {"token": token_a, "id": tid, "step": owner}
    else:
        msg = (
            f"C1 injectivity FAIL: id {tid} is claimed by both {token_a!r} and "
            f"{token_b!r} (step {owner!r}) — overlapping blocks."
        )
        fix = (
            "block offsets overlap; rebuild the spec so each step's block starts "
            "at the running offset (id = offset + local_index, 0-based). "
            "Reordering shifts every id → vocab_hash changes → retrain required."
        )
        data = {"id": tid, "token_a": token_a, "token_b": token_b, "step": owner}
    report.add(
        Diagnostic(contract="C1", severity=Severity.ERROR, message=msg, fix=fix, data=data)
    )


# ---------------------------------------------------------------------------
# C1 density — the laid-out ids must be exactly 0..vocab_size-1, no gaps.
# ---------------------------------------------------------------------------


def _check_density(
    report: ContractReport, id_to_token: dict[int, str], vocab_size: int
) -> None:
    ids = set(id_to_token.keys())
    expected = set(range(vocab_size))
    if ids != expected:
        report.dense = False
        missing = sorted(expected - ids)[:8]
        extra = sorted(ids - expected)[:8]
        report.add(
            Diagnostic(
                contract="C1",
                severity=Severity.ERROR,
                message=(
                    f"C1 density FAIL: ids are not dense 0..{vocab_size - 1} "
                    f"(missing={missing}, out-of-range={extra})."
                ),
                fix="a dead/skipped id breaks density; lay blocks contiguously.",
                data={"missing": missing, "out_of_range": extra, "vocab_size": vocab_size},
            )
        )


# ---------------------------------------------------------------------------
# C2 determinism — flag fitted-artifact strategies (quantile/kmeans amount).
# ---------------------------------------------------------------------------


def _check_determinism(report: ContractReport, spec: TokenizerSpec) -> None:
    fitted_steps = [s.name for s in spec.steps if s.fitted]
    is_fitted_amount = spec.amount_strategy in (
        AmountStrategy.QUANTILE,
        AmountStrategy.KMEANS,
    )
    if fitted_steps or is_fitted_amount:
        report.has_fitted_artifact = True
        names = fitted_steps or ["amt"]
        report.add(
            Diagnostic(
                contract="C2",
                severity=Severity.WARNING,
                message=(
                    f"C2: fitted-artifact strategy in use "
                    f"(amount_strategy={spec.amount_strategy.value}, steps={names}). "
                    "Vocab is no longer config-only; fitted state MUST be persisted "
                    "with the Corpus to keep determinism recoverable."
                ),
                fix="use amount_strategy=fixed for the deterministic default path, "
                "or accept the fitted-artifact burden (state is persisted).",
                data={"amount_strategy": spec.amount_strategy.value, "fitted_steps": names},
            )
        )
    else:
        report.add(
            Diagnostic(
                contract="C2",
                severity=Severity.INFO,
                message="C2 determinism OK — vocab is config-only, no fitted artifact.",
                data={"amount_strategy": spec.amount_strategy.value},
            )
        )


# ---------------------------------------------------------------------------
# C3 grammar — chunk_size derivation + corpus-line builder.
# ---------------------------------------------------------------------------


def derive_chunk_size(context_len: int, tokens_per_txn: int) -> int:
    """``chunk_size = context_len // (tokens_per_txn + 1)`` — the +1 is the
    per-transaction ``<sep>`` (C3)."""
    return context_len // (tokens_per_txn + 1)


def _check_grammar(
    report: ContractReport, context_len: int, tokens_per_txn: int, chunk_size: int
) -> None:
    if chunk_size < 1:
        report.add(
            Diagnostic(
                contract="C3",
                severity=Severity.ERROR,
                message=(
                    f"C3 grammar FAIL: chunk_size={chunk_size} (context_len="
                    f"{context_len}, tokens_per_txn={tokens_per_txn}); a chunk holds "
                    "less than one transaction."
                ),
                fix="raise context_len or lower tokens_per_txn so "
                "context_len // (tokens_per_txn + 1) >= 1.",
                data={"context_len": context_len, "tokens_per_txn": tokens_per_txn},
            )
        )
    else:
        report.add(
            Diagnostic(
                contract="C3",
                severity=Severity.INFO,
                message=(
                    f"C3: chunk_size {chunk_size} = {context_len} // "
                    f"({tokens_per_txn} + 1) — derived & announced."
                ),
                data={
                    "context_len": context_len,
                    "tokens_per_txn": tokens_per_txn,
                    "chunk_size": chunk_size,
                },
            )
        )


def to_corpus_lines(
    token_df: pd.DataFrame,
    group_cols: list[str],
    chunk_size: int,
    *,
    field_cols: Optional[list[str]] = None,
    bos: str = "<bos>",
    eos: str = "<eos>",
    sep: str = "<sep>",
) -> list[str]:
    """Assemble corpus lines from a per-transaction token dataframe (C3 grammar).

    Each input row is one transaction whose field token columns (``field_cols``,
    default = all columns not in ``group_cols``) join with a space into a txn.
    Rows are grouped by ``group_cols`` (the entity), then sliced into windows of
    at most ``chunk_size`` transactions. Each window becomes a line:

        ``<bos> txn (<sep> txn)* <eos>``

    Group order and within-group row order are preserved (callers sort first for
    C6). Returns the list of corpus-line strings."""
    if field_cols is None:
        field_cols = [c for c in token_df.columns if c not in group_cols]

    lines: list[str] = []
    grouper = token_df.groupby(group_cols, sort=False) if group_cols else [(None, token_df)]
    for _, grp in grouper:
        txns = [
            " ".join(str(grp.iloc[i][c]) for c in field_cols) for i in range(len(grp))
        ]
        for start in range(0, len(txns), chunk_size):
            window = txns[start : start + chunk_size]
            body = f" {sep} ".join(window)
            lines.append(f"{bos} {body} {eos}")
    return lines


# ---------------------------------------------------------------------------
# vocab_hash — stable sha256 over the ordered (token,id) pairs.
# ---------------------------------------------------------------------------


def compute_vocab_hash(vocab: dict[str, int]) -> str:
    """Deterministic ``sha256:<hex>`` over the sorted ``(token,id)`` pairs (C1
    signature). Independent of insertion order; changes ⇒ retrain required."""
    h = hashlib.sha256()
    for token, tid in sorted(vocab.items(), key=lambda kv: kv[1]):
        h.update(f"{token}\t{tid}\n".encode("utf-8"))
    return f"sha256:{h.hexdigest()}"


# ---------------------------------------------------------------------------
# The compiler.
# ---------------------------------------------------------------------------


def compile_spec(spec: TokenizerSpec, context_len: int = 4096) -> CompiledTokenizer:
    report = ContractReport()

    # C1: lay out specials + step blocks (0-based local indices) and check
    # injectivity as we go.
    vocab, id_to_token = _layout_vocab(spec, report)
    vocab_size = len(id_to_token)

    # C1: density — the ids must be exactly 0..vocab_size-1.
    _check_density(report, id_to_token, vocab_size)

    # C2: determinism / fitted-artifact flag.
    _check_determinism(report, spec)

    # C3: grammar / chunk derivation.
    tokens_per_txn = spec.tokens_per_txn()
    chunk_size = derive_chunk_size(context_len, tokens_per_txn)
    _check_grammar(report, context_len, tokens_per_txn, chunk_size)

    vocab_hash = compute_vocab_hash(vocab)
    grammar = ChunkGrammar(
        context_len=context_len,
        tokens_per_txn=tokens_per_txn,
        chunk_size=chunk_size,
    )

    return CompiledTokenizer(
        vocab=vocab,
        id_to_token=id_to_token,
        vocab_size=vocab_size,
        vocab_hash=vocab_hash,
        tokens_per_txn=tokens_per_txn,
        chunk_size=chunk_size,
        context_len=context_len,
        grammar=grammar,
        report=report,
        spec=spec,
        fitted_state=None,
    )


__all__ = [
    "compile_spec",
    "to_corpus_lines",
    "derive_chunk_size",
    "compute_vocab_hash",
]
