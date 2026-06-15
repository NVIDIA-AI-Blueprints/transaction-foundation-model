"""C3: ``chunk_size = context_len // (tokens_per_txn + 1)`` is DERIVED and
announced (build brief §"chunk derivation"). Pins the three reference points:

  * financial:            tokens_per_txn 12 → chunk_size 4096 // 13 == 315
  * financial + TDIF:     tokens_per_txn 13 → chunk_size 4096 // 14 == 292, vocab 6283
  * chain (DEX):          tokens_per_event 7 → chunk_size 4096 // 8  == 512
"""

from __future__ import annotations

from .golden_helpers import compiled_chain, compiled_financial


def test_financial_chunk_size_is_315():
    ct = compiled_financial()
    assert ct.tokens_per_txn == 12
    assert ct.context_len == 4096
    assert ct.chunk_size == 315
    # The grammar object carries the same derivation.
    assert ct.grammar.chunk_size == 315
    assert ct.grammar.tokens_per_txn == 12
    assert ct.grammar.context_len == 4096
    # Recompute the formula explicitly.
    assert ct.chunk_size == ct.context_len // (ct.tokens_per_txn + 1)


def test_financial_with_time_delta_derivation():
    ct = compiled_financial(include_time_delta=True)
    assert ct.tokens_per_txn == 13, "TDIF appends a 13th field"
    assert ct.chunk_size == 292, "4096 // (13+1) == 292"
    assert ct.vocab_size == 6283, "6251 + 32 TDIF bins == 6283"
    assert ct.chunk_size == ct.context_len // (ct.tokens_per_txn + 1)
    # The TDIF token family is present and exactly 32 bins.
    tdif = [t for t in ct.vocab if t.startswith("TDIF")]
    assert len(tdif) == 32


def test_chain_chunk_size_is_512():
    ct = compiled_chain()
    assert ct.tokens_per_txn == 7, "chain has 7 tokens per event"
    assert ct.chunk_size == 512, "4096 // (7+1) == 512"
    assert ct.chunk_size == ct.context_len // (ct.tokens_per_txn + 1)
    # Chain vocab is DERIVED (not hardcoded) — assert it's in the documented
    # neighborhood (~5081 illustrative) and consistent with its parts, not a fixed
    # magic number.
    assert 4000 < ct.vocab_size < 7000
    assert ct.report.passed, ct.report.diagnostics


def test_chunk_size_is_a_pure_function_of_context_len():
    """Non-default context_len re-derives chunk_size by the same formula (C3)."""
    from loom.engine import compile_spec, financial_spec

    from .golden_helpers import require_engine

    for ctx_len in (2048, 4096, 8192):
        ct = require_engine(lambda c=ctx_len: compile_spec(financial_spec(), context_len=c))
        assert ct.context_len == ctx_len
        assert ct.chunk_size == ctx_len // (12 + 1)
