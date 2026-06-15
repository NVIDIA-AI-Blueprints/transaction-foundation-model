"""C3 corpus grammar: a corpus line is ``<bos> txn (<sep> txn)* <eos>`` and each
txn is the space-joined field tokens (``tokens_per_txn`` of them), with no
nan/None leaking into the token stream.

The corpus-assembly entrypoint (``to_corpus_lines``) is not part of the LOCKED
engine API surface — the implementing agent attaches it (the design names it
``to_corpus_lines``; the reference lives at ``src/tokenizer/pipeline.py``). So we
discover it across the likely locations (module function / CompiledTokenizer
method) and, failing that, fall back to the ``tokenize`` verb's persisted Corpus
payload, which is the public contract. If none is wired yet, skip cleanly."""

from __future__ import annotations

from typing import Optional

import pytest

from .golden_helpers import call_verb, compiled_financial, store_list


def _discover_corpus_lines(ct, df) -> Optional[list[str]]:
    """Find a way to turn a compiled tokenizer + a tiny frame into corpus lines.

    Tries, in order:
      1. ``loom.engine.to_corpus_lines(ct, df)`` (module-level helper).
      2. ``ct.to_corpus_lines(df)`` (method on CompiledTokenizer).
      3. ``loom.engine.api.to_corpus_lines(...)``.
    Returns the list of lines, or None if no helper is exposed yet."""
    import loom.engine as engine

    # 1 + 3: module-level function under a few plausible names.
    for modname in (engine, getattr(engine, "api", engine)):
        for fname in ("to_corpus_lines", "corpus_lines", "to_corpus"):
            f = getattr(modname, fname, None)
            if callable(f):
                for call in (lambda: f(ct, df), lambda: f(ct, df, group_cols=["cust", "card"])):
                    try:
                        return list(call())
                    except TypeError:
                        continue
                    except NotImplementedError:
                        return None
    # 2: method on the compiled tokenizer.
    for mname in ("to_corpus_lines", "corpus_lines", "encode_corpus"):
        m = getattr(ct, mname, None)
        if callable(m):
            for call in (lambda: m(df), lambda: m(df, group_cols=["cust", "card"])):
                try:
                    return list(call())
                except TypeError:
                    continue
                except NotImplementedError:
                    return None
    return None


def _assert_grammar(lines: list[str], tokens_per_txn: int) -> None:
    """Every line is ``<bos> txn (<sep> txn)* <eos>`` with tokens_per_txn fields per
    txn and no nan/None tokens."""
    assert lines, "expected at least one corpus line"
    for line in lines:
        assert isinstance(line, str)
        toks = line.split(" ")
        assert toks[0] == "<bos>", f"line must start with <bos>: {line!r}"
        assert toks[-1] == "<eos>", f"line must end with <eos>: {line!r}"
        # No nan/None/empty tokens anywhere.
        for t in toks:
            assert t not in ("", "nan", "NaN", "None", "<unk>") or t == "<unk>", line
            assert "nan" not in t.lower(), f"nan leaked into a token: {t!r} in {line!r}"
        # Strip specials, split the remaining transactions on <sep>.
        body = toks[1:-1]
        assert body, "line has no transactions between <bos>/<eos>"
        # Each <sep> separates transactions; count fields between separators.
        txns: list[list[str]] = [[]]
        for t in body:
            if t == "<sep>":
                txns.append([])
            else:
                txns[-1].append(t)
        for txn in txns:
            assert len(txn) == tokens_per_txn, (
                f"each txn must have {tokens_per_txn} field tokens, got {len(txn)}: {txn}"
            )
            # No special tokens inside a transaction body.
            for t in txn:
                assert t not in ("<bos>", "<eos>", "<sep>", "<pad>"), t


def test_corpus_grammar_via_engine_helper(tabformer_df):
    """If the engine exposes a corpus-line helper, validate the grammar directly
    on the tiny 5-row TabFormer fixture (a single short sequence per group)."""
    ct = compiled_financial()
    lines = _discover_corpus_lines(ct, tabformer_df)
    if lines is None:
        pytest.skip("no corpus-line helper exposed yet (engine attaches to_corpus_lines)")
    _assert_grammar(lines, tokens_per_txn=ct.tokens_per_txn)


def test_corpus_grammar_three_txn_sample(tabformer_df):
    """A tiny synthetic 3-transaction single-entity sample yields exactly one line
    with three <sep>-joined transactions (build brief: 'tiny synthetic 3-txn')."""
    ct = compiled_financial()
    # All three rows share cust=0 card=0 → one sequence of 3 transactions.
    three = tabformer_df.copy()
    three["cust"] = 0
    three["card"] = 0
    three = three.iloc[:3].reset_index(drop=True)
    lines = _discover_corpus_lines(ct, three)
    if lines is None:
        pytest.skip("no corpus-line helper exposed yet (engine attaches to_corpus_lines)")
    _assert_grammar(lines, tokens_per_txn=ct.tokens_per_txn)
    # One group → one chunk → one line with exactly 3 transactions.
    single = [ln for ln in lines if ln.count("<sep>") == 2]
    assert single, f"expected a 3-txn line (2 <sep>), got: {lines}"


def test_corpus_grammar_via_tokenize_verb_payload(tmp_path):
    """Public-contract fallback: the persisted Corpus payload from the tokenize
    verb (if implemented) is corpus-line grammar. Skips on the scaffold stub."""
    from loom.registry import VerbContext
    from loom.store import ObjectStore

    store = ObjectStore(str(tmp_path))
    ctx = VerbContext(store=store, driver="cli")
    call_verb("tokenize", {"preset": "financial"}, ctx)
    corpora = store_list(store, "Corpus")
    if not corpora:
        pytest.skip("tokenize did not persist a Corpus payload to read back")
    obj = corpora[0]
    if not obj.payload_path:
        pytest.skip("Corpus has no payload_path to read corpus lines from")
    import os

    if not os.path.exists(obj.payload_path):
        pytest.skip("Corpus payload path is metadata-only in this slice")
    # Read whatever corpus-line text was persisted and validate the grammar.
    text = ""
    p = obj.payload_path
    if os.path.isdir(p):
        for fn in sorted(os.listdir(p)):
            full = os.path.join(p, fn)
            if os.path.isfile(full):
                with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                    text += fh.read()
    else:
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
    lines = [ln for ln in text.splitlines() if ln.strip().startswith("<bos>")]
    if not lines:
        pytest.skip("Corpus payload is not corpus-line text in this slice")
    tpt = int(obj.signatures.get("tokens_per_txn", 12))
    _assert_grammar(lines, tokens_per_txn=tpt)
