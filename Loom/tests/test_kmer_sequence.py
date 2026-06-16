"""FIX 2 — a generic ``kmer`` strategy + minimal sequence path (the DNA probe).

The generality test: Loom must be able to tokenize a **fixed-alphabet biological
sequence** (DNA over ``{A,C,G,T}``) so the DNA control runs. A new generic
``kmer`` strategy — given a sequence column (a string over an alphabet) and ``k``
(+ ``stride``, overlapping by default) — tokenizes the sequence into k-mer tokens
(vocab = the k-mers, ``4**k`` for DNA, ``KMER_``-prefixed) and emits the
per-position token SEQUENCE for that row (one input sequence → MANY tokens).

**The HARD constraint this file pins (the generality claim):** adding k-mer
required NO change to ``loom/{registry,types,store,tools}.py`` and NO change to
``engine/contracts.py`` — only a new Strategy + the field-map/compile wiring
(+ minimal sequence handling). These tests therefore drive ONLY:

  * ``spec_from_field_map`` (the ``kmer`` field-map keyword → a TokenizerSpec),
  * the LOCKED ``compile_spec`` + C1/C2/C3 (UNCHANGED),
  * ``materialize_corpus_lines`` (the existing corpus path),
  * ``loom.engine.strategies`` (the new Strategy's own vocab/transform),

and assert the k-mer vocab is injective+dense and a known sequence tokenizes to
the expected ordered k-mer token list — all WITHOUT touching the harness/contracts.
If k-mer cannot compile/materialize through these unchanged surfaces, the harness
is secretly tabular-shaped (the build brief's STOP-and-REPORT finding).
"""

from __future__ import annotations

import itertools

import pandas as pd

from loom.engine import compile_spec, materialize_corpus_lines, spec_from_field_map

# DNA — the fixed alphabet for the control. Sorted so expected k-mer ordering is
# unambiguous (a dense, injective vocab must enumerate the alphabet deterministically).
DNA = "ACGT"


# ---------------------------------------------------------------------------
# Helpers — the EXPECTED k-mer vocab + the EXPECTED token sequence for a string.
# These are computed independently of the implementation (the oracle).
# ---------------------------------------------------------------------------


def _expected_kmer_vocab(alphabet: str, k: int) -> list[str]:
    """Every length-``k`` string over ``sorted(alphabet)``, lexicographic order.

    For DNA k=3 this is the 4**3 = 64 codons AAA, AAC, AAG, AAT, ACA, … TTT."""
    letters = sorted(set(alphabet))
    return ["".join(p) for p in itertools.product(letters, repeat=k)]


def _expected_token_seq(seq: str, k: int, stride: int = 1) -> list[str]:
    """The overlapping k-mer token sequence of ``seq`` (the per-position oracle).

    ``KMER_<kmer>`` for each window ``seq[i:i+k]`` at positions ``i = 0, stride,
    2*stride, …`` while a full k-mer fits. For ``GATTACA``, k=3, stride=1 →
    GAT, ATT, TTA, TAC, ACA → 5 tokens."""
    out = []
    for i in range(0, len(seq) - k + 1, stride):
        out.append(f"KMER_{seq[i:i + k]}")
    return out


def _kmer_field_map(*, k: int, stride: int = 1, alphabet: str = DNA, entity="seq_id") -> dict:
    """A minimal single-field ``kmer`` field-map (the DNA control spec).

    The field is NAMED ``kmer`` so its auto-derived token prefix is ``KMER`` (the
    token prefix is the field name uppercased, exactly like every other strategy)
    → tokens are ``KMER_<kmer>`` (e.g. ``KMER_GAT``)."""
    fm = {
        "version": "loom-fieldmap/1",
        "entity": entity,
        "event": "read",
        "context_len": 4096,
        "fields": [
            {
                "name": "kmer",
                "source": "sequence",
                "strategy": "kmer",
                "k": k,
                "stride": stride,
                "alphabet": alphabet,
            }
        ],
    }
    return fm


def _kmer_tokens_in(blob: str) -> list[str]:
    """The ``KMER_*`` tokens (in order) from a corpus-line blob, specials stripped.

    Robust to the exact emission mechanism (one space-joined cell vs. exploded
    rows): we just read the ordered KMER_ tokens out of the materialized corpus."""
    return [t for t in blob.split(" ") if t.startswith("KMER_")]


def _materialized_kmer_tokens(ct, df) -> list[str]:
    """Materialize a corpus over ``df`` and return its ordered ``KMER_*`` tokens.

    ``materialize_corpus_lines`` returns ``(lines, n_positions)``; we unpack the
    lines, join them, and read the k-mer tokens in emission order."""
    lines, _ = materialize_corpus_lines(ct, df)
    return _kmer_tokens_in(" ".join(lines))


def _kmer_token_set(ct) -> set[str]:
    return {t for t in ct.vocab if t.startswith("KMER_")}


def _assert_kmer_vocab(ct, alphabet: str, k: int) -> None:
    """Assert the compiled vocab is the dense k-mer code over ``alphabet`` at ``k``.

    Pins the essential contract while staying ROBUST to whether the strategy mints an
    optional single out-of-alphabet default bucket (an implementation choice that has
    been in flux): EVERY one of the ``len(alphabet)**k`` codons must be present, the
    vocab is injective + dense + C1/C2/C3-clean, and any EXTRA ``KMER_`` token is at
    most ONE default bucket (e.g. ``KMER_UNK``) — never a missing/duplicate codon."""
    codons = {f"KMER_{m}" for m in _expected_kmer_vocab(alphabet, k)}
    assert len(codons) == len(set(alphabet)) ** k
    kmer_tokens = _kmer_token_set(ct)
    # all codons present.
    assert codons <= kmer_tokens, f"missing codons: {sorted(codons - kmer_tokens)[:8]}"
    # any surplus is at most a single default bucket.
    surplus = kmer_tokens - codons
    assert len(surplus) <= 1, f"unexpected extra k-mer tokens: {sorted(surplus)}"
    # injective + dense + clean (the perfect-hash code compiles through C1/C2/C3).
    assert ct.report.passed, [d.message for d in ct.report.diagnostics]
    assert ct.report.injective and ct.report.dense
    assert not ct.report.has_fitted_artifact
    ids = list(ct.vocab.values())
    assert len(ids) == len(set(ids)) and set(ids) == set(range(ct.vocab_size))
    # vocab_size = 5 specials + the k-mer block (codons [+ at most one default]).
    assert ct.vocab_size == 5 + len(kmer_tokens)


# ---------------------------------------------------------------------------
# Compile path — the kmer field-map compiles through the LOCKED gate unchanged.
# ---------------------------------------------------------------------------


def test_kmer_field_map_compiles_to_a_compiled_tokenizer():
    """A DNA ``kmer`` field-map (k=3) compiles via ``spec_from_field_map`` →
    ``compile_spec`` to a CompiledTokenizer — through the EXISTING gate, no
    contract surgery."""
    ct = compile_spec(spec_from_field_map(_kmer_field_map(k=3)))
    # The custom (BYO) preset — the financial/chain dual-driver is untouched.
    assert ct.spec.preset == "custom"
    # One kmer field → one step.
    assert ct.tokens_per_txn == 1


def test_kmer_vocab_is_the_kmer_set_4_pow_k():
    """The vocab is the dense set of 4**k DNA k-mers (``KMER_<kmer>``), enumerating
    a perfect-hash code (injective + dense) — every codon present, no codon missing,
    plus the 5 specials (and at most a single out-of-alphabet default bucket)."""
    k = 3
    assert len(_expected_kmer_vocab(DNA, k)) == 4 ** k == 64
    ct = compile_spec(spec_from_field_map(_kmer_field_map(k=k)))
    _assert_kmer_vocab(ct, DNA, k)


def test_kmer_vocab_is_injective_and_dense_c1_c2_c3_pass():
    """C1 (injective + dense), C2 (config-only, no fitted artifact), C3 (grammar)
    all PASS on the compiled k-mer spec — the k-mer vocab is injective + dense by
    construction (a fixed alphabet enumerates a perfect-hash code)."""
    ct = compile_spec(spec_from_field_map(_kmer_field_map(k=3)))
    assert ct.report.passed, [d.message for d in ct.report.diagnostics]
    assert ct.report.injective  # C1: blocks disjoint, no collisions
    assert ct.report.dense      # C1: ids dense 0..vocab_size-1
    assert not ct.report.has_fitted_artifact  # C2: the alphabet is config, not fitted
    # density + injectivity straight off the vocab too.
    ids = list(ct.vocab.values())
    assert len(ids) == len(set(ids))
    assert set(ids) == set(range(ct.vocab_size))


def test_kmer_vocab_size_scales_as_4_pow_k():
    """The k-mer code scales as ``4**k`` for several k — the generic, alphabet-sized
    dense vocab (all codons present at each k)."""
    for k in (1, 2, 3, 4):
        ct = compile_spec(spec_from_field_map(_kmer_field_map(k=k)))
        _assert_kmer_vocab(ct, DNA, k)


# ---------------------------------------------------------------------------
# Sequence path — one input sequence → MANY k-mer tokens (the expected order).
# ---------------------------------------------------------------------------


def test_tokenizing_a_dna_sequence_yields_the_expected_kmer_sequence():
    """The headline: tokenizing a known DNA string yields the EXPECTED ordered
    k-mer token sequence (one input sequence → many tokens). ``GATTACA``, k=3,
    overlapping → GAT, ATT, TTA, TAC, ACA."""
    seq = "GATTACA"
    k = 3
    n_input_rows = 1
    df = pd.DataFrame({"seq_id": ["r0"], "sequence": [seq]})
    ct = compile_spec(spec_from_field_map(_kmer_field_map(k=k)))
    lines, n_positions = materialize_corpus_lines(ct, df)
    blob = " ".join(lines)
    got = _kmer_tokens_in(blob)
    assert got == _expected_token_seq(seq, k), got
    assert got == ["KMER_GAT", "KMER_ATT", "KMER_TTA", "KMER_TAC", "KMER_ACA"]
    # ONE input sequence row expanded to MANY tokens (the sequence path, not a
    # tabular 1:1): the materialize explodes the row into one position per k-mer.
    assert len(df) == n_input_rows
    assert len(got) == 5 > n_input_rows
    assert n_positions == 5  # the exploded position count, not the input row count
    # one input sequence → exactly one grouped corpus line.
    assert len(lines) == 1


def test_overlapping_default_is_stride_one():
    """The default is OVERLAPPING k-mers (stride 1): an n-length sequence yields
    ``n - k + 1`` tokens."""
    seq = "ACGTACGT"  # length 8
    k = 3
    df = pd.DataFrame({"seq_id": ["r0"], "sequence": [seq]})
    ct = compile_spec(spec_from_field_map(_kmer_field_map(k=k)))
    got = _materialized_kmer_tokens(ct, df)
    assert len(got) == len(seq) - k + 1 == 6
    assert got == _expected_token_seq(seq, k, stride=1)


def test_non_overlapping_stride_equals_k():
    """A non-overlapping tiling (stride == k) yields ``len // k`` disjoint k-mers."""
    seq = "AAACCCGGGTTT"  # length 12
    k = 3
    df = pd.DataFrame({"seq_id": ["r0"], "sequence": [seq]})
    ct = compile_spec(spec_from_field_map(_kmer_field_map(k=k, stride=k)))
    got = _materialized_kmer_tokens(ct, df)
    assert got == ["KMER_AAA", "KMER_CCC", "KMER_GGG", "KMER_TTT"]
    assert got == _expected_token_seq(seq, k, stride=k)


def test_every_emitted_kmer_token_is_in_the_vocab():
    """Round-trip soundness: every k-mer token the materialize path emits is a
    real vocab id (the transform never invents an out-of-vocab token)."""
    seqs = ["GATTACA", "ACGTACGTAC", "TTTTAAAA", "CGCGCGCG"]
    k = 3
    df = pd.DataFrame({"seq_id": [f"r{i}" for i in range(len(seqs))], "sequence": seqs})
    ct = compile_spec(spec_from_field_map(_kmer_field_map(k=k)))
    got = _materialized_kmer_tokens(ct, df)
    assert got, "expected k-mer tokens in the corpus"
    for tok in got:
        assert tok in ct.vocab, f"{tok!r} not in the compiled k-mer vocab"


# ---------------------------------------------------------------------------
# Reusability — the strategy is generic over ANY fixed alphabet (RNA/protein).
# ---------------------------------------------------------------------------


def test_kmer_is_generic_over_an_arbitrary_alphabet():
    """The same ``kmer`` strategy tokenizes ANY fixed-alphabet sequence (RNA over
    ``{A,C,G,U}``) — a new domain is a new alphabet param, not new harness code."""
    rna = "ACGU"
    k = 2
    fm = _kmer_field_map(k=k, alphabet=rna)
    ct = compile_spec(spec_from_field_map(fm))
    _assert_kmer_vocab(ct, rna, k)

    df = pd.DataFrame({"seq_id": ["r0"], "sequence": ["AUGCGU"]})
    got = _materialized_kmer_tokens(ct, df)
    assert got == ["KMER_AU", "KMER_UG", "KMER_GC", "KMER_CG", "KMER_GU"]


def test_kmer_strategy_vocab_via_engine_strategies_directly():
    """The new Strategy exposes the SAME config-only vocab through the engine's
    ``strategies.build_vocab`` surface the compiler uses — proving it is a first-class
    Strategy (the compiler/contracts never special-case it)."""
    import loom.engine.strategies as strat

    k = 2
    spec = spec_from_field_map(_kmer_field_map(k=k))
    step = spec.steps[0]
    vocab = strat.build_vocab(step.strategy)
    codons = {f"KMER_{m}" for m in _expected_kmer_vocab(DNA, k)}
    # build_vocab is the SAME ordered local block the compiler lays out: every codon
    # present, at most one default surplus, and count() agrees with len(build_vocab).
    assert codons <= set(vocab)
    assert len(set(vocab) - codons) <= 1
    assert strat.count(step.strategy) == len(vocab)
    assert len(vocab) in (4 ** k, 4 ** k + 1)


# ---------------------------------------------------------------------------
# The HARD INVARIANT — k-mer flows through the EXISTING compile path only.
# ---------------------------------------------------------------------------


def test_kmer_does_not_perturb_the_dual_driver_preset():
    """Adding k-mer is additive: the financial preset still compiles to vocab 6251
    (HARD INVARIANT #1) — the k-mer strategy lives alongside, never inside, the
    preset path."""
    from loom.engine import financial_spec

    ct = compile_spec(financial_spec())
    assert ct.vocab_size == 6251
    assert ct.report.passed
