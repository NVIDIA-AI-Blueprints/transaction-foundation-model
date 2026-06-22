"""Conformance oracle: the financial (TabFormer) preset compiles to the exact
reference vocabulary — vocab_size 6251, dense, injective, specials at 0-4, and
the 12 field blocks at the documented sizes (build brief §"reference tokenizer").

This is the headline gate: if these pass, Loom's clean CPU compile reproduces the
GPU reference's vocab/grammar/injectivity (the only conformance axis that matters
— merchant-bucket identity is explicitly NOT part of conformance)."""

from __future__ import annotations

from loom.engine import SPECIAL_TOKENS

from .golden_helpers import compiled_financial

# The documented per-field block sizes, in field order (build brief, sums to 6246;
# + 5 specials == 6251). These are the C1 invariant the reference 12 fields imply.
EXPECTED_FIELD_SIZES = {
    "AMT": 7,
    "MERCH": 2000,
    "CAT": 14,
    "MCC": 110,
    "HOUR": 24,
    "DOW": 7,
    "MONTH": 12,
    "CARD": 10,
    "CHIP": 4,
    "ZIP3": 1000,
    "STATE": 58,
    "CUST": 3000,
}


def test_financial_vocab_size_is_6251():
    ct = compiled_financial()
    assert ct.vocab_size == 6251
    assert len(ct.vocab) == 6251
    assert ct.report.passed, ct.report.diagnostics


def test_ids_are_dense_and_injective():
    """C1: every id is unique (injective) and the id space is dense 0..N-1."""
    ct = compiled_financial()
    ids = list(ct.vocab.values())
    assert len(ids) == len(set(ids)), "duplicate ids — vocabulary is not injective"
    assert set(ids) == set(range(ct.vocab_size)), "id space is not dense 0..vocab_size-1"
    # id_to_token is the exact inverse.
    assert len(ct.id_to_token) == ct.vocab_size
    for tok, i in ct.vocab.items():
        assert ct.id_to_token[i] == tok
    assert ct.report.injective and ct.report.dense


def test_special_tokens_occupy_ids_0_through_4():
    ct = compiled_financial()
    assert SPECIAL_TOKENS == ("<pad>", "<bos>", "<eos>", "<sep>", "<unk>")
    for expected_id, tok in enumerate(SPECIAL_TOKENS):
        assert ct.vocab[tok] == expected_id, f"{tok} must be id {expected_id}"


def _block_sizes_by_prefix(ct) -> dict[str, int]:
    """Count vocab tokens by their field prefix (e.g. 'AMT', 'MERCH', 'ZIP3').

    Field tokens are ``PREFIX_...`` (or ``PREFIX:...`` for the mapping families per
    the brief's CAT_/MCC_ examples); we bucket on the leading prefix token so the
    count is independent of the exact suffix scheme."""
    specials = set(SPECIAL_TOKENS)
    sizes: dict[str, int] = {}
    for tok in ct.vocab:
        if tok in specials:
            continue
        # Prefix is the run of leading [A-Z0-9] before the first separator.
        prefix = ""
        for ch in tok:
            if ch.isalnum() and not ch.islower():
                prefix += ch
            else:
                break
        sizes[prefix] = sizes.get(prefix, 0) + 1
    return sizes


def test_field_blocks_have_documented_sizes():
    ct = compiled_financial()
    sizes = _block_sizes_by_prefix(ct)
    for prefix, expected in EXPECTED_FIELD_SIZES.items():
        assert sizes.get(prefix) == expected, (
            f"{prefix} block size {sizes.get(prefix)} != expected {expected}"
        )
    # The 12 field blocks + 5 specials account for every id.
    assert sum(EXPECTED_FIELD_SIZES.values()) + len(SPECIAL_TOKENS) == ct.vocab_size


def test_spec_step_counts_match_documented_sizes():
    """The same invariant straight off the spec (each FieldStep.count())."""
    from loom.engine import financial_spec
    from .golden_helpers import require_engine

    spec = require_engine(lambda: financial_spec())
    counts = {s.name.upper(): require_engine(lambda s=s: s.count()) for s in spec.steps}
    total = 0
    for expected in EXPECTED_FIELD_SIZES.values():
        total += expected
    # tokens_per_txn is the number of steps (12 for financial without time-delta).
    assert require_engine(lambda: spec.tokens_per_txn()) == 12
    assert sum(counts.values()) == total


def test_spot_check_field_token_strings():
    """Spot-check exact token strings the brief documents per field."""
    ct = compiled_financial()
    vocab = ct.vocab
    # AMT 0..6 (no pad).
    assert "AMT_0" in vocab and "AMT_6" in vocab and "AMT_7" not in vocab
    # MERCH hashed buckets 0..1999.
    assert "MERCH_0" in vocab and "MERCH_1999" in vocab and "MERCH_2000" not in vocab
    # HOUR padded width 2: HOUR_00..HOUR_23.
    assert "HOUR_00" in vocab and "HOUR_23" in vocab and "HOUR_24" not in vocab
    # MONTH padded width 2, 01..12 — the field that collided in the reference bug.
    assert "MONTH_01" in vocab and "MONTH_12" in vocab and "MONTH_00" not in vocab
    # CARD 0..9 (no pad).
    assert "CARD_0" in vocab and "CARD_9" in vocab and "CARD_10" not in vocab
    # ZIP3 padded width 3: ZIP3_000..ZIP3_999.
    assert "ZIP3_000" in vocab and "ZIP3_999" in vocab
    # CUST 0..2999 (no pad).
    assert "CUST_0" in vocab and "CUST_2999" in vocab and "CUST_3000" not in vocab


def test_month12_and_card0_are_distinct_ids():
    """The regression assertion for the reference bug: MONTH_12 and CARD_0 are
    distinct (in the reference they collided at id 2179)."""
    ct = compiled_financial()
    assert ct.vocab["MONTH_12"] != ct.vocab["CARD_0"]
