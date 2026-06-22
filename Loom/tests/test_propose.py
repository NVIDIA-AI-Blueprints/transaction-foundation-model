"""The ``propose`` field→strategy classifier — bring-your-own-schema authoring.

A stranger's bank transaction schema (``account_id, txn_amount, mcc, channel,
dr_cr, balance, txn_ts, [is_fraud]``) must propose a sane, reviewable tokenizer
field-map that EXCLUDES the entity (T2) and the target (leakage), bins continuous
amounts, maps low-card categoricals, hashes high-card ones, expands the timestamp
into calendar + inter-event TimeDelta tokens, and drops near-constant / near-unique
columns — then compiles through the LOCKED ``compile_spec`` + C1/C2/C3 unchanged.

These tests pin the classifier rules (the build brief's Ground 1), the
hand-counted estimates, and the field-map ⇄ TokenizerSpec round-trip + the
compiler agreement (the SpecDraft's hand-count == the compiled vocab/grammar).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from loom.eda import leakage_scan
# The SINGLE production field-map compiler (the one `tokenize --spec` uses). These
# tests exercise it directly so a passing assertion reflects what tokenize produces
# — there is no second, divergent compiler in engine/propose.py any more.
from loom.engine import compile_spec, spec_from_field_map
from loom.engine.propose import (
    CONT_BINS_DEFAULT,
    TIMEDELTA_BINS,
    N_SPECIALS,
    SpecDraft,
    propose_spec,
)
from loom.verbs.ingest import _sniff_schema


def fieldmap_to_yaml(fieldmap: dict) -> str:
    """Editable-YAML serializer (the production `propose` verb path: PyYAML)."""
    return "# loom-fieldmap/1\n" + yaml.safe_dump(fieldmap, sort_keys=False)


def fieldmap_from_yaml(text: str) -> dict:
    """Parse a YAML field-map back to a dict (round-trips `fieldmap_to_yaml`)."""
    return yaml.safe_load(text)


# ---------------------------------------------------------------------------
# Fixtures — the synthetic BANK schema (the build brief's worked example).
# ---------------------------------------------------------------------------


def _bank_df(n: int = 50_000, *, with_target: bool, seed: int = 0) -> pd.DataFrame:
    """A realistic bank-transaction frame: ~2000 accounts, repeated categoricals,
    heavy-tailed amounts, a real datetime, plus a near-constant column and a
    near-unique id-shaped NON-entity column to exercise the exclusion gates."""
    rng = np.random.default_rng(seed)
    cols: dict = {
        "account_id": rng.integers(0, 2000, n).astype(str),  # entity (~2000 groups)
        "txn_amount": rng.lognormal(3.0, 1.0, n),            # continuous float
        "mcc": rng.integers(0, 30, n),                        # bounded int small range
        "channel": rng.choice(["POS", "ONLINE", "ATM"], n),   # low-card categorical
        "dr_cr": rng.choice(["DR", "CR"], n),                 # 2-value categorical
        "balance": rng.lognormal(6.0, 1.0, n),               # continuous float
        "txn_ts": pd.to_datetime(
            rng.integers(1_700_000_000, 1_710_000_000, n), unit="s"
        ),                                                    # datetime
        "currency": ["USD"] * n,                              # near-constant
        "session_uuid": [f"sess-{i}" for i in range(n)],      # near-unique id-shaped
    }
    if with_target:
        cols["is_fraud"] = rng.integers(0, 2, n)              # target
    return pd.DataFrame(cols)


def _propose_from(df: pd.DataFrame, *, target=None) -> SpecDraft:
    schema = _sniff_schema(df)
    eda = leakage_scan(df, target=target)
    # Pass the rows frame so low-card categoricals carry their REAL observed value
    # list — exactly what the `propose` verb does (it reads the IngestDataset rows
    # payload). Without it, a `mapping` field would carry only n_values (a count)
    # and the production compiler would collapse every value to the single default.
    return propose_spec(
        schema=schema,
        eda_flags=eda,
        entity="account_id",
        event="txn",
        target=target,
        context_len=4096,
        rows=df,
    )


# ---------------------------------------------------------------------------
# A. Exclusion gate — entity (T2) + target (leakage) + the earns-a-token gate.
# ---------------------------------------------------------------------------


def test_entity_is_excluded_as_T2():
    draft = _propose_from(_bank_df(with_target=False))
    excl = {e.name: e for e in draft.excluded}
    assert "account_id" in excl
    assert excl["account_id"].reason == "entity"
    assert "T2" in excl["account_id"].rationale
    # never appears as an included field / step.
    assert "account_id" not in {f.source for f in draft.fields}
    # the field-map carries it as the grouping entity, not a feature.
    assert draft.fieldmap["entity"] == "account_id"
    assert all(f["source"] != "account_id" for f in draft.fieldmap["fields"])


def test_target_is_excluded_as_leakage():
    draft = _propose_from(_bank_df(with_target=True), target="is_fraud")
    excl = {e.name: e for e in draft.excluded}
    assert "is_fraud" in excl
    assert excl["is_fraud"].reason == "target"
    assert "is_fraud" not in {f.source for f in draft.fields}


def test_near_constant_column_is_flagged_and_dropped():
    draft = _propose_from(_bank_df(with_target=False))
    excl = {e.name: e for e in draft.excluded}
    assert "currency" in excl
    assert excl["currency"].reason == "near-constant"
    assert "currency" not in {f.source for f in draft.fields}


def test_near_unique_id_shaped_nonentity_is_flagged_and_dropped():
    """A near-unique id-shaped NON-entity column (session_uuid) is excluded by
    default (A.3) and surfaced as a REVIEW warning the human can override."""
    # No target → the id-shaped near-unique column hits the A.3 near-unique gate
    # (with a target, random labels can make it spuriously 1:1 → A.2 leakage).
    draft = _propose_from(_bank_df(with_target=False))
    excl = {e.name: e for e in draft.excluded}
    assert "session_uuid" in excl
    assert excl["session_uuid"].reason == "near-unique"
    assert "session_uuid" not in {f.source for f in draft.fields}
    # surfaced as a review warning (HARD INVARIANT #4 — make the human reason).
    assert any("session_uuid" in w for w in draft.warnings)


def test_excluded_columns_embed_the_originating_eda_card():
    """HARD INVARIANT #4: every consumed EDA flag is surfaced in the proposal."""
    draft = _propose_from(_bank_df(with_target=False))
    excl = {e.name: e for e in draft.excluded}
    # the entity + the id-shaped near-unique column carry their EDA Diagnostic.
    assert excl["account_id"].eda is not None
    assert excl["account_id"].eda["data"]["kind"] == "identity_like"
    assert excl["session_uuid"].eda is not None
    assert excl["session_uuid"].eda["data"]["kind"] == "identity_like"


# ---------------------------------------------------------------------------
# B. Strategy decision table — each surviving column gets the right strategy.
# ---------------------------------------------------------------------------


def test_continuous_floats_get_log_bins():
    draft = _propose_from(_bank_df(with_target=False))
    by_src = {f.source: f for f in draft.fields}
    for col in ("txn_amount", "balance"):
        assert col in by_src, f"{col} should be tokenized, not dropped"
        assert by_src[col].strategy == "amount"
        assert by_src[col].token_count == CONT_BINS_DEFAULT
        assert "deterministic" in by_src[col].rationale  # C2-clean, not fitted


def test_bounded_small_int_gets_fixedvocab_0_based():
    draft = _propose_from(_bank_df(with_target=False))
    by_src = {f.source: f for f in draft.fields}
    assert by_src["mcc"].strategy == "fixedvocab"
    # 0-based shift (min=0) — the MONTH_12/CARD_0 min_val>0 collision fix.
    assert draft_field_param(draft, "mcc", "min") == 0


def test_low_card_categoricals_get_mapping_plus_default():
    draft = _propose_from(_bank_df(with_target=False))
    by_src = {f.source: f for f in draft.fields}
    # channel (3 distinct) → Mapping size 3 + 1 default = 4.
    assert by_src["channel"].strategy == "mapping"
    assert by_src["channel"].token_count == 3 + 1
    # dr_cr (2 distinct) → Mapping size 2 + 1 default = 3.
    assert by_src["dr_cr"].strategy == "mapping"
    assert by_src["dr_cr"].token_count == 2 + 1


def test_high_card_categorical_gets_hash():
    """A high-cardinality categorical is routed to HASH and KEPT — NOT dropped as
    "starved" (the sample-aware Fix 1: vocab safety for high-card fields comes from
    the cardinality→strategy routing, not from the occupancy gate nuking them)."""
    rng = np.random.default_rng(1)
    n = 50_000
    df = pd.DataFrame({
        "account_id": rng.integers(0, 2000, n).astype(str),
        "amount": rng.lognormal(3, 1, n),
        # ~3000 distinct merchants → high-card (>=500) → Hash, buckets clamped to
        # 256 (MIN); 50000/256 ~= 195 occ/tok, which clears the LOW sample-aware
        # floor (≤50), so the hash field is KEPT (high-card → hash, not starved).
        "merchant": rng.integers(0, 3000, n).astype(str),
        "ts": pd.to_datetime(rng.integers(1_700_000_000, 1_710_000_000, n), unit="s"),
    })
    draft = propose_spec(schema=_sniff_schema(df), eda_flags=leakage_scan(df),
                         entity="account_id", event="txn", context_len=4096)
    by_src = {f.source: f for f in draft.fields}
    excl = {e.name: e for e in draft.excluded}
    # high-card merchant → HASH, kept (NOT starved): the routing keeps the vocab safe.
    assert "merchant" not in excl, f"high-card must hash, not be starved: {excl.get('merchant')}"
    assert "merchant" in by_src and by_src["merchant"].strategy == "hash"

    # The same routing holds on a much larger corpus — still hash, still kept.
    big = pd.DataFrame({
        "account_id": rng.integers(0, 2000, 5_000_000).astype(str),
        "amount": rng.lognormal(3, 1, 5_000_000),
        "merchant": rng.integers(0, 3000, 5_000_000).astype(str),
    })
    draft2 = propose_spec(schema=_sniff_schema(big), eda_flags=leakage_scan(big),
                          entity="account_id", event="txn", context_len=4096)
    by_src2 = {f.source: f for f in draft2.fields}
    assert "merchant" in by_src2 and by_src2["merchant"].strategy == "hash"


def test_low_card_categorical_on_a_small_sample_is_kept_not_starved():
    """Fix 1 acceptance: a healthy low-cardinality categorical on a ~2,500-row
    SAMPLE is INCLUDED (mapping), NOT dropped as "starved". This is the exact flaw
    the control experiment exposed — the old absolute 1K floor assumed the full
    cloud corpus and nuked a 4-value field at ~640 occ/value and a 3-value field at
    ~855 on a 2,565-row sample. Tokenizer design is a LOCAL laptop-SAMPLE activity.

    Also pins the OLD inconsistency is gone: the gate is UNIFORM (no fixedvocab
    exemption), yet the low sample-aware floor keeps every signal-bearing field."""
    n = 2_565  # the control experiment's sample size
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "account_id": rng.integers(0, 300, n).astype(str),     # entity
        "quad": rng.choice(["NW", "NE", "SW", "SE"], n),        # 4-value (~640 occ/val)
        "tri": rng.choice(["LOW", "MID", "HIGH"], n),           # 3-value (~855 occ/val)
        "amount": rng.lognormal(3, 1, n),                       # continuous float
    })
    draft = propose_spec(schema=_sniff_schema(df), eda_flags=leakage_scan(df),
                         entity="account_id", event="txn", context_len=4096)
    by_src = {f.source: f for f in draft.fields}
    excl = {e.name: e for e in draft.excluded}
    # The signal-bearing low-card categoricals are KEPT as mappings (not starved).
    assert "quad" not in excl, f"4-value field must be kept on a 2.5K sample: {excl.get('quad')}"
    assert "tri" not in excl, f"3-value field must be kept on a 2.5K sample: {excl.get('tri')}"
    assert by_src["quad"].strategy == "mapping" and by_src["quad"].token_count == 4 + 1
    assert by_src["tri"].strategy == "mapping" and by_src["tri"].token_count == 3 + 1
    # ...and the continuous float still bins (no signal field dropped on a sample).
    assert by_src["amount"].strategy == "amount"
    assert not any(e.reason == "starved" for e in draft.excluded)


def test_datetime_expands_to_calendar_plus_timedelta():
    draft = _propose_from(_bank_df(with_target=False))
    by_name = {f.name: f for f in draft.fields}
    ts_steps = {f.name: f for f in draft.fields if f.source == "txn_ts"}
    # 3 calendar parts + 1 inter-event gap = 4 steps off the one timestamp.
    assert len(ts_steps) == 4
    assert {f.strategy for f in ts_steps.values()} == {"calendar", "timedelta"}
    counts = {f.params.get("part", "gap"): f.token_count for f in ts_steps.values()}
    assert counts["hour"] == 24
    assert counts["dow"] == 7
    assert counts["month"] == 12
    # the TimeDelta gap (32 log-bins).
    gap = [f for f in ts_steps.values() if f.strategy == "timedelta"][0]
    assert gap.token_count == TIMEDELTA_BINS


def test_free_text_high_cardinality_is_dropped():
    """A free-text column (> HIGH_CARD_MAX distinct strings, but WITH repetition so
    it is not near-unique) is dropped by default as free-text."""
    n = 1_000_000
    rng = np.random.default_rng(2)
    # ~150K distinct memos over 1M rows: > HIGH_CARD_MAX (100K) AND repeated
    # (unique_ratio ~0.15 < 0.98) so it clears the near-unique gate and reaches the
    # free-text cardinality rule.
    memo_pool = [f"memo phrase {i}" for i in range(150_000)]
    df = pd.DataFrame({
        "account_id": rng.integers(0, 5000, n).astype(str),
        "amount": rng.lognormal(3, 1, n),
        "memo": rng.choice(memo_pool, n),
    })
    draft = propose_spec(schema=_sniff_schema(df), eda_flags=leakage_scan(df),
                         entity="account_id", event="txn", context_len=4096)
    excl = {e.name: e for e in draft.excluded}
    assert "memo" in excl and excl["memo"].reason == "free-text"
    assert "memo" not in {f.source for f in draft.fields}


# ---------------------------------------------------------------------------
# E. Derived numbers — the hand-counted estimates match the brief + the compiler.
# ---------------------------------------------------------------------------


def test_estimates_match_the_worked_example():
    """The brief's worked bank example: tokens_per_event=9, chunk_size=409."""
    draft = _propose_from(_bank_df(with_target=False))
    # amount(1)+mcc(1)+channel(1)+dr_cr(1)+balance(1)+[hour,dow,month](3)+gap(1) = 9
    assert draft.tokens_per_event == 9
    assert draft.chunk_size == 4096 // (9 + 1)  # == 409
    # vocab = 5 specials + hand-summed per-field counts.
    expected_vocab = N_SPECIALS + (8 + 30 + 4 + 3 + 8 + 24 + 7 + 12 + 32)
    assert draft.vocab_size == expected_vocab


def test_handcount_matches_the_compiled_vocab_and_grammar():
    """The SpecDraft's hand-counted vocab_size / tokens_per_event / chunk_size are
    EXACTLY what the LOCKED compiler derives (C1/C3) from the compiled field-map."""
    draft = _propose_from(_bank_df(with_target=False))
    spec = spec_from_field_map(draft.fieldmap)
    ct = compile_spec(spec, context_len=4096)
    assert ct.vocab_size == draft.vocab_size
    assert ct.tokens_per_txn == draft.tokens_per_event
    assert ct.chunk_size == draft.chunk_size


# ---------------------------------------------------------------------------
# The compile path — the custom field-map runs through C1/C2/C3 unchanged.
# ---------------------------------------------------------------------------


def test_compiled_custom_spec_passes_C1_C2_C3():
    draft = _propose_from(_bank_df(with_target=False))
    spec = spec_from_field_map(draft.fieldmap)
    ct = compile_spec(spec, context_len=4096)
    assert ct.report.passed, [d.message for d in ct.report.diagnostics]
    assert ct.report.injective    # C1: blocks disjoint
    assert ct.report.dense        # C1: ids dense 0..vocab_size-1
    assert not ct.report.has_fitted_artifact  # C2: threshold bins, no fitted state


def test_custom_preset_string_is_not_a_builtin_preset():
    """spec_from_field_map uses preset='custom' so the financial/chain dual-driver
    byte-identity (HARD INVARIANT #1) is untouched."""
    draft = _propose_from(_bank_df(with_target=False))
    spec = spec_from_field_map(draft.fieldmap)
    assert spec.preset == "custom"
    assert spec.preset not in ("financial", "chain")
    assert spec.entity == "account_id"


def test_spec_from_field_map_refuses_to_tokenize_the_entity():
    """The production compiler REFUSES (raises) a hand-edited field-map that lists
    the entity as a field — the entity can never be smuggled into the vocab as a
    feature (HARD INVARIANT #4: identity comes from history, not an ID embedding)."""
    import pytest

    fieldmap = {
        "version": "loom-fieldmap/1",
        "entity": "account_id",
        "context_len": 4096,
        "fields": [
            {"name": "acct", "source": "account_id", "strategy": "hash", "buckets": 4096},
            {"name": "amt", "source": "amount", "strategy": "amount", "bins": 8},
        ],
    }
    with pytest.raises(ValueError, match="entity"):
        spec_from_field_map(fieldmap)


def test_spec_from_field_map_refuses_to_tokenize_the_target():
    """Likewise the declared target/label column is refused as a field (leakage)."""
    import pytest

    fieldmap = {
        "version": "loom-fieldmap/1",
        "entity": "account_id",
        "target": "is_fraud",
        "context_len": 4096,
        "fields": [
            {"name": "label", "source": "is_fraud", "strategy": "fixedvocab", "min": 0, "max": 1},
            {"name": "amt", "source": "amount", "strategy": "amount", "bins": 8},
        ],
    }
    with pytest.raises(ValueError, match="target"):
        spec_from_field_map(fieldmap)


# ---------------------------------------------------------------------------
# Serialization — the editable YAML round-trips and is the same artifact.
# ---------------------------------------------------------------------------


def test_fieldmap_yaml_roundtrip_is_stable():
    draft = _propose_from(_bank_df(with_target=False))
    y = fieldmap_to_yaml(draft.fieldmap)
    assert "loom-fieldmap/1" in y
    fm = fieldmap_from_yaml(y)
    assert fm == draft.fieldmap
    # a second pass is byte-identical (deterministic serialization).
    assert fieldmap_to_yaml(fm) == y


def test_edited_yaml_compiles_through_the_same_gate():
    """A human edits the YAML (e.g. raise amount bins to 16) and it recompiles."""
    draft = _propose_from(_bank_df(with_target=False))
    fm = fieldmap_from_yaml(fieldmap_to_yaml(draft.fieldmap))
    for f in fm["fields"]:
        if f["source"] == "txn_amount":
            f["bins"] = 16
    spec = spec_from_field_map(fm)
    ct = compile_spec(spec, context_len=4096)
    assert ct.report.passed
    # the edit added 8 ids (16 - 8) to the vocab.
    assert ct.vocab_size == draft.vocab_size + 8


def test_propose_is_deterministic():
    """Same inputs → identical draft (pure, no data fitting)."""
    df = _bank_df(with_target=False)
    a = _propose_from(df)
    b = _propose_from(df)
    assert a.to_dict() == b.to_dict()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def draft_field_param(draft: SpecDraft, source: str, key: str):
    for f in draft.fields:
        if f.source == source:
            return f.params.get(key)
    raise AssertionError(f"no field with source {source!r}")
