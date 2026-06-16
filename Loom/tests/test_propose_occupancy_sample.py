"""FIX 1 — the proposer's occupancy gate must be SAMPLE-AWARE + UNIFORM.

A control experiment exposed a real flaw: on a 2,565-row SAMPLE of a non-finance
schema the proposer dropped EVERY signal-bearing field as ``"starved"`` (a 4-value
categorical at ~640 occ/value, a 3-value at ~855) because of an ABSOLUTE big-data
occupancy floor (``MIN_OCC_PER_TOKEN = 1_000``), and kept only the timestamp
tokens — so the representation recovered NONE of the planted structure (ARI ≈ 0).
That contradicts Loom's principle that **tokenizer design is a LOCAL laptop-SAMPLE
activity** (the absolute floor silently assumes the full cloud corpus).

These tests pin the CORRECTED behavior (the build brief's FIX 1):

  * a healthy low-cardinality categorical on a ~2,500-row SAMPLE is KEPT (mapping),
    NOT dropped as "starved" — the gate is sample-aware, not an absolute 1K floor;
  * the gate is UNIFORM — no ``fixedvocab`` exemption; the SAME low floor decides
    every field type (the old code kept 24-value calendar tokens at ~107 occ while
    dropping a 4-value categorical at 640 — inconsistent);
  * the GENUINE junk drops are UNCHANGED + correct: near-constant (≤1 effective
    value), near-unique id-shaped, mostly-null/sparse, target (leakage), entity (T2);
  * a high-cardinality field routes to **Hash** (NOT dropped as "starved") — vocab
    safety comes from cardinality→strategy routing, not from nuking the field;
  * the proposer keeps SIGNAL on a small sample (the planted categoricals survive).

The acceptance criterion is the headline assertion of
:func:`test_four_value_categorical_on_2500_row_sample_is_kept_as_mapping`.

These tests drive ONLY the pure ``propose_spec`` authoring entrypoint + the LOCKED
``spec_from_field_map`` / ``compile_spec`` gate — no harness/contract surgery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loom.eda import leakage_scan
from loom.engine import compile_spec, spec_from_field_map
from loom.engine.propose import SpecDraft, propose_spec
from loom.verbs.ingest import _sniff_schema

# The control-experiment SAMPLE size that exposed the flaw. Deliberately FAR below
# the old absolute 1K-occ-per-token floor for every low-cardinality field so the
# OLD behavior would (wrongly) starve them and the NEW behavior must keep them.
SAMPLE_N = 2_565


# ---------------------------------------------------------------------------
# Fixtures — a non-finance SAMPLE with PLANTED signal-bearing categoricals.
# ---------------------------------------------------------------------------


def _sample_df(n: int = SAMPLE_N, *, seed: int = 0) -> pd.DataFrame:
    """A ~2,565-row SAMPLE of a non-finance schema with planted low-card structure.

    Mirrors the control experiment:
      * ``seq_id``       — the grouping entity (T2), ~250 groups;
      * ``state4``       — a 4-value categorical at ~n/4 ≈ 640 occ/value (SIGNAL);
      * ``phase3``       — a 3-value categorical at ~n/3 ≈ 855 occ/value (SIGNAL);
      * ``ts``           — a real datetime (calendar + inter-event gap survive);
      * ``region``       — a high-cardinality categorical (~thousands distinct) → Hash;
      * ``const_col``    — near-constant (1 value) → dropped (no information);
      * ``row_uuid``     — near-unique id-shaped non-entity → dropped/hashed;
      * ``mostly_null``  — present on <1% of rows → dropped (sparse).

    EVERY per-token occupancy here is well under the OLD 1K floor, so the OLD gate
    starved all of them; the NEW sample-aware gate must KEEP the healthy ones.
    """
    rng = np.random.default_rng(seed)
    cols: dict = {
        "seq_id": rng.integers(0, 250, n).astype(str),                 # entity (~250 groups)
        "state4": rng.choice(["S0", "S1", "S2", "S3"], n),             # 4-value categorical (signal)
        "phase3": rng.choice(["P", "Q", "R"], n),                       # 3-value categorical (signal)
        "ts": pd.to_datetime(
            rng.integers(1_700_000_000, 1_700_500_000, n), unit="s"
        ),                                                              # datetime
        # ~3,000 distinct over 2,565 rows but REPEATED enough to stay below the
        # near-unique ratio (we sample WITH replacement from a 3,000-value pool, so
        # only ~2,000 distinct actually appear → unique_ratio ~0.8 < 0.98) → high-card.
        "region": rng.integers(0, 3000, n).astype(str),               # high-card categorical → Hash
        "const_col": ["ONLY"] * n,                                     # near-constant → drop
        "row_uuid": [f"row-{i}" for i in range(n)],                    # near-unique id-shaped → drop
    }
    df = pd.DataFrame(cols)
    # mostly_null: present on ~0.5% of rows (well under the 99% coverage floor).
    null_mask = rng.random(n) > 0.005
    mostly = rng.choice(["a", "b"], n).astype(object)
    mostly[null_mask] = None
    df["mostly_null"] = mostly
    return df


def _propose(df: pd.DataFrame, *, target=None) -> SpecDraft:
    schema = _sniff_schema(df)
    eda = leakage_scan(df, target=target)
    return propose_spec(
        schema=schema,
        eda_flags=eda,
        entity="seq_id",
        event="ev",
        target=target,
        context_len=4096,
        rows=df,
    )


def _by_src(draft: SpecDraft) -> dict:
    return {f.source: f for f in draft.fields}


def _excl(draft: SpecDraft) -> dict:
    return {e.name: e for e in draft.excluded}


# ---------------------------------------------------------------------------
# The headline acceptance — a healthy low-card categorical on a SAMPLE is KEPT.
# ---------------------------------------------------------------------------


def test_four_value_categorical_on_2500_row_sample_is_kept_as_mapping():
    """ACCEPTANCE (FIX 1): a 4-value categorical with reasonable coverage on a
    ~2,500-row SAMPLE is INCLUDED as a ``mapping`` field — NOT dropped as "starved".

    At 2,565 rows / 4 values ≈ 640 occ/value, the OLD absolute 1K floor would have
    starved it; the NEW sample-aware floor must keep it (the unblocker)."""
    draft = _propose(_sample_df())
    by_src = _by_src(draft)
    excl = _excl(draft)

    assert "state4" in by_src, (
        "the 4-value categorical must be KEPT on a 2.5k-row sample, not starved; "
        f"it was excluded as {excl.get('state4').reason if 'state4' in excl else '??'}"
    )
    f = by_src["state4"]
    assert f.strategy == "mapping"
    # mapping = the 4 observed values + exactly 1 default.
    assert f.token_count == 4 + 1
    # explicitly NOT recorded as a starved exclusion.
    assert "state4" not in excl
    assert all(e.reason != "starved" or e.name != "state4" for e in draft.excluded)


def test_three_value_categorical_on_sample_is_also_kept():
    """The 3-value categorical (~855 occ/value) is likewise kept (uniform rule)."""
    draft = _propose(_sample_df())
    by_src = _by_src(draft)
    assert "phase3" in by_src, "the 3-value categorical must be kept on a sample"
    assert by_src["phase3"].strategy == "mapping"
    assert by_src["phase3"].token_count == 3 + 1


def test_no_signal_bearing_field_is_dropped_as_starved_on_a_sample():
    """The regression for the exposed flaw: NO healthy low-card categorical is
    excluded with reason ``"starved"`` on a small sample (the bug nuked all of them)."""
    draft = _propose(_sample_df())
    starved = {e.name for e in draft.excluded if e.reason == "starved"}
    assert "state4" not in starved and "phase3" not in starved, (
        f"signal-bearing categoricals were starved on a sample: {starved}"
    )


def test_sample_keeps_signal_not_only_the_timestamp_tokens():
    """The flaw left ONLY the timestamp/calendar tokens (ARI ≈ 0). After the fix the
    proposed representation must carry the planted categorical signal too — i.e. the
    included sources are more than just the timestamp-derived steps."""
    draft = _propose(_sample_df())
    included_sources = {f.source for f in draft.fields}
    assert {"state4", "phase3"} <= included_sources, (
        "the sample-aware proposer must keep the planted categorical signal, not "
        f"only the timestamp tokens; included sources = {sorted(included_sources)}"
    )
    # the timestamp still expands (calendar + gap) — those were never the problem.
    assert "ts" in included_sources


# ---------------------------------------------------------------------------
# The gate is UNIFORM — no field type is special (no fixedvocab exemption).
# ---------------------------------------------------------------------------


def test_gate_is_uniform_low_card_categorical_treated_like_a_bounded_int():
    """UNIFORMITY (FIX 1): the OLD gate exempted ``fixedvocab`` (so 24-value calendar
    tokens at ~107 occ survived while a 4-value categorical at 640 was dropped) —
    inconsistent. Under the uniform sample-aware floor, BOTH a small bounded int and
    a small categorical survive on the same sample (neither is special-cased)."""
    df = _sample_df()
    # add a bounded small-range int (10 values → ~256 occ/value on the sample).
    rng = np.random.default_rng(7)
    df = df.copy()
    df["small_int"] = rng.integers(0, 10, len(df))
    draft = _propose(df)
    by_src = _by_src(draft)
    # the bounded int survives (fixedvocab) AND the 4-value categorical survives
    # (mapping) on the SAME sample — the floor is applied uniformly.
    assert "small_int" in by_src and by_src["small_int"].strategy == "fixedvocab"
    assert "state4" in by_src and by_src["state4"].strategy == "mapping"


# ---------------------------------------------------------------------------
# High-cardinality routes to Hash (NOT starved) — vocab safety via routing.
# ---------------------------------------------------------------------------


def test_high_card_field_routes_to_hash_not_starved_on_a_sample():
    """A high-cardinality (~thousands distinct) categorical must route to **Hash**,
    NOT be dropped as "starved". Vocab safety comes from the cardinality→strategy
    routing (high-card → Hash), never from nuking the field on a small sample."""
    draft = _propose(_sample_df())
    by_src = _by_src(draft)
    excl = _excl(draft)
    assert "region" in by_src, (
        "a high-card categorical must be hashed, not starved; "
        f"region was excluded as {excl['region'].reason if 'region' in excl else '??'}"
    )
    assert by_src["region"].strategy == "hash"
    assert "region" not in excl


def test_bank_merchant_high_card_goes_to_hash_on_a_sample():
    """The brief's bank example: ``merchant_name`` (high-card) → hash on a SAMPLE,
    NOT dropped as starved (the explicit acceptance call-out)."""
    rng = np.random.default_rng(3)
    n = SAMPLE_N
    df = pd.DataFrame(
        {
            "account_id": rng.integers(0, 250, n).astype(str),
            "amount": rng.lognormal(3.0, 1.0, n),
            "merchant_name": rng.integers(0, 4000, n).astype(str),  # high-card
            "channel": rng.choice(["POS", "ONLINE", "ATM"], n),     # low-card (kept)
        }
    )
    draft = propose_spec(
        schema=_sniff_schema(df),
        eda_flags=leakage_scan(df),
        entity="account_id",
        event="txn",
        context_len=4096,
        rows=df,
    )
    by_src = _by_src(draft)
    excl = _excl(draft)
    assert "merchant_name" in by_src, "merchant_name must hash, not starve"
    assert by_src["merchant_name"].strategy == "hash"
    assert "merchant_name" not in excl
    # and the low-card channel is kept too (sample-aware).
    assert "channel" in by_src and by_src["channel"].strategy == "mapping"


# ---------------------------------------------------------------------------
# The GENUINE junk drops are UNCHANGED + correct (these exclusions must STAY).
# ---------------------------------------------------------------------------


def test_near_constant_column_is_still_excluded():
    """A near-constant column (≤1 effective value) is still EXCLUDED (no information)
    — the genuine drop must NOT regress under the sample-aware floor."""
    draft = _propose(_sample_df())
    excl = _excl(draft)
    assert "const_col" in excl and excl["const_col"].reason == "near-constant"
    assert "const_col" not in {f.source for f in draft.fields}


def test_near_unique_id_shaped_nonentity_is_still_excluded_or_hashed():
    """A near-unique id-shaped NON-entity column is still EXCLUDED by default (drop or
    hash on explicit opt-in) — never silently kept as a per-row feature (T2 spirit)."""
    draft = _propose(_sample_df())
    excl = _excl(draft)
    by_src = _by_src(draft)
    # never minted as a per-row mapping feature.
    if "row_uuid" in by_src:
        # the only acceptable INCLUSION is an explicit hash (cardinality routing).
        assert by_src["row_uuid"].strategy == "hash"
    else:
        assert "row_uuid" in excl and excl["row_uuid"].reason == "near-unique"
        assert any("row_uuid" in w for w in draft.warnings)


def test_mostly_null_sparse_column_is_still_excluded():
    """A mostly-null column (present on <1% of rows) is still EXCLUDED as sparse —
    the coverage floor is independent of the occupancy fix and must STAY."""
    draft = _propose(_sample_df())
    excl = _excl(draft)
    assert "mostly_null" in excl and excl["mostly_null"].reason == "sparse"
    assert "mostly_null" not in {f.source for f in draft.fields}


def test_entity_and_target_are_still_excluded():
    """T2 (entity) + leakage (target) exclusions are untouched by the occupancy fix."""
    df = _sample_df()
    rng = np.random.default_rng(11)
    df = df.copy()
    df["label"] = rng.integers(0, 2, len(df))
    draft = _propose(df, target="label")
    excl = _excl(draft)
    assert "seq_id" in excl and excl["seq_id"].reason == "entity"
    assert "label" in excl and excl["label"].reason == "target"
    assert "seq_id" not in {f.source for f in draft.fields}
    assert "label" not in {f.source for f in draft.fields}


# ---------------------------------------------------------------------------
# The kept SAMPLE field-map still compiles through the LOCKED gate (C1/C2/C3).
# ---------------------------------------------------------------------------


def test_sample_aware_proposal_compiles_through_the_locked_gate():
    """The richer (signal-bearing) field-map the sample-aware proposer emits still
    compiles through the LOCKED ``spec_from_field_map`` + ``compile_spec`` (C1/C2/C3),
    and the hand-count matches the compiled vocab/grammar — the fix changes WHICH
    fields survive, never the contract gate."""
    draft = _propose(_sample_df())
    spec = spec_from_field_map(draft.fieldmap)
    ct = compile_spec(spec, context_len=4096)
    assert ct.report.passed, [d.message for d in ct.report.diagnostics]
    assert ct.report.injective and ct.report.dense
    assert not ct.report.has_fitted_artifact
    assert ct.vocab_size == draft.vocab_size
    assert ct.tokens_per_txn == draft.tokens_per_event
    assert ct.chunk_size == draft.chunk_size


def test_sample_aware_proposal_is_deterministic():
    """Same SAMPLE inputs → identical draft (pure, no fitting)."""
    df = _sample_df()
    assert _propose(df).to_dict() == _propose(df).to_dict()
