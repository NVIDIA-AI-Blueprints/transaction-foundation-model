"""Discriminating tests for the constant-memory streaming vocab learner.

The entire correctness argument for ``--streaming on`` is *parity*: a bounded,
one-pass walk over EVERY row must produce the SAME schema / EDA diagnostics /
``SpecDraft`` as the in-RAM path that materializes the whole frame — while never
retaining an unbounded distinct set. These tests pin that argument (the T1–T5
plan in the build spec):

  * **T1 equivalence** — ``stream_stats`` → ``propose_spec(col_stats=…, rows=None)``
    is byte-identical to ``_sniff_schema`` + ``leakage_scan`` →
    ``propose_spec(rows=full_df)``; ``to_schema()`` matches ``_sniff_schema`` per
    field (dtype / null_frac 6dp / n_unique). This pins the dtype-promotion lattice
    (the headline risk) against a real ``read_csv`` inference.
  * **T2 sampling-misses** — a 20 % HEAD sample's ``merchant_state`` value set is a
    STRICT SUBSET of the full set (the tail-clustered rare states collapse to UNK),
    which is *why* streaming is required.
  * **T3 bounded / cap** — a > ``LOW_CARD_MAX`` column flips ``hash_bound`` / frees
    its value LIST / routes to Hash exactly like in-RAM, and the live distinct set
    is NEVER retained above the cap.
  * **T4 back-compat byte-identity** — a small input on the in-RAM path (col_stats
    None) is unchanged.
  * **T5 leakage parity** — ``to_eda_diagnostics(target)`` emits the SAME
    identity_like / target_correlated / target_determines cards as
    ``leakage_scan(full_df, target)`` — INCLUDING for a high-cardinality near-unique
    column (the case the LOW_CARD ``n_unique`` sentinel used to silently drop).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from loom.eda import leakage_scan
from loom.engine.propose import (
    HIGH_CARD_MAX,
    LOW_CARD_MAX,
    propose_spec,
)
from loom.engine.streaming import CHUNK_ROWS, ColStat, stream_stats
from loom.verbs.ingest import _sniff_schema


# ---------------------------------------------------------------------------
# Fixture: a deterministic MEDIUM csv on disk with a planted low-card
# ``merchant_state`` (224 distinct) whose 10 rarest states are clustered in the
# TAIL, so a HEAD sample misses them. Kept CI-fast (the parity / cap / lattice
# logic is scale-invariant); a couple of full chunks still exercise the chunked
# reader's dtype=str path against a real read_csv inference.
# ---------------------------------------------------------------------------

_N_ROWS = 60_000
_N_COMMON_STATES = 214
_RARE_STATES = [f"RARE_{i:02d}" for i in range(10)]  # 10 tail-clustered rares → 224 total


def _build_frame() -> pd.DataFrame:
    rng = np.random.RandomState(1234)
    n = _N_ROWS
    common = [f"ST{i:03d}" for i in range(_N_COMMON_STATES)]
    # body: only the common states (so a HEAD sample sees ~214 distinct)
    state = list(rng.choice(common, size=n))
    # plant each rare state exactly once, clustered in the TAIL (last 200 rows)
    for k, rare in enumerate(_RARE_STATES):
        state[n - 1 - k] = rare

    df = pd.DataFrame(
        {
            # low-card categorical → Mapping with a real value list (the headline col)
            "merchant_state": state,
            # bounded small int → FixedVocab (dtype int64, no nulls)
            "mcc": rng.randint(0, 30, size=n),
            # continuous float → amount bins; min/max enrichment on streaming
            "amount": np.round(rng.uniform(0.5, 5000.0, size=n), 2),
            # an int column with a null → must promote int→float64 on BOTH paths
            "opt_int": rng.randint(0, 5, size=n).astype("float64"),
            # high-card categorical → Hash (over LOW_CARD_MAX distinct)
            "merchant_name": [f"M{ i % (LOW_CARD_MAX + 120) }" for i in range(n)],
            # plain label / target
            "label": rng.randint(0, 2, size=n),
        }
    )
    # inject the null into opt_int so it promotes to float64 (matches read_csv)
    df.loc[0, "opt_int"] = np.nan
    return df


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


def _stream_over_file(path: Path, column_order: list[str], target=None):
    """Drive ``stream_stats`` over the file EXACTLY like the ingest verb does:
    chunked ``read_csv`` with ``dtype=str, keep_default_na=True, na_values=['']``."""

    def _chunks():
        reader = pd.read_csv(
            path,
            sep=",",
            chunksize=4000,  # several chunks so the parallel-merge moments are exercised
            dtype=str,
            keep_default_na=True,
            na_values=[""],
        )
        for chunk in reader:
            yield chunk

    return stream_stats(_chunks(), column_order=column_order, target=target)


@pytest.fixture(scope="module")
def medium_csv(tmp_path_factory) -> Path:
    df = _build_frame()
    path = tmp_path_factory.mktemp("stream") / "corpus.csv"
    return _write_csv(df, path)


@pytest.fixture(scope="module")
def full_df(medium_csv) -> pd.DataFrame:
    # The in-RAM path reads the file with default read_csv (dtype inference).
    return pd.read_csv(medium_csv)


# ---------------------------------------------------------------------------
# T1 — equivalence (the core parity claim).
# ---------------------------------------------------------------------------


def test_T1_schema_matches_sniff(medium_csv, full_df):
    column_order = list(pd.read_csv(medium_csv, nrows=0).columns)
    stats = _stream_over_file(medium_csv, column_order)
    streamed = stats.to_schema()
    inram = _sniff_schema(full_df)

    assert streamed["n_rows"] == inram["n_rows"]
    assert streamed["n_cols"] == inram["n_cols"]
    for col in column_order:
        s = streamed["columns"][col]
        r = inram["columns"][col]
        # dtype-promotion lattice must byte-match read_csv's inference (the risk).
        assert s["dtype"] == r["dtype"], f"{col}: dtype {s['dtype']} != {r['dtype']}"
        assert s["null_frac"] == r["null_frac"], f"{col}: null_frac mismatch"
        # n_unique must be the TRUE count (not a LOW_CARD sentinel) — for the
        # high-card column too, since the proposer's near-unique gate reads it.
        assert s["n_unique"] == r["n_unique"], (
            f"{col}: n_unique {s['n_unique']} != {r['n_unique']}"
        )


def test_T1_specdraft_byte_identical(medium_csv, full_df):
    column_order = list(pd.read_csv(medium_csv, nrows=0).columns)
    stats = _stream_over_file(medium_csv, column_order, target="label")

    # streaming path: schema + eda from the bundle, rows=None, col_stats=…
    stream_schema = stats.to_schema()
    stream_eda = stats.to_eda_diagnostics(target="label")
    draft_stream = propose_spec(
        schema=stream_schema,
        eda_flags=stream_eda,
        target="label",
        rows=None,
        col_stats=stats.columns,
    )

    # in-RAM path: _sniff_schema + leakage_scan over the full frame, rows=full_df.
    inram_schema = _sniff_schema(full_df)
    inram_eda = leakage_scan(full_df, target="label")
    draft_inram = propose_spec(
        schema=inram_schema,
        eda_flags=inram_eda,
        target="label",
        rows=full_df,
        col_stats=None,
    )

    d_s = draft_stream.to_dict()
    d_r = draft_inram.to_dict()
    # The streaming schema carries an EXTRA min/max on numeric cols (intentional
    # enrichment); that lives in `schema`, not the SpecDraft — the draft itself
    # (fieldmap, fields, excluded, vocab_size, …) must be byte-identical.
    assert d_s == d_r, "SpecDraft diverged between streaming and in-RAM paths"

    # and the headline column carries ALL 224 observed values + the UNK default.
    ms = next(f for f in draft_stream.fields if f.source == "merchant_state")
    assert ms.strategy == "mapping"
    vals = ms.params["values"]
    assert len(vals) == 224, f"expected 224 merchant_state values, got {len(vals)}"
    assert set(_RARE_STATES).issubset(set(vals)), "tail rares missing from streamed vocab"


# ---------------------------------------------------------------------------
# T2 — sampling misses (the motivation): a 20% HEAD sample is a STRICT SUBSET.
# ---------------------------------------------------------------------------


def test_T2_head_sample_misses_tail_states(medium_csv, full_df):
    head = full_df.head(int(len(full_df) * 0.20))
    sample_states = set(head["merchant_state"].dropna().astype(str).unique())
    full_states = set(full_df["merchant_state"].dropna().astype(str).unique())

    assert sample_states < full_states, "sample should be a STRICT subset"
    missed = full_states - sample_states
    # the tail-clustered rares are exactly what a head sample drops
    assert set(_RARE_STATES).issubset(missed)
    assert len(full_states) == 224
    assert len(sample_states) <= 214 + len(_RARE_STATES)  # rares absent from the head


# ---------------------------------------------------------------------------
# T3 — bounded / cap: high-card column → hash_bound, values freed, routes to Hash;
# the live distinct set is NEVER retained above LOW_CARD_MAX.
# ---------------------------------------------------------------------------


def test_T3_high_card_caps_and_routes_to_hash(medium_csv, full_df, monkeypatch):
    column_order = list(pd.read_csv(medium_csv, nrows=0).columns)

    # instrument the collector: assert it never retains > LOW_CARD_MAX strings.
    max_live = {"merchant_name": 0}
    orig = ColStat._observe_distinct

    def spy(self, vals):
        if self.name in max_live and self._distinct is not None:
            max_live[self.name] = max(max_live[self.name], len(self._distinct))
        orig(self, vals)
        if self.name in max_live and self._distinct is not None:
            max_live[self.name] = max(max_live[self.name], len(self._distinct))

    monkeypatch.setattr(ColStat, "_observe_distinct", spy)

    stats = _stream_over_file(medium_csv, column_order)
    cs = stats.columns["merchant_name"]

    assert cs.hash_bound is True
    assert cs.values is None  # value LIST freed at the cap
    # but the COUNT is the TRUE distinct (LOW_CARD_MAX + 120), not a sentinel.
    true_distinct = full_df["merchant_name"].dropna().nunique()
    assert cs.n_unique == true_distinct
    assert cs.n_unique >= LOW_CARD_MAX
    # bounded memory: the live set never exceeded the cap.
    assert max_live["merchant_name"] < LOW_CARD_MAX, (
        f"distinct set grew to {max_live['merchant_name']} >= {LOW_CARD_MAX}"
    )

    # routes to Hash identically to in-RAM (values None → Hash strategy).
    inram_schema = _sniff_schema(full_df)
    inram_eda = leakage_scan(full_df, target="label")
    draft = propose_spec(
        schema=stats.to_schema(),
        eda_flags=stats.to_eda_diagnostics(target="label"),
        target="label",
        rows=None,
        col_stats=stats.columns,
    )
    draft_inram = propose_spec(
        schema=inram_schema, eda_flags=inram_eda, target="label", rows=full_df
    )
    mn = next((f for f in draft.fields if f.source == "merchant_name"), None)
    mn_inram = next((f for f in draft_inram.fields if f.source == "merchant_name"), None)
    assert mn is not None and mn.strategy == "hash"
    assert mn_inram is not None and mn_inram.strategy == "hash"
    assert mn.to_dict() == mn_inram.to_dict()


# ---------------------------------------------------------------------------
# T4 — back-compat byte-identity: a SMALL input with col_stats None is unchanged.
# ---------------------------------------------------------------------------


def test_T4_back_compat_col_stats_none(tabformer_df):
    schema = _sniff_schema(tabformer_df)
    eda = leakage_scan(tabformer_df, target=None)

    # the canonical (current-main) call: rows frame, col_stats omitted entirely.
    draft_default = propose_spec(schema=schema, eda_flags=eda, rows=tabformer_df)
    # an explicit col_stats=None must reproduce the default branch EXACTLY.
    draft_explicit_none = propose_spec(
        schema=schema, eda_flags=eda, rows=tabformer_df, col_stats=None
    )
    assert draft_default.to_dict() == draft_explicit_none.to_dict()

    # and passing col_stats=None changes nothing about the value enumeration:
    state_field = next(
        (f for f in draft_default.fields if f.source == "state"), None
    )
    if state_field is not None and state_field.strategy == "mapping":
        assert "values" in state_field.params  # rows frame still drove the vocab


# ---------------------------------------------------------------------------
# T5 — leakage parity, INCLUDING a high-cardinality near-unique column.
# This is the case the LOW_CARD n_unique sentinel used to silently suppress.
# ---------------------------------------------------------------------------


def _diag_set(diags) -> set:
    """Normalize a list of Diagnostic (live OR dict) to a {(column, kind)} set."""
    out = set()
    for d in diags:
        data = d.data if hasattr(d, "data") else d.get("data", {})
        out.add((data.get("column"), data.get("kind")))
    return out


def test_T5_leakage_parity_high_card_near_unique(tmp_path):
    """A 30k-row near-unique, non-id-named string column that DETERMINES the target.
    In-RAM ``leakage_scan`` flags it identity_like AND target_determines. Streaming
    MUST emit the same cards — the cap freed the value LIST but the COUNT and the
    determinism tracking survive."""
    n = 30_000
    blobs = [f"blob_{i:06d}" for i in range(n)]  # all distinct → near-unique (ratio 1.0)
    target = [("A" if i % 2 == 0 else "B") for i in range(n)]  # each blob → one label
    df = pd.DataFrame({"token_blob": blobs, "label": target})
    path = tmp_path / "leak.csv"
    df.to_csv(path, index=False)

    column_order = list(pd.read_csv(path, nrows=0).columns)
    stats = _stream_over_file(path, column_order, target="label")

    inram = _diag_set(
        d for d in leakage_scan(df, target="label") if d.data.get("column") == "token_blob"
    )
    streamed = _diag_set(
        d
        for d in stats.to_eda_diagnostics(target="label")
        if d["data"].get("column") == "token_blob"
    )
    assert inram == streamed, f"leakage parity broken: in-RAM {inram} != stream {streamed}"
    assert ("token_blob", "identity_like") in streamed
    assert ("token_blob", "target_determines") in streamed

    # the value LIST is still freed (high-card) but the COUNT is the true distinct.
    cs = stats.columns["token_blob"]
    assert cs.values is None and cs.hash_bound is True
    assert cs.n_unique == n


def test_T5_leakage_parity_full_fixture(medium_csv, full_df):
    """Whole-fixture parity for ALL columns (under- and over-cap), with a target.
    Pins identity_like / target_correlated / target_determines by (column, kind)."""
    column_order = list(pd.read_csv(medium_csv, nrows=0).columns)
    stats = _stream_over_file(medium_csv, column_order, target="label")

    inram = _diag_set(leakage_scan(full_df, target="label"))
    streamed = _diag_set(stats.to_eda_diagnostics(target="label"))
    assert inram == streamed, f"whole-fixture leakage parity broken: {inram} ^ {streamed}"


def test_T5_target_correlated_parity_off_boundary(tmp_path):
    """A numeric column highly (but not perfectly) correlated with a numeric target,
    off the 0.95 boundary — streamed Welford moments must agree with pandas corr to
    within tolerance and emit the SAME target_correlated card."""
    rng = np.random.RandomState(7)
    n = 20_000
    base = rng.uniform(0, 100, size=n)
    target = base + rng.normal(0, 1.0, size=n)  # |corr| well above 0.95
    df = pd.DataFrame({"feat": np.round(base, 3), "label": np.round(target, 3)})
    path = tmp_path / "corr.csv"
    df.to_csv(path, index=False)

    column_order = list(pd.read_csv(path, nrows=0).columns)
    stats = _stream_over_file(path, column_order, target="label")

    inram = _diag_set(leakage_scan(df, target="label"))
    streamed = _diag_set(stats.to_eda_diagnostics(target="label"))
    assert ("feat", "target_correlated") in inram
    assert ("feat", "target_correlated") in streamed

    # the abs_corr values must agree to a tight tolerance (Welford parallel merge).
    inram_corr = next(
        d.data["abs_corr"]
        for d in leakage_scan(df, target="label")
        if d.data.get("column") == "feat" and d.data.get("kind") == "target_correlated"
    )
    stream_corr = next(
        d["data"]["abs_corr"]
        for d in stats.to_eda_diagnostics(target="label")
        if d["data"].get("column") == "feat" and d["data"].get("kind") == "target_correlated"
    )
    assert abs(inram_corr - stream_corr) < 1e-3


# ---------------------------------------------------------------------------
# Round-trip: the bundle must survive extras['col_stats'] JSON serialization.
# ---------------------------------------------------------------------------


def test_streamingstats_roundtrips_through_json(medium_csv):
    import json

    from loom.engine.streaming import StreamingStats

    column_order = list(pd.read_csv(medium_csv, nrows=0).columns)
    stats = _stream_over_file(medium_csv, column_order, target="label")
    blob = json.dumps(stats.to_dict())  # must be JSON-able (stored INLINE on extras)
    back = StreamingStats.from_dict(json.loads(blob))

    assert back.to_schema() == stats.to_schema()
    assert back.to_eda_diagnostics("label") == stats.to_eda_diagnostics("label")
