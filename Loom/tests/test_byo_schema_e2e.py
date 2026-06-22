"""Task D — the STRANGER'S-BANK end-to-end proof (bring-your-own-schema).

A first-time user — a data scientist at a bank with their OWN transaction schema
(``account_id, txn_amount, mcc, channel, dr_cr, balance, merchant_name, txn_ts,
is_fraud``) that matches NEITHER hardcoded preset (``financial`` = TabFormer's
exact columns, ``chain`` = DEX) — must be able to tokenize their data out of the
box, with no GPU and no preset match. This module proves the whole authoring flow:

    loom ingest  --in bank.csv --entity account_id --event txn --target is_fraud
      → loom propose  --in IngestDataset/1
      → (human reviews/tweaks the emitted spec)
      → loom tokenize --in IngestDataset/1 --spec TokenizerSpec/1   → a Corpus

The conformance gates (every one a HARD INVARIANT from the build brief):

  1. ``propose`` EXCLUDES the entity (``account_id``, T2 — identity comes from
     history, not an embedding) and the target (``is_fraud`` — leakage) from the
     vocab, and SURFACES the originating EDA leakage flag so the human sees *why*
     (INVARIANT #4). The surviving columns are classified per the doc §1–§3 rule
     set: continuous floats → log-bins, small-range int → FixedVocab, low-card
     categoricals → mapping, high-card → hash, timestamp → calendar + TimeDelta.
  2. ``tokenize --spec`` compiles THAT custom field-map (``preset="custom"``)
     through the EXISTING C1/C2/C3 contracts UNCHANGED → a contract-checked Corpus
     with REAL materialized corpus lines (verdict PASS), and a deliberately-broken
     edited spec is REFUSED_CONTRACT with the named C1 diff and NO Corpus written
     (the contracts are the safety net for a stranger's schema — INVARIANT #2).
  3. REGRESSION (the dual-driver byte-identity, INVARIANT #1): the existing
     ``loom tokenize --preset financial`` path stays byte-identical across the two
     faces and still compiles vocab 6251 — the new ``--spec`` path is purely
     additive and never perturbs the preset path.

These ``assert``s are STRICT. They only skip on the one "not-landed-yet" condition
the rest of the suite uses (a seam still raising ``NotImplementedError`` / the
``propose`` verb or the ``--spec`` arg not yet registered) — exactly the
``golden_helpers`` discipline: a real implementation that returns the WRONG answer
is a hard failure, never a skip.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from loom import REGISTRY
from loom.registry import VerbContext
from loom.store import ObjectStore
from loom.tools import dispatch
from loom.types import Severity, Status, Verdict


# ===========================================================================
# Fixtures — the stranger's bank schema (matches NO preset).
# ===========================================================================


#: The bank's raw columns, in order. NONE of these are a ``financial`` (cust,
#: card, amount, mcc, merchant, chip, zip, state, datetime) or ``chain`` (wallet,
#: timestamp, venue, side, item, size_usd) preset column set — this is a wholly
#: novel schema that today's two presets cannot tokenize.
_BANK_COLUMNS = [
    "account_id",   # grouping ENTITY → excluded from the vocab (T2)
    "txn_amount",   # continuous $ float → log-spaced threshold bins
    "mcc",          # small-range int code → FixedVocab
    "channel",      # low-card categorical (POS/ONLINE/ATM) → mapping
    "dr_cr",        # 2-value categorical (DR/CR) → mapping
    "balance",      # continuous float → log-spaced threshold bins
    "merchant_name",  # high-cardinality string → hash
    "txn_ts",       # datetime → calendar (hour/dow/month) + inter-event TimeDelta
    "is_fraud",     # declared TARGET → excluded (leakage)
]


def _make_bank_df(n: int, *, seed: int = 7) -> pd.DataFrame:
    """A synthetic bank transaction frame over the stranger schema.

    Deterministic (seeded). The continuous columns (``txn_amount``/``balance``)
    are drawn from a small set of repeated values so they are coarse (NOT
    near-unique) — a real proposer bins them rather than flagging them id-shaped;
    ``merchant_name`` is high-cardinality (~1.1K distinct) so it classifies as a
    hash; ``account_id`` repeats per entity (50 accounts) so it is a grouping key,
    not a row id."""
    rng = np.random.RandomState(seed)
    base = pd.Timestamp("2026-01-01 00:00:00")
    return pd.DataFrame(
        {
            "account_id": [f"ACC{i % 50:04d}" for i in range(n)],
            "txn_amount": rng.choice(
                [4.50, 12.0, 25.0, 80.0, 150.0, 500.0, 1200.0, 4800.0], n
            ),
            "mcc": rng.choice([5411, 5814, 4111, 5942, 5912, 5732], n),
            "channel": rng.choice(["POS", "ONLINE", "ATM"], n),
            "dr_cr": rng.choice(["DR", "CR"], n),
            "balance": rng.choice([100.0, 500.0, 1500.0, 5000.0, 12000.0], n),
            "merchant_name": [f"MERCH_{rng.randint(0, 1500)}" for _ in range(n)],
            "txn_ts": pd.to_datetime(
                pd.Series(
                    [base + pd.Timedelta(hours=int(h)) for h in rng.randint(0, 24 * 90, n)]
                )
            ).astype(str),
            "is_fraud": rng.choice([0, 1], n, p=[0.92, 0.08]),
        }
    )[_BANK_COLUMNS]


def _sniffed_schema(df: pd.DataFrame, *, n_rows: int) -> dict:
    """Reproduce ``ingest._sniff_schema`` from a representative frame but OVERRIDE
    ``n_rows`` to ``n_rows`` — so the pure classifier's occupancy gate
    (``corpus_events / token_count >= 1000``) sees a corpus large enough that the
    high-card ``merchant_name`` field earns a hash (256 buckets ⇒ needs ~256K
    events), without paying to materialize 256K rows in a verb test."""
    cols = {}
    for c in df.columns:
        s = df[c]
        cols[str(c)] = {
            "dtype": str(s.dtype),
            "null_frac": round(float(s.isna().mean()), 6),
            "n_unique": int(s.dropna().nunique()),
        }
    return {"n_rows": int(n_rows), "n_cols": int(len(df.columns)), "columns": cols}


@pytest.fixture
def store(tmp_path, monkeypatch) -> ObjectStore:
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    return ObjectStore(str(tmp_path))


def _ctx(store: ObjectStore, **kw) -> VerbContext:
    kw.setdefault("driver", "cli")
    kw.setdefault("interactive", True)
    return VerbContext(store=store, **kw)


# ===========================================================================
# Landing guards — skip ONLY on "not landed yet", assert strictly otherwise.
# ===========================================================================


def _require_propose_verb() -> None:
    """Skip iff the ``propose`` verb is not registered yet (a parallel Implement
    agent has not landed it). Once registered, every assertion below is strict."""
    if "propose" not in REGISTRY:
        pytest.skip("`propose` verb not registered yet (BYO-schema seam not landed)")


def _require_spec_arg() -> None:
    """Skip iff ``tokenize`` does not yet advertise the additive ``--spec`` arg."""
    if "spec" not in (REGISTRY["tokenize"].params or {}).get("properties", {}):
        pytest.skip("`tokenize --spec` not wired yet (the additive field-map path)")


def _call(verb: str, args: dict, ctx: VerbContext):
    """Invoke a verb ``fn``; turn a still-stubbed downstream seam into a skip (the
    repo's ``golden_helpers`` discipline) but let real results through to strict
    asserts."""
    try:
        return REGISTRY[verb].fn(dict(args), ctx)
    except NotImplementedError:  # pragma: no cover - a seam is still a stub
        pytest.skip(f"{verb}: a downstream seam is still a stub (NotImplementedError)")


def _ingest_bank(store: ObjectStore, tmp_path_dir, df: pd.DataFrame):
    """``loom ingest`` the bank frame; return the IngestDataset pathspec."""
    csv = tmp_path_dir / "bank.csv"
    df.to_csv(csv, index=False)
    res = _call(
        "ingest",
        {"in": str(csv), "name": "stranger-bank", "entity": "account_id",
         "event": "txn", "target": "is_fraud"},
        _ctx(store),
    )
    assert res.status is Status.OK
    assert res.outputs, "ingest must persist an IngestDataset"
    pathspec = res.outputs[0].pathspec
    assert pathspec.startswith("IngestDataset/")
    return pathspec


# ===========================================================================
# 1. The PURE classifier — the full §F worked example, including merchant→hash.
#    Pure + instant (no materialization): it reads the SNIFFED SCHEMA only, so we
#    declare a large corpus (n_rows) and assert the complete field→strategy map.
# ===========================================================================


def test_propose_classifier_maps_every_stranger_field(store, tmp_path):
    """End-to-end through the verbs (``ingest`` → ``propose``) on the bank schema,
    proving the EXACT §1–§3 classifier outcome — at a corpus size where the
    high-card ``merchant_name`` decisively earns a HASH (256K events).

    The proposer reads ``ingest``'s sniffed schema + EDA flags; the occupancy gate
    is a function of ``n_rows``, not of the materialized frame, so we ingest a
    small representative frame and re-drive ``propose`` on a hand-built large-``n``
    IngestDataset payload — keeping the test instant while exercising the real
    verb + real classifier."""
    _require_propose_verb()

    # A small representative frame for the ingest round-trip + a large declared
    # corpus so the hash field is not starved (occupancy floor = 1000 occ/token).
    small = _make_bank_df(2000)
    in_path = _ingest_bank(store, tmp_path, small)

    # Re-stamp the persisted IngestDataset's sniffed schema to a 260K-row corpus
    # (the classifier reads schema["n_rows"]; cardinalities/dtypes are unchanged).
    obj = store.get(in_path)
    obj.extras["schema"] = _sniffed_schema(small, n_rows=260_000)
    from loom.store import _write_json_atomic  # the store's own atomic writer
    _write_json_atomic(store._obj_dir(obj.ref) / "object.json", obj.to_dict())

    res = _call("propose", {"in": in_path}, _ctx(store))
    assert res.status is Status.OK, res.summary
    assert res.verdict is Verdict.REVIEW  # a PROPOSAL the human reviews/edits
    assert res.outputs, "propose must emit a reviewable TokenizerSpec object"
    assert res.outputs[0].pathspec.startswith("TokenizerSpec/")

    data = res.data
    incl = {f["name"]: f for f in data["included"]}
    excl = {e["column"]: e for e in data["excluded"]}

    # --- INVARIANT #4: the entity (T2) and the target (leakage) are EXCLUDED ---
    assert "account_id" in excl and excl["account_id"]["reason"] == "entity"
    assert "is_fraud" in excl and excl["is_fraud"]["reason"] == "target"
    # ...and NEITHER survives as a tokenized field, by name OR as a step source
    # (a hand guard against the entity/target leaking back into the vocab).
    included_sources = {f["source"] for f in data["included"]}
    assert "account_id" not in incl and "account_id" not in included_sources
    assert "is_fraud" not in incl and "is_fraud" not in included_sources

    # --- the §F field→strategy classification (each column, exactly) ----------
    # continuous floats → log-spaced bins (the "amount" strategy keyword).
    assert incl["txn_amount"]["strategy"] == "amount"
    assert incl["balance"]["strategy"] == "amount"
    # small-range int code → FixedVocab.
    assert incl["mcc"]["strategy"] == "fixedvocab"
    # low-card categoricals → mapping (values + a default).
    assert incl["channel"]["strategy"] == "mapping"
    assert incl["dr_cr"]["strategy"] == "mapping"
    # high-cardinality string → hash (the merchant/counterparty rule).
    assert incl["merchant_name"]["strategy"] == "hash", (
        "merchant_name (~1.1K distinct over a 260K corpus) must earn a HASH, not be "
        f"starved/dropped: {excl.get('merchant_name')}"
    )
    # timestamp → calendar tokens (hour/dow/month) + an inter-event TimeDelta.
    ts_steps = {f["name"]: f for f in data["included"] if f["source"] == "txn_ts"}
    strategies = {f["strategy"] for f in ts_steps.values()}
    assert "calendar" in strategies, "txn_ts must expand to calendar tokens"
    assert "timedelta" in strategies, "txn_ts must add an inter-event TimeDelta gap"
    # the calendar family covers hour, dow, month (3 calendar steps) + 1 timedelta.
    cal_parts = {f["params"].get("part") for f in ts_steps.values() if f["strategy"] == "calendar"}
    assert {"hour", "dow", "month"} <= cal_parts

    # --- the hand-counted derived numbers are finite + self-consistent (E) -----
    assert isinstance(data["vocab_size"], int) and data["vocab_size"] > 0
    tpe = data["tokens_per_event"]
    assert tpe == len(data["included"]), "tokens_per_event == count of included field-steps"
    assert data["chunk_size"] == 4096 // (tpe + 1), "chunk_size = context_len // (tpe+1)"


def test_propose_surfaces_the_entity_leakage_flag_for_review(store, tmp_path):
    """INVARIANT #4 (the human-facing half): ``propose`` surfaces the ORIGINATING
    EDA leakage flag for the excluded entity on the result card, so a human sees
    *why* ``account_id`` was dropped (and could override) — it does not silently
    drop it. The entity's identity-like EDA card travels onto the proposal."""
    _require_propose_verb()
    in_path = _ingest_bank(store, tmp_path, _make_bank_df(2000))
    res = _call("propose", {"in": in_path}, _ctx(store))
    assert res.status is Status.OK

    # The entity exclusion is surfaced as a diagnostic that names the column AND
    # carries the ingest EDA card's identity-like reasoning (not a bare drop).
    entity_cards = [
        d for d in res.diagnostics
        if "account_id" in (d.message or "") and "EXCLUDED" in (d.message or "")
    ]
    assert entity_cards, "the excluded entity must be surfaced as a review diagnostic"
    card = entity_cards[0]
    assert "identity" in (card.message or "").lower(), (
        "the surfaced exclusion must carry the originating identity-like EDA reasoning"
    )


# ===========================================================================
# 2. tokenize --spec: compile the proposed field-map → a contract-checked Corpus.
#    Run on a moderate frame so the corpus materializes fast; the columns that
#    survive the occupancy gate at this size still cover every strategy family
#    EXCEPT the hash (asserted at scale in test 1), proving the compile path.
# ===========================================================================


def test_tokenize_spec_compiles_a_real_custom_corpus(store, tmp_path):
    """``loom tokenize --in IngestDataset/1 --spec TokenizerSpec/1`` compiles the
    PROPOSED custom field-map (``preset="custom"``) through the existing C1/C2/C3
    contracts UNCHANGED and materializes a REAL Corpus from the bank rows:

      * verdict PASS, a ``Corpus/<n>`` object persisted (``wrote_corpus``),
      * a finite vocab + a custom (non-financial/chain) preset,
      * real corpus lines (``<bos> txn (<sep> txn)* <eos>``) with ``n_txns`` == the
        ingested row count — proving a stranger's schema tokenizes out of the box."""
    _require_propose_verb()
    _require_spec_arg()

    df = _make_bank_df(12_000)  # > occupancy for amount(8)/channel(4)/dr_cr(3)
    in_path = _ingest_bank(store, tmp_path, df)
    prop = _call("propose", {"in": in_path}, _ctx(store))
    assert prop.status is Status.OK and prop.outputs
    spec_pathspec = prop.outputs[0].pathspec

    # Compile THAT proposed spec into a corpus from the SAME ingested rows.
    tok = _call("tokenize", {"in": in_path, "spec": spec_pathspec}, _ctx(store))
    assert tok.status is Status.OK, tok.summary
    assert tok.verdict is Verdict.PASS, tok.summary
    data = tok.data

    # The custom field-map compiled — NOT a preset (INVARIANT #1: preset path
    # untouched, this is the new additive custom path).
    assert data["preset"] == "custom"
    assert isinstance(data["vocab_size"], int) and data["vocab_size"] > 0

    # A real Corpus was persisted, content-addressed, with materialized lines.
    assert data["wrote_corpus"] is True
    assert data["pathspec"].startswith("Corpus/")
    assert store.list("Corpus"), "a PASS custom-spec tokenize must persist a Corpus"
    # Every ingested row is a transaction in the corpus (real materialization).
    assert data["n_txns"] == len(df)
    assert data["n_lines"] >= 1

    # The persisted corpus lines obey the C3 grammar for the custom tokens/txn.
    corpus = store.get(data["pathspec"])
    assert corpus.payload_path
    payload = json.loads(open(corpus.payload_path, encoding="utf-8").read())
    lines = payload["corpus_lines"]
    assert lines, "the custom Corpus must carry real corpus lines"
    tpt = int(data["tokens_per_txn"])
    for line in lines[:20]:
        toks = line.split(" ")
        assert toks[0] == "<bos>" and toks[-1] == "<eos>"
        body = toks[1:-1]
        for txn in " ".join(body).split(" <sep> "):
            assert len(txn.split(" ")) == tpt, (
                f"each txn must carry {tpt} field tokens: {txn!r}"
            )
        # No nan/None leaked into the stranger-schema token stream.
        assert not any("nan" in t.lower() for t in toks), line

    # --- the proposer-emitted MAPPING fields carry REAL signal, not a constant ---
    # The proposer reads the ingested rows and enumerates each low-card categorical's
    # observed values into the field-map, so the compiled MappingPassthrough maps
    # each value (POS/ONLINE/ATM, DR/CR) to its OWN token. A mapping that carried only
    # a cardinality count would collapse every value to the single `_UNK` default —
    # a PASS verdict over a semantically-empty corpus. Assert it tokenizes to >1 token.
    vocab = payload["vocab"]
    for prefix, col in (("CHANNEL", "channel"), ("DR_CR", "dr_cr")):
        toks = sorted(t for t in vocab if t.startswith(prefix + "_"))
        non_default = [t for t in toks if not t.endswith("_UNK")]
        assert len(non_default) > 1, (
            f"the proposed mapping for {col!r} must tokenize each observed value to "
            f"its own token (not collapse to the default): got {toks}"
        )
    # And those real value-tokens actually appear in the corpus stream (not just the
    # vocab) — the signal survives all the way to materialized lines.
    blob = " ".join(lines[:50])
    assert "CHANNEL_POS" in blob or "CHANNEL_ONLINE" in blob or "CHANNEL_ATM" in blob
    assert "DR_CR_DR" in blob or "DR_CR_CR" in blob


def test_tokenize_spec_vocab_matches_the_proposal(store, tmp_path):
    """The compiled custom corpus's vocab is the one the proposal accounted for:
    the proposed ``tokens_per_event`` equals the compiled ``tokens_per_txn`` (the
    field-step count is invariant across propose→compile)."""
    _require_propose_verb()
    _require_spec_arg()

    df = _make_bank_df(12_000)
    in_path = _ingest_bank(store, tmp_path, df)
    prop = _call("propose", {"in": in_path}, _ctx(store))
    assert prop.status is Status.OK
    tok = _call("tokenize", {"in": in_path, "spec": prop.outputs[0].pathspec}, _ctx(store))
    assert tok.status is Status.OK

    # propose's tokens_per_event (count of included field-steps) == compiled
    # tokens_per_txn — the proposal and the compiled spec agree on the grammar.
    assert tok.data["tokens_per_txn"] == prop.data["tokens_per_event"]
    # The proposed chunk_size formula is the one the compiler derives (C3).
    assert tok.data["chunk_size"] == prop.data["chunk_size"]


def test_tokenize_spec_accepts_a_human_edited_yaml_file(store, tmp_path):
    """The flow's middle step: a human downloads the proposed field-map, edits the
    YAML, and re-feeds it as a FILE path to ``tokenize --spec`` (the
    ``<file-or-pathspec>`` surface). A hand-written ``loom-fieldmap/1`` over the
    stranger schema compiles to the same contract-checked custom Corpus."""
    _require_propose_verb()
    _require_spec_arg()
    yaml = pytest.importorskip("yaml")

    df = _make_bank_df(12_000)
    in_path = _ingest_bank(store, tmp_path, df)

    # A minimal, hand-authored field-map (the columns that earn a token at this
    # size), exercising the file-path resolution + every non-hash strategy family.
    fieldmap = {
        "version": "loom-fieldmap/1",
        "entity": "account_id",     # EXCLUDED (T2)
        "event": "txn",
        "target": "is_fraud",       # EXCLUDED (leakage)
        "context_len": 4096,
        "fields": [
            {"name": "amt", "source": "txn_amount", "strategy": "amount", "bins": 8},
            {"name": "mcc", "source": "mcc", "strategy": "fixedvocab", "min": 0, "max": 5},
            {"name": "chan", "source": "channel", "strategy": "mapping",
             "values": ["POS", "ONLINE", "ATM"], "default": "UNK"},
            {"name": "drcr", "source": "dr_cr", "strategy": "mapping",
             "values": ["DR", "CR"], "default": "UNK"},
            {"name": "bal", "source": "balance", "strategy": "amount", "bins": 8},
            {"name": "ts", "source": "txn_ts", "strategy": "calendar"},  # → hour+dow+month
            {"name": "gap", "source": "txn_ts", "strategy": "timedelta", "bins": 32},
        ],
    }
    fm_path = tmp_path / "edited_fieldmap.yaml"
    fm_path.write_text(yaml.safe_dump(fieldmap, sort_keys=False))

    tok = _call("tokenize", {"in": in_path, "spec": str(fm_path)}, _ctx(store))
    assert tok.status is Status.OK, tok.summary
    assert tok.verdict is Verdict.PASS
    assert tok.data["preset"] == "custom"
    assert tok.data["wrote_corpus"] is True
    assert tok.data["n_txns"] == len(df)


# ===========================================================================
# 2b. The contracts are the SAFETY NET — a deliberately-broken edited spec is
#     REFUSED with the named C1 diff and NO Corpus (INVARIANT #2).
# ===========================================================================


def test_broken_edited_spec_is_refused_with_named_c1_diff(store, tmp_path):
    """A stranger could hand-edit the YAML into a non-injective vocab. The compiled
    custom spec MUST run through the existing C1/C2/C3 contracts: a bad field-map
    (two fields emitting the SAME token strings → a C1 injectivity collision) is
    REFUSED_CONTRACT with the named C1 diff and NO Corpus written — never a
    silently-broken corpus (INVARIANT #2 — the contracts are the safety net)."""
    _require_spec_arg()
    yaml = pytest.importorskip("yaml")

    # Two FixedVocab fields sharing prefix DUP over the same range → DUP_0..DUP_3
    # are each claimed twice ⇒ C1 injectivity FAIL.
    broken = {
        "version": "loom-fieldmap/1",
        "entity": "account_id",
        "event": "txn",
        "context_len": 4096,
        "fields": [
            {"name": "a", "source": "col_a", "strategy": "fixedvocab",
             "min": 0, "max": 3, "prefix": "DUP"},
            {"name": "b", "source": "col_b", "strategy": "fixedvocab",
             "min": 0, "max": 3, "prefix": "DUP"},
        ],
    }
    fm_path = tmp_path / "broken_fieldmap.yaml"
    fm_path.write_text(yaml.safe_dump(broken, sort_keys=False))

    res = _call("tokenize", {"spec": str(fm_path)}, _ctx(store))
    assert res.status is Status.REFUSED_CONTRACT, res.summary
    assert res.verdict is Verdict.FAIL
    assert res.data.get("wrote_corpus") is False
    assert store.list("Corpus") == [], "a refused custom spec must NOT persist a Corpus"

    # The named C1 ERROR diff travels on the refusal envelope (the human sees the
    # exact colliding tokens), and the verb's summary names the failing contract.
    c1 = [d for d in res.diagnostics if d.contract == "C1" and d.severity is Severity.ERROR]
    assert c1, "the C1 injectivity ERROR must travel on the refusal envelope"
    assert "DUP" in (c1[0].message or ""), "the named diff must point at the colliding token"
    assert "C1" in res.summary


def test_broken_spec_refusal_is_byte_identical_across_faces(store, tmp_path):
    """The custom ``--spec`` refusal envelope is dual-driver byte-identical: the CLI
    ``fn`` result equals the agent ``dispatch`` result char-for-char — the locked
    invariant holds on the NEW custom path's REFUSED case too, not just PASS."""
    _require_spec_arg()
    yaml = pytest.importorskip("yaml")

    broken = {
        "version": "loom-fieldmap/1",
        "entity": "account_id",
        "context_len": 4096,
        "fields": [
            {"name": "a", "source": "col_a", "strategy": "fixedvocab",
             "min": 0, "max": 3, "prefix": "DUP"},
            {"name": "b", "source": "col_b", "strategy": "fixedvocab",
             "min": 0, "max": 3, "prefix": "DUP"},
        ],
    }
    fm_path = tmp_path / "broken_fieldmap.yaml"
    fm_path.write_text(yaml.safe_dump(broken, sort_keys=False))
    args = {"spec": str(fm_path)}

    cli = _call("tokenize", dict(args), _ctx(store)).to_json()
    tool = dispatch("loom.tokenize", dict(args)).to_json()
    assert cli == tool


# ===========================================================================
# 3. REGRESSION — the preset path is byte-identical (INVARIANT #1). The new
#    --spec branch is purely additive; with --spec ABSENT nothing changes.
# ===========================================================================


def test_preset_financial_unchanged_and_byte_identical(store):
    """``loom tokenize --preset financial`` is byte-identical across the two faces
    and still compiles vocab 6251 — the new ``--spec`` path did NOT perturb the
    preset path (the dual-driver regression, INVARIANT #1)."""
    cli = REGISTRY["tokenize"].fn({"preset": "financial"}, _ctx(store))
    tool = dispatch("loom.tokenize", {"preset": "financial"})
    assert cli.to_json() == tool.to_json()
    payload = json.loads(cli.to_json())
    assert payload["verb"] == "tokenize"
    assert payload["data"]["vocab_size"] == 6251


@pytest.mark.parametrize("preset", ["financial", "chain"])
def test_both_presets_compile_with_spec_absent(store, preset):
    """With ``--spec`` ABSENT, both presets compile a PASS Corpus exactly as before
    — the additive field-map path is inert on the preset drivers."""
    res = REGISTRY["tokenize"].fn({"preset": preset}, _ctx(store))
    assert res.status is Status.OK
    assert res.verdict is Verdict.PASS
    data = json.loads(res.to_json())["data"]
    assert data["preset"] == preset
    assert data["vocab_size"] > 0
    # The preset path never emits the "custom" marker.
    assert data["preset"] != "custom"


def test_tokenize_schema_keeps_preset_and_gains_spec_only(store):
    """``tokenize`` still advertises the LOCKED preset surface, plus the new
    additive ``--spec`` arg — and never leaks the generic ``representation`` flag
    (it stays pinned to event-sequence). The preset enum is untouched."""
    props = REGISTRY["tokenize"].params["properties"]
    assert "preset" in props, "the preset arg must remain on the locked tokenize surface"
    assert "representation" not in props, "tokenize must not expose the representation flag"
    # The additive arg, once landed, is advertised on BOTH faces (the agent
    # input_schema IS params), so an agent can drive --spec too.
    if "spec" in props:
        from loom.tools import all_tool_schemas

        schema = {s["name"]: s for s in all_tool_schemas()}["loom.tokenize"]
        assert "spec" in schema["input_schema"]["properties"]
