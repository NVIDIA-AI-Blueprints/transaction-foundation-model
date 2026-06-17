"""Proposer quality-gap regression — the two gaps the TabFormer ground-truth
control exposed (NOT architecture, just classification/display QUALITY).

The BYO proposer (``engine/propose.py`` + ``engine/spec.py`` + ``verbs/propose.py``)
generalizes (synthetic ARI 1.0, DNA k-mer ARI 1.0) but the TabFormer control found
two gaps:

  * GAP 1 (meaningful) — a currency/numeric-coercible STRING column (e.g. the
    TabFormer ``amount`` arriving as ``"$12.50"``) was classified as ``hash`` when
    high-cardinality, losing the magnitude ordering. A numeric-string amount must
    instead be detected as CONTINUOUS → the ``amount`` (log-bins) strategy, exactly
    like a float column — but ONLY after the entity/target/id-shaped/near-unique-NAME
    exclusions, so an ``account_id``-style numeric(-string) id is STILL excluded.
  * GAP 2 (cosmetic) — ``loom propose --json`` reported ``data.fields`` with
    ``name: None`` and an INCLUDED count of 0, even though fields ARE included. The
    verb's ``data`` must surface the real INCLUDED fields (name + strategy) + count
    from the ``SpecDraft.fields`` FieldProposals.

These tests pin the FIXED behavior and guard the HARD INVARIANTS: an id-shaped
numeric(-string) column is NEVER swallowed into ``amount``; a genuine free-text
high-card column still hashes; and the financial preset stays byte-identical
(vocab 6251 + dual-driver), so neither fix perturbs the conformance oracle.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from loom import REGISTRY
from loom.eda import leakage_scan
from loom.engine import compile_spec, financial_spec, spec_from_field_map
from loom.engine.propose import propose_spec
from loom.engine.spec import materialize_corpus_lines, preprocess_field_map
from loom.registry import VerbContext
from loom.store import ObjectStore
from loom.tools import dispatch
from loom.types import Status
from loom.verbs.ingest import _sniff_schema

# The byte-identical financial signature (the conformance oracle, mirrored from
# tests/test_fixes_regression.py). Neither GAP fix may move a locked preset id.
FINANCIAL_VOCAB_SIZE = 6251
FINANCIAL_VOCAB_HASH = (
    "sha256:ba0e0daa6c1d64a1028e428b7981a82a69fe45a42cc42161277df04aa9152ce4"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _dollarize(values) -> list[str]:
    """Format a numeric series as TabFormer-style ``"$x.yz"`` strings."""
    return ["$%.2f" % float(v) for v in values]


def _propose(df: pd.DataFrame, *, entity: str, target=None, n_rows: int | None = None):
    """Run the pure classifier the way the verb does (sniffed schema + EDA flags +
    the rows frame so categoricals carry their observed values). ``n_rows`` overrides
    the sniffed corpus size (to drive the occupancy gate without materializing)."""
    schema = _sniff_schema(df)
    if n_rows is not None:
        schema["n_rows"] = int(n_rows)
    return propose_spec(
        schema=schema,
        eda_flags=leakage_scan(df, target=target),
        entity=entity,
        event="txn",
        target=target,
        context_len=4096,
        rows=df,
    )


def _ctx(store: ObjectStore) -> VerbContext:
    return VerbContext(store=store, driver="cli", interactive=True)


def _require_propose_verb() -> None:
    if "propose" not in REGISTRY:
        pytest.skip("`propose` verb not registered yet (BYO-schema seam not landed)")


# ===========================================================================
# GAP 1 — a $-formatted (numeric-coercible) amount STRING → the `amount`
# (log-bins) strategy, NOT hash. And the field-map compiles + tokenizes the
# "$12.50"-style values (currency formatting stripped before binning).
# ===========================================================================


def test_high_card_dollar_string_amount_is_continuous_not_hash():
    """The control's exact gap: ``amount`` arrives as a HIGH-cardinality ``"$x"``
    string. Pre-fix the proposer saw a high-card *string* → ``hash`` (magnitude
    ordering lost). The fix detects a numeric-coercible string column and routes it
    to CONTINUOUS → the ``amount`` (log-bins) strategy, exactly like a float."""
    rng = np.random.RandomState(0)
    n = 50_000
    amounts = rng.lognormal(3.0, 1.0, n).round(2)  # ~10K+ distinct → high-card
    df = pd.DataFrame(
        {
            "account_id": [f"A{i % 2000}" for i in range(n)],
            "amount": _dollarize(amounts),  # $-formatted STRING column
        }
    )
    # The column is genuinely a high-cardinality STRING (not a float dtype).
    schema = _sniff_schema(df)
    assert "int" not in schema["columns"]["amount"]["dtype"].lower()
    assert "float" not in schema["columns"]["amount"]["dtype"].lower()
    assert schema["columns"]["amount"]["n_unique"] >= 500  # high-card → would-be hash

    draft = _propose(df, entity="account_id")
    by_src = {f.source: f for f in draft.fields}
    excl = {e.name: e for e in draft.excluded}
    assert "amount" in by_src, f"$-amount must be tokenized, not dropped: {excl.get('amount')}"
    assert by_src["amount"].strategy == "amount", (
        f"$-string amount must bin (continuous), not {by_src['amount'].strategy!r}"
    )
    # It must NOT be hashed (the pre-fix mis-classification) and NOT dropped.
    assert "amount" not in excl
    assert by_src["amount"].strategy != "hash"


def test_dollar_string_amount_is_treated_like_a_float_column():
    """A numeric-coercible string amount earns the SAME proposal as the equivalent
    FLOAT column — same `amount` strategy and same bin count — so currency formatting
    is purely cosmetic to the classifier."""
    rng = np.random.RandomState(1)
    n = 50_000
    amounts = rng.lognormal(3.0, 1.0, n).round(2)
    df_str = pd.DataFrame(
        {"account_id": [f"A{i % 2000}" for i in range(n)], "amount": _dollarize(amounts)}
    )
    df_float = pd.DataFrame(
        {"account_id": [f"A{i % 2000}" for i in range(n)], "amount": amounts.astype(float)}
    )
    str_amt = {f.source: f for f in _propose(df_str, entity="account_id").fields}["amount"]
    flt_amt = {f.source: f for f in _propose(df_float, entity="account_id").fields}["amount"]
    assert str_amt.strategy == flt_amt.strategy == "amount"
    assert str_amt.token_count == flt_amt.token_count


def test_proposed_dollar_amount_fieldmap_compiles_and_tokenizes_dollar_values():
    """The field-map the proposer emits for a $-amount column compiles clean (C1/C2/C3)
    AND materialize tokenizes "$12.50"-style values — the `__amt__` recipe strips the
    `$`/`,` formatting before log-binning (mirrors the financial preset)."""
    rng = np.random.RandomState(2)
    n = 50_000
    amounts = rng.lognormal(3.0, 1.0, n).round(2)
    train = pd.DataFrame(
        {"account_id": [f"A{i % 2000}" for i in range(n)], "amount": _dollarize(amounts)}
    )
    draft = _propose(train, entity="account_id")
    spec = spec_from_field_map(draft.fieldmap)
    ct = compile_spec(spec, context_len=4096)
    assert ct.report.passed, [d.message for d in ct.report.diagnostics]
    assert ct.report.injective and ct.report.dense
    assert not ct.report.has_fitted_artifact  # C2: threshold bins, no fitted state

    # Tokenize concrete "$x"-style values; the recipe must strip `$` and `,`.
    sample = pd.DataFrame(
        {
            "account_id": ["A0", "A0", "A1", "A1"],
            "amount": ["$12.50", "$1,500.00", "$0.99", "$4,800.00"],
        }
    )
    pre = preprocess_field_map(sample, spec)
    amt_src = next(s.source for s in spec.steps if s.source.startswith("__amt__"))
    binned = pre[amt_src].tolist()
    # All four parse to a real (positive) magnitude → a bin in 0..bins-1 (no NaN→0
    # collapse from a failed parse), and the ordering is preserved: $0.99 < $12.50
    # < $1,500 < $4,800 ⇒ monotonically non-decreasing bin indices.
    bins = int(amt_src.rsplit("__", 1)[-1])
    assert all(0 <= b <= bins - 1 for b in binned)
    # magnitude ordering is preserved: $0.99 < $12.50 < $1,500 < $4,800.
    assert binned[2] <= binned[0] <= binned[1] <= binned[3]

    lines, n_txns = materialize_corpus_lines(ct, sample)
    assert n_txns == 4
    blob = " ".join(lines)
    amt_prefix = next(
        s.strategy.prefix for s in spec.steps if s.source.startswith("__amt__")
    )
    assert f"{amt_prefix}_" in blob  # the $-values minted real AMT_* tokens


# ===========================================================================
# GAP 1 HARD INVARIANT — the numeric-string detection must NOT swallow an
# id-shaped numeric / numeric-string column: it stays EXCLUDED, never binned.
# ===========================================================================


def test_numeric_string_account_id_is_still_excluded_not_binned():
    """An ``account_id`` numeric-STRING column that is NOT the declared entity is
    still EXCLUDED (id-shaped name / near-unique), never coerced to continuous —
    the numeric-string→`amount` detection must run AFTER the id exclusions."""
    rng = np.random.RandomState(3)
    n = 50_000
    df = pd.DataFrame(
        {
            # entity is wallet; account_id is a FEATURE column here.
            "wallet": [f"w{i % 2000}" for i in range(n)],
            "account_id": [str(rng.randint(0, 800)) for _ in range(n)],  # numeric-string id
            "amount": _dollarize(rng.lognormal(3, 1, n).round(2)),
        }
    )
    draft = _propose(df, entity="wallet")
    by_src = {f.source: f for f in draft.fields}
    excl = {e.name: e for e in draft.excluded}
    assert "account_id" in excl, "id-shaped numeric-string must be EXCLUDED"
    assert "account_id" not in by_src
    # critically: it was NOT swallowed into the continuous `amount` strategy.
    assert excl["account_id"].reason != "amount"
    assert by_src.get("account_id") is None
    # the real amount column DID bin (the fix is active, the id was still excluded).
    assert by_src["amount"].strategy == "amount"


def test_pure_numeric_id_column_is_excluded_not_binned():
    """A pure-INTEGER id-shaped column (``account_id``) is excluded by its id name,
    not binned as a small-range int / continuous — the id exclusion precedes the
    numeric routing."""
    rng = np.random.RandomState(4)
    n = 50_000
    df = pd.DataFrame(
        {
            "wallet": [f"w{i % 2000}" for i in range(n)],
            "account_id": rng.randint(0, 2000, n),  # numeric id-shaped NON-entity
            "amount": rng.lognormal(3, 1, n),
        }
    )
    draft = _propose(df, entity="wallet")
    by_src = {f.source: f for f in draft.fields}
    excl = {e.name: e for e in draft.excluded}
    assert "account_id" in excl and "account_id" not in by_src
    assert by_src["amount"].strategy == "amount"


def test_declared_entity_numeric_string_is_excluded_as_T2():
    """A numeric-string ENTITY column is excluded as T2 (the entity gate runs first),
    never coerced to continuous."""
    rng = np.random.RandomState(5)
    n = 50_000
    df = pd.DataFrame(
        {
            "account_id": [str(rng.randint(0, 2000)) for _ in range(n)],  # numeric-string entity
            "amount": _dollarize(rng.lognormal(3, 1, n).round(2)),
        }
    )
    draft = _propose(df, entity="account_id")
    excl = {e.name: e for e in draft.excluded}
    assert excl["account_id"].reason == "entity"
    assert "account_id" not in {f.source for f in draft.fields}


# ===========================================================================
# GAP 1 boundary — a genuine free-text / high-card NON-numeric string still
# hashes (the detection only fires on numeric-coercible strings).
# ===========================================================================


def test_genuine_free_text_high_card_string_still_hashes():
    """A genuine high-card categorical of NON-numeric strings (merchant/memo) is NOT
    numeric-coercible, so the detection leaves it alone → it still routes to HASH."""
    rng = np.random.RandomState(6)
    n = 50_000
    pool = [f"MERCH_{i}" for i in range(3000)]  # high-card, repeated, non-numeric
    df = pd.DataFrame(
        {
            "account_id": [f"A{i % 2000}" for i in range(n)],
            "amount": _dollarize(rng.lognormal(3, 1, n).round(2)),
            "merchant": rng.choice(pool, n),
        }
    )
    draft = _propose(df, entity="account_id")
    by_src = {f.source: f for f in draft.fields}
    excl = {e.name: e for e in draft.excluded}
    assert "merchant" not in excl, f"high-card text must hash, not drop: {excl.get('merchant')}"
    assert by_src["merchant"].strategy == "hash"
    # the numeric $-amount alongside it still bins (continuous) — both fire correctly.
    assert by_src["amount"].strategy == "amount"


def test_very_high_card_free_text_still_dropped_as_free_text():
    """An extremely high-cardinality NON-numeric free-text column (> HIGH_CARD_MAX,
    but repeated so not near-unique) is still dropped as free-text — the numeric
    detection does not rescue it."""
    n = 1_000_000
    rng = np.random.RandomState(8)
    memo_pool = [f"memo phrase {i}" for i in range(150_000)]  # > 100K distinct, repeated
    df = pd.DataFrame(
        {
            "account_id": rng.randint(0, 5000, n).astype(str),
            "amount": rng.lognormal(3, 1, n),
            "memo": rng.choice(memo_pool, n),
        }
    )
    draft = _propose(df, entity="account_id")
    excl = {e.name: e for e in draft.excluded}
    assert "memo" in excl and excl["memo"].reason == "free-text"
    assert "memo" not in {f.source for f in draft.fields}


# ===========================================================================
# GAP 2 — `loom propose` result `data.fields` lists the real INCLUDED field
# names + strategies + the count (the display fix). The pre-fix bug reported
# name: None and an INCLUDED count of 0.
# ===========================================================================


def _drive_propose(store: ObjectStore, df: pd.DataFrame, tmp_path):
    """ingest → propose through the real verbs; return the propose VerbResult."""
    csv = tmp_path / "gap2.csv"
    df.to_csv(csv, index=False)
    ig = REGISTRY["ingest"].fn(
        {"in": str(csv), "name": "gap2", "entity": "account_id", "event": "txn"},
        _ctx(store),
    )
    assert ig.status is Status.OK and ig.outputs
    return REGISTRY["propose"].fn({"in": ig.outputs[0].pathspec}, _ctx(store))


def _bank_gap2_df(n: int = 2000, *, seed: int = 7) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "account_id": [f"ACC{i % 50:04d}" for i in range(n)],
            "amount": _dollarize(rng.choice([4.5, 12.0, 80.0, 500.0, 1200.0, 4800.0], n)),
            "channel": rng.choice(["POS", "ONLINE", "ATM"], n),
            "dr_cr": rng.choice(["DR", "CR"], n),
        }
    )


def _fields_block(data: dict):
    """The included-fields list from the verb `data`, tolerant of the exact key the
    GAP-2 fix lands under (``data["fields"]`` per the brief, or a nested
    ``{"included": [...]}``). Skips if the fix has not landed yet."""
    fields = data.get("fields")
    if fields is None:
        pytest.skip("GAP-2 display fix not landed yet (data has no `fields` block)")
    if isinstance(fields, dict):  # e.g. {"included": [...], "count": n}
        fields = fields.get("included") or fields.get("fields") or []
    return list(fields)


def test_propose_data_fields_lists_real_included_names_and_strategies(store, tmp_path):
    """The GAP-2 fix: `data.fields` carries the INCLUDED field name + strategy for
    every tokenized field (no `name: None`), and the count is the real INCLUDED
    count (not 0)."""
    _require_propose_verb()
    res = _drive_propose(store, _bank_gap2_df(), tmp_path)
    assert res.status is Status.OK
    data = res.data
    fields = _fields_block(data)

    # The count is the real INCLUDED count, not 0 — there ARE tokenized fields.
    assert len(fields) > 0, "data.fields reports 0 INCLUDED fields (the pre-fix bug)"
    assert len(fields) == data["tokens_per_event"], (
        "data.fields count must equal tokens_per_event (the included field-step count)"
    )

    # Every entry surfaces a real name (not None) + a strategy.
    for f in fields:
        name = f.get("name") if isinstance(f, dict) else getattr(f, "name", None)
        strat = f.get("strategy") if isinstance(f, dict) else getattr(f, "strategy", None)
        assert name, f"data.fields entry has a None/empty name (the pre-fix bug): {f}"
        assert strat, f"data.fields entry {name!r} has no strategy: {f}"

    # The names/strategies match the field-map's emitted field steps exactly.
    fm_fields = {e["name"]: e["strategy"] for e in data["fieldmap"]["fields"]}
    surfaced = {
        (f.get("name") if isinstance(f, dict) else getattr(f, "name")):
        (f.get("strategy") if isinstance(f, dict) else getattr(f, "strategy"))
        for f in fields
    }
    assert surfaced == fm_fields, "data.fields must mirror the emitted field-map steps"


def test_propose_data_fields_survive_json_roundtrip(store, tmp_path):
    """The surfaced fields are pure-JSON (the agent/CLI `--json` envelope) — name +
    strategy survive a `to_json()` round-trip with no None names / 0 count."""
    _require_propose_verb()
    res = _drive_propose(store, _bank_gap2_df(), tmp_path)
    payload = json.loads(res.to_json())
    fields = _fields_block(payload["data"])
    assert len(fields) > 0
    assert all((f.get("name") if isinstance(f, dict) else None) for f in fields)


def test_propose_data_fields_include_the_dollar_amount_field(store, tmp_path):
    """End-to-end tie of GAP 1 + GAP 2: a $-amount column appears in `data.fields`
    with name + the `amount` strategy (the fix surfaces the continuous amount field,
    not a None-named placeholder)."""
    _require_propose_verb()
    res = _drive_propose(store, _bank_gap2_df(), tmp_path)
    fields = _fields_block(res.data)
    amt = [
        f for f in fields
        if (f.get("source") if isinstance(f, dict) else getattr(f, "source", None)) == "amount"
        or (f.get("name") if isinstance(f, dict) else getattr(f, "name", None)) == "amount"
    ]
    assert amt, "the $-amount field must be surfaced in data.fields"
    strat = amt[0].get("strategy") if isinstance(amt[0], dict) else getattr(amt[0], "strategy")
    assert strat == "amount"


# ===========================================================================
# Regression — neither GAP fix may perturb the financial conformance oracle:
# vocab 6251 + byte-identical vocab_hash + dual-driver byte-identity.
# ===========================================================================


def test_financial_preset_vocab_6251_and_hash_byte_identical():
    """The financial preset is byte-identical (vocab 6251 + the retrain-trigger
    vocab_hash) — the GAP fixes touch only the BYO (custom) path, never the preset."""
    ct = compile_spec(financial_spec())
    assert ct.vocab_size == FINANCIAL_VOCAB_SIZE
    assert len(ct.vocab) == FINANCIAL_VOCAB_SIZE
    assert ct.report.passed
    assert ct.vocab_hash == FINANCIAL_VOCAB_HASH, (
        "financial vocab_hash drifted — a GAP fix perturbed a locked preset id"
    )
    # tokens/chunk grammar unchanged; the time-delta variant is still 6283.
    assert ct.tokens_per_txn == 12
    assert ct.chunk_size == 4096 // (12 + 1)  # == 315
    assert compile_spec(financial_spec(include_time_delta=True)).vocab_size == 6283


def test_financial_amount_step_strips_currency_unchanged():
    """The financial preset's amount preprocess still strips `$`/`,` and bins to the
    reference 0..6 — the same currency-stripping the BYO `amount` recipe mirrors, so
    GAP 1's recipe change did not disturb the preset path."""
    from loom.engine.spec import _amount_bin

    s = pd.Series(["$12.50", "$1,500.00", "$0.99", "$87.20", "$4,800.00"])
    out = _amount_bin(s).tolist()
    # thresholds [10,50,100,500,1000,5000]: count of thresholds crossed →
    # 12.50→1, 1500→5 (10/50/100/500/1000), 0.99→0, 87.20→2, 4800→5.
    assert out == [1, 5, 0, 2, 5]


def test_financial_dual_driver_is_byte_identical(tmp_path, monkeypatch):
    """The dual-driver tokenize stays byte-identical across the two faces (CLI
    `--json` == agent `dispatch`) for the financial preset — unperturbed by the GAP
    fixes (DESIGN.md §2.1 narrow waist)."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    store = ObjectStore(str(tmp_path))
    args = {"in": "IngestDataset/1", "preset": "financial"}

    cli_ctx = VerbContext(store=store, driver="cli", interactive=True)
    try:
        cli_result = REGISTRY["tokenize"].fn(dict(args), cli_ctx)
        tool_result = dispatch("loom.tokenize", dict(args))
    except NotImplementedError:  # pragma: no cover - a seam is still a stub
        pytest.skip("a verb seam is still a stub (NotImplementedError)")

    assert cli_result.to_json() == tool_result.to_json(), (
        "tokenize financial: CLI --json envelope diverged from the agent tool result"
    )


@pytest.fixture
def store(tmp_path, monkeypatch) -> ObjectStore:
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    return ObjectStore(str(tmp_path))
