"""Bring-your-own-schema field-map → TokenizerSpec compile path (Task B).

A first-time user with their OWN tabular schema declares a field→strategy
field-map (the SpecDraft the ``propose`` verb emits) and ``tokenize --spec`` must
compile THAT field-map into a contract-checked Corpus — ARBITRARY columns flow
through the EXISTING ``compile_spec`` + C1/C2/C3 unchanged.

These tests pin:
  1. the brief's bank schema (account_id, txn_amount, mcc, channel, dr_cr,
     balance, txn_ts) compiles to the documented tokens_per_txn=9, chunk_size=409,
     and the hand-counted vocab — with the entity EXCLUDED (T2);
  2. the financial PRESET path is BYTE-UNCHANGED (still vocab 6251, and the
     adapter's ``_build_spec`` returns the identical preset spec when ``--spec``
     is absent) — the dual-driver regression / HARD INVARIANT #1;
  3. a deliberately-colliding field-map FAILS C1 with the named diagnostic (the
     contracts are the safety net / HARD INVARIANT #2);
  4. the entity / target columns can NEVER be smuggled back in as a field step
     (HARD INVARIANT #4).
"""

from __future__ import annotations

import json
import textwrap

import pandas as pd
import pytest

from loom.adapters.event_sequence import _build_spec
from loom.engine import compile_spec, financial_spec, spec_from_field_map
from loom.engine.spec import materialize_corpus_lines


# ── The brief's bank schema field-map (the worked example, Ground-1/F) ──────


def _bank_field_map() -> dict:
    """account_id, txn_amount, mcc, channel, dr_cr, balance, txn_ts (entity=account_id)."""
    return {
        "entity": "account_id",      # EXCLUDED from the vocab (T2)
        "event": "txn",
        "context_len": 4096,
        "corpus_events": 50_000,
        "fields": [
            {"name": "amt", "source": "txn_amount", "strategy": "amount", "bins": 8},
            {"name": "mcc", "source": "mcc", "strategy": "hash", "buckets": 4096},
            {
                "name": "chan",
                "source": "channel",
                "strategy": "mapping",
                "values": ["POS", "ONLINE", "ATM"],
                "default": "UNK",
            },
            {"name": "drcr", "source": "dr_cr", "strategy": "fixedvocab", "min": 0, "max": 1},
            {"name": "bal", "source": "balance", "strategy": "amount", "bins": 8},
            {"name": "ts", "source": "txn_ts", "strategy": "calendar"},  # → HOUR+DOW+MONTH
            {"name": "gap", "source": "txn_ts", "strategy": "timedelta", "bins": 32},
        ],
    }


def _bank_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_id": ["a", "a", "b", "b"],
            "txn_amount": [12.5, 4.0, 1500.0, 0.99],
            "mcc": [5411, 5814, 4111, 5942],
            "channel": ["POS", "ONLINE", "ATM", "ONLINE"],
            "dr_cr": [0, 1, 0, 1],
            "balance": [100.0, 96.0, 8500.0, 8499.0],
            "txn_ts": [
                "2026-01-02 09:15",
                "2026-01-02 13:40",
                "2026-02-14 18:05",
                "2026-03-30 23:59",
            ],
        }
    )


def test_bank_field_map_compiles_to_expected_vocab_and_chunk():
    """The brief's worked example: tokens_per_txn=9, chunk_size=4096//10=409, and
    the hand-counted vocab; C1/C2/C3 all pass on the compiled custom spec."""
    ct = compile_spec(spec_from_field_map(_bank_field_map()))

    assert ct.spec.preset == "custom"  # NOT financial/chain (HARD INVARIANT #1)
    assert ct.tokens_per_txn == 9, "amt,mcc,chan,drcr,bal + (hour,dow,month) + gap"
    assert ct.chunk_size == 409, "4096 // (9 + 1)"

    # Hand-count (Ground-1/E): 5 specials + 8 amt + 4096 mcc + (3+1 default) chan
    # + 2 drcr + 8 bal + 24 hour + 7 dow + 12 month + 32 gap.
    expected = 5 + 8 + 4096 + 4 + 2 + 8 + 24 + 7 + 12 + 32
    assert ct.vocab_size == expected == 4198
    assert len(ct.vocab) == ct.vocab_size

    # C1: dense + injective; C2: deterministic (no fitted artifact); C3: grammar OK.
    ids = list(ct.vocab.values())
    assert set(ids) == set(range(ct.vocab_size))
    assert ct.report.passed and ct.report.injective and ct.report.dense
    assert ct.report.has_fitted_artifact is False


def test_bank_field_map_excludes_entity_from_vocab():
    """HARD INVARIANT #4: the entity (account_id) NEVER earns a token."""
    ct = compile_spec(spec_from_field_map(_bank_field_map()))
    assert not any(tok.startswith("ACCOUNT") for tok in ct.vocab)
    assert ct.spec.entity == "account_id"
    # The documented calendar + threshold tokens DO appear.
    for tok in ("AMT_0", "AMT_7", "DRCR_0", "HOUR_00", "HOUR_23", "DOW_0", "MONTH_00", "GAP_0"):
        assert tok in ct.vocab, tok
    # MONTH is 0-based (the MONTH_12 ≡ CARD_0 collision fix) → no MONTH_12.
    assert "MONTH_11" in ct.vocab and "MONTH_12" not in ct.vocab


def test_bank_field_map_materializes_a_valid_corpus():
    """The compiled custom spec materializes corpus lines over a stranger's frame:
    entity-grouped, calendar tokens + per-entity inter-event gap, NO entity token."""
    ct = compile_spec(spec_from_field_map(_bank_field_map()))
    lines, n_txns = materialize_corpus_lines(ct, _bank_df())
    assert n_txns == 4
    # Two entities (a, b) → two grouped lines (each < chunk_size txns).
    assert len(lines) == 2
    blob = " ".join(lines)
    assert "<bos>" in blob and "<eos>" in blob and "<sep>" in blob
    # Field tokens present; the entity value is NOT emitted as a token.
    assert "CHAN_POS" in blob and "DRCR_0" in blob and "HOUR_09" in blob
    assert " a " not in blob and " b " not in blob


# ── The PRESET path is byte-unchanged (the dual-driver regression) ──────────


def test_financial_preset_unchanged_vocab_6251():
    """HARD INVARIANT #1: the financial preset still compiles to vocab 6251."""
    ct = compile_spec(financial_spec())
    assert ct.vocab_size == 6251
    assert ct.report.passed


def test_adapter_build_spec_preset_path_byte_identical_without_spec():
    """``_build_spec`` with NO ``--spec`` returns the IDENTICAL preset spec it
    returned before the additive ``--spec`` branch existed — same steps, same
    preset, same compiled vocab_hash."""
    # financial (the default) and chain presets, exactly as the verb passes them.
    fin = _build_spec({"preset": "financial"})
    assert fin.preset == "financial"
    fin_ct = compile_spec(fin)
    assert fin_ct.vocab_size == 6251

    # The compiled vocab_hash is the retrain-trigger signature; pin it equals the
    # direct factory's, proving the no-spec branch did not perturb the preset spec.
    direct_ct = compile_spec(financial_spec())
    assert fin_ct.vocab_hash == direct_ct.vocab_hash
    assert fin.step_names() == financial_spec().step_names()

    chain = _build_spec({"preset": "chain"})
    assert chain.preset == "chain"
    assert compile_spec(chain).vocab_hash == compile_spec(__import__(
        "loom.engine", fromlist=["chain_spec"]).chain_spec()).vocab_hash


def test_default_no_args_is_still_financial_preset():
    """No preset, no spec → the financial conformance oracle, unchanged."""
    spec = _build_spec({})
    assert spec.preset == "financial"
    assert compile_spec(spec).vocab_size == 6251


# ── --spec routes to the custom path (additive); file + dict both resolve ───


def test_adapter_routes_spec_dict_to_custom_path():
    """A ``--spec`` dict (the verb-resolved field-map) compiles to a custom spec,
    NOT a preset — and is contract-checked through the same compile gate."""
    spec = _build_spec({"spec": _bank_field_map()})
    assert spec.preset == "custom"
    ct = compile_spec(spec)
    assert ct.tokens_per_txn == 9 and ct.chunk_size == 409 and ct.report.passed


def test_adapter_resolves_spec_from_yaml_and_json_files(tmp_path):
    """``--spec <file>`` resolves a YAML or JSON field-map file to the same spec."""
    fm = {
        "entity": "account_id",
        "event": "txn",
        "context_len": 4096,
        "fields": [
            {"name": "amt", "source": "txn_amount", "strategy": "amount", "bins": 8},
            {"name": "drcr", "source": "dr_cr", "strategy": "fixedvocab", "min": 0, "max": 1},
            {"name": "ts", "source": "txn_ts", "strategy": "calendar", "part": "hour"},
        ],
    }
    json_path = tmp_path / "fm.json"
    json_path.write_text(json.dumps(fm), encoding="utf-8")
    yaml_path = tmp_path / "fm.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """
            entity: account_id
            event: txn
            context_len: 4096
            fields:
              - {name: amt,  source: txn_amount, strategy: amount,    bins: 8}
              - {name: drcr, source: dr_cr,      strategy: fixedvocab, min: 0, max: 1}
              - {name: ts,   source: txn_ts,     strategy: calendar,   part: hour}
            """
        ),
        encoding="utf-8",
    )
    from_json = compile_spec(_build_spec({"spec": str(json_path)}))
    from_yaml = compile_spec(_build_spec({"spec": str(yaml_path)}))
    # amt(8) + drcr(2) + hour(24) + 5 specials.
    assert from_json.vocab_size == 5 + 8 + 2 + 24
    assert from_json.vocab_hash == from_yaml.vocab_hash
    assert from_json.tokens_per_txn == 3


# ── Generalization: a multi-timestamp schema compiles out of the box ────────


def test_two_datetime_columns_get_disjoint_calendar_blocks():
    """A schema with TWO datetime columns (created_at + updated_at — very common)
    must compile: each ``calendar`` field gets a DISJOINT token block. The token
    prefix is uniquified per field (``HOUR`` → ``HOUR1`` for the second timestamp)
    so the two HOUR/DOW/MONTH blocks don't collide on C1 injectivity — the
    stranger's multi-timestamp flow no longer dead-ends out of the box."""
    fm = {
        "entity": "acct",
        "event": "ev",
        "context_len": 4096,
        "fields": [
            {"name": "created", "source": "created_at", "strategy": "calendar"},
            {"name": "updated", "source": "updated_at", "strategy": "calendar"},
        ],
    }
    ct = compile_spec(spec_from_field_map(fm))
    assert ct.report.passed and ct.report.injective and ct.report.dense
    # 6 steps (2 timestamps × HOUR+DOW+MONTH), disjoint prefixes for the second ts.
    assert ct.tokens_per_txn == 6
    hour_prefixes = {t.rsplit("_", 1)[0] for t in ct.vocab if t.startswith("HOUR")}
    assert hour_prefixes == {"HOUR", "HOUR1"}, hour_prefixes
    # vocab = 5 specials + 2 × (24 + 7 + 12) = 5 + 86 = 91.
    assert ct.vocab_size == 5 + 2 * (24 + 7 + 12)


def test_two_amount_columns_with_default_names_do_not_collide():
    """Two continuous columns whose AUTO-DERIVED prefixes would clash (both named
    ``amt``) get disjoint blocks — auto-derived prefixes are uniquified."""
    fm = {
        "entity": "acct",
        "fields": [
            {"name": "amt", "source": "fee", "strategy": "amount", "bins": 8},
            {"name": "amt", "source": "total", "strategy": "amount", "bins": 8},
        ],
    }
    ct = compile_spec(spec_from_field_map(fm))
    assert ct.report.passed and ct.report.injective
    assert ct.tokens_per_txn == 2


# ── A bad field-map is REFUSED by the contracts (the safety net) ────────────


def _colliding_field_map() -> dict:
    """Two field steps share the SAME prefix with overlapping ranges → duplicate
    token strings (e.g. ``DUP_3``) → C1 must refuse (HARD INVARIANT #2)."""
    return {
        "entity": "acct",
        "event": "ev",
        "fields": [
            {"name": "a", "source": "x", "strategy": "fixedvocab", "prefix": "DUP", "min": 0, "max": 5},
            {"name": "b", "source": "y", "strategy": "fixedvocab", "prefix": "DUP", "min": 3, "max": 8},
        ],
    }


def test_colliding_field_map_fails_c1_with_named_diagnostic():
    """A bad stranger-schema spec is REFUSED with the named C1 diff, never a
    silent-broken corpus — the compiled custom spec runs the SAME contracts."""
    ct = compile_spec(spec_from_field_map(_colliding_field_map()))
    assert ct.report.passed is False, "a colliding field-map must not pass C1"
    assert ct.report.injective is False
    c1 = [d for d in ct.report.diagnostics if d.contract == "C1"]
    assert c1, "expected a named C1 diagnostic"
    assert any(d.severity.value == "error" for d in c1)
    blob = " ".join((d.message or "") + " " + str(d.data) for d in c1)
    assert "DUP" in blob, f"C1 must name the colliding token(s): {blob!r}"


def test_colliding_field_map_routes_through_adapter_and_still_fails_c1():
    """The same refusal holds via the adapter's ``--spec`` path (the verb's gate)."""
    ct = compile_spec(_build_spec({"spec": _colliding_field_map()}))
    assert ct.report.passed is False
    assert any(d.contract == "C1" and d.severity.value == "error" for d in ct.report.diagnostics)


# ── The entity / target can never be smuggled back in (HARD INVARIANT #4) ───


def test_hand_edited_field_map_cannot_tokenize_the_entity():
    with pytest.raises(ValueError, match="entity"):
        spec_from_field_map(
            {"entity": "acct", "fields": [{"name": "acct", "source": "acct", "strategy": "hash"}]}
        )


def test_hand_edited_field_map_cannot_tokenize_the_target():
    with pytest.raises(ValueError, match="target"):
        spec_from_field_map(
            {
                "entity": "acct",
                "target": "is_fraud",
                "fields": [
                    {"name": "lbl", "source": "is_fraud", "strategy": "fixedvocab", "min": 0, "max": 1}
                ],
            }
        )
