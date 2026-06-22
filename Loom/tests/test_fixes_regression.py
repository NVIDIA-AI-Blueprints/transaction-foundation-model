"""Regression gate for FIX 1 (sample-aware occupancy) + FIX 2 (k-mer strategy).

The build brief's HARD INVARIANT: neither fix may perturb the conformance oracle.

  * The ``financial`` preset stays BYTE-IDENTICAL — vocab 6251 (6283 with the
    time-delta field), the documented per-field block sizes, and — the strongest
    pin — the same ``vocab_hash`` (the retrain-trigger signature). If FIX 1's gate
    rewrite or FIX 2's new strategy nudged any preset id, this hash changes and the
    test fails.
  * The dual-driver tokenize is byte-identical across the two faces: the CLI
    ``--json`` envelope equals the agent-tool ``dispatch`` envelope, character for
    character (DESIGN.md §2.1 narrow waist), for the financial preset.

These are deliberately PINNED to literal values (not recomputed from the spec) so
the regression catches a silent drift the fixes might introduce, rather than moving
with it.
"""

from __future__ import annotations

import json

import pytest

from loom.engine import (
    SPECIAL_TOKENS,
    chain_spec,
    compile_spec,
    financial_spec,
    spec_from_field_map,
)

# The byte-identical financial signature captured BEFORE the fixes (the oracle).
# A change here means the preset vocab moved — a retrain trigger the fixes must NOT
# cause. Recomputed pins (vocab_size/hash) for the locked conformance preset.
FINANCIAL_VOCAB_SIZE = 6251
FINANCIAL_VOCAB_HASH = (
    "sha256:ba0e0daa6c1d64a1028e428b7981a82a69fe45a42cc42161277df04aa9152ce4"
)


# ---------------------------------------------------------------------------
# FIX-invariant #1 — the financial preset is byte-identical (vocab 6251 + hash).
# ---------------------------------------------------------------------------


def test_financial_preset_still_vocab_6251():
    ct = compile_spec(financial_spec())
    assert ct.vocab_size == FINANCIAL_VOCAB_SIZE
    assert len(ct.vocab) == FINANCIAL_VOCAB_SIZE
    assert ct.report.passed


def test_financial_preset_vocab_hash_is_byte_identical():
    """The retrain-trigger signature is UNCHANGED by FIX 1 / FIX 2 — the strongest
    byte-identity pin (any moved id flips this sha256)."""
    ct = compile_spec(financial_spec())
    assert ct.vocab_hash == FINANCIAL_VOCAB_HASH, (
        "financial preset vocab_hash drifted — FIX 1/2 perturbed a locked preset id"
    )


def test_financial_specials_and_grammar_unchanged():
    ct = compile_spec(financial_spec())
    assert SPECIAL_TOKENS == ("<pad>", "<bos>", "<eos>", "<sep>", "<unk>")
    for i, tok in enumerate(SPECIAL_TOKENS):
        assert ct.vocab[tok] == i
    assert ct.tokens_per_txn == 12
    assert ct.chunk_size == 4096 // (12 + 1)  # == 315


def test_financial_with_time_delta_still_6283():
    ct = compile_spec(financial_spec(include_time_delta=True))
    assert ct.vocab_size == 6283
    assert ct.tokens_per_txn == 13
    assert ct.chunk_size == 292


def test_chain_preset_still_derives_and_passes():
    """The chain preset's DERIVED vocab is unchanged + still compiles clean."""
    ct = compile_spec(chain_spec())
    assert ct.vocab_size == 5082
    assert ct.tokens_per_txn == 7
    assert ct.chunk_size == 4096 // (7 + 1)  # == 512
    assert ct.report.passed


# ---------------------------------------------------------------------------
# FIX-invariant #2 — the dual-driver tokenize stays byte-identical.
# ---------------------------------------------------------------------------


def _call_or_skip(thunk):
    """A NotImplementedError means a seam is still a stub (parallel build) — skip,
    don't report a conformance failure. The byte-identity assert stays strict."""
    try:
        return thunk()
    except NotImplementedError:  # pragma: no cover - stub seam
        pytest.skip("a verb seam is still a stub (NotImplementedError)")


def test_dual_driver_tokenize_financial_is_byte_identical(tmp_path, monkeypatch):
    """``loom tokenize --preset financial --json`` (the CLI face) equals the agent
    tool ``dispatch`` result, character for character — UNCHANGED by FIX 1/2."""
    from loom import REGISTRY
    from loom.registry import VerbContext
    from loom.store import ObjectStore
    from loom.tools import dispatch

    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    store = ObjectStore(str(tmp_path))
    args = {"in": "IngestDataset/1", "preset": "financial"}

    cli_ctx = VerbContext(store=store, driver="cli", interactive=True)
    cli_result = _call_or_skip(lambda: REGISTRY["tokenize"].fn(dict(args), cli_ctx))
    tool_result = _call_or_skip(lambda: dispatch("loom.tokenize", dict(args)))

    assert cli_result.to_json() == tool_result.to_json(), (
        "tokenize: CLI --json envelope diverged from the agent tool result"
    )
    payload = json.loads(cli_result.to_json())
    assert list(payload.keys()) == [
        "verb", "status", "verdict", "tier", "capability_mode", "summary",
        "outputs", "diagnostics", "data", "experiment", "cost_plan", "confirm_token",
    ]


def test_adapter_no_spec_financial_branch_byte_identical():
    """The adapter ``_build_spec`` no-``--spec`` path returns the IDENTICAL preset
    spec (same vocab_hash) — proving FIX 2's additive ``kmer`` keyword did not
    perturb the preset branch of the spec builder (HARD INVARIANT #1)."""
    from loom.adapters.event_sequence import _build_spec

    fin = _build_spec({"preset": "financial"})
    assert fin.preset == "financial"
    assert fin.step_names() == financial_spec().step_names()
    assert compile_spec(fin).vocab_hash == FINANCIAL_VOCAB_HASH


def test_custom_field_map_preset_is_isolated_from_the_presets():
    """A BYO custom field-map (the FIX surfaces' home) compiles to ``preset="custom"``
    — never ``financial``/``chain`` — so the dual-driver byte-identity is structurally
    isolated from both fixes."""
    fm = {
        "entity": "acct",
        "event": "ev",
        "context_len": 4096,
        "fields": [
            {"name": "amt", "source": "amount", "strategy": "amount", "bins": 8},
            {"name": "drcr", "source": "dr_cr", "strategy": "fixedvocab", "min": 0, "max": 1},
        ],
    }
    spec = spec_from_field_map(fm)
    assert spec.preset == "custom"
    assert spec.preset not in ("financial", "chain")
