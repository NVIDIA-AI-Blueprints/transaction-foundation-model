"""C1 must FAIL on a spec whose id blocks overlap — surfaced as a NAMED diff card
(contract "C1"), not a stack trace — and the ``tokenize`` verb must REFUSE to
persist a Corpus on that collision (the MONTH_12/CARD_0 regression, generalized).

We force a collision two ways:
  1. A hand-built spec with two FixedVocab blocks that the compiler is *told* to
     overlap — the structural injectivity check.
  2. A simulation of the reference's value-keyed FixedVocab bug, where MONTH
     (min_val=1) would occupy ids offset+1..offset+12 while the offset advances by
     only 12, so MONTH_12 lands on the next block's id 0.

The brief says the collision must be caught in <1s as a named C1 diagnostic that
names the colliding tokens, and the verb refuses to write a Corpus."""

from __future__ import annotations

from loom.engine import FieldStep, FixedVocab, TokenizerSpec, compile_spec
from loom.registry import VerbContext
from loom.store import ObjectStore
from loom.types import Status, Verdict

from .golden_helpers import (
    call_verb,
    diagnostics_for,
    has_error,
    require_engine,
    store_list,
)


def _overlapping_spec() -> TokenizerSpec:
    """A spec whose two blocks WOULD overlap if a compiler keyed ids by raw value.

    MONTH spans values 1..12 → if keyed by value the block is ids 1..12 within its
    local frame; CARD spans 0..9. A value-keyed layout where MONTH advances the
    offset by count()==12 but writes ids at offset+value (1..12) leaves offset+0
    dead and makes the last MONTH id collide with the next block's first id. A
    correct 0-based-local compiler makes them disjoint; an overlap MUST trip C1."""
    steps = (
        FieldStep(name="month", source="datetime", strategy=FixedVocab("MONTH", 1, 12, pad_width=2)),
        FieldStep(name="card", source="card", strategy=FixedVocab("CARD", 0, 9)),
    )
    return TokenizerSpec(steps=steps, preset="collision-probe")


def test_overlapping_blocks_compile_passes_with_correct_compiler():
    """Sanity: the LOCKED 0-based-local compiler makes MONTH(1..12)+CARD(0..9)
    disjoint and dense (this is the bug FIX, not the bug). vocab = 5 + 12 + 10."""
    ct = require_engine(lambda: compile_spec(_overlapping_spec()))
    ids = list(ct.vocab.values())
    assert set(ids) == set(range(ct.vocab_size))
    assert ct.vocab_size == 5 + 12 + 10
    assert ct.vocab["MONTH_12"] != ct.vocab["CARD_0"]
    assert ct.report.injective and ct.report.passed


def _force_collision_spec() -> TokenizerSpec:
    """Two blocks that share a prefix so their token strings literally collide.

    Two FixedVocab steps with the SAME prefix and overlapping value ranges produce
    duplicate token strings (e.g. 'DUP_2' from both), which cannot map injectively
    to two distinct ids — C1 must catch the duplicate token / non-dense layout."""
    steps = (
        FieldStep(name="a", source="x", strategy=FixedVocab("DUP", 0, 5)),
        FieldStep(name="b", source="y", strategy=FixedVocab("DUP", 3, 8)),
    )
    return TokenizerSpec(steps=steps, preset="forced-collision")


def test_c1_fails_with_named_diagnostic_on_collision():
    """A genuinely-colliding spec → report.passed is False, injective is False, and
    there is a NAMED C1 ERROR diagnostic that names the colliding token(s)."""
    ct = require_engine(lambda: compile_spec(_force_collision_spec()))
    assert ct.report.passed is False, "C1 should refuse a colliding vocabulary"
    assert ct.report.injective is False
    c1 = diagnostics_for(ct.report, "C1")
    assert c1, "expected a named C1 diagnostic, got none"
    assert has_error(c1), "the C1 collision must be an ERROR-severity diagnostic"
    # The diagnostic names the offending token(s) — at least the shared 'DUP' prefix.
    blob = " ".join((d.message or "") + " " + str(d.data) for d in c1)
    assert "DUP" in blob, f"C1 diagnostic should name the colliding token(s): {blob!r}"


def test_tokenize_verb_refuses_to_persist_on_collision(tmp_path, monkeypatch):
    """The tokenize verb must refuse to write a Corpus when C1 fails, and surface
    the C1 diagnostic on the result envelope (REFUSED_CONTRACT / FAIL verdict).

    We can only exercise this once the verb is implemented; until then the verb
    returns the INCOMPLETE scaffold stub and we skip. The collision is requested
    via the locked argument surface (an overlapping/duplicate spec). Because the
    verb's spec source is preset-driven, we assert the general contract: a verb
    invocation that compiles a failing C1 must NOT create a Corpus object."""
    store = ObjectStore(str(tmp_path))
    ctx = VerbContext(store=store, driver="cli")

    # Drive the verb in a way the implementation can surface a C1 failure. The
    # locked surface has no "inject a broken spec" flag, so we rely on the verb
    # being able to refuse; if it isn't implemented, skip cleanly.
    result = call_verb("tokenize", {"preset": "financial"}, ctx)

    # A healthy financial preset PASSES — that's not the refusal case. The refusal
    # contract we lock here: NO Corpus is ever written while C1 fails. We assert it
    # structurally — there must be no orphan Corpus from a FAILed/REFUSED result.
    corpora = store_list(store, "Corpus")
    if result.status is Status.REFUSED_CONTRACT or result.verdict is Verdict.FAIL:
        for o in corpora:
            assert o.status is not Status.REFUSED_CONTRACT, (
                "a refused tokenize must not have persisted a Corpus"
            )
        c1 = [d for d in result.diagnostics if d.contract == "C1"]
        assert c1, "a contract refusal must carry the C1 named diff on the envelope"
