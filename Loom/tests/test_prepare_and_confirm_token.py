"""Implement C — the generic ``prepare`` verb + ``tokenize`` bound alias + the
confirm-token HMAC (ARCHITECTURE §8 recast, §10 steps 3 & 5).

Three conformance gates:
  1. ``prepare`` (default) and ``tokenize`` compile the IDENTICAL event-sequence
     corpus (financial vocab 6251); ``tokenize`` stays byte-identical to v0.1 on
     both faces (the locked dual-driver invariant) and ``prepare`` is byte-identical
     except for the ``verb`` field.
  2. The write-refusal is CONTRACT-NAME-AGNOSTIC: a representation that reports any
     ERROR-severity Diagnostic (a C1 collision; a C3 grammar fail; a wholly novel
     contract name) is refused with NO Corpus written and its named diffs on the
     envelope — the verb never names a contract in the decision.
  3. The confirm-token: mint → validate → replay-rejected → expired-rejected →
     wrong-plan-rejected (HMAC over plan_hash+expiry+nonce; single-use; 15-min).
"""

from __future__ import annotations

import json

import pytest

from loom import REGISTRY
from loom.registry import VerbContext
from loom.store import ObjectStore
from loom.tools import dispatch, make_confirm_token, validate_confirm_token
from loom.types import Diagnostic, Severity, Status, Verdict


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ctx(store, **kw) -> VerbContext:
    kw.setdefault("driver", "cli")
    kw.setdefault("interactive", True)
    return VerbContext(store=store, **kw)


# ===========================================================================
# 1. prepare ≡ tokenize ≡ event-sequence; dual-driver byte-identity
# ===========================================================================


def test_tokenize_and_prepare_compile_identical_financial_corpus(tmp_path, monkeypatch):
    """``prepare`` (default representation) and ``tokenize`` compile the same
    financial corpus — vocab 6251, identical signature, identical persisted Corpus
    pathspec (content-addressed) — differing only in the envelope ``verb`` field."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    store = ObjectStore(str(tmp_path))

    tk = json.loads(REGISTRY["tokenize"].fn({"preset": "financial"}, _ctx(store)).to_json())
    pr = json.loads(REGISTRY["prepare"].fn({"preset": "financial"}, _ctx(store)).to_json())

    assert tk["verb"] == "tokenize" and pr["verb"] == "prepare"
    assert tk["data"]["vocab_size"] == pr["data"]["vocab_size"] == 6251
    assert tk["data"]["vocab_hash"] == pr["data"]["vocab_hash"]
    # Content-addressed: both wrote (or hit) the SAME Corpus object.
    assert tk["data"]["wrote_corpus"] and pr["data"]["wrote_corpus"]
    assert tk["data"]["pathspec"] == pr["data"]["pathspec"]
    # Everything except the verb field is identical between the two faces.
    tk.pop("verb"), pr.pop("verb")
    assert json.dumps(tk) == json.dumps(pr)


@pytest.mark.parametrize(
    "args",
    [
        {"in": "IngestDataset/1", "preset": "financial"},
        {"in": "IngestDataset/1", "preset": "financial", "include_time_delta": True},
        {"preset": "chain"},
        {"preset": "financial", "amount_strategy": "quantile"},
        {"preset": "financial", "drop_step": "cust"},
    ],
)
def test_tokenize_dual_driver_byte_identical(args, tmp_path, monkeypatch):
    """``loom tokenize --json`` (CLI fn) == ``dispatch("loom.tokenize", …)`` (agent),
    char-for-char — the locked dual-driver invariant, preserved through the alias."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    store = ObjectStore(str(tmp_path))
    cli = REGISTRY["tokenize"].fn(dict(args), _ctx(store)).to_json()
    tool = dispatch("loom.tokenize", dict(args)).to_json()
    assert cli == tool


def test_prepare_dual_driver_byte_identical(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    store = ObjectStore(str(tmp_path))
    args = {"representation": "event-sequence", "preset": "chain"}
    cli = REGISTRY["prepare"].fn(dict(args), _ctx(store)).to_json()
    tool = dispatch("loom.prepare", dict(args)).to_json()
    assert cli == tool


def test_tokenize_schema_has_no_representation_flag():
    """The ``tokenize`` agent tool schema is the v0.1 schema verbatim — no
    ``representation`` flag leaks onto it (it is pinned in the binding)."""
    assert "representation" not in REGISTRY["tokenize"].params["properties"]
    assert "representation" in REGISTRY["prepare"].params["properties"]


def test_tokenize_still_registered_for_the_agent_face():
    """``dispatch("loom.tokenize")`` must resolve a REGISTRY entry (the alias is a
    real registration, not just an argparse alias) so the agent face is unbroken."""
    assert "tokenize" in REGISTRY
    assert REGISTRY["tokenize"].name == "tokenize"


# ===========================================================================
# 2. the contract-name-agnostic write-refusal (the §8 fix)
# ===========================================================================


def test_c3_grammar_refusal_no_corpus_and_identical_diagnostics(tmp_path, monkeypatch):
    """A C3 grammar fail (reachable via ``context_len=5`` → chunk_size 0) is
    refused with NO Corpus written and the port's named diffs on the envelope —
    the status/verdict/diagnostics/data blocks AND the summary string are
    byte-identical to v0.1 (invariant #3), because the event-sequence port supplies
    the optional ``refusal_summary`` hook that reconstructs the v0.1 wording from its
    OWN already-refused cards. The write-gate stays name-agnostic regardless."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    store = ObjectStore(str(tmp_path))
    r = REGISTRY["tokenize"].fn({"preset": "financial", "context_len": 5}, _ctx(store))
    assert r.status is Status.REFUSED_CONTRACT
    assert r.verdict is Verdict.FAIL
    assert store.list("Corpus") == [], "a refused prepare must not persist a Corpus"
    assert r.data["wrote_corpus"] is False
    # The port's own C3 ERROR card is surfaced on the envelope.
    c3 = [d for d in r.diagnostics if d.contract == "C3" and d.severity is Severity.ERROR]
    assert c3, "the C3 grammar ERROR diagnostic must travel on the refusal envelope"


# v0.1 ``tokenize.py`` REFUSED summary strings (git c2d16d2), captured verbatim as
# the golden the new generic ``prepare`` must reproduce on the event-sequence
# representation (invariant #3 — the REFUSED envelope, not just the PASS one).
_V01_C3_SUMMARY = (
    "REFUSED_CONTRACT: C3 grammar failed — chunk_size=0 holds less than one "
    "transaction; no Corpus written."
)
_V01_C1_SUMMARY = (
    "REFUSED_CONTRACT: C1 injectivity/density failed for preset 'financial' — no "
    "Corpus written (the named diff explains the collision; reordering shifts every "
    "id ⇒ vocab_hash changes ⇒ retrain required)."
)


@pytest.mark.parametrize("context_len", [5, 12])
def test_c3_refused_summary_byte_identical_to_v01(context_len, tmp_path, monkeypatch):
    """The C3-grammar REFUSED *summary* (reachable through the LOCKED tokenize arg
    surface, e.g. ``--context_len 12``) is byte-identical to v0.1 on BOTH faces.

    This is the review-finding fix: the write-gate is name-agnostic, but the
    message-rendering helper offers the port a cosmetic ``refusal_summary`` hook so
    the event-sequence rep mirrors its v0.1 wording verbatim (``chunk_size=0``, as
    v0.1 interpolated from the same small-context compile)."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    store = ObjectStore(str(tmp_path))
    args = {"preset": "financial", "context_len": context_len}
    r = REGISTRY["tokenize"].fn(dict(args), _ctx(store))
    assert r.status is Status.REFUSED_CONTRACT
    assert r.summary == _V01_C3_SUMMARY
    # Dual-driver: the REFUSED envelope (incl. this summary) is byte-identical
    # across the CLI fn and the agent dispatch — invariant #3 on the refusal case.
    cli = REGISTRY["tokenize"].fn(dict(args), _ctx(store)).to_json()
    tool = dispatch("loom.tokenize", dict(args)).to_json()
    assert cli == tool


def test_c1_refused_summary_byte_identical_to_v01():
    """The C1-injectivity/density REFUSED *summary* mirrors v0.1 verbatim. C1 is not
    reachable through the clean-compile arg surface, so we exercise the renderer
    directly via the event-sequence port's ``refusal_summary`` hook over a forged C1
    ERROR card — proving the cosmetic hook reproduces the v0.1 C1 wording too."""
    from loom.adapters.event_sequence import EventSequenceRepresentation
    from loom.engine import compile_spec
    from loom.engine.spec import financial_spec

    rep = EventSequenceRepresentation()
    compiled = compile_spec(financial_spec(merchant_hash_size=2000), context_len=4096)
    forged_c1 = [
        Diagnostic(
            contract="C1",
            severity=Severity.ERROR,
            message="C1 injectivity FAIL: id 17 is claimed by both 'MONTH_12' and 'CARD_0'.",
            fix="lay step blocks out at 0-based local indices.",
        )
    ]
    assert rep.refusal_summary(compiled, forged_c1) == _V01_C1_SUMMARY


class _CollidingRepr:
    """A stub representation whose contracts ALWAYS report an ERROR — under an
    arbitrary, novel contract name. It proves the verb's write-gate is purely an
    ERROR-scan: it never names C1/C3 and gates a representation it has never heard
    of. ``compile`` returns a tiny duck-typed object the verb only reads for
    ``vocab_size``/``vocab_hash``/grammar in the PASS path (never reached here)."""

    name = "colliding-probe"
    produces_tensor_contract = "probe/none"

    def build_spec(self, args):
        return {"args": dict(args)}

    def compile(self, spec, *, context_len):
        return {"spec": spec, "context_len": context_len}

    def contracts(self, compiled):
        return [
            Diagnostic(
                contract="X9-INJECTIVITY",  # a name the verb has never seen
                severity=Severity.ERROR,
                message="probe: ids MONTH_12 and CARD_0 collide on the same id 17",
                fix="lay blocks out at 0-based local indices.",
                data={"colliding": ["MONTH_12", "CARD_0"], "id": 17},
            )
        ]

    def representation_passed(self, compiled):
        return not any(d.severity is Severity.ERROR for d in self.contracts(compiled))

    def signatures(self, compiled):  # never reached on the refusal path
        return {"representation": self.name, "representation_signature": "probe"}


def test_collision_refused_under_a_novel_contract_name(tmp_path, monkeypatch):
    """The write-gate refuses ANY representation reporting an ERROR — even under a
    contract name the verb has never seen — with no Corpus, the named diff carried
    through, and the verb never naming a contract in its decision."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    store = ObjectStore(str(tmp_path))
    from loom.ports import REPRESENTATIONS

    probe = _CollidingRepr()
    REPRESENTATIONS[probe.name] = probe
    try:
        r = REGISTRY["prepare"].fn({"representation": probe.name}, _ctx(store))
    finally:
        REPRESENTATIONS.pop(probe.name, None)

    assert r.status is Status.REFUSED_CONTRACT
    assert r.verdict is Verdict.FAIL
    assert store.list("Corpus") == []
    assert r.data["wrote_corpus"] is False
    # The novel-named ERROR card with its colliding-token diff is on the envelope.
    x9 = [d for d in r.diagnostics if d.contract == "X9-INJECTIVITY"]
    assert x9 and "MONTH_12" in (x9[0].message or "")
    # And the contract name appears in the summary ONLY because it came from the
    # port's own card — the verb surfaced it generically, never hardcoded it.
    assert "X9-INJECTIVITY" in r.summary


def test_unknown_representation_is_refused_no_corpus(tmp_path, monkeypatch):
    """An unregistered representation name → REFUSED_CONTRACT, a named
    ``REPRESENTATION`` diagnostic, and no Corpus."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    store = ObjectStore(str(tmp_path))
    r = REGISTRY["prepare"].fn({"representation": "no-such-rep"}, _ctx(store))
    assert r.status is Status.REFUSED_CONTRACT
    assert store.list("Corpus") == []
    assert any(d.contract == "REPRESENTATION" for d in r.diagnostics)


# ===========================================================================
# 3. the confirm-token lifecycle (HMAC; single-use; 15-min; plan-scoped)
# ===========================================================================


def test_confirm_token_mint_validate_replay_expire_wrongplan(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    plan_a = "planhash-AAAA"
    plan_b = "planhash-BBBB"

    # mint → validate (first use succeeds, burns the nonce).
    tok = make_confirm_token(plan_a)
    assert validate_confirm_token(tok, plan_a) is True

    # replay → rejected (single-use: the nonce is burned).
    assert validate_confirm_token(tok, plan_a) is False

    # wrong-plan → rejected (plan-hash-scoped; a fresh token for A fails against B).
    tok2 = make_confirm_token(plan_a)
    assert validate_confirm_token(tok2, plan_b) is False
    # ...and that rejection did NOT burn the nonce: it still validates for A.
    assert validate_confirm_token(tok2, plan_a) is True


def test_confirm_token_rejects_tampered_and_malformed(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    plan = "planhash-CCCC"
    tok = make_confirm_token(plan)
    parts = tok.split(".")
    assert len(parts) == 4  # plan_hash . expiry . nonce . hmac

    # tampered HMAC → rejected (unforgeable).
    tampered = ".".join(parts[:3] + ["deadbeef" * 8])
    assert validate_confirm_token(tampered, plan) is False
    # malformed → rejected, no crash.
    assert validate_confirm_token("not-a-token", plan) is False
    assert validate_confirm_token(None, plan) is False
    assert validate_confirm_token("", plan) is False


def test_confirm_token_rejects_expired(tmp_path, monkeypatch):
    """A token whose expiry has passed is rejected (15-min window). We forge an
    expiry in the past using the SAME HMAC the minter uses, so only the clock — not
    the signature — gates it."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    import time as _time

    from loom import tools

    plan = "planhash-DDDD"
    past = int(_time.time()) - 1
    nonce = "feedface" * 4
    mac = tools._confirm_mac(plan, past, nonce)
    expired = f"{plan}.{past}.{nonce}.{mac}"
    assert validate_confirm_token(expired, plan) is False
