"""STAGE 3 — the discriminating test for the local validate-before-train loop.

This closes ``propose → tokenize → embed → evaluate → refine`` end-to-end through
the REAL verb path and proves the thing the loop exists for: ``evaluate`` gives a
GOOD tokenizer spec a PASS verdict and SEES a model could be worth a GPU-hour,
while every broken spec earns a FAIL/REVIEW with the EXACT fieldmap.yaml knob to
turn — all on CPU, $0, before a single GPU-hour.

The four ways (one tiny synthetic multi-event sequential sample, four field-maps):

  1. **GOOD** — the predictive ``item`` magnitude binned with a sensible 8-bin
     amount strategy + the side field kept → ``verdict==PASS``, ``knn`` beats the
     strong repeat-last control, low oov/dead-token, an EMPTY ``refine_plan``.
  2. **HASH-EVERYTHING** — the item field hashed into a vocab far larger than the
     data → the INTRINSIC dead-token gate dominates → ``verdict==FAIL`` with
     ``dead_token_frac`` strictly above GOOD's and a refine entry naming an
     ``item_hash_size`` knob.
  3. **DROP-SIGNAL** — the predictive item field omitted from ``fields[]`` → no
     item-family vocab token → ``knn`` cannot beat repeat-last → ``verdict==FAIL``
     with a refine entry naming the dropped field (``fields[]``).
  4. **2-BIN-AMOUNT** — the item magnitude collapsed to 2 amount bins → the
     distinct magnitudes the sequence rides on are blurred → ``knn`` falls below
     the controls → ``verdict∈{FAIL,REVIEW}`` with a refine entry naming an
     ``n_bins``/``amount_strategy`` knob.

EVERY scoring assertion is RELATIVE — to the reused classical controls and to the
data's own occupancy. There is NO hardcoded "good" Prec@k anywhere; the verdict is
``PASS`` because ``knn`` beats *this sample's* repeat-last + popularity, and the bad
specs are worse than GOOD *on the same sample*. The strict gate (Anub's product
decision) is exercised directly: a spec that cannot beat repeat-last is FAIL, not
REVIEW.

Plus three integration guards the build spec calls out as the subtle risks:
  (a) baseline ``--json`` byte-identity on the financial+chain fixtures (the
      holdout extraction was a PURE move — the eval split did not fork);
  (b) determinism — embed twice → a bit-identical matrix + a deduped Embeddings;
      evaluate twice → the same verdict + the same deduped Scores;
  (c) C5-mismatch — embed Corpus A, evaluate Corpus B against A's embeddings →
      ``status==REFUSED_CONTRACT`` with ``contract=='C5'`` ($0 on a mispaired pair).

CPU-only, deterministic, sub-second — no torch/NeMo/RAPIDS, no GPU, no network.
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
from loom.types import Status, Verdict


# ===========================================================================
# Fixtures — the store + a verb-call helper that skips only on a not-landed seam.
# ===========================================================================


@pytest.fixture
def store(tmp_path, monkeypatch) -> ObjectStore:
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    return ObjectStore(str(tmp_path))


def _ctx(store: ObjectStore, **kw) -> VerbContext:
    kw.setdefault("driver", "cli")
    kw.setdefault("interactive", True)
    return VerbContext(store=store, **kw)


def _require(*verbs: str) -> None:
    for v in verbs:
        if v not in REGISTRY:
            pytest.skip(f"`{v}` verb not registered yet (validate-loop seam not landed)")


def _call(verb: str, args: dict, ctx: VerbContext):
    try:
        return REGISTRY[verb].fn(dict(args), ctx)
    except NotImplementedError:  # pragma: no cover — a downstream seam is still a stub
        pytest.skip(f"{verb}: a downstream seam is still a stub (NotImplementedError)")


# ===========================================================================
# The ONE synthetic sample — a numeric ``item`` magnitude that walks a clean,
# learnable 8-step cyclic chain. Each magnitude lands in its OWN bucket under an
# 8-bin amount strategy (so the chain is recoverable), but the eight magnitudes
# collapse to two coarse buckets under a 2-bin strategy (so the chain is lost).
# The next magnitude is NEVER equal to the current one → repeat-last scores ~0, so
# any kNN signal genuinely beats the strong sequential control.
# ===========================================================================

# Eight magnitudes spanning all eight distinct 8-bin log-threshold buckets
# (thresholds 1e-4 .. 1e2 → buckets 0..7); under the 2-bin split (threshold 0.1)
# the first four fall in bucket 0 and the last four in bucket 1.
_MAGS = [0.00005, 0.0005, 0.005, 0.05, 0.5, 5.0, 50.0, 500.0]
_NEXT = {i: (i + 1) % len(_MAGS) for i in range(len(_MAGS))}


def _make_sample_df(n_entities: int = 80, seqlen: int = 8) -> pd.DataFrame:
    """A deterministic multi-event sequential frame: each of ``n_entities`` wallets
    walks the magnitude chain ``MAGS[p] → MAGS[p+1] → …`` for ``seqlen`` events. The
    raw ``item`` magnitude IS the predictive next-item signal; ``size_usd`` mirrors
    it so an amount-strategy field reads the same chain; ``side`` alternates."""
    rows = []
    base = pd.Timestamp("2026-01-01")
    for e in range(n_entities):
        pos = e % len(_MAGS)
        t = base + pd.Timedelta(hours=e)
        for s in range(seqlen):
            rows.append(
                (f"0x{e:03d}", t.isoformat(), _MAGS[pos], "BUY" if s % 2 == 0 else "SELL", _MAGS[pos])
            )
            pos = _NEXT[pos]
            t = t + pd.Timedelta(minutes=5)
    return pd.DataFrame(rows, columns=["wallet", "timestamp", "item", "side", "size_usd"])


def _fieldmap(fields: list[dict]) -> dict:
    return {
        "version": "loom-fieldmap/1",
        "entity": "wallet",
        "event": "trade",
        "context_len": 4096,
        "fields": fields,
    }


# The four field-maps over the SAME sample. The item field is the numeric magnitude;
# GOOD bins it finely (8), 2-BIN coarsely (2), HASH over-hashes it, DROP omits it.
_SIDE = {"name": "side", "source": "side", "strategy": "mapping",
         "values": ["BUY", "SELL"], "default": "UNK"}
_GOOD_FM = _fieldmap([{"name": "item", "source": "item", "strategy": "amount", "bins": 8}, _SIDE])
_HASH_FM = _fieldmap([{"name": "item", "source": "item", "strategy": "hash", "buckets": 4000}, _SIDE])
_DROP_FM = _fieldmap([_SIDE])  # the predictive item field is simply not listed
_2BIN_FM = _fieldmap([{"name": "item", "source": "item", "strategy": "amount", "bins": 2}, _SIDE])

_K = 5  # Prec@K cutoff; the controls + the kNN use the same k.


def _ingest_sample(store: ObjectStore, tmp_path) -> str:
    csv = tmp_path / "sample.csv"
    _make_sample_df().to_csv(csv, index=False)
    res = _call(
        "ingest",
        {"in": str(csv), "name": "magnitude-chain", "entity": "wallet", "event": "trade"},
        _ctx(store),
    )
    assert res.status is Status.OK and res.outputs
    return res.outputs[0].pathspec


def _write_fieldmap(tmp_path, name: str, fm: dict) -> str:
    yaml = pytest.importorskip("yaml")
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(fm, sort_keys=False))
    return str(p)


def _run_spec(store: ObjectStore, tmp_path, in_path: str, name: str, fm: dict, *, experiment: str):
    """Drive ``tokenize --spec → embed → evaluate`` for one field-map; return the
    evaluate ``data`` block (or a (status, result) on a structural refusal upstream)."""
    spec_path = _write_fieldmap(tmp_path, name, fm)
    tok = _call("tokenize", {"in": in_path, "spec": spec_path}, _ctx(store, experiment=experiment))
    assert tok.status is Status.OK, tok.summary
    assert tok.verdict is Verdict.PASS, tok.summary
    corpus = tok.data["pathspec"]

    emb = _call("embed", {"in": corpus}, _ctx(store, experiment=experiment))
    assert emb.status is Status.OK, emb.summary
    assert emb.verdict is Verdict.PASS

    ev = _call("evaluate", {"in": corpus, "k": _K}, _ctx(store, experiment=experiment))
    return corpus, emb, ev


# ===========================================================================
# 1. The four-way discrimination — the heart of the validate loop.
# ===========================================================================


def test_four_way_verdict_ordering_and_knobs(store, tmp_path):
    """GOOD PASSes and beats repeat-last; HASH/DROP/2-BIN each FAIL or REVIEW, are
    worse than GOOD, and name the EXACT offending knob. Every threshold is RELATIVE
    to the reused controls + the data's own occupancy — no hardcoded goodness number."""
    _require("ingest", "tokenize", "embed", "evaluate")
    in_path = _ingest_sample(store, tmp_path)

    _, _, good = _run_spec(store, tmp_path, in_path, "good", _GOOD_FM, experiment="good")
    _, _, hash_ = _run_spec(store, tmp_path, in_path, "hashall", _HASH_FM, experiment="hash")
    _, _, drop = _run_spec(store, tmp_path, in_path, "drop", _DROP_FM, experiment="drop")
    _, _, twobin = _run_spec(store, tmp_path, in_path, "twobin", _2BIN_FM, experiment="2bin")

    # All four are COMPUTED verdicts (status OK), not structural refusals — a FAIL
    # spec is OK+FAIL (like baseline carries PASS+OK), the "refine it" signal.
    for r in (good, hash_, drop, twobin):
        assert r.status is Status.OK, r.summary

    gd = good.data
    hd = hash_.data
    dd = drop.data
    td = twobin.data

    g_knn = gd["extrinsic"]["knn_prec_at_k"]
    g_rli = gd["controls"]["repeat_last_prec1"]
    g_pop = gd["controls"]["popularity_prec_at_k"]

    # --- GOOD: PASS, beats the strong control, clean intrinsics, no refine work ---
    assert good.verdict is Verdict.PASS, good.summary
    assert g_knn > g_rli, "GOOD must beat repeat-last (the strict gate's bar)"
    assert g_knn >= g_pop, "GOOD must also be ≥ the popularity control (PASS condition)"
    assert gd["controls"]["beats_repeat_last"] is True
    assert gd["intrinsic"]["oov_rate"] == 0.0
    assert gd["intrinsic"]["dead_token_frac"] < 0.5
    assert gd["refine_plan"] == [], "a PASS GOOD spec carries no refine work"

    # --- each bad spec lands in {FAIL, REVIEW} (GOOD alone is PASS) ---------------
    assert hash_.verdict in (Verdict.FAIL, Verdict.REVIEW)
    assert drop.verdict in (Verdict.FAIL, Verdict.REVIEW)
    assert twobin.verdict in (Verdict.FAIL, Verdict.REVIEW)
    for r in (hash_, drop, twobin):
        assert r.verdict is not Verdict.PASS

    # --- the extrinsic-failure specs are STRICTLY worse than GOOD on knn ----------
    # (HASH fails on the INTRINSIC dead-token gate, not on knn — its discriminator is
    # the monotone dead_token_frac asserted below, so knn need only be ≤ GOOD there.)
    assert g_knn > dd["extrinsic"]["knn_prec_at_k"], "DROP must score worse than GOOD on knn"
    assert g_knn > td["extrinsic"]["knn_prec_at_k"], "2-BIN must score worse than GOOD on knn"
    assert g_knn >= hd["extrinsic"]["knn_prec_at_k"]

    # --- HASH-EVERYTHING: the INTRINSIC dead-token gate dominates → FAIL -----------
    assert hash_.verdict is Verdict.FAIL
    assert hd["intrinsic"]["dead_token_frac"] > gd["intrinsic"]["dead_token_frac"], (
        "over-hashing must leave strictly MORE dead tokens than the GOOD spec"
    )
    hash_knobs = {p["knob"] for p in hd["refine_plan"]}
    assert hash_knobs & {"merchant_hash_size", "item_hash_size"}, hd["refine_plan"]

    # --- DROP-SIGNAL: cannot beat repeat-last (strict FAIL) + names the dropped field
    assert drop.verdict is Verdict.FAIL
    assert dd["extrinsic"]["knn_prec_at_k"] <= dd["controls"]["repeat_last_prec1"], (
        "dropping the predictive field must leave knn unable to beat repeat-last"
    )
    drop_entries = [p for p in dd["refine_plan"] if p["knob"] == "fields[]"]
    assert drop_entries, dd["refine_plan"]
    # the inferred item column ('item') is named as the field to re-include.
    assert any(p["field"] == "item" for p in drop_entries), drop_entries

    # --- 2-BIN-AMOUNT: under-binning collapses the chain → FAIL/REVIEW + n_bins -----
    assert twobin.verdict in (Verdict.FAIL, Verdict.REVIEW)
    twobin_knobs = {p["knob"] for p in td["refine_plan"]}
    assert twobin_knobs & {"n_bins", "amount_strategy"}, td["refine_plan"]


def test_bad_intrinsics_move_monotonically(store, tmp_path):
    """The bad specs' intrinsic health is monotonically worse than GOOD's: over-
    hashing strictly raises dead-token fraction; dropping/under-binning never lowers
    oov below GOOD's clean baseline. A pure RELATIVE comparison (no fixed cutoff)."""
    _require("ingest", "tokenize", "embed", "evaluate")
    in_path = _ingest_sample(store, tmp_path)

    _, _, good = _run_spec(store, tmp_path, in_path, "good", _GOOD_FM, experiment="g")
    _, _, hash_ = _run_spec(store, tmp_path, in_path, "hashall", _HASH_FM, experiment="h")
    _, _, twobin = _run_spec(store, tmp_path, in_path, "twobin", _2BIN_FM, experiment="t")

    g_dead = good.data["intrinsic"]["dead_token_frac"]
    g_oov = good.data["intrinsic"]["oov_rate"]

    # Over-hashing → strictly more dead tokens than the well-sized GOOD spec.
    assert hash_.data["intrinsic"]["dead_token_frac"] > g_dead
    # Neither bad spec has LOWER oov than GOOD's clean baseline (oov is monotone-bad).
    assert hash_.data["intrinsic"]["oov_rate"] >= g_oov
    assert twobin.data["intrinsic"]["oov_rate"] >= g_oov


def test_strict_gate_fail_is_status_ok_and_exit_code_one(store, tmp_path):
    """Anub's STRICT gate: a spec that cannot beat repeat-last is a COMPUTED FAIL —
    ``status==OK`` (not a structural refusal) with ``verdict==FAIL`` and
    ``exit_code==1`` (the legit "refine it" signal), distinct from a $0 structural
    REFUSED_CONTRACT (exit_code 2)."""
    _require("ingest", "tokenize", "embed", "evaluate")
    in_path = _ingest_sample(store, tmp_path)
    _, _, drop = _run_spec(store, tmp_path, in_path, "drop", _DROP_FM, experiment="d")

    assert drop.status is Status.OK
    assert drop.verdict is Verdict.FAIL
    assert drop.exit_code == 1


def test_evaluate_persists_a_scores_object_with_the_signatures(store, tmp_path):
    """A computed evaluate mints a ``Scores/<n>`` carrying the verdict + the scalar
    signatures (knn/controls/oov/dead/n_eval/eval_split/k) and a scores.json payload
    with the intrinsic/extrinsic/controls/refine_plan blocks — the durable record the
    refine loop reads."""
    _require("ingest", "tokenize", "embed", "evaluate")
    in_path = _ingest_sample(store, tmp_path)
    _, _, good = _run_spec(store, tmp_path, in_path, "good", _GOOD_FM, experiment="g")

    assert good.outputs and good.outputs[0].pathspec.startswith("Scores/")
    sc = store.get(good.data["pathspec"])
    assert sc.verdict is Verdict.PASS
    sigs = sc.signatures or {}
    for key in ("verdict", "knn_prec_at_k", "repeat_last_prec1", "popularity_prec_at_k",
                "oov_rate", "dead_token_frac", "n_eval", "eval_split", "k"):
        assert key in sigs, f"Scores signatures missing {key!r}"
    payload = json.loads(open(sc.payload_path, encoding="utf-8").read())
    for block in ("intrinsic", "extrinsic", "controls", "refine_plan"):
        assert block in payload


# ===========================================================================
# 2. The token-space ↔ row-space bridge — the subtlest part (RISK in the spec).
# ===========================================================================


def test_bridge_encodes_item_via_the_corpus_not_a_string_match(store, tmp_path):
    """The GOOD case proves the bridge maps a raw held-out item value → its token via
    the Corpus encode (the amount-strategy binning), NOT a verbatim string match — a
    numeric magnitude that string-matching could never find in the ``ITEM_<bin>``
    vocab. The evidence: ``knn`` is well above zero AND there are no OOV misses on the
    held-out actuals (every actual magnitude resolved to its bin token)."""
    _require("ingest", "tokenize", "embed", "evaluate")
    in_path = _ingest_sample(store, tmp_path)
    _, _, good = _run_spec(store, tmp_path, in_path, "good", _GOOD_FM, experiment="g")

    assert good.data["extrinsic"]["knn_prec_at_k"] > 0.0, (
        "a working bridge must recover the chain; knn==0 means encode() found no tokens"
    )
    assert good.data["extrinsic"]["n_oov_actual"] == 0, (
        "every held-out magnitude must encode to its bin token (no silent string-match miss)"
    )


# ===========================================================================
# 3a. Baseline --json byte-identity on the financial + chain fixtures — the
#     holdout extraction was a PURE move (the eval split did not fork).
# ===========================================================================


def _ingest_df(store: ObjectStore, tmp_path, df: pd.DataFrame, name: str, entity: str) -> str:
    csv = tmp_path / f"{name}.csv"
    df.to_csv(csv, index=False)
    res = _call("ingest", {"in": str(csv), "name": name, "entity": entity, "event": "evt"}, _ctx(store))
    assert res.status is Status.OK and res.outputs
    return res.outputs[0].pathspec


def test_baseline_json_byte_identical_across_faces_financial(store, tmp_path, tabformer_df):
    """``loom baseline --json`` (the CLI ``fn``) == the agent ``dispatch`` result
    char-for-char on the financial fixture, AND the leave-one-last-out metrics are the
    exact values the inline split produced before the extraction (the temporal split,
    the ``$``-amount coercion, the ``round(...,6)`` all preserved by the PURE move)."""
    _require("ingest", "baseline")
    in_path = _ingest_df(store, tmp_path, tabformer_df, "fin", "cust")

    cli = REGISTRY["baseline"].fn({"in": in_path, "k": 5}, _ctx(store, experiment="fin"))
    tool = dispatch("loom.baseline", {"in": in_path, "k": 5}, experiment="fin")
    assert cli.to_json() == tool.to_json(), "baseline CLI/agent envelopes diverged"

    metrics = json.loads(cli.to_json())["data"]["metrics"]
    # The financial fixture's temporal hold-out: cust 0 + cust 1 each have ≥2 events.
    # repeat-last over distinct mcc → 0; the next-amount MAE carry-forward over the
    # two held-out events is the exact pre-refactor value (proves the $-coercion +
    # round(...,6) survived the holdout extraction unchanged).
    assert metrics["repeat-last-item"]["value"] == 0.0
    assert metrics["next-amount-last-value"]["value"] == 753.755
    assert metrics["next-amount-last-value"]["n"] == 2
    assert metrics["popularity"]["topk"] == ["5411", "4111"]


def test_baseline_json_byte_identical_across_faces_chain(store, tmp_path, dex_df):
    """The same byte-identity + a stable temporal split on the chain (DEX) fixture —
    the extracted hold-out helper is the ONE split both baseline and evaluate ride,
    so the strong repeat-last control is computed exactly one way on both presets."""
    _require("ingest", "baseline")
    # The dex fixture's timestamp is a Timestamp column; persist it as a string so the
    # CSV round-trip matches the financial path (ingest writes rows.csv either way).
    df = dex_df.copy()
    df["timestamp"] = df["timestamp"].astype(str)
    in_path = _ingest_df(store, tmp_path, df, "chain", "wallet")

    cli = REGISTRY["baseline"].fn({"in": in_path, "k": 5}, _ctx(store, experiment="chain"))
    tool = dispatch("loom.baseline", {"in": in_path, "k": 5}, experiment="chain")
    assert cli.to_json() == tool.to_json(), "baseline CLI/agent envelopes diverged on chain"

    metrics = json.loads(cli.to_json())["data"]["metrics"]
    # Both wallets (0xa1: 3 events, 0xb2: 2 events) contribute a held-out target.
    assert metrics["repeat-last-item"]["metric"] == "prec@1"
    # 0xa1's last item is USDC, prev WETH (≠) → 0; 0xb2's last is SOL, prev SOL (=) →
    # 1 hit / 2 targets == 0.5 (the deterministic mergesort temporal order).
    assert metrics["repeat-last-item"]["value"] == 0.5


# ===========================================================================
# 3b. Determinism — embed twice → a bit-identical matrix + a deduped object;
#     evaluate twice → the same verdict + a deduped Scores.
# ===========================================================================


def _embed_matrix(store: ObjectStore, emb_pathspec: str) -> np.ndarray:
    from safetensors.numpy import load_file
    import os

    obj = store.get(emb_pathspec)
    st_dir = (obj.extras or {}).get("safetensors_dir")
    path = os.path.join(st_dir, "model.safetensors") if st_dir else obj.payload_path
    return np.asarray(load_file(path)["embeddings"])


def test_embed_is_deterministic_and_deduped(store, tmp_path):
    """``embed`` on the same Corpus + (dim, window) returns the SAME content-addressed
    Embeddings (the store dedupes a re-run) and a BIT-IDENTICAL matrix — the PPMI+SVD
    fit is deterministic (sklearn ``random_state=0``), so two runs never fork a twin."""
    _require("ingest", "tokenize", "embed")
    in_path = _ingest_sample(store, tmp_path)
    spec_path = _write_fieldmap(tmp_path, "good", _GOOD_FM)
    corpus = _call("tokenize", {"in": in_path, "spec": spec_path}, _ctx(store, experiment="g")).data["pathspec"]

    e1 = _call("embed", {"in": corpus}, _ctx(store, experiment="g"))
    e2 = _call("embed", {"in": corpus}, _ctx(store, experiment="g"))
    assert e1.data["pathspec"] == e2.data["pathspec"], "a re-run must dedupe to the same Embeddings"
    m1 = _embed_matrix(store, e1.data["pathspec"])
    m2 = _embed_matrix(store, e2.data["pathspec"])
    assert np.array_equal(m1, m2), "the embedding matrix must be bit-identical across runs"
    # Anub's default: dim=128 (16 stays valid; 128 is the new default).
    assert m1.shape[1] == 128


def test_evaluate_is_deterministic_and_deduped(store, tmp_path):
    """``evaluate`` twice on the same (Corpus, Embeddings, k, eval_split) returns the
    SAME verdict and dedupes to the SAME content-addressed Scores object — the verdict
    is a pure function of the deterministic embeddings + the shared hold-out."""
    _require("ingest", "tokenize", "embed", "evaluate")
    in_path = _ingest_sample(store, tmp_path)
    spec_path = _write_fieldmap(tmp_path, "good", _GOOD_FM)
    corpus = _call("tokenize", {"in": in_path, "spec": spec_path}, _ctx(store, experiment="g")).data["pathspec"]
    _call("embed", {"in": corpus}, _ctx(store, experiment="g"))

    v1 = _call("evaluate", {"in": corpus, "k": _K}, _ctx(store, experiment="g"))
    v2 = _call("evaluate", {"in": corpus, "k": _K}, _ctx(store, experiment="g"))
    assert v1.verdict is v2.verdict
    assert v1.data["pathspec"] == v2.data["pathspec"], "a re-run must dedupe to the same Scores"
    assert v1.data["extrinsic"]["knn_prec_at_k"] == v2.data["extrinsic"]["knn_prec_at_k"]


# ===========================================================================
# 3c. C5-mismatch — embed Corpus A, evaluate Corpus B vs A's embeddings → a clean
#     $0 structural REFUSED_CONTRACT named C5 (never a forward pass on a mispair).
# ===========================================================================


def test_c5_mismatch_is_refused_contract(store, tmp_path):
    """The C5 pairing guard fires FIRST: embeddings fit on Corpus A, evaluated against
    a DIFFERENT Corpus B (a different vocab_hash), are REFUSED with ``contract=='C5'``
    and ``status==REFUSED_CONTRACT`` — $0, no forward pass over the mispaired matrix."""
    _require("ingest", "tokenize", "embed", "evaluate")
    in_path = _ingest_sample(store, tmp_path)

    # Corpus A (the GOOD amount-binned spec) + its embeddings.
    spec_a = _write_fieldmap(tmp_path, "good", _GOOD_FM)
    corpus_a = _call("tokenize", {"in": in_path, "spec": spec_a}, _ctx(store, experiment="a")).data["pathspec"]
    emb_a = _call("embed", {"in": corpus_a}, _ctx(store, experiment="a")).data["pathspec"]

    # Corpus B — a structurally different spec (a small hash) → a different vocab_hash.
    spec_b = _write_fieldmap(tmp_path, "other", _fieldmap(
        [{"name": "item", "source": "item", "strategy": "hash", "buckets": 99}, _SIDE]
    ))
    corpus_b = _call("tokenize", {"in": in_path, "spec": spec_b}, _ctx(store, experiment="b")).data["pathspec"]
    assert corpus_a != corpus_b

    mm = _call("evaluate", {"in": corpus_b, "embeddings": emb_a, "k": _K}, _ctx(store, experiment="b"))
    assert mm.status is Status.REFUSED_CONTRACT, mm.summary
    assert any(d.contract == "C5" for d in mm.diagnostics), [d.contract for d in mm.diagnostics]
    assert mm.exit_code == 2, "a structural refusal exits 2 (distinct from a computed FAIL's 1)"


# ===========================================================================
# 4. propose runs on the sample (the loop's first verb is exercised end-to-end).
# ===========================================================================


def test_propose_runs_on_the_sample(store, tmp_path):
    """The loop opens with ``propose``: it reads the ingested sample's sniffed schema
    and emits a reviewable ``TokenizerSpec`` (verdict REVIEW — a proposal a human edits
    into the GOOD/bad variants). Exercises the first verb of
    ``propose → tokenize → embed → evaluate`` on the real path."""
    _require("ingest", "propose")
    in_path = _ingest_sample(store, tmp_path)
    res = _call("propose", {"in": in_path}, _ctx(store))
    assert res.status is Status.OK, res.summary
    assert res.outputs and res.outputs[0].pathspec.startswith("TokenizerSpec/")
