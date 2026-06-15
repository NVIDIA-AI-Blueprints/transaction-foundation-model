"""HiveMind capture tests: per-verb aggregation + the IP boundary + empty tolerance.

:func:`loom.hivemind.capture_corpus` is the **capture / skillify** stage of the
self-improvement loop (`design-spec.md` §5): it reads the learnings flywheel
(``learnings/rollouts.jsonl``) back and distills the rows for one ``/loom-*`` verb
into a small :class:`~loom.hivemind.VerbCorpus` digest the SkillOpt scorer reads.
It is pure -- no Metaflow, no LLM -- so these run everywhere on a fixture JSONL
written through the real :class:`loom.learnings.Learnings` writer.

The load-bearing behaviours pinned here, mirroring the corpus/learnings contract:

* aggregation -- ``n_rollouts`` / ``n_success`` / ``success_rate``, the metric
  distribution where a metric is present, the verdict/status histogram, and the
  recurring failure modes (most-common-first) are computed correctly;
* **the IP boundary** -- rows whose ``owned_by != "general"`` are EXCLUDED before
  anything is aggregated (a tenant-tagged row never folds into the cross-tenant
  moat), and the ``command == verb`` filter scopes capture to one verb; and
* **empty / missing tolerance** -- a missing file, an empty corpus, or a corpus
  with no rows for the verb yields an empty digest (never raises).
"""

from __future__ import annotations

from loom.config import LoomConfig
from loom.hivemind import (
    FailureMode,
    MetricDistribution,
    VerbCorpus,
    capture_corpus,
)
from loom.learnings import GENERAL, Learnings, LearningRecord, Outcome, TaskSpec


# ---------------------------------------------------------------------------
# Fixtures: write rollout rows through the real Learnings writer.
# ---------------------------------------------------------------------------


def _record(
    *,
    command: str = "eda",
    owned_by: str = GENERAL,
    tenant: str = "default",
    success: bool = True,
    best_metric: float | None = None,
    inputs: dict | None = None,
    reflection: str | None = None,
    ts: float = 1.0,
) -> LearningRecord:
    """Build one rollout row, defaulting everything the test does not assert on.

    Keyword-only so each call reads as a small, explicit override of just the
    fields a given test cares about (``owned_by`` for the boundary tests,
    ``success`` / ``inputs`` / ``reflection`` for the failure-mode tests).
    """
    return LearningRecord(
        command=command,
        task=TaskSpec(
            data_ref="IngestDataset/123",
            goal="profile the data object",
            metric="n/a",
            experiment_id="exp-hm",
        ),
        inputs=inputs if inputs is not None else {},
        outcome=Outcome(
            best_metric=best_metric,
            submission_ok=success,
            node_count=0,
        ),
        artifacts=["EdaFlow/1", "cards/abc"],
        success=success,
        model=None,
        tenant=tenant,
        owned_by=owned_by,
        reflection=reflection,
        ts=ts,
    )


def _store(tmp_path) -> Learnings:
    """A learnings store backed by a fresh JSONL file under ``tmp_path``."""
    cfg = LoomConfig(learnings_path=str(tmp_path / "learnings" / "rollouts.jsonl"))
    return Learnings(cfg)


def _capture(store: Learnings, verb: str, **kwargs) -> VerbCorpus:
    """Capture ``verb`` from the store's backing path (the CLI's one reader)."""
    return capture_corpus(verb, store.path, **kwargs)


# ---------------------------------------------------------------------------
# Empty / missing corpus tolerance (capture must never crash the loop).
# ---------------------------------------------------------------------------


def test_capture_missing_corpus_is_empty(tmp_path) -> None:
    """A learnings file that does not exist yields an empty digest, not an error."""
    missing = str(tmp_path / "nope" / "rollouts.jsonl")
    corpus = capture_corpus("eda", missing)

    assert isinstance(corpus, VerbCorpus)
    assert corpus.verb == "eda"
    assert corpus.is_empty
    assert corpus.n_rollouts == 0
    assert corpus.n_success == 0
    assert corpus.success_rate == 0.0
    assert corpus.failure_modes == []
    assert corpus.failure_labels == []
    assert corpus.verdict_histogram == {}
    assert corpus.metric.n == 0
    assert corpus.owned_by_filter == GENERAL


def test_capture_empty_when_no_rows_for_verb(tmp_path) -> None:
    """A corpus with rows only for *other* verbs yields an empty digest for the verb."""
    store = _store(tmp_path)
    store.record(_record(command="validate"))
    store.record(_record(command="features"))

    corpus = _capture(store, "eda")
    assert corpus.is_empty
    assert corpus.n_rollouts == 0


# ---------------------------------------------------------------------------
# Aggregation: counts, success rate, metric distribution, histograms.
# ---------------------------------------------------------------------------


def test_capture_aggregates_counts_and_success_rate(tmp_path) -> None:
    """n_rollouts / n_success / success_rate are computed over the verb's rows."""
    store = _store(tmp_path)
    store.record(_record(command="eda", success=True))
    store.record(_record(command="eda", success=True))
    store.record(_record(command="eda", success=True))
    store.record(_record(command="eda", success=False))
    # A row for a different verb must not be counted.
    store.record(_record(command="validate", success=True))

    corpus = _capture(store, "eda")
    assert corpus.n_rollouts == 4
    assert corpus.n_success == 3
    assert corpus.success_rate == 0.75
    assert not corpus.is_empty


def test_capture_metric_distribution_over_reported_metrics(tmp_path) -> None:
    """The metric distribution covers only rows that reported a numeric metric."""
    store = _store(tmp_path)
    store.record(_record(command="optimize", best_metric=0.80))
    store.record(_record(command="optimize", best_metric=0.90))
    store.record(_record(command="optimize", best_metric=0.70))
    # A row with no metric (read-only style) must not skew the distribution.
    store.record(_record(command="optimize", best_metric=None))

    corpus = _capture(store, "optimize")
    assert corpus.n_rollouts == 4
    dist = corpus.metric
    assert isinstance(dist, MetricDistribution)
    assert dist.n == 3
    assert dist.min == 0.70
    assert dist.max == 0.90
    assert abs(dist.mean - (0.80 + 0.90 + 0.70) / 3) < 1e-9


def test_capture_metric_all_none_when_no_metrics(tmp_path) -> None:
    """A read-only verb (no metric reported) yields an all-None distribution."""
    store = _store(tmp_path)
    store.record(_record(command="eda", best_metric=None))
    store.record(_record(command="eda", best_metric=None))

    corpus = _capture(store, "eda")
    assert corpus.metric.n == 0
    assert corpus.metric.min is None
    assert corpus.metric.mean is None
    assert corpus.metric.max is None


def test_capture_verdict_histogram(tmp_path) -> None:
    """The verdict histogram counts each row's status signal (explicit + derived)."""
    store = _store(tmp_path)
    # Explicit verdicts in the typed summary (inputs), normalized lowercase.
    store.record(_record(command="validate", success=True, inputs={"verdict": "PASS"}))
    store.record(_record(command="validate", success=True, inputs={"verdict": "pass"}))
    store.record(_record(command="validate", success=False, inputs={"verdict": "REVIEW"}))
    # No explicit verdict -> derived from the success bool.
    store.record(_record(command="validate", success=False, inputs={}))

    corpus = _capture(store, "validate")
    hist = corpus.verdict_histogram
    # PASS/pass collapse to one normalized key.
    assert hist.get("pass") == 2
    assert hist.get("review") == 1
    assert hist.get("failure") == 1
    assert sum(hist.values()) == corpus.n_rollouts


# ---------------------------------------------------------------------------
# Recurring failure modes (most-common-first; the scorer's soft-coverage target).
# ---------------------------------------------------------------------------


def test_capture_failure_modes_ranked(tmp_path) -> None:
    """Failure modes aggregate over FAILED rows only, most-common-first."""
    store = _store(tmp_path)
    # Two leakage failures, one timeout failure -> leakage ranks first.
    store.record(_record(command="features", success=False, inputs={"leakage": True}))
    store.record(_record(command="features", success=False, inputs={"leakage": True}))
    store.record(
        _record(command="features", success=False, reflection="timeout while building")
    )
    # A SUCCESS must not contribute a failure mode.
    store.record(_record(command="features", success=True))

    corpus = _capture(store, "features")
    assert corpus.n_rollouts == 4
    assert corpus.n_success == 1
    labels = corpus.failure_labels
    assert labels[0] == "leakage"  # most common failure first
    assert "timeout" in labels
    # The success row contributed nothing to the failure tally.
    counts = {fm.label: fm.count for fm in corpus.failure_modes}
    assert counts["leakage"] == 2
    assert counts["timeout"] == 1
    assert all(isinstance(fm, FailureMode) for fm in corpus.failure_modes)


def test_capture_failure_mode_from_nonsuccess_verdict(tmp_path) -> None:
    """A non-success verdict in the typed summary becomes the failure label."""
    store = _store(tmp_path)
    store.record(_record(command="validate", success=False, inputs={"verdict": "REVIEW"}))
    store.record(_record(command="validate", success=False, inputs={"verdict": "REVIEW"}))

    corpus = _capture(store, "validate")
    assert corpus.failure_labels[0] == "review"


# ---------------------------------------------------------------------------
# THE IP BOUNDARY: owned_by != "general" rows are EXCLUDED (the cross-tenant moat).
# ---------------------------------------------------------------------------


def test_capture_excludes_tenant_owned_rows(tmp_path) -> None:
    """A tenant-tagged row (owned_by != general) is never folded into the moat.

    The load-bearing IP boundary, mirroring ``Learnings.general`` /
    ``Corpus.general``: the default ``owned_by_filter="general"`` keeps only the
    cross-tenant slice. Two general rows + two tenant rows for the same verb must
    aggregate to exactly the two general rows.
    """
    store = _store(tmp_path)
    store.record(_record(command="eda", owned_by=GENERAL, success=True))
    store.record(_record(command="eda", owned_by=GENERAL, success=False))
    store.record(_record(command="eda", owned_by="acme", tenant="acme", success=True))
    store.record(_record(command="eda", owned_by="globex", tenant="globex", success=True))

    corpus = _capture(store, "eda")
    # Only the two general rows are captured; the tenant rows are excluded.
    assert corpus.n_rollouts == 2
    assert corpus.n_success == 1
    assert corpus.success_rate == 0.5
    assert corpus.owned_by_filter == GENERAL


def test_capture_empty_when_all_rows_tenant_owned(tmp_path) -> None:
    """If every row for the verb is tenant-owned, the general digest is empty."""
    store = _store(tmp_path)
    store.record(_record(command="eda", owned_by="acme", tenant="acme"))
    store.record(_record(command="eda", owned_by="acme", tenant="acme"))

    corpus = _capture(store, "eda")
    assert corpus.is_empty
    assert corpus.n_rollouts == 0


def test_capture_can_scope_to_a_tenant(tmp_path) -> None:
    """Passing a tenant owned_by_filter scopes capture to that tenant's own rows.

    The boundary is general by default but is a parameter: a tenant may capture
    its OWN rows by passing its owner tag, and that capture excludes both the
    general rows and other tenants' rows.
    """
    store = _store(tmp_path)
    store.record(_record(command="eda", owned_by=GENERAL))
    store.record(_record(command="eda", owned_by="acme", tenant="acme"))
    store.record(_record(command="eda", owned_by="acme", tenant="acme"))
    store.record(_record(command="eda", owned_by="globex", tenant="globex"))

    corpus = capture_corpus("eda", store.path, owned_by_filter="acme")
    assert corpus.n_rollouts == 2
    assert corpus.owned_by_filter == "acme"
