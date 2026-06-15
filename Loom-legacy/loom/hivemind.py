"""HiveMind capture: raw learnings traces -> per-verb skill evidence (the flywheel's left half).

This module is the **capture / skillify** stage of Loom's self-improvement loop
(`design-spec.md` §5): Loom usage leaves a trace corpus (one
:class:`~loom.learnings.LearningRecord` per command/run in
``learnings/rollouts.jsonl``); HiveMind reads that corpus back and distills it
into a small, per-verb **digest** -- the evidence :mod:`loom.skillopt` then scores
a ``SKILL.md`` against. It is the read counterpart of
:class:`loom.learnings.Learnings`'s write: where the controller/CLI *append* one
rollout row per run, :func:`capture_corpus` *aggregates* those rows for one verb
into a :class:`VerbCorpus`.

The IP boundary is load-bearing here and mirrors
:meth:`loom.corpus.Corpus.general` / :meth:`loom.learnings.Learnings.general`: the
cross-tenant moat may only learn from records whose ``owned_by == "general"``.
:func:`capture_corpus` filters to ``owned_by == owned_by_filter`` (``"general"``
by default) **and** ``command == verb`` before it aggregates anything, so a
tenant-tagged row is never folded into the cross-tenant skill evidence.

The digest is deliberately small and JSON-able -- success rate, the metric
distribution where a metric is present, the verdict/status histogram, and the
recurring failure modes (the most common error/verdict among the failures). No
raw rows, no code, no secrets ever leave the corpus through here; the learnings
schema itself carries none. This module imports only the standard library +
Loom core, so it is pure, Metaflow-free, and unit-testable on a fixture JSONL.
A missing or empty corpus yields an empty digest -- capture never crashes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from loom.config import LoomConfig
from loom.learnings import GENERAL, Learnings, LearningRecord

# How many recurring failure modes the digest surfaces (the long tail is noise).
_TOP_FAILURE_MODES = 5


@dataclass
class MetricDistribution:
    """The min/mean/max of the metric values present across a verb's rollouts.

    Only rollouts that reported a numeric ``outcome.best_metric`` contribute, so a
    read-only verb (which records no metric) yields an all-``None`` distribution
    with ``n == 0``. The skill scorer reads this as soft evidence of how the verb
    is performing in the field; it is direction-agnostic (it does not assume a
    higher metric is better -- that is a per-task convention).

    Attributes:
        n: Number of rollouts that reported a numeric metric.
        min: The smallest reported metric, or ``None`` when ``n == 0``.
        mean: The arithmetic mean of the reported metrics, or ``None``.
        max: The largest reported metric, or ``None``.
    """

    n: int = 0
    min: float | None = None
    mean: float | None = None
    max: float | None = None


@dataclass
class FailureMode:
    """One recurring failure signature aggregated across a verb's failed rollouts.

    A failure mode is keyed by the most specific failure signal a row carries: an
    exception/verdict label drawn from the row's ``reflection`` / status, falling
    back to a generic ``"unsuccessful"`` when a row failed without a label. The
    ``label`` is what the skill scorer matches its SKILL.md text against (a skill
    that names + mitigates a common failure mode covers the corpus better).

    Attributes:
        label: The normalized failure signature (e.g. ``"leakage"``,
            ``"timeout"``, ``"unsuccessful"``).
        count: How many failed rollouts shared this signature.
    """

    label: str
    count: int = 0


@dataclass
class VerbCorpus:
    """A per-verb digest of the learnings corpus -- HiveMind's captured evidence.

    The small, JSON-able rollup :func:`capture_corpus` produces for one
    ``/loom-*`` verb: how many (general-owned) rollouts it has, how often they
    succeeded, the metric distribution where present, the verdict/status
    histogram, and the recurring failure modes. This is the evidence
    :func:`loom.skillopt.optimize_skill` scores a candidate ``SKILL.md`` against;
    it is intentionally derived (no raw rows / code / secrets).

    Attributes:
        verb: The ``/loom-*`` verb (the ``command`` field) this digest covers.
        n_rollouts: Number of general-owned rollouts captured for the verb.
        n_success: Number of those rollouts whose ``success`` was true.
        success_rate: ``n_success / n_rollouts`` in ``[0, 1]``; ``0.0`` for an
            empty corpus.
        metric: The metric distribution across rollouts that reported one.
        verdict_histogram: Count of each verdict/status label seen (derived from
            the rows' status signals), most-common first when materialized.
        failure_modes: The recurring failure signatures among failed rollouts,
            most-common first, capped to the top few.
        owned_by_filter: The IP-boundary filter applied (``"general"`` by
            default) -- recorded so a reader can confirm the cross-tenant
            discipline held.
    """

    verb: str
    n_rollouts: int = 0
    n_success: int = 0
    success_rate: float = 0.0
    metric: MetricDistribution = field(default_factory=MetricDistribution)
    verdict_histogram: dict[str, int] = field(default_factory=dict)
    failure_modes: list[FailureMode] = field(default_factory=list)
    owned_by_filter: str = GENERAL

    @property
    def is_empty(self) -> bool:
        """Whether no rollouts were captured (a missing/empty/filtered-out corpus)."""
        return self.n_rollouts == 0

    @property
    def failure_labels(self) -> list[str]:
        """The recurring failure-mode labels, most-common first (scorer convenience)."""
        return [fm.label for fm in self.failure_modes]


def capture_corpus(
    verb: str,
    learnings_path: str,
    owned_by_filter: str = GENERAL,
) -> VerbCorpus:
    """Capture the learnings corpus for one verb into a :class:`VerbCorpus` digest (pure).

    HiveMind's "capture / skillify": read ``learnings/rollouts.jsonl``, filter to
    the rows that both target ``verb`` (``command == verb``) **and** are within the
    IP boundary (``owned_by == owned_by_filter``, ``"general"`` by default), then
    aggregate them into a small per-verb digest. Tenant-tagged rows are excluded
    from the cross-tenant moat *before* anything is aggregated, mirroring
    :meth:`loom.learnings.Learnings.general`.

    The function is pure and Metaflow-free: it reads the JSONL through
    :class:`loom.learnings.Learnings` (which tolerates a missing file and partial
    lines) and computes only derived aggregates. A missing, empty, or
    fully-filtered-out corpus yields an empty digest (``n_rollouts == 0``) rather
    than raising -- capture must never crash the loop.

    Args:
        verb: The ``/loom-*`` command to capture evidence for (matched against each
            row's ``command`` field, e.g. ``"eda"`` or ``"validate"``).
        learnings_path: Filesystem path of the backing ``rollouts.jsonl`` corpus.
        owned_by_filter: The IP-boundary owner to keep. Defaults to ``"general"``
            (the only slice a cross-tenant skill-optimizer may train on); passing a
            tenant value scopes capture to that tenant's own rows.

    Returns:
        A :class:`VerbCorpus` digest for the verb over the filtered rows.
    """
    # Read through the learnings store so we reuse its missing-file / partial-line
    # tolerance and its schema rehydration. We point a minimal config at the path
    # rather than touching the file directly, keeping one reader of the corpus.
    store = Learnings(LoomConfig(learnings_path=learnings_path))
    rows = [
        rec
        for rec in store.all()
        if rec.command == verb and rec.owned_by == owned_by_filter
    ]
    return _aggregate(verb, rows, owned_by_filter)


def _aggregate(
    verb: str,
    rows: list[LearningRecord],
    owned_by_filter: str,
) -> VerbCorpus:
    """Fold the already-filtered rows for ``verb`` into a :class:`VerbCorpus` (pure).

    Args:
        verb: The verb being captured.
        rows: The rollout rows already filtered to ``verb`` + the IP boundary.
        owned_by_filter: The owner filter that was applied (recorded on the digest).

    Returns:
        The aggregated digest. An empty ``rows`` list yields an empty digest.
    """
    n_rollouts = len(rows)
    if n_rollouts == 0:
        return VerbCorpus(verb=verb, owned_by_filter=owned_by_filter)

    n_success = sum(1 for r in rows if r.success)

    metrics = [
        float(r.outcome.best_metric)
        for r in rows
        if r.outcome is not None and isinstance(r.outcome.best_metric, (int, float))
    ]
    metric_dist = MetricDistribution()
    if metrics:
        metric_dist = MetricDistribution(
            n=len(metrics),
            min=min(metrics),
            mean=sum(metrics) / len(metrics),
            max=max(metrics),
        )

    verdict_counter: Counter[str] = Counter(_verdict_label(r) for r in rows)
    failure_counter: Counter[str] = Counter(
        _failure_label(r) for r in rows if not r.success
    )

    failure_modes = [
        FailureMode(label=label, count=count)
        for label, count in failure_counter.most_common(_TOP_FAILURE_MODES)
    ]

    return VerbCorpus(
        verb=verb,
        n_rollouts=n_rollouts,
        n_success=n_success,
        success_rate=n_success / n_rollouts,
        metric=metric_dist,
        verdict_histogram=dict(verdict_counter.most_common()),
        failure_modes=failure_modes,
        owned_by_filter=owned_by_filter,
    )


def _verdict_label(record: LearningRecord) -> str:
    """Derive a coarse verdict/status label for one rollout (for the histogram).

    The learnings schema has no dedicated verdict column, so we synthesize one
    from the signals a row does carry: an explicit ``verdict``/``status`` in the
    ``inputs`` (the typed JSON summary downstream verbs surface), else the
    success bool. Normalized to lowercase so ``PASS``/``pass`` collapse together.

    Args:
        record: The rollout row.

    Returns:
        A short status label (e.g. ``"pass"``, ``"blocked"``, ``"success"``,
        ``"failure"``).
    """
    inputs = record.inputs if isinstance(record.inputs, dict) else {}
    for key in ("verdict", "status", "decision"):
        value = inputs.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return "success" if record.success else "failure"


def _failure_label(record: LearningRecord) -> str:
    """Derive a recurring-failure signature for one *failed* rollout (pure).

    Picks the most specific failure signal the row carries: an explicit
    verdict/status from the typed summary in ``inputs`` (e.g. ``"REVIEW"``,
    ``"BLOCKED"``), a leakage flag, a free-text ``reflection`` keyword, finally a
    generic ``"unsuccessful"``. Normalized to lowercase so the same failure mode
    aggregates across rows. Never echoes raw row content beyond a short label.

    Args:
        record: A rollout row whose ``success`` is false.

    Returns:
        A short, normalized failure label.
    """
    inputs = record.inputs if isinstance(record.inputs, dict) else {}

    # 1) An explicit non-success verdict/status from the typed summary.
    for key in ("verdict", "status", "decision"):
        value = inputs.get(key)
        if isinstance(value, str) and value.strip():
            label = value.strip().lower()
            if label not in ("pass", "ok", "success", "allow", "allowed"):
                return label

    # 2) A leakage flag is a distinct, common, nameable failure mode.
    if inputs.get("leakage"):
        return "leakage"

    # 3) A short keyword from the free-text reflection (sanitized to one token).
    reflection = record.reflection
    if isinstance(reflection, str) and reflection.strip():
        first = reflection.strip().split()[0].lower()
        token = "".join(ch for ch in first if ch.isalnum() or ch in "-_")
        if token:
            return token

    # 4) Generic fallback: failed without a more specific signal.
    return "unsuccessful"


__all__ = [
    "VerbCorpus",
    "MetricDistribution",
    "FailureMode",
    "capture_corpus",
]
