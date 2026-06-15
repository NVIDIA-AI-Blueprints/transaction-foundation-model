"""SkillOpt: the deterministic SKILL.md scorer + the never-worse promotion GATE (the moat's heart).

This module is the **optimize** stage of Loom's self-improvement loop
(`design-spec.md` §5, §6 #5) -- the right half of the flywheel whose left half is
:mod:`loom.hivemind`. HiveMind reads the raw usage traces
(``learnings/rollouts.jsonl``) back into a per-verb :class:`~loom.hivemind.VerbCorpus`
digest; SkillOpt then scores each ``/loom-*`` ``SKILL.md`` (the *trainable artifact*:
every command IS a skill) against that captured evidence + the 7-point acceptance
contract from ``skills/CONVENTIONS.md`` / ``skills/_TEMPLATE/SKILL.md``, and decides
-- with a pure, machine-checkable GATE -- whether a *candidate* skill may replace the
incumbent. The gate is the exact parallel of :func:`flows.deploy.deploy_gate`: it
**fails closed** and **NEVER deploys a worse skill**.

Two safety invariants are load-bearing and unit-testable here:

* **Hard before soft.** The 7-point contract is a set of HARD constraints. A
  candidate that misses *any* hard constraint is DISQUALIFIED (``hard_ok=False``) and
  can **never** be promoted -- even if its soft (corpus-coverage) score is higher.
  A polished skill that stopped recording a learnings row, or that started naming a
  backend, is a regression no matter how well it reads.
* **No-regression + slow-update-with-selection.** Among the hard-valid candidates we
  *select the best*, but we promote it **only if** its total beats the incumbent's by
  a margin. A tie or a regression KEEPS the incumbent. This mirrors the spec's
  ``no-regression`` + ``slow_update_gate_with_selection: true`` posture.

The whole engine is **deterministic and LLM-free**: :class:`ContractCorpusScorer`
scores by text/contract inspection only, and candidate texts are supplied by the
caller (a file, the identity incumbent, or an optional pluggable LLM proposer). That
makes the loop fully testable without a model. Secrets are never read or logged; this
module imports only the standard library + :mod:`loom.hivemind`, so it is pure,
Metaflow-free, and importable in any environment.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Protocol

from loom.hivemind import VerbCorpus

# The promotion margin: a candidate's total must beat the incumbent's by at least
# this much to be promoted. A strictly-positive epsilon makes the gate a genuine
# *slow-update* gate -- a tie (or a sub-epsilon improvement) KEEPS the incumbent, so
# noise never churns a shipped skill. Parallels deploy's "fails closed" posture.
_PROMOTION_MARGIN = 1e-9

# The sentinel "total" of a hard-disqualified skill: a large negative number so a
# contract-violating candidate can never out-total any hard-valid skill, regardless
# of how high its soft score is. (We keep it finite/JSON-able rather than ``-inf`` so
# the audit row serializes cleanly.)
_DISQUALIFIED_TOTAL = -1.0e9

# Base credit a hard-valid skill earns before its soft coverage is added, so a valid
# skill's total lives in ``[base, base + 1]`` and is always far above the
# disqualified sentinel. The soft term then breaks ties between valid candidates.
_HARD_BASE = 1.0


@dataclass
class SkillScore:
    """The score of one ``SKILL.md`` text: a hard gate + a soft coverage number.

    The unit the GATE compares. ``hard_ok`` is the 7-point acceptance contract
    verdict (all-or-nothing); ``soft`` is a ``[0, 1]`` measure of how well the text
    covers the captured corpus failure modes + the convention surface; ``total`` folds
    them so that **any** hard miss sinks the skill below every hard-valid one.

    Attributes:
        hard_ok: Whether the text satisfies every HARD acceptance constraint. A
            single miss makes this ``False`` -> the skill is DISQUALIFIED.
        soft: A ``[0, 1]`` soft score -- corpus-failure-mode coverage + convention
            adherence. Only a tie-breaker among hard-valid skills; never rescues a
            hard-disqualified one.
        total: The single comparable number the gate ranks on. ``base + soft`` when
            ``hard_ok`` (so it sits in ``[base, base + 1]``), else a large negative
            sentinel so a contract violator can never out-rank a valid skill.
        detail: A small JSON-able breakdown: each hard check's pass/fail, the soft
            sub-scores, and the failure modes the text did / did not address.
    """

    hard_ok: bool
    soft: float
    total: float
    detail: dict = field(default_factory=dict)


@dataclass
class SkillOptResult:
    """The outcome of one :func:`optimize_skill` run -- promote-or-keep + the evidence.

    Mirrors the shape of :func:`flows.deploy.build_deploy_plan`'s plan dict: a clear
    boolean decision, a human-readable verdict, the scores it was based on, and an
    audit blob the CLI persists as a ``command="skillopt"`` learnings row. It is
    side-effect free -- it decides, it never writes the skill (the gated ``--apply``
    overwrite lives in the CLI, exactly like deploy's ``--apply``).

    Attributes:
        verb: The ``/loom-*`` verb whose skill was optimized.
        promoted: Whether a candidate WON the gate and should replace the incumbent.
            ``False`` keeps the incumbent (no candidate, a tie, a regression, or every
            candidate hard-disqualified).
        verdict: A short status line -- ``"PROMOTE"`` / ``"KEEP"`` / ``"KEEP_NO_CANDIDATE"``
            / ``"KEEP_ALL_DISQUALIFIED"`` -- the CLI/CI reads.
        incumbent_score: The :class:`SkillScore` of the shipped skill.
        candidate_scores: The :class:`SkillScore` of every candidate, in input order.
        winner_index: Index into ``candidate_scores`` of the promoted candidate, or
            ``None`` when the incumbent is kept.
        winner_text: The text that should be shipped -- the winning candidate when
            ``promoted``, else the incumbent text (so the caller always has the
            current-best to write to a sidecar / report).
        gate_detail: A JSON-able record of the gate decision (margin, the best
            candidate's total vs the incumbent's, why it promoted/kept).
        audit: A small JSON-able blob for the learnings row (verb, promoted, verdict,
            scores) -- no raw skill text, no secrets.
    """

    verb: str
    promoted: bool
    verdict: str
    incumbent_score: SkillScore
    candidate_scores: list[SkillScore] = field(default_factory=list)
    winner_index: int | None = None
    winner_text: str = ""
    gate_detail: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)


class SkillScorer(Protocol):
    """The scorer seam: turn a ``SKILL.md`` text + a corpus digest into a :class:`SkillScore`.

    A structural protocol (no inheritance required) so the gate is testable with a
    trivial fake scorer and a real one is swappable without touching
    :func:`optimize_skill`. The default implementation is the deterministic,
    LLM-free :class:`ContractCorpusScorer`.
    """

    def score(self, skill_text: str, corpus: VerbCorpus) -> SkillScore:
        """Score one ``SKILL.md`` text against the verb's captured corpus.

        Args:
            skill_text: The full ``SKILL.md`` text to score.
            corpus: The :class:`~loom.hivemind.VerbCorpus` digest for the verb.

        Returns:
            The :class:`SkillScore` for the text.
        """
        ...


# ---------------------------------------------------------------------------
# The default deterministic scorer: the 7-point contract (HARD) + corpus coverage (SOFT).
# ---------------------------------------------------------------------------


class ContractCorpusScorer:
    """Deterministic ``SKILL.md`` scorer: the 7-point acceptance contract + corpus coverage.

    The default :class:`SkillScorer`. It needs **no LLM**: it scores a skill text
    purely by inspecting it against the acceptance contract in
    ``skills/CONVENTIONS.md`` / ``skills/_TEMPLATE/SKILL.md`` (the HARD gate) and
    against the verb's captured :class:`~loom.hivemind.VerbCorpus` (the SOFT
    coverage). That determinism is what lets the whole self-improvement loop be
    unit-tested without a model.

    The seven HARD checks (any miss => ``hard_ok=False`` => DISQUALIFIED), drawn
    from the template's "Acceptance test" + the frontmatter contract:

    1. **frontmatter** -- a YAML frontmatter block declaring ``name`` /
       ``description`` / ``when_to_use``;
    2. **approval tier** -- names its sandbox tier (read-only / workspace-write /
       expensive-mutate / irreversible-external);
    3. **speaks the interface** -- shells out to the ``loom`` CLI / Loom's provider
       interface and does **not** call a backend directly or touch raw S3;
    4. **mandated artifact** -- states the Metaflow run + ``@card`` deliverable;
    5. **learnings row** -- records a ``learnings`` row to the flywheel;
    6. **exit-gate self-test** -- ships an executable self-test for its gate;
    7. **single-arg + dual-invocation** -- a single free-text arg and dual
       (user-typed + model-auto-load) invocation.

    The SOFT score (``[0, 1]``) is the mean of two equally-weighted terms: how many of
    the corpus's recurring failure-mode labels the text names/addresses, and a small
    convention-adherence bonus (mentions lineage/fingerprint, budget/cost, no-raw-S3,
    sanitize/secrets). A skill that explicitly speaks to the failures the field is
    actually hitting covers the corpus better and edges out one that does not.
    """

    #: Tier vocabulary (any one present satisfies the tier check). Matched
    #: case-insensitively against the text; hyphen/slash/space variants all count.
    _TIER_TERMS = (
        "read-only",
        "workspace-write",
        "expensive",
        "mutate",
        "irreversible",
        "external",
    )

    #: Convention-surface markers the soft score rewards (small adherence bonus).
    _CONVENTION_MARKERS = (
        ("lineage", ("lineage", "fingerprint", "pathspec")),
        ("budget", ("budget", "cost cap", "step cap", "wall-clock")),
        ("no_raw_s3", ("never touch raw s3", "no raw s3", "raw s3")),
        ("sanitize", ("sanitize", "never persist secret", "no secrets", "secret")),
    )

    def score(self, skill_text: str, corpus: VerbCorpus) -> SkillScore:
        """Score ``skill_text`` against the contract (hard) + the corpus (soft).

        Args:
            skill_text: The full ``SKILL.md`` text.
            corpus: The verb's captured corpus digest (drives the soft coverage).

        Returns:
            The :class:`SkillScore`. ``total`` is ``base + soft`` when every hard
            check passes, else a large negative sentinel (so a contract violator can
            never out-rank a hard-valid skill, whatever its soft score).
        """
        text = skill_text or ""
        low = text.lower()

        hard_checks = {
            "frontmatter": self._has_frontmatter(text),
            "approval_tier": self._declares_tier(low),
            "speaks_interface": self._speaks_interface(low),
            "mandated_artifact": self._states_artifact(low),
            "learnings_row": self._records_learnings(low),
            "exit_gate_self_test": self._has_self_test(low),
            "single_arg_dual_invocation": self._single_arg_dual_invocation(low),
        }
        hard_ok = all(hard_checks.values())

        soft, soft_detail = self._soft_score(low, corpus)

        if hard_ok:
            total = _HARD_BASE + soft
        else:
            total = _DISQUALIFIED_TOTAL

        detail = {
            "hard_checks": hard_checks,
            "hard_ok": hard_ok,
            "hard_misses": sorted(k for k, ok in hard_checks.items() if not ok),
            "soft": soft,
            "soft_detail": soft_detail,
        }
        return SkillScore(hard_ok=hard_ok, soft=soft, total=total, detail=detail)

    # -- the seven HARD checks (pure text inspection) -----------------------

    @staticmethod
    def _has_frontmatter(text: str) -> bool:
        """Check 1: a YAML frontmatter block declaring name/description/when_to_use.

        The frontmatter is the leading ``---`` ... ``---`` fence; the three required
        keys must appear inside it (so a stray ``description:`` in prose does not
        count).
        """
        stripped = text.lstrip()
        if not stripped.startswith("---"):
            return False
        # Body between the opening fence and the next ``---`` line.
        body = stripped[3:]
        end = body.find("\n---")
        front = body if end == -1 else body[:end]
        front_low = front.lower()
        return (
            re.search(r"^\s*name\s*:", front, re.MULTILINE) is not None
            and "description:" in front_low
            and "when_to_use:" in front_low
        )

    def _declares_tier(self, low: str) -> bool:
        """Check 2: the skill names an approval tier from the matrix."""
        return any(term in low for term in self._TIER_TERMS)

    @staticmethod
    def _speaks_interface(low: str) -> bool:
        """Check 3: speaks Loom's interface (the ``loom`` CLI), not a backend.

        Requires a mention of the ``loom`` CLI / provider interface AND that it does
        not advertise calling Metaflow/AIDE directly. A skill may *name* Metaflow as
        the default backend (that is allowed), so we only fail on an explicit
        "call ... directly" admission, mirroring the convention's wording.
        """
        speaks = (
            "loom " in low
            or "`loom" in low
            or "loom cli" in low
            or "provider interface" in low
            or "mlops interface" in low
        )
        # An explicit admission of calling a backend directly is a contract break.
        calls_backend_directly = (
            "call metaflow directly" in low
            or "call aide directly" in low
            or "import a concrete adapter" in low
            or "imports a concrete adapter" in low
        )
        return speaks and not calls_backend_directly

    @staticmethod
    def _states_artifact(low: str) -> bool:
        """Check 4: states the mandated Metaflow run + ``@card`` artifact."""
        has_card = "@card" in low or "card" in low
        has_run = "metaflow run" in low or "run + " in low or "run pathspec" in low or "pathspec" in low
        return has_card and has_run

    @staticmethod
    def _records_learnings(low: str) -> bool:
        """Check 5: records a learnings row to the flywheel corpus."""
        return "learnings" in low and (
            "row" in low or "rollouts.jsonl" in low or "flywheel" in low
        )

    @staticmethod
    def _has_self_test(low: str) -> bool:
        """Check 6: ships an executable exit-gate self-test."""
        return "self-test" in low or "self test" in low or "exit gate" in low or "exit-gate" in low

    @staticmethod
    def _single_arg_dual_invocation(low: str) -> bool:
        """Check 7: a single free-text arg AND dual (user + model) invocation.

        The ``argument-hint`` frontmatter key signals the single free-text arg; the
        ``when_to_use`` / dual-invocation wording (or ``disable-model-invocation``,
        which is the deliberate dual-invocation control) signals the dual surface.
        """
        single_arg = "argument-hint" in low or "single free-text arg" in low or "free-text arg" in low
        dual = (
            "dual-invocation" in low
            or "dual invocation" in low
            or "when_to_use" in low
            or "disable-model-invocation" in low
            or "auto-load" in low
        )
        return single_arg and dual

    # -- the SOFT coverage score -------------------------------------------

    def _soft_score(self, low: str, corpus: VerbCorpus) -> tuple[float, dict]:
        """Compute the ``[0, 1]`` soft coverage score + a small detail breakdown.

        Two equally-weighted terms:

        * **failure-mode coverage** -- the fraction of the corpus's recurring
          failure-mode labels the text names/addresses (``1.0`` when the corpus has
          no failure modes -- nothing to cover);
        * **convention adherence** -- the fraction of the convention-surface markers
          (lineage, budget, no-raw-S3, sanitize) the text mentions.

        Args:
            low: The lowercased skill text.
            corpus: The verb's captured corpus digest.

        Returns:
            ``(soft, detail)`` where ``soft`` is in ``[0, 1]``.
        """
        # Term 1: failure-mode coverage.
        labels = list(corpus.failure_labels) if corpus is not None else []
        addressed: list[str] = []
        missed: list[str] = []
        for label in labels:
            token = (label or "").strip().lower()
            if token and token in low:
                addressed.append(label)
            else:
                missed.append(label)
        if labels:
            coverage = len(addressed) / len(labels)
        else:
            coverage = 1.0  # nothing to cover -> full marks on this term.

        # Term 2: convention adherence.
        conv_hits: list[str] = []
        for name, markers in self._CONVENTION_MARKERS:
            if any(marker in low for marker in markers):
                conv_hits.append(name)
        adherence = len(conv_hits) / len(self._CONVENTION_MARKERS)

        soft = (coverage + adherence) / 2.0
        # Clamp defensively to [0, 1] (the inputs already are, but be safe).
        soft = max(0.0, min(1.0, soft))

        detail = {
            "failure_mode_coverage": coverage,
            "failure_modes_addressed": addressed,
            "failure_modes_missed": missed,
            "convention_adherence": adherence,
            "convention_hits": conv_hits,
        }
        return soft, detail


# ---------------------------------------------------------------------------
# The GATE: score the incumbent + candidates, promote the best ONLY if it wins.
# ---------------------------------------------------------------------------


def optimize_skill(
    verb: str,
    incumbent_text: str,
    candidate_texts: list[str],
    scorer: SkillScorer,
    corpus: VerbCorpus,
    margin: float = _PROMOTION_MARGIN,
) -> SkillOptResult:
    """Score the incumbent + candidates and apply the never-worse promotion GATE (pure).

    The heart of the moat's safety -- the exact parallel of
    :func:`flows.deploy.deploy_gate`. It scores the shipped (incumbent) ``SKILL.md``
    and every candidate with ``scorer``, then decides whether to PROMOTE the best
    candidate or KEEP the incumbent, under two invariants:

    * **Hard before soft (DISQUALIFICATION).** Only ``hard_ok`` candidates are even
      eligible. A candidate that violates any 7-point contract check is filtered out
      *before* selection -- it can never be promoted even if its soft score is higher
      (its ``total`` is the disqualified sentinel, far below any valid skill).
    * **Selection + no-regression + slow-update.** Among the hard-valid candidates we
      pick the one with the highest ``total`` (selection), and promote it **only if**
      ``best.total > incumbent.total + margin``. A tie or any regression KEEPS the
      incumbent (``promoted=False``). A strictly-positive ``margin`` makes this a
      genuine slow-update gate, so noise never churns a shipped skill.

    The function is side-effect free: it returns the decision + the winning text, but
    never writes anything. The gated in-place overwrite (``--apply``) lives in the
    CLI, mirroring deploy's apply-only-when-allowed posture.

    Args:
        verb: The ``/loom-*`` verb whose skill is being optimized.
        incumbent_text: The currently-shipped ``SKILL.md`` text.
        candidate_texts: Proposed replacement texts (from a file, the identity
            candidate, or an LLM proposer). May be empty (then the incumbent is
            kept and only scored/reported).
        scorer: The :class:`SkillScorer` to score every text with (default:
            :class:`ContractCorpusScorer`).
        corpus: The verb's captured :class:`~loom.hivemind.VerbCorpus` digest.
        margin: The strictly-positive improvement a candidate must clear to be
            promoted (no-regression + slow-update). Defaults to a tiny epsilon.

    Returns:
        A :class:`SkillOptResult` carrying the decision, every score, the winning
        text (the candidate when promoted, else the incumbent), the gate detail, and
        a small audit blob for the learnings row.
    """
    incumbent_score = scorer.score(incumbent_text, corpus)
    candidate_scores = [scorer.score(text, corpus) for text in candidate_texts]

    # Eligible = hard-valid candidates only. A contract violator is excluded here, so
    # it can never be selected/promoted -- the hard gate dominates the soft score.
    eligible = [
        (idx, score)
        for idx, score in enumerate(candidate_scores)
        if score.hard_ok
    ]

    best_index: int | None = None
    best_score: SkillScore | None = None
    for idx, score in eligible:
        if best_score is None or score.total > best_score.total:
            best_index, best_score = idx, score

    # Decide promote-vs-keep.
    if not candidate_texts:
        promoted = False
        verdict = "KEEP_NO_CANDIDATE"
    elif best_score is None:
        # Candidates exist but every one is hard-disqualified.
        promoted = False
        verdict = "KEEP_ALL_DISQUALIFIED"
    elif best_score.total > incumbent_score.total + margin:
        promoted = True
        verdict = "PROMOTE"
    else:
        # The best valid candidate does not beat the incumbent by the margin
        # (a tie or a regression) -> keep the incumbent. Never deploy a worse skill.
        promoted = False
        verdict = "KEEP"

    winner_index = best_index if promoted else None
    winner_text = candidate_texts[best_index] if (promoted and best_index is not None) else incumbent_text

    gate_detail = {
        "verdict": verdict,
        "promoted": promoted,
        "margin": margin,
        "incumbent_total": incumbent_score.total,
        "incumbent_hard_ok": incumbent_score.hard_ok,
        "best_candidate_index": best_index,
        "best_candidate_total": best_score.total if best_score is not None else None,
        "best_candidate_hard_ok": best_score.hard_ok if best_score is not None else None,
        "n_candidates": len(candidate_texts),
        "n_eligible": len(eligible),
        "n_disqualified": len(candidate_texts) - len(eligible),
    }

    audit = {
        "verb": verb,
        "promoted": promoted,
        "verdict": verdict,
        "incumbent": {
            "hard_ok": incumbent_score.hard_ok,
            "soft": incumbent_score.soft,
            "total": incumbent_score.total,
            "hard_misses": incumbent_score.detail.get("hard_misses", []),
        },
        "candidates": [
            {
                "index": idx,
                "hard_ok": s.hard_ok,
                "soft": s.soft,
                "total": s.total,
                "hard_misses": s.detail.get("hard_misses", []),
            }
            for idx, s in enumerate(candidate_scores)
        ],
        "winner_index": winner_index,
    }

    return SkillOptResult(
        verb=verb,
        promoted=promoted,
        verdict=verdict,
        incumbent_score=incumbent_score,
        candidate_scores=candidate_scores,
        winner_index=winner_index,
        winner_text=winner_text,
        gate_detail=gate_detail,
        audit=audit,
    )


def unified_diff(incumbent_text: str, candidate_text: str, verb: str) -> str:
    """Render a unified diff of the incumbent vs. a candidate ``SKILL.md`` (pure).

    A small convenience the CLI uses to show *what would change* before any
    ``--apply``. Side-effect free; returns the diff as a single string (empty when
    the texts are identical).

    Args:
        incumbent_text: The currently-shipped skill text.
        candidate_text: The proposed replacement text.
        verb: The verb, used to label the diff's from/to paths.

    Returns:
        The unified-diff text (empty string when there is no change).
    """
    incumbent_lines = (incumbent_text or "").splitlines(keepends=True)
    candidate_lines = (candidate_text or "").splitlines(keepends=True)
    diff = difflib.unified_diff(
        incumbent_lines,
        candidate_lines,
        fromfile=f"skills/{verb}/SKILL.md (incumbent)",
        tofile=f"skills/{verb}/SKILL.md (candidate)",
    )
    return "".join(diff)


__all__ = [
    "SkillScore",
    "SkillOptResult",
    "SkillScorer",
    "ContractCorpusScorer",
    "optimize_skill",
    "unified_diff",
]
