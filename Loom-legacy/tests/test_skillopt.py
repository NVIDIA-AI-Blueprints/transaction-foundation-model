"""SkillOpt tests: the deterministic scorer, the never-worse GATE, and CLI wiring.

:mod:`loom.skillopt` is the **optimize** half of the self-improvement loop
(`design-spec.md` §5, §6 #5) -- the moat's safety gate, the exact parallel of
:func:`flows.deploy.deploy_gate`. The scorer (:class:`ContractCorpusScorer`) and
the gate (:func:`optimize_skill`) are pure, deterministic, and LLM-free, so these
run everywhere on small in-memory SKILL.md fixtures + a fixture
:class:`~loom.hivemind.VerbCorpus`.

The load-bearing invariants pinned here mirror the deploy-gate self-test:

* **the scorer** -- a contract-complete SKILL.md is ``hard_ok``; a SKILL.md that
  drops any one of the 7 hard contract checks is DISQUALIFIED (``hard_ok=False``);
  the soft score rises with corpus-failure-mode coverage + convention adherence;
* **THE GATE SELF-TEST (fails closed)** -- a contract-violating candidate is NEVER
  promoted even with a *higher soft score*; a regressing/tying candidate is NEVER
  promoted (no-regression + slow-update); selection picks the best *valid*
  candidate; an empty candidate list keeps the incumbent;
* **--apply writes only when promoted** -- the gated in-place overwrite happens
  iff the gate ALLOWED (mirrors deploy's apply-only-when-allowed posture); and
* **CLI arg-parse** -- ``loom skillopt --verb V [--candidate P | --propose]
  [--apply]`` parses into the handler (skipped until the subcommand is wired).
"""

from __future__ import annotations

import json

import pytest

from loom.hivemind import FailureMode, MetricDistribution, VerbCorpus
from loom.skillopt import (
    ContractCorpusScorer,
    SkillOptResult,
    SkillScore,
    optimize_skill,
    unified_diff,
)


# ---------------------------------------------------------------------------
# SKILL.md fixtures: a contract-complete one, and contract-violating variants.
# ---------------------------------------------------------------------------


def _good_skill(extra_body: str = "") -> str:
    """A contract-complete SKILL.md that passes all 7 HARD checks.

    Verified against :class:`ContractCorpusScorer`'s seven checks: frontmatter
    (name/description/when_to_use), an approval tier, speaks the loom CLI
    interface (not a backend / raw S3), the run + @card artifact, a learnings row,
    an exit-gate self-test, and a single-arg + dual-invocation surface. The
    optional ``extra_body`` lets a test add failure-mode / convention text to
    raise the SOFT score without touching the hard checks.
    """
    return f"""---
name: loom-eda
description: Profile a dataset (read-only). Triggers on "profile this data".
when_to_use: explore a dataframe before features
argument-hint: "dataset_ref + focus"
---

# loom-eda

This verb does a read-only profiling job through the loom CLI.

## 3. Run
Speak only Loom's provider interface by shelling out to the `loom` CLI; never
touch raw S3. The MLOps default is Metaflow but the verb stays backend-swappable.

## 5. Deliver
Hand back a Metaflow run + an @card and a typed JSON summary, and append a
learnings row to the flywheel (learnings/rollouts.jsonl) on every run.

## Composition
Exit gate: emit a typed VERDICT. The gate ships an executable self-test on a
known-bad fixture that asserts it BLOCKS.

Dual-invocation: works user-typed and model-auto-loaded.
{extra_body}
"""


def _skill_missing_learnings() -> str:
    """A polished SKILL.md that violates exactly ONE hard check: no learnings row.

    A skill that stopped recording a learnings row is a regression no matter how
    well it reads -- the scorer must DISQUALIFY it (``hard_ok=False``).
    """
    return """---
name: loom-eda
description: Profile a dataset (read-only). Triggers on "profile this data".
when_to_use: explore a dataframe before features
argument-hint: "dataset_ref + focus"
---

# loom-eda

Run through the `loom` CLI; never touch raw S3.
Hand back a Metaflow run + an @card and a typed JSON summary.
Exit gate ships an executable self-test that asserts it BLOCKS.
Dual-invocation: works user-typed and model-auto-loaded.
"""


def _skill_names_backend() -> str:
    """A SKILL.md that violates the interface check: it calls the backend directly.

    Contract-complete on every other axis, but it admits "call Metaflow
    directly" -- a backend break the scorer must DISQUALIFY.
    """
    return """---
name: loom-eda
description: Profile a dataset (read-only). Triggers on "profile this data".
when_to_use: explore a dataframe before features
argument-hint: "dataset_ref + focus"
---

# loom-eda

We call Metaflow directly and import a concrete adapter for speed.
Hand back a Metaflow run + an @card; append a learnings row to the flywheel.
Exit gate ships an executable self-test that asserts it BLOCKS.
Dual-invocation: works user-typed and model-auto-loaded.
"""


def _corpus_with_failures(*labels: str) -> VerbCorpus:
    """A VerbCorpus whose failure modes are ``labels`` (the soft-coverage target)."""
    return VerbCorpus(
        verb="eda",
        n_rollouts=10,
        n_success=6,
        success_rate=0.6,
        metric=MetricDistribution(n=0),
        verdict_histogram={"pass": 6, "review": 4},
        failure_modes=[FailureMode(label=l, count=1) for l in labels],
    )


def _empty_corpus() -> VerbCorpus:
    """An empty (no-failure-mode) corpus -- full marks on the coverage term."""
    return VerbCorpus(verb="eda")


# ---------------------------------------------------------------------------
# The scorer: HARD all-or-nothing + SOFT coverage.
# ---------------------------------------------------------------------------


def test_scorer_good_skill_is_hard_ok() -> None:
    """A contract-complete SKILL.md passes every hard check (hard_ok=True)."""
    scorer = ContractCorpusScorer()
    score = scorer.score(_good_skill(), _empty_corpus())

    assert isinstance(score, SkillScore)
    assert score.hard_ok is True
    assert score.detail["hard_misses"] == []
    # All seven named checks passed.
    assert all(score.detail["hard_checks"].values())
    # total = base + soft for a valid skill (well above the disqualified sentinel).
    assert score.total >= 1.0
    assert score.total == pytest.approx(1.0 + score.soft)


def test_scorer_missing_learnings_is_disqualified() -> None:
    """Dropping the learnings-row check DISQUALIFIES the skill (hard_ok=False)."""
    scorer = ContractCorpusScorer()
    score = scorer.score(_skill_missing_learnings(), _empty_corpus())

    assert score.hard_ok is False
    assert "learnings_row" in score.detail["hard_misses"]
    # A disqualified skill sinks far below any valid one, whatever its soft score.
    assert score.total < 0


def test_scorer_naming_backend_is_disqualified() -> None:
    """Admitting a direct backend call DISQUALIFIES on the interface check."""
    scorer = ContractCorpusScorer()
    score = scorer.score(_skill_names_backend(), _empty_corpus())

    assert score.hard_ok is False
    assert "speaks_interface" in score.detail["hard_misses"]


def test_scorer_soft_rises_with_failure_mode_coverage() -> None:
    """Naming the corpus's failure modes raises the soft (coverage) score.

    Two contract-complete skills differ only in whether they NAME the corpus's
    recurring failure modes; the one that addresses them must score strictly
    higher on soft (and thus total), since both are hard_ok.
    """
    scorer = ContractCorpusScorer()
    corpus = _corpus_with_failures("leakage", "timeout")

    blind = scorer.score(_good_skill(), corpus)
    aware = scorer.score(
        _good_skill(extra_body="It guards against leakage and a timeout budget cap."),
        corpus,
    )

    assert blind.hard_ok and aware.hard_ok
    assert aware.soft > blind.soft
    assert aware.total > blind.total


def test_scorer_empty_skill_text_is_disqualified() -> None:
    """An empty / blank skill text fails the hard gate (defensive)."""
    scorer = ContractCorpusScorer()
    assert scorer.score("", _empty_corpus()).hard_ok is False


# ---------------------------------------------------------------------------
# THE GATE (optimize_skill): the never-worse promotion self-tests.
# ---------------------------------------------------------------------------


def test_gate_promotes_strictly_better_candidate() -> None:
    """A hard-valid candidate that beats the incumbent on total IS promoted."""
    scorer = ContractCorpusScorer()
    corpus = _corpus_with_failures("leakage")

    incumbent = _good_skill()  # hard_ok, but does not name the failure mode
    candidate = _good_skill(extra_body="Explicitly handles leakage before features.")

    result = optimize_skill("eda", incumbent, [candidate], scorer, corpus)
    assert isinstance(result, SkillOptResult)
    assert result.promoted is True
    assert result.verdict == "PROMOTE"
    assert result.winner_index == 0
    assert result.winner_text == candidate
    assert result.gate_detail["best_candidate_total"] > result.gate_detail["incumbent_total"]


def test_gate_disqualified_candidate_never_promoted_even_with_higher_soft() -> None:
    """THE SELF-TEST: a contract violator is NEVER promoted, even if its soft is higher.

    The candidate addresses EVERY corpus failure mode + every convention marker
    (max soft) but violates a hard contract check (no learnings row). The hard
    gate dominates: it must be DISQUALIFIED and the incumbent KEPT. This is the
    moat's fail-closed guarantee -- a polished-but-broken skill never ships.
    """
    scorer = ContractCorpusScorer()
    corpus = _corpus_with_failures("leakage", "timeout")

    incumbent = _good_skill()  # hard_ok, lower soft
    # A contract VIOLATOR (no learnings row) stuffed with the failure modes +
    # convention surface, so its SOFT score would top the incumbent's if soft won.
    high_soft_violator = """---
name: loom-eda
description: Profile a dataset. Triggers on "profile this data".
when_to_use: explore a dataframe
argument-hint: "dataset_ref"
---

# loom-eda
Run through the `loom` CLI; never touch raw S3.
Hand back a Metaflow run + an @card.
Exit gate ships an executable self-test that asserts it BLOCKS.
Dual-invocation: works user-typed and model-auto-loaded.
It addresses leakage and a timeout, with lineage/fingerprint, a budget/cost cap,
no raw S3, and it sanitizes secrets.
"""
    cand_score = scorer.score(high_soft_violator, corpus)
    inc_score = scorer.score(incumbent, corpus)
    # The violator's SOFT really is higher than the incumbent's -- soft alone would
    # promote it -- yet it is hard-disqualified.
    assert cand_score.soft > inc_score.soft
    assert cand_score.hard_ok is False

    result = optimize_skill("eda", incumbent, [high_soft_violator], scorer, corpus)
    assert result.promoted is False
    assert result.verdict == "KEEP_ALL_DISQUALIFIED"
    assert result.winner_index is None
    assert result.winner_text == incumbent  # the incumbent is kept


def test_gate_regression_never_promoted() -> None:
    """A hard-valid candidate that does NOT beat the incumbent is KEPT (no-regression).

    The candidate is contract-complete but has a strictly LOWER soft (it ignores
    the corpus failure modes the incumbent addresses) -> a regression -> KEEP.
    """
    scorer = ContractCorpusScorer()
    corpus = _corpus_with_failures("leakage")

    incumbent = _good_skill(extra_body="Handles leakage carefully before features.")
    candidate = _good_skill()  # same hard pass, but lower soft (no leakage mention)

    result = optimize_skill("eda", incumbent, [candidate], scorer, corpus)
    assert result.promoted is False
    assert result.verdict == "KEEP"
    assert result.winner_text == incumbent


def test_gate_tie_keeps_incumbent_slow_update() -> None:
    """An exactly-tying candidate is KEPT -- the slow-update margin (no churn)."""
    scorer = ContractCorpusScorer()
    corpus = _empty_corpus()

    incumbent = _good_skill()
    candidate = _good_skill()  # identical contract surface -> identical total

    inc = scorer.score(incumbent, corpus)
    cand = scorer.score(candidate, corpus)
    assert cand.total == inc.total  # a genuine tie

    result = optimize_skill("eda", incumbent, [candidate], scorer, corpus)
    assert result.promoted is False
    assert result.verdict == "KEEP"


def test_gate_selects_best_valid_candidate() -> None:
    """Among several hard-valid candidates, the highest-total one is selected."""
    scorer = ContractCorpusScorer()
    corpus = _corpus_with_failures("leakage", "timeout", "drift")

    incumbent = _good_skill()
    weak = _good_skill(extra_body="Mentions leakage only.")
    best = _good_skill(
        extra_body="Handles leakage, a timeout cap, and drift; with lineage and "
        "a budget/cost cap; never raw S3; sanitizes secrets."
    )
    mid = _good_skill(extra_body="Handles leakage and timeout.")

    result = optimize_skill("eda", incumbent, [weak, best, mid], scorer, corpus)
    assert result.promoted is True
    assert result.verdict == "PROMOTE"
    # `best` is index 1 and has the highest soft/total of the three.
    assert result.winner_index == 1
    assert result.winner_text == best


def test_gate_disqualified_loses_to_valid_in_selection() -> None:
    """A higher-soft DISQUALIFIED candidate never beats a valid one in selection."""
    scorer = ContractCorpusScorer()
    corpus = _corpus_with_failures("leakage")

    incumbent = _good_skill()
    valid = _good_skill(extra_body="Handles leakage before features.")
    # Disqualified (no learnings row) but maximally soft.
    violator = _skill_missing_learnings()

    result = optimize_skill("eda", incumbent, [violator, valid], scorer, corpus)
    assert result.promoted is True
    assert result.winner_index == 1  # the valid candidate, not the violator
    assert result.winner_text == valid


def test_gate_no_candidate_keeps_incumbent() -> None:
    """An empty candidate list keeps the incumbent and reports it (no crash)."""
    scorer = ContractCorpusScorer()
    result = optimize_skill("eda", _good_skill(), [], scorer, _empty_corpus())

    assert result.promoted is False
    assert result.verdict == "KEEP_NO_CANDIDATE"
    assert result.winner_index is None
    assert result.winner_text == _good_skill()
    assert result.candidate_scores == []


def test_gate_result_audit_is_json_able() -> None:
    """The audit blob (what the CLI persists as a learnings row) round-trips JSON."""
    scorer = ContractCorpusScorer()
    corpus = _corpus_with_failures("leakage")
    result = optimize_skill(
        "eda", _good_skill(), [_good_skill(extra_body="Handles leakage.")], scorer, corpus
    )
    assert json.loads(json.dumps(result.audit)) == result.audit
    assert json.loads(json.dumps(result.gate_detail)) == result.gate_detail
    assert result.audit["verb"] == "eda"
    assert result.audit["promoted"] is True


def test_unified_diff_empty_when_identical_else_shows_change() -> None:
    """The winner-vs-incumbent diff is empty when identical, non-empty otherwise."""
    assert unified_diff(_good_skill(), _good_skill(), "eda") == ""
    diff = unified_diff(_good_skill(), _good_skill(extra_body="A new line."), "eda")
    assert diff
    assert "A new line." in diff


# ---------------------------------------------------------------------------
# --apply writes only when promoted (mirror deploy's apply-only-when-allowed).
# ---------------------------------------------------------------------------
#
# The gated in-place overwrite is the CLI's job; the safety RULE it enforces is
# pure and testable here: write the winning text iff the gate promoted. We pin
# the rule against optimize_skill's verdict so the CLI handler (which simply
# obeys it) inherits the guarantee.


def _apply_if_promoted(incumbent_path, result: SkillOptResult) -> bool:
    """Mirror the CLI's gated overwrite: write the winner iff promoted.

    Returns whether an in-place write happened. This is the exact rule the
    ``--apply`` branch obeys -- the real skill file is overwritten ONLY when the
    gate ALLOWED (``result.promoted``).
    """
    if not result.promoted:
        return False
    incumbent_path.write_text(result.winner_text, encoding="utf-8")
    return True


def test_apply_overwrites_only_when_promoted(tmp_path) -> None:
    """--apply overwrites the shipped skill ONLY when the gate promoted."""
    scorer = ContractCorpusScorer()
    corpus = _corpus_with_failures("leakage")

    skill_path = tmp_path / "SKILL.md"
    incumbent = _good_skill()
    skill_path.write_text(incumbent, encoding="utf-8")

    # A winning candidate -> --apply writes it in place.
    better = _good_skill(extra_body="Handles leakage before features.")
    promote = optimize_skill("eda", incumbent, [better], scorer, corpus)
    assert _apply_if_promoted(skill_path, promote) is True
    assert skill_path.read_text(encoding="utf-8") == better


def test_apply_is_noop_when_gate_keeps(tmp_path) -> None:
    """--apply is a no-op (the shipped skill is untouched) when the gate KEEPS.

    The fail-closed guarantee at the file level: a disqualified or regressing
    candidate must never overwrite the shipped skill even under --apply.
    """
    scorer = ContractCorpusScorer()
    corpus = _corpus_with_failures("leakage")

    skill_path = tmp_path / "SKILL.md"
    incumbent = _good_skill(extra_body="Handles leakage before features.")
    skill_path.write_text(incumbent, encoding="utf-8")

    # A contract-violating candidate (disqualified) -> KEEP -> no write.
    keep = optimize_skill(
        "eda", incumbent, [_skill_missing_learnings()], scorer, corpus
    )
    assert keep.promoted is False
    assert _apply_if_promoted(skill_path, keep) is False
    assert skill_path.read_text(encoding="utf-8") == incumbent  # untouched


# ---------------------------------------------------------------------------
# CLI arg-parsing for `loom skillopt` (skips until the subcommand is wired).
# ---------------------------------------------------------------------------


def _skillopt_in_parser() -> bool:
    """Whether the `skillopt` subcommand is wired into the CLI parser yet."""
    from loom.cli import _build_parser

    parser = _build_parser()
    try:
        parser.parse_args(["skillopt", "--verb", "loom-eda"])
    except SystemExit:
        return False
    except Exception:
        return False
    return True


_skillopt_required = pytest.mark.skipif(
    not _skillopt_in_parser(),
    reason="loom skillopt subcommand not wired into the CLI yet (wire-agent deliverable)",
)


@_skillopt_required
def test_cli_skillopt_parses_verb_and_candidate() -> None:
    """`loom skillopt --verb V --candidate PATH` parses into the handler."""
    from loom.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        ["skillopt", "--verb", "loom-eda", "--candidate", "cand.md"]
    )
    assert args.command == "skillopt"
    assert args.verb == "loom-eda"
    assert args.candidate == "cand.md"
    # --apply is OFF by default (the gated mutate is opt-in).
    assert args.apply is False


@_skillopt_required
def test_cli_skillopt_apply_off_by_default() -> None:
    """--apply defaults OFF; skillopt PROPOSES (read-only) unless --apply is passed."""
    from loom.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["skillopt", "--verb", "loom-eda"])
    assert args.apply is False


@_skillopt_required
def test_cli_skillopt_apply_flag_parses() -> None:
    """`loom skillopt --verb V --apply` flips the gated-mutate flag on."""
    from loom.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["skillopt", "--verb", "loom-eda", "--apply"])
    assert args.apply is True


@_skillopt_required
def test_cli_skillopt_propose_flag_parses() -> None:
    """`loom skillopt --verb V --propose` selects the (optional) LLM proposer."""
    from loom.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["skillopt", "--verb", "loom-eda", "--propose"])
    assert getattr(args, "propose", False) is True


@_skillopt_required
def test_cli_skillopt_requires_verb() -> None:
    """`loom skillopt` requires --verb (the skill being optimized)."""
    from loom.cli import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["skillopt"])
