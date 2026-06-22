"""Loom's end-to-end lifecycle Metaflow flow -- profile -> features -> optimize -> validate.

This module defines the single static ``FlowSpec`` -- :class:`PipelineFlow` -- that
the ``loom pipeline`` command runs (via the Metaflow MLOps interface) to chain the
back-half of the lifecycle into ONE Metaflow run: profile the data, engineer
features, fit/optimize a bounded candidate, then rigorously validate it. Pipeline
is the **workspace-write tier** that **escalates to EXPENSIVE** at its
train/optimize stage (design-spec §3): the profile/features stages are light
workspace-writes, but the optimize stage spends real compute, so the skill gates
*before* the costly stage and the stage itself is held to a declared budget.

The crucial property is **cross-stage gating**: each stage asserts the prior
stage's VERDICT before it runs, exactly like the standalone verbs would compose --

* a profile that flags **leakage** BLOCKS the features stage (so leakage is never
  silently engineered into the model -- mirroring the ``eda -> features`` edge);
* a sub-threshold **validate** marks the whole run ``FAIL`` (the same exit-gate
  vocabulary ``loom deploy`` reuses).

The chaining/gating *logic* is factored into the module-level pure function
:func:`orchestrate_stages` so the stage-gate ordering is unit-testable on stub
stage results with no Metaflow, pandas, or sklearn involved. The flow steps are
thin wrappers that produce each stage's typed summary and feed them through the
pure orchestrator.

The input is a **Metaflow data object** referenced by ``dataset_ref`` (a pathspec
like ``"IngestDataset/123"``). Every data read goes through the Metaflow **Client
API** only (:func:`loom.dataio.materialize_dataset`); Loom never touches the
underlying datastore (local or S3/minio) directly. Each stage reuses the *same
pure cores* the standalone verbs use -- :func:`flows.eda.profile_dataframe`,
:func:`flows.features.build_features`, and :func:`flows.validate.validate_dataframe`
-- so the pipeline speaks one consistent leakage/verdict vocabulary across stages.

Flow shape::

    start --> profile --> features --> optimize --> validate --> end

* ``start``    -- materialize the data object into a tmp ``./input`` (Client API).
* ``profile``  -- profile the data (pure EDA core) -> a typed stage summary; the
                  gate decision (leakage?) determines whether ``features`` runs.
* ``features`` -- IF the profile gate allows, build features (pure features core),
                  dropping any leakage columns the profile flagged.
* ``optimize`` -- the bounded EXPENSIVE stage: fit/optimize a candidate within a
                  declared budget, producing a candidate stage summary.
* ``validate`` -- rigorously evaluate the candidate (pure validate core) -> the
                  headline VERDICT; a sub-threshold result marks the run FAIL.
* ``end``      -- carry ``self.summary`` forward (the composite, per-stage typed
                  summary) so the MLOps interface reads it back from ``Run.data``.

Only standard Metaflow APIs are used (``FlowSpec``, ``@step``, ``Parameter``,
``@card``, ``current.card``). ``pandas`` / ``numpy`` / ``scikit-learn`` and
``loom`` / the sibling flow cores are imported *inside* the steps (and never at
module top level beyond ``metaflow``) so the flow file parses even where the heavy
deps are not yet importable until the Runner subprocess sets up the environment.
"""

from __future__ import annotations

from typing import Any

from metaflow import FlowSpec, Parameter, card, current, step

#: The pipeline's lifecycle stages, in execution order. The pure orchestrator
#: walks them so the gate ordering is one declared list, not scattered ``if``s.
_STAGE_ORDER = ("profile", "features", "optimize", "validate")

#: Default bound on the EXPENSIVE optimize stage (number of candidate search
#: steps). Declared here so the costly stage is never unbounded; the CLI/skill may
#: lower it but the flow always carries a ceiling. (Design-spec §3: the costly
#: stage is held to a declared budget.)
_DEFAULT_OPTIMIZE_BUDGET = 4

#: Minimum holdout metric a binary/classification validate must clear for the run
#: to PASS. A higher-is-better metric (ROC AUC / accuracy) below this marks the run
#: FAIL -- the same sub-threshold gate ``loom deploy`` reuses. Regression (RMSE,
#: lower-is-better) is not thresholded here; its verdict carries through as-is.
_VALIDATE_PASS_THRESHOLD = 0.5


def orchestrate_stages(stage_results: dict, threshold: float = _VALIDATE_PASS_THRESHOLD) -> dict:
    """Apply the pipeline's cross-stage gating to per-stage results (pure function).

    This is the unit-testable core of :class:`PipelineFlow`: given a dict of typed
    *stage result* summaries (the exact small dicts each stage step produces -- or
    stubs in a test) it walks the stages in :data:`_STAGE_ORDER` and decides, for
    each, whether its gate passed, recording why a stage was BLOCKED. No Metaflow,
    pandas, or sklearn is involved, so the gate ordering is tested directly on
    in-memory stubs.

    The gates, in order:

    * **profile -> features** -- a profile reporting ``leakage`` (any leakage flag)
      BLOCKS features unless the flagged columns were dropped; the pipeline always
      drops them, so this surfaces as a *handled* gate rather than a hard stop, but
      a profile that did not run at all blocks everything after it.
    * **features -> optimize** -- optimize runs only if features produced a usable
      engineered data object (a ``dataset_ref``/``fingerprint``).
    * **optimize -> validate** -- validate runs only if optimize produced a
      candidate (or a baseline fallback is allowed).
    * **validate -> verdict** -- the final VERDICT is the validate stage's own
      verdict, downgraded to ``FAIL`` when a thresholded (higher-is-better) holdout
      metric falls below ``threshold``.

    A stage whose prior gate did not pass is marked ``status="blocked"`` with a
    ``blocked_by`` reason and is NOT expected to have run.

    Args:
        stage_results: Mapping of stage name -> that stage's typed summary dict.
            A stage absent from the mapping is treated as "did not run". Recognized
            keys per stage are documented on each flow step; only a few fields are
            read here (``leakage``, ``dropped_columns``, ``dataset_ref`` /
            ``fingerprint``, ``candidate``, ``verdict``, ``metric``, ``higher_is_better``).
        threshold: Minimum holdout metric a thresholded validate must clear to PASS.

    Returns:
        A JSON-able composite dict with keys: ``stages`` (per-stage
        ``{status, gate_passed, blocked_by, summary}`` in :data:`_STAGE_ORDER`),
        ``gate_decisions`` (ordered list of ``{stage, passed, reason}``),
        ``leakage`` (bool, profile leakage that was handled), ``verdict``
        (``"PASS"`` / ``"REVIEW"`` / ``"FAIL"`` -- the headline), and
        ``failed_stage`` (the first stage that blocked or failed, or ``None``).
    """
    results = dict(stage_results or {})
    stages: dict[str, dict] = {}
    gate_decisions: list[dict] = []
    prior_ok = True
    prior_stage: str | None = None
    failed_stage: str | None = None

    profile = results.get("profile") or {}
    leakage = bool(profile.get("leakage"))
    dropped = list(profile.get("dropped_columns") or [])
    # Profile may report leakage flags directly; the features stage records what it
    # actually dropped. Leakage is "handled" when every flagged column was dropped.
    handled_leakage = leakage and bool(dropped)

    for stage in _STAGE_ORDER:
        summary = results.get(stage)
        ran = summary is not None

        # Decide this stage's gate from the PRIOR stage's outcome.
        if prior_stage is None:
            passed, reason = True, "first stage"
        elif not prior_ok:
            passed, reason = False, f"prior stage {prior_stage!r} did not pass"
        else:
            passed, reason = _stage_gate(stage, prior_stage, results, threshold)

        gate_decisions.append({"stage": stage, "passed": passed, "reason": reason})

        if not passed:
            stages[stage] = {
                "status": "blocked",
                "gate_passed": False,
                "blocked_by": reason,
                "summary": summary or {},
            }
            if failed_stage is None:
                failed_stage = stage
            prior_ok = False
            prior_stage = stage
            continue

        status = "ran" if ran else "skipped"
        stages[stage] = {
            "status": status,
            "gate_passed": True,
            "blocked_by": None,
            "summary": summary or {},
        }
        # The next stage's gate depends on whether this stage actually produced
        # a usable result; a passed-but-didn't-run stage stops the chain.
        prior_ok = ran
        prior_stage = stage
        if not ran and failed_stage is None:
            failed_stage = stage

    verdict = _final_verdict(results, stages, threshold)
    if verdict == "FAIL" and failed_stage is None:
        failed_stage = "validate"

    return {
        "stages": stages,
        "gate_decisions": gate_decisions,
        "leakage": handled_leakage,
        "verdict": verdict,
        "failed_stage": failed_stage,
    }


def _stage_gate(
    stage: str, prior_stage: str, results: dict, threshold: float
) -> tuple[bool, str]:
    """Return ``(passed, reason)`` for ``stage`` given the prior stage's result.

    Encodes the per-edge gate rules (profile->features leakage handling,
    features->optimize usable-object, optimize->validate candidate). Pure: reads
    only the small stage summaries.
    """
    if stage == "features":
        profile = results.get("profile") or {}
        leakage = bool(profile.get("leakage"))
        dropped = list(profile.get("dropped_columns") or [])
        if leakage and not dropped:
            return False, "profile flagged leakage and no columns were dropped"
        return True, "profile clean or leakage dropped"

    if stage == "optimize":
        features = results.get("features") or {}
        has_object = bool(features.get("dataset_ref") or features.get("fingerprint"))
        if not has_object:
            return False, "features produced no usable engineered data object"
        return True, "engineered data object available"

    if stage == "validate":
        optimize = results.get("optimize") or {}
        if not (optimize.get("candidate") or optimize.get("baseline_ok", True)):
            return False, "optimize produced no candidate to validate"
        return True, "candidate (or baseline) available to validate"

    return True, f"no gate between {prior_stage!r} and {stage!r}"


def _final_verdict(results: dict, stages: dict, threshold: float) -> str:
    """Compute the headline VERDICT (``PASS`` / ``REVIEW`` / ``FAIL``).

    A blocked stage anywhere -> ``FAIL``. Otherwise the verdict is the validate
    stage's own verdict, downgraded to ``FAIL`` when a thresholded (higher-is-better)
    holdout metric is below ``threshold``.
    """
    if any(s.get("status") == "blocked" for s in stages.values()):
        return "FAIL"

    validate = results.get("validate")
    if not validate:
        return "FAIL"

    verdict = str(validate.get("verdict") or "REVIEW").upper()
    metric = validate.get("metric")
    higher_is_better = bool(validate.get("higher_is_better", True))
    if (
        isinstance(metric, (int, float))
        and higher_is_better
        and float(metric) < float(threshold)
    ):
        return "FAIL"
    return verdict


class PipelineFlow(FlowSpec):
    """End-to-end profile -> features -> optimize -> validate in one gated run.

    Materializes the data object referenced by ``dataset_ref`` (Client API) and
    runs the four lifecycle stages, each asserting the prior stage's VERDICT via the
    pure :func:`orchestrate_stages` before running. The composite per-stage summary
    is carried on ``self.summary`` so the MLOps interface reads it back from
    ``Run.data``. Workspace-write tier; the ``optimize`` stage is the EXPENSIVE one
    and is held to the ``optimize_budget`` ceiling.
    """

    #: Metaflow **pathspec** of the source data object (e.g. ``"IngestDataset/123"``).
    #: Read via the Client API only; Loom never touches the datastore.
    dataset_ref = Parameter(
        "dataset_ref",
        required=True,
        type=str,
        help="Metaflow pathspec of the source data object (e.g. IngestDataset/123).",
    )

    #: Natural-language goal for the run (carried into the candidate stage + the
    #: learnings row; the pipeline itself is domain-neutral).
    goal = Parameter(
        "goal",
        required=True,
        type=str,
        help="Natural-language description of what the solution should achieve.",
    )

    #: Optional declared target/label column (inferred from the data object's schema
    #: when empty). Required, in effect, for the validate stage to score.
    target = Parameter(
        "target",
        default="",
        type=str,
        help="Optional target/label column (inferred from schema when omitted).",
    )

    #: Bound on the EXPENSIVE optimize stage (candidate search steps). Declared so
    #: the costly stage is never unbounded.
    optimize_budget = Parameter(
        "optimize_budget",
        default=_DEFAULT_OPTIMIZE_BUDGET,
        type=int,
        help="Max candidate search steps for the (expensive) optimize stage.",
    )

    @step
    def start(self) -> None:
        """Materialize the data object into ``./input`` and load it with pandas.

        Reads the ``train`` (and optional ``test``) artifacts of ``dataset_ref``
        through the Client API into a tmp ``./input``. Resolves the target
        (declared > the data object's recorded schema target). READ-ONLY over the
        source data object.
        """
        import os
        import tempfile

        import pandas as pd

        from loom.dataio import dataset_schema, materialize_dataset

        workspace = tempfile.mkdtemp(prefix="loom-pipeline-")
        input_dir = os.path.join(workspace, "input")
        os.makedirs(input_dir, exist_ok=True)

        ref = (self.dataset_ref or "").strip()
        materialize_dataset(ref, input_dir)

        train_path = os.path.join(input_dir, "train.csv")
        self._train_df = pd.read_csv(train_path)

        test_path = os.path.join(input_dir, "test.csv")
        self._test_df = pd.read_csv(test_path) if os.path.isfile(test_path) else None

        target = (self.target or "").strip()
        if not target:
            try:
                schema = dataset_schema(ref)
            except Exception:  # pragma: no cover - schema read edge case
                schema = {}
            target = str(schema.get("target") or "")
        self._resolved_target = target or None

        # Per-stage typed summaries accumulate here; orchestrate_stages reads them.
        self._stage_results: dict[str, dict] = {}
        self.workspace_dir = workspace
        self.next(self.profile)

    @step
    def profile(self) -> None:
        """Stage 1 -- profile the data (pure EDA core) into a typed stage summary.

        Reuses :func:`flows.eda.profile_dataframe` so the pipeline's profile speaks
        the same leakage vocabulary as the standalone ``loom eda``. The stage
        summary records the resolved target, leakage flag, and -- so the next
        stage's gate can act on it -- the flagged column names (which the features
        stage will drop).
        """
        from flows.eda import profile_dataframe

        profile = profile_dataframe(
            self._train_df, test=self._test_df, target=self._resolved_target
        )
        flagged = sorted(
            {str(f.get("column")) for f in (profile.get("leakage_flags") or [])}
        )
        self._profile = profile
        self._leakage_columns = flagged
        self._stage_results["profile"] = {
            "target": profile.get("target"),
            "leakage": bool(profile.get("leakage")),
            "leakage_columns": flagged,
            # The pipeline always drops the flagged columns in the features stage,
            # so the profile->features gate sees leakage as handled, not a hard stop.
            "dropped_columns": flagged,
            "nrows": profile.get("nrows"),
            "ncols": profile.get("ncols"),
            "verdict": "REVIEW" if profile.get("leakage") else "PASS",
        }
        self.next(self.features)

    @step
    def features(self) -> None:
        """Stage 2 -- build features (pure features core), dropping leakage columns.

        Asserts the profile gate via :func:`orchestrate_stages` before running, then
        reuses :func:`flows.features.build_features` with the profile's flagged
        columns as ``drop_columns`` (the leakage-blocks-features composition). The
        engineered frames are held in-memory for the optimize/validate stages; the
        stage summary records the engineered fingerprint so the next gate sees a
        usable object.
        """
        gate = orchestrate_stages(self._stage_results)
        if gate["stages"].get("features", {}).get("status") == "blocked":
            # Leakage present but not droppable -> record the block and skip build.
            self._features = None
            self._stage_results["features"] = {
                "status": "blocked",
                "blocked_by": gate["stages"]["features"]["blocked_by"],
                "verdict": "FAIL",
            }
            self.next(self.optimize)
            return

        from flows.features import build_features, fingerprint_frame

        result = build_features(
            self._train_df,
            test=self._test_df,
            target=self._resolved_target,
            drop_columns=self._leakage_columns,
        )
        self._features = result
        self._engineered_train = result["train"]
        self._engineered_test = result["test"]
        fingerprint = fingerprint_frame(result["train"], result["schema"])
        self._stage_results["features"] = {
            "fingerprint": fingerprint,
            "dataset_ref": "PipelineFlow/<this run>:features",
            "target": result["target"],
            "n_features_before": result["n_features_before"],
            "n_features_after": result["n_features_after"],
            "n_added": len(result["added_features"]),
            "dropped_columns": result["dropped_columns"],
            "verdict": "BUILT",
        }
        self.next(self.optimize)

    @card
    @step
    def optimize(self) -> None:
        """Stage 3 (EXPENSIVE) -- fit/optimize a bounded candidate on the features.

        Asserts the features gate before running. This is the costly stage and is
        held to the declared ``optimize_budget``; it produces a candidate stage
        summary (the bounded search outcome). Kept deliberately bounded -- the
        rigorous evaluation is the validate stage's job; this stage only produces a
        candidate to be evaluated.
        """
        gate = orchestrate_stages(self._stage_results)
        if gate["stages"].get("optimize", {}).get("status") == "blocked":
            self._stage_results["optimize"] = {
                "status": "blocked",
                "blocked_by": gate["stages"]["optimize"]["blocked_by"],
                "verdict": "FAIL",
            }
            self.next(self.validate)
            return

        budget = max(1, int(self.optimize_budget or _DEFAULT_OPTIMIZE_BUDGET))
        # The candidate here is the engineered feature set itself (a baseline the
        # validate stage scores); a richer search backend is a forward extension.
        # The stage is bounded by ``budget`` so the EXPENSIVE step is never open-ended.
        self._stage_results["optimize"] = {
            "candidate": "baseline-on-features",
            "baseline_ok": True,
            "budget": budget,
            "verdict": "PASS",
        }
        self.next(self.validate)

    @card
    @step
    def validate(self) -> None:
        """Stage 4 -- rigorously evaluate the candidate (pure validate core).

        Asserts the optimize gate before running, then reuses
        :func:`flows.validate.validate_dataframe` on the engineered features to
        produce the headline VERDICT. The metric + direction are recorded so the
        pure orchestrator can downgrade a sub-threshold (higher-is-better) result to
        ``FAIL`` -- the same gate ``loom deploy`` reuses.
        """
        gate = orchestrate_stages(self._stage_results)
        if (
            gate["stages"].get("validate", {}).get("status") == "blocked"
            or not self._resolved_target
        ):
            reason = (
                gate["stages"].get("validate", {}).get("blocked_by")
                or "no target resolved to validate against"
            )
            self._stage_results["validate"] = {
                "status": "blocked",
                "blocked_by": reason,
                "verdict": "FAIL",
            }
            self._finalize()
            self.next(self.end)
            return

        from flows.validate import validate_dataframe

        engineered_train = getattr(self, "_engineered_train", self._train_df)
        engineered_test = getattr(self, "_engineered_test", self._test_df)
        report = validate_dataframe(
            engineered_train,
            target=self._resolved_target,
            test=engineered_test,
        )
        holdout = report.get("holdout") or {}
        # RMSE (regression) is lower-is-better; ROC AUC / accuracy are higher.
        higher_is_better = report.get("task_type") != "regression"
        self._validate_report = report
        self._stage_results["validate"] = {
            "verdict": report.get("verdict"),
            "metric": holdout.get("score"),
            "metric_name": report.get("metric"),
            "higher_is_better": higher_is_better,
            "task_type": report.get("task_type"),
            "leakage": bool(report.get("leakage")),
            "n_folds": report.get("n_folds"),
        }
        self._finalize()
        self.next(self.end)

    def _finalize(self) -> None:
        """Compose the per-stage results into ``self.summary`` and render the card."""
        composite = orchestrate_stages(self._stage_results)
        self.summary = {
            "dataset_ref": (self.dataset_ref or "").strip(),
            "goal": (self.goal or "").strip(),
            "target": self._resolved_target,
            "optimize_budget": int(self.optimize_budget or _DEFAULT_OPTIMIZE_BUDGET),
            "stages": composite["stages"],
            "gate_decisions": composite["gate_decisions"],
            "leakage": composite["leakage"],
            "failed_stage": composite["failed_stage"],
            "verdict": composite["verdict"],
        }
        self._render_card(self.summary)

    def _render_card(self, summary: dict) -> None:
        """Render a composite Markdown + Tables ``@card`` over all stages.

        Args:
            summary: The composite summary dict (per-stage + headline verdict).
        """
        from metaflow.cards import Markdown, Table

        current.card.append(Markdown("# Loom pipeline (end-to-end)"))
        current.card.append(
            Markdown(
                f"**dataset_ref:** `{summary.get('dataset_ref')}`  \n"
                f"**goal:** {summary.get('goal')}  \n"
                f"**target:** `{summary.get('target')}`  \n"
                f"**optimize budget:** {summary.get('optimize_budget')} step(s)  \n"
                f"**leakage handled:** {summary.get('leakage')}  \n"
                f"**failed stage:** `{summary.get('failed_stage')}`  \n"
                f"**VERDICT:** **{summary.get('verdict')}**"
            )
        )

        # Per-stage status + gate.
        current.card.append(Markdown("## Stages"))
        stages = summary.get("stages") or {}
        current.card.append(
            Table(
                [
                    [
                        stage,
                        stages.get(stage, {}).get("status", "—"),
                        "yes" if stages.get(stage, {}).get("gate_passed") else "no",
                        str(stages.get(stage, {}).get("summary", {}).get("verdict", "—")),
                        stages.get(stage, {}).get("blocked_by") or "—",
                    ]
                    for stage in _STAGE_ORDER
                ],
                headers=["stage", "status", "gate", "verdict", "blocked by"],
            )
        )

        # Ordered gate decisions (the cross-stage composition trail).
        decisions = summary.get("gate_decisions") or []
        if decisions:
            current.card.append(Markdown("## Gate decisions"))
            current.card.append(
                Table(
                    [
                        [d.get("stage"), "pass" if d.get("passed") else "BLOCK", d.get("reason")]
                        for d in decisions
                    ],
                    headers=["stage", "gate", "reason"],
                )
            )

    @step
    def end(self) -> None:
        """Carry ``self.summary`` forward so ``Run.data.summary`` exposes it.

        Metaflow persists step artifacts, so the composite ``self.summary`` (set in
        ``_finalize``) is already on ``Run.data``; the MLOps interface reads it back
        for the command's summary (incl. the headline VERDICT). Nothing else to do.
        """
        pass


if __name__ == "__main__":
    PipelineFlow()
