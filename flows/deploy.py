"""Loom's deployment Metaflow flow -- the validate-VERDICT gate is the centerpiece.

This module defines the single static ``FlowSpec`` -- :class:`DeployFlow` -- that the
``loom deploy`` command runs (via the Metaflow MLOps interface) to **promote** a
validated solution. Deploy is the **irreversible / external tier** of the approval
matrix (design-spec §3; ``CONVENTIONS.md`` §1): the real external action mutates
something off-box (a schedule, a registry entry), so its skill sets
``disable-model-invocation: true`` and the costly/irreversible apply is behind an
explicit ``--apply`` flag that is **OFF by default**. The default run produces a
deployment **PLAN** + a registry/manifest artifact + a *staged* register only --
**no external mutation**.

The load-bearing invariant is the **cross-verb exit gate**: deploy MUST assert the
upstream :class:`flows.validate.ValidateFlow` ``VERDICT == PASS`` before it will
deploy. A sub-threshold or leaky validation (verdict ``REVIEW``/``FAIL``, leakage
present, or a holdout below a declared floor) **BLOCKS** the deploy -- the gate
refuses rather than fails open. That decision is computed by the pure, unit-testable
:func:`deploy_gate`, so the gate is asserted in plain Python (a step, not a prompt)
and ships with an executable self-test that feeds a known-bad validate summary and
asserts BLOCK (see ``tests/test_deploy.py``).

The input is the upstream validate (or solution) **run pathspec**; the validate
report is read back through the Metaflow **Client API** only
(:func:`loom.dataio` style ``metaflow.Run(...).data``). Loom never touches the
underlying datastore (local or S3/minio) directly -- that is Metaflow's concern.

Flow shape::

    start --> plan --> end

* ``start`` -- resolve the upstream validate run via the Client API and read its
               ``report`` artifact (the typed validate summary carrying ``verdict``
               / ``holdout`` / ``leakage``). READ-ONLY over the upstream run.
* ``plan``  -- compute the gate decision with :func:`deploy_gate`; assemble a
               deployment PLAN + a registry manifest (model ref + lineage + the
               validate metric) via the pure :func:`build_deploy_plan`; only when
               ``apply`` is true AND the gate allowed does it perform the real,
               env/config-driven external action (else it stays a dry-run/staged
               register). Renders the ``@card`` (what would deploy, the GATE
               decision, lineage). Stores the typed summary on ``self.summary``.
* ``end``   -- carry ``self.summary`` forward so ``Run.data.summary`` exposes it to
               the MLOps interface's Client-API read.

The deploy *logic* is factored into the module-level pure functions
:func:`deploy_gate` and :func:`build_deploy_plan` so they are unit-testable on a
small in-memory validate summary with no Metaflow involved. The flow step is a thin
wrapper that reads the upstream report and calls them.

Only standard Metaflow APIs are used (``FlowSpec``, ``@step``, ``Parameter``,
``@card``, and the ``current.card`` append API). ``metaflow`` / ``loom`` are
imported *inside* the steps so the flow file parses even where they are not yet
importable until the Runner subprocess sets up the environment.
"""

from __future__ import annotations

from typing import Any

from metaflow import FlowSpec, Parameter, card, current, step

#: The single upstream verdict that allows a deploy. Anything else (``REVIEW`` /
#: ``FAIL`` / missing) BLOCKS -- the gate refuses rather than fails open.
_PASS_VERDICT = "PASS"

#: Default minimum sealed-holdout score the validate report must clear to deploy.
#: ``None`` means "no floor enforced beyond the verdict" -- a caller can pass a
#: concrete ``min_holdout`` to add a hard numeric bar on top of the verdict. The
#: gate is direction-agnostic about the metric *name* but treats a higher holdout
#: as better (the convention validate already uses for its PASS metrics); a
#: regression RMSE bar is expressed by the caller as a max, not handled here.
_DEFAULT_MIN_HOLDOUT = None


def deploy_gate(
    validate_summary: dict | None,
    min_holdout: float | None = _DEFAULT_MIN_HOLDOUT,
) -> dict:
    """Decide whether a deploy may proceed from an upstream validate summary (pure).

    This is the **centerpiece exit gate** of :class:`DeployFlow` and the reason
    deploy is a Loom verb rather than a loose ``mlflow register`` call: it asserts
    the upstream :func:`flows.validate.validate_dataframe` report's ``VERDICT`` (and,
    optionally, a hard holdout floor) **in plain Python** so the decision is
    machine-checkable and unit-testable -- never a prompt the model can talk itself
    past. It **fails closed**: anything other than an explicit, leak-free
    ``PASS`` that clears the (optional) floor BLOCKS the deploy.

    The rules, in order (any one BLOCKS):

    * **missing report** -- no validate summary at all -> BLOCK (you cannot deploy
      what was never validated);
    * **verdict != PASS** -- a ``REVIEW`` (leakage present) or ``FAIL`` verdict
      BLOCKS (a sub-threshold validation must not read as deployable);
    * **leakage present** -- ``leakage`` truthy BLOCKS even if a verdict slipped to
      ``PASS`` (defence in depth: an implausible score is explained, not shipped);
    * **no holdout score** -- a report with no sealed-holdout number BLOCKS (the
      headline "did not peek" number is the thing a promotion decision trusts);
    * **below the floor** -- when ``min_holdout`` is given and the holdout score is
      below it, BLOCK.

    Args:
        validate_summary: The typed validate report dict (from a ``ValidateFlow``
            run's ``report`` artifact), or ``None`` when none could be read. Read
            keys: ``verdict``, ``leakage``, ``holdout`` (``{"score", "n"}``),
            ``metric``, ``target``.
        min_holdout: Optional hard minimum sealed-holdout score the report must
            clear (in addition to the verdict). ``None`` -> no numeric floor beyond
            the verdict.

    Returns:
        A JSON-able gate decision dict with keys: ``allow`` (bool -- ``True`` only
        when every rule passes), ``decision`` (``"ALLOW"``/``"BLOCK"``),
        ``reasons`` (list of human-readable blocking reasons; empty when allowed),
        ``verdict`` (the upstream verdict echoed), ``holdout`` (the holdout score or
        ``None``), ``min_holdout`` (the floor applied or ``None``), and ``leakage``
        (bool).
    """
    reasons: list[str] = []

    summary = validate_summary if isinstance(validate_summary, dict) else None
    if summary is None:
        return {
            "allow": False,
            "decision": "BLOCK",
            "reasons": [
                "no upstream validate report found; deploy requires a "
                "loom-validate run with VERDICT==PASS (you cannot deploy what was "
                "never validated)."
            ],
            "verdict": None,
            "holdout": None,
            "min_holdout": min_holdout,
            "leakage": False,
        }

    verdict = str(summary.get("verdict") or "").strip().upper()
    leakage = bool(summary.get("leakage"))
    holdout_block = summary.get("holdout") or {}
    holdout_score = holdout_block.get("score")
    has_holdout = isinstance(holdout_score, (int, float))

    if verdict != _PASS_VERDICT:
        reasons.append(
            f"upstream validate VERDICT is {verdict or 'MISSING'!r}, not "
            f"{_PASS_VERDICT!r}; a sub-threshold / REVIEW / FAIL validation blocks "
            "deploy."
        )

    if leakage:
        reasons.append(
            "upstream validate reported leakage flags; an implausible score must "
            "be explained, not shipped (leakage blocks deploy)."
        )

    if not has_holdout:
        reasons.append(
            "upstream validate report carries no sealed-holdout score; the "
            "headline 'did not peek' number is required to deploy."
        )
    elif min_holdout is not None and float(holdout_score) < float(min_holdout):
        reasons.append(
            f"sealed-holdout score {float(holdout_score):.6g} is below the deploy "
            f"floor {float(min_holdout):.6g}."
        )

    allow = not reasons
    return {
        "allow": allow,
        "decision": "ALLOW" if allow else "BLOCK",
        "reasons": reasons,
        "verdict": verdict or None,
        "holdout": float(holdout_score) if has_holdout else None,
        "min_holdout": min_holdout,
        "leakage": leakage,
    }


def build_deploy_plan(
    source_run: str,
    validate_summary: dict | None,
    gate: dict,
    apply: bool,
    target: str | None = None,
    commit: str | None = None,
) -> dict:
    """Assemble a deployment PLAN + registry manifest from the gated inputs (pure).

    This is the unit-testable core of :class:`DeployFlow`'s ``plan`` step: given the
    upstream validate summary and the :func:`deploy_gate` decision, it builds the
    JSON-able deployment plan + the registry/manifest the run carries as its
    artifact (model ref + lineage + the validate metric). It is **side-effect free**
    -- it never performs the external action; it only describes *what would* (or,
    when applied + allowed, *what did*) deploy. Domain-neutral: the target is an
    env/config-driven string, never a hardcoded customer/vertical.

    The manifest's ``status`` reflects the safety posture:

    * ``"BLOCKED"`` -- the gate did not allow (regardless of ``apply``); no
      register, staged or applied;
    * ``"PLANNED"`` -- gate allowed but ``apply`` is False (the default): a staged
      register only, no external mutation;
    * ``"APPLIED"`` -- gate allowed AND ``apply`` is True: the manifest the flow's
      real external action will/did write.

    Args:
        source_run: The upstream validate/solution run pathspec being promoted.
        validate_summary: The upstream validate report dict (for the lineage +
            metric the manifest records), or ``None``.
        gate: The :func:`deploy_gate` decision dict.
        apply: Whether the real external action was requested (``--apply``). OFF by
            default; only meaningful when ``gate["allow"]`` is True.
        target: The env/config-driven deployment target name (e.g. a registry or
            schedule id). Domain-neutral; never a hardcoded customer.
        commit: Optional source commit recorded in the lineage for traceability.

    Returns:
        A JSON-able plan dict with keys: ``source_run``, ``target``, ``apply``,
        ``status`` (``"BLOCKED"``/``"PLANNED"``/``"APPLIED"``), ``gate`` (the
        decision), ``manifest`` (model ref + lineage + validate metric), and
        ``verdict`` (a top-level status line for downstream/CI -- ``"DEPLOYED"`` /
        ``"STAGED"`` / ``"BLOCKED"``).
    """
    summary = validate_summary if isinstance(validate_summary, dict) else {}
    allow = bool(gate.get("allow"))

    if not allow:
        status = "BLOCKED"
    elif apply:
        status = "APPLIED"
    else:
        status = "PLANNED"

    holdout = summary.get("holdout") or {}
    manifest = {
        # The thing being promoted: the validated solution/run is the model ref in
        # v0.1 (a candidate is data, not a separate object), so the source run IS
        # the model reference. Lineage points back to exactly what was validated.
        "model_ref": source_run,
        "lineage": {
            "validate_run": source_run,
            "dataset_ref": summary.get("dataset_ref"),
            "target": summary.get("target"),
            "commit": commit,
        },
        "validate_metric": {
            "metric": summary.get("metric"),
            "holdout": holdout.get("score"),
            "cv_mean": (summary.get("cv") or {}).get("mean"),
            "verdict": summary.get("verdict"),
        },
        "status": status,
    }

    top_verdict = {
        "BLOCKED": "BLOCKED",
        "PLANNED": "STAGED",
        "APPLIED": "DEPLOYED",
    }[status]

    return {
        "source_run": source_run,
        "target": target,
        "apply": bool(apply),
        "status": status,
        "gate": gate,
        "manifest": manifest,
        "verdict": top_verdict,
    }


# ---------------------------------------------------------------------------
# Client-API read helper (no datastore access; importable, lazily uses Metaflow).
# ---------------------------------------------------------------------------


def read_validate_summary(run_pathspec: str) -> dict | None:
    """Read a validate run's ``report`` summary via the Metaflow Client API.

    Resolves the ``"<FlowName>/<run_id>"`` pathspec to a ``metaflow.Run`` and reads
    its ``report`` artifact off ``.data`` (the typed validate summary). Best-effort:
    any failure (unresolvable run, metadata down, no report) yields ``None`` so the
    :func:`deploy_gate` then BLOCKS on a missing report rather than crashing.
    ``metaflow`` is imported lazily so importing this module never requires it.

    Args:
        run_pathspec: The upstream validate run pathspec (e.g. ``ValidateFlow/12``).

    Returns:
        The validate report dict, or ``None`` if it could not be read.
    """
    from metaflow import Run, namespace

    try:
        namespace(None)
    except Exception:  # pragma: no cover - namespace API edge case
        pass

    try:
        run = Run((run_pathspec or "").strip())
        data = run.data
    except Exception:  # noqa: BLE001 - unresolvable run / metadata down
        return None
    if data is None:  # pragma: no cover - a successful run has data
        return None
    report = getattr(data, "report", None)
    return dict(report) if isinstance(report, dict) else None


class DeployFlow(FlowSpec):
    """Gated promotion of a validated solution -- the validate-VERDICT gate enforced.

    Reads the upstream validate run's ``report`` via the Client API, computes the
    :func:`deploy_gate` decision (asserting ``VERDICT == PASS`` + an optional holdout
    floor), assembles a deployment PLAN + registry manifest with
    :func:`build_deploy_plan`, and emits a Metaflow run + an ``@card``. The real
    external action runs **only** when ``apply`` is True AND the gate allowed; the
    default (``apply=False``) is a dry-run / staged register with NO external
    mutation. The typed summary is carried on ``self.summary`` so the MLOps interface
    reads it back from ``Run.data``. Irreversible/external tier -- always gated; the
    skill sets ``disable-model-invocation: true``.
    """

    #: Pathspec of the upstream **validate** run whose VERDICT gates this deploy
    #: (e.g. ``ValidateFlow/12``). The validate report is read via the Client API.
    #: One of ``validate_run`` / ``solution_run`` is required.
    validate_run = Parameter(
        "validate_run",
        default="",
        type=str,
        help="Pathspec of the upstream validate run whose VERDICT gates deploy.",
    )

    #: Pathspec of a solution/optimize run to promote. Treated as the deploy source
    #: when no ``validate_run`` is given; its run is still read for a ``report``
    #: artifact (and the gate BLOCKS if it carries no PASS validate report).
    solution_run = Parameter(
        "solution_run",
        default="",
        type=str,
        help="Pathspec of a solution run to promote (read for a validate report).",
    )

    #: Whether to perform the real external action. **OFF by default** -- the
    #: default run produces a PLAN + staged register only. Even when True, the
    #: external apply runs ONLY if the gate allowed (sub-threshold validate blocks).
    apply = Parameter(
        "apply",
        default=False,
        type=bool,
        help="Perform the real external deploy (OFF by default; gate must allow).",
    )

    #: Optional hard minimum sealed-holdout score the validate report must clear, on
    #: top of the verdict. Empty/0 -> no numeric floor beyond the verdict.
    min_holdout = Parameter(
        "min_holdout",
        default=0.0,
        type=float,
        help="Optional minimum sealed-holdout score required to deploy.",
    )

    #: Env/config-driven deployment target name (e.g. a registry/schedule id).
    #: Domain-neutral; never a hardcoded customer/vertical. Empty -> a neutral
    #: ``LOOM_DEPLOY_TARGET`` env value, else the literal ``"staged-registry"``.
    target = Parameter(
        "target",
        default="",
        type=str,
        help="Deployment target name (env/config-driven; never a hardcoded customer).",
    )

    #: Optional source commit recorded in the manifest lineage for traceability.
    commit = Parameter(
        "commit",
        default="",
        type=str,
        help="Optional source commit recorded in the deploy manifest lineage.",
    )

    @step
    def start(self) -> None:
        """Resolve the upstream validate run and read its report via the Client API.

        The deploy *source* is the ``validate_run`` when given, else the
        ``solution_run``. Either way the source run is read for a ``report``
        artifact (the typed validate summary) through the Client API only -- never
        touching the datastore. A run with no readable report yields ``None``, which
        the gate in ``plan`` then BLOCKS on. READ-ONLY over the upstream run.
        """
        validate_run = (self.validate_run or "").strip()
        solution_run = (self.solution_run or "").strip()

        self._source_run = validate_run or solution_run
        self._validate_summary = (
            read_validate_summary(self._source_run) if self._source_run else None
        )
        self.next(self.plan)

    @card
    @step
    def plan(self) -> None:
        """Compute the gate, assemble the plan, and (only if allowed + applied) deploy.

        Delegates the gate decision to the pure :func:`deploy_gate` and the plan/
        manifest assembly to :func:`build_deploy_plan`, so both are unit-testable
        without Metaflow. The real external action is performed only when ``apply``
        is True AND the gate allowed; the default stays a dry-run / staged register.
        Renders the ``@card`` (what would deploy, the GATE decision, lineage) and
        stores the typed summary on ``self.summary``.
        """
        import os

        min_holdout = float(self.min_holdout or 0.0) or None
        gate = deploy_gate(self._validate_summary, min_holdout=min_holdout)

        # Resolve the target domain-neutrally: explicit > env (LOOM_DEPLOY_TARGET) >
        # a neutral staged-registry literal. NEVER a hardcoded customer/vertical.
        target = (self.target or "").strip()
        if not target:
            target = (os.environ.get("LOOM_DEPLOY_TARGET") or "").strip()
        if not target:
            target = "staged-registry"

        commit = (self.commit or "").strip() or None
        apply = bool(self.apply)

        plan = build_deploy_plan(
            self._source_run,
            self._validate_summary,
            gate,
            apply=apply,
            target=target,
            commit=commit,
        )

        # The real external action runs ONLY when applied AND the gate allowed. The
        # action itself is env/config-driven and side-effect-isolated in a helper so
        # the default (and any blocked) path performs no external mutation at all.
        applied_detail = None
        if apply and gate["allow"]:
            applied_detail = self._apply_external(plan, target)
        plan["applied_detail"] = applied_detail

        self.summary = plan
        self._render_card(plan)
        self.next(self.end)

    @staticmethod
    def _apply_external(plan: dict, target: str) -> dict:
        """Perform the real, env/config-driven external deploy action (apply path).

        Called ONLY from ``plan`` when ``apply`` is True and the gate allowed. In
        v0.1 the external action writes the registry manifest to the env/config
        sink: an outbox directory under ``LOOM_DEPLOY_DIR`` (default a local
        ``deploy/`` registry). This is the one place a mutation happens; it is kept
        in a single isolated helper so the default and blocked paths provably never
        reach it. The sink is **never** a hardcoded customer/vertical -- it is the
        env-driven target string.

        Args:
            plan: The assembled deploy plan (its manifest is the registry entry).
            target: The resolved deployment target name.

        Returns:
            A small JSON-able detail dict ``{"sink", "entry"}`` describing where the
            manifest was registered. Best-effort: a failure is reported in-band
            rather than raised, so the run records the attempt.
        """
        import json
        import os
        import time

        try:
            sink_dir = (os.environ.get("LOOM_DEPLOY_DIR") or "deploy").strip()
            os.makedirs(sink_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            safe_target = "".join(
                ch if ch.isalnum() or ch in "-_." else "_" for ch in target
            )
            entry = os.path.join(sink_dir, f"{safe_target}-{stamp}.json")
            with open(entry, "w", encoding="utf-8") as fh:
                json.dump(plan["manifest"], fh, indent=2, sort_keys=True)
            return {"sink": sink_dir, "entry": entry}
        except Exception as exc:  # noqa: BLE001 - record the attempt, never crash
            return {"sink": target, "entry": None, "error": str(exc)}

    def _render_card(self, plan: dict) -> None:
        """Render a Markdown + Tables ``@card`` (what would deploy, gate, lineage).

        Args:
            plan: The JSON-able plan dict from :func:`build_deploy_plan`.
        """
        from metaflow.cards import Markdown, Table

        gate = plan.get("gate") or {}
        manifest = plan.get("manifest") or {}
        lineage = manifest.get("lineage") or {}
        metric = manifest.get("validate_metric") or {}

        current.card.append(Markdown("# Loom deployment plan"))
        current.card.append(
            Markdown(
                f"**source run:** `{plan.get('source_run')}`  \n"
                f"**target:** `{plan.get('target')}`  \n"
                f"**apply:** {plan.get('apply')} "
                f"(real external action {'ON' if plan.get('apply') else 'OFF — staged plan only'})  \n"
                f"**GATE:** **{gate.get('decision')}** "
                f"(upstream VERDICT `{gate.get('verdict')}`)  \n"
                f"**VERDICT:** **{plan.get('verdict')}**"
            )
        )

        # Gate decision + reasons (the centerpiece: why deploy is/ isn't allowed).
        current.card.append(Markdown("## Gate decision"))
        reasons = gate.get("reasons") or []
        if reasons:
            current.card.append(
                Table(
                    [[i, r] for i, r in enumerate(reasons, start=1)],
                    headers=["#", "blocking reason"],
                )
            )
        else:
            current.card.append(
                Markdown(
                    "_Gate ALLOWED: upstream validate VERDICT==PASS, no leakage, "
                    "holdout clears the floor._"
                )
            )

        # The validate metric the manifest records (what was trusted).
        current.card.append(Markdown("## Validate metric (what would deploy)"))
        current.card.append(
            Table(
                [
                    [
                        metric.get("metric") or "n/a",
                        metric.get("holdout") if metric.get("holdout") is not None else "n/a",
                        metric.get("cv_mean") if metric.get("cv_mean") is not None else "n/a",
                        metric.get("verdict") or "n/a",
                    ]
                ],
                headers=["metric", "sealed holdout", "cv mean", "verdict"],
            )
        )

        # Lineage (model ref + what it traces back to).
        current.card.append(Markdown("## Lineage"))
        current.card.append(
            Table(
                [
                    ["model_ref", manifest.get("model_ref") or "n/a"],
                    ["validate_run", lineage.get("validate_run") or "n/a"],
                    ["dataset_ref", lineage.get("dataset_ref") or "n/a"],
                    ["target", lineage.get("target") or "n/a"],
                    ["commit", lineage.get("commit") or "n/a"],
                    ["manifest status", manifest.get("status") or "n/a"],
                ],
                headers=["field", "value"],
            )
        )

        applied = plan.get("applied_detail")
        if applied:
            current.card.append(Markdown("## Applied (external action performed)"))
            current.card.append(
                Markdown(
                    f"Registered manifest at `{applied.get('entry') or applied.get('sink')}`."
                    + (f"  \n_error: {applied['error']}_" if applied.get("error") else "")
                )
            )
        elif plan.get("status") == "PLANNED":
            current.card.append(
                Markdown(
                    "_Staged plan only — no external mutation (re-run with "
                    "`--apply` to deploy once the gate ALLOWS)._"
                )
            )

    @step
    def end(self) -> None:
        """Carry ``self.summary`` forward so ``Run.data.summary`` exposes it.

        Metaflow persists step artifacts, so ``self.summary`` (set in ``plan``) is
        already on ``Run.data``; the MLOps interface reads it back for the command's
        summary. Nothing else to do.
        """
        pass


if __name__ == "__main__":
    DeployFlow()
