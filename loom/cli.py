"""Command-line interface for Loom.

Exposes the ``loom`` console script (wired in ``pyproject.toml`` as
``loom = loom.cli:main``). Subcommands:

``loom ingest --source PATH [--name NAME]``
    Run the :class:`flows.ingest_dataset.IngestDataset` flow once (via
    ``metaflow.Runner``) to turn a local dataset into a **Metaflow data object**
    and print its **pathspec** (e.g. ``IngestDataset/123``). That pathspec is the
    ``dataset_ref`` to hand to ``loom run --dataset <pathspec>``. This is the one
    external->Metaflow boundary; thereafter Loom reads the data only through the
    Metaflow Client API and never touches the datastore (local or S3/minio).

``loom run [--data DIR] [--dataset PATHSPEC] --goal STR --metric STR [--steps N]
         [--mlops metaflow|local] [--search aide]``
    Build a :class:`~loom.types.Task` and a :class:`~loom.config.LoomConfig` from
    the parsed arguments (provider/budget overrides on top of the env/.env/YAML
    config), run the task through :func:`loom.controller.run_loom`, and print the
    best metric, the artifact paths from the returned
    :class:`~loom.types.SearchResult`, and a short leaderboard read from the
    execution provider's ``runs()``. The ``metaflow`` provider consumes
    ``--dataset`` (a data-object pathspec); the ``local`` dev provider consumes
    ``--data`` (a local dir).

``loom datasets``
    List the ingested **Metaflow data objects** (``IngestDataset`` runs tagged
    ``loom_dataset``) via the Metaflow Client API, printing each one's pathspec,
    name, and row/schema summary. Read-only; never touches the datastore.

``loom eda --dataset PATHSPEC [--target COL]``
    Profile a data object (read-only) by running the
    :class:`flows.eda.EdaFlow` through Loom's MLOps **interface**
    (``ExecutionProvider.run_flow``) -- schema, dtypes, missingness, numeric
    summary, target balance, top correlations, and leakage flags. Output is a
    Metaflow run + an ``@card``; the command prints a profile summary + the card
    reference and appends a ``command="eda"`` learnings row. Read-only tier --
    never prompts.

``loom validate --dataset PATHSPEC [--target COL] [--solution RUN] [--sensitive COL]``
    Rigorously evaluate a baseline/solution against a data object by running the
    :class:`flows.validate.ValidateFlow` through Loom's MLOps **interface**
    (``ExecutionProvider.run_flow``) -- a sealed holdout distinct from a
    stratified/purged K-fold CV, probability calibration (curve + Brier),
    per-slice / fairness metrics when a sensitive column is given, and leakage
    flags. With no ``--solution`` a gradient-boosted-trees baseline is fit. Output
    is a Metaflow run + an ``@card``; the command prints a validation summary +
    the card reference and appends a ``command="validate"`` learnings row.
    Workspace-write tier (light; the read-only evaluation runs in its own
    workspace and does not prompt).

``loom report (--experiment ID | --runs PATHSPEC,...)``
    Assemble an experiment's runs + metrics + lineage into a structured
    analysis/model-card by running the read-only :class:`flows.report.ReportFlow`
    through the MLOps interface. Output is a Metaflow run + an ``@card``; the
    command prints a report summary + the card reference and appends a
    ``command="report"`` learnings row. Read-only tier -- never prompts.

``loom viz (--dataset PATHSPEC | --run PATHSPEC) [--target COL] [--kind KIND]``
    Generate standard plots from a data object (distributions, correlation
    heatmap, target-vs-feature) or a run's results (metric-over-nodes,
    leaderboard) by running the read-only :class:`flows.viz.VizFlow` through the
    MLOps interface; the figures are emitted as ``@card`` images. The command
    prints a plot summary + the card reference and appends a ``command="viz"``
    learnings row. Read-only tier -- never prompts.

``loom features --dataset PATHSPEC [--target COL] [--from EDA-RUN] [--recipe NAME]``
    Build engineered features (domain-neutral transforms) from a data object into a
    **NEW** Metaflow data object by running the :class:`flows.features.FeaturesFlow`
    through the MLOps interface; the produced run's pathspec (``FeaturesFlow/<id>``)
    is the ``dataset_ref`` downstream verbs consume. Composes with ``loom eda`` via
    ``--from`` (the EDA run's leakage-flagged columns are dropped). Output is a run +
    an ``@card``; the command prints a feature-build summary + the card and appends a
    ``command="features"`` learnings row. Workspace-write tier (light/auto).

``loom pipeline --dataset PATHSPEC --goal STR [--target COL]``
    Run the end-to-end lifecycle (profile -> features -> a bounded optimize step ->
    validate) as ONE gated Metaflow run via the
    :class:`flows.pipeline.PipelineFlow`; each stage asserts the prior stage's
    VERDICT (leakage handled before features; a sub-threshold validate marks the run
    FAIL). Output is a composite ``@card``; the command prints a per-stage summary +
    the headline VERDICT and appends a ``command="pipeline"`` learnings row.
    Workspace-write tier that escalates to EXPENSIVE at the bounded optimize stage.

``loom deploy (--validate RUN | --solution RUN) [--apply]``
    Promote a validated solution by running the :class:`flows.deploy.DeployFlow`
    through the MLOps interface. The cross-verb exit gate asserts the upstream
    ``loom validate`` VERDICT==PASS before deploying (a sub-threshold validate
    BLOCKS). The real external action is OFF by default (``--apply``): the default is
    a deployment PLAN + a staged registry manifest, no external mutation. Output is a
    run + an ``@card``; the command prints the plan + the GATE decision + the VERDICT
    and appends a ``command="deploy"`` learnings row. Irreversible/external tier --
    always gated; never model-auto-invoked.

``loom ops [--flow NAME | --experiment ID | --dataset PATHSPEC --reference PATHSPEC]``
    Monitor run health + the leaderboard for a flow/experiment, plus a simple data
    DRIFT check (a data object vs. a reference), by running the read-only
    :class:`flows.ops.OpsFlow` through the MLOps interface. Output is a run + an
    ``@card``; the command prints a health/drift summary + the card and appends a
    ``command="ops"`` learnings row. Read-only tier -- never prompts.

``loom collab (--run PATHSPEC | --experiment ID) [--send]``
    Assemble a sanitized, shareable bundle (report/model-card + lineage manifest) of
    a run by running the :class:`flows.collab.CollabFlow` through the MLOps
    interface. The off-box SEND is OFF by default (``--send``): the default builds
    only -- no data leaves the box; a send routes to an env/config-driven sink
    (``LOOM_COLLAB_WEBHOOK`` / ``LOOM_COLLAB_OUTBOX``), never a hardcoded target.
    Output is a run + an ``@card``; the command prints the bundle summary + the
    would-send target and appends a ``command="collab"`` learnings row. Build =
    workspace-write; send = irreversible/external (gated; never model-auto-invoked).

``loom proxy serve [--host 127.0.0.1] [--port 8088]``
    Launch the **Loom gateway** (see :mod:`loom.proxy.server`): an
    Anthropic-passthrough server that authenticates callers by a Loom-issued
    ``LOOM_API_KEY``, injects Loom's system prompt, forwards to the real Anthropic
    API with a *server-side* ``ANTHROPIC_API_KEY`` the caller never sees, and logs
    every call centrally (the moat corpus). This is the server side of the
    opt-in ``--model-provider loom-proxy`` route. Both keys are read from the
    environment on the server; a missing server-side ``ANTHROPIC_API_KEY`` fails
    fast with an actionable message.

``loom skillopt --verb <loom-eda|...> [--candidate PATH | --propose] [--apply]``
    Run the self-improvement loop's **OPTIMIZE** stage over one verb's
    ``SKILL.md`` -- the moat (design-spec §5). HiveMind captures the verb's
    learnings corpus (:func:`loom.hivemind.capture_corpus`, filtered to the
    ``owned_by=general`` IP boundary), then SkillOpt's deterministic scorer grades
    the incumbent SKILL.md + any candidate on the 7-point acceptance contract
    (HARD) + corpus failure-mode coverage (SOFT) and applies a never-worse
    promotion GATE (the exact parallel of :func:`flows.deploy.deploy_gate`): the
    best hard-valid candidate is promoted ONLY if it beats the incumbent by a
    margin -- a contract violator or a regression can never win. **Safe by
    default:** the default PROPOSES (writes a sidecar ``SKILL.candidate.md`` + prints
    the corpus digest + the gate VERDICT + a unified diff); the real in-place
    overwrite is behind ``--apply`` and runs ONLY when the gate PROMOTED (mirroring
    ``loom deploy --apply``). Appends a ``command="skillopt"`` learnings audit row.
    The loop is deterministic / LLM-free; ``--propose`` is an optional pluggable
    proposer (a clearly-marked no-op when no model is configured). Read-only by
    default; ``--apply`` is the gated mutate.

This module is dependency-light at import time: it imports only the standard
library and Loom core. The heavy optional dependencies (AIDE, Metaflow, ...)
are pulled in only when the controller resolves a provider that needs them, at
``run`` time.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from typing import Mapping, Optional, Sequence

from loom.config import LoomConfig
from loom.controller import run_loom
from loom.registry import get_execution
from loom.types import SearchResult, Task

# How many leaderboard rows to display after a run.
_LEADERBOARD_LIMIT = 10


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with the ``run`` subcommand.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="loom",
        description=(
            "Loom: an agentic CLI for the full data-science "
            "lifecycle (ports-and-adapters provider architecture). Run with no "
            "subcommand (or `loom chat`) to drop into the interactive REPL."
        ),
    )
    # The --no-ui escape (also LOOM_NO_UI): force plain, non-Rich output for
    # CI/pipes. It only affects the interactive REPL path; the one-shot
    # subcommands print plain text regardless.
    parser.add_argument(
        "--no-ui",
        dest="no_ui",
        action="store_true",
        help="Disable the Rich interactive UI (plain output; for CI/pipes). "
        "Also via the LOOM_NO_UI environment variable.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # `loom chat`: an explicit alias for launching the interactive REPL (the same
    # branded loop a bare `loom` drops into). Carries no flags of its own; the
    # REPL reads the env/.env/YAML config like every other verb.
    chat_parser = subparsers.add_parser(
        "chat",
        help="Launch the interactive Loom REPL (same as running `loom` with no command).",
        description=(
            "Drop into the branded interactive Loom shell: a thin loop over the "
            "same verbs, with a themed render layer, interactive approval gates, "
            "and a streaming search. Identical to invoking `loom` with no "
            "subcommand. The read-only/lifecycle verbs work without an API key; "
            "only the LLM verbs (run / pipeline) need one."
        ),
    )
    chat_parser.set_defaults(func=_cmd_chat)

    run_parser = subparsers.add_parser(
        "run",
        help="Run a task end-to-end through the configured providers.",
        description=(
            "Propose, execute, score, and record candidate solutions for a "
            "task, then report the best solution and a short leaderboard."
        ),
    )
    run_parser.add_argument(
        "--data",
        default=None,
        metavar="DIR",
        help=(
            "Path to a local input data directory (staged into ./input by the "
            "'local' provider). Optional when --dataset is given."
        ),
    )
    run_parser.add_argument(
        "--dataset",
        default=None,
        metavar="PATHSPEC",
        help=(
            "Metaflow pathspec of an ingested data object (e.g. IngestDataset/123 "
            "from `loom ingest`). The 'metaflow' provider reads it via the "
            "Metaflow Client API; Loom never touches the datastore directly."
        ),
    )
    run_parser.add_argument(
        "--goal",
        required=True,
        metavar="STR",
        help="Natural-language description of what the solution should achieve.",
    )
    run_parser.add_argument(
        "--metric",
        required=True,
        metavar="STR",
        help="Natural-language description of how a solution is evaluated.",
    )
    run_parser.add_argument(
        "--steps",
        type=int,
        default=None,
        metavar="N",
        help="Number of search steps to run (overrides the config budget).",
    )
    run_parser.add_argument(
        "--mlops",
        default=None,
        metavar="metaflow|local",
        help="Execution ('muscle') provider name (overrides config).",
    )
    run_parser.add_argument(
        "--search",
        default=None,
        metavar="aide",
        help="Search ('brain') provider name (overrides config).",
    )
    run_parser.add_argument(
        "--code-provider",
        default=None,
        metavar="NAME",
        help=(
            "Model ('LLM backend') provider for the code role (overrides config), "
            "e.g. anthropic-api | openai-api | openrouter | nim | openai-compat | "
            "claude-subscription | codex-subscription."
        ),
    )
    run_parser.add_argument(
        "--feedback-provider",
        default=None,
        metavar="NAME",
        help="Model provider for the feedback/judge role (overrides config).",
    )
    run_parser.add_argument(
        "--model-provider",
        default=None,
        metavar="NAME",
        help=(
            "Shorthand setting BOTH --code-provider and --feedback-provider "
            "(the per-role flags, if given, take precedence)."
        ),
    )
    run_parser.add_argument(
        "--experiment-id",
        default=None,
        metavar="ID",
        help="Stable experiment id (default: a generated 'loom-<uuid>').",
    )
    run_parser.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Optional path to a YAML config file (lowest precedence).",
    )
    run_parser.set_defaults(func=_cmd_run)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest a local dataset into a Metaflow data object (once).",
        description=(
            "Run the IngestDataset flow to turn a local dir/CSV into a Metaflow "
            "data object and print its pathspec (the --dataset value for "
            "`loom run`). This is the one external->Metaflow boundary; Loom "
            "thereafter reads the data only via the Metaflow Client API and never "
            "touches the datastore (local or S3/minio) directly."
        ),
    )
    ingest_parser.add_argument(
        "--source",
        required=True,
        metavar="PATH",
        help=(
            "Local directory (with train.csv[, test.csv]) or a single .csv file "
            "to ingest once into a Metaflow data object."
        ),
    )
    ingest_parser.add_argument(
        "--name",
        default=None,
        metavar="NAME",
        help="Optional dataset name (also tagged as loom_dataset:<name>).",
    )
    ingest_parser.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Optional path to a YAML config file (for the Metaflow profile).",
    )
    ingest_parser.set_defaults(func=_cmd_ingest)

    eda_parser = subparsers.add_parser(
        "eda",
        help="Profile an ingested data object (read-only) -> a Metaflow run + @card.",
        description=(
            "Run the read-only EDA flow against a Metaflow data object: schema, "
            "dtypes, missingness, numeric summary, target balance, top feature "
            "correlations, and simple leakage flags. The work executes as a "
            "Metaflow run through Loom's MLOps interface and produces an @card; "
            "Loom reads the data only via the Metaflow Client API and never "
            "touches the datastore directly. EDA is read-only and never prompts."
        ),
    )
    eda_parser.add_argument(
        "--dataset",
        required=True,
        metavar="PATHSPEC",
        help=(
            "Metaflow pathspec of an ingested data object (e.g. IngestDataset/123 "
            "from `loom ingest`) to profile."
        ),
    )
    eda_parser.add_argument(
        "--target",
        default=None,
        metavar="COL",
        help="Optional target/label column name (inferred when omitted).",
    )
    eda_parser.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Optional path to a YAML config file (for the Metaflow profile).",
    )
    eda_parser.set_defaults(func=_cmd_eda)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Rigorously validate a baseline/solution against a data object -> run + @card.",
        description=(
            "Run the rigorous validation flow against a Metaflow data object: a "
            "sealed holdout distinct from a stratified/purged K-fold CV, "
            "probability calibration (curve + Brier), per-slice / fairness metrics "
            "when a sensitive column is given, and leakage flags. With no "
            "--solution a gradient-boosted-trees baseline is fit; with --solution a "
            "prior optimize run's best solution is evaluated. The work executes as "
            "a Metaflow run through Loom's MLOps interface and produces an @card; "
            "Loom reads the data only via the Client API and never touches the "
            "datastore. Validate is the workspace-write tier (light, no prompt for "
            "the read-only evaluation in its own workspace)."
        ),
    )
    validate_parser.add_argument(
        "--dataset",
        required=True,
        metavar="PATHSPEC",
        help=(
            "Metaflow pathspec of an ingested data object (e.g. IngestDataset/123) "
            "to validate against."
        ),
    )
    validate_parser.add_argument(
        "--target",
        default=None,
        metavar="COL",
        help="Target/label column to evaluate against (inferred from schema when omitted).",
    )
    validate_parser.add_argument(
        "--solution",
        default=None,
        metavar="RUN",
        help="Optional pathspec of a prior optimize run to evaluate (else a baseline).",
    )
    validate_parser.add_argument(
        "--sensitive",
        default=None,
        metavar="COL",
        help="Optional sensitive column for per-slice / fairness metrics.",
    )
    validate_parser.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Optional path to a YAML config file (for the Metaflow profile).",
    )
    validate_parser.set_defaults(func=_cmd_validate)

    report_parser = subparsers.add_parser(
        "report",
        help="Assemble an experiment's runs + metrics + lineage (read-only) -> run + @card.",
        description=(
            "Run the read-only report flow to gather an experiment's runs, their "
            "metrics, and their lineage (Flow/Run + tags + learnings rows) into a "
            "structured analysis/model-card via the Client API. The work executes "
            "as a Metaflow run through Loom's MLOps interface and produces an "
            "@card; it trains nothing and writes nothing back. Read-only -- never "
            "prompts. Give exactly one of --experiment or --runs."
        ),
    )
    report_group = report_parser.add_mutually_exclusive_group(required=True)
    report_group.add_argument(
        "--experiment",
        default=None,
        metavar="ID",
        help="Experiment id to report on (the loom_experiment tag).",
    )
    report_group.add_argument(
        "--runs",
        default=None,
        metavar="PATHSPEC,...",
        help="Comma-separated run pathspecs to report on (alternative to --experiment).",
    )
    report_parser.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Optional path to a YAML config file (for the Metaflow profile).",
    )
    report_parser.set_defaults(func=_cmd_report)

    viz_parser = subparsers.add_parser(
        "viz",
        help="Plot a data object or a run (read-only) -> a Metaflow run + @card images.",
        description=(
            "Run the read-only visualization flow to generate standard plots from "
            "a Metaflow data object (distributions, correlation heatmap, "
            "target-vs-feature) or a run's results (metric-over-nodes, "
            "leaderboard), emitted as @card images. The work executes as a "
            "Metaflow run through Loom's MLOps interface; Loom reads the data only "
            "via the Client API and never touches the datastore. Read-only -- "
            "never prompts. Give exactly one of --dataset or --run."
        ),
    )
    viz_group = viz_parser.add_mutually_exclusive_group(required=True)
    viz_group.add_argument(
        "--dataset",
        default=None,
        metavar="PATHSPEC",
        help="Metaflow pathspec of a data object to plot (e.g. IngestDataset/123).",
    )
    viz_group.add_argument(
        "--run",
        default=None,
        metavar="PATHSPEC",
        help="Metaflow pathspec of a run to plot (e.g. EvalCandidate/42).",
    )
    viz_parser.add_argument(
        "--target",
        default=None,
        metavar="COL",
        help="Optional target column for target-vs-feature plots (dataset input).",
    )
    viz_parser.add_argument(
        "--kind",
        default=None,
        metavar="KIND",
        help="Plot family for a dataset (distributions|correlation|target|all).",
    )
    viz_parser.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Optional path to a YAML config file (for the Metaflow profile).",
    )
    viz_parser.set_defaults(func=_cmd_viz)

    features_parser = subparsers.add_parser(
        "features",
        help="Build engineered features into a NEW data object -> a run + @card.",
        description=(
            "Run the feature-engineering flow against a Metaflow data object: "
            "domain-neutral transforms (numeric scaling/interactions, categorical "
            "encoding, datetime parts, simple aggregations) built into a brand-new "
            "Metaflow data object whose pathspec (FeaturesFlow/<id>) every "
            "downstream verb consumes via --dataset. Composes with `loom eda` via "
            "--from: an upstream EDA run's leakage-flagged columns are DROPPED "
            "before building. The work executes as a Metaflow run through Loom's "
            "MLOps interface and produces an @card (feature list, before/after "
            "schema, null/variance stats); Loom reads the data only via the Client "
            "API and never touches the datastore. Workspace-write tier (light/auto, "
            "network off): it reads the source read-only and writes only into its "
            "own workspace."
        ),
    )
    features_parser.add_argument(
        "--dataset",
        required=True,
        metavar="PATHSPEC",
        help=(
            "Metaflow pathspec of the source data object (e.g. IngestDataset/123) "
            "to build features from."
        ),
    )
    features_parser.add_argument(
        "--target",
        default=None,
        metavar="COL",
        help="Optional target/label column to preserve untouched (inferred when omitted).",
    )
    features_parser.add_argument(
        "--from",
        dest="from_eda",
        default=None,
        metavar="EDA-RUN",
        help=(
            "Optional upstream `loom eda` run pathspec (e.g. EdaFlow/12); its "
            "leakage-flagged columns are dropped before building (the eda->features "
            "composition edge)."
        ),
    )
    features_parser.add_argument(
        "--recipe",
        default=None,
        metavar="NAME",
        help="Optional transform recipe: minimal | full (default full).",
    )
    features_parser.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Optional path to a YAML config file (for the Metaflow profile).",
    )
    features_parser.set_defaults(func=_cmd_features)

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the end-to-end lifecycle (profile->features->optimize->validate) -> run + @card.",
        description=(
            "Run the end-to-end lifecycle flow against a Metaflow data object: "
            "profile -> features -> a bounded candidate/optimize step -> validate, "
            "chained into ONE gated Metaflow run. Each stage asserts the prior "
            "stage's VERDICT before running (a profile that flags leakage blocks/"
            "handles features; a sub-threshold validate marks the run FAIL). The "
            "optimize stage is the bounded EXPENSIVE step (held to a declared "
            "budget). The work executes as a Metaflow run through Loom's MLOps "
            "interface and produces a composite @card; Loom reads the data only via "
            "the Client API. Workspace-write tier that escalates to EXPENSIVE at "
            "the train/optimize stage."
        ),
    )
    pipeline_parser.add_argument(
        "--dataset",
        required=True,
        metavar="PATHSPEC",
        help=(
            "Metaflow pathspec of the source data object (e.g. IngestDataset/123) "
            "to run the lifecycle against."
        ),
    )
    pipeline_parser.add_argument(
        "--goal",
        required=True,
        metavar="STR",
        help="Natural-language description of what the solution should achieve.",
    )
    pipeline_parser.add_argument(
        "--target",
        default=None,
        metavar="COL",
        help="Optional target/label column (inferred from the data object's schema when omitted).",
    )
    pipeline_parser.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Optional path to a YAML config file (for the Metaflow profile).",
    )
    pipeline_parser.set_defaults(func=_cmd_pipeline)

    deploy_parser = subparsers.add_parser(
        "deploy",
        help="Gate on a validate VERDICT==PASS and produce a deploy PLAN -> run + @card.",
        description=(
            "Run the gated deployment flow to promote a validated solution. The "
            "centerpiece is the cross-verb exit gate: deploy asserts the upstream "
            "`loom validate` VERDICT==PASS (and an optional holdout floor) before "
            "it will deploy -- a sub-threshold / REVIEW / FAIL / leaky validation "
            "BLOCKS the deploy. The real external action is OFF by default: the "
            "default run produces a deployment PLAN + a staged registry manifest "
            "(model ref + lineage + the validate metric) with NO external "
            "mutation. Only --apply performs the real, env/config-driven external "
            "action, and only when the gate ALLOWED. The work executes as a "
            "Metaflow run through Loom's MLOps interface and produces an @card "
            "(what would deploy, the GATE decision, lineage). Irreversible/external "
            "tier -- always gated; never model-auto-invoked. Give exactly one of "
            "--validate or --solution."
        ),
    )
    deploy_group = deploy_parser.add_mutually_exclusive_group(required=True)
    deploy_group.add_argument(
        "--validate",
        dest="validate_run",
        default=None,
        metavar="RUN",
        help="Pathspec of the upstream validate run whose VERDICT gates deploy (e.g. ValidateFlow/12).",
    )
    deploy_group.add_argument(
        "--solution",
        dest="solution_run",
        default=None,
        metavar="RUN",
        help="Pathspec of a solution run to promote (read for a validate report).",
    )
    deploy_parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Perform the real external deploy (OFF by default; the gate must ALLOW "
            "and the upstream validate VERDICT must be PASS)."
        ),
    )
    deploy_parser.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Optional path to a YAML config file (for the Metaflow profile).",
    )
    deploy_parser.set_defaults(func=_cmd_deploy)

    ops_parser = subparsers.add_parser(
        "ops",
        help="Monitor run health, the leaderboard, and data drift (read-only) -> run + @card.",
        description=(
            "Run the read-only monitoring flow: recent run health + success/"
            "failure counts for a flow or experiment, its leaderboard, and a "
            "simple data-object DRIFT check (compare a data object's schema / "
            "summary stats to a reference data object). The work executes as a "
            "Metaflow run through Loom's MLOps interface and produces an @card "
            "(run health, leaderboard, drift table); Loom reads everything via the "
            "Client API and never touches the datastore. Read-only tier -- trains "
            "nothing, writes nothing back, never prompts. Give a --flow, an "
            "--experiment, or a --dataset + --reference drift pair."
        ),
    )
    ops_parser.add_argument(
        "--flow",
        dest="flow_name",
        default=None,
        metavar="NAME",
        help="Flow name whose recent run health to read (e.g. ValidateFlow).",
    )
    ops_parser.add_argument(
        "--experiment",
        default=None,
        metavar="ID",
        help="Experiment id whose runs + leaderboard to read (the loom_experiment tag).",
    )
    ops_parser.add_argument(
        "--dataset",
        default=None,
        metavar="PATHSPEC",
        help="Data object pathspec to drift-check against --reference.",
    )
    ops_parser.add_argument(
        "--reference",
        default=None,
        metavar="PATHSPEC",
        help="Reference data object pathspec for the drift comparison (with --dataset).",
    )
    ops_parser.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Optional path to a YAML config file (for the Metaflow profile).",
    )
    ops_parser.set_defaults(func=_cmd_ops)

    collab_parser = subparsers.add_parser(
        "collab",
        help="Assemble a sanitized shareable bundle of a run -> run + @card (SEND off by default).",
        description=(
            "Run the collaboration flow to assemble a sanitized, shareable bundle "
            "of a run: its report/model-card + a lineage manifest (pathspecs + "
            "fingerprints + commit) as a Metaflow run + @card. Everything is "
            "sanitized (no secrets, no raw rows). The off-box SEND is OFF by "
            "default: the default run builds the bundle only -- no data leaves the "
            "box. Only --send pushes to the env/config-driven sink "
            "(LOOM_COLLAB_WEBHOOK or LOOM_COLLAB_OUTBOX -- never a hardcoded "
            "target), and the send is the irreversible/external action. The work "
            "executes as a Metaflow run through Loom's MLOps interface; Loom reads "
            "the run only via the Client API. Build = workspace-write; send = "
            "irreversible/external (always gated; never model-auto-invoked). Give "
            "exactly one of --run or --experiment."
        ),
    )
    collab_group = collab_parser.add_mutually_exclusive_group(required=True)
    collab_group.add_argument(
        "--run",
        dest="run_pathspec",
        default=None,
        metavar="PATHSPEC",
        help="Pathspec of the run whose report/card to bundle (e.g. ValidateFlow/12).",
    )
    collab_group.add_argument(
        "--experiment",
        default=None,
        metavar="ID",
        help="Experiment id to bundle (alternative to --run).",
    )
    collab_parser.add_argument(
        "--send",
        action="store_true",
        help=(
            "Send the bundle off-box to the env/config-driven sink (OFF by "
            "default; build-only otherwise)."
        ),
    )
    collab_parser.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Optional path to a YAML config file (for the Metaflow profile).",
    )
    collab_parser.set_defaults(func=_cmd_collab)

    train_parser = subparsers.add_parser(
        "train",
        help="Build a model through the model-builder seam (gated) -> a run + @card.",
        description=(
            "Run the model-training flow to build a model stated in Loom DS-intent "
            "vocabulary (objective / budget / backbone / metric). The training "
            "backend (the `local` torch-free PPMI+SVD CPU stand-in, or `nemo` -- a "
            "lowering compiler) is resolved by config (model_builder_provider); the "
            "skill speaks the interface and never names a backend. `pretrain` is "
            "launch-and-track (AIDE never tree-searches it -- use `loom optimize` "
            "for cheap scalars like tokenization/heads/embeddings). EXPENSIVE/MUTATE "
            "tier: the cost PLAN (hours / $ / GPU-count for the budget) is surfaced "
            "at the gate, and the real heavy GPU launch is OFF by default behind "
            "--launch (the deploy --apply posture); with no GPU target it refuses "
            "cleanly (REFUSED_NO_GPU_TARGET) without launching. The work executes as "
            "a Metaflow run through Loom's MLOps interface and produces an @card; "
            "Loom reads the data only via the Client API and never touches the "
            "datastore. The `local` adapter actually builds (backbone / "
            "IngestDataset-shaped embeddings) on CPU; the produced run's pathspec is "
            "a first-class dataset_ref."
        ),
    )
    train_parser.add_argument(
        "--dataset",
        required=True,
        metavar="PATHSPEC",
        help=(
            "Metaflow pathspec of the sequences data object (e.g. IngestDataset/123 "
            "from `loom ingest`) to build the model from."
        ),
    )
    train_parser.add_argument(
        "--objective",
        default=None,
        metavar="OBJ",
        help="Pretraining objective: next-event | masked-field | contrastive (default next-event).",
    )
    train_parser.add_argument(
        "--budget",
        default=None,
        metavar="BUDGET",
        help="Training budget: probe | small | full (default probe; physics at the gate).",
    )
    train_parser.add_argument(
        "--capability",
        default=None,
        metavar="CAP",
        help=(
            "Capability to build (default pretrain): pretrain | tokenize | finetune "
            "| embed | serve. /loom-train only invokes launch-and-track capabilities; "
            "a searchable one is redirected to `loom optimize`."
        ),
    )
    train_parser.add_argument(
        "--backbone",
        dest="backbone_ref",
        default=None,
        metavar="PATHSPEC",
        help="Pathspec of a frozen backbone for finetune/embed (e.g. TrainFlow/12).",
    )
    train_parser.add_argument(
        "--metric",
        default=None,
        metavar="STR",
        help="Evaluation metric the build is steered toward (default fraud-pr-auc).",
    )
    train_parser.add_argument(
        "--launch",
        action="store_true",
        help=(
            "Perform the real heavy GPU launch (OFF by default; needs a configured "
            "gpu_target -- with none it refuses cleanly without launching)."
        ),
    )
    train_parser.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Optional path to a YAML config file (for the Metaflow profile).",
    )
    train_parser.set_defaults(func=_cmd_train)

    datasets_parser = subparsers.add_parser(
        "datasets",
        help="List ingested Metaflow data objects (via the Client API).",
        description=(
            "List the data objects produced by `loom ingest`, read through the "
            "Metaflow Client API (IngestDataset runs tagged loom_dataset). Prints "
            "pathspec, name, and row count / column summary per data object. "
            "Read-only; never touches the datastore directly."
        ),
    )
    datasets_parser.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Optional path to a YAML config file (for the Metaflow profile).",
    )
    datasets_parser.set_defaults(func=_cmd_datasets)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Diagnose the local Loom + Metaflow datastore stack (read-only).",
        description=(
            "Run a READ-ONLY health check of the local stack and print PASS / "
            "WARN / FAIL per check with an actionable fix for each failure, then a "
            "one-line VERDICT. Checks: the Python venv + `import loom`; `import "
            "metaflow`; the datastore environment variables "
            "(METAFLOW_DEFAULT_DATASTORE / METAFLOW_DATASTORE_SYSROOT_S3 / "
            "METAFLOW_S3_ENDPOINT_URL / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY "
            "/ METAFLOW_DEFAULT_METADATA / METAFLOW_USER); datastore reachability "
            "(a socket probe to the METAFLOW_S3_ENDPOINT_URL host:port and/or a "
            "Metaflow Client API listing); and a `loom datasets`-style Client-API "
            "smoke that counts ingested data objects (zero is fine). Diagnoses "
            "only -- it never installs, mutates, or prompts, and never touches the "
            "datastore except through the Metaflow Client API or a TCP socket "
            "probe to the configured endpoint. Exit code 0 when no check FAILs."
        ),
    )
    doctor_parser.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Optional path to a YAML config file (for the Metaflow profile).",
    )
    doctor_parser.set_defaults(func=_cmd_doctor)

    proxy_parser = subparsers.add_parser(
        "proxy",
        help="Run the Loom model gateway (the central data-collection moat path).",
        description=(
            "Operate Loom's LLM gateway. The 'serve' action launches an "
            "Anthropic-passthrough server that authenticates callers by a "
            "Loom-issued LOOM_API_KEY, injects Loom's system prompt, forwards to "
            "the real Anthropic API with a server-side ANTHROPIC_API_KEY the "
            "caller never sees, and logs every call centrally (the moat corpus)."
        ),
    )
    proxy_sub = proxy_parser.add_subparsers(dest="proxy_action", metavar="<action>")
    proxy_serve = proxy_sub.add_parser(
        "serve",
        help="Launch the Loom gateway (uvicorn).",
        description=(
            "Launch the Loom gateway. Reads the server-side real vendor key "
            "(ANTHROPIC_API_KEY) and the accepted Loom key(s) (LOOM_API_KEY / "
            "LOOM_API_KEYS) from the environment; fails fast if the vendor key is "
            "missing. This is the server side of the opt-in '--model-provider "
            "loom-proxy' route."
        ),
    )
    proxy_serve.add_argument(
        "--host",
        default="127.0.0.1",
        metavar="HOST",
        help="Bind address (default 127.0.0.1 — loopback only until hosted).",
    )
    proxy_serve.add_argument(
        "--port",
        type=int,
        default=8088,
        metavar="PORT",
        help="Bind port (default 8088, matching the loom_api_base default).",
    )
    proxy_serve.set_defaults(func=_cmd_proxy_serve)
    # No action -> print the proxy help and exit non-zero (mirrors top-level).
    proxy_parser.set_defaults(func=_cmd_proxy, _proxy_parser=proxy_parser)

    skillopt_parser = subparsers.add_parser(
        "skillopt",
        help="Optimize a /loom-* SKILL.md against the learnings corpus (gated; proposes by default).",
        description=(
            "Run the self-improvement loop's OPTIMIZE stage over one verb's "
            "SKILL.md (the moat, design-spec §5). HiveMind captures the verb's "
            "learnings corpus (`learnings/rollouts.jsonl`, filtered to the "
            "`owned_by=general` IP boundary), then SkillOpt's deterministic scorer "
            "grades the incumbent SKILL.md and any candidate(s) on the 7-point "
            "acceptance contract (HARD) + corpus failure-mode coverage (SOFT) and "
            "applies a never-worse promotion GATE (parallel to `loom deploy`): the "
            "BEST hard-valid candidate is promoted ONLY if it beats the incumbent "
            "by a margin -- a contract violator or a regression can NEVER win. "
            "SAFE BY DEFAULT: with no --apply the command only PROPOSES -- it "
            "prints the corpus digest + the gate VERDICT + a unified diff and "
            "writes the winning text to a sidecar `skills/<verb>/SKILL.candidate.md`; "
            "the real in-place overwrite happens ONLY with --apply AND only when the "
            "gate ALLOWED (promoted). Every run appends a `command=\"skillopt\"` "
            "learnings audit row. The whole loop is deterministic / LLM-free; "
            "--propose is an OPTIONAL pluggable proposer that is a clearly-marked "
            "no-op when no model is configured. Read-only by default; --apply is the "
            "gated mutate (editing the shipped skill library)."
        ),
    )
    skillopt_parser.add_argument(
        "--verb",
        required=True,
        metavar="loom-eda|...",
        help=(
            "The /loom-* verb whose SKILL.md to optimize (with or without the "
            "`loom-` prefix; resolved to skills/<verb>/SKILL.md)."
        ),
    )
    skillopt_parser.add_argument(
        "--candidate",
        default=None,
        metavar="PATH",
        help=(
            "Path to a candidate SKILL.md text to score against the incumbent "
            "(the deterministic, LLM-free candidate source). Repeatable score "
            "input; mutually exclusive with --propose."
        ),
    )
    skillopt_parser.add_argument(
        "--propose",
        action="store_true",
        help=(
            "Ask the OPTIONAL pluggable LLM proposer for a candidate (a "
            "clearly-marked no-op/stub when no model is configured). Mutually "
            "exclusive with --candidate; with neither, the command just scores + "
            "reports the incumbent and the corpus digest."
        ),
    )
    skillopt_parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Overwrite skills/<verb>/SKILL.md IN PLACE -- ONLY when the gate "
            "PROMOTED a candidate (OFF by default; the default proposes a sidecar). "
            "This is the gated mutate of the shipped skill library."
        ),
    )
    skillopt_parser.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Optional path to a YAML config file (for the learnings path / owner).",
    )
    skillopt_parser.set_defaults(func=_cmd_skillopt)

    telemetry_parser = subparsers.add_parser(
        "telemetry",
        help="Inspect + export the distillation-grade trajectory telemetry corpus.",
        description=(
            "Operate Loom's telemetry layer -- the distillation-grade trajectory "
            "capture (events + the interaction-root trajectory model + the proxy "
            "LLM I/O), correlated by a trajectory_id and bridged to the LOOM-DS-1 "
            "corpus. The 'status' action is a read-only summary (event/trajectory "
            "counts, general-vs-tenant split, the OTel exporter state, the paths); "
            "'export' assembles the trajectories and writes a LOOM-DS-1 SFT/teacher "
            "dataset (general-only + content REDACTED BY DEFAULT); 'trace' shows one "
            "assembled trajectory. Content is redacted unless --with-content; the "
            "export trains ONLY on owned_by=general (the IP boundary)."
        ),
    )
    telemetry_sub = telemetry_parser.add_subparsers(
        dest="telemetry_action", metavar="<action>"
    )

    telemetry_status = telemetry_sub.add_parser(
        "status",
        help="Read-only telemetry summary (counts, IP split, OTel state, paths).",
        description=(
            "Print a read-only summary of the telemetry corpus: the telemetry / "
            "trajectories paths, the event count, the assembled-trajectory count, "
            "the general-vs-tenant-owned split (the IP boundary), and whether the "
            "optional OTel exporter is enabled/available. Reads only; never emits, "
            "exports, or prompts."
        ),
    )
    telemetry_status.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Optional path to a YAML config file (for the telemetry paths).",
    )
    telemetry_status.set_defaults(func=_cmd_telemetry_status)

    telemetry_export = telemetry_sub.add_parser(
        "export",
        help="Assemble trajectories -> a LOOM-DS-1 dataset JSONL (general-only, redacted).",
        description=(
            "Assemble the correlated telemetry events + proxy LLM calls + "
            "command-level rollouts into ordered trajectories, then build the "
            "LOOM-DS-1 SFT/teacher distillation dataset and write it as JSONL. The "
            "IP boundary is enforced: only owned_by==general trajectories are "
            "included (--owned-by overrides the filter). Content (prompts/outputs) "
            "is REDACTED BY DEFAULT to typed sentinels unless --with-content is "
            "passed. Workspace-write: the only thing it writes is the export file."
        ),
    )
    telemetry_export.add_argument(
        "--owned-by",
        default="general",
        metavar="OWNER",
        help="IP-owner tag a trajectory must carry to be exported (default general).",
    )
    telemetry_export.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Output JSONL path for the dataset (default telemetry/loom-ds-1.jsonl).",
    )
    telemetry_export.add_argument(
        "--to-dataset",
        dest="to_dataset",
        default=None,
        metavar="NAME",
        help=(
            "Also INGEST the assembled corpus as a VERSIONED, content-addressed "
            "Metaflow data object under NAME (via the same IngestDataset seam as "
            "`loom ingest`) and print its pathspec -- the durable, lossless, "
            "no-sampling sink for scale. The --out JSONL is still written. Needs "
            "the metaflow MLOps provider."
        ),
    )
    telemetry_export.add_argument(
        "--with-content",
        action="store_true",
        help=(
            "Emit raw prompt/output content instead of the redaction sentinels "
            "(OFF by default; the operator must opt in)."
        ),
    )
    telemetry_export.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Optional path to a YAML config file (for the telemetry paths).",
    )
    telemetry_export.set_defaults(func=_cmd_telemetry_export)

    telemetry_trace = telemetry_sub.add_parser(
        "trace",
        help="Show one assembled trajectory by id (read-only).",
        description=(
            "Assemble and print a single trajectory by its trajectory_id (the "
            "experiment id when pinned): the task context, the ordered steps "
            "(LLM calls / tool / exec), and the terminal outcome + reward. "
            "Read-only; content stays redacted unless --with-content."
        ),
    )
    telemetry_trace.add_argument(
        "--trajectory",
        required=True,
        metavar="ID",
        help="The trajectory id to show (e.g. an experiment id like loom-abc123).",
    )
    telemetry_trace.add_argument(
        "--with-content",
        action="store_true",
        help="Show raw content instead of the redaction sentinels (OFF by default).",
    )
    telemetry_trace.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Optional path to a YAML config file (for the telemetry paths).",
    )
    telemetry_trace.set_defaults(func=_cmd_telemetry_trace)

    # No action -> print the telemetry help and exit non-zero (mirrors proxy).
    telemetry_parser.set_defaults(
        func=_cmd_telemetry, _telemetry_parser=telemetry_parser
    )

    return parser


def _build_config(args: argparse.Namespace) -> LoomConfig:
    """Build a :class:`LoomConfig`, layering CLI flags over env/.env/YAML.

    The base config is loaded from the environment (and an optional YAML file),
    then the explicit CLI flags (``--mlops``, ``--search``, ``--steps``) are
    applied as highest-precedence overrides. Unset flags leave the
    env/file-derived values untouched.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The fully resolved configuration for this run. No secret material is
        read onto it; adapters consume keys/endpoints from the environment.
    """
    overrides: dict[str, object] = {}
    # Use getattr so this helper also serves subcommands (e.g. ``ingest``) whose
    # parser does not define the run-only flags.
    if getattr(args, "mlops", None) is not None:
        overrides["mlops_provider"] = args.mlops
    if getattr(args, "search", None) is not None:
        overrides["search_provider"] = args.search
    if getattr(args, "steps", None) is not None:
        overrides["budget"] = {"steps": args.steps}

    # Model providers: --model-provider sets both roles; the per-role flags, when
    # given, take precedence (so --model-provider X --feedback-provider Y works).
    both = getattr(args, "model_provider", None)
    code_provider = getattr(args, "code_provider", None) or both
    feedback_provider = getattr(args, "feedback_provider", None) or both
    if code_provider is not None:
        overrides["code_provider"] = code_provider
    if feedback_provider is not None:
        overrides["feedback_provider"] = feedback_provider

    # ``--config`` is a subcommand flag; the top-level (no-subcommand REPL launch)
    # and ``chat`` namespaces do not define it, so read it defensively like the
    # other subcommand-specific flags above.
    return LoomConfig.load(yaml_path=getattr(args, "config", None), overrides=overrides)


# Substrings that suggest an exception is really a missing/invalid LLM credential.
_AUTH_ERROR_MARKERS = (
    "api_key",
    "auth",
    "authentication",
    "credential",
    "unauthorized",
    "401",
)


def _llm_preflight(config: LoomConfig) -> list[str]:
    """Return hints for any missing LLM credentials/login for the model providers.

    Delegates to the resolved :class:`~loom.providers.model.ModelProvider` for
    each role (``"code"`` and ``"feedback"``): each provider knows exactly which
    credential or CLI login its route needs. The search brain calls an LLM to
    write solutions, so a run cannot succeed without a working model backend.
    Best-effort: an empty list means no obvious problem.

    Args:
        config: The resolved Loom configuration.

    Returns:
        Deduplicated hint strings from the providers (empty if all look ready).
    """
    from loom.registry import get_model

    hints: list[str] = []
    seen: set[str] = set()
    pairs = (
        (config.code_provider, "code"),
        (config.feedback_provider, "feedback"),
    )
    for provider_name, role in pairs:
        try:
            provider = get_model(provider_name)(config)
        except Exception as exc:  # noqa: BLE001 - surface as an actionable hint
            hints.append(f"model provider {provider_name!r} for the {role} role "
                         f"could not be loaded: {exc}")
            continue
        for hint in provider.preflight(role):
            if hint not in seen:
                seen.add(hint)
                hints.append(hint)
    return hints


def _judge_preflight(config: LoomConfig) -> Optional[str]:
    """Return a fail-fast message if the feedback route cannot serve the judge.

    AIDE's feedback judge (``submit_review``) always calls the backend with a
    ``func_spec`` (tool/function calling). A route whose ``judge_capable`` is
    ``False`` will crash mid-run, so fail fast with an actionable message.

    Args:
        config: The resolved Loom configuration.

    Returns:
        An error message string if the feedback route is not judge-capable, else
        ``None``.
    """
    from loom.registry import get_model

    try:
        provider = get_model(config.feedback_provider)(config)
        route = provider.resolve("feedback")
    except Exception:  # noqa: BLE001 - the credential preflight reports load errors
        return None

    if route.judge_capable:
        return None

    # OpenRouter routes through AIDE's OpenAI-compatible backend only for
    # non-reserved (provider/model) slugs; a bad-shaped feedback slug is a
    # distinct failure mode from "model lacks tool calling", so surface the
    # provider's precise shape message when that is the cause.
    if config.feedback_provider == "openrouter":
        try:
            from loom.providers.model.openrouter import feedback_slug_shape_error

            shape_error = feedback_slug_shape_error(config.feedback_model)
        except Exception:  # noqa: BLE001 - fall back to the generic message below
            shape_error = None
        if shape_error is not None:
            return f"the feedback provider 'openrouter' cannot run the judge: {shape_error}"

    hint = (
        f"the feedback provider {config.feedback_provider!r} resolves to a route "
        f"that cannot run the judge (model {route.model_name!r} lacks tool "
        "calling). AIDE's feedback step (submit_review) requires function/tool "
        "calling."
    )
    if config.feedback_provider == "openrouter":
        hint += (
            " For OpenRouter, pick a tool-capable feedback slug "
            "(see https://openrouter.ai/models?supported_parameters=tools)."
        )
    return hint


def _looks_like_auth_error(exc: BaseException) -> bool:
    """Heuristic: does ``exc`` look like a missing/invalid LLM credential?"""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _AUTH_ERROR_MARKERS)


def _format_metric(value: object) -> str:
    """Render a metric value for display, tolerating ``None``/non-floats."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_leaderboard_row(rank: int, row: dict) -> str:
    """Render one leaderboard row, tolerating both provider row shapes.

    Different execution providers emit differently-shaped run dicts and the CLI
    must render either without hard-coding one provider's keys:

    * the ``local`` provider (Loom corpus) emits AIDE-node rows with ``metric`` /
      ``node_id`` / ``stage`` (the search-native shape);
    * the ``metaflow`` provider emits Metaflow run rows with ``run_id`` (a
      pathspec) / ``submission_ok`` / ``exec_time`` / ``exc_type`` and no metric
      (the flow records execution outcome, not a scored search metric).

    A row is treated as the search shape when it carries a ``metric``,
    ``node_id``, or ``stage`` key; otherwise it is rendered as a Metaflow run row
    (pathspec + submission flag + exec time). This avoids the old "metric=n/a
    node=?" output for ``--mlops metaflow`` while keeping the local leaderboard
    unchanged.

    Args:
        rank: 1-based display rank.
        row: One run dict from an execution provider's ``runs()``.

    Returns:
        A single formatted leaderboard line (no trailing newline).
    """
    prefix = f"  {rank:>2}. "

    # Search-native shape (local provider / AIDE corpus): prefer metric/node/stage.
    if "metric" in row or "node_id" in row or "stage" in row:
        metric = _format_metric(row.get("metric"))
        node_id = row.get("node_id", row.get("id", "?"))
        stage = row.get("stage", "")
        suffix = f"  [{stage}]" if stage else ""
        return f"{prefix}metric={metric}  node={node_id}{suffix}"

    # Metaflow run shape: a pathspec + submission flag + exec time. There is no
    # scored metric here, so report the run's execution outcome instead.
    pathspec = row.get("run_id", row.get("pathspec", "?"))
    submission_ok = bool(row.get("submission_ok", False))
    exec_time = row.get("exec_time")
    sub = "ok" if submission_ok else "no"
    parts = [f"{prefix}run={pathspec}", f"submission={sub}"]
    if isinstance(exec_time, (int, float)):
        parts.append(f"exec_time={float(exec_time):.6g}s")
    exc_type = row.get("exc_type")
    if exc_type:
        parts.append(f"exc={exc_type}")
    return "  ".join(parts)


def _print_result(result: SearchResult, leaderboard: list[dict]) -> None:
    """Print the best metric, artifact paths, and a short leaderboard.

    Args:
        result: The search result returned by the controller.
        leaderboard: Ranked run dicts from the execution provider's ``runs()``.
    """
    print("Loom run complete.")
    print(f"  best metric : {_format_metric(result.best_metric)}")
    print(f"  nodes       : {result.node_count}")
    print(f"  journal     : {result.journal_path or 'n/a'}")
    print(f"  tree        : {result.tree_path or 'n/a'}")

    if result.best_code:
        line_count = result.best_code.count("\n") + 1
        print(f"  best code   : {line_count} line(s) found")
    else:
        print("  best code   : none produced")

    if leaderboard:
        print("")
        print(f"Leaderboard (top {min(_LEADERBOARD_LIMIT, len(leaderboard))}):")
        for rank, row in enumerate(leaderboard[:_LEADERBOARD_LIMIT], start=1):
            print(_format_leaderboard_row(rank, row))


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Handle ``loom ingest``: run IngestDataset and print the data-object pathspec.

    Runs :class:`flows.ingest_dataset.IngestDataset` once via ``metaflow.Runner``
    (the one external->Metaflow boundary), tagging the run ``loom_dataset:<name>``,
    and prints the resulting **pathspec** (``IngestDataset/<run_id>``) -- the
    ``dataset_ref`` to pass to ``loom run --dataset <pathspec>``. Loom never
    touches the underlying datastore; Metaflow persists the artifacts per the
    active profile.

    Args:
        args: Parsed command-line arguments for the ``ingest`` subcommand.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    source = os.path.abspath(args.source)
    if not os.path.exists(source):
        print(f"error: --source path does not exist: {source}", file=sys.stderr)
        return 2

    config = _build_config(args)
    name = (args.name or "").strip()

    print(f"Ingesting {source!r} into a Metaflow data object...")
    pathspec, error = _ingest_source(source, name, config)
    if error is not None:
        print(f"error: {error}", file=sys.stderr)
        # Metaflow-absent is a setup/config problem (exit 2); a flow failure is a
        # runtime failure (exit 1), preserving the original exit codes.
        return 2 if "Metaflow is required" in error else 1

    print("Ingest complete.")
    print(f"  dataset_ref : {pathspec}")
    print("")
    print("Run a task against it with:")
    print(f"  loom run --dataset {pathspec} --mlops metaflow --goal '...' --metric '...'")
    return 0


def _ingest_source(
    source: str, name: str, config: LoomConfig
) -> tuple[Optional[str], Optional[str]]:
    """Run :class:`flows.ingest_dataset.IngestDataset` on ``source`` (the seam).

    The single, reusable external->Metaflow ingest boundary -- shared by
    ``loom ingest`` and ``loom telemetry export --to-dataset`` so the corpus is
    persisted as a **versioned, content-addressed Metaflow data object** through
    exactly the path :func:`_cmd_ingest` uses (``metaflow.Runner`` +
    ``IngestDataset`` + the resulting pathspec). Metaflow owns the datastore; Loom
    never touches it.

    Args:
        source: Absolute path to the local dir/CSV to ingest once.
        name: Optional dataset name (also applied as a ``loom_dataset:<name>`` tag).
        config: The active configuration (for the Metaflow profile).

    Returns:
        ``(pathspec, None)`` on success, or ``(None, error_message)`` on failure
        (Metaflow absent, the Runner raising, or a non-successful flow status).
    """
    # Lazy import: Metaflow is an optional dependency, pulled in only when an
    # operation that needs it (ingest) is actually invoked.
    try:
        from metaflow import Runner
    except Exception as exc:  # noqa: BLE001 - actionable hint, no traceback
        return None, (
            f"Metaflow is required to ingest a data object but could not be "
            f"imported: {exc}\nInstall it (it ships with Loom's deps) and try "
            "again."
        )

    from flows import INGEST_DATASET_FLOW_PATH

    runner_kwargs: dict[str, object] = {}
    if config.metaflow_profile:
        runner_kwargs["profile"] = config.metaflow_profile

    run_kwargs: dict[str, object] = {"source": source}
    if name:
        # The flow's name Parameter is id'd ``dataset_name`` (``name`` is reserved
        # by Metaflow's CLI options).
        run_kwargs["dataset_name"] = name
        run_kwargs["tags"] = [f"loom_dataset:{name}"]

    try:
        with Runner(
            INGEST_DATASET_FLOW_PATH,
            show_output=False,
            **runner_kwargs,
        ) as runner:
            executing = runner.run(**run_kwargs)
            status = getattr(executing, "status", None)
            run = executing.run
    except Exception as exc:  # noqa: BLE001 - surface as an actionable message
        return None, (
            f"failed to run IngestDataset flow: {type(exc).__name__}: {exc}"
        )

    # The Runner returns once the subprocess exits; a failed flow still yields an
    # ExecutingRun, so check status before claiming success (otherwise we would
    # return a pathspec whose run has no data object).
    if status is not None and status != "successful":
        return None, (
            f"IngestDataset did not complete successfully (status={status!r}). "
            "Re-run with the Metaflow logs for details."
        )

    pathspec = getattr(run, "pathspec", None) or str(run)
    return pathspec, None


def _print_eda_summary(dataset_ref: str, result: object) -> None:
    """Print a compact EDA profile summary + the ``@card`` reference.

    Args:
        dataset_ref: The profiled data object's pathspec.
        result: The :class:`~loom.types.RunResult` returned by the MLOps
            interface's ``run_flow``.
    """
    summary = getattr(result, "summary", None) or {}
    print("Loom EDA complete.")
    print(f"  dataset_ref : {dataset_ref}")
    print(f"  run         : {getattr(result, 'pathspec', None) or 'n/a'}")
    print(f"  card        : {getattr(result, 'card_path', None) or 'n/a'}")

    if summary:
        nrows = summary.get("nrows", "?")
        ncols = summary.get("ncols", "?")
        target = summary.get("target")
        inferred = " (inferred)" if summary.get("target_inferred") else ""
        print(f"  rows x cols : {nrows} x {ncols}")
        print(f"  target      : {target if target is not None else 'none'}{inferred}")

        balance = summary.get("target_balance")
        if balance:
            shown = list(balance.items())[:5]
            pretty = ", ".join(f"{k}={v}" for k, v in shown)
            more = "" if len(balance) <= 5 else f" (+{len(balance) - 5} more)"
            print(f"  balance     : {pretty}{more}")

        flags = summary.get("leakage_flags") or []
        if flags:
            print(f"  LEAKAGE     : {len(flags)} flag(s) -- review before features:")
            for flag in flags[:10]:
                print(
                    f"    - {flag.get('column')} [{flag.get('kind')}]: "
                    f"{flag.get('detail')}"
                )
        else:
            print("  leakage     : none detected")


def _cmd_eda(args: argparse.Namespace) -> int:
    """Handle ``loom eda``: profile a data object via the MLOps interface.

    Resolves the configured MLOps execution provider, runs the read-only
    :class:`flows.eda.EdaFlow` through its ``run_flow`` seam (never importing a
    concrete backend or touching the datastore), prints a profile summary + the
    ``@card`` reference, and appends a ``command="eda"`` learnings row. EDA is the
    read-only tier of the approval matrix: it never prompts.

    Args:
        args: Parsed arguments for the ``eda`` subcommand.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    from flows import EDA_FLOW_PATH

    config = _build_config(args)
    dataset_ref = (args.dataset or "").strip()
    target = (getattr(args, "target", None) or "").strip() or None

    if not dataset_ref:
        print(
            "error: --dataset is required (a `loom ingest` pathspec, e.g. "
            "IngestDataset/123).",
            file=sys.stderr,
        )
        return 2

    try:
        execution = get_execution(config.mlops_provider)(config)
    except Exception as exc:  # noqa: BLE001 - actionable hint
        print(
            f"error: could not load the MLOps provider "
            f"{config.mlops_provider!r}: {exc}",
            file=sys.stderr,
        )
        return 2

    tags = [
        "loom_command:eda",
        f"loom_dataset_ref:{dataset_ref}",
        f"loom_tenant:{config.tenant}",
        f"loom_owned_by:{config.owned_by}",
    ]

    print(f"Profiling data object {dataset_ref!r} (read-only)...")
    try:
        result = execution.run_flow(
            EDA_FLOW_PATH,
            {"dataset_ref": dataset_ref, "target": target},
            tags=tags,
        )
    except NotImplementedError as exc:
        # The 'local' provider cannot run lifecycle flows; guide to metaflow.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface as an actionable message
        print(
            f"error: failed to run the EDA flow: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    if not getattr(result, "successful", False):
        print(
            f"error: the EDA run did not complete successfully: "
            f"{getattr(result, 'error', None) or 'unknown error'}",
            file=sys.stderr,
        )
        # Still record the failed rollout below so the flywheel sees it.

    _print_eda_summary(dataset_ref, result)
    _record_eda_learning(config, dataset_ref, target, result)

    return 0 if getattr(result, "successful", False) else 1


def _record_eda_learning(
    config: LoomConfig,
    dataset_ref: str,
    target: Optional[str],
    result: object,
) -> None:
    """Append one ``command="eda"`` learnings row for this profile run.

    Best-effort and sanitized: the row carries only references (the data-object
    pathspec, the run pathspec/card) and small derived flags (leakage, target) --
    never raw rows or secrets. A failure to record never fails the command.

    Args:
        config: The active Loom configuration.
        dataset_ref: The profiled data object's pathspec.
        target: The declared target column, or ``None``.
        result: The :class:`~loom.types.RunResult` from ``run_flow``.
    """
    try:
        from loom.learnings import Learnings, LearningRecord, Outcome, TaskSpec

        summary = getattr(result, "summary", None) or {}
        successful = bool(getattr(result, "successful", False))
        leakage = bool(summary.get("leakage"))
        flag_kinds = sorted(
            {str(f.get("kind")) for f in (summary.get("leakage_flags") or [])}
        )

        artifacts = [
            ref
            for ref in (
                getattr(result, "pathspec", None),
                getattr(result, "card_path", None),
            )
            if ref
        ]

        record = LearningRecord(
            command="eda",
            task=TaskSpec(
                data_ref=dataset_ref,
                goal="profile the data object (read-only EDA)",
                metric="n/a (read-only profiling)",
                experiment_id=dataset_ref,
            ),
            inputs={
                "target": target,
                "mlops_provider": config.mlops_provider,
                "leakage": leakage,
                "leakage_flag_kinds": flag_kinds,
                "resolved_target": summary.get("target"),
                "nrows": summary.get("nrows"),
                "ncols": summary.get("ncols"),
            },
            outcome=Outcome(
                best_metric=None,
                submission_ok=successful,
                node_count=0,
            ),
            artifacts=artifacts,
            success=successful,
            model=None,
            tenant=config.tenant,
            owned_by=config.owned_by,
            reflection=(
                "leakage flags present; review before features"
                if leakage
                else None
            ),
        )
        Learnings(config).record(record)
    except Exception:  # noqa: BLE001 - learnings are best-effort, never fatal
        pass


def _print_validate_summary(dataset_ref: str, result: object) -> None:
    """Print a compact validation summary + the ``@card`` reference.

    Args:
        dataset_ref: The validated data object's pathspec.
        result: The :class:`~loom.types.RunResult` returned by ``run_flow``.
    """
    summary = getattr(result, "summary", None) or {}
    print("Loom validation complete.")
    print(f"  dataset_ref : {dataset_ref}")
    print(f"  run         : {getattr(result, 'pathspec', None) or 'n/a'}")
    print(f"  card        : {getattr(result, 'card_path', None) or 'n/a'}")

    if summary:
        cv = summary.get("cv") or {}
        holdout = summary.get("holdout") or {}
        metric = summary.get("metric", "score")
        verdict = summary.get("verdict", "?")
        print(f"  target      : {summary.get('target')} ({summary.get('task_type')})")
        cv_mean = cv.get("mean")
        cv_std = cv.get("std")
        if cv_mean is not None:
            print(
                f"  CV {metric:<8}: {cv_mean:.6g}"
                + (f" +/- {cv_std:.4g}" if cv_std is not None else "")
                + f" ({summary.get('n_folds')}-fold)"
            )
        if holdout.get("score") is not None:
            print(
                f"  holdout     : {holdout.get('score'):.6g} "
                f"(sealed, n={holdout.get('n')})"
            )
        cal = summary.get("calibration")
        if cal and cal.get("brier") is not None:
            print(f"  calibration : Brier={cal.get('brier')}")
        slices = summary.get("slice_metrics")
        if slices:
            shown = ", ".join(
                f"{g}={m.get('score')}" for g, m in list(slices.items())[:5]
            )
            print(f"  fairness    : {shown}")
        flags = summary.get("leakage_flags") or []
        if flags:
            print(f"  LEAKAGE     : {len(flags)} flag(s) -- explain before trusting:")
            for flag in flags[:10]:
                print(
                    f"    - {flag.get('column')} [{flag.get('kind')}]: "
                    f"{flag.get('detail')}"
                )
        print(f"  VERDICT     : {verdict}")


def _cmd_validate(args: argparse.Namespace) -> int:
    """Handle ``loom validate``: rigorously evaluate via the MLOps interface.

    Resolves the configured MLOps execution provider, runs the
    :class:`flows.validate.ValidateFlow` through its ``run_flow`` seam (never
    importing a concrete backend or touching the datastore), prints a validation
    summary + the ``@card`` reference, and appends a ``command="validate"``
    learnings row. Validate is the workspace-write tier: the read-only evaluation
    runs in its own Metaflow workspace and does not prompt.

    Args:
        args: Parsed arguments for the ``validate`` subcommand.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    from flows import VALIDATE_FLOW_PATH

    config = _build_config(args)
    dataset_ref = (args.dataset or "").strip()
    target = (getattr(args, "target", None) or "").strip() or None
    solution = (getattr(args, "solution", None) or "").strip() or None
    sensitive = (getattr(args, "sensitive", None) or "").strip() or None

    if not dataset_ref:
        print(
            "error: --dataset is required (a `loom ingest` pathspec, e.g. "
            "IngestDataset/123).",
            file=sys.stderr,
        )
        return 2

    try:
        execution = get_execution(config.mlops_provider)(config)
    except Exception as exc:  # noqa: BLE001 - actionable hint
        print(
            f"error: could not load the MLOps provider "
            f"{config.mlops_provider!r}: {exc}",
            file=sys.stderr,
        )
        return 2

    tags = [
        "loom_command:validate",
        f"loom_dataset_ref:{dataset_ref}",
        f"loom_tenant:{config.tenant}",
        f"loom_owned_by:{config.owned_by}",
    ]

    print(f"Validating against data object {dataset_ref!r} (workspace-write)...")
    try:
        result = execution.run_flow(
            VALIDATE_FLOW_PATH,
            {
                "dataset_ref": dataset_ref,
                "target": target,
                "solution_run": solution,
                "sensitive": sensitive,
            },
            tags=tags,
        )
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface as an actionable message
        print(
            f"error: failed to run the validate flow: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    if not getattr(result, "successful", False):
        print(
            f"error: the validate run did not complete successfully: "
            f"{getattr(result, 'error', None) or 'unknown error'}",
            file=sys.stderr,
        )

    _print_validate_summary(dataset_ref, result)
    _record_validate_learning(config, dataset_ref, target, solution, result)

    return 0 if getattr(result, "successful", False) else 1


def _record_validate_learning(
    config: LoomConfig,
    dataset_ref: str,
    target: Optional[str],
    solution: Optional[str],
    result: object,
) -> None:
    """Append one ``command="validate"`` learnings row for this validation run.

    Best-effort and sanitized: the row carries only references (the data-object
    pathspec, the run pathspec/card, the evaluated solution ref) and small derived
    scalars (holdout score, verdict, leakage) -- never raw rows or secrets. A
    failure to record never fails the command.
    """
    try:
        from loom.learnings import Learnings, LearningRecord, Outcome, TaskSpec

        summary = getattr(result, "summary", None) or {}
        successful = bool(getattr(result, "successful", False))
        holdout = summary.get("holdout") or {}
        best_metric = holdout.get("score")
        leakage = bool(summary.get("leakage"))

        artifacts = [
            ref
            for ref in (
                getattr(result, "pathspec", None),
                getattr(result, "card_path", None),
            )
            if ref
        ]

        record = LearningRecord(
            command="validate",
            task=TaskSpec(
                data_ref=dataset_ref,
                goal="validate a baseline/solution (CV + sealed holdout + calibration)",
                metric=str(summary.get("metric") or "n/a"),
                experiment_id=solution or dataset_ref,
            ),
            inputs={
                "target": target,
                "solution_run": solution,
                "mlops_provider": config.mlops_provider,
                "task_type": summary.get("task_type"),
                "n_folds": summary.get("n_folds"),
                "holdout_fraction": summary.get("holdout_fraction"),
                "leakage": leakage,
                "verdict": summary.get("verdict"),
            },
            outcome=Outcome(
                best_metric=float(best_metric)
                if isinstance(best_metric, (int, float))
                else None,
                submission_ok=successful,
                node_count=0,
            ),
            artifacts=artifacts,
            success=successful,
            model=None,
            tenant=config.tenant,
            owned_by=config.owned_by,
            reflection=(
                "leakage flags present; explain the score before trusting"
                if leakage
                else None
            ),
        )
        Learnings(config).record(record)
    except Exception:  # noqa: BLE001 - learnings are best-effort, never fatal
        pass


def _print_report_summary(experiment_id: Optional[str], result: object) -> None:
    """Print a compact experiment-report summary + the ``@card`` reference."""
    summary = getattr(result, "summary", None) or {}
    print("Loom report complete.")
    print(f"  experiment  : {experiment_id or '(by pathspecs)'}")
    print(f"  run         : {getattr(result, 'pathspec', None) or 'n/a'}")
    print(f"  card        : {getattr(result, 'card_path', None) or 'n/a'}")

    if summary:
        print(
            f"  runs        : {summary.get('n_runs')} "
            f"({summary.get('n_successful')} successful)"
        )
        best = summary.get("best_metric")
        if best is not None:
            print(f"  best metric : {best} ({summary.get('best_run')})")
        spread = summary.get("metric_spread")
        if spread:
            print(
                f"  spread      : min={spread.get('min')} "
                f"mean={spread.get('mean')} max={spread.get('max')}"
            )
        leaderboard = summary.get("leaderboard") or []
        if leaderboard:
            print("  leaderboard :")
            for rank, row in enumerate(leaderboard[:_LEADERBOARD_LIMIT], start=1):
                metric = row.get("metric")
                metric_s = _format_metric(metric)
                print(f"    {rank:>2}. {row.get('pathspec', '?')}  metric={metric_s}")
        print(f"  VERDICT     : {summary.get('verdict')}")


def _cmd_report(args: argparse.Namespace) -> int:
    """Handle ``loom report``: assemble an experiment report via the MLOps interface.

    Resolves the configured MLOps execution provider, runs the read-only
    :class:`flows.report.ReportFlow` through its ``run_flow`` seam (never importing
    a concrete backend or touching the datastore), prints a report summary + the
    ``@card`` reference, and appends a ``command="report"`` learnings row.
    Read-only: it never prompts.

    Args:
        args: Parsed arguments for the ``report`` subcommand.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    from flows import REPORT_FLOW_PATH

    config = _build_config(args)
    experiment_id = (getattr(args, "experiment", None) or "").strip() or None
    runs = (getattr(args, "runs", None) or "").strip() or None

    try:
        execution = get_execution(config.mlops_provider)(config)
    except Exception as exc:  # noqa: BLE001 - actionable hint
        print(
            f"error: could not load the MLOps provider "
            f"{config.mlops_provider!r}: {exc}",
            file=sys.stderr,
        )
        return 2

    tags = [
        "loom_command:report",
        f"loom_tenant:{config.tenant}",
        f"loom_owned_by:{config.owned_by}",
    ]
    if experiment_id:
        tags.append(f"loom_experiment:{experiment_id}")

    target_desc = experiment_id or runs
    print(f"Assembling report for {target_desc!r} (read-only)...")
    try:
        result = execution.run_flow(
            REPORT_FLOW_PATH,
            {"experiment_id": experiment_id, "run_pathspecs": runs},
            tags=tags,
        )
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface as an actionable message
        print(
            f"error: failed to run the report flow: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    if not getattr(result, "successful", False):
        print(
            f"error: the report run did not complete successfully: "
            f"{getattr(result, 'error', None) or 'unknown error'}",
            file=sys.stderr,
        )

    _print_report_summary(experiment_id, result)
    _record_report_learning(config, experiment_id, runs, result)

    return 0 if getattr(result, "successful", False) else 1


def _record_report_learning(
    config: LoomConfig,
    experiment_id: Optional[str],
    runs: Optional[str],
    result: object,
) -> None:
    """Append one ``command="report"`` learnings row for this report run.

    Best-effort and sanitized: references + small scalars only (n_runs, best
    metric, verdict). A failure to record never fails the command.
    """
    try:
        from loom.learnings import Learnings, LearningRecord, Outcome, TaskSpec

        summary = getattr(result, "summary", None) or {}
        successful = bool(getattr(result, "successful", False))
        best_metric = summary.get("best_metric")

        artifacts = [
            ref
            for ref in (
                getattr(result, "pathspec", None),
                getattr(result, "card_path", None),
            )
            if ref
        ]

        record = LearningRecord(
            command="report",
            task=TaskSpec(
                data_ref=None,
                goal="assemble an experiment report (runs + metrics + lineage)",
                metric="n/a (read-only report)",
                experiment_id=experiment_id or (runs or "report"),
            ),
            inputs={
                "experiment_id": experiment_id,
                "run_pathspecs": runs,
                "mlops_provider": config.mlops_provider,
                "n_runs": summary.get("n_runs"),
                "n_successful": summary.get("n_successful"),
                "verdict": summary.get("verdict"),
            },
            outcome=Outcome(
                best_metric=float(best_metric)
                if isinstance(best_metric, (int, float))
                else None,
                submission_ok=successful,
                node_count=int(summary.get("n_runs") or 0),
            ),
            artifacts=artifacts,
            success=successful,
            model=None,
            tenant=config.tenant,
            owned_by=config.owned_by,
            reflection=None,
        )
        Learnings(config).record(record)
    except Exception:  # noqa: BLE001 - learnings are best-effort, never fatal
        pass


def _print_viz_summary(source_ref: str, result: object) -> None:
    """Print a compact viz summary + the ``@card`` reference."""
    summary = getattr(result, "summary", None) or {}
    print("Loom viz complete.")
    print(f"  source      : {source_ref}")
    print(f"  run         : {getattr(result, 'pathspec', None) or 'n/a'}")
    print(f"  card        : {getattr(result, 'card_path', None) or 'n/a'}")

    if summary:
        plots = summary.get("plots") or []
        print(f"  kind        : {summary.get('kind')} ({summary.get('source')})")
        if plots:
            names = ", ".join(p.get("name", "?") for p in plots)
            print(f"  plots       : {len(plots)} -- {names}")
        else:
            print("  plots       : none (no numeric columns / no scored runs)")


def _cmd_viz(args: argparse.Namespace) -> int:
    """Handle ``loom viz``: plot a data object or a run via the MLOps interface.

    Resolves the configured MLOps execution provider, runs the read-only
    :class:`flows.viz.VizFlow` through its ``run_flow`` seam (never importing a
    concrete backend or touching the datastore), prints a plot summary + the
    ``@card`` reference, and appends a ``command="viz"`` learnings row. Read-only:
    it never prompts.

    Args:
        args: Parsed arguments for the ``viz`` subcommand.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    from flows import VIZ_FLOW_PATH

    config = _build_config(args)
    dataset_ref = (getattr(args, "dataset", None) or "").strip() or None
    run_pathspec = (getattr(args, "run", None) or "").strip() or None
    target = (getattr(args, "target", None) or "").strip() or None
    kind = (getattr(args, "kind", None) or "").strip() or None

    try:
        execution = get_execution(config.mlops_provider)(config)
    except Exception as exc:  # noqa: BLE001 - actionable hint
        print(
            f"error: could not load the MLOps provider "
            f"{config.mlops_provider!r}: {exc}",
            file=sys.stderr,
        )
        return 2

    source_ref = dataset_ref or run_pathspec or ""
    tags = [
        "loom_command:viz",
        f"loom_tenant:{config.tenant}",
        f"loom_owned_by:{config.owned_by}",
    ]
    if dataset_ref:
        tags.append(f"loom_dataset_ref:{dataset_ref}")

    print(f"Plotting {source_ref!r} (read-only)...")
    try:
        result = execution.run_flow(
            VIZ_FLOW_PATH,
            {
                "dataset_ref": dataset_ref,
                "run_pathspec": run_pathspec,
                "target": target,
                "kind": kind,
            },
            tags=tags,
        )
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface as an actionable message
        print(
            f"error: failed to run the viz flow: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    if not getattr(result, "successful", False):
        print(
            f"error: the viz run did not complete successfully: "
            f"{getattr(result, 'error', None) or 'unknown error'}",
            file=sys.stderr,
        )

    _print_viz_summary(source_ref, result)
    _record_viz_learning(config, dataset_ref, run_pathspec, kind, result)

    return 0 if getattr(result, "successful", False) else 1


def _record_viz_learning(
    config: LoomConfig,
    dataset_ref: Optional[str],
    run_pathspec: Optional[str],
    kind: Optional[str],
    result: object,
) -> None:
    """Append one ``command="viz"`` learnings row for this plotting run.

    Best-effort and sanitized: references + small scalars only (the source ref,
    plot count). A failure to record never fails the command.
    """
    try:
        from loom.learnings import Learnings, LearningRecord, Outcome, TaskSpec

        summary = getattr(result, "summary", None) or {}
        successful = bool(getattr(result, "successful", False))
        source_ref = dataset_ref or run_pathspec
        plots = summary.get("plots") or []

        artifacts = [
            ref
            for ref in (
                getattr(result, "pathspec", None),
                getattr(result, "card_path", None),
            )
            if ref
        ]

        record = LearningRecord(
            command="viz",
            task=TaskSpec(
                data_ref=source_ref,
                goal="generate read-only plots of a data object or a run",
                metric="n/a (read-only viz)",
                experiment_id=source_ref or "viz",
            ),
            inputs={
                "dataset_ref": dataset_ref,
                "run_pathspec": run_pathspec,
                "kind": kind,
                "mlops_provider": config.mlops_provider,
                "n_plots": len(plots),
                "plot_kinds": [p.get("name") for p in plots],
            },
            outcome=Outcome(
                best_metric=None,
                submission_ok=successful,
                node_count=0,
            ),
            artifacts=artifacts,
            success=successful,
            model=None,
            tenant=config.tenant,
            owned_by=config.owned_by,
            reflection=None,
        )
        Learnings(config).record(record)
    except Exception:  # noqa: BLE001 - learnings are best-effort, never fatal
        pass


def _print_features_summary(dataset_ref: str, result: object) -> None:
    """Print a compact feature-build summary + the ``@card`` reference.

    Args:
        dataset_ref: The source data object's pathspec.
        result: The :class:`~loom.types.RunResult` returned by ``run_flow``.
    """
    summary = getattr(result, "summary", None) or {}
    pathspec = getattr(result, "pathspec", None)
    print("Loom feature build complete.")
    print(f"  source      : {dataset_ref}")
    # The produced run's pathspec is itself the NEW dataset_ref downstream verbs use.
    print(f"  new dataset : {pathspec or 'n/a'}")
    print(f"  card        : {getattr(result, 'card_path', None) or 'n/a'}")

    if summary:
        before = summary.get("n_features_before", "?")
        after = summary.get("n_features_after", "?")
        added = summary.get("n_added", "?")
        target = summary.get("target")
        print(f"  target      : {target if target is not None else 'none'}")
        print(f"  recipe      : {summary.get('recipe')}")
        print(f"  features    : {before} -> {after} (+{added})")
        dropped = summary.get("dropped_columns") or []
        if dropped:
            shown = ", ".join(str(c) for c in dropped[:10])
            print(f"  LEAKAGE     : dropped {len(dropped)} flagged column(s): {shown}")
        else:
            print("  leakage     : none dropped (none flagged upstream)")
        print(f"  fingerprint : {summary.get('fingerprint') or 'n/a'}")
        print(f"  VERDICT     : {summary.get('verdict', '?')}")


def _cmd_features(args: argparse.Namespace) -> int:
    """Handle ``loom features``: build engineered features via the MLOps interface.

    Resolves the configured MLOps execution provider, runs the
    :class:`flows.features.FeaturesFlow` through its ``run_flow`` seam (never
    importing a concrete backend or touching the datastore), prints a feature-build
    summary + the ``@card`` reference, and appends a ``command="features"`` learnings
    row. Features is the workspace-write tier: it reads the source read-only and
    writes a NEW data object only into this run's own Metaflow workspace (the
    produced run's pathspec is the ``dataset_ref`` for downstream verbs). It composes
    with ``loom eda`` via ``--from`` (the EDA run's leakage-flagged columns are
    dropped before building).

    Args:
        args: Parsed arguments for the ``features`` subcommand.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    from flows import FEATURES_FLOW_PATH

    config = _build_config(args)
    dataset_ref = (args.dataset or "").strip()
    target = (getattr(args, "target", None) or "").strip() or None
    eda_run = (getattr(args, "from_eda", None) or "").strip() or None
    recipe = (getattr(args, "recipe", None) or "").strip() or None

    if not dataset_ref:
        print(
            "error: --dataset is required (a `loom ingest` / `loom features` "
            "pathspec, e.g. IngestDataset/123).",
            file=sys.stderr,
        )
        return 2

    try:
        execution = get_execution(config.mlops_provider)(config)
    except Exception as exc:  # noqa: BLE001 - actionable hint
        print(
            f"error: could not load the MLOps provider "
            f"{config.mlops_provider!r}: {exc}",
            file=sys.stderr,
        )
        return 2

    tags = [
        "loom_command:features",
        f"loom_dataset_ref:{dataset_ref}",
        f"loom_tenant:{config.tenant}",
        f"loom_owned_by:{config.owned_by}",
    ]

    print(f"Building features from data object {dataset_ref!r} (workspace-write)...")
    try:
        result = execution.run_flow(
            FEATURES_FLOW_PATH,
            {
                "dataset_ref": dataset_ref,
                "target": target,
                "eda_run": eda_run,
                "recipe": recipe,
            },
            tags=tags,
        )
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface as an actionable message
        print(
            f"error: failed to run the features flow: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    if not getattr(result, "successful", False):
        print(
            f"error: the features run did not complete successfully: "
            f"{getattr(result, 'error', None) or 'unknown error'}",
            file=sys.stderr,
        )

    _print_features_summary(dataset_ref, result)
    _record_features_learning(config, dataset_ref, target, eda_run, recipe, result)

    return 0 if getattr(result, "successful", False) else 1


def _record_features_learning(
    config: LoomConfig,
    dataset_ref: str,
    target: Optional[str],
    eda_run: Optional[str],
    recipe: Optional[str],
    result: object,
) -> None:
    """Append one ``command="features"`` learnings row for this feature build.

    Best-effort and sanitized: references + small scalars only (the source ref, the
    new data object's pathspec, feature counts, the fingerprint, the leakage-drop
    flag) -- never raw rows or secrets. A failure to record never fails the command.
    """
    try:
        from loom.learnings import Learnings, LearningRecord, Outcome, TaskSpec

        summary = getattr(result, "summary", None) or {}
        successful = bool(getattr(result, "successful", False))
        dropped = summary.get("dropped_columns") or []

        artifacts = [
            ref
            for ref in (
                getattr(result, "pathspec", None),
                getattr(result, "card_path", None),
            )
            if ref
        ]

        record = LearningRecord(
            command="features",
            task=TaskSpec(
                data_ref=dataset_ref,
                goal="build engineered features into a new data object",
                metric="n/a (feature engineering)",
                experiment_id=dataset_ref,
            ),
            inputs={
                "target": target,
                "from_eda": eda_run,
                "recipe": recipe,
                "mlops_provider": config.mlops_provider,
                "n_features_before": summary.get("n_features_before"),
                "n_features_after": summary.get("n_features_after"),
                "n_added": summary.get("n_added"),
                "refused_leakage": bool(summary.get("refused_leakage")),
                "dropped_columns": list(dropped),
                "fingerprint": summary.get("fingerprint"),
            },
            outcome=Outcome(
                best_metric=None,
                submission_ok=successful,
                node_count=0,
            ),
            artifacts=artifacts,
            success=successful,
            model=None,
            tenant=config.tenant,
            owned_by=config.owned_by,
            reflection=(
                "leakage columns dropped before building (eda->features gate)"
                if dropped
                else None
            ),
        )
        Learnings(config).record(record)
    except Exception:  # noqa: BLE001 - learnings are best-effort, never fatal
        pass


def _print_pipeline_summary(dataset_ref: str, goal: str, result: object) -> None:
    """Print a compact end-to-end pipeline summary + the ``@card`` reference.

    Args:
        dataset_ref: The source data object's pathspec.
        goal: The natural-language goal driving the run.
        result: The :class:`~loom.types.RunResult` returned by ``run_flow``.
    """
    summary = getattr(result, "summary", None) or {}
    print("Loom pipeline complete.")
    print(f"  dataset_ref : {dataset_ref}")
    print(f"  goal        : {goal}")
    print(f"  run         : {getattr(result, 'pathspec', None) or 'n/a'}")
    print(f"  card        : {getattr(result, 'card_path', None) or 'n/a'}")

    if summary:
        print(f"  target      : {summary.get('target')}")
        print(f"  leakage     : {'handled' if summary.get('leakage') else 'none'}")
        stages = summary.get("stages") or {}
        if stages:
            order = ("profile", "features", "optimize", "validate")
            shown = [s for s in order if s in stages] or list(stages.keys())
            line = ", ".join(
                f"{s}={stages.get(s, {}).get('status', '?')}" for s in shown
            )
            print(f"  stages      : {line}")
        failed = summary.get("failed_stage")
        if failed:
            print(f"  failed stage: {failed}")
        print(f"  VERDICT     : {summary.get('verdict', '?')}")


def _cmd_pipeline(args: argparse.Namespace) -> int:
    """Handle ``loom pipeline``: run the end-to-end lifecycle via the MLOps interface.

    Resolves the configured MLOps execution provider, runs the
    :class:`flows.pipeline.PipelineFlow` through its ``run_flow`` seam (never
    importing a concrete backend or touching the datastore), prints a per-stage
    summary + the headline VERDICT + the ``@card`` reference, and appends a
    ``command="pipeline"`` learnings row. Pipeline is the workspace-write tier that
    escalates to EXPENSIVE at its bounded optimize stage; each stage asserts the
    prior stage's VERDICT (leakage handled before features; a sub-threshold validate
    marks the run FAIL).

    Args:
        args: Parsed arguments for the ``pipeline`` subcommand.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    from flows import PIPELINE_FLOW_PATH

    config = _build_config(args)
    dataset_ref = (args.dataset or "").strip()
    goal = (args.goal or "").strip()
    target = (getattr(args, "target", None) or "").strip() or None

    if not dataset_ref:
        print(
            "error: --dataset is required (a `loom ingest` / `loom features` "
            "pathspec, e.g. IngestDataset/123).",
            file=sys.stderr,
        )
        return 2

    try:
        execution = get_execution(config.mlops_provider)(config)
    except Exception as exc:  # noqa: BLE001 - actionable hint
        print(
            f"error: could not load the MLOps provider "
            f"{config.mlops_provider!r}: {exc}",
            file=sys.stderr,
        )
        return 2

    tags = [
        "loom_command:pipeline",
        f"loom_dataset_ref:{dataset_ref}",
        f"loom_tenant:{config.tenant}",
        f"loom_owned_by:{config.owned_by}",
    ]

    print(
        f"Running the end-to-end pipeline on {dataset_ref!r} "
        "(workspace-write -> EXPENSIVE at optimize)..."
    )
    try:
        result = execution.run_flow(
            PIPELINE_FLOW_PATH,
            {"dataset_ref": dataset_ref, "goal": goal, "target": target},
            tags=tags,
        )
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface as an actionable message
        print(
            f"error: failed to run the pipeline flow: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    if not getattr(result, "successful", False):
        print(
            f"error: the pipeline run did not complete successfully: "
            f"{getattr(result, 'error', None) or 'unknown error'}",
            file=sys.stderr,
        )

    _print_pipeline_summary(dataset_ref, goal, result)
    _record_pipeline_learning(config, dataset_ref, goal, target, result)

    return 0 if getattr(result, "successful", False) else 1


def _record_pipeline_learning(
    config: LoomConfig,
    dataset_ref: str,
    goal: str,
    target: Optional[str],
    result: object,
) -> None:
    """Append one ``command="pipeline"`` learnings row for this lifecycle run.

    Best-effort and sanitized: references + small scalars only (the data-object ref,
    the headline verdict, the failed stage, the validate holdout metric) -- never raw
    rows or secrets. A failure to record never fails the command.
    """
    try:
        from loom.learnings import Learnings, LearningRecord, Outcome, TaskSpec

        summary = getattr(result, "summary", None) or {}
        successful = bool(getattr(result, "successful", False))
        stages = summary.get("stages") or {}
        validate_summary = (stages.get("validate") or {}).get("summary") or {}
        best_metric = validate_summary.get("metric")
        leakage = bool(summary.get("leakage"))

        artifacts = [
            ref
            for ref in (
                getattr(result, "pathspec", None),
                getattr(result, "card_path", None),
            )
            if ref
        ]

        record = LearningRecord(
            command="pipeline",
            task=TaskSpec(
                data_ref=dataset_ref,
                goal=goal or "run the end-to-end lifecycle",
                metric=str(validate_summary.get("metric_name") or "n/a"),
                experiment_id=dataset_ref,
            ),
            inputs={
                "target": target,
                "mlops_provider": config.mlops_provider,
                "leakage": leakage,
                "failed_stage": summary.get("failed_stage"),
                "verdict": summary.get("verdict"),
                "optimize_budget": summary.get("optimize_budget"),
            },
            outcome=Outcome(
                best_metric=float(best_metric)
                if isinstance(best_metric, (int, float))
                else None,
                submission_ok=successful and summary.get("verdict") == "PASS",
                node_count=len(stages),
            ),
            artifacts=artifacts,
            success=successful,
            model=None,
            tenant=config.tenant,
            owned_by=config.owned_by,
            reflection=(
                f"pipeline failed at stage {summary.get('failed_stage')!r}"
                if summary.get("failed_stage")
                else None
            ),
        )
        Learnings(config).record(record)
    except Exception:  # noqa: BLE001 - learnings are best-effort, never fatal
        pass


def _print_deploy_summary(source_run: str, result: object) -> None:
    """Print a compact deployment-plan summary + the GATE decision + the ``@card``.

    Args:
        source_run: The upstream validate/solution run pathspec being promoted.
        result: The :class:`~loom.types.RunResult` returned by ``run_flow``.
    """
    summary = getattr(result, "summary", None) or {}
    print("Loom deploy complete.")
    print(f"  source run  : {source_run}")
    print(f"  run         : {getattr(result, 'pathspec', None) or 'n/a'}")
    print(f"  card        : {getattr(result, 'card_path', None) or 'n/a'}")

    if summary:
        gate = summary.get("gate") or {}
        decision = gate.get("decision", "?")
        print(f"  target      : {summary.get('target')}")
        print(f"  apply       : {summary.get('apply')} "
              f"(real external action {'ON' if summary.get('apply') else 'OFF — staged plan only'})")
        print(f"  upstream    : validate VERDICT={gate.get('verdict')}")
        print(f"  GATE        : {decision}")
        reasons = gate.get("reasons") or []
        if reasons:
            print("  blocked by  :")
            for reason in reasons[:10]:
                print(f"    - {reason}")
        applied = summary.get("applied_detail")
        if applied and applied.get("entry"):
            print(f"  registered  : {applied.get('entry')}")
        print(f"  VERDICT     : {summary.get('verdict', '?')}")


def _cmd_deploy(args: argparse.Namespace) -> int:
    """Handle ``loom deploy``: gate on the validate VERDICT and plan a deploy.

    Resolves the configured MLOps execution provider, runs the
    :class:`flows.deploy.DeployFlow` through its ``run_flow`` seam (never importing a
    concrete backend or touching the datastore), prints the deployment plan + the
    GATE decision + the headline VERDICT + the ``@card`` reference, and appends a
    ``command="deploy"`` learnings row. Deploy is the irreversible/external tier: it
    asserts the upstream ``loom validate`` VERDICT==PASS (a sub-threshold validate
    BLOCKS the deploy -- the cross-verb exit gate), and the real external action is
    behind ``--apply`` (OFF by default; default is a PLAN + staged register, no
    external mutation).

    Args:
        args: Parsed arguments for the ``deploy`` subcommand.

    Returns:
        Process exit code (0 on success, non-zero on failure). A clean run whose gate
        BLOCKED still returns the run's success code -- the BLOCK is the correct,
        machine-checkable outcome, surfaced in the summary's GATE/VERDICT.
    """
    from flows import DEPLOY_FLOW_PATH

    config = _build_config(args)
    validate_run = (getattr(args, "validate_run", None) or "").strip() or None
    solution_run = (getattr(args, "solution_run", None) or "").strip() or None
    apply = bool(getattr(args, "apply", False))

    source_run = validate_run or solution_run or ""

    try:
        execution = get_execution(config.mlops_provider)(config)
    except Exception as exc:  # noqa: BLE001 - actionable hint
        print(
            f"error: could not load the MLOps provider "
            f"{config.mlops_provider!r}: {exc}",
            file=sys.stderr,
        )
        return 2

    tags = [
        "loom_command:deploy",
        f"loom_tenant:{config.tenant}",
        f"loom_owned_by:{config.owned_by}",
    ]

    print(
        f"Planning deploy from {source_run!r} "
        f"(irreversible/external; apply={'ON' if apply else 'OFF — staged plan only'})..."
    )
    try:
        result = execution.run_flow(
            DEPLOY_FLOW_PATH,
            {
                "validate_run": validate_run,
                "solution_run": solution_run,
                "apply": apply,
            },
            tags=tags,
        )
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface as an actionable message
        print(
            f"error: failed to run the deploy flow: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    if not getattr(result, "successful", False):
        print(
            f"error: the deploy run did not complete successfully: "
            f"{getattr(result, 'error', None) or 'unknown error'}",
            file=sys.stderr,
        )

    _print_deploy_summary(source_run, result)
    _record_deploy_learning(config, source_run, apply, result)

    return 0 if getattr(result, "successful", False) else 1


def _record_deploy_learning(
    config: LoomConfig,
    source_run: str,
    apply: bool,
    result: object,
) -> None:
    """Append one ``command="deploy"`` learnings row for this deploy run.

    Best-effort and sanitized: references + small scalars only (the source run, the
    gate decision, the manifest status, the validate metric the gate trusted) --
    never raw rows or secrets. A failure to record never fails the command.
    """
    try:
        from loom.learnings import Learnings, LearningRecord, Outcome, TaskSpec

        summary = getattr(result, "summary", None) or {}
        successful = bool(getattr(result, "successful", False))
        gate = summary.get("gate") or {}
        allowed = bool(gate.get("allow"))
        manifest = summary.get("manifest") or {}
        metric_block = manifest.get("validate_metric") or {}
        holdout = metric_block.get("holdout")

        artifacts = [
            ref
            for ref in (
                getattr(result, "pathspec", None),
                getattr(result, "card_path", None),
            )
            if ref
        ]

        record = LearningRecord(
            command="deploy",
            task=TaskSpec(
                data_ref=None,
                goal="promote a validated solution (gated on validate VERDICT==PASS)",
                metric=str(metric_block.get("metric") or "n/a"),
                experiment_id=source_run or "deploy",
            ),
            inputs={
                "source_run": source_run,
                "apply": apply,
                "mlops_provider": config.mlops_provider,
                "gate_decision": gate.get("decision"),
                "gate_allow": allowed,
                "upstream_verdict": gate.get("verdict"),
                "manifest_status": summary.get("status"),
                "target": summary.get("target"),
            },
            outcome=Outcome(
                best_metric=float(holdout)
                if isinstance(holdout, (int, float))
                else None,
                # A deploy "succeeds" as a rollout only when the gate ALLOWED;
                # a clean run that BLOCKED is a correct refusal, not a submission.
                submission_ok=successful and allowed,
                node_count=0,
            ),
            artifacts=artifacts,
            success=successful,
            model=None,
            tenant=config.tenant,
            owned_by=config.owned_by,
            reflection=(
                "deploy BLOCKED by the exit gate: "
                + "; ".join(gate.get("reasons") or [])
                if not allowed
                else None
            ),
        )
        Learnings(config).record(record)
    except Exception:  # noqa: BLE001 - learnings are best-effort, never fatal
        pass


def _print_train_summary(dataset_ref: str, result: object) -> None:
    """Print a compact training summary + the gate STATUS + the ``@card`` reference.

    Surfaces the headline status line (PLAN / PLANNED / REFUSED_NO_GPU_TARGET /
    BUILT) the model-builder seam produced, plus the cost PLAN (physics at the
    gate), the produced artifact pathspec, and -- for the ``local`` adapter -- the
    backbone/embeddings fingerprint.

    Args:
        dataset_ref: The data object's pathspec the model was built from.
        result: The :class:`~loom.types.RunResult` returned by ``run_flow``.
    """
    summary = getattr(result, "summary", None) or {}
    print("Loom train complete.")
    print(f"  dataset_ref : {dataset_ref}")
    print(f"  run         : {getattr(result, 'pathspec', None) or 'n/a'}")
    print(f"  card        : {getattr(result, 'card_path', None) or 'n/a'}")

    if summary:
        print(
            f"  backend     : {summary.get('backend')} "
            f"(model_builder_provider {summary.get('model_builder_provider')})"
        )
        print(
            f"  capability  : {summary.get('capability')} "
            f"(mode {summary.get('capability_mode')})"
        )
        print(
            f"  objective   : {summary.get('objective')} "
            f"(budget {summary.get('budget')})"
        )
        cost = summary.get("cost") or {}
        if cost.get("headline"):
            print(f"  cost (gate) : {cost.get('headline')}")
        launch = summary.get("launch")
        print(
            f"  launch      : {launch} "
            f"(real GPU launch {'ON' if launch else 'OFF — plan only'}; "
            f"posture {summary.get('launch_posture')})"
        )
        if summary.get("gpu_target") is not None:
            print(f"  gpu_target  : {summary.get('gpu_target')}")
        artifact = summary.get("artifact_pathspec")
        print(f"  artifact    : {artifact or 'none'} ({summary.get('artifact_kind')})")
        if summary.get("fingerprint"):
            print(f"  fingerprint : {summary.get('fingerprint')}")
        if summary.get("error"):
            print(f"  refused     : {summary.get('error')}")
        # The headline status line (PLAN / PLANNED / REFUSED_NO_GPU_TARGET / BUILT).
        print(f"  STATUS      : {summary.get('status', '?')}")


def _cmd_train(args: argparse.Namespace) -> int:
    """Handle ``loom train``: build a model through the MLOps interface (gated).

    Resolves the configured MLOps execution provider, runs the
    :class:`flows.train.TrainFlow` through its ``run_flow`` seam (never importing a
    concrete model-builder backend or touching the datastore), prints a training
    summary + the gate STATUS + the ``@card`` reference, and appends a
    ``command="train"`` learnings row. Train is the EXPENSIVE/MUTATE, ALWAYS-GATE
    tier: ``pretrain`` is launch-and-track, the cost PLAN is surfaced at the gate,
    and the real heavy GPU launch is OFF by default behind ``--launch`` (with no
    GPU target it refuses cleanly without launching). The training backend is
    resolved by config (``model_builder_provider``); the CLI speaks the interface.

    Args:
        args: Parsed arguments for the ``train`` subcommand.

    Returns:
        Process exit code (0 on success, non-zero on failure). A clean run that
        cleanly PLANNED / REFUSED (no GPU target) still returns the run's success
        code -- the gate status is the correct outcome, surfaced in the summary.
    """
    from flows import TRAIN_FLOW_PATH

    config = _build_config(args)
    dataset_ref = (args.dataset or "").strip()
    objective = (getattr(args, "objective", None) or "").strip() or "next-event"
    budget = (getattr(args, "budget", None) or "").strip() or "probe"
    capability = (getattr(args, "capability", None) or "").strip() or "pretrain"
    backbone_ref = (getattr(args, "backbone_ref", None) or "").strip() or ""
    metric = (getattr(args, "metric", None) or "").strip() or "fraud-pr-auc"
    launch = bool(getattr(args, "launch", False))

    if not dataset_ref:
        print(
            "error: --dataset is required (a `loom ingest` pathspec, e.g. "
            "IngestDataset/123).",
            file=sys.stderr,
        )
        return 2

    try:
        execution = get_execution(config.mlops_provider)(config)
    except Exception as exc:  # noqa: BLE001 - actionable hint
        print(
            f"error: could not load the MLOps provider "
            f"{config.mlops_provider!r}: {exc}",
            file=sys.stderr,
        )
        return 2

    tags = [
        "loom_command:train",
        f"loom_dataset_ref:{dataset_ref}",
        f"loom_tenant:{config.tenant}",
        f"loom_owned_by:{config.owned_by}",
    ]

    print(
        f"Training a model from {dataset_ref!r} "
        f"(EXPENSIVE/MUTATE; launch={'ON' if launch else 'OFF — plan only'})..."
    )
    try:
        result = execution.run_flow(
            TRAIN_FLOW_PATH,
            {
                "dataset_ref": dataset_ref,
                "capability": capability,
                "objective": objective,
                "budget": budget,
                "backbone_ref": backbone_ref,
                "metric": metric,
                "launch": launch,
            },
            tags=tags,
        )
    except NotImplementedError as exc:
        # The 'local' dev MLOps provider cannot run lifecycle flows; guide to metaflow.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface as an actionable message
        print(
            f"error: failed to run the train flow: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    if not getattr(result, "successful", False):
        print(
            f"error: the train run did not complete successfully: "
            f"{getattr(result, 'error', None) or 'unknown error'}",
            file=sys.stderr,
        )

    _print_train_summary(dataset_ref, result)
    _record_train_learning(
        config, dataset_ref, capability, objective, budget, metric, launch, result
    )

    return 0 if getattr(result, "successful", False) else 1


def _record_train_learning(
    config: LoomConfig,
    dataset_ref: str,
    capability: str,
    objective: str,
    budget: str,
    metric: str,
    launch: bool,
    result: object,
) -> None:
    """Append one ``command="train"`` learnings row for this training run.

    Best-effort and sanitized: references + small derived scalars only (the data
    object pathspec, the produced artifact pathspec, the backend, the gate status,
    the cost line, the fingerprint) -- never raw rows or secrets. A failure to
    record never fails the command.

    Args:
        config: The active Loom configuration.
        dataset_ref: The data object pathspec the model was built from.
        capability: The invoked capability.
        objective: The pretraining objective.
        budget: The training budget.
        metric: The evaluation metric.
        launch: Whether the real heavy GPU launch was requested.
        result: The :class:`~loom.types.RunResult` from ``run_flow``.
    """
    try:
        from loom.learnings import Learnings, LearningRecord, Outcome, TaskSpec

        summary = getattr(result, "summary", None) or {}
        successful = bool(getattr(result, "successful", False))
        status = str(summary.get("status") or "")
        # A train "succeeds" as a build only when the local adapter actually built;
        # a clean PLANNED / REFUSED_NO_GPU_TARGET is a correct gate outcome, not a
        # produced artifact (mirrors the deploy BLOCK posture).
        built = status == "BUILT"
        cost = summary.get("cost") or {}

        artifacts = [
            ref
            for ref in (
                getattr(result, "pathspec", None),
                getattr(result, "card_path", None),
                summary.get("artifact_pathspec"),
            )
            if ref
        ]

        record = LearningRecord(
            command="train",
            task=TaskSpec(
                data_ref=dataset_ref,
                goal=f"build a model ({capability}) via the model-builder seam",
                metric=metric or "n/a",
                experiment_id=dataset_ref,
            ),
            inputs={
                "capability": capability,
                "objective": objective,
                "budget": budget,
                "metric": metric,
                "launch": launch,
                "mlops_provider": config.mlops_provider,
                "model_builder_provider": summary.get("model_builder_provider"),
                "backend": summary.get("backend"),
                "capability_mode": summary.get("capability_mode"),
                "launch_posture": summary.get("launch_posture"),
                "gpu_target": summary.get("gpu_target"),
                "est_usd": cost.get("est_usd"),
                "gpu_hours": cost.get("gpu_hours"),
                "status": status,
                "fingerprint": summary.get("fingerprint"),
            },
            outcome=Outcome(
                best_metric=None,
                submission_ok=successful and built,
                node_count=0,
            ),
            artifacts=artifacts,
            success=successful,
            model=None,
            tenant=config.tenant,
            owned_by=config.owned_by,
            reflection=(
                f"train did not produce an artifact (status={status}): "
                + str(summary.get("error") or "see gate status")
                if (successful and not built)
                else None
            ),
        )
        Learnings(config).record(record)
    except Exception:  # noqa: BLE001 - learnings are best-effort, never fatal
        pass


def _print_ops_summary(result: object) -> None:
    """Print a compact ops/monitoring summary + the ``@card`` reference.

    Args:
        result: The :class:`~loom.types.RunResult` returned by ``run_flow``.
    """
    summary = getattr(result, "summary", None) or {}
    print("Loom ops complete.")
    print(f"  run         : {getattr(result, 'pathspec', None) or 'n/a'}")
    print(f"  card        : {getattr(result, 'card_path', None) or 'n/a'}")

    if summary:
        health = summary.get("health") or {}
        scope = health.get("flow_name") or health.get("experiment_id") or "(none)"
        print(f"  scope       : {scope}")
        print(
            f"  runs        : {health.get('n_runs')} "
            f"({health.get('n_successful')} ok, {health.get('n_failed')} failed)"
        )
        rate = health.get("success_rate")
        print(f"  success rate: {rate if rate is not None else 'n/a'}")
        print(f"  run health  : {health.get('status')}")
        drift = summary.get("drift")
        if drift is not None:
            flags = drift.get("drift_flags") or []
            print(
                f"  drift       : {drift.get('status')} "
                f"({len(flags)} column(s) flagged; "
                f"+{len(drift.get('added') or [])} -{len(drift.get('removed') or [])} cols)"
            )
        print(f"  VERDICT     : {summary.get('status', '?')}")


def _cmd_ops(args: argparse.Namespace) -> int:
    """Handle ``loom ops``: read-only run/drift monitoring via the MLOps interface.

    Resolves the configured MLOps execution provider, runs the read-only
    :class:`flows.ops.OpsFlow` through its ``run_flow`` seam (never importing a
    concrete backend or touching the datastore), prints a run-health / leaderboard /
    drift summary + the ``@card`` reference, and appends a ``command="ops"`` learnings
    row. Ops is the read-only tier: it trains nothing, writes nothing back, and never
    prompts.

    Args:
        args: Parsed arguments for the ``ops`` subcommand.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    from flows import OPS_FLOW_PATH

    config = _build_config(args)
    flow_name = (getattr(args, "flow_name", None) or "").strip() or None
    experiment = (getattr(args, "experiment", None) or "").strip() or None
    dataset_ref = (getattr(args, "dataset", None) or "").strip() or None
    reference = (getattr(args, "reference", None) or "").strip() or None

    if not (flow_name or experiment or (dataset_ref and reference)):
        print(
            "error: give a monitoring target: --flow NAME, --experiment ID, or a "
            "--dataset PATHSPEC --reference PATHSPEC drift pair.",
            file=sys.stderr,
        )
        return 2

    try:
        execution = get_execution(config.mlops_provider)(config)
    except Exception as exc:  # noqa: BLE001 - actionable hint
        print(
            f"error: could not load the MLOps provider "
            f"{config.mlops_provider!r}: {exc}",
            file=sys.stderr,
        )
        return 2

    tags = [
        "loom_command:ops",
        f"loom_tenant:{config.tenant}",
        f"loom_owned_by:{config.owned_by}",
    ]
    if dataset_ref:
        tags.append(f"loom_dataset_ref:{dataset_ref}")

    scope = flow_name or experiment or f"{dataset_ref} vs {reference}"
    print(f"Monitoring {scope!r} (read-only)...")
    try:
        result = execution.run_flow(
            OPS_FLOW_PATH,
            {
                "flow_name": flow_name,
                "experiment": experiment,
                "dataset_ref": dataset_ref,
                "reference": reference,
            },
            tags=tags,
        )
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface as an actionable message
        print(
            f"error: failed to run the ops flow: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    if not getattr(result, "successful", False):
        print(
            f"error: the ops run did not complete successfully: "
            f"{getattr(result, 'error', None) or 'unknown error'}",
            file=sys.stderr,
        )

    _print_ops_summary(result)
    _record_ops_learning(config, flow_name, experiment, dataset_ref, reference, result)

    return 0 if getattr(result, "successful", False) else 1


def _record_ops_learning(
    config: LoomConfig,
    flow_name: Optional[str],
    experiment: Optional[str],
    dataset_ref: Optional[str],
    reference: Optional[str],
    result: object,
) -> None:
    """Append one ``command="ops"`` learnings row for this monitoring run.

    Best-effort and sanitized: references + small scalars only (the scope, run
    counts, drift status) -- never raw rows or secrets. A failure to record never
    fails the command.
    """
    try:
        from loom.learnings import Learnings, LearningRecord, Outcome, TaskSpec

        summary = getattr(result, "summary", None) or {}
        successful = bool(getattr(result, "successful", False))
        health = summary.get("health") or {}
        drift = summary.get("drift")
        drift_status = drift.get("status") if isinstance(drift, dict) else None

        artifacts = [
            ref
            for ref in (
                getattr(result, "pathspec", None),
                getattr(result, "card_path", None),
            )
            if ref
        ]

        record = LearningRecord(
            command="ops",
            task=TaskSpec(
                data_ref=dataset_ref,
                goal="monitor run health / leaderboard / data drift (read-only)",
                metric="n/a (read-only ops)",
                experiment_id=experiment or flow_name or dataset_ref or "ops",
            ),
            inputs={
                "flow_name": flow_name,
                "experiment": experiment,
                "dataset_ref": dataset_ref,
                "reference": reference,
                "mlops_provider": config.mlops_provider,
                "n_runs": health.get("n_runs"),
                "success_rate": health.get("success_rate"),
                "run_health": health.get("status"),
                "drift_status": drift_status,
            },
            outcome=Outcome(
                best_metric=None,
                submission_ok=successful,
                node_count=int(health.get("n_runs") or 0),
            ),
            artifacts=artifacts,
            success=successful,
            model=None,
            tenant=config.tenant,
            owned_by=config.owned_by,
            reflection=(
                f"ops flagged {drift_status or summary.get('status')}: review"
                if summary.get("status") == "ATTENTION"
                else None
            ),
        )
        Learnings(config).record(record)
    except Exception:  # noqa: BLE001 - learnings are best-effort, never fatal
        pass


def _print_collab_summary(source_ref: str, result: object) -> None:
    """Print a compact collaboration-bundle summary + the ``@card`` reference.

    Args:
        source_ref: The run/experiment pathspec the bundle is built from.
        result: The :class:`~loom.types.RunResult` returned by ``run_flow``.
    """
    summary = getattr(result, "summary", None) or {}
    print("Loom collab complete.")
    print(f"  source      : {source_ref}")
    print(f"  run         : {getattr(result, 'pathspec', None) or 'n/a'}")
    print(f"  card        : {getattr(result, 'card_path', None) or 'n/a'}")

    if summary:
        send = bool(summary.get("send"))
        print(f"  send        : {send} "
              f"(off-box send {'ON' if send else 'OFF — build only'})")
        print(f"  would-send  : {summary.get('sink') or 'none configured'}")
        print(f"  sent        : {summary.get('sent')}")
        lineage = summary.get("lineage") or {}
        if lineage.get("fingerprint"):
            print(f"  fingerprint : {lineage.get('fingerprint')}")
        sent_detail = summary.get("sent_detail")
        if sent_detail and not sent_detail.get("sent") and sent_detail.get("error"):
            print(f"  send error  : {sent_detail.get('error')}")
        print(f"  VERDICT     : {summary.get('verdict', '?')}")


def _cmd_collab(args: argparse.Namespace) -> int:
    """Handle ``loom collab``: assemble a shareable bundle via the MLOps interface.

    Resolves the configured MLOps execution provider, runs the
    :class:`flows.collab.CollabFlow` through its ``run_flow`` seam (never importing a
    concrete backend or touching the datastore), prints the bundle summary + the
    would-send target + the ``@card`` reference, and appends a ``command="collab"``
    learnings row. Collab is workspace-write to BUILD the (sanitized) bundle; the
    off-box SEND is behind ``--send`` (OFF by default; the default builds only -- no
    data leaves the box), is gated, and routes to an env/config-driven sink
    (``LOOM_COLLAB_WEBHOOK`` / ``LOOM_COLLAB_OUTBOX``) -- never a hardcoded target.

    Args:
        args: Parsed arguments for the ``collab`` subcommand.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    from flows import COLLAB_FLOW_PATH

    config = _build_config(args)
    run_pathspec = (getattr(args, "run_pathspec", None) or "").strip() or None
    experiment = (getattr(args, "experiment", None) or "").strip() or None
    send = bool(getattr(args, "send", False))

    source_ref = run_pathspec or experiment or ""

    try:
        execution = get_execution(config.mlops_provider)(config)
    except Exception as exc:  # noqa: BLE001 - actionable hint
        print(
            f"error: could not load the MLOps provider "
            f"{config.mlops_provider!r}: {exc}",
            file=sys.stderr,
        )
        return 2

    tags = [
        "loom_command:collab",
        f"loom_tenant:{config.tenant}",
        f"loom_owned_by:{config.owned_by}",
    ]
    if experiment:
        tags.append(f"loom_experiment:{experiment}")

    print(
        f"Assembling a shareable bundle of {source_ref!r} "
        f"(send={'ON' if send else 'OFF — build only'})..."
    )
    try:
        result = execution.run_flow(
            COLLAB_FLOW_PATH,
            {
                "run_pathspec": run_pathspec,
                "experiment": experiment,
                "send": send,
            },
            tags=tags,
        )
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface as an actionable message
        print(
            f"error: failed to run the collab flow: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    if not getattr(result, "successful", False):
        print(
            f"error: the collab run did not complete successfully: "
            f"{getattr(result, 'error', None) or 'unknown error'}",
            file=sys.stderr,
        )

    _print_collab_summary(source_ref, result)
    _record_collab_learning(config, run_pathspec, experiment, send, result)

    return 0 if getattr(result, "successful", False) else 1


def _record_collab_learning(
    config: LoomConfig,
    run_pathspec: Optional[str],
    experiment: Optional[str],
    send: bool,
    result: object,
) -> None:
    """Append one ``command="collab"`` learnings row for this bundle run.

    Best-effort and sanitized: references + small scalars only (the source ref, the
    send intent + outcome, the would-send sink, the bundle fingerprint) -- never raw
    rows or secrets. A failure to record never fails the command.
    """
    try:
        from loom.learnings import Learnings, LearningRecord, Outcome, TaskSpec

        summary = getattr(result, "summary", None) or {}
        successful = bool(getattr(result, "successful", False))
        source_ref = run_pathspec or experiment
        lineage = summary.get("lineage") or {}

        artifacts = [
            ref
            for ref in (
                getattr(result, "pathspec", None),
                getattr(result, "card_path", None),
            )
            if ref
        ]

        record = LearningRecord(
            command="collab",
            task=TaskSpec(
                data_ref=None,
                goal="assemble a sanitized shareable bundle of a run",
                metric="n/a (collaboration bundle)",
                experiment_id=source_ref or "collab",
            ),
            inputs={
                "run_pathspec": run_pathspec,
                "experiment": experiment,
                "send": send,
                "mlops_provider": config.mlops_provider,
                "sink": summary.get("sink"),
                "sent": bool(summary.get("sent")),
                "fingerprint": lineage.get("fingerprint"),
            },
            outcome=Outcome(
                best_metric=None,
                submission_ok=successful,
                node_count=0,
            ),
            artifacts=artifacts,
            success=successful,
            model=None,
            tenant=config.tenant,
            owned_by=config.owned_by,
            reflection=(
                "bundle sent off-box to the configured sink"
                if summary.get("sent")
                else None
            ),
        )
        Learnings(config).record(record)
    except Exception:  # noqa: BLE001 - learnings are best-effort, never fatal
        pass


# ---------------------------------------------------------------------------
# `loom doctor`: a READ-ONLY health check of the local Loom + Metaflow stack.
#
# Each check is a small, pure function returning a ``DoctorCheck`` -- a (name,
# status, detail, fix) tuple where status is one of PASS / WARN / FAIL. They are
# factored out of the command handler so the suite can unit-test them on a
# stubbed env without invoking the CLI. The handler runs them in order, prints a
# line per check + a fix for each non-PASS, then a one-line VERDICT, and returns
# a non-zero exit code iff any check FAILs (a WARN never fails the verdict).
#
# Datastore reachability is verified WITHOUT any object-store SDK: a plain TCP
# socket probe to the configured endpoint's host:port (read from the env) and,
# when Metaflow is importable, a Client-API listing. There is no object-store
# SDK import and no datastore-URI literal anywhere in this code -- the endpoint
# and datastore root are read from the environment at call time (the source scan
# in tests/test_dataio.py keeps it that way).
# ---------------------------------------------------------------------------

# Status constants (ordered worst-last for the verdict roll-up).
_DOCTOR_PASS = "PASS"
_DOCTOR_WARN = "WARN"
_DOCTOR_FAIL = "FAIL"

# The datastore environment variables `loom doctor` and the setup script agree
# on (the 7 exports the verified minikube recipe writes to .env.metaflow). The
# AWS_* pair is the datastore credential; the rest select the datastore/metadata
# backends and the run owner. Values are read from the env at call time -- never
# defaulted to a literal datastore URI here.
_DOCTOR_DATASTORE_ENV_VARS = (
    "METAFLOW_DEFAULT_DATASTORE",
    "METAFLOW_DATASTORE_SYSROOT_S3",
    "METAFLOW_S3_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "METAFLOW_DEFAULT_METADATA",
    "METAFLOW_USER",
)

# The endpoint env var whose host:port the reachability probe connects to.
_DOCTOR_ENDPOINT_ENV_VAR = "METAFLOW_S3_ENDPOINT_URL"

# The one-liner that re-runs the verified setup + this doctor.
_DOCTOR_SETUP_FIX = (
    "run the local datastore setup, then re-source the env: "
    "`bash scripts/setup_metaflow_minikube.sh` then "
    "`source .env.metaflow && loom doctor`."
)


def _doctor_check(name: str, status: str, detail: str, fix: str = "") -> dict:
    """Build one doctor check result.

    Args:
        name: Short check label (shown on the line).
        status: One of :data:`_DOCTOR_PASS` / :data:`_DOCTOR_WARN` /
            :data:`_DOCTOR_FAIL`.
        detail: Human-readable detail for the line.
        fix: Actionable remediation, shown only for a non-PASS check.

    Returns:
        A dict with ``name`` / ``status`` / ``detail`` / ``fix`` keys.
    """
    return {"name": name, "status": status, "detail": detail, "fix": fix}


def _doctor_check_loom() -> dict:
    """Check (a): the Python venv is usable and ``import loom`` works.

    This handler is itself running inside ``loom``, so reaching this code means
    ``import loom`` already succeeded; we re-import defensively to surface the
    interpreter + package location for the line.
    """
    try:
        import loom as _loom  # noqa: F401 - imported for the location/version

        where = getattr(_loom, "__file__", "?")
        return _doctor_check(
            "python/venv + import loom",
            _DOCTOR_PASS,
            f"python {sys.version.split()[0]} @ {sys.executable}; loom @ {where}",
        )
    except Exception as exc:  # noqa: BLE001 - effectively unreachable; defensive
        return _doctor_check(
            "python/venv + import loom",
            _DOCTOR_FAIL,
            f"could not import loom: {type(exc).__name__}: {exc}",
            fix=(
                "activate the Loom venv and install the package, e.g. "
                "`source .venv/bin/activate && pip install -e .` from the repo root."
            ),
        )


def _doctor_check_metaflow() -> dict:
    """Check (b): ``import metaflow`` works.

    Metaflow is an optional dependency at ``import loom`` time but required for
    every datastore-backed verb, so a missing import is a hard FAIL here (the
    fix points at the install).
    """
    try:
        import metaflow  # noqa: F401

        version = getattr(metaflow, "__version__", "?")
        return _doctor_check(
            "import metaflow",
            _DOCTOR_PASS,
            f"metaflow {version}",
        )
    except Exception as exc:  # noqa: BLE001 - actionable, no traceback
        return _doctor_check(
            "import metaflow",
            _DOCTOR_FAIL,
            f"metaflow is not importable: {type(exc).__name__}: {exc}",
            fix=(
                "install Metaflow into the active venv (it ships with Loom's "
                "deps): `pip install -e .` (or `pip install metaflow`)."
            ),
        )


def _doctor_check_datastore_env(env: Mapping[str, str]) -> dict:
    """Check (c): the datastore environment variables are present and sane.

    Verifies every variable in :data:`_DOCTOR_DATASTORE_ENV_VARS` is set and
    non-empty, and that ``METAFLOW_DEFAULT_DATASTORE`` and
    ``METAFLOW_DEFAULT_METADATA`` carry recognized values. Returns a single
    rolled-up check (FAIL if any required var is missing/blank; otherwise PASS).

    Args:
        env: The environment mapping to inspect (``os.environ`` in production;
            a stub in tests).

    Returns:
        The check result; the fix names the exact missing vars and the
        source-the-env one-liner.
    """
    missing = [
        var
        for var in _DOCTOR_DATASTORE_ENV_VARS
        if not (env.get(var) or "").strip()
    ]
    if missing:
        return _doctor_check(
            "datastore env vars",
            _DOCTOR_FAIL,
            f"{len(missing)} of {len(_DOCTOR_DATASTORE_ENV_VARS)} unset/blank: "
            + ", ".join(missing),
            fix=(
                "export the datastore env (the 7 vars the setup writes) -- "
                + _DOCTOR_SETUP_FIX
            ),
        )

    # All present: a light sanity pass on the two enum-ish selectors. A datastore
    # other than 's3' / metadata other than 'local'/'service' still works, so
    # flag it as a WARN (not a FAIL) -- the local recipe uses s3 + local.
    datastore = (env.get("METAFLOW_DEFAULT_DATASTORE") or "").strip().lower()
    metadata = (env.get("METAFLOW_DEFAULT_METADATA") or "").strip().lower()
    odd: list[str] = []
    if datastore not in ("s3", "local", "azure", "gs"):
        odd.append(f"METAFLOW_DEFAULT_DATASTORE={datastore!r}")
    if metadata not in ("local", "service"):
        odd.append(f"METAFLOW_DEFAULT_METADATA={metadata!r}")
    if odd:
        return _doctor_check(
            "datastore env vars",
            _DOCTOR_WARN,
            "all set; unusual selector value(s): " + ", ".join(odd),
            fix=(
                "the verified local recipe uses METAFLOW_DEFAULT_DATASTORE=s3 + "
                "METAFLOW_DEFAULT_METADATA=local; double-check these if a verb "
                "cannot reach the datastore."
            ),
        )

    return _doctor_check(
        "datastore env vars",
        _DOCTOR_PASS,
        f"all {len(_DOCTOR_DATASTORE_ENV_VARS)} set "
        f"(datastore={datastore}, metadata={metadata}, "
        f"user={(env.get('METAFLOW_USER') or '').strip()})",
    )


def _doctor_parse_host_port(endpoint: str) -> Optional[tuple[str, int]]:
    """Parse an endpoint URL into a ``(host, port)`` pair for a socket probe.

    Uses :func:`urllib.parse.urlparse`; falls back to the scheme's default port
    (80 for http, 443 for https) when the URL omits an explicit port. Returns
    ``None`` when no host can be parsed.

    Args:
        endpoint: The endpoint URL (e.g. the value of
            ``METAFLOW_S3_ENDPOINT_URL``). No URI literal is hard-coded here --
            the value is supplied by the caller from the environment.

    Returns:
        ``(host, port)`` or ``None`` if unparseable.
    """
    from urllib.parse import urlparse

    text = (endpoint or "").strip()
    if not text:
        return None
    # urlparse needs a scheme to populate .hostname/.port; assume http if absent.
    if "://" not in text:
        text = "http://" + text
    parsed = urlparse(text)
    host = parsed.hostname
    if not host:
        return None
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return host, int(port)


def _doctor_probe_endpoint(endpoint: str, timeout: float = 3.0) -> dict:
    """Check (d): the datastore endpoint is reachable via a TCP socket probe.

    Opens (and immediately closes) a TCP connection to the endpoint's
    host:port. This deliberately uses **no** object-store SDK and **no**
    datastore-URI literal -- it only resolves the host:port from the supplied
    endpoint string (the env's ``METAFLOW_S3_ENDPOINT_URL``) and checks the
    socket. Metaflow itself, configured by the env, owns all real datastore I/O.

    Args:
        endpoint: The endpoint URL to probe (from the environment).
        timeout: Socket connect timeout in seconds.

    Returns:
        PASS when the socket connects; WARN when the endpoint env var is unset
        (the env check already FAILs that case, so reachability is only a WARN
        here); FAIL when the host:port refuses/times out.
    """
    import socket

    target = (endpoint or "").strip()
    if not target:
        return _doctor_check(
            "datastore reachable",
            _DOCTOR_WARN,
            f"no {_DOCTOR_ENDPOINT_ENV_VAR} set; cannot probe (see the env check)",
            fix=_DOCTOR_SETUP_FIX,
        )

    parsed = _doctor_parse_host_port(target)
    if parsed is None:
        return _doctor_check(
            "datastore reachable",
            _DOCTOR_FAIL,
            f"could not parse a host:port from {_DOCTOR_ENDPOINT_ENV_VAR}={target!r}",
            fix=(
                f"set {_DOCTOR_ENDPOINT_ENV_VAR} to a URL like "
                "http://localhost:9000 (the local minio endpoint)."
            ),
        )

    host, port = parsed
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as exc:
        return _doctor_check(
            "datastore reachable",
            _DOCTOR_FAIL,
            f"cannot connect to {host}:{port} ({exc.__class__.__name__}: {exc})",
            fix=(
                "start/port-forward the datastore, e.g. "
                "`kubectl port-forward -n loom svc/minio 9000:9000 9001:9001`, "
                "or re-run the setup -- " + _DOCTOR_SETUP_FIX
            ),
        )
    return _doctor_check(
        "datastore reachable",
        _DOCTOR_PASS,
        f"TCP connect to {host}:{port} ok",
    )


def _doctor_client_api_smoke(config: LoomConfig) -> dict:
    """Check (e): a ``loom datasets``-style Client-API smoke (count data objects).

    Lists ``IngestDataset`` runs through the Metaflow Client API exactly as
    ``loom datasets`` does (no datastore touch), counting the ingested data
    objects. Zero is tolerated (a fresh stack has none) -- that is a PASS with a
    hint to run ``loom ingest``; a Client-API error is a WARN (the metadata
    service may simply be empty/local and unconfigured), never a hard FAIL,
    because the env + reachability checks already cover the load-bearing setup.

    Args:
        config: The active Loom configuration (for the Metaflow profile).

    Returns:
        The check result (PASS with a count, or WARN on a Client-API error /
        when Metaflow is not importable).
    """
    try:
        from metaflow import Flow, namespace
    except Exception as exc:  # noqa: BLE001 - the metaflow check already FAILs
        return _doctor_check(
            "client-api smoke",
            _DOCTOR_WARN,
            f"skipped (metaflow not importable: {type(exc).__name__})",
            fix="install Metaflow (see the `import metaflow` check above).",
        )

    if config.metaflow_profile:
        os.environ.setdefault("METAFLOW_PROFILE", config.metaflow_profile)
    try:
        namespace(None)
    except Exception:  # pragma: no cover - namespace API edge case
        pass

    try:
        flow = Flow("IngestDataset")
        runs = list(flow.runs("loom_dataset")) or list(flow.runs())
        count = sum(1 for r in runs if _doctor_run_ok(r))
    except Exception as exc:  # noqa: BLE001 - no flow yet / metadata down
        return _doctor_check(
            "client-api smoke",
            _DOCTOR_WARN,
            f"no ingested data objects readable via the Client API "
            f"({type(exc).__name__})",
            fix=(
                "ingest one with `loom ingest --source <path>` once the datastore "
                "is up; zero is fine on a fresh stack."
            ),
        )

    if count == 0:
        return _doctor_check(
            "client-api smoke",
            _DOCTOR_PASS,
            "Client API reachable; 0 ingested data objects (fresh stack)",
            fix="",
        )
    return _doctor_check(
        "client-api smoke",
        _DOCTOR_PASS,
        f"Client API reachable; {count} ingested data object(s)",
    )


def _doctor_run_ok(run: object) -> bool:
    """Return whether a Metaflow run is a readable, successful data object.

    Mirrors the per-run guard in :func:`_cmd_datasets`: ``.successful`` loads
    lazily and can raise on a corrupt/legacy run, so swallow any access error
    and treat the run as not-ok.
    """
    try:
        return bool(run.successful)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - an unreadable run does not count
        return False


def _doctor_verdict(checks: Sequence[dict]) -> tuple[str, bool]:
    """Roll a list of checks into a one-line VERDICT + an ok flag.

    Args:
        checks: The check results.

    Returns:
        ``(verdict_line, ok)`` where ``ok`` is ``False`` iff any check FAILed
        (a WARN never fails the verdict).
    """
    fails = sum(1 for c in checks if c["status"] == _DOCTOR_FAIL)
    warns = sum(1 for c in checks if c["status"] == _DOCTOR_WARN)
    total = len(checks)
    if fails:
        return (
            f"FAIL -- {fails} check(s) failed, {warns} warning(s) "
            f"({total} total). Fix the FAIL line(s) above and re-run `loom doctor`.",
            False,
        )
    if warns:
        return (
            f"PASS (with {warns} warning(s)) -- the stack is usable; "
            "review the WARN line(s) above.",
            True,
        )
    return (f"PASS -- all {total} checks green; the local stack is ready.", True)


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Handle ``loom doctor``: a read-only diagnosis of the local stack.

    Runs the factored check functions in order, prints a ``[STATUS] name --
    detail`` line per check (with a ``fix:`` line for each non-PASS), then a
    one-line VERDICT. Diagnoses only: it never installs, mutates, prompts, or
    touches the datastore except through the Metaflow Client API or a TCP socket
    probe to the configured endpoint. Exit code is 0 when no check FAILs.

    Args:
        args: Parsed arguments for the ``doctor`` subcommand.

    Returns:
        ``0`` when no check FAILed, else ``1``.
    """
    config = _build_config(args)
    env = os.environ

    checks: list[dict] = [
        _doctor_check_loom(),
        _doctor_check_metaflow(),
        _doctor_check_datastore_env(env),
        _doctor_probe_endpoint(env.get(_DOCTOR_ENDPOINT_ENV_VAR, "")),
        _doctor_client_api_smoke(config),
    ]

    print("Loom doctor -- local stack health check (read-only):")
    for check in checks:
        print(f"  [{check['status']:<4}] {check['name']} -- {check['detail']}")
        if check["status"] != _DOCTOR_PASS and check["fix"]:
            print(f"         fix: {check['fix']}")

    verdict, ok = _doctor_verdict(checks)
    print("")
    print(f"VERDICT: {verdict}")
    return 0 if ok else 1


def _cmd_datasets(args: argparse.Namespace) -> int:
    """Handle ``loom datasets``: list ingested data objects via the Client API.

    Reads :class:`flows.ingest_dataset.IngestDataset` runs tagged ``loom_dataset``
    through the Metaflow Client API and prints, per data object, its pathspec, its
    name, and a row-count / schema summary. Read-only: it never touches the
    datastore directly.

    Args:
        args: Parsed arguments for the ``datasets`` subcommand.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    config = _build_config(args)

    try:
        from metaflow import Flow, namespace
    except Exception as exc:  # noqa: BLE001 - actionable hint, no traceback
        print(
            f"error: Metaflow is required for `loom datasets` but could not be "
            f"imported: {exc}",
            file=sys.stderr,
        )
        return 2

    # Point the Client API at the configured profile's metadata service. The
    # Client reads METAFLOW_PROFILE from the environment.
    if config.metaflow_profile:
        os.environ.setdefault("METAFLOW_PROFILE", config.metaflow_profile)

    # See all runs regardless of the user namespace `loom ingest` ran under.
    try:
        namespace(None)
    except Exception:  # pragma: no cover - namespace API edge case
        pass

    try:
        flow = Flow("IngestDataset")
        runs = list(flow.runs("loom_dataset"))
        # Fall back to all IngestDataset runs if none carry the loom_dataset tag
        # (an unnamed ingest is not tagged).
        if not runs:
            runs = list(flow.runs())
    except Exception:  # noqa: BLE001 - no such flow yet / metadata down
        print("No ingested data objects found (run `loom ingest` first).")
        return 0

    rows: list[tuple[str, str, str]] = []
    for run in runs:
        # Everything below reads artifacts that load LAZILY on access (the
        # ``successful`` flag and ``.data`` both fetch datastore blobs), so a
        # corrupt/legacy run can raise on access (e.g. a missing blob -> a
        # TypeError that getattr does not swallow). Guard the whole per-run body
        # so one unreadable run is skipped gracefully and never breaks the list.
        try:
            if not bool(run.successful):
                continue
            pathspec = getattr(run, "pathspec", None) or str(run)
            data = run.data
            name = ""
            schema = None
            if data is not None:
                name = str(getattr(data, "dataset_name", "") or "")
                schema = getattr(data, "schema", None)
        except Exception:  # noqa: BLE001 - skip unreadable runs gracefully
            continue
        if isinstance(schema, dict):
            nrows = schema.get("nrows", "?")
            ncols = len(schema.get("columns", []) or [])
            target = schema.get("target")
            detail = f"nrows={nrows} cols={ncols}" + (
                f" target={target}" if target else ""
            )
        else:
            detail = "schema n/a"
        rows.append((pathspec, name or "-", detail))

    if not rows:
        print("No ingested data objects found (run `loom ingest` first).")
        return 0

    print(f"Ingested data objects ({len(rows)}):")
    for pathspec, name, detail in rows:
        print(f"  {pathspec}  ·  {name}  ·  {detail}")
    return 0


def _cmd_proxy(args: argparse.Namespace) -> int:
    """Handle a bare ``loom proxy`` (no action): print help, return non-zero.

    Mirrors the top-level no-subcommand behavior so ``loom proxy`` alone is
    discoverable rather than a silent no-op.

    Args:
        args: Parsed arguments for the ``proxy`` subcommand.

    Returns:
        ``1`` (no action chosen).
    """
    parser = getattr(args, "_proxy_parser", None)
    if parser is not None:
        parser.print_help()
    return 1


def _cmd_proxy_serve(args: argparse.Namespace) -> int:
    """Handle ``loom proxy serve``: launch the Loom gateway under uvicorn.

    Delegates to :func:`loom.proxy.server.serve`, which reads the server-side
    ``ANTHROPIC_API_KEY`` (the real vendor key) and the accepted ``LOOM_API_KEY``
    / ``LOOM_API_KEYS`` from the environment and fails fast (via ``SystemExit``)
    with an actionable message if the vendor key is missing. The proxy deps
    (starlette / uvicorn / httpx) are imported lazily here so a missing optional
    dependency only affects this command, not the rest of the CLI.

    Args:
        args: Parsed arguments for the ``proxy serve`` action (``--host`` /
            ``--port``).

    Returns:
        Process exit code (``0`` on a clean shutdown; ``2`` on a missing-key /
        import failure surfaced as an actionable message).
    """
    try:
        from loom.proxy.server import serve
    except Exception as exc:  # noqa: BLE001 - actionable hint, no traceback
        print(
            f"error: the Loom gateway needs starlette + uvicorn + httpx but they "
            f"could not be imported: {exc}\nInstall them (they ship with Loom's "
            "deps) and try again.",
            file=sys.stderr,
        )
        return 2

    try:
        serve(host=args.host, port=args.port)
    except SystemExit as exc:
        # serve() raises SystemExit with an actionable message on a missing key.
        message = exc.code
        if isinstance(message, str):
            print(message, file=sys.stderr)
            return 2
        return int(message or 0)
    except KeyboardInterrupt:  # pragma: no cover - interactive shutdown
        print("\nLoom gateway stopped.", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# `loom telemetry`: inspect + export the distillation-grade trajectory corpus.
#
# The telemetry layer ADDS trajectory correlation on top of Loom's existing
# captures: it stamps one stable trajectory_id across the telemetry events
# (telemetry/events.jsonl), the proxy LLM calls (learnings/proxy_calls.jsonl),
# and the command-level rollouts (learnings/rollouts.jsonl), so a full agent
# trajectory can be re-stitched and distilled into the LOOM-DS-1 corpus. These
# handlers are read-only / workspace-write (only `export` writes -- the dataset
# file). They never emit telemetry, run a flow, or touch the datastore.
# ---------------------------------------------------------------------------


def _telemetry_collect_trajectories(config: LoomConfig, *, verbose: bool = False):
    """Assemble every trajectory in the corpus by JOINing the three signals.

    Reads the telemetry events, the proxy LLM call rows, and the command-level
    rollouts, groups them by ``trajectory_id`` (falling back to a rollout's
    ``task.experiment_id`` when it carries no explicit id, mirroring the additive
    integration), and calls the pure :func:`loom.telemetry.assemble_trajectory`
    per id. Pure read: no telemetry is emitted, no flow runs.

    Args:
        config: The active configuration (the telemetry / proxy / learnings paths).
        verbose: Unused placeholder for symmetry; kept off the hot path.

    Returns:
        A list of assembled :class:`~loom.telemetry.TrajectoryRecord`, one per id.
    """
    from loom.learnings import Learnings
    from loom.telemetry import assemble_trajectory, read_events
    from loom.telemetry.sink import read_jsonl

    events = read_events(config)
    proxy_calls = read_jsonl(config.proxy_log_path)
    rollouts = [_dataclass_asdict_safe(r) for r in Learnings(config).all()]

    # Index the rollouts by their join key: the explicit trajectory_id when set,
    # else the task.experiment_id (the additive fallback).
    rollout_by_id: dict[str, dict] = {}
    for roll in rollouts:
        tid = roll.get("trajectory_id") or (roll.get("task") or {}).get("experiment_id")
        if tid is not None:
            rollout_by_id.setdefault(str(tid), roll)

    # The set of all trajectory ids appearing anywhere across the three signals.
    ids: set[str] = set()
    for row in events:
        tid = row.get("trajectory_id")
        if tid:
            ids.add(str(tid))
    for call in proxy_calls:
        tid = call.get("trajectory_id")
        if tid:
            ids.add(str(tid))
    ids.update(rollout_by_id.keys())

    trajectories = []
    for tid in sorted(ids):
        trajectories.append(
            assemble_trajectory(
                tid,
                events,
                proxy_calls,
                rollout_by_id.get(tid),
            )
        )
    return trajectories


def _dataclass_asdict_safe(obj: object) -> dict:
    """Convert a LearningRecord (a dataclass) into a plain dict for the JOIN."""
    import dataclasses as _dc

    if _dc.is_dataclass(obj):
        return _dc.asdict(obj)
    return dict(obj) if isinstance(obj, dict) else {}


def _cmd_telemetry(args: argparse.Namespace) -> int:
    """Handle a bare ``loom telemetry`` (no action): print help, return non-zero.

    Mirrors :func:`_cmd_proxy` so a bare ``loom telemetry`` is discoverable.

    Args:
        args: Parsed arguments for the ``telemetry`` subcommand.

    Returns:
        ``1`` (no action chosen).
    """
    parser = getattr(args, "_telemetry_parser", None)
    if parser is not None:
        parser.print_help()
    return 1


def _cmd_telemetry_status(args: argparse.Namespace) -> int:
    """Handle ``loom telemetry status``: a read-only telemetry-corpus summary.

    Prints the telemetry/trajectories paths, the event count, the
    assembled-trajectory count, the general-vs-tenant-owned split (the IP
    boundary), and the optional OTel exporter state. Reads only -- it never emits
    telemetry, exports, or prompts.

    Args:
        args: Parsed arguments for the ``telemetry status`` action.

    Returns:
        ``0`` always (a pure read).
    """
    from loom.telemetry import bootstrap_ops_telemetry, read_events
    from loom.telemetry.events import content_logging_enabled, telemetry_enabled

    config = _build_config(args)

    events = read_events(config)
    trajectories = _telemetry_collect_trajectories(config)
    general = sum(1 for t in trajectories if t.owned_by == "general")
    tenant_owned = len(trajectories) - general

    otel = bootstrap_ops_telemetry(config)

    print("Loom telemetry -- COMPLETE training-data corpus, not observability (read-only):")
    print(f"  events path      : {config.telemetry_path}")
    print(f"  trajectories path: {config.trajectories_path}")
    print(f"  proxy calls path : {config.proxy_log_path}")
    print(f"  learnings path   : {config.learnings_path}")
    print(f"  events           : {len(events)}")
    print(f"  trajectories     : {len(trajectories)}")
    print(f"  IP boundary      : {general} general · {tenant_owned} tenant-owned")
    print(
        f"  capture enabled  : {'on' if telemetry_enabled() else 'off'} "
        "(LOOM_TELEMETRY)"
    )
    print(
        f"  content logging  : {'on' if content_logging_enabled() else 'off (redacted)'} "
        "(LOOM_LOG_CONTENT)"
    )
    print(
        f"  OTel ops mirror  : "
        f"{'on' if otel.enabled else 'off'} / "
        f"{'available' if otel.available else 'sdk-absent'}"
        + (f" [{', '.join(otel.exporters)}]" if otel.exporters else "")
        + " (LOOM_TELEMETRY_OTEL_OPS; ops-only, NOT the corpus)"
    )
    return 0


def _cmd_telemetry_export(args: argparse.Namespace) -> int:
    """Handle ``loom telemetry export``: assemble trajectories -> a LOOM-DS-1 JSONL.

    Assembles the correlated signals into trajectories, builds the LOOM-DS-1
    SFT/teacher dataset (general-only by default -- the IP boundary; content
    REDACTED BY DEFAULT unless ``--with-content``), and writes it to ``--out``
    (default ``telemetry/loom-ds-1.jsonl``, anchored next to the trajectories
    path). Workspace-write: the export file is the only thing written.

    With ``--to-dataset NAME`` it ALSO ingests the same assembled corpus as a
    **versioned, content-addressed Metaflow data object** (the durable, lossless,
    no-sampling sink for scale) through the same ``IngestDataset`` seam
    ``loom ingest`` uses, and prints the produced pathspec. The corpus is NEVER
    routed through an observability/metrics endpoint.

    Args:
        args: Parsed arguments for the ``telemetry export`` action.

    Returns:
        ``0`` on success (or when ``--to-dataset`` ingest fails, a non-zero code).
    """
    import dataclasses as _dc

    from loom.telemetry import build_distillation_dataset
    from loom.telemetry.sink import append_trajectory

    config = _build_config(args)
    owned_by = (getattr(args, "owned_by", None) or "general").strip() or "general"
    with_content = bool(getattr(args, "with_content", False))
    to_dataset = (getattr(args, "to_dataset", None) or "").strip()

    out_path = getattr(args, "out", None)
    if out_path:
        out_path = os.path.abspath(out_path)
    else:
        # Default beside the trajectories path so it lives in the telemetry dir.
        out_path = os.path.join(
            os.path.dirname(config.trajectories_path) or ".", "loom-ds-1.jsonl"
        )

    trajectories = _telemetry_collect_trajectories(config)
    examples = build_distillation_dataset(
        trajectories,
        owned_by_filter=owned_by,
        with_content=with_content,
    )

    # Excluded count: trajectories the IP boundary kept out of the export.
    excluded = len(trajectories) - len(examples)

    # Truncate any prior export, then append each example as one JSONL row. The
    # --out file export is always produced (kept regardless of --to-dataset).
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    open(out_path, "w", encoding="utf-8").close()
    rows = [_dc.asdict(ex) for ex in examples]
    for row in rows:
        append_trajectory(out_path, row)

    print("Loom telemetry export complete (the LOOM-DS-1 distillation dataset).")
    print(f"  owned_by filter : {owned_by} (the IP boundary)")
    print(
        f"  content         : "
        f"{'raw (--with-content)' if with_content else 'REDACTED (default)'}"
    )
    print(f"  trajectories    : {len(trajectories)} assembled")
    print(f"  examples written: {len(examples)}")
    print(f"  excluded (IP)   : {excluded} (owned_by != {owned_by})")
    print(f"  out             : {out_path}")

    # The durable, lossless, no-sampling sink for scale: ingest the corpus as a
    # versioned Metaflow data object via the SAME seam `loom ingest` uses, on
    # Loom's "data is a Metaflow artifact" thesis. Never an observability sink.
    if to_dataset:
        return _telemetry_export_to_dataset(config, rows, to_dataset, out_path)

    return 0


def _telemetry_export_to_dataset(
    config: LoomConfig,
    rows: list[dict],
    name: str,
    out_path: str,
) -> int:
    """Ingest the assembled SFT corpus as a versioned Metaflow data object.

    Writes the corpus losslessly to a CSV staging file (the nested ``context`` /
    ``tools_trajectory`` / ``teacher_output`` fields JSON-encoded so they
    round-trip), then runs it through the SAME ``IngestDataset`` seam
    :func:`_cmd_ingest` uses (:func:`_ingest_source`) so it becomes a durable,
    content-addressed, versioned data object addressable by pathspec -- a
    first-class ``dataset_ref``. The corpus never touches an observability/metrics
    endpoint. Needs the metaflow MLOps provider (guarded like the other lifecycle
    paths).

    Args:
        config: The active configuration (for the Metaflow profile).
        rows: The assembled SFT examples as plain dicts.
        name: The dataset name for the produced data object.
        out_path: The already-written ``--out`` JSONL (for the staging dir).

    Returns:
        ``0`` when the data object was produced, else a non-zero exit code.
    """
    import csv as _csv
    import json as _json
    import tempfile

    # The metaflow MLOps provider is required for a data object, exactly as the
    # other lifecycle paths guard it.
    if config.mlops_provider != "metaflow":
        print(
            f"error: --to-dataset produces a Metaflow data object and needs the "
            f"metaflow MLOps provider, but mlops_provider is "
            f"{config.mlops_provider!r}. Re-run with --mlops metaflow (or set "
            "LOOM_MLOPS_PROVIDER=metaflow).",
            file=sys.stderr,
        )
        return 2

    # Stage the corpus as a CSV IngestDataset can read. Nested fields are
    # JSON-encoded strings so the data object is a LOSSLESS round-trip of the
    # JSONL export (the whole corpus, never sampled).
    field_names = [
        "trajectory_id",
        "verb",
        "context",
        "teacher_output",
        "tools_trajectory",
        "reward",
        "weight",
        "owned_by",
        "metric",
        "success",
    ]
    staging_dir = tempfile.mkdtemp(prefix="loom-ds-ingest-")
    train_csv = os.path.join(staging_dir, "train.csv")
    with open(train_csv, "w", encoding="utf-8", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            encoded = {}
            for key in field_names:
                value = row.get(key)
                if isinstance(value, (list, dict)):
                    encoded[key] = _json.dumps(value, ensure_ascii=False)
                else:
                    encoded[key] = value
            writer.writerow(encoded)

    print(f"  to-dataset      : ingesting {len(rows)} example(s) as {name!r}...")
    pathspec, error = _ingest_source(staging_dir, name, config)
    if error is not None:
        print(f"error: {error}", file=sys.stderr)
        return 2 if "Metaflow is required" in error else 1

    print("  data object     : a versioned, content-addressed Metaflow data object")
    print(f"  dataset_ref     : {pathspec}")
    print("")
    print("This is the durable, lossless, no-sampling corpus sink (a dataset_ref).")
    print("Inspect it with:")
    print(f"  loom eda --dataset {pathspec} --mlops metaflow")
    return 0


def _cmd_telemetry_trace(args: argparse.Namespace) -> int:
    """Handle ``loom telemetry trace``: show one assembled trajectory by id.

    Assembles and prints a single trajectory (task context, ordered steps,
    terminal outcome + reward). Read-only; content stays redacted unless
    ``--with-content``.

    Args:
        args: Parsed arguments for the ``telemetry trace`` action.

    Returns:
        ``0`` when the trajectory was found, else ``1``.
    """
    from loom.learnings import Learnings
    from loom.telemetry import assemble_trajectory, read_events
    from loom.telemetry.sink import read_jsonl

    config = _build_config(args)
    tid = (getattr(args, "trajectory", None) or "").strip()
    with_content = bool(getattr(args, "with_content", False))

    if not tid:
        print("error: --trajectory <id> is required.", file=sys.stderr)
        return 2

    events = read_events(config)
    proxy_calls = read_jsonl(config.proxy_log_path)

    rollout = None
    for roll in Learnings(config).all():
        roll_tid = roll.trajectory_id or (
            roll.task.experiment_id if roll.task else None
        )
        if roll_tid == tid:
            rollout = _dataclass_asdict_safe(roll)
            break

    traj = assemble_trajectory(tid, events, proxy_calls, rollout)

    if traj.event_count == 0 and not traj.steps and rollout is None:
        print(
            f"No trajectory found for id {tid!r}. Known ids come from "
            "`loom telemetry status` (or run with LOOM_TELEMETRY=1 to capture).",
            file=sys.stderr,
        )
        return 1

    print(f"Trajectory {traj.trajectory_id}")
    print(f"  verb        : {traj.verb}")
    print(f"  owned_by    : {traj.owned_by}  ·  tenant: {traj.tenant}")
    print(f"  events      : {traj.event_count}")
    if traj.duration_ms is not None:
        print(f"  duration_ms : {traj.duration_ms}")
    if traj.task:
        for key in ("goal", "metric", "data_ref", "experiment_id"):
            if key in traj.task:
                value = traj.task[key]
                if not with_content and key in ("goal", "metric"):
                    value = f"<REDACTED:task.{key}>"
                print(f"  task.{key:<7}: {value}")

    print(f"  steps       : {len(traj.steps)}")
    for step in traj.steps:
        attrs = step.attributes or {}
        bits = "  ".join(f"{k}={v}" for k, v in attrs.items())
        print(f"    {step.index:>2}. [{step.kind}] {bits}")
        if step.llm_response is not None:
            text = step.llm_response.get("response_text")
            if text is not None:
                shown = text if with_content else "<REDACTED:output>"
                print(f"        response: {shown}")

    out = traj.outcome
    print(
        f"  outcome     : metric={_format_metric(out.metric)} "
        f"verdict={out.verdict or 'n/a'} success={out.success} "
        f"reward={_format_metric(out.reward)}"
    )
    return 0


# ---------------------------------------------------------------------------
# `loom skillopt`: the self-improvement loop's OPTIMIZE stage (the moat v0.2).
#
# This is the one ops/meta verb that optimizes the OTHER /loom-* skills. It
# captures a verb's learnings corpus (HiveMind), scores the incumbent SKILL.md +
# any candidate against the 7-point acceptance contract (HARD) + corpus coverage
# (SOFT), and applies the never-worse promotion GATE -- the exact parallel of
# `loom deploy`'s exit gate. SAFE BY DEFAULT: it PROPOSES (writes a sidecar
# candidate + prints the gate verdict + a diff); the in-place SKILL.md overwrite
# is behind --apply and runs ONLY when the gate PROMOTED a candidate. The whole
# loop is deterministic + LLM-free (the scorer + a file/identity candidate
# source); --propose is an OPTIONAL pluggable adapter that is a clearly-marked
# no-op when no model is configured.
# ---------------------------------------------------------------------------

# Where the in-repo skill library lives relative to this file (loom/cli.py ->
# the repo root's skills/ dir). Resolved at call time, never a hardcoded abs path.
def _skills_root() -> str:
    """Return the absolute path of the in-repo ``skills/`` directory.

    Anchored off this module's location (``loom/cli.py``) so it resolves the
    same regardless of the launch cwd.

    Returns:
        The absolute ``skills/`` directory path.
    """
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skills")


def _normalize_verb(verb: str) -> str:
    """Normalize a ``--verb`` to the ``loom-<name>`` skill-folder convention.

    Accepts ``eda`` / ``loom-eda`` / ``loom eda`` and returns ``loom-eda`` -- the
    name of the skill folder under ``skills/``.

    Args:
        verb: The raw ``--verb`` value.

    Returns:
        The normalized ``loom-<name>`` verb string.
    """
    token = (verb or "").strip().lower().replace(" ", "-")
    if not token:
        return token
    return token if token.startswith("loom-") else f"loom-{token}"


def _skill_command(verb: str) -> str:
    """The ``command`` learnings field a verb's rollouts carry (no ``loom-`` prefix).

    The lifecycle CLI records rollout rows with ``command="eda"`` (the bare verb),
    so the corpus capture must strip the ``loom-`` skill-folder prefix.

    Args:
        verb: The normalized ``loom-<name>`` verb.

    Returns:
        The bare command name (e.g. ``"eda"``).
    """
    return verb[len("loom-"):] if verb.startswith("loom-") else verb


def _propose_candidate(verb: str, incumbent_text: str, corpus: object) -> Optional[str]:
    """The OPTIONAL pluggable LLM proposer seam for ``--propose`` (a no-op stub by default).

    The self-improvement loop is deterministic + LLM-free by design: the scorer +
    gate need no model, and the load-bearing candidate source is a file
    (``--candidate``) or the identity incumbent. ``--propose`` is the *optional*
    adapter that would ask a model to draft a better SKILL.md. In v0.2 no model is
    wired here, so this is a **clearly-marked no-op**: it returns ``None`` (no
    candidate proposed) and the caller reports that the proposer is a stub. Wiring a
    real proposer means returning a candidate text from this one function -- the
    gate then scores + gates it exactly like a ``--candidate`` file, so a model's
    draft can never bypass the never-worse contract gate.

    Args:
        verb: The normalized ``loom-<name>`` verb being optimized.
        incumbent_text: The currently-shipped SKILL.md text.
        corpus: The captured :class:`~loom.hivemind.VerbCorpus` digest.

    Returns:
        A proposed candidate SKILL.md text, or ``None`` when no proposer is
        configured (the default).
    """
    return None


def _print_skillopt_corpus(corpus: object) -> None:
    """Print a compact HiveMind corpus digest for the verb being optimized.

    Args:
        corpus: The :class:`~loom.hivemind.VerbCorpus` digest.
    """
    n = getattr(corpus, "n_rollouts", 0)
    print("Captured corpus (HiveMind):")
    print(f"  owned_by    : {getattr(corpus, 'owned_by_filter', 'general')} (IP boundary)")
    print(f"  rollouts    : {n}")
    if not n:
        print("  (empty corpus -- scoring against the contract only; the moat compounds from run #1)")
        return
    print(
        f"  success     : {getattr(corpus, 'n_success', 0)}/{n} "
        f"({getattr(corpus, 'success_rate', 0.0):.0%})"
    )
    metric = getattr(corpus, "metric", None)
    if metric is not None and getattr(metric, "n", 0):
        print(
            f"  metric      : min={_format_metric(metric.min)} "
            f"mean={_format_metric(metric.mean)} max={_format_metric(metric.max)} "
            f"(n={metric.n})"
        )
    verdicts = getattr(corpus, "verdict_histogram", {}) or {}
    if verdicts:
        shown = ", ".join(f"{k}={v}" for k, v in list(verdicts.items())[:6])
        print(f"  verdicts    : {shown}")
    failures = getattr(corpus, "failure_modes", []) or []
    if failures:
        shown = ", ".join(f"{fm.label}={fm.count}" for fm in failures[:6])
        print(f"  failures    : {shown}")


def _print_skillopt_result(result: object) -> None:
    """Print the gate VERDICT + the incumbent/candidate scores for a skillopt run.

    Args:
        result: The :class:`~loom.skillopt.SkillOptResult` to render.
    """
    inc = result.incumbent_score
    print("SkillOpt gate:")
    print(f"  verb        : {result.verb}")
    print(f"  VERDICT     : {result.verdict} (promoted={result.promoted})")
    inc_misses = inc.detail.get("hard_misses", []) if isinstance(inc.detail, dict) else []
    print(
        f"  incumbent   : hard_ok={inc.hard_ok} soft={inc.soft:.4g} total={inc.total:.6g}"
        + (f"  hard_misses={inc_misses}" if inc_misses else "")
    )
    for idx, score in enumerate(result.candidate_scores):
        misses = score.detail.get("hard_misses", []) if isinstance(score.detail, dict) else []
        marker = "  <- WINNER" if (result.promoted and idx == result.winner_index) else ""
        print(
            f"  candidate {idx}: hard_ok={score.hard_ok} soft={score.soft:.4g} "
            f"total={score.total:.6g}"
            + (f"  hard_misses={misses}" if misses else "")
            + marker
        )
    gate = result.gate_detail or {}
    if gate.get("n_disqualified"):
        print(
            f"  disqualified: {gate.get('n_disqualified')} candidate(s) violated a "
            "hard contract constraint (excluded before selection)"
        )


def _cmd_skillopt(args: argparse.Namespace) -> int:
    """Handle ``loom skillopt``: capture -> score -> gate -> propose (or gated --apply).

    Resolves the verb's ``skills/<verb>/SKILL.md`` (the trainable artifact),
    captures the verb's learnings corpus through HiveMind
    (:func:`loom.hivemind.capture_corpus`, filtered to the ``owned_by=general`` IP
    boundary), gathers candidate texts (from ``--candidate`` files; or ``--propose``
    -- the optional LLM proposer stub; or none = score+report the incumbent), runs
    the pure :func:`loom.skillopt.optimize_skill` never-worse GATE, and prints the
    corpus digest + the gate VERDICT + a unified diff of the winner vs. the
    incumbent. SAFE BY DEFAULT: the default PROPOSES -- it writes the winning text to
    a sidecar ``skills/<verb>/SKILL.candidate.md`` and reports; ``--apply`` overwrites
    ``skills/<verb>/SKILL.md`` IN PLACE only when the gate PROMOTED a candidate
    (mirroring ``loom deploy --apply``). Every run appends a ``command="skillopt"``
    learnings audit row.

    Args:
        args: Parsed arguments for the ``skillopt`` subcommand.

    Returns:
        Process exit code (0 on success, non-zero on a usage/IO error). A clean run
        whose gate KEPT the incumbent still returns 0 -- KEEP is the correct,
        machine-checkable outcome, surfaced in the VERDICT.
    """
    from loom.hivemind import capture_corpus
    from loom.skillopt import ContractCorpusScorer, optimize_skill, unified_diff

    config = _build_config(args)
    verb = _normalize_verb(getattr(args, "verb", "") or "")
    candidate_path = (getattr(args, "candidate", None) or "").strip() or None
    propose = bool(getattr(args, "propose", False))
    apply = bool(getattr(args, "apply", False))

    if not verb:
        print("error: --verb is required (e.g. --verb loom-eda).", file=sys.stderr)
        return 2
    if candidate_path and propose:
        print(
            "error: pass at most one candidate source: --candidate PATH or "
            "--propose (not both).",
            file=sys.stderr,
        )
        return 2

    skills_root = _skills_root()
    skill_path = os.path.join(skills_root, verb, "SKILL.md")
    if not os.path.isfile(skill_path):
        print(
            f"error: no incumbent skill found at {skill_path} (is --verb {verb!r} a "
            "real /loom-* verb under skills/?).",
            file=sys.stderr,
        )
        return 2

    try:
        with open(skill_path, "r", encoding="utf-8") as fh:
            incumbent_text = fh.read()
    except OSError as exc:
        print(f"error: could not read the incumbent skill {skill_path}: {exc}", file=sys.stderr)
        return 2

    # Gather candidate texts. Deterministic, LLM-free by default: a --candidate
    # file, an optional --propose stub, or none (score + report the incumbent).
    candidate_texts: list[str] = []
    if candidate_path:
        if not os.path.isfile(candidate_path):
            print(
                f"error: --candidate path does not exist: {candidate_path}",
                file=sys.stderr,
            )
            return 2
        try:
            with open(candidate_path, "r", encoding="utf-8") as fh:
                candidate_texts.append(fh.read())
        except OSError as exc:
            print(f"error: could not read the candidate {candidate_path}: {exc}", file=sys.stderr)
            return 2

    # Capture the verb's learnings corpus (HiveMind). The command field the
    # lifecycle CLI records is the bare verb (e.g. "eda"), and the IP boundary keeps
    # ONLY owned_by="general" rows. Tolerant of a missing/empty corpus.
    command = _skill_command(verb)
    corpus = capture_corpus(command, config.learnings_path, owned_by_filter="general")

    proposer_note = ""
    if propose:
        proposed = _propose_candidate(verb, incumbent_text, corpus)
        if proposed is not None:
            candidate_texts.append(proposed)
        else:
            proposer_note = (
                "  (--propose: no LLM proposer configured -- this is a no-op stub; "
                "scoring the incumbent only. Supply --candidate PATH for a real "
                "candidate.)"
            )

    print(
        f"Optimizing skill {verb!r} (SKILL.md is the trainable artifact; "
        f"apply={'ON' if apply else 'OFF — proposes a sidecar candidate'})..."
    )
    if proposer_note:
        print(proposer_note)
    _print_skillopt_corpus(corpus)

    scorer = ContractCorpusScorer()
    result = optimize_skill(verb, incumbent_text, candidate_texts, scorer, corpus)

    print("")
    _print_skillopt_result(result)

    # Show what would change (winner vs. incumbent). Empty when the incumbent is kept.
    if result.promoted:
        diff = unified_diff(incumbent_text, result.winner_text, verb)
        if diff:
            print("")
            print("Winner vs. incumbent (unified diff):")
            # Cap the inline diff defensively; large output spills nowhere here, so
            # just truncate the rendered text rather than flood the transcript.
            if len(diff) > 20000:
                diff = diff[:20000] + "\n... (diff truncated)\n"
            print(diff)

    # SAFE BY DEFAULT: default PROPOSES (sidecar), --apply is the gated mutate.
    exit_code = 0
    if apply:
        if result.promoted:
            try:
                with open(skill_path, "w", encoding="utf-8") as fh:
                    fh.write(result.winner_text)
                print("")
                print(f"--apply: gate PROMOTED -> overwrote {skill_path} in place.")
            except OSError as exc:
                print(f"error: could not overwrite {skill_path}: {exc}", file=sys.stderr)
                exit_code = 1
        else:
            print("")
            print(
                f"--apply: gate did NOT promote (VERDICT={result.verdict}); "
                "the incumbent is kept UNCHANGED (never deploy a worse skill)."
            )
    elif result.promoted:
        sidecar = os.path.join(skills_root, verb, "SKILL.candidate.md")
        try:
            with open(sidecar, "w", encoding="utf-8") as fh:
                fh.write(result.winner_text)
            print("")
            print(
                f"Proposed: wrote the winning candidate to {sidecar} "
                "(review it, then re-run with --apply to overwrite in place)."
            )
        except OSError as exc:
            print(f"error: could not write the sidecar {sidecar}: {exc}", file=sys.stderr)
            exit_code = 1
    else:
        print("")
        print(
            f"Proposed nothing to promote (VERDICT={result.verdict}); "
            "the incumbent stands."
        )

    _record_skillopt_learning(config, verb, candidate_path, propose, apply, result)
    return exit_code


def _record_skillopt_learning(
    config: LoomConfig,
    verb: str,
    candidate_path: Optional[str],
    propose: bool,
    apply: bool,
    result: object,
) -> None:
    """Append one ``command="skillopt"`` learnings audit row for this optimize run.

    Best-effort and sanitized: the row carries only the verb, the gate decision +
    verdict, the incumbent/candidate scores (the audit blob the gate produced), and
    the apply/propose flags -- never raw skill text or secrets. A failure to record
    never fails the command. The row is itself owned_by=general (a skill edit is a
    cross-tenant moat artifact), feeding the flywheel that drives the next round.

    Args:
        config: The active Loom configuration.
        verb: The normalized ``loom-<name>`` verb optimized.
        candidate_path: The ``--candidate`` path, or ``None``.
        propose: Whether the optional proposer was requested.
        apply: Whether the gated in-place overwrite was requested.
        result: The :class:`~loom.skillopt.SkillOptResult`.
    """
    try:
        from loom.learnings import Learnings, LearningRecord, Outcome, TaskSpec

        audit = getattr(result, "audit", None) or {}
        promoted = bool(getattr(result, "promoted", False))
        verdict = str(getattr(result, "verdict", "") or "")
        inc_score = getattr(result, "incumbent_score", None)
        # A skillopt rollout "succeeds" as a flywheel signal only when it actually
        # promoted a better skill; a clean KEEP is a correct refusal, not a win
        # (mirrors the deploy BLOCK posture). The metric is the winner's total score.
        winner_total = None
        winner_index = getattr(result, "winner_index", None)
        if promoted and winner_index is not None:
            scores = getattr(result, "candidate_scores", []) or []
            if 0 <= winner_index < len(scores):
                winner_total = scores[winner_index].total
        elif inc_score is not None:
            winner_total = inc_score.total

        record = LearningRecord(
            command="skillopt",
            task=TaskSpec(
                data_ref=None,
                goal=f"optimize the {verb} SKILL.md against the learnings corpus (gated)",
                metric="skill total (hard contract + soft corpus coverage)",
                experiment_id=verb,
            ),
            inputs={
                "verb": verb,
                "candidate_path": candidate_path,
                "propose": propose,
                "apply": apply,
                "promoted": promoted,
                "verdict": verdict,
                "n_candidates": (result.gate_detail or {}).get("n_candidates"),
                "n_eligible": (result.gate_detail or {}).get("n_eligible"),
                "n_disqualified": (result.gate_detail or {}).get("n_disqualified"),
                "incumbent_hard_ok": getattr(inc_score, "hard_ok", None),
                "incumbent_total": getattr(inc_score, "total", None),
                "audit": audit,
            },
            outcome=Outcome(
                best_metric=float(winner_total)
                if isinstance(winner_total, (int, float))
                else None,
                # Promoted only counts as a "submission" when --apply actually wrote
                # it; a proposed sidecar is staged, not shipped (deploy --apply parity).
                submission_ok=promoted and apply,
                node_count=len(getattr(result, "candidate_scores", []) or []),
            ),
            artifacts=[],
            success=promoted,
            model=None,
            tenant=config.tenant,
            # A general-method skill edit is a cross-tenant moat artifact; the audit
            # carries no tenant facts (only scores + the contract verdict).
            owned_by="general",
            reflection=(
                f"skillopt KEPT the incumbent ({verdict}); never deploy a worse skill"
                if not promoted
                else None
            ),
        )
        Learnings(config).record(record)
    except Exception:  # noqa: BLE001 - learnings are best-effort, never fatal
        pass


def _cmd_run(args: argparse.Namespace) -> int:
    """Handle ``loom run``: build a task + config, run it, print results.

    Args:
        args: Parsed command-line arguments for the ``run`` subcommand.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    config = _build_config(args)

    # Resolve the two input kinds. ``--dataset`` is a Metaflow data-object
    # pathspec (read via the Client API); ``--data`` is a local dir (the dev
    # fallback). A local --data, if given, must exist.
    dataset_ref = (args.dataset or "").strip() or None
    data_dir = ""
    if args.data:
        data_dir = os.path.abspath(args.data)
        if not os.path.isdir(data_dir):
            print(
                f"error: --data directory does not exist: {data_dir}",
                file=sys.stderr,
            )
            return 2

    # Pre-flight (input): the metaflow provider's input IS a Metaflow data
    # object, so it needs a --dataset pathspec (a local --data is only a
    # fallback for the 'local' provider). Guide the user to ingest first.
    if config.mlops_provider == "metaflow" and not dataset_ref and not data_dir:
        print(
            "error: the metaflow provider needs a Metaflow data object. Ingest "
            "your data first, then pass the printed pathspec:\n"
            "  loom ingest --source <path>\n"
            "  loom run --dataset <pathspec> --mlops metaflow ...\n"
            "(Or use the Metaflow-free dev path: --mlops local --data <dir>.)",
            file=sys.stderr,
        )
        return 2
    if not dataset_ref and not data_dir:
        print(
            "error: no input given. Pass --dataset <pathspec> (a `loom ingest` "
            "data object) or --data <dir> (a local directory).",
            file=sys.stderr,
        )
        return 2

    # Pre-flight (judge): AIDE's feedback step always uses tool calling, so a
    # feedback route that is not judge-capable would crash mid-run. Fail fast.
    judge_problem = _judge_preflight(config)
    if judge_problem is not None:
        print(f"error: {judge_problem}", file=sys.stderr)
        return 2

    # Pre-flight: the search brain needs an LLM credential. Fail fast with an
    # actionable message rather than a deep backend traceback at the first call.
    cred_hints = _llm_preflight(config)
    if cred_hints:
        print(
            "error: no LLM credential found for the configured model(s):",
            file=sys.stderr,
        )
        for hint in cred_hints:
            print(f"  - {hint}", file=sys.stderr)
        print(
            "\nLoom's search brain (AIDE) calls an LLM to write solutions. "
            "Set a key and re-run, e.g.:\n"
            "  export ANTHROPIC_API_KEY=...                       # Claude (default)\n"
            "  # or an OpenAI-compatible / NVIDIA NIM endpoint:\n"
            "  export OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1\n"
            "  export OPENAI_API_KEY=...\n"
            "See the README 'Configuration' section.",
            file=sys.stderr,
        )
        return 2

    experiment_id = args.experiment_id or f"loom-{uuid.uuid4().hex[:12]}"

    task = Task(
        data_dir=data_dir,
        goal=args.goal,
        eval=args.metric,
        experiment_id=experiment_id,
        tenant=config.tenant,
        dataset_ref=dataset_ref,
    )

    print(
        f"Running task {experiment_id!r} "
        f"(search={config.search_provider}, mlops={config.mlops_provider}, "
        f"steps={config.budget.steps})..."
    )

    try:
        result = run_loom(task, config)
    except Exception as exc:  # noqa: BLE001 - translate to an actionable message
        if _looks_like_auth_error(exc):
            print(
                f"\nerror: the LLM call failed with what looks like an "
                f"authentication problem:\n  {type(exc).__name__}: {exc}\n"
                "Check that the API key for your configured model is set and "
                "valid (see the README 'Configuration' section).",
                file=sys.stderr,
            )
            return 1
        raise

    # Read the leaderboard from the execution provider (best-effort). Resolving
    # the provider class again and instantiating it from config is cheap and
    # keeps the controller's return type minimal (it returns only SearchResult).
    leaderboard: list[dict] = []
    try:
        exec_cls = get_execution(config.mlops_provider)
        leaderboard = exec_cls(config).runs(experiment_id)
    except Exception:
        # A leaderboard is informational only; never fail the run over it.
        leaderboard = []

    _print_result(result, leaderboard)
    return 0


def _ui_disabled(args: argparse.Namespace) -> bool:
    """Return whether the Rich interactive UI is disabled (``--no-ui`` / env).

    The ``--no-ui`` flag (or a truthy ``LOOM_NO_UI`` environment variable) forces
    the REPL to plain, color-stripped output -- the CI/pipes posture. Reads the
    env lazily so a normal interactive run pays nothing.

    Args:
        args: The parsed top-level namespace.

    Returns:
        ``True`` when the UI should be plain (no Rich styling).
    """
    if getattr(args, "no_ui", False):
        return True
    return (os.environ.get("LOOM_NO_UI") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _launch_repl(args: argparse.Namespace) -> int:
    """Build the config and launch the interactive REPL (the no-command path).

    The single seam from the one-shot CLI into the interactive UI. ``rich`` and
    ``prompt_toolkit`` are imported here, LAZILY (inside this function), so a
    stripped environment that never opens the REPL still runs every one-shot
    subcommand without those packages installed.

    Args:
        args: The parsed top-level namespace (for the ``--no-ui`` flag).

    Returns:
        The REPL loop's exit code. A missing Rich/prompt_toolkit install yields
        an actionable message (and falls back to plain help), never a traceback.
    """
    config = _build_config(args)
    try:
        from loom.ui.repl import run_repl
    except Exception as exc:  # noqa: BLE001 - actionable hint, no traceback
        print(
            "error: the interactive Loom REPL needs `rich` + `prompt_toolkit` "
            f"but they could not be imported: {exc}\n"
            "Install them (they ship with Loom's deps) or use the one-shot "
            "subcommands (run `loom --help`).",
            file=sys.stderr,
        )
        return 2
    return run_repl(config, use_rich=not _ui_disabled(args))


def _cmd_chat(args: argparse.Namespace) -> int:
    """Handle ``loom chat``: launch the interactive REPL (explicit alias).

    Identical to invoking ``loom`` with no subcommand; provided as a discoverable
    verb. Delegates to :func:`_launch_repl` (which lazily imports the UI deps).

    Args:
        args: Parsed arguments for the ``chat`` subcommand.

    Returns:
        The REPL loop's exit code.
    """
    return _launch_repl(args)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``loom`` console script.

    With NO subcommand, ``loom`` drops into the branded interactive REPL
    (:mod:`loom.ui.repl`) -- the launch-into-REPL shape. Every existing
    one-shot subcommand still dispatches through its handler unchanged. ``rich``
    / ``prompt_toolkit`` are imported lazily inside the REPL path so a stripped
    environment still runs the one-shot commands.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``). Accepting
            it explicitly makes the CLI straightforward to unit-test.

    Returns:
        Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        # No subcommand -> the interactive REPL.
        return _launch_repl(args)

    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - module executed as a script
    raise SystemExit(main())
