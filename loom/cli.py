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
from typing import Optional, Sequence

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
            "Loom: a general-purpose, domain-neutral automated ML engine "
            "(ports-and-adapters provider architecture)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

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

    return LoomConfig.load(yaml_path=args.config, overrides=overrides)


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

    # Lazy import: Metaflow is an optional dependency, pulled in only when an
    # operation that needs it (ingest) is actually invoked.
    try:
        from metaflow import Runner
    except Exception as exc:  # noqa: BLE001 - actionable hint, no traceback
        print(
            f"error: Metaflow is required for `loom ingest` but could not be "
            f"imported: {exc}\nInstall it (it ships with Loom's deps) and try "
            "again.",
            file=sys.stderr,
        )
        return 2

    from flows import INGEST_DATASET_FLOW_PATH

    name = (args.name or "").strip()
    runner_kwargs: dict[str, object] = {}
    if config.metaflow_profile:
        runner_kwargs["profile"] = config.metaflow_profile

    run_kwargs: dict[str, object] = {"source": source}
    if name:
        # The flow's name Parameter is id'd ``dataset_name`` (``name`` is reserved
        # by Metaflow's CLI options).
        run_kwargs["dataset_name"] = name
        run_kwargs["tags"] = [f"loom_dataset:{name}"]

    print(f"Ingesting {source!r} into a Metaflow data object...")
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
        print(
            f"error: failed to run IngestDataset flow: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    # The Runner returns once the subprocess exits; a failed flow still yields an
    # ExecutingRun, so check status before claiming success (otherwise we would
    # print a pathspec whose run has no data object).
    if status is not None and status != "successful":
        print(
            f"error: IngestDataset did not complete successfully (status="
            f"{status!r}). Re-run with the Metaflow logs for details.",
            file=sys.stderr,
        )
        return 1

    pathspec = getattr(run, "pathspec", None) or str(run)
    print("Ingest complete.")
    print(f"  dataset_ref : {pathspec}")
    print("")
    print("Run a task against it with:")
    print(f"  loom run --dataset {pathspec} --mlops metaflow --goal '...' --metric '...'")
    return 0


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


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``loom`` console script.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``). Accepting
            it explicitly makes the CLI straightforward to unit-test.

    Returns:
        Process exit code. With no subcommand, prints help and returns 1.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - module executed as a script
    raise SystemExit(main())
