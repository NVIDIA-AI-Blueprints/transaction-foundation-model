"""Command-line interface for Loom.

Exposes the ``loom`` console script (wired in ``pyproject.toml`` as
``loom = loom.cli:main``). v0.1 ships a single subcommand:

``loom run --data DIR --goal STR --metric STR [--steps N]
         [--mlops metaflow|local] [--search aide]``

The command builds a :class:`~loom.types.Task` and a
:class:`~loom.config.LoomConfig` from the parsed arguments (provider/budget
overrides on top of the env/.env/YAML-derived config), runs the task through
:func:`loom.controller.run_loom`, and prints the best metric, the artifact
paths from the returned :class:`~loom.types.SearchResult`, and a short
leaderboard read from the execution provider's ``runs()``.

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
        required=True,
        metavar="DIR",
        help="Path to the task's input data directory (staged into ./input).",
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
    if args.mlops is not None:
        overrides["mlops_provider"] = args.mlops
    if args.search is not None:
        overrides["search_provider"] = args.search
    if args.steps is not None:
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
            metric = _format_metric(row.get("metric"))
            node_id = row.get("node_id", row.get("id", "?"))
            stage = row.get("stage", "")
            suffix = f"  [{stage}]" if stage else ""
            print(f"  {rank:>2}. metric={metric}  node={node_id}{suffix}")


def _cmd_run(args: argparse.Namespace) -> int:
    """Handle ``loom run``: build a task + config, run it, print results.

    Args:
        args: Parsed command-line arguments for the ``run`` subcommand.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    data_dir = os.path.abspath(args.data)
    if not os.path.isdir(data_dir):
        print(f"error: --data directory does not exist: {data_dir}", file=sys.stderr)
        return 2

    config = _build_config(args)

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
