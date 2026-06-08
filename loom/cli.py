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
from typing import Sequence

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

    return LoomConfig.load(yaml_path=args.config, overrides=overrides)


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

    result = run_loom(task, config)

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
