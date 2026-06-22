"""The human face — argparse over the REGISTRY (DESIGN.md §2.1).

One subcommand per registered verb (generated from the same declaration the
agent tools come from). Global flags: ``--json`` (print the raw result envelope,
byte-identical to the agent tool result), ``--experiment <id>`` (the join key),
and ``-q/--quiet`` (print only the output pathspec on stdout, everything human to
stderr — §2.4). The CLI renders a :class:`~loom.types.VerbResult` as a pretty
card (human) or raw JSON (``--json``); exit code comes from the envelope.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Optional

import json

from .registry import REGISTRY, Verb, VerbContext
from .store import default_store
from .tools import all_tool_schemas
from .types import Severity, Status, Tier, Verdict, VerbResult


def _global_flags_parser() -> argparse.ArgumentParser:
    """The global control flags (``--json``/``--experiment``/``-q``) as a reusable
    parent parser (DESIGN.md §2.4).

    Sharing them via ``parents=`` puts them on BOTH the top-level parser and every
    subparser, so they're accepted in either position — ``loom -q tokenize …`` and
    ``loom tokenize … -q`` both work. The defaults are ``argparse.SUPPRESS`` so a
    flag that is *not* passed at the subparser level never clobbers a value already
    set in the pre-verb position (without SUPPRESS, the post-verb ``store_true``
    default ``False`` would overwrite a pre-verb ``--json``). ``main`` then reads
    each control off the namespace with a ``getattr`` fallback default."""
    g = argparse.ArgumentParser(add_help=False)
    g.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                   help="print the raw result envelope (byte-identical to the agent tool result)")
    g.add_argument("--experiment", default=argparse.SUPPRESS,
                   help="experiment id — the join key threading runs together")
    g.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS,
                   help="print only the output pathspec on stdout (humans to stderr)")
    return g


def _add_verb_subparser(
    sub: argparse._SubParsersAction, verb: Verb, parents: list[argparse.ArgumentParser]
) -> None:
    """Generate a subparser for a verb from its JSON-Schema ``params``.

    Each top-level property becomes a flag (``--name``); ``in``/positional inputs
    are handled by the verb's own arg conventions. This is intentionally simple
    for the scaffold — the verb-implementing agents refine per-verb argument
    surfaces; the schema is the source of truth. ``parents`` carries the global
    control flags so they're accepted AFTER the verb too (DESIGN.md §2.4)."""
    p = sub.add_parser(verb.name, help=verb.summary, aliases=list(verb.aliases),
                       parents=parents)
    props = (verb.params or {}).get("properties", {})
    required = set((verb.params or {}).get("required", []))
    for arg_name, schema in props.items():
        flag = "--" + arg_name.replace("_", "-")
        jtype = schema.get("type")
        kwargs: dict[str, Any] = {"help": schema.get("description", "")}
        if jtype == "boolean":
            kwargs["action"] = "store_true"
        elif jtype == "integer":
            kwargs["type"] = int
        elif jtype == "number":
            kwargs["type"] = float
        if arg_name in required and jtype != "boolean":
            kwargs["required"] = True
        p.add_argument(flag, dest=arg_name, **kwargs)
    # A positional input pathspec/source is common to the verbs; accept it loosely.
    p.add_argument("input", nargs="?", default=None,
                   help="input pathspec or source (verb-specific)")
    p.set_defaults(_verb=verb)


def build_parser() -> argparse.ArgumentParser:
    global_flags = _global_flags_parser()
    parser = argparse.ArgumentParser(
        prog="loom",
        description="Loom — typed verbs you compile before you spend. "
                    "A human and a Claude/Codex agent drive the identical verbs.",
        parents=[global_flags],
    )
    sub = parser.add_subparsers(dest="verb", metavar="<verb>")
    for verb in REGISTRY.values():
        _add_verb_subparser(sub, verb, parents=[global_flags])
    return parser


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_VERDICT_GLYPH = {
    Verdict.PASS: "✓",      # check
    Verdict.REVIEW: "⚠",    # warning
    Verdict.FAIL: "✗",      # cross
    Verdict.INCOMPLETE: "…",
}
_SEVERITY_GLYPH = {
    Severity.INFO: "·",
    Severity.WARNING: "⚠",
    Severity.ERROR: "✗",
}


def render_card(result: VerbResult) -> str:
    """Render a VerbResult as a human-readable terminal card (DESIGN.md §7.2)."""
    lines: list[str] = []
    glyph = _VERDICT_GLYPH.get(result.verdict, "")
    head = f"{glyph} {result.verb}  status={result.status.value}  verdict={result.verdict.value}"
    if result.experiment:
        head += f"  experiment={result.experiment}"
    lines.append(head)
    meta = f"  tier={result.tier.value}"
    if result.capability_mode.value != "none":
        meta += f"  capability={result.capability_mode.value}"
    lines.append(meta)
    if result.summary:
        lines.append(f"  {result.summary}")
    for o in result.outputs:
        lines.append(f"  → {o.pathspec}")
    if result.cost_plan and result.cost_plan.usd is not None:
        cp = result.cost_plan
        lines.append(f"  COST (derived={cp.derived}): ~${cp.usd}  confidence={cp.confidence}")
    for d in result.diagnostics:
        sg = _SEVERITY_GLYPH.get(d.severity, "")
        lines.append(f"  {sg} {d.contract}: {d.message}")
        if d.fix:
            lines.append(f"      fix: {d.fix}")
    if result.confirm_token:
        lines.append(f"  confirm_token: {result.confirm_token}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    # ``verbs`` is a CLI-ONLY manifest command, intercepted BEFORE the
    # registry-driven argparse (DESIGN.md §2.1; PI.md §B.4 the bridge seam). It is
    # deliberately NOT a registered Verb: registering it would enumerate it into
    # ``all_tool_schemas()`` and thereby register it as a Pi tool the agent could
    # call — but it is plumbing for the Node bridge, not a domain verb. It prints
    # ``json.dumps(all_tool_schemas())`` (the 3 verb schemas, each carrying
    # ``_loom.{tier, capability_mode, disable_model_invocation}``) and exits 0 so
    # the bridge can read the tool manifest without booting an agent.
    args_list = sys.argv[1:] if argv is None else argv
    if args_list and args_list[0] == "verbs":
        print(json.dumps(all_tool_schemas()))
        return 0

    parser = build_parser()
    ns = parser.parse_args(argv)

    if not getattr(ns, "verb", None):
        parser.print_help()
        return 0

    verb: Verb = ns._verb

    # Global control flags carry ``argparse.SUPPRESS`` defaults so a flag unset in
    # the post-verb position can't clobber a value set pre-verb; read each off the
    # namespace regardless of position, falling back to the unset default.
    want_json = getattr(ns, "json", False)
    want_quiet = getattr(ns, "quiet", False)
    experiment = getattr(ns, "experiment", None)

    # Collect verb args from the namespace (everything except global/control keys).
    # A ``store_true`` flag that was NOT passed shows up as ``False``; we drop those
    # so an unset boolean flag never overrides a verb/preset default with False
    # (e.g. chain's T2 default is identity-token OFF, expressed as the verb
    # defaulting ``no_identity_token`` True — a bare ``loom tokenize --preset chain``
    # must not force it False). A boolean flag is meaningful only when set True.
    control = {"json", "experiment", "quiet", "verb", "_verb", "input"}
    args: dict[str, Any] = {
        k: v
        for k, v in vars(ns).items()
        if k not in control and v is not None and v is not False
    }
    if ns.input is not None:
        args.setdefault("in", ns.input)

    ctx = VerbContext(
        store=default_store(),
        experiment=experiment,
        driver="cli",
        interactive=sys.stdin.isatty(),
        quiet=want_quiet,
    )

    result = verb.fn(args, ctx)

    if want_quiet:
        # Machine-pipeable: only the output pathspec on stdout (§2.4).
        for o in result.outputs:
            print(o.pathspec)
        if not result.outputs:
            print(render_card(result), file=sys.stderr)
    elif want_json:
        print(result.to_json())
    else:
        print(render_card(result))

    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
