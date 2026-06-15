"""The agent face — Anthropic-style tool schemas from the REGISTRY (DESIGN.md §2.1, §5.3).

One verb declaration generates both faces. This module emits the agent tool
schemas (``loom.<verb>``) from the same ``REGISTRY`` the CLI enumerates, and
dispatches a tool call to the verb ``fn``, returning the identical
:class:`~loom.types.VerbResult` the CLI ``--json`` face produces.

The ``confirm_token`` handshake (§5.3) is wired here as the shape gated verbs use:
a PLAN result carries a single-use, plan-hash-scoped, expiring ``confirm_token``;
the agent makes a SECOND call passing it back. No verb is gated in this Phase-0
slice (tokenize/ingest/baseline are cheap CPU writes), but the round-trip shape is
present so launch verbs inherit it unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

from .registry import REGISTRY, Verb, VerbContext
from .store import default_store
from .types import CapabilityMode, Status, Tier, VerbResult


def tool_schema(verb: Verb) -> dict[str, Any]:
    """Emit one Anthropic-style tool schema for a verb (name/description/input).

    ``disable-model-invocation`` is set True for irreversible / launch-and-track
    verbs so an agent structurally cannot fire money/destruction (§4, §5.2). None
    of the Phase-0 verbs are gated, but the flag is derived from tier/capability so
    launch verbs inherit it automatically."""
    gated = (
        verb.tier is Tier.IRREVERSIBLE
        or verb.capability_mode is CapabilityMode.LAUNCH_AND_TRACK
    )
    return {
        "name": f"loom.{verb.name}",
        "description": verb.summary,
        "input_schema": verb.params,
        # Loom-specific metadata the agent runtime reads (not part of the
        # Anthropic schema proper, but carried alongside it).
        "_loom": {
            "tier": verb.tier.value,
            "capability_mode": verb.capability_mode.value,
            "disable_model_invocation": gated,
        },
    }


def all_tool_schemas() -> list[dict[str, Any]]:
    """Emit tool schemas for every registered verb."""
    return [tool_schema(v) for v in REGISTRY.values()]


def dispatch(
    name: str,
    input_json: dict[str, Any],
    *,
    confirm_token: Optional[str] = None,
    experiment: Optional[str] = None,
) -> VerbResult:
    """Dispatch an agent tool call ``loom.<verb>(input_json)`` to the verb ``fn``.

    Accepts either ``"tokenize"`` or ``"loom.tokenize"``. Builds an agent-driver
    :class:`VerbContext` (interactive=False — an agent never owns a TTY, §7.5) and
    threads the optional ``confirm_token`` for the second-call handshake. Returns
    the dual-driver :class:`VerbResult` (the agent serializes it via
    ``.to_json()``)."""
    verb_name = name[len("loom.") :] if name.startswith("loom.") else name
    if verb_name not in REGISTRY:
        return VerbResult(
            verb=verb_name,
            status=Status.FAIL,
            verdict=__import__("loom.types", fromlist=["Verdict"]).Verdict.FAIL,
            tier=Tier.READ_ONLY,
            capability_mode=CapabilityMode.NONE,
            summary=f"unknown verb: {verb_name!r}",
        )
    verb = REGISTRY[verb_name]
    ctx = VerbContext(
        store=default_store(),
        experiment=experiment or input_json.get("experiment"),
        driver="agent",
        interactive=False,
        confirm_token=confirm_token or input_json.get("confirm_token"),
    )
    return verb.fn(dict(input_json), ctx)


def make_confirm_token(plan_hash: str) -> str:
    """Mint a single-use, plan-hash-scoped, expiring confirm_token (§5.3).

    STUB: launch verbs will implement minting (bound to the compiled-plan hash,
    default 15-min expiry, single-use). Cheap verbs in this slice accept any
    matching token. The signature is the locked shape."""
    raise NotImplementedError


def validate_confirm_token(token: Optional[str], plan_hash: str) -> bool:
    """Validate a confirm_token against a compiled-plan hash (single-use, unexpired,
    bound to this plan). STUB — locked shape for gated verbs (§5.3, §9)."""
    raise NotImplementedError
