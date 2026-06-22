"""The NARROW WAIST — one typed verb declaration, two faces (DESIGN.md §2.1).

A verb is declared exactly once via :func:`register`. From that single
declaration Loom generates (a) the human CLI ``loom <verb> … [--json]`` and
(b) the agent tool ``loom.<verb>(…)``. Both faces read the same argument schema,
tier, capability mode, and call the same ``fn`` returning a single
:class:`~loom.types.VerbResult`.

LOCKED: implementers register their verb with ``@register(...)`` and write a
function ``fn(args: dict, ctx: VerbContext) -> VerbResult``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .types import CapabilityMode, Tier, VerbResult

# A verb implementation: takes the validated argument dict and a runtime context,
# returns the dual-driver result envelope.
VerbFn = Callable[[dict[str, Any], "VerbContext"], VerbResult]


@dataclass
class VerbContext:
    """Runtime context handed to every verb ``fn``.

    Carries the workspace store, the resolved experiment, and the driver-face
    signal so a verb can mint/validate a ``confirm_token`` correctly (§5.3) and
    decide non-interactive behavior (§2.4). The store is typed loosely here to
    avoid an import cycle with :mod:`loom.store`.
    """

    store: Any                      # loom.store.ObjectStore
    experiment: Optional[str] = None
    driver: str = "cli"             # "cli" | "agent"
    interactive: bool = True        # False when stdin is not a TTY / piped (§2.4)
    confirm_token: Optional[str] = None  # the agent's second-call token (§5.3)
    quiet: bool = False             # -q: print only the output pathspec on stdout
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Verb:
    """A single verb declaration (the typed contract).

    Attributes
    ----------
    name : the verb name, e.g. ``"tokenize"``.
    summary : one-line human summary, also the agent tool description.
    tier : the verb's :class:`~loom.types.Tier` (a property, not a flag).
    capability_mode : the verb's :class:`~loom.types.CapabilityMode`.
    params : a JSON-Schema dict describing the argument set (drives both the
        argparse CLI and the agent tool's ``input_schema``).
    fn : the implementation ``fn(args, ctx) -> VerbResult``.
    aliases : optional alternate CLI names.
    """

    name: str
    summary: str
    tier: Tier
    capability_mode: CapabilityMode
    params: dict[str, Any]
    fn: VerbFn
    aliases: tuple[str, ...] = ()


# The single source of truth both faces enumerate over.
REGISTRY: dict[str, Verb] = {}


def register(
    name: str,
    *,
    summary: str,
    tier: Tier,
    capability_mode: CapabilityMode = CapabilityMode.NONE,
    params: Optional[dict[str, Any]] = None,
    aliases: tuple[str, ...] = (),
) -> Callable[[VerbFn], VerbFn]:
    """Decorator that registers a verb ``fn`` as a typed contract.

    Usage::

        @register("tokenize", summary="compile a tokenizer spec to a Corpus",
                  tier=Tier.WORKSPACE_WRITE, params=TOKENIZE_PARAMS)
        def tokenize(args: dict, ctx: VerbContext) -> VerbResult:
            ...

    Returns the original function unchanged (so it stays directly callable in
    tests). Re-registering the same name overwrites the prior declaration, which
    lets verb modules be reloaded safely.
    """

    def _decorator(fn: VerbFn) -> VerbFn:
        REGISTRY[name] = Verb(
            name=name,
            summary=summary,
            tier=tier,
            capability_mode=capability_mode,
            params=params or {"type": "object", "properties": {}},
            fn=fn,
            aliases=aliases,
        )
        return fn

    return _decorator


def get(name: str) -> Verb:
    """Look a verb up by name (raises ``KeyError`` if unregistered)."""
    return REGISTRY[name]
