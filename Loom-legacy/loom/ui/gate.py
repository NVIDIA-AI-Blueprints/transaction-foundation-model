"""The approval UX: the four-tier cost/data gate (CONVENTIONS §1).

This is Loom's interactive parallel of Claude Code's deny-first approval cascade,
specialized to the approval matrix in ``skills/CONVENTIONS.md`` §1. The gate is
**orthogonal to capability**: it decides only whether an action may proceed,
based on its declared *tier*, never on what the action does.

The tiers and their behaviour:

* ``read-only`` -- reads are non-destructive; **allow, never prompt.**
* ``workspace-write`` -- light/auto within a declared budget; **allow with a
  one-line auto note**, no prompt.
* ``expensive`` / ``mutate`` -- **always gate:** an interactive y/N confirm that
  shows the cost / rows / operation before running.
* ``irreversible`` / ``external`` -- **always gate, deny-first:** the same
  interactive confirm, and (per the matrix) the model never auto-fires it -- only
  the user does.

:func:`gate` returns a :class:`Decision` (allow / deny + the tier + a human
reason). The ``confirm`` callable is **injectable** so tests drive the y/N answer
without a TTY (and so the REPL can wire it to a real Rich prompt). Rendering is
done through a Loom console (headless-capable over a buffer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rich.console import Console


# ---------------------------------------------------------------------------
# Tiers (CONVENTIONS §1).
# ---------------------------------------------------------------------------

#: Reads -- never prompt.
READ_ONLY = "read-only"
#: Light/auto within budget -- allow with a one-line note, no prompt.
WORKSPACE_WRITE = "workspace-write"
#: GPU/large/full-scan/registry-edit -- always gate (y/N confirm).
EXPENSIVE = "expensive"
#: Deploy / off-box / drop-gold-table -- always gate, deny-first, user-only.
IRREVERSIBLE = "irreversible"

#: Aliases the matrix uses interchangeably, normalized to the canonical tier.
_TIER_ALIASES = {
    READ_ONLY: READ_ONLY,
    "read": READ_ONLY,
    "readonly": READ_ONLY,
    WORKSPACE_WRITE: WORKSPACE_WRITE,
    "workspace": WORKSPACE_WRITE,
    "write": WORKSPACE_WRITE,
    "light": WORKSPACE_WRITE,
    "auto": WORKSPACE_WRITE,
    EXPENSIVE: EXPENSIVE,
    "mutate": EXPENSIVE,
    "expensive/mutate": EXPENSIVE,
    "expensive / mutate": EXPENSIVE,
    IRREVERSIBLE: IRREVERSIBLE,
    "external": IRREVERSIBLE,
    "irreversible/external": IRREVERSIBLE,
    "irreversible / external": IRREVERSIBLE,
}

#: The tiers that require an interactive confirm.
_GATED_TIERS = frozenset({EXPENSIVE, IRREVERSIBLE})


@dataclass(frozen=True)
class Decision:
    """The outcome of a :func:`gate` call.

    Attributes:
        allow: Whether the action may proceed.
        tier: The canonical tier that was evaluated.
        prompted: Whether an interactive confirm was shown (``False`` for the two
            auto-allow tiers).
        reason: A short human-readable explanation (the auto-note, or why a
            gated action was allowed/denied).
    """

    allow: bool
    tier: str
    prompted: bool
    reason: str

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        """A Decision is truthy iff it allows."""
        return self.allow


def normalize_tier(tier: str) -> str:
    """Map a declared tier string (with aliases) to the canonical tier.

    Args:
        tier: The tier as declared by the verb (e.g. ``"mutate"``,
            ``"irreversible / external"``).

    Returns:
        One of :data:`READ_ONLY` / :data:`WORKSPACE_WRITE` / :data:`EXPENSIVE` /
        :data:`IRREVERSIBLE`. An unknown tier is treated as :data:`EXPENSIVE`
        (fail safe: gate it rather than wave it through).
    """
    return _TIER_ALIASES.get((tier or "").strip().lower(), EXPENSIVE)


def _default_confirm(prompt: str) -> bool:
    """The default confirm: a deny-first Rich y/N prompt over the real terminal.

    Used only when no ``confirm`` callable is injected. Defaults to **No** so a
    bare Enter / EOF / interrupt denies (deny-first).

    Args:
        prompt: The one-line question to show.

    Returns:
        ``True`` only if the user explicitly affirms.
    """
    try:
        from rich.prompt import Confirm

        return bool(Confirm.ask(prompt, default=False))
    except Exception:  # noqa: BLE001 - EOF / no TTY / interrupt all mean "deny"
        return False


def _format_detail(detail: Optional[Mapping[str, Any]]) -> str:
    """Render the cost/rows/operation detail mapping as a compact one-liner."""
    if not detail:
        return ""
    parts = [f"{k}={v}" for k, v in detail.items() if v is not None]
    return "; ".join(parts)


def gate(
    tier: str,
    action: str,
    detail: Optional[Mapping[str, Any]] = None,
    *,
    confirm: Optional[Callable[[str], bool]] = None,
    console: Optional["Console"] = None,
) -> Decision:
    """Evaluate the approval gate for an action and (when gated) prompt for it.

    Enforces CONVENTIONS §1: read-only and workspace-write auto-allow (the
    latter with a one-line note); expensive/mutate and irreversible/external
    always require an interactive y/N confirm that surfaces the cost / rows /
    operation, and deny-first (a bare Enter / EOF denies).

    Args:
        tier: The declared tier (aliases accepted; see :func:`normalize_tier`).
        action: A short label for what is about to happen (e.g.
            ``"deploy to production registry"``).
        detail: Optional mapping of the cost/rows/operation to show at the gate
            (e.g. ``{"rows": 1_200_000, "est_usd": 4.20, "op": "overwrite"}``).
        confirm: Injectable callable taking the prompt string and returning a
            bool. Defaults to a deny-first Rich prompt; tests inject a lambda so
            no TTY is needed.
        console: Optional Loom console (headless-capable) used to print the gate
            context line before prompting. ``None`` builds a default console only
            if something needs printing.

    Returns:
        A :class:`Decision`.
    """
    canonical = normalize_tier(tier)
    detail_str = _format_detail(detail)

    # Auto-allow tiers: no prompt.
    if canonical == READ_ONLY:
        return Decision(
            allow=True,
            tier=canonical,
            prompted=False,
            reason=f"read-only: {action} runs free (non-destructive).",
        )
    if canonical == WORKSPACE_WRITE:
        note = f"workspace-write (auto): {action} within budget; network off"
        if detail_str:
            note += f" [{detail_str}]"
        return Decision(allow=True, tier=canonical, prompted=False, reason=note + ".")

    # Gated tiers: always an interactive confirm, deny-first.
    confirm = confirm or _default_confirm

    # Surface the cost/operation context BEFORE the prompt (the matrix mandate:
    # "Show estimated cost/rows and the exact operation before running").
    if console is None:
        from loom.ui.theme import get_console

        console = get_console()

    label = "EXPENSIVE/MUTATE" if canonical == EXPENSIVE else "IRREVERSIBLE/EXTERNAL"
    extra = (
        " The model proposes; only you can fire this."
        if canonical == IRREVERSIBLE
        else ""
    )
    from loom.ui.theme import make_panel
    from rich.text import Text

    lines = Text()
    lines.append(f"{action}\n", style="loom.ink")
    if detail_str:
        lines.append(f"{detail_str}\n", style="loom.warning")
    lines.append(f"This is the {label} tier -- it always gates.{extra}",
                 style="loom.stone")
    console.print(make_panel(f"Approval required: {label}", lines,
                             border_style="loom.warning"))

    question = f"Proceed with {action}?"
    if confirm(question):
        return Decision(
            allow=True,
            tier=canonical,
            prompted=True,
            reason=f"{label}: user approved {action}.",
        )
    return Decision(
        allow=False,
        tier=canonical,
        prompted=True,
        reason=f"{label}: BLOCKED -- {action} not approved.",
    )


__all__ = [
    "READ_ONLY",
    "WORKSPACE_WRITE",
    "EXPENSIVE",
    "IRREVERSIBLE",
    "Decision",
    "normalize_tier",
    "gate",
]
