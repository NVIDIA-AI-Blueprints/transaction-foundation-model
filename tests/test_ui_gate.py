"""Tests for ``loom.ui.gate`` -- the four-tier cost/data approval gate.

The gate enforces CONVENTIONS §1, deny-first, and is **orthogonal to
capability**: it decides only whether an action may proceed, from its declared
*tier*. These tests pin the whole tier -> behavior matrix:

* ``read-only`` -> allow, **never prompts** (the injected confirm is never
  called);
* ``workspace-write`` -> allow with a one-line auto note, **never prompts**;
* ``expensive`` / ``mutate`` -> **always prompts**; a confirm returning ``False``
  BLOCKS, a ``True`` allows; the cost/operation detail is surfaced before the
  prompt;
* ``irreversible`` / ``external`` -> **always prompts**, deny-first, and the
  context names it user-only;
* tier aliases normalize (``mutate`` -> expensive, ``external`` ->
  irreversible), an unknown tier fails safe to expensive (gated, not waved
  through);
* a :class:`~loom.ui.gate.Decision` is truthy iff it allows.

The ``confirm`` callable is injected so no TTY is needed; the console is built
headless over a ``StringIO`` so the gate's pre-prompt context line is asserted
without a terminal.
"""

from __future__ import annotations

from io import StringIO

import pytest

from loom.ui import gate as gate_mod
from loom.ui import theme


def _console() -> theme.Console:
    return theme.get_console(file=StringIO(), force_terminal=False, width=120)


def _never(_prompt: str) -> bool:
    """A confirm that fails the test if it is ever called (auto-allow tiers)."""
    raise AssertionError("the gate must not prompt for an auto-allow tier")


# ---------------------------------------------------------------------------
# Auto-allow tiers: never prompt.
# ---------------------------------------------------------------------------


def test_read_only_allows_and_never_prompts() -> None:
    decision = gate_mod.gate("read-only", "profile the data object", confirm=_never, console=_console())
    assert decision.allow is True
    assert decision.prompted is False
    assert decision.tier == gate_mod.READ_ONLY
    assert bool(decision) is True  # truthy iff allow
    assert "read-only" in decision.reason


def test_workspace_write_allows_with_note_and_never_prompts() -> None:
    decision = gate_mod.gate(
        "workspace-write",
        "build engineered features",
        {"rows": 1000},
        confirm=_never,
        console=_console(),
    )
    assert decision.allow is True
    assert decision.prompted is False
    assert decision.tier == gate_mod.WORKSPACE_WRITE
    # the one-line auto note mentions it ran within budget, network off
    assert "auto" in decision.reason and "budget" in decision.reason


# ---------------------------------------------------------------------------
# Gated tiers: always prompt; deny-first; the confirm decides.
# ---------------------------------------------------------------------------


def test_expensive_prompts_and_deny_blocks() -> None:
    calls: list[str] = []

    def deny(prompt: str) -> bool:
        calls.append(prompt)
        return False

    decision = gate_mod.gate(
        "expensive",
        "launch the heavy GPU training run",
        {"gpu_hours": 96, "est_usd": 288},
        confirm=deny,
        console=_console(),
    )
    assert calls, "an expensive action must prompt"
    assert decision.allow is False  # deny BLOCKS
    assert decision.prompted is True
    assert decision.tier == gate_mod.EXPENSIVE
    assert "BLOCKED" in decision.reason


def test_expensive_approve_allows() -> None:
    decision = gate_mod.gate(
        "expensive",
        "overwrite the gold feature table",
        {"rows": 1_200_000},
        confirm=lambda _p: True,
        console=_console(),
    )
    assert decision.allow is True
    assert decision.prompted is True


def test_expensive_surfaces_cost_detail_before_prompt() -> None:
    # The matrix mandates showing estimated cost/rows + the exact operation.
    console = _console()
    gate_mod.gate(
        "expensive",
        "full-table scan",
        {"rows": 5_000_000, "est_usd": 12.5},
        confirm=lambda _p: False,
        console=console,
    )
    out = console.file.getvalue()
    assert "full-table scan" in out
    assert "rows=5000000" in out and "est_usd=12.5" in out
    assert "EXPENSIVE" in out


def test_irreversible_prompts_deny_first_and_is_user_only() -> None:
    console = _console()
    decision = gate_mod.gate(
        "irreversible",
        "deploy to the external registry",
        {"target": "prod"},
        confirm=lambda _p: True,
        console=console,
    )
    assert decision.allow is True
    assert decision.prompted is True
    assert decision.tier == gate_mod.IRREVERSIBLE
    out = console.file.getvalue()
    assert "IRREVERSIBLE/EXTERNAL" in out
    # the context names it user-only ("the model proposes; only you can fire this")
    assert "only you" in out.lower()


def test_irreversible_deny_blocks() -> None:
    decision = gate_mod.gate("external", "send the bundle off-box", confirm=lambda _p: False, console=_console())
    assert decision.allow is False
    assert decision.prompted is True
    assert decision.tier == gate_mod.IRREVERSIBLE


# ---------------------------------------------------------------------------
# Tier normalization + the fail-safe.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alias, expected",
    [
        ("read", gate_mod.READ_ONLY),
        ("readonly", gate_mod.READ_ONLY),
        ("workspace", gate_mod.WORKSPACE_WRITE),
        ("light", gate_mod.WORKSPACE_WRITE),
        ("auto", gate_mod.WORKSPACE_WRITE),
        ("mutate", gate_mod.EXPENSIVE),
        ("expensive / mutate", gate_mod.EXPENSIVE),
        ("external", gate_mod.IRREVERSIBLE),
        ("irreversible / external", gate_mod.IRREVERSIBLE),
    ],
)
def test_normalize_tier_aliases(alias: str, expected: str) -> None:
    assert gate_mod.normalize_tier(alias) == expected


def test_unknown_tier_fails_safe_to_expensive_and_gates() -> None:
    # An unknown tier must NOT be waved through: it normalizes to EXPENSIVE and
    # therefore prompts (deny-first), rather than silently auto-allowing.
    assert gate_mod.normalize_tier("totally-made-up") == gate_mod.EXPENSIVE
    decision = gate_mod.gate("totally-made-up", "do a risky thing", confirm=lambda _p: False, console=_console())
    assert decision.prompted is True
    assert decision.allow is False


# ---------------------------------------------------------------------------
# Decision dataclass.
# ---------------------------------------------------------------------------


def test_decision_is_truthy_iff_allow() -> None:
    assert bool(gate_mod.Decision(allow=True, tier="read-only", prompted=False, reason="ok"))
    assert not bool(gate_mod.Decision(allow=False, tier="expensive", prompted=True, reason="blocked"))
