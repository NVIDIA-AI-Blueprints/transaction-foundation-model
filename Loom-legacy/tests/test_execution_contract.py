"""Gating test: Loom's ExecutionResult must stay field-identical to AIDE's.

This is the contract that lets any Loom :class:`~loom.providers.ExecutionProvider`
be handed straight to AIDE as an ``exec_callback``: the AIDE adapter converts a
loom :class:`~loom.types.ExecutionResult` into ``aide.interpreter.ExecutionResult``
with a flat field-for-field copy
(``aide.interpreter.ExecutionResult(**dataclasses.asdict(result))``). If the two
dataclasses ever drift apart, that conversion breaks — so this test fails fast.

The whole module needs a real AIDE install to compare against, so it is skipped
(never errored) when ``aide`` is absent.
"""

from __future__ import annotations

import dataclasses

import pytest

from loom.types import ExecutionResult as LoomExecutionResult

# AIDE is an optional, heavy dependency; skip the entire module when missing so
# the pure-Python suite still runs. The conversion seam is meaningless without
# AIDE present anyway.
aide = pytest.importorskip("aide")
import aide.interpreter as aide_interpreter  # noqa: E402  (after importorskip)

AideExecutionResult = aide_interpreter.ExecutionResult


def _field_specs(cls: type) -> list[tuple[str, object, object]]:
    """Return ``(name, type, default)`` for each field of a dataclass.

    ``default`` is :data:`dataclasses.MISSING` when the field has no default,
    which lets us compare both the field *order* and the *optionality* of the
    two dataclasses, not merely the set of names.
    """
    return [
        (f.name, f.type, f.default)
        for f in dataclasses.fields(cls)
    ]


def test_both_are_dataclasses() -> None:
    """Sanity check that both sides are dataclasses we can introspect."""
    assert dataclasses.is_dataclass(LoomExecutionResult)
    assert dataclasses.is_dataclass(AideExecutionResult)


def test_field_names_and_order_match() -> None:
    """Field names and their order must be identical between the two types."""
    loom_names = [f.name for f in dataclasses.fields(LoomExecutionResult)]
    aide_names = [f.name for f in dataclasses.fields(AideExecutionResult)]
    assert loom_names == aide_names == [
        "term_out",
        "exec_time",
        "exc_type",
        "exc_info",
        "exc_stack",
    ]


def test_field_defaults_match() -> None:
    """Optional fields (and their default values) must match.

    ``exc_info`` and ``exc_stack`` default to ``None``; ``term_out``,
    ``exec_time`` and ``exc_type`` are required (no default).
    """
    loom_defaults = {
        f.name: f.default for f in dataclasses.fields(LoomExecutionResult)
    }
    aide_defaults = {
        f.name: f.default for f in dataclasses.fields(AideExecutionResult)
    }
    assert loom_defaults == aide_defaults


def test_loom_to_aide_roundtrip() -> None:
    """A loom result converts to an AIDE result with a flat asdict copy."""
    loom_result = LoomExecutionResult(
        term_out=["hello\n", "Execution time: 1 second seconds (time limit is 1 hour)."],
        exec_time=0.42,
        exc_type=None,
        exc_info=None,
        exc_stack=None,
    )

    aide_result = AideExecutionResult(**dataclasses.asdict(loom_result))

    assert aide_result.term_out == loom_result.term_out
    assert aide_result.exec_time == loom_result.exec_time
    assert aide_result.exc_type == loom_result.exc_type
    assert aide_result.exc_info == loom_result.exc_info
    assert aide_result.exc_stack == loom_result.exc_stack


def test_aide_to_loom_roundtrip_with_exception_fields() -> None:
    """An AIDE result (incl. exception metadata) converts back to a loom result.

    Mirrors the exact shape AIDE's interpreter produces on a raised exception:
    an ``exc_info`` dict and an ``exc_stack`` list of
    ``(filename, lineno, name, line)`` tuples.
    """
    aide_result = AideExecutionResult(
        term_out=["Traceback (most recent call last):\n"],
        exec_time=0.01,
        exc_type="ValueError",
        exc_info={"args": ["boom"], "msg": "boom"},
        exc_stack=[("solution.py", 3, "<module>", "raise ValueError('boom')")],
    )

    loom_result = LoomExecutionResult(**dataclasses.asdict(aide_result))

    assert dataclasses.asdict(loom_result) == dataclasses.asdict(aide_result)
    assert loom_result.exc_type == "ValueError"
    assert loom_result.exc_info == {"args": ["boom"], "msg": "boom"}
    assert loom_result.exc_stack == [
        ("solution.py", 3, "<module>", "raise ValueError('boom')")
    ]
