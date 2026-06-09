"""Loom's interactive CLI UI package.

A thin, Rich-based render + approval layer that makes every ``loom`` verb
beautiful and drives a branded interactive REPL. Modeled on Feynman's lean
themed terminal library and Claude Code's loop/approval patterns, but it owns
no engine logic: it renders the typed summaries the verbs in
:mod:`loom.cli` already produce and gates per the approval matrix in
``skills/CONVENTIONS.md`` §1.

Submodules:

* :mod:`loom.ui.theme` -- the Loom color palette, the shared ``get_console``
  factory (headless-capable over a ``file``), the ``info``/``success``/
  ``warning``/``error``/``section``/``panel`` helpers, and ``banner(config)``.
* :mod:`loom.ui.render` -- PURE functions mapping a verb's typed summary dict
  (or a :class:`~loom.types.RunResult`/:class:`~loom.types.SearchResult`) to a
  Rich renderable.
* :mod:`loom.ui.gate` -- the approval UX: :func:`loom.ui.gate.gate` enforces the
  read-only / workspace-write / expensive / irreversible tiers with an
  injectable confirm callable.

Everything here imports ``rich`` (and, in the REPL, ``prompt_toolkit``) and is
intended to be imported LAZILY from the CLI so a stripped environment still runs
the one-shot subcommands.
"""

from __future__ import annotations

__all__ = ["theme", "render", "gate"]
