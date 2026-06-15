"""Verb package — importing it self-registers every verb into the REGISTRY.

Each verb module calls ``@register(...)`` at import time, so importing this
package wires both faces (CLI + agent tools). Imports are GUARDED: a verb module
that is mid-build (raising at import) is logged and skipped rather than breaking
``import loom`` for everyone (DESIGN.md item ordering — verbs land incrementally).
"""

from __future__ import annotations

import importlib
import logging

_log = logging.getLogger("loom.verbs")

# The Phase-0 verbs, in build order (tokenize → ingest → baseline).
_VERB_MODULES = ("tokenize", "ingest", "baseline")

for _name in _VERB_MODULES:
    try:
        importlib.import_module(f"{__name__}.{_name}")
    except Exception as exc:  # pragma: no cover - defensive wiring
        _log.warning("verb module %r failed to import and was skipped: %s", _name, exc)
