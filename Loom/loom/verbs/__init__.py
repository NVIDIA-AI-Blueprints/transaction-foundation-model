"""Verb package — importing it self-registers every verb into the REGISTRY.

Each verb module calls ``@register(...)`` at import time, so importing this
package wires both faces (CLI + agent tools). Imports are GUARDED: a verb module
that is mid-build (raising at import) is logged and skipped rather than breaking
``import loom`` for everyone (DESIGN.md item ordering — verbs land incrementally).
"""

from __future__ import annotations

import importlib
import logging

from ..registry import register
from ..types import CapabilityMode, Tier

_log = logging.getLogger("loom.verbs")

# The verbs, in build order. ``tokenize`` is no longer a standalone module — it is
# the ``representation="event-sequence"`` binding of the generic ``prepare`` verb
# (ARCHITECTURE §4/§8), registered explicitly below after ``prepare`` imports.
# ``ingest``/``baseline`` are unchanged; ``pretrain`` is the gated launch verb
# (ARCHITECTURE §10 step 6).
_VERB_MODULES = ("prepare", "ingest", "baseline", "pretrain")

for _name in _VERB_MODULES:
    try:
        importlib.import_module(f"{__name__}.{_name}")
    except Exception as exc:  # pragma: no cover - defensive wiring
        _log.warning("verb module %r failed to import and was skipped: %s", _name, exc)


# --- register the generic ``prepare`` verb + the ``tokenize`` bound alias -----
# Done here (not via @register decorators in prepare.py) so the import-order guard
# above governs landing, and so BOTH faces resolve:
#   * ``loom prepare``  / ``loom.prepare``  → prepare_fn  (generic; representation arg)
#   * ``loom tokenize`` / ``loom.tokenize`` → tokenize_fn (representation pinned to
#     event-sequence; envelope ``verb`` field pinned to "tokenize")
#
# WHY a separately-registered ``tokenize`` and not an argparse ``aliases=`` entry:
# the LOCKED dual-driver byte-identity invariant requires the JSON envelope's
# ``verb`` field to read "tokenize" (not "prepare") AND requires
# ``dispatch("loom.tokenize", …)`` to resolve a REGISTRY["tokenize"] entry. An
# argparse alias routes to the SAME Verb object (so the emitted ``verb`` field
# would read "prepare") and would NOT create a REGISTRY["tokenize"] key for the
# agent face. A distinct registration whose ``fn`` delegates to the same
# ``_prepare_impl`` with the representation pinned is the only wiring that keeps
# ``loom tokenize --json`` and ``dispatch("loom.tokenize")`` byte-identical to
# v0.1 on both faces. ``tokenize`` IS "prepare bound to event-sequence" — the
# binding lives in ``tokenize_fn``.
try:  # pragma: no cover - wiring guard, mirrors the per-module guard above
    from .prepare import (
        PREPARE_PARAMS,
        TOKENIZE_PARAMS,
        prepare_fn,
        tokenize_fn,
    )

    register(
        "prepare",
        summary="prepare a corpus from a data-representation adapter "
                "(contracts checked, <1s, no GPU)",
        tier=Tier.WORKSPACE_WRITE,
        capability_mode=CapabilityMode.NONE,
        params=PREPARE_PARAMS,
    )(prepare_fn)

    register(
        "tokenize",
        summary="compile a declarative tokenizer spec to a Corpus "
                "(C1/C2/C3 checked, <1s, no GPU)",
        tier=Tier.WORKSPACE_WRITE,
        capability_mode=CapabilityMode.NONE,
        params=TOKENIZE_PARAMS,
    )(tokenize_fn)
except Exception as exc:  # pragma: no cover - defensive wiring
    _log.warning("prepare/tokenize verbs failed to register and were skipped: %s", exc)
