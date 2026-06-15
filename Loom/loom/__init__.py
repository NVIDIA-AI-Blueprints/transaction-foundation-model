"""Loom — an agent harness for training SOTA foundation models.

A small set of sharp, typed verbs you *compile before you spend*, driven
identically by a human at a terminal (``loom <verb> …``) and a Claude/Codex agent
(``loom.<verb>(…)``), where contract violations surface as named diffs caught for
free. See ``DESIGN.md`` for the authoritative product + UX spec.

Phase-0 slice (ZERO GPU): the ``tokenize`` contract compiler, ``ingest`` +
EDA leakage gate, ``baseline``, the typed-contract narrow waist, a local
content-addressed object store, and ``--experiment`` threading.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Public re-exports — the names code and tests import from the top level.
from .types import (  # noqa: F401
    CapabilityMode,
    CostPlan,
    DataObjectRef,
    Diagnostic,
    Severity,
    Status,
    Tier,
    Verdict,
    VerbResult,
)
from .registry import REGISTRY, Verb, VerbContext, register  # noqa: F401

# Importing the verbs package self-registers every verb into REGISTRY so both
# faces (CLI + tools) see them. Guarded so the package still imports if a verb
# module is mid-build (a missing/broken verb is logged, not fatal).
from . import verbs  # noqa: F401,E402

__all__ = [
    "__version__",
    "CapabilityMode",
    "CostPlan",
    "DataObjectRef",
    "Diagnostic",
    "Severity",
    "Status",
    "Tier",
    "Verdict",
    "VerbResult",
    "REGISTRY",
    "Verb",
    "VerbContext",
    "register",
]
