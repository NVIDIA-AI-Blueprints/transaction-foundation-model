"""Loom core types — the dual-driver result envelope and its parts.

These types are the *contract* every verb implementation and both driver faces
(human CLI + agent tool) share. The envelope rendered by ``loom <verb> --json``
is byte-identical to the envelope returned by the agent tool ``loom.<verb>(...)``
(DESIGN.md §2.1). LOCKED: implementers code against these names verbatim.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums — string-valued so they serialize cleanly into the JSON envelope.
# ---------------------------------------------------------------------------


class Verdict(str, Enum):
    """The machine-checkable outcome of a verb (DESIGN.md §0, §7)."""

    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"
    INCOMPLETE = "INCOMPLETE"  # e.g. a STOPPED_AT_BUDGET partial checkpoint (§6)


class Status(str, Enum):
    """Top-level call status. ``PLAN`` is the cheap/launch pre-commit state that
    carries a ``confirm_token`` for the agent's second call (§5.3); the
    ``REFUSED_*`` family are structural refusals surfaced as named diagnostics."""

    OK = "OK"
    PLAN = "PLAN"
    REFUSED_NO_METRIC = "REFUSED_NO_METRIC"
    REFUSED_NO_BASELINE = "REFUSED_NO_BASELINE"
    REFUSED_NO_GPU_TARGET = "REFUSED_NO_GPU_TARGET"
    REFUSED_AGENT_CANNOT_LAUNCH = "REFUSED_AGENT_CANNOT_LAUNCH"
    REFUSED_NONINTERACTIVE_LAUNCH = "REFUSED_NONINTERACTIVE_LAUNCH"
    REFUSED_SPEND_CAP = "REFUSED_SPEND_CAP"
    REFUSED_STALE = "REFUSED_STALE"
    REFUSED_CONTRACT = "REFUSED_CONTRACT"  # a C1/C2/C3/C6 violation refused the write
    FAIL = "FAIL"


class Tier(str, Enum):
    """A property of the verb, not a flag (DESIGN.md §2.2/§4). It cannot be
    bypassed; gating reads it off the verb declaration."""

    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    EXPENSIVE = "expensive"
    IRREVERSIBLE = "irreversible"


class CapabilityMode(str, Enum):
    """How a verb participates in the search/launch machinery (DESIGN.md §2.2,
    §4.5). ``searchable`` work can be tree-searched by the AIDE brain;
    ``launch-and-track`` work structurally cannot."""

    NONE = "none"
    SEARCHABLE = "searchable"
    LAUNCH_AND_TRACK = "launch-and-track"


class Severity(str, Enum):
    """Severity of a single contract diagnostic."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataObjectRef:
    """A pathspec reference — Loom's "everything is a file" handle (§2.4).

    Rendered/parsed as ``"Type/<n>"`` (e.g. ``"Corpus/204"``).
    """

    kind: str
    n: int

    @property
    def pathspec(self) -> str:
        return f"{self.kind}/{self.n}"

    def __str__(self) -> str:  # so f-strings render the pathspec
        return self.pathspec

    @classmethod
    def parse(cls, pathspec: str) -> "DataObjectRef":
        kind, _, n = pathspec.partition("/")
        if not _ or not n.isdigit():
            raise ValueError(f"not a valid pathspec 'Type/<n>': {pathspec!r}")
        return cls(kind=kind, n=int(n))


@dataclass
class CostPlan:
    """A *derived* cost plan (DESIGN.md §4.2). PLACEHOLDER for this Phase-0 slice
    (no GPU): the fields the gating model will use are present and carried on the
    envelope, but ``tokenize``/``ingest``/``baseline`` are cheap CPU verbs that
    leave them ``None``/zero. ``derived`` distinguishes a computed estimate from a
    label — it must never be a hardcoded number once a GPU verb populates it.
    """

    derived: bool = False
    usd: Optional[float] = None
    confidence: Optional[str] = None  # "LOW" | "MEDIUM" | "HIGH"
    tokens: Optional[int] = None
    params: Optional[int] = None
    seq_len: Optional[int] = None
    gpu_target: Optional[str] = None
    # The binding envelope a human approves (§4.3); None until a launch verb fills it.
    envelope: Optional[dict[str, Any]] = None
    inputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Diagnostic:
    """One named contract finding — the named-diff card, not a stack trace
    (DESIGN.md §7.2). Renders as a terminal card (human) and an element of the
    ``diagnostics[]`` array (agent); the two carry identical data."""

    contract: str  # "C1" | "C2" | "C3" | "C6" | "EDA" | ...
    severity: Severity
    message: str
    fix: Optional[str] = None  # the offered one-line fix
    data: dict[str, Any] = field(default_factory=dict)  # structured detail (ids, deltas)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "severity": self.severity.value,
            "message": self.message,
            "fix": self.fix,
            "data": self.data,
        }


# ---------------------------------------------------------------------------
# The dual-driver result envelope — ONE dataclass, two renderers.
# ---------------------------------------------------------------------------


@dataclass
class VerbResult:
    """The single result envelope shared by both driver faces (DESIGN.md §2.1).

    The CLI renders this as a pretty card; ``--json`` prints ``to_json()`` raw;
    the agent tool returns ``to_json()``. The ``--json`` text is byte-identical to
    the tool result. LOCKED shape — do not add fields without threading them
    through both faces.
    """

    verb: str
    status: Status
    verdict: Verdict
    tier: Tier
    capability_mode: CapabilityMode
    summary: str = ""
    outputs: list[DataObjectRef] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    experiment: Optional[str] = None
    cost_plan: Optional[CostPlan] = None
    # Single-use, plan-hash-scoped, expiring token for the agent's confirm
    # round-trip on a PLAN (§5.3); None unless status == PLAN.
    confirm_token: Optional[str] = None

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form of the envelope (keys are stable, order is stable)."""
        return {
            "verb": self.verb,
            "status": self.status.value,
            "verdict": self.verdict.value,
            "tier": self.tier.value,
            "capability_mode": self.capability_mode.value,
            "summary": self.summary,
            "outputs": [o.pathspec for o in self.outputs],
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "data": self.data,
            "experiment": self.experiment,
            "cost_plan": self.cost_plan.to_dict() if self.cost_plan else None,
            "confirm_token": self.confirm_token,
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """The machine envelope. ``loom <verb> --json`` and the agent tool emit
        exactly this string (default: compact, sort-stable keys via to_dict)."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @property
    def exit_code(self) -> int:
        """Process exit code for the CLI face (read-only/OK/PLAN succeed; a FAIL
        verdict or a refusal is non-zero)."""
        if self.verdict is Verdict.FAIL or self.status is Status.FAIL:
            return 1
        if self.status.value.startswith("REFUSED_"):
            return 2
        return 0
