"""``--experiment`` threading helpers (DESIGN.md §2.3, §7.7).

``--experiment <id>`` is the join key that threads every run together; the
baseline lives under the same id (house rule #3) so a report always contains its
own control. This slice provides the resolution + persistence helpers; the
campaign-spec ``.loom`` parser (metric/goal/envelope) is wired here as a stub for
the ``pretrain`` agent to fill (it does not gate ``tokenize``/``ingest``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Experiment:
    """A resolved experiment record (the ``.loom`` campaign, when present).

    For this Phase-0 slice only ``id`` is load-bearing (it threads objects).
    ``goal``/``metric``/``baseline``/``envelope`` are parsed from the campaign
    file by ``pretrain`` later; ``metric`` absence drives ``REFUSED_NO_METRIC``
    for mutating GPU verbs (NOT for tokenize/ingest, which are non-gated here).
    """

    id: str
    goal: Optional[str] = None
    metric: Optional[str] = None
    baseline: Optional[str] = None
    dataset: Optional[str] = None
    envelope: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


def resolve_experiment(explicit: Optional[str]) -> Optional[str]:
    """Resolve the active experiment id: explicit ``--experiment`` flag wins,
    else ``$LOOM_EXPERIMENT``, else None."""
    if explicit:
        return explicit
    return os.environ.get("LOOM_EXPERIMENT") or None


def load_campaign(experiment_id: str, *, search_dir: Optional[str] = None) -> Optional[Experiment]:
    """Load ``experiments/<id>.loom`` if it exists (YAML). Returns None if there is
    no campaign file (the experiment is then id-only). TODO(pretrain): parse
    metric/goal/envelope and enforce ``REFUSED_NO_METRIC`` for launch verbs."""
    raise NotImplementedError
