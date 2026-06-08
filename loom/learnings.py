"""The Loom learnings flywheel: append-only JSONL of command-level rollouts.

Where the :mod:`loom.corpus` captures one :class:`~loom.types.NodeRecord` per
*search node* (the fine-grained substrate), the learnings flywheel captures one
:class:`LearningRecord` per *command/run* -- the command-level rollup that sits
on top of the corpus. One ``run_loom`` call appends exactly one learning row.

This is the **moat fuel** (`design-spec.md` §5, `feedback-loop-tech.md` §4b):
each row is shaped to be a **SkillOpt rollout input** -- a trajectory of one
attempt at a task plus its outcome -- so the feedback/skill-optimization
provider can later consume ``learnings/rollouts.jsonl`` directly to optimize each
``/loom-*`` ``SKILL.md`` against Loom's own held-out metric, with no extra
instrumentation. The fields mirror that schema: a task spec (``data_ref`` /
``goal`` / ``metric``), the ``inputs`` that drove the run, the ``outcome``
(``best_metric``, ``submission_ok``), the ``artifacts`` produced (as a Metaflow
pathspec / journal / tree -- never raw bytes), a ``success`` bool, the ``model``,
routing/ownership metadata, and a free-text ``reflection`` slot.

This module enforces the same single, generic IP boundary as the corpus:

* A record's ``owned_by`` field is the IP owner. The sentinel ``"general"``
  means the record is *not* owned by any specific tenant and may be used across
  tenants. Any other value tags the record as tenant-owned.
* :meth:`Learnings.general` returns only the ``"general"`` records -- the slice
  a cross-tenant skill-optimizer may train on. Tenant-owned rows stay isolated.

Secrets are never persisted: a :class:`LearningRecord` carries only task spec,
metrics, artifact *references* (pathspecs/paths), and routing/ownership metadata
-- never API keys or endpoints. This module imports only the standard library +
Loom core, so it is importable in any environment.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from loom.config import LoomConfig

# Sentinel ``owned_by`` value marking a record as cross-tenant / unowned. Mirrors
# :data:`loom.corpus.GENERAL` so the two stores share one IP-boundary vocabulary.
GENERAL = "general"


@dataclass
class TaskSpec:
    """The task a learning row was rolled out against (the SkillOpt task).

    A compact, JSON-friendly projection of :class:`~loom.types.Task`: the parts
    that define *what* was attempted and *how it was scored*, grouped so the
    rollout's supervision target is self-contained.

    Attributes:
        data_ref: Reference to the input data object. The Metaflow ``dataset_ref``
            pathspec when present, else the local ``data_dir`` -- whichever the
            active execution provider consumed. Never raw data, only the ref.
        goal: Natural-language description of what the solution should achieve.
        metric: Natural-language description of how a solution is evaluated (the
            validation metric the search was asked to optimize).
        experiment_id: Stable identifier grouping runs/attempts for the same task
            (the key a contrastive reflection step joins attempts on).
    """

    data_ref: str | None
    goal: str
    metric: str
    experiment_id: str


@dataclass
class Outcome:
    """The scalar outcome of a rollout (the SkillOpt score signal).

    Attributes:
        best_metric: Validation metric of the best solution found, or ``None`` if
            no viable solution was produced. The continuous ("soft") score.
        submission_ok: Whether the run produced a usable/submittable solution.
            The boolean ("hard") score; ``True`` when a best solution exists.
        node_count: Number of search nodes explored in this rollout.
    """

    best_metric: float | None
    submission_ok: bool
    node_count: int = 0


@dataclass
class LearningRecord:
    """One command/run rollout, persisted by :class:`Learnings`.

    Appended (as JSONL) once per ``run_loom`` call. Shaped as a SkillOpt rollout
    input: a single attempt at :attr:`task` under :attr:`inputs`, with its
    :attr:`outcome`, the :attr:`artifacts` it produced, and a free-text
    :attr:`reflection`. The ``owned_by`` field is the IP boundary: rows owned by
    a specific tenant are tagged and excluded from the cross-tenant "general" set.

    Attributes:
        ts: Epoch timestamp (seconds) when the record was created. Stamped on
            write if the caller leaves it ``0.0``.
        command: The ``/loom-*`` command / skill this rollout exercised (e.g.
            ``"loom-optimize"``). The trajectory's owning SKILL.md.
        task: The task spec rolled out against (see :class:`TaskSpec`).
        inputs: The args/knobs that drove the run (budget dials, provider
            selection, ...) -- the controllable part of the rollout.
        outcome: The scalar outcome / score signal (see :class:`Outcome`).
        artifacts: Pathspec(s) / paths of artifacts the run produced -- a
            Metaflow run pathspec, journal path, tree/card path. References only,
            never inlined bytes (large output spills to an Artifact, per spec §2).
        success: Whether the rollout is considered a success (mirrors
            ``outcome.submission_ok``; kept as a top-level flag for cheap
            filtering by the optimizer).
        model: Identifier of the model that generated the solution code.
        tenant: Tenant the run belongs to (multi-tenant boundary).
        owned_by: IP owner of the record; ``"general"`` means it may be used by a
            cross-tenant skill-optimizer, any other value tags it as
            tenant-owned.
        reflection: Free-text slot for a natural-language reflection on the run
            (judge summary, what worked / failed). Optional; never the sole gate
            signal downstream.
    """

    command: str
    task: TaskSpec
    inputs: dict[str, Any]
    outcome: Outcome
    artifacts: list[str] = field(default_factory=list)
    success: bool = False
    model: str | None = None
    tenant: str = "default"
    owned_by: str = GENERAL
    reflection: str | None = None
    ts: float = field(default=0.0)


class Learnings:
    """Append-only JSONL store of :class:`LearningRecord` objects.

    Constructed from a :class:`~loom.config.LoomConfig` so the controller can
    wire it with ``Learnings(config)`` symmetrically with the corpus and the
    providers. The JSONL file lives at ``config.learnings_path`` (anchored
    absolute at config load); its parent directory is created on demand at first
    write.

    Attributes:
        path: Filesystem path of the backing JSONL file.
    """

    def __init__(self, config: LoomConfig) -> None:
        """Create a learnings store backed by ``config.learnings_path``.

        Args:
            config: The active Loom configuration. Only ``learnings_path`` is
                read; no secret material is touched.
        """
        self.path: str = config.learnings_path

    def record(self, learning: LearningRecord) -> None:
        """Append a single :class:`LearningRecord` to the flywheel as JSONL.

        The record's parent directory is created lazily, a default timestamp is
        stamped if the caller left ``ts`` unset, and the row is flushed so a
        crash mid-run still leaves a valid prefix of complete lines.

        Args:
            learning: The command-level rollout record to persist. Must not
                contain secret material (the schema has no field for any).
        """
        if not learning.ts:
            learning = dataclasses.replace(learning, ts=time.time())

        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        line = json.dumps(dataclasses.asdict(learning), ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()

    def all(self) -> list[LearningRecord]:
        """Read and return every learning record, in file order.

        Returns:
            All persisted records, in file order. An empty list if the backing
            file does not exist yet.
        """
        return list(self._iter_records())

    def general(self) -> list[LearningRecord]:
        """Return only the cross-tenant ("general") records.

        The IP boundary in action, mirroring :meth:`loom.corpus.Corpus.general`:
        a cross-tenant skill-optimizer may train on the result of this method
        only. Records whose ``owned_by`` is any value other than ``"general"``
        (i.e. owned by a specific tenant) are excluded.

        Returns:
            The records whose ``owned_by == "general"``, in file order.
        """
        return [rec for rec in self._iter_records() if rec.owned_by == GENERAL]

    def _iter_records(self) -> Iterator[LearningRecord]:
        """Yield records from the JSONL file, tolerating partial/blank lines.

        A missing file yields nothing. Blank lines are skipped, and only the
        fields recognized by :class:`LearningRecord` are passed through (with the
        nested ``task``/``outcome`` rehydrated into their dataclasses) so the
        reader stays forward-compatible with extra columns.
        """
        if not os.path.isfile(self.path):
            return

        valid_fields = {f.name for f in dataclasses.fields(LearningRecord)}
        task_fields = {f.name for f in dataclasses.fields(TaskSpec)}
        outcome_fields = {f.name for f in dataclasses.fields(Outcome)}

        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                clean = {k: v for k, v in raw.items() if k in valid_fields}
                if isinstance(clean.get("task"), dict):
                    clean["task"] = TaskSpec(
                        **{k: v for k, v in clean["task"].items() if k in task_fields}
                    )
                if isinstance(clean.get("outcome"), dict):
                    clean["outcome"] = Outcome(
                        **{
                            k: v
                            for k, v in clean["outcome"].items()
                            if k in outcome_fields
                        }
                    )
                yield LearningRecord(**clean)


__all__ = ["Learnings", "LearningRecord", "TaskSpec", "Outcome", "GENERAL"]
