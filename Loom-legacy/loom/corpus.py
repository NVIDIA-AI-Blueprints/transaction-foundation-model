"""The Loom corpus: append-only JSONL store of finished search nodes.

Every :class:`~loom.types.NodeRecord` a search provider emits (via the
``on_node`` callback) is appended here as one JSON object per line. The corpus
is the substrate a cross-tenant "moat" model may eventually train on, which is
why it enforces a single, generic IP boundary:

* A record's ``owned_by`` field is the IP owner. The sentinel ``"general"``
  means the record is *not* owned by any specific tenant and may be used across
  tenants. Any other value tags the record as tenant-owned.
* :meth:`Corpus.general` returns only the ``"general"`` records -- the slice a
  cross-tenant model is allowed to see. Tenant-owned records stay isolated.

The boundary is intentionally domain-neutral: the corpus knows nothing about any
customer or vertical, only the generic ``owned_by`` tag.

Secrets are never persisted: a :class:`NodeRecord` carries only code, terminal
output, metrics, and routing/ownership metadata -- never API keys or endpoints.
This module imports only the standard library + Loom core, so it is importable
in any environment.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from typing import Iterable, Iterator

from loom.config import LoomConfig
from loom.types import NodeRecord

# Sentinel ``owned_by`` value marking a record as cross-tenant / unowned.
GENERAL = "general"


class Corpus:
    """Append-only JSONL store of :class:`NodeRecord` objects.

    The corpus is constructed from a :class:`~loom.config.LoomConfig` so the
    controller can wire it with ``Corpus(config)`` symmetrically with the
    providers. The JSONL file lives at ``config.corpus_path``; its parent
    directory is created on demand at first write.

    Attributes:
        path: Filesystem path of the backing JSONL file.
    """

    def __init__(self, config: LoomConfig) -> None:
        """Create a corpus backed by ``config.corpus_path``.

        Args:
            config: The active Loom configuration. Only ``corpus_path`` is read;
                no secret material is touched.
        """
        self.path: str = config.corpus_path

    def record(self, node: NodeRecord) -> None:
        """Append a single :class:`NodeRecord` to the corpus as JSONL.

        This is the sink passed to a search provider as ``on_node``. The
        record's parent directory is created lazily, a default timestamp is
        stamped if the caller left ``ts`` unset, and the row is flushed so a
        crash mid-run still leaves a valid prefix of complete lines.

        Args:
            node: The finished-node record to persist. Must not contain secret
                material (the :class:`NodeRecord` schema has no field for any).
        """
        if not node.ts:
            node = dataclasses.replace(node, ts=time.time())

        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        line = json.dumps(dataclasses.asdict(node), ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()

    def all(self) -> list[NodeRecord]:
        """Read and return every record in the corpus.

        Returns:
            All persisted records, in file order. An empty list if the corpus
            file does not exist yet.
        """
        return list(self._iter_records())

    def general(self) -> list[NodeRecord]:
        """Return only the cross-tenant ("general") records.

        This is the IP boundary in action: a cross-tenant moat model may train
        on the result of :meth:`general` only. Records whose ``owned_by`` is any
        value other than ``"general"`` (i.e. owned by a specific tenant) are
        excluded.

        Returns:
            The records whose ``owned_by == "general"``, in file order.
        """
        return [rec for rec in self._iter_records() if rec.owned_by == GENERAL]

    def for_experiment(self, experiment_id: str) -> list[NodeRecord]:
        """Return every record belonging to ``experiment_id``, in file order.

        Args:
            experiment_id: The experiment to filter by.

        Returns:
            The matching records (across all owners/tenants).
        """
        return [
            rec for rec in self._iter_records() if rec.experiment_id == experiment_id
        ]

    def _iter_records(self) -> Iterator[NodeRecord]:
        """Yield records from the JSONL file, tolerating partial/blank lines.

        A missing file yields nothing. Blank lines are skipped, and only the
        fields recognized by :class:`NodeRecord` are passed through so the
        reader stays forward-compatible with extra columns.
        """
        if not os.path.isfile(self.path):
            return

        valid_fields = {f.name for f in dataclasses.fields(NodeRecord)}
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                clean = {k: v for k, v in raw.items() if k in valid_fields}
                yield NodeRecord(**clean)


def write_general_subset(records: Iterable[NodeRecord], dest_path: str) -> int:
    """Write the ``"general"`` subset of ``records`` to ``dest_path`` as JSONL.

    A small convenience for materializing the cross-tenant training slice to a
    standalone file (e.g. to hand to a moat-model training job) without exposing
    tenant-owned records.

    Args:
        records: Records to filter.
        dest_path: Destination JSONL path; its parent directory is created on
            demand.

    Returns:
        The number of records written.
    """
    parent = os.path.dirname(dest_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    written = 0
    with open(dest_path, "w", encoding="utf-8") as fh:
        for rec in records:
            if rec.owned_by != GENERAL:
                continue
            fh.write(json.dumps(dataclasses.asdict(rec), ensure_ascii=False) + "\n")
            written += 1
    return written


__all__ = ["Corpus", "GENERAL", "write_general_subset"]
