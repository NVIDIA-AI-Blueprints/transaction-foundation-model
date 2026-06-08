"""Metaflow flows used by Loom.

This package holds the static Metaflow ``FlowSpec`` classes Loom drives, each run
as a subprocess via ``metaflow.Runner``:

* :class:`flows.ingest_dataset.IngestDataset` -- the **one external->Metaflow
  boundary**. ``loom ingest`` runs it once to turn a local dataset into a
  Metaflow **data object** (artifacts addressable by pathspec). Thereafter Loom
  reads that data only through the Metaflow Client API (see :mod:`loom.dataio`),
  and the datastore (local or S3/minio) is an opaque detail Metaflow owns.
* :class:`flows.eval_candidate.EvalCandidate` -- the per-candidate evaluation
  flow run by the Metaflow execution provider (see
  :mod:`loom.providers.metaflow_exec`).
* :class:`flows.eda.EdaFlow` -- the read-only EDA/profiling flow the ``loom eda``
  lifecycle command runs through the MLOps interface's ``run_flow`` seam.

Design invariant (mirrors the repo CLAUDE.md): a *candidate* solution is never
turned into a new flow. ``EvalCandidate`` is ONE static flow class, and each
candidate enters it as **data** (an ``IncludeFile`` parameter), so the flow
definition is stable across every evaluation.

Importing this package must not require Metaflow to be installed: each flow
module imports ``metaflow`` at its top level (a flow file genuinely needs it), so
they are imported lazily by the provider/CLI rather than eagerly here. We
therefore keep this package ``__init__`` import-light and expose only the path
helpers below.
"""

from __future__ import annotations

import os

_FLOWS_DIR = os.path.dirname(os.path.abspath(__file__))

#: Absolute path to the static evaluation flow file. The Metaflow ``Runner``
#: takes a flow *file path*, so the provider resolves the flow this way rather
#: than importing the flow class (which would require Metaflow at import time).
EVAL_CANDIDATE_FLOW_PATH: str = os.path.join(_FLOWS_DIR, "eval_candidate.py")

#: Absolute path to the dataset-ingest flow file. ``loom ingest`` runs this via
#: ``metaflow.Runner`` to produce the data object referenced by pathspec.
INGEST_DATASET_FLOW_PATH: str = os.path.join(_FLOWS_DIR, "ingest_dataset.py")

#: Absolute path to the read-only EDA/profiling flow file. ``loom eda`` runs this
#: through the MLOps interface's ``run_flow`` seam to profile a data object.
EDA_FLOW_PATH: str = os.path.join(_FLOWS_DIR, "eda.py")

__all__ = [
    "EVAL_CANDIDATE_FLOW_PATH",
    "INGEST_DATASET_FLOW_PATH",
    "EDA_FLOW_PATH",
]
