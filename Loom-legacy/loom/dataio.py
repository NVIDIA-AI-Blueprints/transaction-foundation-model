"""Materialize Loom's input data object through the Metaflow Client API.

Loom's input data is a **Metaflow data object** -- a set of Metaflow *artifacts*
produced by :class:`flows.ingest_dataset.IngestDataset` and referenced by a
**pathspec** (e.g. ``"IngestDataset/123"``). This module is the *only* place
Loom reads that data, and it reads it through exactly one door: the **Metaflow
Client API** (``metaflow.Run(pathspec).data.<artifact>``).

The load-bearing invariant
---------------------------
The datastore backing those artifacts (local or object storage such as
S3/minio) is an **opaque implementation detail that Metaflow owns**. Loom is
agnostic to it and only ever sees artifacts:

* **No direct object-store access.** Loom code contains no object-storage SDK,
  no bucket-URI literals, and no raw-URI Metaflow datastore handle anywhere. Where
  the bytes physically live is configured solely in the Metaflow profile /
  environment, never in Loom code. (The test suite scans this module's source to
  keep it that way.)
* **Client API only.** We resolve a pathspec to a ``metaflow.Run`` and read its
  ``.data`` artifact proxy. Whether that proxy fetches from a local directory or
  a remote object store is Metaflow's concern, transparently handled by the
  profile the caller's environment selects (``METAFLOW_PROFILE`` / ``METAFLOW_*``).

``metaflow`` is imported *lazily inside the functions* so that importing this
module (and therefore ``loom``) never requires Metaflow to be installed -- only
calling a materialize/schema helper does.

Public surface
--------------
* :func:`resolve_run` -- pathspec/``"IngestDataset/<id>"`` -> a usable ``Run``.
* :func:`materialize_dataset` -- write the run's ``.train``/``.test`` artifacts to
  ``dest_dir`` as ``train.csv`` / ``test.csv``.
* :func:`dataset_schema` -- return the run's ``.schema`` artifact (a dict).
"""

from __future__ import annotations

import os
from typing import Any


def resolve_run(dataset_ref: str) -> Any:
    """Resolve a dataset reference to a Metaflow ``Run`` via the Client API.

    Accepts either a full pathspec or the ``"IngestDataset/<run_id>"`` short
    form (they are in fact the same shape -- ``<FlowName>/<run_id>`` -- so this
    helper normalizes/validates and hands the string to ``metaflow.Run``). The
    returned object is a live ``metaflow.Run`` whose ``.data`` proxy reads the
    ingested artifacts through whatever datastore the active Metaflow profile
    points at. **Loom never touches that datastore directly.**

    Args:
        dataset_ref: A Metaflow pathspec identifying the ingested data object,
            e.g. ``"IngestDataset/123"``. A leading/trailing whitespace is
            stripped; an empty value is rejected.

    Returns:
        A ``metaflow.Run`` for ``dataset_ref``.

    Raises:
        ValueError: If ``dataset_ref`` is empty or not a ``<flow>/<run_id>``
            pathspec.
        ImportError: If the ``metaflow`` package is not installed.
    """
    # Lazy import: keeps ``import loom.dataio`` (and ``import loom``) working
    # without Metaflow installed. The Client API is the ONLY data door.
    from metaflow import Run

    ref = (dataset_ref or "").strip()
    if not ref:
        raise ValueError(
            "dataset_ref is empty; expected a Metaflow pathspec like "
            "'IngestDataset/123' (run `loom ingest` to produce one)."
        )

    # A run pathspec is exactly two non-empty components: <FlowName>/<run_id>.
    # (Step/task pathspecs have more; we deliberately target the Run level so
    # ``.data`` resolves to the flow's end-step artifacts.)
    parts = [p for p in ref.split("/") if p]
    if len(parts) != 2:
        raise ValueError(
            f"dataset_ref {ref!r} is not a run pathspec; expected "
            "'<FlowName>/<run_id>' (e.g. 'IngestDataset/123')."
        )

    return Run(ref)


def materialize_dataset(dataset_ref: str, dest_dir: str) -> None:
    """Materialize an ingested data object to ``dest_dir`` via the Client API.

    Reads the ``train`` (and, when present, ``test``) artifacts from the
    Metaflow run identified by ``dataset_ref`` and writes them as ``train.csv``
    (and ``test.csv``) into ``dest_dir``, which is created if needed. The data is
    fetched purely through ``metaflow.Run(dataset_ref).data`` -- whether the
    bytes come from a local datastore or S3/minio is Metaflow's concern, opaque
    to Loom.

    This is the one boundary at which an ingested data object becomes plain CSVs
    in a host-local workspace (so a candidate solution and AIDE's data-preview
    can read ``./input/train.csv`` exactly as they do under the local provider).

    Args:
        dataset_ref: The Metaflow pathspec of the ingested data object (see
            :func:`resolve_run`).
        dest_dir: Local directory to write the dataset CSVs into. Created if it
            does not exist.

    Raises:
        ValueError: If ``dataset_ref`` is malformed (see :func:`resolve_run`) or
            the run exposes no ``train`` artifact.
        ImportError: If ``metaflow`` is not installed.
    """
    run = resolve_run(dataset_ref)
    data = run.data
    if data is None:  # pragma: no cover - a finished IngestDataset always has data
        raise ValueError(
            f"dataset_ref {dataset_ref!r} resolved to a run with no data; "
            "ensure `loom ingest` completed successfully."
        )

    os.makedirs(dest_dir, exist_ok=True)

    # ``train`` is required; ``test`` is optional (a single-CSV ingest may have
    # split it, but a caller could also ingest train-only). We read the artifacts
    # by name through the Client-API data proxy -- never by reaching into a
    # datastore. The artifacts are pandas DataFrames written by IngestDataset.
    train = getattr(data, "train", None)
    if train is None:
        raise ValueError(
            f"dataset_ref {dataset_ref!r} has no 'train' artifact; was it "
            "produced by `loom ingest` (IngestDataset)?"
        )
    train.to_csv(os.path.join(dest_dir, "train.csv"), index=False)

    test = getattr(data, "test", None)
    if test is not None:
        test.to_csv(os.path.join(dest_dir, "test.csv"), index=False)


def dataset_schema(dataset_ref: str) -> dict:
    """Return the ingested data object's ``schema`` artifact via the Client API.

    Reads ``metaflow.Run(dataset_ref).data.schema`` -- the columns/dtypes/nrows
    (and ``target`` when present) dict that :class:`IngestDataset` recorded at
    ingest time. As everywhere in this module, the read goes through the Client
    API only; the underlying datastore is never touched directly.

    Args:
        dataset_ref: The Metaflow pathspec of the ingested data object.

    Returns:
        The schema dict (empty ``{}`` if the run carries no ``schema`` artifact).

    Raises:
        ValueError: If ``dataset_ref`` is malformed (see :func:`resolve_run`).
        ImportError: If ``metaflow`` is not installed.
    """
    run = resolve_run(dataset_ref)
    data = run.data
    if data is None:  # pragma: no cover - a finished IngestDataset always has data
        return {}
    schema = getattr(data, "schema", None)
    return dict(schema) if schema else {}


__all__ = ["resolve_run", "materialize_dataset", "dataset_schema"]
