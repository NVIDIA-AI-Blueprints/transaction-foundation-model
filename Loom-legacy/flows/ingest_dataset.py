"""Loom's dataset-ingest Metaflow flow -- the one external->Metaflow boundary.

This module defines :class:`IngestDataset`, the single point at which data from
*outside* Metaflow (a local directory or CSV the user points at) crosses into a
**Metaflow data object**. Running it once turns a plain dataset into Metaflow
artifacts (``train``/``test``/``schema``) addressable by **pathspec** (e.g.
``"IngestDataset/123"``). That pathspec is the reference every downstream Loom
step uses: the data is thereafter read *only* through the Metaflow Client API
(see :mod:`loom.dataio`), and the datastore backing it (local or S3/minio) is an
opaque detail Metaflow owns.

Why a flow (and not just a copy): ingesting *through* Metaflow is what makes the
dataset a versioned, content-addressed, profile-backed data object -- the same
object whether Loom runs locally now or in a cluster later. Loom never writes to
S3 itself; it hands the source to Metaflow and lets the configured datastore
persist the artifacts.

Flow shape::

    start --> end

* ``start`` -- load the ``source`` with pandas. A *directory* is expected to
               contain ``train.csv`` (plus optional ``test.csv`` /
               ``sample_submission.csv``); a single ``.csv`` file is
               train/test-split. Stores ``self.train`` / ``self.test``
               (DataFrames), ``self.schema`` (a dict of columns/dtypes/nrows and
               a detected ``target`` if present), and ``self.dataset_name``.
* ``end``   -- no-op; the artifacts on ``self`` are the data object.

The run is tagged ``loom_dataset:<name>`` so the Client API can find it by name.

Only standard Metaflow APIs are used (``FlowSpec``, ``@step``, ``Parameter``).
``pandas`` is imported *inside* the step so the flow file parses even where
pandas is not importable until the Runner subprocess sets up the environment.
"""

from __future__ import annotations

from metaflow import FlowSpec, Parameter, step

#: Fraction of a single-CSV source held out as the test split when the source is
#: one file (no separate test.csv). A deterministic tail split keeps ingest
#: reproducible without importing a splitter dependency.
_DEFAULT_TEST_FRACTION = 0.2


class IngestDataset(FlowSpec):
    """Ingest a local dataset once into a Metaflow data object.

    The produced run's pathspec (``"IngestDataset/<run_id>"``) is the
    ``dataset_ref`` passed to ``loom run --dataset <pathspec>``. The flow is run
    via ``metaflow.Runner`` by ``loom ingest`` (see :mod:`loom.cli`).
    """

    #: Local directory or CSV file to ingest ONCE. After ingest, the data lives
    #: as Metaflow artifacts and is referenced by pathspec; this source path is
    #: never read again by Loom.
    source = Parameter(
        "source",
        required=True,
        type=str,
        help="Local dir (with train.csv[, test.csv]) or a single .csv to ingest.",
    )

    #: Human-friendly dataset name; also applied as a ``loom_dataset:<name>`` tag.
    #: Defaults to the source's basename when empty. The parameter *id* is
    #: ``dataset_name`` (the ``--dataset_name`` flag) rather than ``name`` because
    #: ``name`` is reserved by Metaflow's CLI options; the class attribute is
    #: ``name_param`` (read as ``self.name_param``) so the resolved value can be
    #: persisted under the distinct ``self.dataset_name`` artifact -- Metaflow
    #: forbids overwriting a Parameter attribute.
    name_param = Parameter(
        "dataset_name",
        default="",
        type=str,
        help="Optional dataset name (also tagged as loom_dataset:<name>).",
    )

    @step
    def start(self) -> None:
        """Load the source into ``train``/``test`` DataFrames + a ``schema`` dict.

        Directory source: read ``train.csv`` (required) and ``test.csv`` if
        present. Single-CSV source: read it and hold out a deterministic tail as
        the test split. Records a lightweight schema (columns, dtypes, row count,
        and a detected ``target`` column when one is obvious).
        """
        import os

        import pandas as pd

        src = (self.source or "").strip()
        if not src or not os.path.exists(src):
            raise ValueError(f"source path does not exist: {src!r}")

        # Resolve a dataset name: explicit (the ``dataset_name`` Parameter, read
        # as ``self.name_param``), else the source basename (sans ext). The
        # resolved value is stored as the distinct ``dataset_name`` artifact.
        requested = (self.name_param or "").strip()
        base = os.path.basename(os.path.normpath(src))
        if os.path.isfile(src):
            base = os.path.splitext(base)[0]
        self.dataset_name = requested or base

        test_df = None
        if os.path.isdir(src):
            train_path = os.path.join(src, "train.csv")
            if not os.path.isfile(train_path):
                raise ValueError(
                    f"source directory {src!r} has no train.csv; a directory "
                    "ingest expects train.csv (and optionally test.csv)."
                )
            train_df = pd.read_csv(train_path)
            test_path = os.path.join(src, "test.csv")
            if os.path.isfile(test_path):
                test_df = pd.read_csv(test_path)
        else:
            # Single CSV: deterministic tail split into train/test.
            full = pd.read_csv(src)
            n_test = int(len(full) * _DEFAULT_TEST_FRACTION)
            if n_test > 0:
                train_df = full.iloc[:-n_test].reset_index(drop=True)
                test_df = full.iloc[-n_test:].reset_index(drop=True)
            else:
                train_df = full.reset_index(drop=True)

        self.train = train_df
        self.test = test_df
        self.schema = self._build_schema(train_df, test_df)

        self.next(self.end)

    @staticmethod
    def _build_schema(train_df: object, test_df: object) -> dict:
        """Build a lightweight schema dict for the ingested data.

        Records the train columns and their (stringified) dtypes, the train row
        count, and a detected ``target`` column. The target is inferred as a
        column present in ``train`` but absent from ``test`` (the usual
        train/test asymmetry); when no test split exists, a column literally
        named ``target`` is used if present, else ``None``.

        Args:
            train_df: The train DataFrame.
            test_df: The test DataFrame, or ``None``.

        Returns:
            A JSON-friendly dict: ``columns``/``dtypes``/``nrows``/``target``.
        """
        columns = [str(c) for c in train_df.columns]
        dtypes = {str(c): str(train_df[c].dtype) for c in train_df.columns}

        target = None
        if test_df is not None:
            test_cols = {str(c) for c in test_df.columns}
            only_in_train = [c for c in columns if c not in test_cols]
            if len(only_in_train) == 1:
                target = only_in_train[0]
        if target is None and "target" in columns:
            target = "target"

        return {
            "columns": columns,
            "dtypes": dtypes,
            "nrows": int(len(train_df)),
            "target": target,
        }

    @step
    def end(self) -> None:
        """Finalize the data object.

        The artifacts (``train`` / ``test`` / ``schema`` / ``dataset_name``) are
        already persisted on ``self`` by Metaflow, so they are exposed on
        ``Run.data`` and addressable by this run's pathspec. Nothing else to do.
        """
        pass


if __name__ == "__main__":
    IngestDataset()
