"""Tests for :mod:`loom.dataio` -- the Client-API data door.

Loom's input data is a **Metaflow data object** (an Artifact) referenced by
**pathspec** and read ONLY through the Metaflow **Client API**
(``metaflow.Run(pathspec).data.<artifact>``). The datastore behind it (local or
S3/minio) is an opaque detail Metaflow owns. This module pins two things:

* the materialize / schema helpers parse a Run's ``.data.train``/``.test``/
  ``.schema`` artifacts correctly. We mock a tiny object exposing that shape so
  the tests need no live Metaflow cluster (and the pure-Python suite stays
  green), patching ``metaflow.Run`` in for the resolve path.
* the **no-S3 principle**: ``loom/dataio.py`` contains no ``boto3``, no
  ``"s3://"`` literal, no ``mc``, and no raw-URI ``metaflow.S3`` usage. Storage
  config lives only in the Metaflow profile/env, never in Loom code.

The helpers lazy-import ``metaflow`` *inside* the functions, so this module
imports ``loom.dataio`` without Metaflow installed; the parsing tests inject a
fake ``metaflow`` module via :data:`sys.modules` rather than requiring a real
one. A live-cluster smoke is out of scope here (mirrors the workspace-contract
test's gating).
"""

from __future__ import annotations

import os
import sys
import types

import pytest

from loom import dataio


# ---------------------------------------------------------------------------
# Fakes: a tiny stand-in for a finished Metaflow Run exposing the Client-API
# shape ``Run(pathspec).data.{train,test,schema}``. We avoid importing pandas by
# using a minimal DataFrame-like object with a ``.to_csv(path, index=...)``.
# ---------------------------------------------------------------------------


class _FakeFrame:
    """A minimal DataFrame stand-in that can write itself as CSV.

    Mirrors just enough of the pandas surface ``materialize_dataset`` uses
    (``to_csv(path, index=False)``). ``columns`` is exposed so a future schema
    test could read it; here it is informational.
    """

    def __init__(self, header: list[str], rows: list[list[object]]) -> None:
        self.columns = header
        self._rows = rows

    def to_csv(self, path: str, index: bool = True) -> None:
        import csv

        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(self.columns)
            w.writerows(self._rows)


class _FakeData:
    """The ``Run.data`` artifact proxy: attribute access yields artifacts."""

    def __init__(self, **artifacts: object) -> None:
        for key, value in artifacts.items():
            setattr(self, key, value)


class _FakeRun:
    """A finished ``metaflow.Run`` stand-in exposing ``.data`` and ``.pathspec``."""

    def __init__(self, pathspec: str, data: _FakeData) -> None:
        self.pathspec = pathspec
        self.data = data


@pytest.fixture
def fake_metaflow(monkeypatch):
    """Install a fake ``metaflow`` module whose ``Run`` returns a captured run.

    Returns a setter ``set_run(fake_run)`` the test uses to choose what
    ``metaflow.Run(pathspec)`` resolves to. ``Run`` records the pathspec it was
    called with so tests can assert the Client-API call shape.
    """
    holder: dict[str, object] = {"run": None, "called_with": None}

    def _Run(pathspec: str):
        holder["called_with"] = pathspec
        return holder["run"]

    fake_module = types.ModuleType("metaflow")
    fake_module.Run = _Run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metaflow", fake_module)

    def set_run(run: object) -> None:
        holder["run"] = run

    set_run.holder = holder  # type: ignore[attr-defined]
    return set_run


# ---------------------------------------------------------------------------
# resolve_run: pathspec validation + Client-API call shape.
# ---------------------------------------------------------------------------


def test_resolve_run_accepts_short_form_pathspec(fake_metaflow) -> None:
    """``IngestDataset/<id>`` resolves via ``metaflow.Run`` (the Client API)."""
    run = _FakeRun("IngestDataset/123", _FakeData())
    fake_metaflow(run)

    resolved = dataio.resolve_run("IngestDataset/123")

    assert resolved is run
    # The exact string was handed to the Client API's Run(...) unchanged.
    assert fake_metaflow.holder["called_with"] == "IngestDataset/123"


def test_resolve_run_strips_whitespace(fake_metaflow) -> None:
    """Surrounding whitespace is trimmed before the Client-API call."""
    run = _FakeRun("IngestDataset/7", _FakeData())
    fake_metaflow(run)

    dataio.resolve_run("  IngestDataset/7  ")

    assert fake_metaflow.holder["called_with"] == "IngestDataset/7"


@pytest.mark.parametrize("bad", ["", "   ", "IngestDataset", "a/b/c", "/", "//"])
def test_resolve_run_rejects_non_run_pathspecs(fake_metaflow, bad) -> None:
    """Empty or non ``<flow>/<run_id>`` references raise ``ValueError``."""
    fake_metaflow(_FakeRun("X/1", _FakeData()))
    with pytest.raises(ValueError):
        dataio.resolve_run(bad)


# ---------------------------------------------------------------------------
# materialize_dataset: writes train.csv (+ test.csv) from .data artifacts.
# ---------------------------------------------------------------------------


def test_materialize_writes_train_and_test(fake_metaflow, tmp_path) -> None:
    """train + test artifacts are written as CSVs into dest_dir via the API."""
    train = _FakeFrame(["id", "x", "target"], [[0, 1.0, 1], [1, 2.0, 0]])
    test = _FakeFrame(["id", "x"], [[2, 3.0]])
    fake_metaflow(_FakeRun("IngestDataset/1", _FakeData(train=train, test=test)))

    dest = tmp_path / "input"
    dataio.materialize_dataset("IngestDataset/1", str(dest))

    train_csv = dest / "train.csv"
    test_csv = dest / "test.csv"
    assert train_csv.is_file()
    assert test_csv.is_file()
    assert train_csv.read_text(encoding="utf-8").splitlines()[0] == "id,x,target"
    assert test_csv.read_text(encoding="utf-8").splitlines()[0] == "id,x"
    # dest_dir is created if absent.
    assert os.path.isdir(dest)


def test_materialize_train_only_when_no_test(fake_metaflow, tmp_path) -> None:
    """A train-only data object writes train.csv and no test.csv."""
    train = _FakeFrame(["a", "b"], [[1, 2]])
    fake_metaflow(_FakeRun("IngestDataset/2", _FakeData(train=train, test=None)))

    dest = tmp_path / "input"
    dataio.materialize_dataset("IngestDataset/2", str(dest))

    assert (dest / "train.csv").is_file()
    assert not (dest / "test.csv").exists()


def test_materialize_requires_train_artifact(fake_metaflow, tmp_path) -> None:
    """A run with no ``train`` artifact is a clear ValueError, not a crash."""
    fake_metaflow(_FakeRun("IngestDataset/3", _FakeData(schema={"nrows": 0})))
    with pytest.raises(ValueError):
        dataio.materialize_dataset("IngestDataset/3", str(tmp_path / "input"))


# ---------------------------------------------------------------------------
# dataset_schema: reads the .schema artifact dict.
# ---------------------------------------------------------------------------


def test_dataset_schema_reads_schema_artifact(fake_metaflow) -> None:
    """The ``schema`` artifact dict is returned verbatim (as a dict)."""
    schema = {
        "columns": ["id", "x", "target"],
        "dtypes": {"id": "int64", "x": "float64", "target": "int64"},
        "nrows": 2,
        "target": "target",
    }
    fake_metaflow(_FakeRun("IngestDataset/4", _FakeData(schema=schema)))

    out = dataio.dataset_schema("IngestDataset/4")

    assert out == schema
    assert isinstance(out, dict)


def test_dataset_schema_empty_when_absent(fake_metaflow) -> None:
    """A run without a ``schema`` artifact yields an empty dict, not an error."""
    fake_metaflow(_FakeRun("IngestDataset/5", _FakeData()))
    assert dataio.dataset_schema("IngestDataset/5") == {}


# ---------------------------------------------------------------------------
# The load-bearing no-S3 principle, asserted on the source itself.
# ---------------------------------------------------------------------------


def test_dataio_never_touches_s3_directly() -> None:
    """``loom/dataio.py`` reads only via the Client API -- never S3 directly.

    Loom code must never talk to the datastore: no ``boto3``, no ``"s3://"``
    literal, no ``mc`` shell-out, and no raw-URI ``metaflow.S3`` usage. Storage
    config lives only in the Metaflow profile/env. We assert this on the module
    source so a regression that reaches for S3 fails loudly here.
    """
    source = _read_module_source(dataio)

    lowered = source.lower()
    assert "boto3" not in lowered
    assert "s3://" not in lowered
    # No `from metaflow import S3` / `metaflow.S3(` raw-URI datastore access.
    assert "import s3" not in lowered
    assert "metaflow.s3" not in lowered
    # The only data door is the Client API.
    assert "from metaflow import run" in lowered


def test_no_loom_module_imports_boto3_or_s3_uri() -> None:
    """Belt-and-suspenders: no Loom source file reaches for S3 directly.

    Scans every ``.py`` under ``loom/`` and ``flows/`` for the forbidden S3
    markers, documenting the principle across the whole codebase (the datastore
    is Metaflow's opaque concern). Comments/docstrings that *name* the principle
    use phrasings like "S3/minio" or "direct S3", which do not contain the
    literal markers below.
    """
    roots = [
        os.path.dirname(os.path.dirname(os.path.abspath(dataio.__file__))),
    ]
    repo_root = roots[0]
    offenders: list[str] = []
    for sub in ("loom", "flows"):
        base = os.path.join(repo_root, sub)
        for dirpath, _dirs, files in os.walk(base):
            if "__pycache__" in dirpath:
                continue
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                with open(path, "r", encoding="utf-8") as fh:
                    text = fh.read().lower()
                if "boto3" in text or "s3://" in text:
                    offenders.append(os.path.relpath(path, repo_root))
    assert not offenders, f"files reach for S3 directly: {offenders}"


def _read_module_source(module) -> str:
    """Return the on-disk source of ``module``."""
    with open(module.__file__, "r", encoding="utf-8") as fh:
        return fh.read()
