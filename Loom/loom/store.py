"""A LOCAL content-addressed object store (DESIGN.md §7.1, §6, §9). NO Metaflow.

Every verb output is an immutable, versioned :class:`DataObject` addressed by a
pathspec ``Type/<n>``. ``Type/<n>`` ids are minted by an **atomic integer
counter** (per-kind), the local stand-in for Metaflow's run-id allocation (§9 —
the v0.2 execution adapter replaces this). Objects are content-addressed by a
``content_id`` (``source_fingerprint + spec_hash``) so re-running ``ingest`` /
``tokenize`` with the same source+spec returns the EXISTING object rather than
forking a twin (§6 idempotency).

The store lives under a workspace dir (default ``.loom/objects``). TODO(v0.2):
swap this for the Metaflow metadata/datastore adapter; the public API
(``put``/``get``/``new_ref``) is the seam.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .types import DataObjectRef, Status, Verdict


@dataclass
class DataObject:
    """An immutable, lineage-carrying object in the store (DESIGN.md §7.1).

    Attributes
    ----------
    ref : its pathspec handle ``Type/<n>``.
    kind : object type (``"IngestDataset"`` | ``"Corpus"`` | ``"Baseline"`` | ...).
    content_id : the content address (``source_fingerprint + spec_hash``) used for
        idempotent dedupe (§6).
    parents : parent pathspecs (the lineage edges).
    producer_verb / producer_args : the exact verb + args that made it (replay).
    signatures : contract signatures it satisfies (e.g. ``{"vocab_hash": ...,
        "vocab_size": ..., "tokens_per_txn": ...}``) — C1 travels with the object.
    verdict / status : its VERDICT and lifecycle status (§6).
    experiment : the join key (``--experiment``).
    payload_path : path to the heavy payload on disk (corpus lines, vocab json),
        or ``None`` for metadata-only objects.
    cost_actuals / envelope : carried on GPU objects (placeholder this slice).
    created_at : ISO timestamp.
    """

    ref: DataObjectRef
    kind: str
    content_id: str
    parents: list[str] = field(default_factory=list)
    producer_verb: str = ""
    producer_args: dict[str, Any] = field(default_factory=dict)
    signatures: dict[str, Any] = field(default_factory=dict)
    verdict: Verdict = Verdict.PASS
    status: Status = Status.OK
    experiment: Optional[str] = None
    payload_path: Optional[str] = None
    cost_actuals: Optional[dict[str, Any]] = None
    envelope: Optional[dict[str, Any]] = None
    created_at: Optional[str] = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def pathspec(self) -> str:
        return self.ref.pathspec

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ref"] = self.ref.pathspec
        d["verdict"] = self.verdict.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataObject":
        d = dict(d)
        d["ref"] = DataObjectRef.parse(d["ref"])
        d["verdict"] = Verdict(d.get("verdict", "PASS"))
        d["status"] = Status(d.get("status", "OK"))
        return cls(**d)


class ObjectStore:
    """A local content-addressed object store rooted at a workspace dir.

    Layout::

        <root>/.loom/objects/<Kind>/<n>/object.json     # metadata
        <root>/.loom/objects/<Kind>/<n>/payload/...     # heavy payload
        <root>/.loom/objects/_counters.json             # atomic id counters
        <root>/.loom/objects/_index.json                # content_id -> pathspec
    """

    def __init__(self, root: Optional[str | os.PathLike] = None) -> None:
        base = Path(root) if root is not None else Path.cwd()
        self.root = base
        self.objects_dir = base / ".loom" / "objects"

    # -- internal layout helpers -----------------------------------------

    def _ensure_dir(self) -> None:
        self.objects_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _counters_path(self) -> Path:
        return self.objects_dir / "_counters.json"

    @property
    def _index_path(self) -> Path:
        return self.objects_dir / "_index.json"

    def _obj_dir(self, ref: DataObjectRef) -> Path:
        return self.objects_dir / ref.kind / str(ref.n)

    # -- id minting -------------------------------------------------------

    def new_ref(self, kind: str) -> DataObjectRef:
        """Mint the next ``Kind/<n>`` ref via an atomic per-kind counter (§9).

        The counter file is read-modify-written under an exclusive lock so two
        concurrent drivers in one workspace get distinct ids (never a collision) —
        the local stand-in for Metaflow's run-id allocation. ids start at 1.
        """
        self._ensure_dir()
        with _file_lock(self.objects_dir / "_counters.lock"):
            counters = _read_json(self._counters_path, default={})
            n = int(counters.get(kind, 0)) + 1
            counters[kind] = n
            _write_json_atomic(self._counters_path, counters)
        return DataObjectRef(kind=kind, n=n)

    # -- write / read -----------------------------------------------------

    def put(
        self,
        obj: DataObject,
        *,
        payload: Optional[bytes | str] = None,
        payload_name: str = "payload",
    ) -> DataObject:
        """Persist an object (and optional payload) immutably. Idempotent on
        ``content_id``: if an object with the same content_id exists, returns the
        existing one without writing a twin (§6)."""
        self._ensure_dir()
        # Idempotency: a prior object with the same content_id wins (no twin).
        if obj.content_id:
            existing = self.find_by_content(obj.content_id)
            if existing is not None:
                return existing

        if obj.created_at is None:
            obj.created_at = datetime.now(timezone.utc).isoformat()

        odir = self._obj_dir(obj.ref)
        odir.mkdir(parents=True, exist_ok=True)

        # Heavy payload (corpus lines, vocab json, rows csv) lives beside metadata.
        if payload is not None:
            payload_dir = odir / "payload"
            payload_dir.mkdir(parents=True, exist_ok=True)
            payload_file = payload_dir / payload_name
            mode = "wb" if isinstance(payload, (bytes, bytearray)) else "w"
            with open(payload_file, mode) as fh:
                fh.write(payload)
            obj.payload_path = str(payload_file)

        # Metadata.
        _write_json_atomic(odir / "object.json", obj.to_dict())

        # Content index → pathspec, for idempotent dedupe (§6).
        if obj.content_id:
            with _file_lock(self.objects_dir / "_index.lock"):
                index = _read_json(self._index_path, default={})
                index[obj.content_id] = obj.pathspec
                _write_json_atomic(self._index_path, index)

        return obj

    def get(self, pathspec: str | DataObjectRef) -> DataObject:
        """Load an object by pathspec (raises ``KeyError`` if absent)."""
        ref = pathspec if isinstance(pathspec, DataObjectRef) else DataObjectRef.parse(pathspec)
        meta = self._obj_dir(ref) / "object.json"
        if not meta.exists():
            raise KeyError(f"no object at pathspec {ref.pathspec!r}")
        return DataObject.from_dict(_read_json(meta, default={}))

    def find_by_content(self, content_id: str) -> Optional[DataObject]:
        """Return the existing object for a content_id, or None (idempotency)."""
        if not self._index_path.exists():
            return None
        index = _read_json(self._index_path, default={})
        pathspec = index.get(content_id)
        if not pathspec:
            return None
        try:
            return self.get(pathspec)
        except KeyError:
            return None

    def list(self, kind: Optional[str] = None) -> list[DataObject]:
        """List objects, optionally filtered by kind (``loom ls`` backend)."""
        if not self.objects_dir.exists():
            return []
        objs: list[DataObject] = []
        kinds = [kind] if kind else [
            p.name for p in self.objects_dir.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        ]
        for k in kinds:
            kdir = self.objects_dir / k
            if not kdir.is_dir():
                continue
            for ndir in kdir.iterdir():
                meta = ndir / "object.json"
                if meta.exists():
                    objs.append(DataObject.from_dict(_read_json(meta, default={})))
        objs.sort(key=lambda o: (o.kind, o.ref.n))
        return objs

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def content_id(source_fingerprint: str, spec_hash: str) -> str:
        """Compose the content address from a source fingerprint + spec hash."""
        h = hashlib.sha256()
        h.update(str(source_fingerprint).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(spec_hash).encode("utf-8"))
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Small filesystem helpers — atomic JSON write + an advisory cross-process lock
# (the local stand-in for Metaflow's atomic metadata service, §9).
# ---------------------------------------------------------------------------


def _read_json(path: Path, *, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, ValueError):
        return default


def _write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON to a temp file then ``os.replace`` it in — atomic on POSIX so a
    concurrent reader never sees a half-written file (§9)."""
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, default=str)
    os.replace(tmp, path)


class _file_lock:
    """An advisory exclusive lock over a lock file (``fcntl.flock``).

    Used to make per-kind id minting and the content index atomic across two
    concurrent drivers in one workspace (§9). Falls back to a best-effort no-op
    where ``fcntl`` is unavailable (non-POSIX); the counter write is still atomic
    via ``os.replace``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = None

    def __enter__(self) -> "_file_lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w")
        try:
            import fcntl

            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):  # pragma: no cover - non-POSIX fallback
            pass
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._fh is not None:
            try:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):  # pragma: no cover
                pass
            self._fh.close()
            self._fh = None


def default_store() -> ObjectStore:
    """The workspace store for the current directory (``$LOOM_WORKSPACE`` or CWD)."""
    return ObjectStore(os.environ.get("LOOM_WORKSPACE"))
