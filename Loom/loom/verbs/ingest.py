"""``ingest`` — register a dataset as a versioned data-object (DESIGN.md item #2).

Reads a source (a directory/file of parquet/csv, or an in-memory dataframe handed
in via ``ctx.extras["dataframe"]``), sniffs schema, runs the EDA leakage gate
(:func:`loom.eda.leakage_scan`, flagging identity-like columns), records
provenance (a content fingerprint of the source + a snapshot/date note), and
persists an ``IngestDataset`` :class:`~loom.store.DataObject`.

IDEMPOTENT (§6): the object is content-addressed by ``source_fingerprint +
spec_hash``; re-ingesting the same source+spec returns the EXISTING object rather
than forking a twin. ``--force`` is the explicit escape hatch to re-pull a moving
source as a new object. No transform happens here — that is ``tokenize``'s job.

LOCKED params (``INGEST_PARAMS``). This is a workspace-write verb: cheap, CPU, no
GPU gating; the envelope still carries tier / capability_mode / cost_plan so the
gating model downstream reads the same shape.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..eda import leakage_scan
from ..registry import VerbContext, register
from ..types import (
    CapabilityMode,
    CostPlan,
    Diagnostic,
    Severity,
    Status,
    Tier,
    Verdict,
    VerbResult,
)

INGEST_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "in": {"type": "string", "description": "source path/URI to ingest"},
        "name": {"type": "string", "description": "human name for the dataset"},
        "entity": {"type": "string", "description": "grouping entity column (e.g. wallet, cust)"},
        "event": {"type": "string", "description": "event row semantics (e.g. trade, txn)"},
        "target": {"type": "string", "description": "label column for the leakage scan"},
        "force": {"type": "boolean", "description": "re-pull a moving source as a new object (§6)"},
        "streaming": {
            "type": "string",
            "enum": ["on", "off"],
            "description": "constant-memory vocab pass for a disk CSV/TSV source LARGER THAN RAM "
                           "(default 'off' = the exact in-RAM path). 'on' learns the COMPLETE vocab "
                           "in one bounded pass (every rare category seen, no UNK) and OMITS the "
                           "rows.csv payload; CSV/TSV only.",
        },
        "confirm_token": {"type": "string", "description": "agent second-call confirm token (§5.3)"},
    },
}

# Extensions we know how to read off disk, in sniff order.
_PARQUET_EXTS = (".parquet", ".pq")
_CSV_EXTS = (".csv", ".csv.gz", ".tsv")


def _read_csv(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix == ".tsv" else ","
    return pd.read_csv(path, sep=sep)


def _read_one(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if any(suffixes.endswith(e) for e in _PARQUET_EXTS):
        return pd.read_parquet(path)
    return _read_csv(path)


def _collect_source_files(root: Path) -> list[Path]:
    """List the data files under a source path (file → [file]; dir → sorted glob)."""
    if root.is_file():
        return [root]
    files: list[Path] = []
    for ext in (*_PARQUET_EXTS, *_CSV_EXTS, ".tsv"):
        files.extend(root.rglob(f"*{ext}"))
    # Deterministic order so the fingerprint is stable across runs/filesystems.
    return sorted(set(files), key=lambda p: str(p))


def _load_source(in_path: str, ctx: VerbContext) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resolve a source into a dataframe + a provenance fingerprint dict.

    Two paths: an in-memory dataframe handed in via ``ctx.extras["dataframe"]``
    (test/agent path), or a path on disk (a single file or a directory of
    parquet/csv). The fingerprint hashes file paths + sizes + mtimes (cheap and
    deterministic) for disk sources, or the dataframe content for in-memory ones.
    """
    injected = ctx.extras.get("dataframe") if ctx.extras else None
    if injected is not None:
        df = injected if isinstance(injected, pd.DataFrame) else pd.DataFrame(injected)
        # Content fingerprint: stable hash of the frame's bytes + shape.
        h = hashlib.sha256()
        h.update(f"{df.shape}".encode())
        h.update(json.dumps(list(map(str, df.columns)), sort_keys=True).encode())
        try:
            h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
        except (TypeError, ValueError):
            h.update(df.to_csv(index=True).encode())
        provenance = {
            "source": in_path or "<dataframe>",
            "source_kind": "dataframe",
            "files": [],
            "n_rows": int(len(df)),
        }
        return df, {"fingerprint": h.hexdigest(), "provenance": provenance}

    root = Path(in_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"ingest source does not exist: {in_path}")
    files = _collect_source_files(root)
    if not files:
        raise FileNotFoundError(f"no parquet/csv files found under: {in_path}")

    frames = [_read_one(f) for f in files]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    h = hashlib.sha256()
    file_meta: list[dict[str, Any]] = []
    for f in files:
        st = f.stat()
        rel = str(f.relative_to(root)) if root.is_dir() else f.name
        h.update(rel.encode())
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
        file_meta.append({"path": rel, "size": st.st_size, "mtime": int(st.st_mtime)})

    provenance = {
        "source": str(root),
        "source_kind": "directory" if root.is_dir() else "file",
        "files": file_meta,
        "n_rows": int(len(df)),
    }
    return df, {"fingerprint": h.hexdigest(), "provenance": provenance}


def _disk_fingerprint(root: Path, files: list[Path]) -> tuple[str, list[dict[str, Any]]]:
    """The disk source fingerprint (relpath + size + int(mtime) per file, in the
    same order as :func:`_collect_source_files`).

    Factored out of :func:`_load_source` so the streaming path produces a
    BYTE-IDENTICAL fingerprint — and therefore the SAME ``content_id`` — as the
    in-RAM path for the same source. (The fingerprint is file-metadata only; no
    byte read, so it is O(1) on huge files. Decision #4.)"""
    h = hashlib.sha256()
    file_meta: list[dict[str, Any]] = []
    for f in files:
        st = f.stat()
        rel = str(f.relative_to(root)) if root.is_dir() else f.name
        h.update(rel.encode())
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
        file_meta.append({"path": rel, "size": st.st_size, "mtime": int(st.st_mtime)})
    return h.hexdigest(), file_meta


def _is_parquet(path: Path) -> bool:
    suffixes = "".join(path.suffixes).lower()
    return any(suffixes.endswith(e) for e in _PARQUET_EXTS)


class StreamingUnsupported(ValueError):
    """``--streaming on`` asked for on a source the v1 streamer cannot handle
    (parquet, or an injected dataframe). Surfaced as a clean REFUSED envelope —
    the caller falls back to NOTHING (the user must re-run without the flag)."""


def _load_source_streaming(
    in_path: str, ctx: VerbContext, *, target: Optional[str]
) -> tuple["StreamingStats", dict[str, Any]]:
    """The constant-memory disk-CSV/TSV path (decision #1/#5/#6).

    Resolves the SAME files in the SAME order as :func:`_load_source`, builds the
    SAME metadata fingerprint (so ``content_id`` is unchanged), and folds one
    chunked ``read_csv`` iterator PER FILE through
    :func:`loom.engine.streaming.stream_stats`. Refuses parquet (CSV/TSV only in
    v1) and the injected-dataframe path (that is always in-RAM). Never reads the
    whole frame into memory; never prints.

    Returns ``(stats, {"fingerprint", "provenance"})`` mirroring
    :func:`_load_source`'s second element so the caller's content-addressing is
    identical."""
    from ..engine import streaming  # local import: keeps the in-RAM path import-light

    if ctx.extras and ctx.extras.get("dataframe") is not None:
        raise StreamingUnsupported(
            "--streaming on is for a disk CSV/TSV source; an injected dataframe "
            "is already in memory — re-run without --streaming"
        )

    root = Path(in_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"ingest source does not exist: {in_path}")
    files = _collect_source_files(root)
    if not files:
        raise FileNotFoundError(f"no parquet/csv files found under: {in_path}")
    parquet = [f for f in files if _is_parquet(f)]
    if parquet:
        raise StreamingUnsupported(
            "--streaming on supports CSV/TSV only in v1 (a >RAM parquet directory "
            "is out of scope — it would add a pyarrow row-group streamer); re-run "
            "without --streaming to use the full-frame parquet path"
        )

    fingerprint, file_meta = _disk_fingerprint(root, files)

    # The column order is the header of the FIRST file (the same order a single
    # read_csv / pd.concat would yield; _collect_source_files is already sorted).
    column_order = list(pd.read_csv(files[0], sep=_sep_for(files[0]), nrows=0).columns)

    def _chunks() -> Any:
        # One chunked iterator per file, concatenated in _collect_source_files
        # order (matches the in-RAM pd.concat([_read_one(f) for f in files])).
        for f in files:
            reader = pd.read_csv(
                f,
                sep=_sep_for(f),
                chunksize=streaming.CHUNK_ROWS,
                dtype=str,
                keep_default_na=True,
                na_values=[""],
            )
            for chunk in reader:
                yield chunk

    stats = streaming.stream_stats(_chunks(), column_order=column_order, target=target)
    stats.source_fingerprint = fingerprint

    provenance = {
        "source": str(root),
        "source_kind": "directory" if root.is_dir() else "file",
        "files": file_meta,
        "n_rows": int(stats.n_rows),
    }
    return stats, {"fingerprint": fingerprint, "provenance": provenance}


def _sep_for(path: Path) -> str:
    """The field separator the in-RAM ``_read_csv`` uses (TSV → tab, else comma)."""
    return "\t" if path.suffix == ".tsv" else ","


def _sniff_schema(df: pd.DataFrame) -> dict[str, Any]:
    """A cheap schema sniff: column → dtype + null-fraction + cardinality."""
    n = max(len(df), 1)
    cols: dict[str, Any] = {}
    for c in df.columns:
        s = df[c]
        cols[str(c)] = {
            "dtype": str(s.dtype),
            "null_frac": round(float(s.isna().mean()), 6),
            "n_unique": int(s.dropna().nunique()),
        }
    return {"n_rows": int(len(df)), "n_cols": int(len(df.columns)), "columns": cols}


def _spec_hash(args: dict[str, Any]) -> str:
    """Hash the ingest spec (the knobs that change the object's identity)."""
    spec = {
        "name": args.get("name"),
        "entity": args.get("entity"),
        "event": args.get("event"),
        "target": args.get("target"),
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()


def _ingest_result(
    ref,
    verdict: Verdict,
    eda_diags: list[Diagnostic],
    *,
    name: Optional[str],
    entity: Optional[str],
    event: Optional[str],
    schema: dict[str, Any],
    provenance: dict[str, Any],
    content_id: str,
    experiment: Optional[str],
) -> VerbResult:
    """The single OK envelope for an ingested IngestDataset.

    Both the fresh-write path and the idempotent-hit path build the envelope here,
    so the result is byte-identical for the same source+spec regardless of whether
    the object was just written or already existed (§2.1 dual-driver byte-identity
    / §6 idempotency). The pathspec is content-addressed, identical across faces."""
    n_rows = provenance.get("n_rows")
    n_cols = schema.get("n_cols")
    if name:
        summary = f"{ref.pathspec} '{name}' rows={n_rows} eda VERDICT: {verdict.value}"
    else:
        summary = f"{ref.pathspec} rows={n_rows} cols={n_cols} eda VERDICT: {verdict.value}"
    return VerbResult(
        verb="ingest",
        status=Status.OK,
        verdict=verdict,
        tier=Tier.WORKSPACE_WRITE,
        capability_mode=CapabilityMode.NONE,
        summary=summary,
        outputs=[ref],
        diagnostics=eda_diags,
        data={
            "pathspec": ref.pathspec,
            "name": name,
            "entity": entity,
            "event": event,
            "content_id": content_id,
            "schema": schema,
            "eda": {
                "verdict": verdict.value,
                "n_flags": len(eda_diags),
                "flags": [d.data for d in eda_diags],
            },
            "provenance": provenance,
        },
        experiment=experiment,
        cost_plan=CostPlan(),
    )


@register(
    "ingest",
    summary="register a dataset as a versioned, content-addressed object (schema sniff + EDA leakage gate)",
    tier=Tier.WORKSPACE_WRITE,
    capability_mode=CapabilityMode.NONE,
    params=INGEST_PARAMS,
)
def _ingest(args: dict[str, Any], ctx: VerbContext) -> VerbResult:
    from ..store import DataObject  # local import: store is the v0.2 seam

    in_path = args.get("in") or ""
    name = args.get("name")
    entity = args.get("entity")
    event = args.get("event")
    target = args.get("target")
    force = bool(args.get("force", False))
    streaming_on = str(args.get("streaming") or "off").lower() == "on"
    experiment = ctx.experiment

    # --- load the source + fingerprint it -------------------------------------
    # Default (flag off) / injected dataframe / parquet → the EXACT in-RAM path:
    # ``df`` is the full frame, ``stats`` is None, and everything below is
    # byte-identical to current main. ``--streaming on`` over a disk CSV/TSV →
    # ``stats`` is a constant-memory StreamingStats bundle and ``df`` is None
    # (decision #1: streaming runs ONLY for an explicit flag on a disk CSV/TSV).
    df: Optional[pd.DataFrame] = None
    stats = None
    try:
        if streaming_on and not (ctx.extras and ctx.extras.get("dataframe") is not None):
            stats, src = _load_source_streaming(in_path, ctx, target=target)
        else:
            df, src = _load_source(in_path, ctx)
    except StreamingUnsupported as exc:
        # ``--streaming on`` on a source the v1 streamer can't take → clean REFUSED
        # (no object written); the user re-runs without the flag.
        return VerbResult(
            verb="ingest",
            status=Status.FAIL,
            verdict=Verdict.FAIL,
            tier=Tier.WORKSPACE_WRITE,
            capability_mode=CapabilityMode.NONE,
            summary=f"ingest refused --streaming: {exc}",
            diagnostics=[
                Diagnostic(
                    contract="EDA",
                    severity=Severity.ERROR,
                    message=str(exc),
                    fix="re-run `loom ingest` WITHOUT --streaming for this source "
                    "(streaming v1 is disk CSV/TSV only)",
                )
            ],
            experiment=experiment,
            cost_plan=CostPlan(),
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        return VerbResult(
            verb="ingest",
            status=Status.FAIL,
            verdict=Verdict.FAIL,
            tier=Tier.WORKSPACE_WRITE,
            capability_mode=CapabilityMode.NONE,
            summary=f"ingest failed to read source: {exc}",
            diagnostics=[
                Diagnostic(
                    contract="EDA",
                    severity=Severity.ERROR,
                    message=str(exc),
                    fix="check the --in path points to a readable parquet/csv file or directory",
                )
            ],
            experiment=experiment,
            cost_plan=CostPlan(),
        )

    fingerprint = src["fingerprint"]
    provenance = src["provenance"]
    spec_hash = _spec_hash(args)
    content_id = ctx.store.content_id(fingerprint, spec_hash)

    # --- idempotency (§6): same source+spec → return the EXISTING object with an
    # envelope byte-identical to the fresh-write one (§2.1 — neither two sequential
    # identical calls nor the two driver faces may diverge). All the envelope
    # fields are reconstructed from the persisted object so the byte-identity
    # invariant holds regardless of fresh-write vs idempotent-hit. -------------
    if not force:
        existing = ctx.store.find_by_content(content_id)
        if existing is not None:
            ex = existing.extras
            ex_report = ex.get("ingest_report", {})
            eda_diags = [
                Diagnostic(
                    contract=d.get("contract", "EDA"),
                    severity=Severity(d.get("severity", "warning")),
                    message=d.get("message", ""),
                    fix=d.get("fix"),
                    data=d.get("data", {}),
                )
                for d in ex_report.get("eda_diagnostics", [])
            ]
            return _ingest_result(
                existing.ref,
                existing.verdict,
                eda_diags,
                name=ex.get("name"),
                entity=ex.get("entity"),
                event=ex.get("event"),
                schema=ex.get("schema", {}),
                provenance=ex_report.get("provenance", provenance),
                content_id=content_id,
                experiment=experiment,
            )

    # --- schema sniff + EDA leakage gate --------------------------------------
    # In-RAM path: the same _sniff_schema + leakage_scan over the full frame.
    # Streaming path: the SAME-shaped schema (to_schema, plus min/max on numeric
    # cols) and the SAME Diagnostic dicts (to_eda_diagnostics) reduced from the
    # constant-memory accumulators — no frame ever resident.
    if stats is not None:
        schema = stats.to_schema()
        eda_diags = [
            Diagnostic(
                contract=d.get("contract", "EDA"),
                severity=Severity(d.get("severity", "warning")),
                message=d.get("message", ""),
                fix=d.get("fix"),
                data=d.get("data", {}) or {},
            )
            for d in stats.to_eda_diagnostics(target)
        ]
    else:
        schema = _sniff_schema(df)
        eda_diags = leakage_scan(df, target=target)
    eda_verdict = Verdict.REVIEW if eda_diags else Verdict.PASS

    ingest_report = {
        "eda_verdict": eda_verdict.value,
        "eda_diagnostics": [d.to_dict() for d in eda_diags],
        "provenance": provenance,
        "snapshot_at": datetime.now(timezone.utc).isoformat(),
    }

    # --- persist the IngestDataset object (content-addressed) -----------------
    ref = ctx.store.new_ref("IngestDataset")
    obj = DataObject(
        ref=ref,
        kind="IngestDataset",
        content_id=content_id,
        parents=[],
        producer_verb="ingest",
        producer_args={
            "in": in_path,
            "name": name,
            "entity": entity,
            "event": event,
            "target": target,
        },
        signatures={
            "source_fingerprint": fingerprint,
            "spec_hash": spec_hash,
            "n_rows": provenance["n_rows"],
            "n_cols": schema["n_cols"],
        },
        verdict=eda_verdict,
        status=Status.OK,
        experiment=experiment,
        created_at=datetime.now(timezone.utc).isoformat(),
        extras={
            "name": name,
            "entity": entity,
            "event": event,
            "target": target,
            "schema": schema,
            "ingest_report": ingest_report,
        },
    )
    if stats is not None:
        # Streaming path (decision #2/#6): the complete vocab travels INLINE on
        # extras['col_stats'] (the JSON-able StreamingStats bundle propose reads
        # frame-free), so we SKIP the whole rows.csv payload — the second OOM
        # source, and vocab-learning never needs rows resident. A tiny stats.json
        # is the heavy payload; the marker makes the omission explicit so a
        # downstream rows reader (prepare/_load_rows) degrades to vocab-only.
        obj.extras["col_stats"] = stats.to_dict()
        obj.extras["rows_payload"] = "streamed-omitted"
        payload = json.dumps(stats.to_dict(), sort_keys=True)
        stored = ctx.store.put(obj, payload=payload, payload_name="stats.json")
    else:
        # Persist the rows as the heavy payload so downstream verbs (baseline) read
        # them. CSV is used because the Phase-0 dep set is numpy/pandas/PyYAML only
        # (no pyarrow/fastparquet) — TODO(v0.2): parquet once an engine is a dep.
        payload = df.to_csv(index=False)
        stored = ctx.store.put(obj, payload=payload, payload_name="rows.csv")

    return _ingest_result(
        stored.ref,
        eda_verdict,
        eda_diags,
        name=name,
        entity=entity,
        event=event,
        schema=schema,
        provenance=provenance,
        content_id=content_id,
        experiment=experiment,
    )
