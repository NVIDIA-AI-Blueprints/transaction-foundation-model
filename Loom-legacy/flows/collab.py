"""Loom's collaboration / share-bundle Metaflow flow.

This module defines the single static ``FlowSpec`` -- :class:`CollabFlow` -- that the
``loom collab`` command runs (via the Metaflow MLOps interface) to **assemble a
shareable bundle** of a run: its report / model-card plus a lineage manifest
(pathspecs + fingerprints + commit). Collab is **workspace-write** to *build* the
bundle (a run + ``@card``), but the actual **SEND off-box** is the
**irreversible / external tier** (data leaving the perimeter), so it is behind an
explicit ``--send`` flag that is **OFF by default**, is gated, and the skill sets
``disable-model-invocation: true``. The default run builds the bundle only -- no
data leaves the box.

The sink for a send is **env/config-driven** -- a ``LOOM_COLLAB_WEBHOOK`` URL or a
local outbox directory (``LOOM_COLLAB_OUTBOX``) -- and is **never** a hardcoded
customer / vertical target; it is domain-neutral. Everything assembled into the
bundle is **sanitized**: only references (pathspecs / fingerprints / commit) and
small derived scalars go in, never raw rows and never secrets.

The input is a run ``run_pathspec`` (whose report/card is the bundle payload) or an
``experiment`` id (bundle the experiment's report). The run/report is read through
the Metaflow **Client API** only; Loom never touches the underlying datastore (local
or S3/minio) directly.

Flow shape::

    start --> bundle --> end

* ``start``  -- resolve the run/experiment and read its ``report``/``summary``
                artifact + lineage (pathspec + tags + commit) via the Client API.
                READ-ONLY over the upstream run.
* ``bundle`` -- assemble + sanitize the shareable bundle with the pure
                :func:`build_bundle`; only when ``send`` is True does it push the
                bundle to the env/config-driven sink (else build-only). Renders the
                ``@card`` (the bundle preview + the would-send target) and stores the
                typed summary on ``self.summary``.
* ``end``    -- carry ``self.summary`` forward so ``Run.data.summary`` exposes it.

The bundle *logic* is factored into the module-level pure function
:func:`build_bundle` (and :func:`sanitize_bundle`) so they are unit-testable on a
small in-memory report dict with no Metaflow involved. The flow step is a thin
wrapper that reads the upstream report and calls them.

Only standard Metaflow APIs are used (``FlowSpec``, ``@step``, ``Parameter``,
``@card``, and the ``current.card`` append API). ``metaflow`` / ``loom`` are
imported *inside* the steps so the flow file parses even where they are not yet
importable until the Runner subprocess sets up the environment.
"""

from __future__ import annotations

from typing import Any

from metaflow import FlowSpec, Parameter, card, current, step

#: Keys whose name looks secret-bearing; any such key is dropped from a bundle
#: during sanitization (defence in depth -- the upstream summaries carry none, but
#: collab ingests DS context that is full of untrusted strings, per CONVENTIONS §7).
_SECRET_KEY_MARKERS = (
    "key",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "authorization",
    "api_key",
    "apikey",
    "access",
    "private",
)

#: Maximum length a string value is kept at in a sanitized bundle, so a stray large
#: free-text blob (a pasted log, a raw row dump) cannot ride along off-box.
_MAX_STR_LEN = 2000


def sanitize_bundle(value: Any, _depth: int = 0) -> Any:
    """Recursively sanitize a value for inclusion in a shareable bundle (pure).

    Collab assembles a bundle that may be sent **off-box**, so everything in it is
    sanitized first (``CONVENTIONS.md`` §7): secret-looking keys are dropped, long
    strings are truncated (so a pasted log / raw-row dump cannot ride along), and
    only JSON-able scalars / lists / dicts survive. Domain-neutral and side-effect
    free.

    Args:
        value: The value to sanitize (typically a report/summary dict).
        _depth: Internal recursion guard (caps nesting to keep a bundle bounded).

    Returns:
        A sanitized, JSON-able copy of ``value`` (secret keys removed, long strings
        truncated, deep nesting flattened to a string).
    """
    if _depth > 6:
        return str(value)[:_MAX_STR_LEN]

    if isinstance(value, dict):
        out: dict = {}
        for k, v in value.items():
            key = str(k)
            if any(marker in key.lower() for marker in _SECRET_KEY_MARKERS):
                # Drop the value but record that a secret-looking key was redacted,
                # so the omission is visible rather than silent.
                out[key] = "<redacted>"
                continue
            out[key] = sanitize_bundle(v, _depth + 1)
        return out

    if isinstance(value, (list, tuple)):
        return [sanitize_bundle(v, _depth + 1) for v in value]

    if isinstance(value, str):
        return value[:_MAX_STR_LEN]

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    # Anything else (a DataFrame, a custom object) is reduced to a short repr -- raw
    # rows / objects never go off-box.
    return str(value)[:_MAX_STR_LEN]


def build_bundle(
    source_ref: str,
    report: dict | None,
    card_path: str | None = None,
    fingerprint: str | None = None,
    commit: str | None = None,
    send: bool = False,
    sink: str | None = None,
) -> dict:
    """Assemble a sanitized, shareable bundle from a run's report + lineage (pure).

    This is the unit-testable core of :class:`CollabFlow`: given a run's report/
    summary dict and its lineage references it builds the JSON-able shareable bundle
    -- the report/model-card payload + a lineage manifest (pathspecs + fingerprints
    + commit) -- with no Metaflow involved. It is **side-effect free**: it never
    sends; it only describes the bundle and *whether* it would be sent and *to where*
    (the env/config-driven sink). The payload is run through :func:`sanitize_bundle`
    so no secrets / raw rows ride along.

    Args:
        source_ref: The run/experiment pathspec the bundle is built from.
        report: The upstream report/summary dict (the model-card payload), or
            ``None`` when none could be read.
        card_path: The upstream run's ``@card`` reference (included in lineage).
        fingerprint: The data/content fingerprint for source-grounding lineage.
        commit: The source commit for traceability.
        send: Whether the off-box send was requested (``--send``). OFF by default;
            this function records the intent but never performs the send.
        sink: The resolved env/config-driven sink (a webhook URL or outbox dir), or
            ``None`` when none is configured. Domain-neutral; never a hardcoded
            customer.

    Returns:
        A JSON-able bundle dict with keys: ``source_ref``, ``payload`` (the
        sanitized report/model-card), ``lineage`` (``{"source_ref", "card_path",
        "fingerprint", "commit"}``), ``send`` (bool intent), ``sink`` (the would-send
        target or ``None``), ``sent`` (bool -- always ``False`` here; the flow sets
        it True only after a real send), and ``verdict`` (``"BUILT"`` when build-only,
        ``"SEND_REQUESTED"`` when a send was requested -- the flow upgrades to
        ``"SENT"`` after a successful push).
    """
    payload = sanitize_bundle(report) if isinstance(report, dict) else {}

    lineage = {
        "source_ref": source_ref,
        "card_path": card_path,
        "fingerprint": fingerprint,
        "commit": commit,
    }

    verdict = "SEND_REQUESTED" if send else "BUILT"
    return {
        "source_ref": source_ref,
        "payload": payload,
        "lineage": lineage,
        "send": bool(send),
        "sink": sink,
        "sent": False,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Client-API read helper (no datastore access; importable, lazily uses Metaflow).
# ---------------------------------------------------------------------------


def read_run_report(run_pathspec: str) -> tuple[dict | None, str | None]:
    """Read a run's report/summary + ``@card`` reference via the Client API.

    Resolves the pathspec to a ``metaflow.Run`` and reads the first of
    ``report``/``summary``/``profile``/``viz`` artifacts off ``.data`` (the typed
    summary the bundle wraps), plus a best-effort ``@card`` reference. Best-effort:
    any failure yields ``(None, None)``. ``metaflow`` is imported lazily.

    Args:
        run_pathspec: The run pathspec whose report to bundle (e.g.
            ``ValidateFlow/12``).

    Returns:
        ``(report_or_None, card_path_or_None)``.
    """
    from metaflow import Run, namespace

    try:
        namespace(None)
    except Exception:  # pragma: no cover - namespace API edge case
        pass

    try:
        run = Run((run_pathspec or "").strip())
        data = run.data
    except Exception:  # noqa: BLE001 - unresolvable run / metadata down
        return None, None
    if data is None:  # pragma: no cover - a successful run has data
        return None, None

    report = None
    for name in ("report", "summary", "profile", "viz"):
        value = getattr(data, name, None)
        if isinstance(value, dict):
            report = dict(value)
            break

    card_path = None
    try:
        from metaflow.cards import get_cards

        for step_obj in list(run):
            try:
                task = step_obj.task
            except Exception:  # pragma: no cover - empty/foreach step
                continue
            if task is None:
                continue
            try:
                cards = get_cards(task)
            except Exception:  # pragma: no cover - no cards for this task
                continue
            for c in cards:
                path = getattr(c, "path", None)
                if path:
                    card_path = str(path)
                    break
            if card_path:
                break
    except Exception:  # pragma: no cover - card plugin unavailable
        card_path = None

    return report, card_path


class CollabFlow(FlowSpec):
    """Assemble a sanitized shareable bundle of a run; SEND is OFF by default.

    Reads the run/experiment's report + lineage via the Client API, assembles +
    sanitizes the shareable bundle with :func:`build_bundle`, and emits a Metaflow
    run + an ``@card``. The off-box send runs **only** when ``send`` is True; the
    default (``send=False``) builds the bundle only -- no data leaves the box. The
    sink is env/config-driven (``LOOM_COLLAB_WEBHOOK`` / ``LOOM_COLLAB_OUTBOX``),
    never a hardcoded customer. The typed summary is carried on ``self.summary`` so
    the MLOps interface reads it back from ``Run.data``. Build = workspace-write;
    send = irreversible/external, gated, ``disable-model-invocation: true``.
    """

    #: Pathspec of the run whose report/card to bundle (e.g. ``ValidateFlow/12``).
    #: One of ``run_pathspec`` / ``experiment`` is required.
    run_pathspec = Parameter(
        "run_pathspec",
        default="",
        type=str,
        help="Pathspec of the run whose report/card to bundle (e.g. ValidateFlow/12).",
    )

    #: Experiment id to bundle (its report run) -- alternative to ``run_pathspec``.
    experiment = Parameter(
        "experiment",
        default="",
        type=str,
        help="Experiment id to bundle (alternative to run_pathspec).",
    )

    #: Whether to perform the real off-box send. **OFF by default** -- the default
    #: run builds the bundle only. The send is the irreversible/external action.
    send = Parameter(
        "send",
        default=False,
        type=bool,
        help="Send the bundle off-box (OFF by default; build-only otherwise).",
    )

    #: Optional source commit recorded in the bundle lineage for traceability.
    commit = Parameter(
        "commit",
        default="",
        type=str,
        help="Optional source commit recorded in the bundle lineage.",
    )

    @step
    def start(self) -> None:
        """Resolve the run/experiment and read its report + lineage via the Client API.

        Reads the upstream run's ``report``/``summary`` artifact + ``@card``
        reference (the bundle payload + lineage) through the Client API only -- never
        touching the datastore. A run with no readable report yields an empty
        payload bundle. READ-ONLY over the upstream run.
        """
        run_pathspec = (self.run_pathspec or "").strip()
        experiment = (self.experiment or "").strip()

        self._source_ref = run_pathspec or experiment
        self._report = None
        self._card_path = None
        if run_pathspec:
            self._report, self._card_path = read_run_report(run_pathspec)
        self.next(self.bundle)

    @card
    @step
    def bundle(self) -> None:
        """Assemble + sanitize the bundle and (only if requested) send it off-box.

        Delegates the bundle assembly to the pure :func:`build_bundle` (which runs
        the payload through :func:`sanitize_bundle`), so the logic is unit-testable
        without Metaflow. The off-box send runs only when ``send`` is True; the
        default stays build-only. Renders the ``@card`` (bundle preview + would-send
        target) and stores the typed summary on ``self.summary``.
        """
        import os

        send = bool(self.send)

        # Resolve the sink domain-neutrally from the environment: a webhook URL or a
        # local outbox dir. NEVER a hardcoded customer/vertical target.
        sink = (os.environ.get("LOOM_COLLAB_WEBHOOK") or "").strip() or None
        if sink is None:
            sink = (os.environ.get("LOOM_COLLAB_OUTBOX") or "").strip() or None

        commit = (self.commit or "").strip() or None
        fingerprint = self._fingerprint(self._report)

        bundle = build_bundle(
            self._source_ref,
            self._report,
            card_path=self._card_path,
            fingerprint=fingerprint,
            commit=commit,
            send=send,
            sink=sink,
        )

        # The real off-box send runs ONLY when requested. Kept in a single isolated
        # helper so the default (build-only) path provably never sends anything.
        if send:
            sent_detail = self._send_bundle(bundle, sink)
            if sent_detail.get("sent"):
                bundle["sent"] = True
                bundle["verdict"] = "SENT"
            bundle["sent_detail"] = sent_detail

        self.summary = bundle
        self._render_card(bundle)
        self.next(self.end)

    @staticmethod
    def _fingerprint(report: dict | None) -> str | None:
        """Compute a stable content fingerprint of the sanitized report (lineage).

        A short hash of the JSON-serialized sanitized report, so the bundle's
        lineage carries a content fingerprint for source-grounding. Best-effort:
        an unserializable report yields ``None``.
        """
        if not isinstance(report, dict):
            return None
        try:
            import hashlib
            import json

            blob = json.dumps(
                sanitize_bundle(report), sort_keys=True, default=str
            ).encode("utf-8")
            return hashlib.sha256(blob).hexdigest()[:16]
        except Exception:  # pragma: no cover - unserializable report
            return None

    @staticmethod
    def _send_bundle(bundle: dict, sink: str | None) -> dict:
        """Perform the real off-box send to the env/config-driven sink (send path).

        Called ONLY from ``bundle`` when ``send`` is True. The sink is the env-driven
        target resolved in ``bundle``: an ``http(s)://`` webhook (POST the sanitized
        bundle JSON) or a local outbox directory (write the bundle JSON as a file).
        This is the one place data leaves the box; it is kept in a single isolated
        helper so the build-only default provably never reaches it. Best-effort: a
        failure is reported in-band rather than raised.

        Args:
            bundle: The assembled, sanitized bundle to send.
            sink: The resolved sink (webhook URL or outbox dir), or ``None``.

        Returns:
            A small JSON-able detail dict ``{"sent", "sink", "target"|"error"}``.
        """
        import json

        if not sink:
            return {
                "sent": False,
                "sink": None,
                "error": (
                    "no collab sink configured; set LOOM_COLLAB_WEBHOOK or "
                    "LOOM_COLLAB_OUTBOX (never a hardcoded target)."
                ),
            }

        payload = json.dumps(bundle, sort_keys=True, default=str).encode("utf-8")

        # Webhook sink: POST the sanitized bundle JSON. urllib only (no extra dep).
        if sink.startswith("http://") or sink.startswith("https://"):
            try:
                import urllib.request

                req = urllib.request.Request(
                    sink,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                    code = getattr(resp, "status", None) or resp.getcode()
                return {"sent": True, "sink": "webhook", "target": sink, "status": code}
            except Exception as exc:  # noqa: BLE001 - record the attempt
                return {"sent": False, "sink": "webhook", "target": sink, "error": str(exc)}

        # Outbox sink: write the bundle JSON into the directory.
        try:
            import os
            import time

            os.makedirs(sink, exist_ok=True)
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            safe_ref = "".join(
                ch if ch.isalnum() or ch in "-_." else "_"
                for ch in str(bundle.get("source_ref") or "bundle")
            )
            entry = os.path.join(sink, f"{safe_ref}-{stamp}.json")
            with open(entry, "wb") as fh:
                fh.write(payload)
            return {"sent": True, "sink": "outbox", "target": entry}
        except Exception as exc:  # noqa: BLE001 - record the attempt
            return {"sent": False, "sink": "outbox", "target": sink, "error": str(exc)}

    def _render_card(self, bundle: dict) -> None:
        """Render a Markdown + Tables ``@card`` (bundle preview + would-send target)."""
        from metaflow.cards import Markdown, Table

        lineage = bundle.get("lineage") or {}
        payload = bundle.get("payload") or {}

        current.card.append(Markdown("# Loom collaboration bundle"))
        current.card.append(
            Markdown(
                f"**source:** `{bundle.get('source_ref')}`  \n"
                f"**send:** {bundle.get('send')} "
                f"(off-box send {'ON' if bundle.get('send') else 'OFF — build only'})  \n"
                f"**would-send target:** `{bundle.get('sink') or 'none configured'}`  \n"
                f"**sent:** {bundle.get('sent')}  \n"
                f"**VERDICT:** **{bundle.get('verdict')}**"
            )
        )

        # Lineage manifest.
        current.card.append(Markdown("## Lineage manifest"))
        current.card.append(
            Table(
                [
                    ["source_ref", lineage.get("source_ref") or "n/a"],
                    ["card_path", lineage.get("card_path") or "n/a"],
                    ["fingerprint", lineage.get("fingerprint") or "n/a"],
                    ["commit", lineage.get("commit") or "n/a"],
                ],
                headers=["field", "value"],
            )
        )

        # Sanitized payload preview (top-level keys only -- the @card is a preview,
        # the full sanitized payload is the run artifact).
        current.card.append(Markdown("## Bundle payload preview (sanitized)"))
        if payload:
            current.card.append(
                Table(
                    [
                        [str(k), _preview_value(v)]
                        for k, v in list(payload.items())[:25]
                    ],
                    headers=["field", "value (preview)"],
                )
            )
        else:
            current.card.append(
                Markdown("_No report payload found on the source run._")
            )

        sent_detail = bundle.get("sent_detail")
        if sent_detail:
            current.card.append(Markdown("## Send (off-box action performed)"))
            if sent_detail.get("sent"):
                current.card.append(
                    Markdown(
                        f"Sent to {sent_detail.get('sink')} "
                        f"`{sent_detail.get('target')}`."
                    )
                )
            else:
                current.card.append(
                    Markdown(f"_Send failed: {sent_detail.get('error')}_")
                )

    @step
    def end(self) -> None:
        """Carry ``self.summary`` forward so ``Run.data.summary`` exposes it.

        Metaflow persists step artifacts, so ``self.summary`` (set in ``bundle``) is
        already on ``Run.data``; the MLOps interface reads it back for the command's
        summary. Nothing else to do.
        """
        pass


def _preview_value(value: Any) -> str:
    """Render a short, one-line preview of a bundle payload value for the card."""
    if isinstance(value, dict):
        return f"{{{len(value)} field(s)}}"
    if isinstance(value, list):
        return f"[{len(value)} item(s)]"
    text = str(value)
    return text if len(text) <= 80 else text[:77] + "..."


if __name__ == "__main__":
    CollabFlow()
