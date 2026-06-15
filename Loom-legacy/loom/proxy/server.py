"""The Loom proxy gateway server — Anthropic passthrough + central logging.

A thin Starlette ASGI app exposing:

* ``POST /v1/messages`` — the Anthropic Messages API shape. It (1) authenticates
  the caller by their Loom key, (2) injects Loom's system prompt, (3) forwards to
  the real Anthropic Messages API using a *server-side* vendor key the user never
  sees, supporting both non-streaming and streaming (SSE) responses, and (4) logs
  every call as one JSONL row to the call log (the moat corpus).
* ``GET /healthz`` — a liveness probe (no auth, no upstream call).

Design notes:

* **Secrets live only in the server process, from the environment.**
  :class:`ProxyConfig` reads the accepted Loom key(s), the real upstream vendor
  key, the upstream URL, and the JSONL log path from the environment at server
  start; none of these are ever placed on a :class:`~loom.config.LoomConfig`, on
  the app object beyond the in-process ``ProxyConfig``, or written to the log.
* **Pure, offline-testable decision helpers.** :func:`authenticate`,
  :func:`inject_loom_system`, and :func:`build_log_row` are network-free module
  functions exercised directly by the tests. The only impure edge — the upstream
  forward — is a single injectable coroutine (``forward``) the tests monkeypatch
  with a fake, so the whole gateway is verifiable with **no** real network.
* **No direct S3 / no datastore.** The only sink is the local JSONL call log at
  ``ProxyConfig.log_path``; its parent directory is created on demand at first
  write (matching :class:`loom.corpus.Corpus`).
* **Robust.** Upstream errors (timeouts, 4xx/5xx) are passed through to the
  caller with the upstream status/body rather than masked; the call is still
  logged (with the error) so failed traffic also feeds the moat.

``starlette`` / ``httpx`` are imported at the top, but ``loom.proxy`` is only
imported by the CLI ``proxy serve`` path and the tests — never by ``loom`` core —
so an optional-dep gap here cannot break core import.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, MutableMapping

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

# Default upstream the gateway forwards to in v0 (Anthropic passthrough). It is a
# server-side default, overridable via ``LOOM_PROXY_UPSTREAM_URL`` for testing /
# pointing at a compatible relay; it is NOT a per-request knob.
_DEFAULT_UPSTREAM_URL = "https://api.anthropic.com/v1/messages"

# Anthropic API version sent upstream when the caller did not supply one.
_DEFAULT_ANTHROPIC_VERSION = "2023-06-01"

# Loom's injected system prompt. Prepended as its own system block ahead of
# whatever system the caller sent, so every call through the gateway carries
# Loom's framing. Intentionally generic / domain-neutral — no tenant or vendor
# specifics (those would taint the cross-tenant moat corpus).
LOOM_SYSTEM_PROMPT = (
    "You are operating inside Loom, an agentic CLI for the full data-science "
    "lifecycle. "
    "Be rigorous, reproducible, and metric-driven: optimize exactly the stated "
    "evaluation metric, never fabricate results, and prefer simple, verifiable "
    "solutions. This request is routed through Loom's model gateway."
)

# Upstream timeout: generous read (streaming can run long), short connect.
_UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)


def _read_keys_file(path: str) -> set[str]:
    """Read accepted Loom keys from a file (one key per line; blanks/#-comments skipped).

    The file half of the gateway's key lookup: mount a keys file into the container
    (``LOOM_API_KEYS_FILE``) and add/rotate caller keys without changing the
    environment or rebuilding the image. Merges with the env-supplied keys. A
    missing or unreadable file yields an empty set (the env keys still apply) and
    never raises — the gateway still fails closed if NO key is configured anywhere.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            return {
                line.strip()
                for line in fh
                if line.strip() and not line.strip().startswith("#")
            }
    except OSError:
        return set()


class ProxyAuthError(Exception):
    """Raised when a caller's Loom key is missing or not on the allowlist."""


@dataclass(frozen=True)
class ProxyConfig:
    """Server-side, environment-derived settings for the Loom gateway.

    All of these are read from the environment of the **server** process (never
    from a :class:`~loom.config.LoomConfig`, never sent to a client). The secret
    material — the accepted Loom key(s) and the real upstream vendor key — lives
    only here, in-process.

    Attributes:
        allowed_keys: The set of Loom keys the gateway accepts from callers
            (from ``LOOM_API_KEY`` and a comma-separated ``LOOM_API_KEYS``).
        upstream_key: The real vendor key the gateway forwards with
            (``ANTHROPIC_API_KEY`` on the server) — never seen by the caller.
        upstream_url: The upstream Messages endpoint (default
            ``https://api.anthropic.com/v1/messages``; override with
            ``LOOM_PROXY_UPSTREAM_URL``).
        log_path: The JSONL call-log path (the moat corpus). Defaults to the Loom
            config's ``proxy_log_path`` (env ``LOOM_PROXY_LOG_PATH``), anchored
            absolute at config load.
        default_tenant: Tenant tag applied when a call sends no ``x-loom-tenant``.
        default_owned_by: ``owned_by`` tag applied when a call sends no
            ``x-loom-owned-by`` (``"general"`` = usable by the cross-tenant moat
            model; any other value = tenant-owned, excluded from the general set).
    """

    allowed_keys: frozenset[str]
    upstream_key: str
    upstream_url: str
    log_path: str
    default_tenant: str = "default"
    default_owned_by: str = "general"

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None
    ) -> "ProxyConfig":
        """Build a :class:`ProxyConfig` from the server environment.

        Reads the accepted Loom key(s), the upstream vendor key, the upstream URL,
        and the log path (via :meth:`LoomConfig.load`, so the path is anchored
        absolute exactly like the corpus/learnings paths).

        Args:
            env: The environment mapping to read (defaults to ``os.environ``).

        Returns:
            A populated :class:`ProxyConfig`. It may have an empty
            ``allowed_keys`` / blank ``upstream_key``; the caller (``serve`` /
            ``create_app``) decides how strict to be — request handling always
            fails closed when either is missing.
        """
        e: Mapping[str, str] = env if env is not None else os.environ

        keys: set[str] = set()
        single = (e.get("LOOM_API_KEY") or "").strip()
        if single:
            keys.add(single)
        for part in (e.get("LOOM_API_KEYS") or "").split(","):
            part = part.strip()
            if part:
                keys.add(part)

        # Also accept keys from a file mounted into the container (one key per
        # line) — the file half of the kv/file lookup, so keys can be rotated
        # without touching the env. Both sources merge into the allowlist.
        keys_file = (e.get("LOOM_API_KEYS_FILE") or "").strip()
        if keys_file:
            keys |= _read_keys_file(keys_file)

        # The log path comes from the Loom config so it is anchored absolute the
        # same way corpus_path / learnings_path are; tenant/owned_by defaults too.
        from loom.config import LoomConfig

        cfg = LoomConfig.load(env=e)

        return cls(
            allowed_keys=frozenset(keys),
            upstream_key=(e.get("ANTHROPIC_API_KEY") or "").strip(),
            upstream_url=(e.get("LOOM_PROXY_UPSTREAM_URL") or _DEFAULT_UPSTREAM_URL).strip(),
            log_path=cfg.proxy_log_path,
            default_tenant=cfg.tenant,
            default_owned_by=cfg.owned_by,
        )


def extract_caller_key(headers: Mapping[str, str]) -> str | None:
    """Pull the caller's presented Loom key from request headers.

    Accepts either ``x-api-key`` (Anthropic's native header, which AIDE's client
    sends) or ``Authorization: Bearer <key>``.

    Args:
        headers: A case-insensitive header mapping (Starlette's
            ``request.headers``; a plain lowercase-keyed dict works in tests).

    Returns:
        The presented key string, or ``None`` if neither header is present.
    """
    api_key = headers.get("x-api-key")
    if api_key:
        return api_key.strip()
    auth = headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[len("bearer ") :].strip()
    return None


def authenticate(headers: Mapping[str, str], allowed_keys: frozenset[str] | set[str]) -> str:
    """Authenticate a caller by their presented Loom key against the allowlist.

    Args:
        headers: The request headers (``x-api-key`` or ``Authorization: Bearer``).
        allowed_keys: The set of accepted Loom keys (from :class:`ProxyConfig`).

    Returns:
        The accepted key (so the caller can be identified; the value itself is
        never logged).

    Raises:
        ProxyAuthError: If the server has no key configured, the caller presented
            none, or the presented key is not on the allowlist.
    """
    if not allowed_keys:
        raise ProxyAuthError(
            "the Loom gateway has no LOOM_API_KEY / LOOM_API_KEYS configured; "
            "refusing all callers"
        )
    presented = extract_caller_key(headers)
    if not presented:
        raise ProxyAuthError(
            "missing Loom key: send it as 'x-api-key' or 'Authorization: Bearer'"
        )
    if presented not in allowed_keys:
        raise ProxyAuthError("invalid Loom key")
    return presented


def inject_loom_system(
    body: Mapping[str, Any], loom_prompt: str = LOOM_SYSTEM_PROMPT
) -> dict[str, Any]:
    """Return a copy of an Anthropic Messages body with Loom's system prepended.

    The Anthropic Messages API accepts ``system`` as either a plain string or a
    list of content blocks. This normalizes both to the block-list form and
    prepends one Loom system block, so every call through the gateway carries
    Loom's framing without dropping the caller's own system instructions.

    Args:
        body: The parsed JSON request body (an Anthropic Messages request).
        loom_prompt: The Loom system text to prepend (defaults to
            :data:`LOOM_SYSTEM_PROMPT`).

    Returns:
        A new dict (the input is not mutated) whose ``system`` is a list of
        content blocks beginning with the Loom block.
    """
    out = dict(body)
    blocks: list[dict[str, Any]] = [{"type": "text", "text": loom_prompt}]

    existing = out.get("system")
    if isinstance(existing, str):
        if existing:
            blocks.append({"type": "text", "text": existing})
    elif isinstance(existing, list):
        blocks.extend(existing)
    # Absent or any other type: keep just the Loom block.

    out["system"] = blocks
    return out


def _response_text(payload: Mapping[str, Any]) -> str:
    """Best-effort extract assistant text from an Anthropic response payload.

    Concatenates the ``text`` of every ``type == "text"`` content block, tolerating
    missing/odd shapes (returns ``""``) so logging never crashes.

    Args:
        payload: A parsed Anthropic Messages response object.

    Returns:
        The concatenated assistant text (may be empty).
    """
    parts: list[str] = []
    content = payload.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


def build_log_row(
    *,
    model: str,
    request_system: Any,
    request_messages: Any,
    response_payload: Mapping[str, Any] | None,
    latency_ms: float,
    streamed: bool,
    status: int,
    error: str | None,
    skill: str | None,
    tenant: str,
    owned_by: str,
    trajectory_id: str | None = None,
) -> dict[str, Any]:
    """Assemble one JSONL call-log row (the moat corpus record).

    Captures the request system + messages, the response text + usage, the model,
    latency, and the IP-boundary tags (``tenant`` / ``owned_by``, the same tags
    :mod:`loom.corpus` uses to keep tenant-confidential data out of the
    cross-tenant general set). **No key material is ever included** (the inputs
    carry none).

    Args:
        model: The model name from the request body.
        request_system: The (Loom-injected) system blocks sent upstream.
        request_messages: The caller's ``messages`` array.
        response_payload: The parsed upstream response (non-streaming), or
            ``None`` for a streamed / errored call.
        latency_ms: Wall-clock upstream latency in milliseconds.
        streamed: Whether the response was streamed (SSE passthrough).
        status: The upstream HTTP status returned to the caller.
        error: An error string if the forward failed / upstream errored, else
            ``None``.
        skill: Optional ``x-loom-skill`` value (which Loom skill made the call).
        tenant: The ``x-loom-tenant`` tag (or the server default).
        owned_by: The ``x-loom-owned-by`` tag (or the server default).
        trajectory_id: Optional telemetry trajectory join key (from the
            ``x-loom-trajectory`` header or ``LOOM_TRAJECTORY_ID`` env). Lets
            :func:`loom.telemetry.assemble_trajectory` stitch this LLM call into the
            owning trajectory. Purely additive; ``None`` when the caller sent none.

    Returns:
        A JSON-serializable dict for one JSONL row.
    """
    usage: dict[str, Any] | None = None
    response_text: str | None = None
    if response_payload is not None:
        raw_usage = response_payload.get("usage")
        if isinstance(raw_usage, Mapping):
            usage = dict(raw_usage)
        response_text = _response_text(response_payload)

    return {
        "ts": time.time(),
        "model": model,
        "system": request_system,
        "messages": request_messages,
        "response_text": response_text,
        "usage": usage,
        "latency_ms": round(latency_ms, 3),
        "streamed": streamed,
        "status": status,
        "error": error,
        "skill": skill,
        "tenant": tenant,
        "owned_by": owned_by,
        "trajectory_id": trajectory_id,
    }


def log_call(log_path: str, row: Mapping[str, Any]) -> None:
    """Append one call-log row to the JSONL moat corpus at ``log_path``.

    The parent directory is created lazily and the line is flushed so a crash
    mid-run still leaves a valid prefix of complete lines (matching
    :meth:`loom.corpus.Corpus.record`). Never writes key material.

    Args:
        log_path: The JSONL call-log path (anchored absolute at config load).
        row: The row to append (JSON-serializable).
    """
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


def _emit_llm_request_event(
    trajectory_id: str,
    model: str,
    skill: str | None,
    tenant: str,
    owned_by: str,
) -> None:
    """Emit a redacted telemetry ``llm_request`` event for this proxied call.

    The trajectory-correlation side-channel that complements the proxy_calls row:
    a small, low-cardinality event tagged with the ``trajectory_id`` so
    :func:`loom.telemetry.assemble_trajectory` can order this LLM call within the
    owning trajectory. A no-op when ``LOOM_TELEMETRY`` is off (``log_event``
    returns ``None``), and entirely best-effort -- any failure (a missing optional
    dep, a config edge) is swallowed so a telemetry hiccup never blocks the
    gateway's real job of forwarding the call. Carries the model name only; the
    request/response bytes live in the proxy_calls row, not in telemetry.

    Args:
        trajectory_id: The trajectory join key (header/env derived).
        model: The model name from the request body.
        skill: The optional ``x-loom-skill`` provenance tag.
        tenant: The resolved tenant tag.
        owned_by: The resolved IP-boundary owner tag.
    """
    try:
        from loom.config import LoomConfig
        from loom.telemetry import log_event

        # Reflect the proxy call's IP-boundary tags onto the config so the event's
        # standard attributes carry the same owned_by/tenant the proxy_calls row
        # does (the distillation export filters on owned_by).
        cfg = LoomConfig.load()
        cfg = dataclasses.replace(cfg, tenant=tenant, owned_by=owned_by)
        log_event(
            "llm_request",
            trajectory_id,
            cfg,
            attrs={
                k: v
                for k, v in (("model", model), ("skill", skill))
                if v
            },
            run_id=trajectory_id,
        )
    except Exception:  # noqa: BLE001 - telemetry is best-effort, never blocks
        pass


def _upstream_headers(headers: Mapping[str, str], vendor_key: str) -> dict[str, str]:
    """Build the headers sent upstream: the server-side vendor key + version.

    The caller's Loom key is NOT forwarded — the gateway swaps in the real vendor
    key so the user never sees it. The Anthropic API version (and any
    ``anthropic-beta``) is passed through if set, else defaulted.

    Args:
        headers: The incoming request headers.
        vendor_key: The server-side real Anthropic key.

    Returns:
        The header dict to send upstream.
    """
    out = {
        "x-api-key": vendor_key,
        "anthropic-version": headers.get("anthropic-version") or _DEFAULT_ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    beta = headers.get("anthropic-beta")
    if beta:
        out["anthropic-beta"] = beta
    return out


# Forwarder coroutine type: (url, json_body, headers, stream) -> httpx.Response.
# Factored out so tests inject a fake (no network).
Forwarder = Callable[..., Awaitable[httpx.Response]]


async def _httpx_forward(
    *, url: str, json_body: Mapping[str, Any], headers: Mapping[str, str], stream: bool
) -> httpx.Response:  # pragma: no cover - exercised only against a live upstream
    """Default forwarder: POST to the upstream via httpx (real network).

    The offline test suite injects a fake forwarder, so this real edge is not
    covered there; it is the production path that actually talks to Anthropic.
    """
    client = httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT)
    if stream:
        request = client.build_request("POST", url, json=json_body, headers=headers)
        return await client.send(request, stream=True)
    try:
        return await client.post(url, json=json_body, headers=headers)
    finally:
        await client.aclose()


def create_app(
    config: ProxyConfig | None = None,
    *,
    forward: Forwarder | None = None,
) -> Starlette:
    """Build the Loom gateway ASGI app.

    Args:
        config: The server-side :class:`ProxyConfig`. Built from the environment
            (:meth:`ProxyConfig.from_env`) if ``None``.
        forward: The upstream forwarder coroutine. Defaults to the real httpx
            forwarder; tests inject a fake to avoid any network.

    Returns:
        A configured :class:`starlette.applications.Starlette` app exposing
        ``POST /v1/messages`` and ``GET /healthz``.
    """
    cfg = config or ProxyConfig.from_env()
    forwarder: Forwarder = forward or _httpx_forward

    async def healthz(_request: Request) -> Response:
        """Liveness probe — no auth, no upstream call."""
        return JSONResponse({"status": "ok", "service": "loom-proxy"})

    async def messages(request: Request) -> Response:
        """Handle ``POST /v1/messages``: auth, inject, forward, log."""
        headers = request.headers

        # (1) Authenticate the caller by their Loom key.
        try:
            authenticate(headers, cfg.allowed_keys)
        except ProxyAuthError as exc:
            return _error(401, "authentication_error", str(exc))

        # The server-side real vendor key (never seen by the caller).
        if not cfg.upstream_key:
            return _error(
                500, "api_error",
                "gateway is missing its server-side ANTHROPIC_API_KEY",
            )

        # Parse the request body.
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - malformed JSON -> 400
            return _error(400, "invalid_request_error", "request body is not JSON")
        if not isinstance(body, Mapping):
            return _error(
                400, "invalid_request_error", "request body must be an object"
            )

        # (2) Inject Loom's system prompt.
        injected = inject_loom_system(body)
        streamed = bool(injected.get("stream"))
        model = str(injected.get("model", ""))

        # IP-boundary + attribution tags (header-driven; fall back to config).
        skill = headers.get("x-loom-skill")
        tenant = headers.get("x-loom-tenant") or cfg.default_tenant
        owned_by = headers.get("x-loom-owned-by") or cfg.default_owned_by
        # Telemetry trajectory join key: prefer the per-call header, else the
        # server-process env (set by a caller that drives one trajectory). Lets
        # the telemetry layer stitch this LLM call into the owning trajectory.
        trajectory_id = (
            headers.get("x-loom-trajectory") or os.environ.get("LOOM_TRAJECTORY_ID")
        )

        # Emit a redacted-by-default telemetry llm_request event correlated by the
        # trajectory id (a no-op when LOOM_TELEMETRY is off, and never raises so the
        # forward is never blocked by a telemetry hiccup). Content is the model name
        # only; the request/response bytes stay in the proxy_calls row, not here.
        if trajectory_id:
            _emit_llm_request_event(trajectory_id, model, skill, tenant, owned_by)

        up_headers = _upstream_headers(headers, cfg.upstream_key)

        # (3) Forward to the upstream (real vendor key), timing the call.
        start = time.perf_counter()
        try:
            upstream = await forwarder(
                url=cfg.upstream_url,
                json_body=injected,
                headers=up_headers,
                stream=streamed,
            )
        except Exception as exc:  # noqa: BLE001 - upstream transport failure
            latency_ms = (time.perf_counter() - start) * 1000.0
            # (4) Log the failed call (still moat fuel) and surface the error.
            log_call(
                cfg.log_path,
                build_log_row(
                    model=model,
                    request_system=injected.get("system"),
                    request_messages=injected.get("messages"),
                    response_payload=None,
                    latency_ms=latency_ms,
                    streamed=streamed,
                    status=502,
                    error=f"{type(exc).__name__}: {exc}",
                    skill=skill,
                    tenant=tenant,
                    owned_by=owned_by,
                    trajectory_id=trajectory_id,
                ),
            )
            return _error(
                502, "api_error",
                f"upstream forward failed: {type(exc).__name__}",
            )

        if streamed:
            # SSE passthrough: stream the upstream bytes straight to the caller.
            # The full text isn't reassembled (buys little, costs memory); the row
            # records that a stream occurred + its status.
            latency_ms = (time.perf_counter() - start) * 1000.0
            log_call(
                cfg.log_path,
                build_log_row(
                    model=model,
                    request_system=injected.get("system"),
                    request_messages=injected.get("messages"),
                    response_payload=None,
                    latency_ms=latency_ms,
                    streamed=True,
                    status=upstream.status_code,
                    error=None if upstream.status_code < 400 else "upstream error",
                    skill=skill,
                    tenant=tenant,
                    owned_by=owned_by,
                    trajectory_id=trajectory_id,
                ),
            )

            async def _stream():
                # Prefer a true byte stream from the upstream (the production
                # path, where the forwarder opened a streaming request). If the
                # response was already buffered (e.g. a test fixture built with
                # eager ``content=``), fall back to yielding its bytes whole — the
                # caller still receives the full SSE body.
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                except httpx.StreamConsumed:
                    yield upstream.content
                finally:
                    await upstream.aclose()

            return StreamingResponse(
                _stream(),
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "text/event-stream"),
            )

        # Non-streaming: read the upstream body, log, pass it through.
        latency_ms = (time.perf_counter() - start) * 1000.0
        try:
            payload = upstream.json()
        except Exception:  # noqa: BLE001 - non-JSON upstream body (e.g. an error page)
            payload = None
        payload_map = payload if isinstance(payload, Mapping) else None

        log_call(
            cfg.log_path,
            build_log_row(
                model=model,
                request_system=injected.get("system"),
                request_messages=injected.get("messages"),
                response_payload=payload_map,
                latency_ms=latency_ms,
                streamed=False,
                status=upstream.status_code,
                error=None if upstream.status_code < 400 else "upstream error",
                skill=skill,
                tenant=tenant,
                owned_by=owned_by,
                trajectory_id=trajectory_id,
            ),
        )

        # Error/normal passthrough: surface the upstream status + body unchanged.
        if payload_map is not None:
            return JSONResponse(payload_map, status_code=upstream.status_code)
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    return Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/v1/messages", messages, methods=["POST"]),
        ]
    )


def _error(status: int, error_type: str, message: str) -> JSONResponse:
    """Render an Anthropic-shaped error envelope with the given HTTP status."""
    return JSONResponse(
        {"type": "error", "error": {"type": error_type, "message": message}},
        status_code=status,
    )


def serve(host: str = "127.0.0.1", port: int = 8088) -> None:  # pragma: no cover - launches a server
    """Launch the gateway with uvicorn (the ``loom proxy serve`` entry point).

    Reads the server-side ``ANTHROPIC_API_KEY`` (the real vendor key) and the
    accepted ``LOOM_API_KEY`` / ``LOOM_API_KEYS`` from the environment. Fails fast
    with an actionable message if either is missing, since the gateway can neither
    forward nor authenticate without them.

    Args:
        host: Bind address (default ``127.0.0.1`` — loopback only until hosted).
        port: Bind port (default ``8088``).
    """
    import uvicorn

    cfg = ProxyConfig.from_env()
    if not cfg.upstream_key:
        raise SystemExit(
            "error: the Loom gateway needs the server-side ANTHROPIC_API_KEY "
            "(the real vendor key it forwards with). Export it and re-run:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "Also set the Loom key callers must present:\n"
            "  export LOOM_API_KEY=loom-..."
        )
    if not cfg.allowed_keys:
        raise SystemExit(
            "error: the Loom gateway has no LOOM_API_KEY / LOOM_API_KEYS set, so "
            "it would reject every caller. Export the key callers must present:\n"
            "  export LOOM_API_KEY=loom-..."
        )

    app = create_app(cfg)
    uvicorn.run(app, host=host, port=port)


__all__ = [
    "LOOM_SYSTEM_PROMPT",
    "ProxyConfig",
    "ProxyAuthError",
    "authenticate",
    "extract_caller_key",
    "inject_loom_system",
    "build_log_row",
    "log_call",
    "create_app",
    "serve",
]
