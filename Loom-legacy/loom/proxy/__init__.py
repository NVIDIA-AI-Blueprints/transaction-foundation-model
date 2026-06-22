"""The Loom proxy gateway — the central LLM data-collection seam (the moat path).

This package hosts Loom's **own LLM gateway**: a thin Anthropic-passthrough
server that Loom's clients call with a Loom-issued ``LOOM_API_KEY`` instead of a
real vendor key. Under the hood (v0) the gateway is *vanilla Anthropic plus
Loom's injected prompts*, but it **logs every call centrally** — request
messages + system, response text + usage, model, latency, and tenant/owner
tags — to a single JSONL corpus. That central capture is the moat: the training
fuel for Loom's eventual distilled small models.

Why an Anthropic passthrough (and not an OpenAI-shaped gateway): the
``loom-proxy`` model provider (see :mod:`loom.providers.model.loom_proxy`) keeps
the ``claude-`` model prefix so AIDE routes to its native Anthropic backend.
Pointing that backend at this gateway (``ANTHROPIC_BASE_URL = loom_api_base``)
means native tool use — and the feedback judge — work with **no**
OpenAI<->Anthropic translation. The gateway speaks the exact
``POST /v1/messages`` shape the Anthropic SDK already emits.

The pieces (all defined in :mod:`loom.proxy.server`):

* :class:`~loom.proxy.server.ProxyConfig` — the server-side, env-derived
  settings (accepted Loom key(s), the real upstream key, the upstream URL, the
  JSONL log path, default tenant/owner tags). The secret material lives only
  here, in the server process, read from the environment.
* :func:`~loom.proxy.server.create_app` — builds the Starlette ASGI app
  (``POST /v1/messages`` + ``GET /healthz``). Accepts an injectable ``forward``
  coroutine so tests exercise the full handler with no network.
* :func:`~loom.proxy.server.serve` — launches the app under uvicorn (wired to
  ``loom proxy serve``); fails fast if the server-side vendor key is missing.
* The pure, network-free decision helpers
  (:func:`~loom.proxy.server.authenticate`,
  :func:`~loom.proxy.server.inject_loom_system`,
  :func:`~loom.proxy.server.build_log_row`, :func:`~loom.proxy.server.log_call`)
  — unit-tested directly so the data-collection logic is verifiable offline.

Secret discipline (mirrors :mod:`loom.config`): the real upstream key
(``ANTHROPIC_API_KEY``) and the allowed Loom key(s) (``LOOM_API_KEY`` /
``LOOM_API_KEYS``) are read from the environment **on the server** and **never**
stored on a :class:`~loom.config.LoomConfig` or written to the call log — the log
row records request/response content and metadata, never key material.

``loom`` core never imports this package; only the CLI's ``proxy serve`` path and
the tests do, so the ASGI/HTTP deps (starlette / uvicorn / httpx) imported by
:mod:`loom.proxy.server` cannot break ``loom`` core import.
"""

from __future__ import annotations

from loom.proxy.server import (
    LOOM_SYSTEM_PROMPT,
    ProxyAuthError,
    ProxyConfig,
    authenticate,
    build_log_row,
    create_app,
    extract_caller_key,
    inject_loom_system,
    log_call,
    serve,
)

__all__ = [
    "LOOM_SYSTEM_PROMPT",
    "ProxyAuthError",
    "ProxyConfig",
    "authenticate",
    "build_log_row",
    "create_app",
    "extract_caller_key",
    "inject_loom_system",
    "log_call",
    "serve",
]
