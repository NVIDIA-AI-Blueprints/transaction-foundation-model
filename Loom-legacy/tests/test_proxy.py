"""Proxy tests: the loom-proxy provider + the gateway server (no real network).

Two surfaces are covered, both offline:

1. The ``loom-proxy`` **model provider** (:mod:`loom.providers.model.loom_proxy`)
   — that ``resolve`` keeps the ``claude-`` prefix and points at the gateway,
   that ``prepare_env`` sets ``ANTHROPIC_BASE_URL`` and copies ``LOOM_API_KEY`` ->
   ``ANTHROPIC_API_KEY``, and that ``preflight`` flags a missing ``LOOM_API_KEY``.
   These run against a throwaway env dict so they never touch the real process
   environment, mirroring ``test_model_providers.py``.

2. The **gateway server** (:mod:`loom.proxy.server`) — the pure decision helpers
   (auth, prompt injection, log-row build, log append) directly, plus the full
   Starlette app driven through a ``TestClient`` with a **fake forwarder injected**
   (no real upstream). The fake lets us assert: Loom's system prompt is injected
   ahead of the caller's system, a JSONL row with the right fields is written to
   the call log, the upstream sees the server-side vendor key (not the caller's
   Loom key), a bad/missing Loom key is rejected with 401 before any forward, and
   a streamed call is passed through and still logged.

No test ships a real key, invokes a real model, or hits the network: the Loom and
vendor keys are fakes set on a throwaway :class:`~loom.proxy.server.ProxyConfig`,
and the upstream forward is replaced by a coroutine returning a synthetic
``httpx.Response``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from loom.config import LoomConfig
from loom.proxy import server as proxy_server
from loom.proxy.server import ProxyAuthError, ProxyConfig
from loom.registry import get_model

# ---------------------------------------------------------------------------
# 1) The loom-proxy model provider.
# ---------------------------------------------------------------------------


def _proxy_provider(**cfg_kwargs: object):
    """Resolve the loom-proxy model provider, instantiated from a config."""
    cfg = LoomConfig(**cfg_kwargs)
    return get_model("loom-proxy")(cfg)


def test_loom_proxy_registered_and_named() -> None:
    """loom-proxy resolves from the registry to a provider named 'loom-proxy'."""
    assert get_model("loom-proxy").name == "loom-proxy"


def test_loom_proxy_resolve_keeps_claude_prefix_and_points_at_gateway() -> None:
    """resolve(): claude- model, base_url = loom_api_base, key_env = LOOM_API_KEY, proxy kind."""
    provider = _proxy_provider(
        code_model="claude-sonnet-4-5",
        feedback_model="claude-opus-4-1",
        loom_api_base="http://127.0.0.1:8088",
    )
    code = provider.resolve("code")
    feedback = provider.resolve("feedback")

    assert code.model_name == "claude-sonnet-4-5"
    # claude- prefix preserved so AIDE routes to its native Anthropic backend.
    assert code.model_name.startswith("claude-")
    assert feedback.model_name == "claude-opus-4-1"
    assert code.base_url == "http://127.0.0.1:8088"
    assert code.key_env == "LOOM_API_KEY"
    assert code.kind == "proxy"
    # Anthropic passthrough -> judge-capable (native tool use) like anthropic-api.
    assert code.judge_capable is True
    assert feedback.judge_capable is True


def test_loom_proxy_prepare_env_redirects_anthropic_at_gateway_with_loom_key() -> None:
    """prepare_env(): ANTHROPIC_BASE_URL = gateway, ANTHROPIC_API_KEY = LOOM_API_KEY."""
    provider = _proxy_provider(loom_api_base="http://127.0.0.1:8088")
    env: dict[str, str] = {"LOOM_API_KEY": "loom-fake-123"}

    provider.prepare_env(env)

    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8088"
    # The Loom key is routed into the var the native Anthropic client reads, so the
    # backend authenticates to the GATEWAY with the Loom key (the gateway swaps in
    # the real vendor key server-side).
    assert env["ANTHROPIC_API_KEY"] == "loom-fake-123"
    # This is an Anthropic-passthrough route: it must NOT set OPENAI_BASE_URL.
    assert "OPENAI_BASE_URL" not in env


def test_loom_proxy_prepare_env_without_key_still_sets_base_url() -> None:
    """prepare_env() never fabricates a key: missing LOOM_API_KEY -> only base_url set."""
    provider = _proxy_provider(loom_api_base="http://127.0.0.1:8088")
    env: dict[str, str] = {}

    provider.prepare_env(env)

    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8088"
    assert "ANTHROPIC_API_KEY" not in env  # no key invented


def test_loom_proxy_preflight_flags_missing_loom_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """preflight(): hint when LOOM_API_KEY missing, empty once set."""
    monkeypatch.delenv("LOOM_API_KEY", raising=False)
    provider = _proxy_provider()

    hints = provider.preflight("code")
    assert hints and any("LOOM_API_KEY" in h for h in hints)

    monkeypatch.setenv("LOOM_API_KEY", "loom-fake")
    assert provider.preflight("code") == []


# ---------------------------------------------------------------------------
# 2a) The gateway server — pure decision helpers (no app, no network).
# ---------------------------------------------------------------------------


def test_authenticate_accepts_allowlisted_key_and_rejects_others() -> None:
    """authenticate(): accepts a listed key (x-api-key or Bearer); rejects the rest."""
    allowed = frozenset({"loom-good", "loom-also"})

    # x-api-key path.
    assert proxy_server.authenticate({"x-api-key": "loom-good"}, allowed) == "loom-good"
    # Bearer path.
    assert (
        proxy_server.authenticate({"authorization": "Bearer loom-also"}, allowed)
        == "loom-also"
    )
    # Wrong key.
    with pytest.raises(ProxyAuthError):
        proxy_server.authenticate({"x-api-key": "loom-wrong"}, allowed)
    # No key presented.
    with pytest.raises(ProxyAuthError):
        proxy_server.authenticate({}, allowed)
    # No key configured on the server -> refuse everyone.
    with pytest.raises(ProxyAuthError):
        proxy_server.authenticate({"x-api-key": "loom-good"}, frozenset())


def test_extract_caller_key_handles_both_header_shapes() -> None:
    """extract_caller_key(): x-api-key wins; Bearer parsed; else None."""
    assert proxy_server.extract_caller_key({"x-api-key": " k1 "}) == "k1"
    assert proxy_server.extract_caller_key({"authorization": "Bearer k2"}) == "k2"
    assert proxy_server.extract_caller_key({}) is None


def test_inject_loom_system_prepends_loom_block_to_string_system() -> None:
    """inject_loom_system(): a string system becomes [loom_block, original_block]."""
    body = {"model": "claude-x", "system": "Be terse.", "messages": []}
    out = proxy_server.inject_loom_system(body)

    assert isinstance(out["system"], list)
    assert out["system"][0]["text"] == proxy_server.LOOM_SYSTEM_PROMPT
    assert out["system"][1]["text"] == "Be terse."
    # The input body is not mutated.
    assert body["system"] == "Be terse."


def test_inject_loom_system_prepends_to_block_list_and_handles_absent() -> None:
    """inject_loom_system(): list system is prepended; absent system -> just Loom block."""
    body = {"system": [{"type": "text", "text": "A"}]}
    out = proxy_server.inject_loom_system(body)
    assert [b["text"] for b in out["system"]] == [proxy_server.LOOM_SYSTEM_PROMPT, "A"]

    bare = proxy_server.inject_loom_system({"model": "claude-x"})
    assert bare["system"] == [{"type": "text", "text": proxy_server.LOOM_SYSTEM_PROMPT}]


def test_build_log_row_captures_fields_and_no_secrets() -> None:
    """build_log_row(): records model/system/messages/usage/tags; carries no key material."""
    response = {
        "content": [{"type": "text", "text": "answer"}],
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }
    row = proxy_server.build_log_row(
        model="claude-sonnet-4-5",
        request_system=[{"type": "text", "text": proxy_server.LOOM_SYSTEM_PROMPT}],
        request_messages=[{"role": "user", "content": "hi"}],
        response_payload=response,
        latency_ms=12.345,
        streamed=False,
        status=200,
        error=None,
        skill="loom-optimize",
        tenant="tala",
        owned_by="tala",
    )

    assert row["model"] == "claude-sonnet-4-5"
    assert row["response_text"] == "answer"
    assert row["usage"] == {"input_tokens": 10, "output_tokens": 2}
    assert row["streamed"] is False
    assert row["status"] == 200
    assert row["skill"] == "loom-optimize"
    # IP-boundary tags carried through (mirrors loom.corpus owned_by semantics).
    assert row["tenant"] == "tala"
    assert row["owned_by"] == "tala"
    # The row schema has no field for key material: build_log_row is given a model
    # name, system/messages, the response payload, and tags — never a key — so a
    # key can never appear. (The "loom-optimize" skill tag is provenance, not a key.)
    assert "key" not in json.dumps(row).lower()
    assert "sk-ant" not in json.dumps(row)


def test_log_call_appends_jsonl_and_creates_parent(tmp_path: Path) -> None:
    """log_call(): creates the parent dir and appends one JSON line per call."""
    log_path = tmp_path / "nested" / "proxy_calls.jsonl"
    proxy_server.log_call(str(log_path), {"a": 1})
    proxy_server.log_call(str(log_path), {"a": 2})

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["a"] for line in lines] == [1, 2]


# ---------------------------------------------------------------------------
# 2b) The full app via TestClient with a FAKE forwarder (no network).
# ---------------------------------------------------------------------------


def _proxy_config(log_path: Path, **overrides: Any) -> ProxyConfig:
    """Build a ProxyConfig with fake keys + a temp log path (no env touched)."""
    base = dict(
        allowed_keys=frozenset({"loom-good"}),
        upstream_key="sk-ant-server-secret",
        upstream_url="https://api.anthropic.com/v1/messages",
        log_path=str(log_path),
    )
    base.update(overrides)
    return ProxyConfig(**base)


def _fake_unary_forwarder(captured: dict[str, Any]):
    """Return a forwarder coroutine that captures its call and returns a fixed reply."""

    async def _forward(*, url, json_body, headers, stream):  # noqa: ANN001
        captured["url"] = url
        captured["json_body"] = json_body
        captured["headers"] = dict(headers)
        captured["stream"] = stream
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "model": json_body.get("model", "claude-x"),
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
        )

    return _forward


def _client(cfg: ProxyConfig, forward):
    """Build a Starlette TestClient over the gateway app."""
    from starlette.testclient import TestClient

    return TestClient(proxy_server.create_app(cfg, forward=forward))


def test_healthz_ok(tmp_path: Path) -> None:
    """GET /healthz returns 200 with a small status payload, no auth required."""
    client = _client(_proxy_config(tmp_path / "log.jsonl"), _fake_unary_forwarder({}))
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_messages_injects_prompt_uses_vendor_key_and_logs(tmp_path: Path) -> None:
    """POST /v1/messages: prompt injected upstream, vendor key (not Loom key) used, row logged."""
    captured: dict[str, Any] = {}
    log_path = tmp_path / "proxy_calls.jsonl"
    client = _client(_proxy_config(log_path), _fake_unary_forwarder(captured))

    resp = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "loom-good",
            "x-loom-skill": "loom-optimize",
            "x-loom-tenant": "tala",
            "x-loom-owned-by": "tala",
        },
        json={
            "model": "claude-sonnet-4-5",
            "system": "Be terse.",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
        },
    )

    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "ok"

    # (a) Loom's system prompt was injected ahead of the caller's system.
    sent_system = captured["json_body"]["system"]
    assert sent_system[0]["text"] == proxy_server.LOOM_SYSTEM_PROMPT
    assert sent_system[1]["text"] == "Be terse."

    # (b) The upstream saw the SERVER-SIDE vendor key, never the caller's Loom key.
    assert captured["headers"]["x-api-key"] == "sk-ant-server-secret"
    assert "loom-good" not in json.dumps(captured["headers"])

    # (c) Exactly one JSONL row was written with the right fields (and no secrets).
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["model"] == "claude-sonnet-4-5"
    assert row["response_text"] == "ok"
    assert row["usage"] == {"input_tokens": 5, "output_tokens": 1}
    assert row["status"] == 200
    assert row["streamed"] is False
    assert row["skill"] == "loom-optimize"
    assert row["tenant"] == "tala"
    assert row["owned_by"] == "tala"
    assert "sk-ant-server-secret" not in lines[0]
    assert "loom-good" not in lines[0]


def test_messages_logs_trajectory_id_from_header(tmp_path: Path) -> None:
    """The x-loom-trajectory header is stamped onto the proxy_calls row (additive).

    The telemetry-layer join key: a caller that drives one trajectory passes its id
    as ``x-loom-trajectory`` so :func:`loom.telemetry.assemble_trajectory` can
    stitch this LLM call into the owning trajectory. A call WITHOUT the header still
    logs (with ``trajectory_id == None``), proving the field is purely additive.
    """
    log_path = tmp_path / "proxy_calls.jsonl"
    client = _client(_proxy_config(log_path), _fake_unary_forwarder({}))

    resp = client.post(
        "/v1/messages",
        headers={"x-api-key": "loom-good", "x-loom-trajectory": "loom-exp-7"},
        json={"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["trajectory_id"] == "loom-exp-7"


def test_messages_without_trajectory_header_logs_none(tmp_path: Path) -> None:
    """No trajectory header (and no env) -> the row carries trajectory_id=None."""
    log_path = tmp_path / "proxy_calls.jsonl"
    client = _client(_proxy_config(log_path), _fake_unary_forwarder({}))

    resp = client.post(
        "/v1/messages",
        headers={"x-api-key": "loom-good"},
        json={"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["trajectory_id"] is None


def test_build_log_row_trajectory_id_defaults_none() -> None:
    """build_log_row(): trajectory_id is an additive, defaulted field."""
    row = proxy_server.build_log_row(
        model="claude-x",
        request_system=None,
        request_messages=[],
        response_payload=None,
        latency_ms=1.0,
        streamed=False,
        status=200,
        error=None,
        skill=None,
        tenant="default",
        owned_by="general",
    )
    assert row["trajectory_id"] is None
    row2 = proxy_server.build_log_row(
        model="claude-x",
        request_system=None,
        request_messages=[],
        response_payload=None,
        latency_ms=1.0,
        streamed=False,
        status=200,
        error=None,
        skill=None,
        tenant="default",
        owned_by="general",
        trajectory_id="t-9",
    )
    assert row2["trajectory_id"] == "t-9"


def test_messages_rejects_bad_key_without_forwarding(tmp_path: Path) -> None:
    """A wrong/missing Loom key -> 401 and the upstream is never called or logged."""
    captured: dict[str, Any] = {}
    log_path = tmp_path / "proxy_calls.jsonl"
    client = _client(_proxy_config(log_path), _fake_unary_forwarder(captured))

    resp = client.post(
        "/v1/messages",
        headers={"x-api-key": "loom-WRONG"},
        json={"model": "claude-x", "messages": []},
    )

    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"
    # The forwarder was never invoked, and nothing was logged.
    assert captured == {}
    assert not log_path.exists()


def test_messages_missing_server_vendor_key_returns_500(tmp_path: Path) -> None:
    """An authenticated caller still gets 500 if the server has no upstream key."""
    captured: dict[str, Any] = {}
    cfg = _proxy_config(tmp_path / "proxy_calls.jsonl", upstream_key="")
    client = _client(cfg, _fake_unary_forwarder(captured))

    resp = client.post(
        "/v1/messages",
        headers={"x-api-key": "loom-good"},
        json={"model": "claude-x", "messages": []},
    )
    assert resp.status_code == 500
    assert captured == {}  # never forwarded


def test_messages_streaming_passthrough_and_logs(tmp_path: Path) -> None:
    """A streamed call passes the SSE bytes through and still logs one row."""

    class _FakeStream(httpx.AsyncByteStream):
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = chunks

        async def __aiter__(self):
            for chunk in self._chunks:
                yield chunk

        async def aclose(self) -> None:
            return None

    captured: dict[str, Any] = {}

    async def _stream_forward(*, url, json_body, headers, stream):  # noqa: ANN001
        captured["stream"] = stream
        captured["headers"] = dict(headers)
        sse = [
            b'event: message_start\ndata: {"type":"message_start"}\n\n',
            b'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n',
            b"event: message_stop\ndata: {}\n\n",
        ]
        return httpx.Response(
            200,
            stream=_FakeStream(sse),
            headers={"content-type": "text/event-stream"},
        )

    log_path = tmp_path / "proxy_calls.jsonl"
    client = _client(_proxy_config(log_path), _stream_forward)

    resp = client.post(
        "/v1/messages",
        headers={"x-api-key": "loom-good"},
        json={"model": "claude-x", "messages": [], "stream": True},
    )

    assert resp.status_code == 200
    # The handler marked the upstream call as a stream and used the vendor key.
    assert captured["stream"] is True
    assert captured["headers"]["x-api-key"] == "sk-ant-server-secret"
    # The SSE bytes were passed through verbatim.
    assert b"message_start" in resp.content
    # And exactly one row was logged, marked streamed.
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["streamed"] is True
    assert row["status"] == 200


def test_messages_upstream_failure_is_passed_through_and_logged(tmp_path: Path) -> None:
    """An upstream transport error -> 502 to the caller and a logged error row."""

    async def _boom(*, url, json_body, headers, stream):  # noqa: ANN001
        raise httpx.ConnectError("upstream down")

    log_path = tmp_path / "proxy_calls.jsonl"
    client = _client(_proxy_config(log_path), _boom)

    resp = client.post(
        "/v1/messages",
        headers={"x-api-key": "loom-good"},
        json={"model": "claude-x", "messages": []},
    )

    assert resp.status_code == 502
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["status"] == 502
    assert row["error"] and "ConnectError" in row["error"]


def test_proxy_config_from_env_reads_keys_and_anchors_log_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ProxyConfig.from_env(): reads keys from env; log path anchored absolute."""
    env = {
        "LOOM_API_KEY": "loom-a",
        "LOOM_API_KEYS": "loom-b, loom-c",
        "ANTHROPIC_API_KEY": "sk-ant-real",
        "LOOM_PROXY_LOG_PATH": "rel/proxy_calls.jsonl",
    }
    cfg = ProxyConfig.from_env(env)

    assert cfg.allowed_keys == frozenset({"loom-a", "loom-b", "loom-c"})
    assert cfg.upstream_key == "sk-ant-real"
    # The log path is anchored absolute by LoomConfig.load (like corpus/learnings).
    import os

    assert os.path.isabs(cfg.log_path)
    assert cfg.log_path.endswith("rel/proxy_calls.jsonl")
