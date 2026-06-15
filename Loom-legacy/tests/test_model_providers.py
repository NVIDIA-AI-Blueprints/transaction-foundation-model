"""Model-provider tests: registry resolution, routes, env materialization, preflight.

These are pure-Python: the model adapters (``loom.providers.model.*``) import only
``loom`` core at module top — none import AIDE eagerly (AIDE is touched lazily only
by the ``cli-bridge`` dispatch override and the optional review-schema helper). So
registry resolution, :meth:`~loom.providers.model.ModelProvider.resolve`,
:meth:`~loom.providers.model.ModelProvider.prepare_env`, and
:meth:`~loom.providers.model.ModelProvider.preflight` all run without AIDE or any
live LLM. The few assertions that genuinely need AIDE installed use
:func:`pytest.importorskip` and never call out to a model.

No test ever invokes a model, ships a key, or relies on real credentials: keys are
faked, and ``prepare_env`` is exercised against a throwaway dict so it cannot touch
the real process environment. Credential/login presence is simulated with
``monkeypatch`` so the same test asserts both the "missing" and "set" branches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest

from loom.config import LoomConfig
from loom.providers.model import ModelProvider, ModelRoute
from loom.registry import MODEL_PROVIDERS, get_model

# Every built-in model provider name the contract requires, by kind.
_API_PROVIDERS = ("anthropic-api", "openai-api")
_OPENAI_COMPAT_PROVIDERS = ("openrouter", "nim", "openai-compat")
_CLI_BRIDGE_PROVIDERS = ("claude-subscription", "codex-subscription")
_ALL_PROVIDERS = _API_PROVIDERS + _OPENAI_COMPAT_PROVIDERS + _CLI_BRIDGE_PROVIDERS

# The credential env vars the providers may read, plus the vars the OpenAI-compatible
# providers materialize. Cleared before each test so presence is fully controlled.
_RELEVANT_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_HTTP_REFERER",
    "OPENROUTER_X_TITLE",
    "NVIDIA_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "LOOM_MODEL_BASE_URL",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every credential/routing env var so presence is test-controlled.

    Without this, a developer's exported ``ANTHROPIC_API_KEY`` (etc.) would leak
    into the "missing credential" assertions and flake them.
    """
    for name in _RELEVANT_ENV:
        monkeypatch.delenv(name, raising=False)


def _provider(name: str, **cfg_kwargs: object) -> ModelProvider:
    """Resolve a model provider by name and instantiate it from a config."""
    cfg = LoomConfig(**cfg_kwargs)
    return get_model(name)(cfg)


# ---------------------------------------------------------------------------
# Registry resolution.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _ALL_PROVIDERS)
def test_registry_resolves_each_provider(name: str) -> None:
    """Each contract provider name resolves to a ``ModelProvider`` subclass.

    Importing ``loom`` self-registers the built-in model adapters (each guarded so
    a missing optional dependency only drops that one). The resolved class must be
    a ``ModelProvider`` whose ``name`` attribute matches its registry key.
    """
    cls = get_model(name)
    assert issubclass(cls, ModelProvider)
    assert cls.name == name
    # And it appears in the public registry dict.
    assert MODEL_PROVIDERS[name] is cls


def test_default_provider_is_anthropic_api() -> None:
    """The unconfigured default keeps Loom's historical Claude behavior."""
    cfg = LoomConfig()
    assert cfg.code_provider == "anthropic-api"
    assert cfg.feedback_provider == "anthropic-api"


def test_unknown_provider_raises_keyerror() -> None:
    """An unregistered model name raises ``KeyError`` listing what's available."""
    with pytest.raises(KeyError):
        get_model("no-such-model-provider")


@pytest.mark.parametrize("name", _ALL_PROVIDERS)
@pytest.mark.parametrize("role", ("code", "feedback"))
def test_resolve_returns_a_modelroute(name: str, role: str, clean_env: None) -> None:
    """``resolve(role)`` returns a well-formed ``ModelRoute`` for every provider/role."""
    route = _provider(name).resolve(role)
    assert isinstance(route, ModelRoute)
    assert isinstance(route.model_name, str) and route.model_name
    assert route.kind in {"api", "openai-compat", "cli-bridge"}
    assert isinstance(route.judge_capable, bool)
    assert route.base_url is None or isinstance(route.base_url, str)
    assert route.key_env is None or isinstance(route.key_env, str)


# ---------------------------------------------------------------------------
# Per-provider route fields.
# ---------------------------------------------------------------------------


def test_anthropic_route_keeps_claude_prefix_and_native_kind(clean_env: None) -> None:
    """anthropic-api: native ``api`` route on the configured ``claude-`` model, no base_url."""
    provider = _provider(
        "anthropic-api", code_model="claude-sonnet-4-5", feedback_model="claude-opus-4-1"
    )
    code = provider.resolve("code")
    feedback = provider.resolve("feedback")

    assert code.model_name == "claude-sonnet-4-5"
    assert code.model_name.startswith("claude-")  # so AIDE routes to the anthropic backend
    assert feedback.model_name == "claude-opus-4-1"
    assert code.kind == "api"
    assert code.base_url is None  # native Anthropic ignores OPENAI_BASE_URL
    assert code.key_env == "ANTHROPIC_API_KEY"
    assert code.judge_capable is True
    assert feedback.judge_capable is True


def test_openai_route_leaves_base_url_unset(clean_env: None) -> None:
    """openai-api: native ``api`` route, ``base_url=None`` (real OpenAI hardcodes its URL)."""
    provider = _provider(
        "openai-api", code_model="gpt-4o", feedback_model="o3"
    )
    code = provider.resolve("code")
    assert code.model_name == "gpt-4o"
    assert code.kind == "api"
    assert code.base_url is None  # MUST stay unset; real OpenAI ignores OPENAI_BASE_URL
    assert code.key_env == "OPENAI_API_KEY"
    assert code.judge_capable is True
    assert provider.resolve("feedback").model_name == "o3"


def test_openrouter_route_uses_openai_compat_path(clean_env: None) -> None:
    """openrouter: ``openai-compat`` kind pointed at the OpenRouter base URL, key_env set."""
    provider = _provider(
        "openrouter",
        code_model="anthropic/claude-sonnet-4.5",
        feedback_model="anthropic/claude-sonnet-4.5",
    )
    code = provider.resolve("code")
    assert code.kind == "openai-compat"
    assert code.base_url == "https://openrouter.ai/api/v1"
    assert code.key_env == "OPENROUTER_API_KEY"
    assert "/" in code.model_name  # provider/model slug form
    # The code role is assumed tool-capable.
    assert code.judge_capable is True


def test_nim_route_uses_configured_base_url(clean_env: None) -> None:
    """nim: ``openai-compat`` route; ``cfg.nim_base_url`` overrides the hosted default."""
    default_provider = _provider("nim", code_model="meta/llama-3.1-70b-instruct")
    default_route = default_provider.resolve("code")
    assert default_route.kind == "openai-compat"
    assert default_route.base_url == "https://integrate.api.nvidia.com/v1"
    assert default_route.key_env == "NVIDIA_API_KEY"

    custom = _provider(
        "nim",
        code_model="meta/llama-3.1-70b-instruct",
        nim_base_url="https://nim.internal.example/v1",
    )
    assert custom.resolve("code").base_url == "https://nim.internal.example/v1"


def test_openai_compat_route_uses_model_base_url(clean_env: None) -> None:
    """openai-compat: ``base_url`` from ``cfg.model_base_url``; passthrough OPENAI_API_KEY."""
    provider = _provider(
        "openai-compat",
        code_model="local-model",
        model_base_url="http://localhost:11434/v1",
    )
    route = provider.resolve("code")
    assert route.kind == "openai-compat"
    assert route.base_url == "http://localhost:11434/v1"
    assert route.key_env == "OPENAI_API_KEY"


@pytest.mark.parametrize(
    "name, expected_model",
    [
        ("claude-subscription", "claude-code-subscription"),
        ("codex-subscription", "codex-mini-latest"),
    ],
)
def test_cli_bridge_routes_are_sentinels_without_keys(
    name: str, expected_model: str, clean_env: None
) -> None:
    """cli-bridge: a sentinel model name (routing to AIDE's slot), no ``key_env``."""
    route = _provider(name).resolve("feedback")
    assert route.kind == "cli-bridge"
    assert route.model_name == expected_model
    assert route.key_env is None  # the CLI owns authentication; Loom holds no key
    assert route.base_url is None  # must NOT set OPENAI_BASE_URL for the sentinel
    assert route.judge_capable is True  # text coerced into the judge's JSON shape


# ---------------------------------------------------------------------------
# prepare_env: materialize routing knobs into a THROWAWAY dict (never os.environ).
# ---------------------------------------------------------------------------


def test_openrouter_prepare_env_sets_base_url_and_copies_key() -> None:
    """openrouter.prepare_env: OPENAI_BASE_URL set + OPENROUTER_API_KEY copied to OPENAI_API_KEY."""
    provider = _provider("openrouter")
    env: dict[str, str] = {"OPENROUTER_API_KEY": "sk-or-fake-123"}

    provider.prepare_env(env)

    assert env["OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"
    # The user's OpenRouter key is moved into the var AIDE's OpenAI client reads.
    assert env["OPENAI_API_KEY"] == "sk-or-fake-123"
    # The original key is left in place (only moved/copied, never deleted).
    assert env["OPENROUTER_API_KEY"] == "sk-or-fake-123"


def test_openrouter_prepare_env_without_key_still_sets_base_url() -> None:
    """openrouter.prepare_env never invents a key: missing key -> only base_url set."""
    provider = _provider("openrouter")
    env: dict[str, str] = {}

    provider.prepare_env(env)

    assert env["OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert "OPENAI_API_KEY" not in env  # no key fabricated


def test_nim_prepare_env_sets_base_url_and_copies_key() -> None:
    """nim.prepare_env: OPENAI_BASE_URL = NIM endpoint + NVIDIA_API_KEY copied to OPENAI_API_KEY."""
    provider = _provider("nim", nim_base_url="https://nim.internal.example/v1")
    env: dict[str, str] = {"NVIDIA_API_KEY": "nvapi-fake-456"}

    provider.prepare_env(env)

    assert env["OPENAI_BASE_URL"] == "https://nim.internal.example/v1"
    assert env["OPENAI_API_KEY"] == "nvapi-fake-456"
    assert env["NVIDIA_API_KEY"] == "nvapi-fake-456"


def test_nim_prepare_env_defaults_to_hosted_endpoint() -> None:
    """nim.prepare_env with no configured base URL falls back to the hosted default."""
    provider = _provider("nim")
    env: dict[str, str] = {}

    provider.prepare_env(env)

    assert env["OPENAI_BASE_URL"] == "https://integrate.api.nvidia.com/v1"


def test_openai_compat_prepare_env_sets_base_url_from_config() -> None:
    """openai-compat.prepare_env materializes OPENAI_BASE_URL from cfg.model_base_url."""
    provider = _provider("openai-compat", model_base_url="http://localhost:8000/v1")
    env: dict[str, str] = {"OPENAI_API_KEY": "sk-local-dummy"}

    provider.prepare_env(env)

    assert env["OPENAI_BASE_URL"] == "http://localhost:8000/v1"
    # OPENAI_API_KEY is passed through unchanged (may be a dummy local token).
    assert env["OPENAI_API_KEY"] == "sk-local-dummy"


def test_native_api_prepare_env_never_sets_openai_base_url() -> None:
    """anthropic-api / openai-api MUST NOT set OPENAI_BASE_URL (would mis-route)."""
    for name in _API_PROVIDERS:
        env: dict[str, str] = {}
        _provider(name).prepare_env(env)
        assert "OPENAI_BASE_URL" not in env, name


def test_cli_bridge_prepare_env_is_noop() -> None:
    """cli-bridge prepare_env materializes nothing (the local CLI owns its auth)."""
    for name in _CLI_BRIDGE_PROVIDERS:
        env: dict[str, str] = {}
        _provider(name).prepare_env(env)
        assert env == {}, name


# ---------------------------------------------------------------------------
# preflight: hints when creds/login absent; empty when the expected env is set.
# ---------------------------------------------------------------------------


def _mentions(hints: Iterable[str], needle: str) -> bool:
    return any(needle in h for h in hints)


def test_anthropic_preflight_hint_then_clean(
    monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """anthropic-api preflight: hint when ANTHROPIC_API_KEY missing, empty once set."""
    provider = _provider("anthropic-api")

    hints = provider.preflight("code")
    assert hints and _mentions(hints, "ANTHROPIC_API_KEY")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    assert provider.preflight("code") == []


def test_openai_preflight_hint_then_clean(
    monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """openai-api preflight: hint when OPENAI_API_KEY missing, empty once set."""
    provider = _provider("openai-api", code_model="gpt-4o", feedback_model="gpt-4o")

    hints = provider.preflight("feedback")
    assert hints and _mentions(hints, "OPENAI_API_KEY")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    assert provider.preflight("feedback") == []


def test_nim_preflight_hint_then_clean(
    monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """nim preflight: hint when NVIDIA_API_KEY missing, empty once set (code role)."""
    provider = _provider("nim", code_model="meta/llama-3.1-70b-instruct")

    hints = provider.preflight("code")
    assert hints and _mentions(hints, "NVIDIA_API_KEY")

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-fake")
    assert provider.preflight("code") == []


def test_nim_preflight_warns_when_feedback_not_tool_capable(
    monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """nim preflight warns on the feedback role when the served model lacks tools."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-fake")
    provider = get_model("nim")(LoomConfig(feedback_model="meta/llama-3.1-70b-instruct"), judge_capable=False)

    hints = provider.preflight("feedback")
    assert hints and _mentions(hints, "tool")
    # The code role has no such warning.
    assert provider.preflight("code") == []


def test_openai_compat_preflight_flags_missing_endpoint_and_key(clean_env: None) -> None:
    """openai-compat preflight flags a missing base URL and a missing key."""
    # No model_base_url, no OPENAI_BASE_URL, no OPENAI_API_KEY in the clean env.
    provider = _provider("openai-compat", code_model="local-model")
    hints = provider.preflight("code")
    assert _mentions(hints, "endpoint") or _mentions(hints, "BASE_URL")
    assert _mentions(hints, "OPENAI_API_KEY")


def test_openai_compat_preflight_clean_when_configured(
    monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """openai-compat preflight is empty once base URL + key are present."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-local-dummy")
    provider = _provider(
        "openai-compat", code_model="local-model", model_base_url="http://localhost:11434/v1"
    )
    assert provider.preflight("code") == []


def test_openrouter_preflight_hint_for_missing_key(clean_env: None) -> None:
    """openrouter preflight hints about the missing OPENROUTER_API_KEY (code role)."""
    provider = _provider("openrouter", code_model="anthropic/claude-sonnet-4.5")
    hints = provider.preflight("code")
    assert _mentions(hints, "OPENROUTER_API_KEY")


def test_claude_subscription_preflight_hint_when_no_cli(
    monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """claude-subscription preflight hints when the `claude` binary is absent."""
    monkeypatch.setattr(
        "loom.providers.model.claude_subscription.shutil.which", lambda _name: None
    )
    hints = _provider("claude-subscription").preflight("code")
    assert hints and _mentions(hints, "claude")


def test_claude_subscription_preflight_hint_when_logged_out(
    monkeypatch: pytest.MonkeyPatch, clean_env: None, tmp_path: Path
) -> None:
    """With the CLI present but no login signal, preflight returns a login hint."""
    monkeypatch.setattr(
        "loom.providers.model.claude_subscription.shutil.which",
        lambda _name: "/usr/local/bin/claude",
    )
    # Point HOME at an empty dir so neither ~/.claude.json nor ~/.claude exists,
    # and ensure no OAuth token env var is set (clean_env cleared it).
    monkeypatch.setattr(
        "loom.providers.model.claude_subscription.Path.home", lambda: tmp_path
    )
    hints = _provider("claude-subscription").preflight("code")
    assert hints and (_mentions(hints, "login") or _mentions(hints, "setup-token"))


def test_claude_subscription_preflight_clean_with_oauth_token(
    monkeypatch: pytest.MonkeyPatch, clean_env: None, tmp_path: Path
) -> None:
    """A logged-in signal (OAuth token env) clears the login hint when the CLI exists."""
    monkeypatch.setattr(
        "loom.providers.model.claude_subscription.shutil.which",
        lambda _name: "/usr/local/bin/claude",
    )
    monkeypatch.setattr(
        "loom.providers.model.claude_subscription.Path.home", lambda: tmp_path
    )
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-fake")
    assert _provider("claude-subscription").preflight("code") == []


def test_codex_subscription_preflight_hint_when_no_cli(
    monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """codex-subscription preflight hints when the `codex` binary is absent."""
    monkeypatch.setattr(
        "loom.providers.model.codex_subscription.shutil.which", lambda _name: None
    )
    hints = _provider("codex-subscription").preflight("code")
    assert hints and _mentions(hints, "codex")


def test_codex_subscription_preflight_hint_when_no_auth_file(
    monkeypatch: pytest.MonkeyPatch, clean_env: None, tmp_path: Path
) -> None:
    """With the CLI present but no ~/.codex/auth.json, preflight hints about login."""
    monkeypatch.setattr(
        "loom.providers.model.codex_subscription.shutil.which",
        lambda _name: "/usr/local/bin/codex",
    )
    monkeypatch.setattr(
        "loom.providers.model.codex_subscription.Path.home", lambda: tmp_path
    )
    hints = _provider("codex-subscription").preflight("code")
    assert hints and _mentions(hints, "auth.json")


def test_codex_subscription_preflight_clean_with_auth_file(
    monkeypatch: pytest.MonkeyPatch, clean_env: None, tmp_path: Path
) -> None:
    """A present ~/.codex/auth.json clears the preflight when the CLI exists."""
    monkeypatch.setattr(
        "loom.providers.model.codex_subscription.shutil.which",
        lambda _name: "/usr/local/bin/codex",
    )
    monkeypatch.setattr(
        "loom.providers.model.codex_subscription.Path.home", lambda: tmp_path
    )
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "auth.json").write_text("{}", encoding="utf-8")
    assert _provider("codex-subscription").preflight("code") == []


# ---------------------------------------------------------------------------
# Judge-capability flag is exposed and respected by the per-role resolve.
# ---------------------------------------------------------------------------


def test_judge_capable_flag_exposed_on_every_route(clean_env: None) -> None:
    """Every provider exposes a boolean ``judge_capable`` on its feedback route."""
    for name in _ALL_PROVIDERS:
        route = _provider(name).resolve("feedback")
        assert isinstance(route.judge_capable, bool), name


def test_nim_feedback_judge_capable_is_configurable(clean_env: None) -> None:
    """nim: judge_capable defaults True, can be set False for tool-less servers."""
    cfg = LoomConfig(feedback_model="meta/llama-3.1-70b-instruct")
    assert get_model("nim")(cfg).resolve("feedback").judge_capable is True
    assert get_model("nim")(cfg, judge_capable=False).resolve("feedback").judge_capable is False


def test_openai_compat_feedback_judge_capable_is_configurable(clean_env: None) -> None:
    """openai-compat: judge_capable defaults True, can be set False to fail fast."""
    cfg = LoomConfig(feedback_model="local-model")
    assert get_model("openai-compat")(cfg).resolve("feedback").judge_capable is True
    assert (
        get_model("openai-compat")(cfg, judge_capable=False).resolve("feedback").judge_capable
        is False
    )


def test_openrouter_feedback_judge_capable_reflects_tool_check(
    monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """openrouter: feedback judge_capable follows the best-effort tool-support check.

    The check is monkeypatched (no network): a definitive ``False`` marks the
    route not judge-capable; ``None`` (offline / unknown) is treated as capable so
    the run is not blocked, while the preflight separately warns.
    """
    import loom.providers.model.openrouter as orouter

    cfg = LoomConfig(
        code_model="anthropic/claude-sonnet-4.5",
        feedback_model="some/tool-less-model",
    )

    # Definitive negative -> not judge-capable, and preflight explains why.
    monkeypatch.setattr(orouter, "feedback_slug_supports_tools", lambda _slug: False)
    route = orouter.OpenRouterModelProvider(cfg).resolve("feedback")
    assert route.judge_capable is False

    # Unknown (offline) -> assumed capable, but preflight should warn.
    monkeypatch.setattr(orouter, "feedback_slug_supports_tools", lambda _slug: None)
    provider = orouter.OpenRouterModelProvider(cfg)
    assert provider.resolve("feedback").judge_capable is True
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")  # silence the key hint
    assert _mentions(provider.preflight("feedback"), "tool")

    # Definitive positive -> judge-capable.
    monkeypatch.setattr(orouter, "feedback_slug_supports_tools", lambda _slug: True)
    assert orouter.OpenRouterModelProvider(cfg).resolve("feedback").judge_capable is True


# ---------------------------------------------------------------------------
# OpenRouter feedback-slug shape guard (must be provider/model form).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_slug",
    ["claude-sonnet-4-5", "gpt-4o", "gemini-1.5-pro", "o3", "local-model", ""],
)
def test_openrouter_shape_error_on_non_slug(bad_slug: str) -> None:
    """A bare (non provider/model) feedback slug yields an actionable shape error."""
    import loom.providers.model.openrouter as orouter

    err = orouter.feedback_slug_shape_error(bad_slug)
    assert err is not None
    # Actionable: mentions the provider/model requirement (or that it's missing).
    assert "provider/model" in err or "configured" in err


@pytest.mark.parametrize(
    "good_slug",
    ["anthropic/claude-sonnet-4.5", "openai/gpt-4o", "meta-llama/llama-3.1-70b"],
)
def test_openrouter_shape_ok_on_provider_model_slug(good_slug: str) -> None:
    """A proper provider/model slug passes the shape guard (no error)."""
    import loom.providers.model.openrouter as orouter

    assert orouter.feedback_slug_shape_error(good_slug) is None


def test_openrouter_bare_feedback_slug_is_not_judge_capable(clean_env: None) -> None:
    """A bare 'claude-' feedback slug routes to native AIDE -> not judge-capable.

    This is the collision the guard prevents: AIDE's name-based routing would send
    'claude-...' to the native anthropic backend instead of the OpenAI-compatible
    OpenRouter diversion, so the slug can never serve the OpenRouter judge.
    """
    cfg = LoomConfig(
        code_provider="openrouter",
        feedback_provider="openrouter",
        code_model="anthropic/claude-sonnet-4.5",
        feedback_model="claude-sonnet-4-5",  # bare reserved-prefix slug, no "/"
    )
    route = get_model("openrouter")(cfg).resolve("feedback")
    assert route.judge_capable is False


def test_openrouter_preflight_flags_bare_feedback_slug(
    monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """preflight surfaces the shape error for a bare feedback slug (and only that)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")  # silence the key hint
    provider = _provider(
        "openrouter",
        code_model="anthropic/claude-sonnet-4.5",
        feedback_model="gpt-4o",  # bare reserved-prefix slug, no "/"
    )
    hints = provider.preflight("feedback")
    assert _mentions(hints, "provider/model")
    # The shape error short-circuits the (network) tool-support warning.
    assert not _mentions(hints, "supports tool calling")


def test_openrouter_preflight_clean_with_good_feedback_slug(
    monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """A proper, tool-capable feedback slug clears the OpenRouter feedback preflight."""
    import loom.providers.model.openrouter as orouter

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    monkeypatch.setattr(orouter, "feedback_slug_supports_tools", lambda _slug: True)
    provider = _provider(
        "openrouter",
        code_model="anthropic/claude-sonnet-4.5",
        feedback_model="anthropic/claude-sonnet-4.5",
    )
    assert provider.preflight("feedback") == []


# ---------------------------------------------------------------------------
# AIDE-dependent surface: only the dispatch override genuinely needs AIDE.
# ---------------------------------------------------------------------------


def test_cli_bridge_install_dispatch_override_patches_aide() -> None:
    """cli-bridge install_dispatch_override binds a loom_query into AIDE's table.

    Genuinely needs AIDE: it runs the dispatch signature smoke test and patches the
    in-memory ``provider_to_query_func`` dict. No model is ever invoked — we only
    assert the dispatch slot now holds our callable. The dict is restored after.
    """
    aide_backend = pytest.importorskip("aide.backend")

    table = aide_backend.provider_to_query_func
    saved = dict(table)
    try:
        _provider("claude-subscription").install_dispatch_override(aide_backend)
        # claude- sentinel routes to the "anthropic" provider key.
        assert table["anthropic"].__name__ == "loom_query"

        _provider("codex-subscription").install_dispatch_override(aide_backend)
        # codex-mini-latest routes to the "openai" provider key.
        assert table["openai"].__name__ == "loom_query"
    finally:
        table.clear()
        table.update(saved)


def test_review_output_schema_matches_aide_submit_review() -> None:
    """The dispatch review schema mirrors AIDE's submit_review func_spec shape."""
    pytest.importorskip("aide.agent")
    from loom.providers.model import _dispatch

    schema = _dispatch.review_output_schema()
    props = schema.get("properties", {})
    for key in ("is_bug", "summary", "metric", "lower_is_better"):
        assert key in props, key
