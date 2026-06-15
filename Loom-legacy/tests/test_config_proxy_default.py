"""DEFAULT-ONCE-HOSTED: the model provider defaults flip on a hosted gateway.

:meth:`loom.config.LoomConfig.load` resolves the ``code_provider`` /
``feedback_provider`` defaults AFTER the YAML/env/override merge so it can tell a
deliberate provider choice apart from the dataclass default. The rule:

* explicit choice (override ``code/feedback/model-provider`` or the
  ``LOOM_CODE/FEEDBACK_PROVIDER`` env) ALWAYS wins;
* else a HOSTED gateway (``LOOM_API_BASE`` set to a non-loopback URL, or
  ``LOOM_PROXY_DEFAULT`` truthy) defaults the roles to ``loom-proxy``;
* else the historical ``anthropic-api`` default is kept unchanged.

Every case passes an explicit ``env`` mapping (often ``{}``) so the suite never
reads the real process environment or a stray ``.env`` — these are pure
resolution tests, no network and no secrets.
"""

from __future__ import annotations

import pytest

from loom.config import LoomConfig

# A representative hosted gateway URL (non-loopback) a client would point at.
_HOSTED_BASE = "https://gateway.loom.zkai.network"


# ---------------------------------------------------------------------------
# No hosting signal -> the historical anthropic-api default is unchanged.
# ---------------------------------------------------------------------------


def test_no_hosting_signal_keeps_anthropic_api_default() -> None:
    """Empty env, default base: providers stay ``anthropic-api`` (unchanged)."""
    cfg = LoomConfig.load(env={})
    assert cfg.code_provider == "anthropic-api"
    assert cfg.feedback_provider == "anthropic-api"


def test_localhost_default_base_does_not_trigger_hosting() -> None:
    """The loopback default base is NOT hosting, so it must not flip the default."""
    cfg = LoomConfig.load(env={"LOOM_API_BASE": "http://127.0.0.1:8088"})
    assert cfg.code_provider == "anthropic-api"
    assert cfg.feedback_provider == "anthropic-api"
    # The base is still recorded; only the provider default is left alone.
    assert cfg.loom_api_base == "http://127.0.0.1:8088"


@pytest.mark.parametrize(
    "base",
    [
        "http://127.0.0.1:9000",  # loopback, non-default port
        "http://localhost:8088",
        "http://[::1]:8088",
        "http://0.0.0.0:8088",
    ],
)
def test_other_loopback_bases_do_not_trigger_hosting(base: str) -> None:
    """Any localhost/loopback base is local dev, not hosting -> no flip."""
    cfg = LoomConfig.load(env={"LOOM_API_BASE": base})
    assert cfg.code_provider == "anthropic-api"
    assert cfg.feedback_provider == "anthropic-api"


# ---------------------------------------------------------------------------
# Hosted gateway -> loom-proxy becomes the default for both roles.
# ---------------------------------------------------------------------------


def test_hosted_base_defaults_to_loom_proxy() -> None:
    """A non-loopback ``LOOM_API_BASE`` flips both roles to ``loom-proxy``."""
    cfg = LoomConfig.load(env={"LOOM_API_BASE": _HOSTED_BASE})
    assert cfg.code_provider == "loom-proxy"
    assert cfg.feedback_provider == "loom-proxy"
    assert cfg.loom_api_base == _HOSTED_BASE


@pytest.mark.parametrize("flag", ["1", "true", "TRUE", "yes", "on"])
def test_loom_proxy_default_truthy_flips_even_on_default_base(flag: str) -> None:
    """``LOOM_PROXY_DEFAULT`` truthy flips the default even on the loopback base."""
    cfg = LoomConfig.load(env={"LOOM_PROXY_DEFAULT": flag})
    assert cfg.code_provider == "loom-proxy"
    assert cfg.feedback_provider == "loom-proxy"


@pytest.mark.parametrize("flag", ["0", "false", "no", "off", ""])
def test_loom_proxy_default_falsy_does_not_flip(flag: str) -> None:
    """A falsy/blank ``LOOM_PROXY_DEFAULT`` is not a hosting signal."""
    cfg = LoomConfig.load(env={"LOOM_PROXY_DEFAULT": flag})
    assert cfg.code_provider == "anthropic-api"
    assert cfg.feedback_provider == "anthropic-api"


# ---------------------------------------------------------------------------
# Explicit provider choice ALWAYS wins over the hosted default.
# ---------------------------------------------------------------------------


def test_explicit_env_provider_overrides_hosted_default() -> None:
    """``LOOM_CODE/FEEDBACK_PROVIDER`` beat the hosted ``loom-proxy`` default."""
    cfg = LoomConfig.load(
        env={
            "LOOM_API_BASE": _HOSTED_BASE,
            "LOOM_CODE_PROVIDER": "openai-api",
            "LOOM_FEEDBACK_PROVIDER": "nim",
        }
    )
    assert cfg.code_provider == "openai-api"
    assert cfg.feedback_provider == "nim"


def test_explicit_override_provider_overrides_hosted_default() -> None:
    """A CLI-style ``overrides`` provider (``--code/feedback-provider``) wins."""
    cfg = LoomConfig.load(
        env={"LOOM_API_BASE": _HOSTED_BASE},
        overrides={"code_provider": "openai-api", "feedback_provider": "openai-api"},
    )
    assert cfg.code_provider == "openai-api"
    assert cfg.feedback_provider == "openai-api"


def test_explicit_anthropic_api_on_hosted_box_is_honored() -> None:
    """Explicitly choosing ``anthropic-api`` opts OUT of the moat even when hosted.

    This is the sentinel case: the value equals the dataclass default, but
    because it was set EXPLICITLY it must NOT be flipped to ``loom-proxy``.
    """
    cfg = LoomConfig.load(
        env={
            "LOOM_API_BASE": _HOSTED_BASE,
            "LOOM_CODE_PROVIDER": "anthropic-api",
            "LOOM_FEEDBACK_PROVIDER": "anthropic-api",
        }
    )
    assert cfg.code_provider == "anthropic-api"
    assert cfg.feedback_provider == "anthropic-api"


def test_explicit_one_role_hosted_default_fills_the_other() -> None:
    """An explicit code provider stands; the unset feedback role still flips."""
    cfg = LoomConfig.load(
        env={"LOOM_API_BASE": _HOSTED_BASE, "LOOM_CODE_PROVIDER": "openai-api"}
    )
    assert cfg.code_provider == "openai-api"
    assert cfg.feedback_provider == "loom-proxy"


def test_explicit_override_wins_over_explicit_env() -> None:
    """The CLI override (highest precedence) beats an env provider on a hosted box."""
    cfg = LoomConfig.load(
        env={"LOOM_API_BASE": _HOSTED_BASE, "LOOM_CODE_PROVIDER": "openai-api"},
        overrides={"code_provider": "nim"},
    )
    assert cfg.code_provider == "nim"
