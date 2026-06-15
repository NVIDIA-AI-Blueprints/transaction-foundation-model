"""Loom proxy gateway model provider (``"loom-proxy"``) -- the moat path.

Routes ``claude-*`` model names through AIDE's native Anthropic backend
(``aide/backend/backend_anthropic.py``) but points that backend at **Loom's own
gateway** instead of ``api.anthropic.com``. The gateway (see
:mod:`loom.proxy.server`) authenticates the caller by a Loom-issued key, injects
Loom's system prompts, forwards to the real Anthropic API using a *server-side*
key the user never sees, and logs every call centrally for model distillation —
this central capture is the moat.

Because v0 is an **Anthropic-passthrough** gateway (same Messages API shape), it
keeps the ``claude-`` model prefix so AIDE routes to its native Anthropic
backend, and native tool-use/judge works with **no** OpenAI<->Anthropic
translation. The Anthropic client honors ``ANTHROPIC_BASE_URL`` and reads its key
from ``ANTHROPIC_API_KEY``; this provider therefore materializes
``ANTHROPIC_BASE_URL = cfg.loom_api_base`` and copies the user's ``LOOM_API_KEY``
into ``ANTHROPIC_API_KEY`` so the backend calls the Loom gateway authenticated by
the Loom key. The real vendor key lives only on the server.

This provider is **opt-in** (``--model-provider loom-proxy``), not the default,
because the gateway is not hosted yet. It becomes the default once the gateway is
hosted (at which point ``loom-proxy`` is the collect-with-consent path and native
``anthropic-api`` / BYO-key / local providers remain the no-egress mode for
sensitive tenants).

Secrets are never read onto the config or this object: :meth:`prepare_env` only
moves/passes through the ``LOOM_API_KEY`` the user already set in the
environment. The gateway URL (``cfg.loom_api_base``) is a non-secret endpoint.
"""

from __future__ import annotations

import os
from typing import MutableMapping

from loom.config import LoomConfig
from loom.providers.model import ModelProvider, ModelRoute
from loom.registry import register_model


@register_model("loom-proxy")
class LoomProxyModelProvider(ModelProvider):
    """Loom gateway backend: Claude via Loom's proxy (the moat path).

    Resolves the configured ``code_model`` / ``feedback_model`` (which must keep
    their ``claude-`` prefix so AIDE routes them to the native Anthropic backend)
    and points that backend at ``cfg.loom_api_base``, authenticated by the user's
    ``LOOM_API_KEY``. The gateway holds the real vendor key server-side; the user
    only ever supplies the Loom key.

    Attributes:
        name: Registry name, ``"loom-proxy"``.
        config: The active :class:`~loom.config.LoomConfig` (supplies the model
            names and the gateway base URL).
    """

    name = "loom-proxy"

    def __init__(self, config: LoomConfig) -> None:
        """Initialize from a Loom config (no secret material is read).

        Args:
            config: The active configuration (supplies the Claude model names and
                ``loom_api_base``). ``LOOM_API_KEY`` is read from the environment
                at the point of use, never off the config.
        """
        self.config = config

    def resolve(self, role: str) -> ModelRoute:
        """Return the Loom-gateway route for ``role``.

        The model name keeps its ``claude-`` prefix so AIDE's name-based
        ``determine_provider`` routes it to the native Anthropic backend; the
        backend is then redirected to the Loom gateway via ``prepare_env``. The
        gateway is an Anthropic passthrough, so the route is judge-capable (native
        tool use) just like ``anthropic-api``.

        Args:
            role: ``"code"`` or ``"feedback"``.

        Returns:
            A ``kind="proxy"`` :class:`ModelRoute` for the configured Claude model
            pointed at ``cfg.loom_api_base`` and keyed on ``LOOM_API_KEY``.
        """
        model = (
            self.config.feedback_model
            if role == "feedback"
            else self.config.code_model
        )
        return ModelRoute(
            model_name=model,
            base_url=self.config.loom_api_base,
            key_env="LOOM_API_KEY",
            judge_capable=True,
            kind="proxy",
        )

    def prepare_env(self, env: MutableMapping[str, str]) -> None:
        """Redirect AIDE's Anthropic backend at the Loom gateway, Loom-keyed.

        Sets ``ANTHROPIC_BASE_URL = cfg.loom_api_base`` and copies the user's
        ``LOOM_API_KEY`` into ``ANTHROPIC_API_KEY`` so the native Anthropic client
        calls the Loom gateway authenticated by the Loom key (the gateway swaps in
        the real server-side vendor key the user never sees). Does NOT set
        ``OPENAI_BASE_URL`` (this is an Anthropic-passthrough route). Never invents
        or stores secrets — only moves the ``LOOM_API_KEY`` the user already set.

        Args:
            env: The mutable environment mapping (typically ``os.environ``).
        """
        env["ANTHROPIC_BASE_URL"] = self.config.loom_api_base
        loom_key = env.get("LOOM_API_KEY")
        if loom_key:
            # The Anthropic backend authenticates with ANTHROPIC_API_KEY; the
            # gateway only accepts the Loom key, so route the Loom key there. The
            # user's real vendor key (if any) is intentionally overwritten in this
            # process env: under loom-proxy all egress goes through the gateway.
            env["ANTHROPIC_API_KEY"] = loom_key

    def preflight(self, role: str) -> list[str]:
        """Hint if ``LOOM_API_KEY`` is unset (the gateway requires it).

        Args:
            role: ``"code"`` or ``"feedback"``.

        Returns:
            A one-item hint list if the Loom key is missing, else empty.
        """
        if not os.environ.get("LOOM_API_KEY"):
            model = self.resolve(role).model_name
            return [
                f"LOOM_API_KEY is not set (the loom-proxy gateway authenticates "
                f"with it; {role} model '{model}' routes through "
                f"{self.config.loom_api_base}). Export it: "
                "export LOOM_API_KEY=loom-..."
            ]
        return []


__all__ = ["LoomProxyModelProvider"]
