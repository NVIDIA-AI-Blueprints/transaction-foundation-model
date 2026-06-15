"""Native OpenAI API model provider (``"openai-api"``).

Routes ``gpt-*`` / ``o<N>`` / ``codex-mini-latest`` model names through AIDE's
OpenAI backend (``aide/backend/backend_openai.py``). For *real* OpenAI model names
that backend hardcodes ``base_url=https://api.openai.com/v1`` and uses the
Responses API, **ignoring** ``OPENAI_BASE_URL`` — so this provider deliberately
does NOT set ``OPENAI_BASE_URL`` (setting it would mis-route other roles or have
no effect here). The Responses-API path implements ``func_spec`` via tools, so the
route is judge-capable for the feedback role.

The credential is ``OPENAI_API_KEY``, read from the environment by the backend.
This provider never reads or stores key material.
"""

from __future__ import annotations

import os
from typing import MutableMapping

from loom.config import LoomConfig
from loom.providers.model import ModelProvider, ModelRoute
from loom.registry import register_model


@register_model("openai-api")
class OpenAiApiModelProvider(ModelProvider):
    """Native OpenAI API backend (``gpt-*`` / ``o<N>``).

    Resolves the configured ``code_model`` / ``feedback_model`` (which must be a
    real OpenAI name so AIDE routes to its OpenAI backend) and relies on
    ``OPENAI_API_KEY`` from the environment. Crucially it leaves
    ``OPENAI_BASE_URL`` untouched, since real OpenAI ignores it.

    Attributes:
        name: Registry name, ``"openai-api"``.
        config: The active :class:`~loom.config.LoomConfig`.
    """

    name = "openai-api"

    def __init__(self, config: LoomConfig) -> None:
        """Initialize from a Loom config (no secret material is read).

        Args:
            config: The active configuration (supplies the model names).
        """
        self.config = config

    def resolve(self, role: str) -> ModelRoute:
        """Return the OpenAI route for ``role``.

        Args:
            role: ``"code"`` or ``"feedback"``.

        Returns:
            A native-API :class:`ModelRoute` for the configured OpenAI model.
            ``base_url`` is ``None`` (the backend hardcodes api.openai.com).
        """
        model = (
            self.config.feedback_model
            if role == "feedback"
            else self.config.code_model
        )
        return ModelRoute(
            model_name=model,
            base_url=None,  # do NOT set OPENAI_BASE_URL: real OpenAI ignores it
            key_env="OPENAI_API_KEY",
            judge_capable=True,
            kind="api",
        )

    def prepare_env(self, env: MutableMapping[str, str]) -> None:
        """No-op: the OpenAI backend reads ``OPENAI_API_KEY`` and hardcodes the URL.

        We must NOT set ``OPENAI_BASE_URL`` here: for real OpenAI model names the
        backend ignores it, and leaving a stray value set could mis-route a
        differently-configured sibling role.

        Args:
            env: The mutable environment mapping (typically ``os.environ``).
        """
        return None

    def preflight(self, role: str) -> list[str]:
        """Hint if ``OPENAI_API_KEY`` is unset.

        Args:
            role: ``"code"`` or ``"feedback"``.

        Returns:
            A one-item hint list if the key is missing, else empty.
        """
        if not os.environ.get("OPENAI_API_KEY"):
            model = self.resolve(role).model_name
            return [
                f"OPENAI_API_KEY is not set (OpenAI models read it; "
                f"{role} model '{model}'). Export it: export OPENAI_API_KEY=sk-..."
            ]
        return []


__all__ = ["OpenAiApiModelProvider"]
