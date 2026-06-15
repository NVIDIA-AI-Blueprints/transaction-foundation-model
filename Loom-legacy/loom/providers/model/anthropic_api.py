"""Native Anthropic API model provider (``"anthropic-api"``) -- the default.

Routes ``claude-*`` model names through AIDE's native Anthropic backend
(``aide/backend/backend_anthropic.py``), which constructs an ``anthropic.Anthropic``
client that reads ``ANTHROPIC_API_KEY`` (and honors ``ANTHROPIC_BASE_URL``) from
the environment. The Anthropic backend implements ``func_spec`` via tool use, so
it is fully judge-capable for the feedback role.

This is the provider used when no model provider is configured, preserving Loom's
historical default behavior (Claude for both code and feedback).

Secrets are never read onto the config or this object: :meth:`prepare_env` only
asserts/passes through what the user already set in the environment.
"""

from __future__ import annotations

import os
from typing import MutableMapping

from loom.config import LoomConfig
from loom.providers.model import ModelProvider, ModelRoute
from loom.registry import register_model


@register_model("anthropic-api")
class AnthropicApiModelProvider(ModelProvider):
    """Native Anthropic API backend (Claude).

    Resolves the configured ``code_model`` / ``feedback_model`` (which must keep
    their ``claude-`` prefix so AIDE routes them to the Anthropic backend) and
    relies on ``ANTHROPIC_API_KEY`` from the environment.

    Attributes:
        name: Registry name, ``"anthropic-api"``.
        config: The active :class:`~loom.config.LoomConfig`.
    """

    name = "anthropic-api"

    def __init__(self, config: LoomConfig) -> None:
        """Initialize from a Loom config (no secret material is read).

        Args:
            config: The active configuration (supplies the model names).
        """
        self.config = config

    def resolve(self, role: str) -> ModelRoute:
        """Return the Claude route for ``role``.

        Args:
            role: ``"code"`` or ``"feedback"``.

        Returns:
            A native-API :class:`ModelRoute` for the configured Claude model.
        """
        model = (
            self.config.feedback_model
            if role == "feedback"
            else self.config.code_model
        )
        return ModelRoute(
            model_name=model,
            base_url=None,
            key_env="ANTHROPIC_API_KEY",
            judge_capable=True,
            kind="api",
        )

    def prepare_env(self, env: MutableMapping[str, str]) -> None:
        """Pass through the Anthropic credential/endpoint the user already set.

        Does not set ``OPENAI_BASE_URL`` (Claude uses the native Anthropic
        client). ``ANTHROPIC_API_KEY`` and an optional ``ANTHROPIC_BASE_URL`` are
        already where the backend reads them; this is a no-op beyond leaving them
        in place (kept explicit for symmetry and to document the contract).

        Args:
            env: The mutable environment mapping (typically ``os.environ``).
        """
        # Nothing to materialize: the native Anthropic client reads
        # ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL directly. We intentionally do
        # NOT touch OPENAI_BASE_URL here (real Claude ignores it, and setting it
        # would mis-route other roles).
        return None

    def preflight(self, role: str) -> list[str]:
        """Hint if ``ANTHROPIC_API_KEY`` is unset.

        Args:
            role: ``"code"`` or ``"feedback"``.

        Returns:
            A one-item hint list if the key is missing, else empty.
        """
        if not os.environ.get("ANTHROPIC_API_KEY"):
            model = self.resolve(role).model_name
            return [
                f"ANTHROPIC_API_KEY is not set (Claude models read it; "
                f"{role} model '{model}'). Export it: "
                "export ANTHROPIC_API_KEY=sk-ant-..."
            ]
        return []


__all__ = ["AnthropicApiModelProvider"]
