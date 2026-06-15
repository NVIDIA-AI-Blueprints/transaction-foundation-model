"""Generic OpenAI-compatible model provider (``"openai-compat"``).

A catch-all for any self-hosted or proxied OpenAI-compatible endpoint —
LiteLLM, vLLM, Ollama (``/v1``), text-generation-inference, a private gateway,
etc. It sets ``OPENAI_BASE_URL`` from ``cfg.model_base_url`` (or whatever the user
already set in ``OPENAI_BASE_URL``) so AIDE's OpenAI backend takes the
chat-completions path, and passes through ``OPENAI_API_KEY`` unchanged (which for
many local servers is a dummy token the server ignores).

Model names are non-reserved slugs the endpoint serves. Tool/func_spec support
varies by server; AIDE's chat-completions path degrades gracefully on
``BadRequestError`` for the code role, but the feedback judge needs tools — so
``judge_capable`` is configurable (default ``True``; set ``False`` for servers
known to lack tool calling so the CLI fails fast).
"""

from __future__ import annotations

import os
from typing import MutableMapping, Optional

from loom.config import LoomConfig
from loom.providers.model import ModelProvider, ModelRoute
from loom.registry import register_model


@register_model("openai-compat")
class OpenAiCompatModelProvider(ModelProvider):
    """Generic self-hosted / proxied OpenAI-compatible backend.

    Attributes:
        name: Registry name, ``"openai-compat"``.
        config: The active :class:`~loom.config.LoomConfig`.
        judge_capable: Whether the served feedback model supports tool calling.
            Defaults to ``True``; set ``False`` for servers lacking tool calling.
    """

    name = "openai-compat"

    def __init__(self, config: LoomConfig, judge_capable: bool = True) -> None:
        """Initialize from a Loom config (no secret material is read).

        Args:
            config: The active configuration (supplies ``model_base_url`` and the
                slug model names).
            judge_capable: Whether the served feedback model supports tool
                calling. Defaults to ``True``.
        """
        self.config = config
        self.judge_capable = judge_capable

    def _base_url(self) -> Optional[str]:
        """Return the endpoint: ``cfg.model_base_url`` or env ``OPENAI_BASE_URL``.

        Returns:
            The configured base URL, or ``None`` if neither is set (the preflight
            then flags it).
        """
        return self.config.model_base_url or os.environ.get("OPENAI_BASE_URL")

    def resolve(self, role: str) -> ModelRoute:
        """Return the OpenAI-compatible route for ``role``.

        Args:
            role: ``"code"`` or ``"feedback"``.

        Returns:
            An ``openai-compat`` :class:`ModelRoute` for the configured slug.
        """
        model = (
            self.config.feedback_model
            if role == "feedback"
            else self.config.code_model
        )
        judge_capable = self.judge_capable if role == "feedback" else True
        return ModelRoute(
            model_name=model,
            base_url=self._base_url(),
            key_env="OPENAI_API_KEY",
            judge_capable=judge_capable,
            kind="openai-compat",
        )

    def prepare_env(self, env: MutableMapping[str, str]) -> None:
        """Materialize ``OPENAI_BASE_URL`` and pass through ``OPENAI_API_KEY``.

        Sets ``OPENAI_BASE_URL`` from ``cfg.model_base_url`` when configured (the
        env value is left as-is otherwise). ``OPENAI_API_KEY`` is passed through
        unchanged — for many local servers it is a dummy token, but it is still
        only ever the value the user supplied.

        Args:
            env: The mutable environment mapping (typically ``os.environ``).
        """
        base_url = self.config.model_base_url
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
        # OPENAI_API_KEY is already where the backend reads it; pass through.

    def preflight(self, role: str) -> list[str]:
        """Hint on a missing base URL/key, or a feedback model lacking tools.

        Args:
            role: ``"code"`` or ``"feedback"``.

        Returns:
            Hint strings (empty if the role looks ready).
        """
        hints: list[str] = []
        if not self._base_url():
            hints.append(
                "No OpenAI-compatible endpoint configured: set LOOM_MODEL_BASE_URL "
                "(cfg.model_base_url) or OPENAI_BASE_URL to your server's /v1 URL."
            )
        if not os.environ.get("OPENAI_API_KEY"):
            hints.append(
                "OPENAI_API_KEY is not set. Many local servers accept a dummy "
                "token (e.g. export OPENAI_API_KEY=sk-local); set whatever your "
                "endpoint expects."
            )
        if role == "feedback" and not self.judge_capable:
            hints.append(
                f"The feedback model {self.resolve('feedback').model_name!r} is "
                "marked not tool-capable; the judge (submit_review) needs tool "
                "calling. Use a tool-capable served model for the feedback role."
            )
        return hints


__all__ = ["OpenAiCompatModelProvider"]
