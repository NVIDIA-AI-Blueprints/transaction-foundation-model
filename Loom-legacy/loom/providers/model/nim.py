"""NVIDIA NIM model provider (``"nim"``) via the OpenAI-compatible route.

NVIDIA NIM (and the hosted ``integrate.api.nvidia.com`` catalog) speak the
OpenAI-compatible chat-completions API. This provider sets ``OPENAI_BASE_URL`` to
the NIM endpoint (``cfg.nim_base_url`` or the hosted default) so AIDE's OpenAI
backend takes the chat-completions path, and copies ``NVIDIA_API_KEY`` into
``OPENAI_API_KEY`` (the var that client reads).

Model names are non-reserved slugs the endpoint serves (so ``determine_provider``
falls through to the ``OPENAI_BASE_URL`` branch). Tool/func_spec support varies by
served model, so ``judge_capable`` is configurable (default ``True``).

This provider never reads or stores key material — :meth:`prepare_env` only moves
through ``NVIDIA_API_KEY`` if the user set it.
"""

from __future__ import annotations

import os
from typing import MutableMapping

from loom.config import LoomConfig
from loom.providers.model import ModelProvider, ModelRoute
from loom.registry import register_model

_DEFAULT_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


@register_model("nim")
class NimModelProvider(ModelProvider):
    """NVIDIA NIM via AIDE's OpenAI-compatible backend.

    Attributes:
        name: Registry name, ``"nim"``.
        config: The active :class:`~loom.config.LoomConfig`.
        judge_capable: Whether NIM-served models are treated as tool-capable for
            the feedback judge. Defaults to ``True``; override per deployment.
    """

    name = "nim"

    def __init__(self, config: LoomConfig, judge_capable: bool = True) -> None:
        """Initialize from a Loom config (no secret material is read).

        Args:
            config: The active configuration (supplies the NIM base URL and the
                slug model names).
            judge_capable: Whether the served feedback model supports tool
                calling (the judge needs it). Defaults to ``True``.
        """
        self.config = config
        self.judge_capable = judge_capable

    def _base_url(self) -> str:
        """Return the NIM endpoint: ``cfg.nim_base_url`` or the hosted default."""
        return self.config.nim_base_url or _DEFAULT_NIM_BASE_URL

    def resolve(self, role: str) -> ModelRoute:
        """Return the NIM route for ``role``.

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
            key_env="NVIDIA_API_KEY",
            judge_capable=judge_capable,
            kind="openai-compat",
        )

    def prepare_env(self, env: MutableMapping[str, str]) -> None:
        """Wire NIM through AIDE's OpenAI-compatible backend.

        Sets ``OPENAI_BASE_URL`` to the NIM endpoint and copies the user's
        ``NVIDIA_API_KEY`` into ``OPENAI_API_KEY``. Never invents a key.

        Args:
            env: The mutable environment mapping (typically ``os.environ``).
        """
        env["OPENAI_BASE_URL"] = self._base_url()
        key = env.get("NVIDIA_API_KEY")
        if key:
            env["OPENAI_API_KEY"] = key

    def preflight(self, role: str) -> list[str]:
        """Hint if ``NVIDIA_API_KEY`` is unset, or the judge is not tool-capable.

        Args:
            role: ``"code"`` or ``"feedback"``.

        Returns:
            Hint strings (empty if the role looks ready).
        """
        hints: list[str] = []
        if not os.environ.get("NVIDIA_API_KEY"):
            hints.append(
                "NVIDIA_API_KEY is not set (copied into OPENAI_API_KEY for the "
                "OpenAI-compatible client). Export it: export NVIDIA_API_KEY=nvapi-..."
            )
        if role == "feedback" and not self.judge_capable:
            hints.append(
                f"The NIM feedback model {self.resolve('feedback').model_name!r} "
                "is marked not tool-capable; the judge (submit_review) needs tool "
                "calling. Use a tool-capable served model for the feedback role."
            )
        return hints


__all__ = ["NimModelProvider"]
