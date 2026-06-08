"""OpenRouter model provider (``"openrouter"``) via the OpenAI-compatible route.

OpenRouter serves many models behind an OpenAI-compatible API. The catch: AIDE's
*dedicated* openrouter backend (``backend_openrouter.py``) raises
``NotImplementedError`` for any ``func_spec``, and the feedback judge always passes
one — so we must NOT let AIDE route to that backend. Instead this provider sets
``OPENAI_BASE_URL="https://openrouter.ai/api/v1"`` so AIDE's *openai* backend takes
the chat-completions path (which implements ``func_spec`` via tools and degrades
gracefully on ``BadRequestError``), and copies ``OPENROUTER_API_KEY`` into
``OPENAI_API_KEY`` (the var that client reads).

Model names come from config in provider/model slug form, e.g.
``"anthropic/claude-sonnet-4.5"`` — a non-reserved slug so ``determine_provider``
falls through to the OPENAI_BASE_URL branch rather than matching ``gpt-``/``claude-``.

Feedback (judge) capability is **per-slug**: not every OpenRouter model supports
tool calling. :func:`feedback_slug_supports_tools` best-effort validates the
feedback slug against OpenRouter's ``/models?supported_parameters=tools`` listing,
caching the result and skipping gracefully when offline (assuming capable, but the
preflight warns). The code role is assumed capable.

Optional attribution headers (``HTTP-Referer`` / ``X-Title``) are recorded on the
route for completeness but NOT injected — injecting them needs a dispatch override
(left as polish).
"""

from __future__ import annotations

import os
from typing import MutableMapping, Optional

from loom.config import LoomConfig
from loom.providers.model import ModelProvider, ModelRoute
from loom.registry import register_model

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_MODELS_TOOLS_URL = "https://openrouter.ai/api/v1/models?supported_parameters=tools"

# Process-lifetime cache for the tool-capable slug listing. ``None`` means "not
# yet fetched / could not fetch"; a set means "fetched (possibly empty)".
_tool_capable_slugs: Optional[set[str]] = None


def _fetch_tool_capable_slugs() -> Optional[set[str]]:
    """Fetch (and cache) the set of OpenRouter slugs that support ``tools``.

    Best-effort and offline-tolerant: returns a cached set on subsequent calls,
    or ``None`` if the listing cannot be fetched/parsed (the caller then treats
    capability as unknown).

    Returns:
        A set of model id slugs that advertise ``tools`` support, or ``None`` if
        the check could not run.
    """
    global _tool_capable_slugs
    if _tool_capable_slugs is not None:
        return _tool_capable_slugs

    try:
        import json
        import urllib.request

        req = urllib.request.Request(
            _MODELS_TOOLS_URL, headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        slugs = {
            item["id"]
            for item in data.get("data", [])
            if isinstance(item, dict) and "id" in item
        }
        _tool_capable_slugs = slugs
        return _tool_capable_slugs
    except Exception:
        # Offline / parse error / rate limited: leave uncached so a later call
        # can retry, and report "unknown".
        return None


def feedback_slug_supports_tools(slug: str) -> Optional[bool]:
    """Best-effort: does OpenRouter slug ``slug`` support tool calling?

    Args:
        slug: The provider/model slug (e.g. ``"anthropic/claude-sonnet-4.5"``).

    Returns:
        ``True`` / ``False`` if the listing could be consulted, or ``None`` if the
        check could not run (offline) — callers should assume capable but warn.
    """
    capable = _fetch_tool_capable_slugs()
    if capable is None:
        return None
    return slug in capable


@register_model("openrouter")
class OpenRouterModelProvider(ModelProvider):
    """OpenRouter via AIDE's OpenAI backend (OpenAI-compatible chat completions).

    Forces the OpenAI-compatible code path (never AIDE's func_spec-less dedicated
    openrouter backend) and validates the feedback slug's tool support.

    Attributes:
        name: Registry name, ``"openrouter"``.
        config: The active :class:`~loom.config.LoomConfig`.
    """

    name = "openrouter"

    def __init__(self, config: LoomConfig) -> None:
        """Initialize from a Loom config (no secret material is read).

        Args:
            config: The active configuration (supplies the slug model names).
        """
        self.config = config

    def resolve(self, role: str) -> ModelRoute:
        """Return the OpenRouter route for ``role``.

        The code role is assumed tool-capable; the feedback role's
        ``judge_capable`` reflects the best-effort tool-support check (assumed
        capable when the check cannot run — the preflight warns in that case).

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

        judge_capable = True
        if role == "feedback":
            supported = feedback_slug_supports_tools(model)
            # Assume capable if the check could not run (None); only mark False on
            # a definitive negative.
            judge_capable = supported is not False

        # Attribution headers are recorded but NOT injected (needs a dispatch
        # override; left as polish). Drawn from env only if the user set them.
        referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        title = os.environ.get("OPENROUTER_X_TITLE")
        headers: dict | None = None
        if referer or title:
            headers = {}
            if referer:
                headers["HTTP-Referer"] = referer
            if title:
                headers["X-Title"] = title

        return ModelRoute(
            model_name=model,
            base_url=_OPENROUTER_BASE_URL,
            key_env="OPENROUTER_API_KEY",
            judge_capable=judge_capable,
            kind="openai-compat",
            attribution_headers=headers,
        )

    def prepare_env(self, env: MutableMapping[str, str]) -> None:
        """Wire OpenRouter through AIDE's OpenAI-compatible backend.

        Sets ``OPENAI_BASE_URL`` to the OpenRouter endpoint and copies the user's
        ``OPENROUTER_API_KEY`` into ``OPENAI_API_KEY`` (the var the OpenAI client
        reads). Never invents a key — only moves through what the user set.

        Args:
            env: The mutable environment mapping (typically ``os.environ``).
        """
        env["OPENAI_BASE_URL"] = _OPENROUTER_BASE_URL
        key = env.get("OPENROUTER_API_KEY")
        if key:
            env["OPENAI_API_KEY"] = key

    def preflight(self, role: str) -> list[str]:
        """Hint on a missing key or a feedback slug that can't run the judge.

        Args:
            role: ``"code"`` or ``"feedback"``.

        Returns:
            Hint strings (empty if the role looks ready).
        """
        hints: list[str] = []
        if not os.environ.get("OPENROUTER_API_KEY"):
            hints.append(
                "OPENROUTER_API_KEY is not set (copied into OPENAI_API_KEY for "
                "the OpenAI-compatible client). Export it: "
                "export OPENROUTER_API_KEY=sk-or-..."
            )

        if role == "feedback":
            slug = self.resolve("feedback").model_name
            supported = feedback_slug_supports_tools(slug)
            if supported is False:
                hints.append(
                    f"OpenRouter feedback slug {slug!r} does not advertise tool "
                    "calling; the judge (submit_review) needs a tool-capable "
                    "model. Pick a tool-capable feedback slug "
                    "(see https://openrouter.ai/models?supported_parameters=tools)."
                )
            elif supported is None:
                hints.append(
                    f"Could not verify that OpenRouter feedback slug {slug!r} "
                    "supports tool calling (offline?). The judge needs tools; if "
                    "the run fails on review, pick a tool-capable slug."
                )
        return hints


__all__ = ["OpenRouterModelProvider", "feedback_slug_supports_tools"]
