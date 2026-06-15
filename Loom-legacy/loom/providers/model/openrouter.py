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

Feedback (judge) capability is **per-slug** and has two gates. First a **shape
guard** (:func:`feedback_slug_shape_error`): the feedback slug must be in
``provider/model`` form (contain a ``"/"``) so it does not collide with AIDE's
reserved ``gpt-``/``claude-``/``gemini-`` regexes — a bare slug would route to a
native backend (or the func_spec-less dedicated openrouter backend) and crash the
judge, so we fail fast. Second a **tool-support check**
(:func:`feedback_slug_supports_tools`): not every OpenRouter model supports tool
calling, so this best-effort validates the slug against OpenRouter's
``/models?supported_parameters=tools`` listing, caching the result and skipping
gracefully when offline (assuming capable, but the preflight warns). The code role
is assumed capable.

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


def feedback_slug_shape_error(slug: str) -> Optional[str]:
    """Return a fail-fast message if ``slug`` is not a routable OpenRouter slug.

    OpenRouter is reached through AIDE's *OpenAI-compatible* backend, which only
    engages when the model name is **not** one of AIDE's reserved prefixes
    (``gpt-`` / ``o<N>`` -> openai, ``claude-`` -> anthropic, ``gemini-`` ->
    gemini). A bare ``claude-...`` or ``gpt-...`` feedback slug would therefore be
    routed to the *native* backend (and AIDE's dedicated openrouter backend
    raises ``NotImplementedError`` for the judge's ``func_spec``), so the judge
    would crash mid-run. Every real OpenRouter id is ``provider/model`` form, so
    we require a ``"/"`` to guarantee the slug routes through the
    ``OPENAI_BASE_URL`` diversion and does not collide with the reserved regexes.

    Args:
        slug: The configured feedback model slug.

    Returns:
        An actionable error string if the slug cannot serve as an OpenRouter
        feedback route, else ``None``.
    """
    cleaned = (slug or "").strip()
    if not cleaned:
        return (
            "no OpenRouter feedback model slug is configured. Set a tool-capable "
            "provider/model slug (e.g. 'anthropic/claude-sonnet-4.5'); the judge "
            "(submit_review) requires a tool-capable model."
        )
    if "/" not in cleaned:
        return (
            f"OpenRouter feedback slug {cleaned!r} is not in provider/model form. "
            "OpenRouter routes through AIDE's OpenAI-compatible backend, which "
            "only engages for non-reserved model names; a bare 'gpt-'/'claude-'/"
            "'gemini-' slug instead collides with AIDE's native-backend regex and "
            "the judge would crash. Use the full 'provider/model' slug (must "
            "contain '/'), e.g. 'anthropic/claude-sonnet-4.5', and ensure it is "
            "tool-capable (see "
            "https://openrouter.ai/models?supported_parameters=tools)."
        )
    return None


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
            # Shape guard first: a non-provider/model slug would mis-route to a
            # native AIDE backend (or the func_spec-less openrouter backend) and
            # crash the judge, so treat it as not judge-capable regardless of the
            # tool-support listing. The CLI judge pre-flight then fails fast with
            # the precise message from :func:`feedback_slug_shape_error`.
            if feedback_slug_shape_error(model) is not None:
                judge_capable = False
            else:
                supported = feedback_slug_supports_tools(model)
                # Assume capable if the check could not run (None); only mark
                # False on a definitive negative.
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
            slug = self.config.feedback_model
            # Shape guard first: a slug that cannot route through the
            # OpenAI-compatible diversion is unconditionally wrong for the judge,
            # so surface that before the (network) tool-support check.
            shape_error = feedback_slug_shape_error(slug)
            if shape_error is not None:
                hints.append(shape_error)
                return hints
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


__all__ = [
    "OpenRouterModelProvider",
    "feedback_slug_supports_tools",
    "feedback_slug_shape_error",
]
