"""Model providers (the third Loom seam) and built-in adapter registration.

Loom is ports-and-adapters. Alongside the *search* ("brain", AIDE) and
*execution* ("muscle", Metaflow/local) ports, this package defines a third port:
the :class:`ModelProvider` — the **LLM backend**, i.e. *which model and how it is
authenticated*. It configures AIDE's **existing** backends (see
``aide/backend/__init__.py``) via configuration and environment variables, plus
an optional runtime dispatch override, **without forking AIDE**.

A :class:`ModelProvider` is resolved once per role (``"code"`` writes solutions,
``"feedback"`` reviews/scores them) and:

* :meth:`~ModelProvider.resolve` returns a :class:`ModelRoute` — the model name
  AIDE should route on plus the env knobs that route it;
* :meth:`~ModelProvider.prepare_env` materializes those knobs into the process
  environment (``OPENAI_BASE_URL`` and the key var AIDE's backend reads) **before
  the agent loop's first query**, because AIDE memoizes its clients with funcy
  ``@once`` — env must be set first;
* :meth:`~ModelProvider.preflight` returns human hints for missing credentials or
  login state (an empty list means the role looks ready);
* :meth:`~ModelProvider.install_dispatch_override` is a no-op by default;
  ``cli-bridge`` adapters override it to install a ``loom_query`` into AIDE's
  ``provider_to_query_func`` dict (see :mod:`loom.providers.model._dispatch`).

AIDE's backend routes purely by model **name** (``determine_provider``):
``gpt-*``/``o<N>``/``codex-mini-latest`` -> openai; ``claude-*`` -> anthropic;
``gemini-*`` -> gemini; otherwise if ``OPENAI_BASE_URL`` is set -> the openai
backend in chat-completions mode against that base URL; else -> the dedicated
openrouter backend. The openrouter backend raises ``NotImplementedError`` for any
``func_spec``, and AIDE's feedback judge **always** passes one, so an
OpenAI-compatible route (``OPENAI_BASE_URL`` set) is required for any non-reserved
slug — never the dedicated openrouter backend for the feedback role.

This module imports only stdlib + ``loom`` core at the top. The built-in adapter
modules are imported at the *bottom*, each inside its own ``try/except`` so a
missing optional dependency cannot break ``import loom.providers.model`` or, by
extension, ``loom`` core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import MutableMapping, Optional


@dataclass(frozen=True)
class ModelRoute:
    """An immutable description of how AIDE should route one model role.

    A :class:`ModelProvider` produces a :class:`ModelRoute` per role. The route
    carries the model *name* AIDE routes on (its ``determine_provider`` is
    name-based) and the environment knobs that steer that routing — but **never**
    any secret material itself. Keys are only ever named (``key_env``) so the
    provider can move/passthrough what the user already set; Loom never invents,
    stores, or logs key values.

    Attributes:
        model_name: The model name AIDE routes on (e.g. ``"claude-sonnet-4-5"``,
            ``"gpt-4o"``, or an OpenAI-compatible slug like
            ``"anthropic/claude-sonnet-4.5"``). For ``cli-bridge`` adapters this
            is a sentinel (e.g. ``"claude-code-subscription"``) consumed by a
            dispatch override rather than a real API model id.
        base_url: The OpenAI-compatible endpoint to materialize as
            ``OPENAI_BASE_URL``, or ``None`` for native API routes (real OpenAI
            ignores ``OPENAI_BASE_URL``; native Anthropic does not use it).
        key_env: The name of the environment variable holding the credential this
            route needs (e.g. ``"ANTHROPIC_API_KEY"``), or ``None`` for routes
            that need no API key (e.g. a ``cli-bridge`` to a logged-in CLI).
        extra_env: Additional non-secret environment variables to pass through
            (e.g. ``ANTHROPIC_BASE_URL`` when the user set it). Values are taken
            from what the user already provided; never fabricated.
        judge_capable: Whether this route can serve the **feedback** judge, which
            always calls AIDE with a ``func_spec`` (tool/function calling).
            ``False`` means the CLI should fail fast for the feedback role.
        kind: The route kind — ``"api"`` (native vendor API),
            ``"openai-compat"`` (chat-completions against an OpenAI-compatible
            ``base_url``), or ``"cli-bridge"`` (shell out to a local CLI via a
            dispatch override).
        attribution_headers: Optional HTTP attribution headers (e.g.
            ``HTTP-Referer`` / ``X-Title`` for OpenRouter). Recorded for
            completeness but **not** injected by the core port (injection needs a
            dispatch override; left as polish).
    """

    model_name: str
    base_url: str | None = None
    key_env: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)
    judge_capable: bool = True
    kind: str = "api"  # api | openai-compat | cli-bridge
    attribution_headers: dict | None = None


class ModelProvider(ABC):
    """Port: the LLM backend — which model to use and how it is authenticated.

    The third Loom seam, alongside :class:`~loom.providers.SearchProvider` and
    :class:`~loom.providers.ExecutionProvider`. A model provider configures
    AIDE's existing backends through configuration and environment (and, for
    ``cli-bridge`` kinds, a runtime dispatch override) without forking AIDE.

    Instantiated uniformly as ``Provider(config)`` (mirroring the other ports) so
    the search adapter can resolve a code provider and a feedback provider from
    :class:`~loom.config.LoomConfig` and wire both symmetrically.

    Attributes:
        name: The registry name of this provider (e.g. ``"anthropic-api"``).
    """

    name: str = "model"

    @abstractmethod
    def resolve(self, role: str) -> ModelRoute:
        """Return the :class:`ModelRoute` for ``role``.

        Args:
            role: The model role, one of ``"code"`` (writes solution code) or
                ``"feedback"`` (reviews/scores executed solutions; AIDE always
                calls this role with a ``func_spec``).

        Returns:
            The route describing the model name and env knobs for ``role``.
        """
        raise NotImplementedError

    def prepare_env(self, env: MutableMapping[str, str]) -> None:
        """Materialize this provider's routing knobs into ``env``.

        Called **before** the agent loop's first query so AIDE's funcy-``@once``
        memoized clients pick up the right base URL and key. Implementations set
        ``OPENAI_BASE_URL`` and/or copy the user's key into the env var AIDE's
        backend reads (``OPENAI_API_KEY`` for OpenAI-compatible routes). They
        **never invent or store secrets** — only move or pass through what the
        user already set. The default is a no-op (native API routes that read
        their key directly, e.g. ``ANTHROPIC_API_KEY``).

        Args:
            env: The mutable environment mapping to materialize into (typically
                ``os.environ``).
        """
        return None

    def preflight(self, role: str) -> list[str]:
        """Return human-readable hints for missing credentials/login for ``role``.

        Best-effort: an empty list means the role looks ready to run.
        Implementations check that the credential or CLI login this route needs
        is present and return one actionable hint per problem found.

        Args:
            role: The model role (``"code"`` or ``"feedback"``).

        Returns:
            A list of hint strings (empty if no obvious problem).
        """
        return []

    def install_dispatch_override(self, aide_backend_module: object) -> None:
        """Install a runtime dispatch override into AIDE's backend, if needed.

        The default is a no-op: API and OpenAI-compatible routes use AIDE's
        existing backends unchanged. ``cli-bridge`` adapters override this to
        register a ``loom_query`` into ``aide.backend.provider_to_query_func``
        (see :mod:`loom.providers.model._dispatch`) that shells out to a local
        CLI while honoring AIDE's ``(output, req_time, in_tokens, out_tokens,
        info)`` return contract.

        Args:
            aide_backend_module: The imported ``aide.backend`` module to patch.

        Returns:
            ``None``.
        """
        return None


__all__ = [
    "ModelRoute",
    "ModelProvider",
]


# ---------------------------------------------------------------------------
# Built-in adapter registration.
#
# Each import is guarded independently: a missing optional dependency for one
# adapter must NOT prevent the others (or core) from importing. Importing each
# module triggers its ``@register_model`` decorator side effects.
# ---------------------------------------------------------------------------

try:  # native Anthropic API ("anthropic-api") -- the default provider.
    from . import anthropic_api  # noqa: F401
except Exception:  # pragma: no cover - optional dependency guard
    pass

try:  # native OpenAI API ("openai-api").
    from . import openai_api  # noqa: F401
except Exception:  # pragma: no cover - optional dependency guard
    pass

try:  # OpenRouter via OpenAI-compatible route ("openrouter").
    from . import openrouter  # noqa: F401
except Exception:  # pragma: no cover - optional dependency guard
    pass

try:  # NVIDIA NIM via OpenAI-compatible route ("nim").
    from . import nim  # noqa: F401
except Exception:  # pragma: no cover - optional dependency guard
    pass

try:  # generic OpenAI-compatible self-host ("openai-compat").
    from . import openai_compat  # noqa: F401
except Exception:  # pragma: no cover - optional dependency guard
    pass

try:  # Claude subscription via local `claude` CLI ("claude-subscription").
    from . import claude_subscription  # noqa: F401
except Exception:  # pragma: no cover - optional dependency guard
    pass

try:  # Codex subscription via local `codex exec` CLI ("codex-subscription").
    from . import codex_subscription  # noqa: F401
except Exception:  # pragma: no cover - optional dependency guard
    pass
