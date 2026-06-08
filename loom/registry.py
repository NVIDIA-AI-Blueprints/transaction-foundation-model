"""Provider registry for Loom.

Maps short provider *names* (as used in configuration, e.g. ``"aide"``,
``"metaflow"``, ``"local"``) to provider *classes*. This is the indirection that
makes Loom pluggable: the controller resolves ``config.search_provider`` and
``config.mlops_provider`` to classes here, never importing concrete adapters
directly.

Built-in adapters register themselves as a side effect of importing
``loom.providers`` (each guarded by its own try/except so a missing optional
dependency cannot break core import). This module itself stays
dependency-light: it imports only the standard library.
"""

from __future__ import annotations

from typing import Callable, Type, TypeVar

# Registries: provider name -> provider class.
_SEARCH_PROVIDERS: dict[str, type] = {}
_EXECUTION_PROVIDERS: dict[str, type] = {}
MODEL_PROVIDERS: dict[str, type] = {}

_T = TypeVar("_T", bound=type)


def register_search(name: str) -> Callable[[_T], _T]:
    """Class decorator that registers a search provider under ``name``.

    Args:
        name: The configuration name to bind the class to (e.g. ``"aide"``).

    Returns:
        A decorator that registers the class and returns it unchanged.
    """

    def _decorator(cls: _T) -> _T:
        _SEARCH_PROVIDERS[name] = cls
        return cls

    return _decorator


def register_execution(name: str) -> Callable[[_T], _T]:
    """Class decorator that registers an execution provider under ``name``.

    Args:
        name: The configuration name to bind the class to (e.g. ``"metaflow"``
            or ``"local"``).

    Returns:
        A decorator that registers the class and returns it unchanged.
    """

    def _decorator(cls: _T) -> _T:
        _EXECUTION_PROVIDERS[name] = cls
        return cls

    return _decorator


def register_model(name: str) -> Callable[[_T], _T]:
    """Class decorator that registers a model provider under ``name``.

    The model provider is the third Loom port: the LLM backend (which model and
    how it is authenticated). See :mod:`loom.providers.model`.

    Args:
        name: The configuration name to bind the class to (e.g.
            ``"anthropic-api"`` or ``"openrouter"``).

    Returns:
        A decorator that registers the class and returns it unchanged.
    """

    def _decorator(cls: _T) -> _T:
        MODEL_PROVIDERS[name] = cls
        return cls

    return _decorator


def _ensure_builtins_loaded() -> None:
    """Import ``loom.providers`` so built-in adapters self-register.

    Importing the providers package triggers the guarded registration of the
    built-in adapters. The import is cheap and idempotent (Python caches it),
    and any optional-dependency failure is swallowed inside ``loom.providers``.
    """
    import loom.providers  # noqa: F401  (import for registration side effects)


def get_search(name: str) -> type:
    """Resolve a registered search provider class by name.

    Args:
        name: The provider name from configuration.

    Returns:
        The registered provider class.

    Raises:
        KeyError: If no search provider is registered under ``name`` (the error
            message lists the names that are available).
    """
    _ensure_builtins_loaded()
    try:
        return _SEARCH_PROVIDERS[name]
    except KeyError:
        available = ", ".join(sorted(_SEARCH_PROVIDERS)) or "<none>"
        raise KeyError(
            f"No search provider registered under {name!r}. Available: {available}."
        ) from None


def get_execution(name: str) -> type:
    """Resolve a registered execution provider class by name.

    Args:
        name: The provider name from configuration.

    Returns:
        The registered provider class.

    Raises:
        KeyError: If no execution provider is registered under ``name`` (the
            error message lists the names that are available).
    """
    _ensure_builtins_loaded()
    try:
        return _EXECUTION_PROVIDERS[name]
    except KeyError:
        available = ", ".join(sorted(_EXECUTION_PROVIDERS)) or "<none>"
        raise KeyError(
            f"No execution provider registered under {name!r}. Available: {available}."
        ) from None


def get_model(name: str) -> type:
    """Resolve a registered model provider class by name.

    Args:
        name: The provider name from configuration (e.g. ``"anthropic-api"``).

    Returns:
        The registered model provider class.

    Raises:
        KeyError: If no model provider is registered under ``name`` (the error
            message lists the names that are available).
    """
    _ensure_builtins_loaded()
    try:
        return MODEL_PROVIDERS[name]
    except KeyError:
        available = ", ".join(sorted(MODEL_PROVIDERS)) or "<none>"
        raise KeyError(
            f"No model provider registered under {name!r}. Available: {available}."
        ) from None


__all__ = [
    "register_search",
    "register_execution",
    "register_model",
    "get_search",
    "get_execution",
    "get_model",
    "MODEL_PROVIDERS",
]
