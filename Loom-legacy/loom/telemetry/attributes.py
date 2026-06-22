"""Standard telemetry attributes for Loom (cardinality-controlled, secret-free).

Modeled on Claude Code's ``telemetryAttributes.ts``: a single function that
assembles the small dict of *standard* dimensions attached to every telemetry
event/span -- who/what/where, never the payload. Two cardinality toggles mirror
``OTEL_METRICS_INCLUDE_*`` so a high-cardinality dimension (a per-run session id,
the build version) can be dropped from metric series:

* ``LOOM_TELEMETRY_INCLUDE_SESSION_ID`` (default **true**) -- include the
  per-process ``session.id``;
* ``LOOM_TELEMETRY_INCLUDE_VERSION`` (default **false**) -- include
  ``app.version`` (off by default: a version churns the dimension on every
  release, which is the cardinality footgun CC's defaults guard against).

The IP-boundary tag (``owned_by``) and the multi-tenant tag (``tenant``) are
always included: they are low-cardinality *and* they are exactly what the
distillation export filters on. **No secret material is ever read here** -- only
the non-secret routing/selection values already on :class:`~loom.config.LoomConfig`
(model names, provider names, tenant/owner), never any key or endpoint.

Pure: this module imports only the standard library + Loom config, so it is
importable (and testable) without OpenTelemetry or any heavy dependency.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Mapping

from loom.config import LoomConfig

# Loom's package version, surfaced as ``app.version`` when the version toggle is
# on. Imported lazily-ish (module-level import of the lightweight ``loom`` package
# only) so this stays dependency-free.
from loom import __version__ as _LOOM_VERSION

# Per-process session id. Generated once at import so every event from one Loom
# process shares a ``session.id`` (the CC ``getSessionId()`` analogue) without a
# stateful singleton object. A fresh process => a fresh session.
_SESSION_ID = uuid.uuid4().hex

# Cardinality defaults, mirroring CC's METRICS_CARDINALITY_DEFAULTS. Session id is
# on (useful + bounded per process); version is off (churns every release).
_CARDINALITY_DEFAULTS = {
    "LOOM_TELEMETRY_INCLUDE_SESSION_ID": True,
    "LOOM_TELEMETRY_INCLUDE_VERSION": False,
}

# Recognized truthy spellings for the env toggles (matches the rest of Loom's
# point-of-use boolean reads).
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off", ""}


def _is_truthy(value: str | None) -> bool:
    """Return whether an env string spells a truthy value."""
    return value is not None and value.strip().lower() in _TRUTHY


def session_id() -> str:
    """Return this process's stable telemetry session id."""
    return _SESSION_ID


def _should_include(
    env_var: str, env: Mapping[str, str]
) -> bool:
    """Resolve a cardinality toggle: env override else the baked-in default.

    Mirrors CC's ``shouldIncludeAttribute``: an unset env var falls back to the
    default; a set env var is parsed as a boolean (any non-truthy spelling, e.g.
    ``"0"``/``"false"``, turns the dimension off).
    """
    default = _CARDINALITY_DEFAULTS[env_var]
    raw = env.get(env_var)
    if raw is None:
        return default
    if raw.strip().lower() in _FALSY:
        return False
    return _is_truthy(raw)


def telemetry_attributes(
    config: LoomConfig,
    *,
    run_id: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the standard telemetry attribute dict for ``config``.

    The low-cardinality dimensions attached to every Loom telemetry event/span:
    the IP-boundary owner, the tenant, the routed model, the search/MLOps
    providers, plus the cardinality-gated ``session.id`` and ``app.version``. An
    optional ``run_id`` (e.g. an experiment id) is included when given.

    Args:
        config: The active Loom configuration. Only non-secret routing/ownership
            values are read; no key material is touched.
        run_id: Optional stable run identifier (e.g. ``task.experiment_id``) to
            stamp as ``run.id``.
        env: Environment mapping for the cardinality toggles (defaults to
            ``os.environ``).

    Returns:
        A JSON-serializable dict of standard attributes. Never contains secrets.
    """
    e: Mapping[str, str] = env if env is not None else os.environ

    attrs: dict[str, Any] = {
        # The IP boundary + multi-tenant tags -- always present, low-cardinality,
        # and exactly what the distillation export filters on.
        "owned_by": config.owned_by,
        "tenant": config.tenant,
        # Routing dimensions (non-secret model/provider *names*, never endpoints).
        "model": config.code_model,
        "search.provider": config.search_provider,
        "mlops.provider": config.mlops_provider,
        "service.name": "loom",
    }

    if run_id:
        attrs["run.id"] = run_id

    if _should_include("LOOM_TELEMETRY_INCLUDE_SESSION_ID", e):
        attrs["session.id"] = _SESSION_ID
    if _should_include("LOOM_TELEMETRY_INCLUDE_VERSION", e):
        attrs["app.version"] = _LOOM_VERSION

    return attrs


__all__ = ["telemetry_attributes", "session_id"]
