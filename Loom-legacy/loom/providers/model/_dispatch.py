"""Runtime dispatch override for AIDE's backend (used by ``cli-bridge`` adapters).

AIDE routes a query by model **name** through ``aide.backend.determine_provider``
to one of the functions in ``aide.backend.provider_to_query_func``. A ``cli-bridge``
:class:`~loom.providers.model.ModelProvider` (Claude / Codex subscription) cannot
use any of AIDE's vendor backends — there is no API key, only a logged-in local
CLI. Instead it installs a wrapper into ``provider_to_query_func`` for the relevant
provider key so that AIDE's existing ``query(...)`` dispatch calls *our* function.

This module never edits files under ``aide/`` — it patches the in-memory module
dict at run setup. It first runs a **signature smoke test** against the SHA-pinned
AIDE so a future upstream reshape fails loudly here rather than mid-run.

The query contract (read from ``aide/backend/__init__.py`` and a backend module):

    query(system_message, user_message, func_spec=None, **model_kwargs)
        -> (output, req_time, in_tokens, out_tokens, info)

When ``func_spec`` is given, ``output`` is the **parsed** result: a ``dict``
matching ``func_spec.json_schema`` (the same shape the openai/anthropic backends
return for a tool call). Otherwise ``output`` is plain text. AIDE's ``query``
wrapper compiles the prompts to Markdown *before* calling the provider func, so a
``loom_query`` receives already-compiled string messages.

The feedback judge is ``aide.agent.review_func_spec`` (name ``submit_review``); its
``json_schema`` requires keys ``is_bug`` (bool), ``summary`` (str), ``metric``
(number|null), ``lower_is_better`` (bool). ``cli-bridge`` adapters coerce their CLI
output into exactly that shape for the feedback role.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional


# The five required keys of AIDE's ``submit_review`` feedback func_spec, for
# best-effort coercion of free-form CLI output into the judge's JSON shape.
_REVIEW_KEYS = ("is_bug", "summary", "metric", "lower_is_better")

QueryFunc = Callable[..., tuple[Any, float, int, int, dict]]


def smoke_test_aide_backend(aide_backend_module: Any) -> None:
    """Assert the SHA-pinned AIDE backend has the surface we patch.

    Verifies the three symbols a dispatch override depends on exist with the
    expected shapes, so an upstream reshape fails loudly at setup rather than
    deep inside a run.

    Args:
        aide_backend_module: The imported ``aide.backend`` module.

    Raises:
        RuntimeError: If any expected symbol is missing or the wrong shape (the
            message names what changed so the SHA pin can be revisited).
    """
    problems: list[str] = []

    if not callable(getattr(aide_backend_module, "query", None)):
        problems.append("aide.backend.query is missing or not callable")
    if not callable(getattr(aide_backend_module, "determine_provider", None)):
        problems.append("aide.backend.determine_provider is missing or not callable")

    table = getattr(aide_backend_module, "provider_to_query_func", None)
    if not isinstance(table, dict):
        problems.append("aide.backend.provider_to_query_func is missing or not a dict")
    elif not all(callable(v) for v in table.values()):
        problems.append(
            "aide.backend.provider_to_query_func holds a non-callable value"
        )

    if problems:
        raise RuntimeError(
            "AIDE backend signature smoke test failed (the SHA-pinned AIDE may "
            "have changed shape); refusing to install a dispatch override:\n  - "
            + "\n  - ".join(problems)
        )


def install_query_override(
    aide_backend_module: Any,
    provider_key: str,
    loom_query: QueryFunc,
) -> None:
    """Install ``loom_query`` into AIDE's dispatch table for ``provider_key``.

    Runs :func:`smoke_test_aide_backend` first, then binds ``loom_query`` into
    ``aide.backend.provider_to_query_func[provider_key]`` so AIDE's existing
    ``query(...)`` dispatch (which routes by model name to a provider key) calls
    our function. Idempotent: installing the same callable twice is harmless.

    The caller is responsible for choosing a ``model_name`` whose
    ``determine_provider`` routes to ``provider_key`` (e.g. a ``claude-`` sentinel
    routes to ``"anthropic"``).

    Args:
        aide_backend_module: The imported ``aide.backend`` module to patch.
        provider_key: The key in ``provider_to_query_func`` to override (e.g.
            ``"anthropic"`` or ``"openai"``).
        loom_query: A query func honoring AIDE's
            ``(output, req_time, in_tokens, out_tokens, info)`` contract.
    """
    smoke_test_aide_backend(aide_backend_module)
    aide_backend_module.provider_to_query_func[provider_key] = loom_query


def coerce_review_json(text: str) -> dict:
    """Best-effort coerce free-form CLI ``text`` into AIDE's submit_review shape.

    A ``cli-bridge`` feedback route returns text; AIDE expects the parsed dict of
    ``func_spec.json_schema`` (``aide.agent.review_func_spec``). This extracts the
    first JSON object from ``text`` and normalizes it to the five required keys,
    filling conservative defaults when a key is absent so the judge never crashes.

    Args:
        text: The raw CLI output (ideally a JSON object, possibly fenced or with
            surrounding prose).

    Returns:
        A dict with keys ``is_bug`` (bool), ``summary`` (str), ``metric``
        (float|None), ``lower_is_better`` (bool).
    """
    obj = _extract_first_json_object(text)
    if not isinstance(obj, dict):
        # No usable JSON: treat as a non-fatal review with the text as summary.
        obj = {}

    is_bug = bool(obj.get("is_bug", False))
    summary = obj.get("summary")
    if not isinstance(summary, str) or not summary:
        summary = text.strip() if isinstance(text, str) else ""

    metric: Optional[float]
    raw_metric = obj.get("metric", None)
    if raw_metric is None:
        metric = None
    else:
        try:
            metric = float(raw_metric)
        except (TypeError, ValueError):
            metric = None

    lower_is_better = bool(obj.get("lower_is_better", False))

    return {
        "is_bug": is_bug,
        "summary": summary,
        "metric": metric,
        "lower_is_better": lower_is_better,
    }


def review_output_schema() -> dict:
    """Return AIDE's submit_review JSON schema for CLIs that accept an explicit one.

    Read lazily from ``aide.agent.review_func_spec`` so the schema stays in lock-
    step with AIDE; falls back to a hand-mirrored copy of the same shape if the
    import is unavailable (e.g. AIDE not importable at the call site).

    Returns:
        The JSON schema dict for the ``submit_review`` function (object with the
        four required ``_REVIEW_KEYS`` properties).
    """
    try:
        from aide.agent import review_func_spec

        return dict(review_func_spec.json_schema)
    except Exception:  # pragma: no cover - fall back to a mirrored schema
        return {
            "type": "object",
            "properties": {
                "is_bug": {"type": "boolean"},
                "summary": {"type": "string"},
                "metric": {"type": "number"},
                "lower_is_better": {"type": "boolean"},
            },
            "required": list(_REVIEW_KEYS),
        }


def _extract_first_json_object(text: str) -> Any:
    """Return the first JSON object parsed from ``text``, or ``None``.

    Tries a direct parse first, then a fenced ```` ```json ```` block, then the
    first balanced ``{...}`` span. Tolerates surrounding prose that CLIs emit.

    Args:
        text: The candidate text.

    Returns:
        The parsed object, or ``None`` if nothing parses.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # 1) Whole string is JSON.
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) A fenced code block (```json ... ``` or ``` ... ```).
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass

    # 3) First balanced top-level object span.
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except Exception:
                        break
        start = text.find("{", start + 1)

    return None


__all__ = [
    "smoke_test_aide_backend",
    "install_query_override",
    "coerce_review_json",
    "review_output_schema",
]
