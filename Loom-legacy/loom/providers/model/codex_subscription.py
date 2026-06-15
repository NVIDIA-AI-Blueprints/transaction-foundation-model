"""Codex subscription model provider (``"codex-subscription"``) — a CLI bridge.

This provider drives the user's **own local** ``codex`` CLI (``codex exec``) instead
of calling an API with a key, reusing the credentials the user already established
at ``~/.codex/auth.json``. It is a ``cli-bridge`` kind: at run setup it installs a
dispatch override into ``aide.backend.provider_to_query_func`` so AIDE's existing
``query(...)`` dispatch shells out to ``codex exec`` for every model call.

Terms-of-service / posture
--------------------------
Login is **human-started**: the user runs the Codex/ChatGPT login flow themselves;
Loom never performs or stores that login. ``~/.codex/auth.json`` is a **full-account
credential** for the user's ChatGPT/Codex account — treat the file as sensitive;
Loom only lets the already-authenticated ``codex`` binary read it (Loom never reads
or transmits the file's contents). Subscription use is subject to a **rolling-window
usage cap** enforced by the provider, so a long AIDE run can hit that cap mid-search;
an API key (the ``openai-api`` provider) avoids the cap for heavy use.

The feedback role is judge-capable: ``codex exec`` is invoked with
``--output-schema`` derived from AIDE's ``submit_review`` ``func_spec.json_schema``,
forcing a JSON answer in exactly the judge's shape (with a text-coercion fallback).

Testability
-----------
Live end-to-end calls require a logged-in local ``codex`` CLI (a valid
``~/.codex/auth.json``) and therefore cannot be unit-tested in CI. The pure
helpers (prompt composition, schema selection, JSON coercion in
:mod:`loom.providers.model._dispatch`, preflight on a fake PATH/HOME) are testable
in isolation; the actual ``subprocess`` call is not.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import MutableMapping

from loom.config import LoomConfig
from loom.providers.model import ModelProvider, ModelRoute
from loom.providers.model import _dispatch
from loom.registry import register_model

# Sentinel routed to AIDE's "openai" provider key. ``determine_provider`` matches
# ``codex-mini-latest`` -> openai, so this lands in the slot our dispatch override
# occupies (we never reach the real OpenAI client because we replace the func).
_SENTINEL_MODEL = "codex-mini-latest"


@register_model("codex-subscription")
class CodexSubscriptionModelProvider(ModelProvider):
    """Drive the user's local ``codex exec`` CLI as the LLM backend.

    Attributes:
        name: Registry name, ``"codex-subscription"``.
        config: The active :class:`~loom.config.LoomConfig`.
    """

    name = "codex-subscription"

    def __init__(self, config: LoomConfig) -> None:
        """Initialize from a Loom config (no secret material is read).

        Args:
            config: The active configuration.
        """
        self.config = config

    def resolve(self, role: str) -> ModelRoute:
        """Return the CLI-bridge route for ``role``.

        Args:
            role: ``"code"`` or ``"feedback"``.

        Returns:
            A ``cli-bridge`` :class:`ModelRoute` with a sentinel model name (which
            routes to AIDE's openai slot) and no ``key_env`` (the CLI owns auth).
        """
        return ModelRoute(
            model_name=_SENTINEL_MODEL,
            base_url=None,
            key_env=None,
            judge_capable=True,
            kind="cli-bridge",
        )

    def prepare_env(self, env: MutableMapping[str, str]) -> None:
        """No env materialization: the ``codex`` CLI owns its own auth.

        We must NOT set ``OPENAI_BASE_URL`` (the sentinel routes by name to the
        openai slot our dispatch override occupies). The CLI reuses
        ``~/.codex/auth.json``; Loom never reads or transmits it.

        Args:
            env: The mutable environment mapping (typically ``os.environ``).
        """
        return None

    def preflight(self, role: str) -> list[str]:
        """Check the ``codex`` binary is on PATH and ``~/.codex/auth.json`` exists.

        Args:
            role: ``"code"`` or ``"feedback"`` (same checks for both).

        Returns:
            Hint strings (empty if the CLI looks installed and logged in).
        """
        hints: list[str] = []
        if shutil.which("codex") is None:
            hints.append(
                "the 'codex' CLI is not on PATH. Install Codex and sign in; see "
                "the README 'Use your Claude/Codex subscription' section."
            )
            return hints

        if not (Path.home() / ".codex" / "auth.json").exists():
            hints.append(
                "no ~/.codex/auth.json found. Start the Codex login flow yourself "
                "(human-started login), then re-run; see the README "
                "'Use your Claude/Codex subscription' section."
            )
        return hints

    def install_dispatch_override(self, aide_backend_module: object) -> None:
        """Install a ``loom_query`` that shells out to ``codex exec``.

        The wrapper honors AIDE's ``(output, req_time, in_tokens, out_tokens,
        info)`` contract. For ``func_spec`` calls (the feedback judge) it writes
        the judge's ``json_schema`` to a temp file and passes it via
        ``--output-schema`` so ``codex exec`` returns JSON in the submit_review
        shape; the result is parsed (with a text-coercion fallback). Installed
        under the ``"openai"`` provider key, which the sentinel routes to.

        Args:
            aide_backend_module: The imported ``aide.backend`` module to patch.
        """

        def loom_query(system_message, user_message, func_spec=None, **model_kwargs):
            prompt = _compose_prompt(system_message, user_message)

            cmd = ["codex", "exec"]
            schema_path: str | None = None
            if func_spec is not None:
                schema = getattr(
                    func_spec, "json_schema", _dispatch.review_output_schema()
                )
                fd, schema_path = tempfile.mkstemp(prefix="loom-codex-schema-", suffix=".json")
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(schema, fh)
                cmd += ["--output-schema", schema_path]
            cmd.append(prompt)

            t0 = time.time()
            try:
                proc = subprocess.run(  # noqa: S603 - user's own trusted CLI
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            finally:
                if schema_path is not None:
                    try:
                        os.unlink(schema_path)
                    except OSError:
                        pass
            req_time = time.time() - t0

            text = (proc.stdout or "").strip()
            if proc.returncode != 0 and not text:
                text = (proc.stderr or "").strip()

            if func_spec is not None:
                output = _dispatch.coerce_review_json(text)
            else:
                output = text

            info = {"model": _SENTINEL_MODEL, "returncode": proc.returncode}
            return output, req_time, 0, 0, info

        _dispatch.install_query_override(aide_backend_module, "openai", loom_query)


def _compose_prompt(system_message, user_message) -> str:
    """Compose a single prompt string for the ``codex exec`` invocation.

    AIDE has already compiled the messages to Markdown strings; ``codex exec``
    takes the prompt as a positional argument. (The judge's JSON shape is enforced
    out-of-band via ``--output-schema``, so no extra schema text is appended here.)

    Args:
        system_message: The compiled system prompt (or ``None``).
        user_message: The compiled user prompt (or ``None``).

    Returns:
        The prompt to pass to ``codex exec``.
    """
    parts: list[str] = []
    if system_message:
        parts.append(str(system_message))
    if user_message:
        parts.append(str(user_message))
    return "\n\n".join(parts)


__all__ = ["CodexSubscriptionModelProvider"]
