"""Claude subscription model provider (``"claude-subscription"``) — a CLI bridge.

This provider drives the user's **own local** ``claude`` CLI (Claude Code) instead
of calling an API with a key. It is a ``cli-bridge`` kind: at run setup it installs
a dispatch override into ``aide.backend.provider_to_query_func`` so AIDE's existing
``query(...)`` dispatch shells out to ``claude -p`` for every model call.

Terms-of-service / posture
--------------------------
Loom **never handles claude.ai credentials**. It only invokes the ``claude`` binary
that the user has already installed and logged in on their own machine (via
``claude /login`` or ``claude setup-token``); authentication, session, and account
state live entirely inside that CLI. Loom passes the prompt in and reads the answer
out — it does not see, store, or transmit any subscription credential. Running your
own logged-in CLI to drive your own work is the intended use of that CLI; you remain
responsible for using it within the CLI's own terms.

Metering reality (as of 2026-06-15)
-----------------------------------
Driving the ``claude`` CLI consumes a **separate Agent SDK / subscription credit
pool**, metered independently from a raw Anthropic ``ANTHROPIC_API_KEY``. Heavy AIDE
runs (many steps × draft/debug/improve, each a full LLM call, plus a judge call per
executed node) can exhaust that pool quickly; for sustained or large runs an API key
(the ``anthropic-api`` provider) is often the better fit. This provider is best for
light, interactive use of an existing subscription.

The feedback role is judge-capable: the CLI returns text, which the dispatch
wrapper coerces into AIDE's ``submit_review`` JSON shape.

Testability
-----------
Live end-to-end calls require a logged-in local ``claude`` CLI and therefore
cannot be unit-tested in CI. The pure helpers (prompt composition, JSON
coercion in :mod:`loom.providers.model._dispatch`, preflight on a fake PATH/HOME)
are testable in isolation; the actual ``subprocess`` call is not.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import MutableMapping

from loom.config import LoomConfig
from loom.providers.model import ModelProvider, ModelRoute
from loom.providers.model import _dispatch
from loom.registry import register_model

# Sentinel routed to AIDE's "anthropic" provider key (it starts with "claude-")
# so determine_provider sends it to the slot our dispatch override occupies.
_SENTINEL_MODEL = "claude-code-subscription"


@register_model("claude-subscription")
class ClaudeSubscriptionModelProvider(ModelProvider):
    """Drive the user's local ``claude`` CLI as the LLM backend.

    Attributes:
        name: Registry name, ``"claude-subscription"``.
        config: The active :class:`~loom.config.LoomConfig`.
    """

    name = "claude-subscription"

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
            A ``cli-bridge`` :class:`ModelRoute` with a ``claude-`` sentinel model
            name and no ``key_env`` (the CLI owns authentication).
        """
        return ModelRoute(
            model_name=_SENTINEL_MODEL,
            base_url=None,
            key_env=None,
            judge_capable=True,
            kind="cli-bridge",
        )

    def prepare_env(self, env: MutableMapping[str, str]) -> None:
        """No env materialization: the ``claude`` CLI owns its own auth.

        We must NOT set ``OPENAI_BASE_URL`` (the sentinel routes by name to the
        anthropic slot our dispatch override occupies). The CLI reads its own
        login state (``CLAUDE_CODE_OAUTH_TOKEN`` or ``~/.claude``); Loom never
        touches claude.ai credentials.

        Args:
            env: The mutable environment mapping (typically ``os.environ``).
        """
        return None

    def preflight(self, role: str) -> list[str]:
        """Check the ``claude`` binary is on PATH and a login signal is present.

        Args:
            role: ``"code"`` or ``"feedback"`` (same checks for both).

        Returns:
            Hint strings (empty if the CLI looks installed and logged in).
        """
        hints: list[str] = []
        if shutil.which("claude") is None:
            hints.append(
                "the 'claude' CLI is not on PATH. Install Claude Code and run "
                "'claude /login' (or 'claude setup-token'); see the README "
                "'Use your Claude/Codex subscription' section."
            )
            # Without the binary, the login check below is moot.
            return hints

        if not self._has_login_signal():
            hints.append(
                "no Claude login signal found (CLAUDE_CODE_OAUTH_TOKEN unset and "
                "no ~/.claude.json / ~/.claude credentials). Run 'claude /login' "
                "or 'claude setup-token'; see the README "
                "'Use your Claude/Codex subscription' section."
            )
        return hints

    @staticmethod
    def _has_login_signal() -> bool:
        """Return whether a Claude CLI login signal is detectable.

        Checks the OAuth token env var first, then the on-disk credential
        locations the CLI uses.

        Returns:
            ``True`` if a login signal is present, else ``False``.
        """
        if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            return True
        home = Path.home()
        if (home / ".claude.json").exists():
            return True
        claude_dir = home / ".claude"
        # The dir alone, or a credentials file within it, counts as a signal.
        if claude_dir.is_dir():
            return True
        return False

    def install_dispatch_override(self, aide_backend_module: object) -> None:
        """Install a ``loom_query`` that shells out to ``claude -p`` (stdin prompt).

        The wrapper honors AIDE's ``(output, req_time, in_tokens, out_tokens,
        info)`` contract. The prompt is fed on stdin with ``--output-format text``
        so large compiled prompts are passed safely and a plain-text answer is
        returned. For ``func_spec`` calls (the feedback judge) it coerces the CLI's
        text answer into the parsed ``submit_review`` dict via
        :func:`loom.providers.model._dispatch.coerce_review_json`. It is installed
        under the ``"anthropic"`` provider key, which the ``claude-`` sentinel
        routes to.

        Args:
            aide_backend_module: The imported ``aide.backend`` module to patch.
        """
        import subprocess
        import time

        def loom_query(system_message, user_message, func_spec=None, **model_kwargs):
            prompt = _compose_prompt(system_message, user_message, func_spec)

            # Pass the prompt on stdin (AIDE's compiled prompts can be very large
            # and contain shell-special characters; stdin avoids argv length /
            # quoting limits). ``--print``/``-p`` is non-interactive;
            # ``--output-format text`` pins a plain-text answer (the parseable
            # contract). We deliberately do NOT pass ``--bare`` -- bare mode forces
            # ANTHROPIC_API_KEY-only auth and never reads OAuth, which would defeat
            # the whole subscription posture this provider exists for.
            cmd = ["claude", "-p", "--output-format", "text"]
            t0 = time.time()
            proc = subprocess.run(  # noqa: S603 - user's own trusted CLI
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                check=False,
            )
            req_time = time.time() - t0
            text = (proc.stdout or "").strip()
            if proc.returncode != 0 and not text:
                text = (proc.stderr or "").strip()

            if func_spec is not None:
                output = _dispatch.coerce_review_json(text)
            else:
                output = text

            info = {"model": _SENTINEL_MODEL, "returncode": proc.returncode}
            # Token counts are not exposed by the CLI; report 0 (informational).
            return output, req_time, 0, 0, info

        _dispatch.install_query_override(aide_backend_module, "anthropic", loom_query)


def _compose_prompt(system_message, user_message, func_spec) -> str:
    """Compose a single prompt string for the ``claude -p`` invocation.

    AIDE has already compiled the messages to Markdown strings. For a
    ``func_spec`` (judge) call, an explicit instruction to emit only the JSON
    object for the function's schema is appended so the text answer can be coerced
    back into AIDE's expected dict.

    Args:
        system_message: The compiled system prompt (or ``None``).
        user_message: The compiled user prompt (or ``None``).
        func_spec: The optional AIDE ``FunctionSpec`` (present for the judge).

    Returns:
        The prompt to pass to ``claude -p``.
    """
    import json

    parts: list[str] = []
    if system_message:
        parts.append(str(system_message))
    if user_message:
        parts.append(str(user_message))

    if func_spec is not None:
        schema = getattr(func_spec, "json_schema", _dispatch.review_output_schema())
        parts.append(
            "Respond with ONLY a single JSON object (no prose, no code fences) "
            "matching this JSON schema:\n" + json.dumps(schema)
        )

    return "\n\n".join(parts)


__all__ = ["ClaudeSubscriptionModelProvider"]
