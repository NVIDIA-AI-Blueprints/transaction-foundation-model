"""The agent face — Anthropic-style tool schemas from the REGISTRY (DESIGN.md §2.1, §5.3).

One verb declaration generates both faces. This module emits the agent tool
schemas (``loom.<verb>``) from the same ``REGISTRY`` the CLI enumerates, and
dispatches a tool call to the verb ``fn``, returning the identical
:class:`~loom.types.VerbResult` the CLI ``--json`` face produces.

The ``confirm_token`` handshake (§5.3) is wired here as the shape gated verbs use:
a PLAN result carries a single-use, plan-hash-scoped, expiring ``confirm_token``;
the agent makes a SECOND call passing it back. No verb is gated in this Phase-0
slice (tokenize/ingest/baseline are cheap CPU writes), but the round-trip shape is
present so launch verbs inherit it unchanged.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from pathlib import Path
from typing import Any, Optional

from .registry import REGISTRY, Verb, VerbContext
from .store import default_store
from .types import CapabilityMode, Status, Tier, VerbResult


def tool_schema(verb: Verb) -> dict[str, Any]:
    """Emit one Anthropic-style tool schema for a verb (name/description/input).

    ``disable-model-invocation`` is set True for irreversible / launch-and-track
    verbs so an agent structurally cannot fire money/destruction (§4, §5.2). None
    of the Phase-0 verbs are gated, but the flag is derived from tier/capability so
    launch verbs inherit it automatically."""
    gated = (
        verb.tier is Tier.IRREVERSIBLE
        or verb.capability_mode is CapabilityMode.LAUNCH_AND_TRACK
    )
    return {
        "name": f"loom.{verb.name}",
        "description": verb.summary,
        "input_schema": verb.params,
        # Loom-specific metadata the agent runtime reads (not part of the
        # Anthropic schema proper, but carried alongside it).
        "_loom": {
            "tier": verb.tier.value,
            "capability_mode": verb.capability_mode.value,
            "disable_model_invocation": gated,
        },
    }


def all_tool_schemas() -> list[dict[str, Any]]:
    """Emit tool schemas for every registered verb."""
    return [tool_schema(v) for v in REGISTRY.values()]


def dispatch(
    name: str,
    input_json: dict[str, Any],
    *,
    confirm_token: Optional[str] = None,
    experiment: Optional[str] = None,
) -> VerbResult:
    """Dispatch an agent tool call ``loom.<verb>(input_json)`` to the verb ``fn``.

    Accepts either ``"tokenize"`` or ``"loom.tokenize"``. Builds an agent-driver
    :class:`VerbContext` (interactive=False — an agent never owns a TTY, §7.5) and
    threads the optional ``confirm_token`` for the second-call handshake. Returns
    the dual-driver :class:`VerbResult` (the agent serializes it via
    ``.to_json()``)."""
    verb_name = name[len("loom.") :] if name.startswith("loom.") else name
    if verb_name not in REGISTRY:
        return VerbResult(
            verb=verb_name,
            status=Status.FAIL,
            verdict=__import__("loom.types", fromlist=["Verdict"]).Verdict.FAIL,
            tier=Tier.READ_ONLY,
            capability_mode=CapabilityMode.NONE,
            summary=f"unknown verb: {verb_name!r}",
        )
    verb = REGISTRY[verb_name]
    ctx = VerbContext(
        store=default_store(),
        experiment=experiment or input_json.get("experiment"),
        driver="agent",
        interactive=False,
        confirm_token=confirm_token or input_json.get("confirm_token"),
    )
    return verb.fn(dict(input_json), ctx)


_CONFIRM_TOKEN_TTL_SEC = 15 * 60  # 15-minute expiry (§5.3 / ARCHITECTURE §1)
_CONFIRM_NONCE_KIND = "_ConfirmNonce"  # store kind where burned nonces live


def _confirm_secret() -> bytes:
    """The HMAC key. A per-workspace secret persisted under the store root so a
    token minted in one process validates in the next (the PLAN→confirm round-trip
    spans two calls), but is unforgeable without filesystem access. Falls back to a
    process-stable secret when the workspace is unwritable."""
    import os

    base = Path(os.environ.get("LOOM_WORKSPACE") or os.getcwd()) / ".loom"
    secret_path = base / "_confirm_secret"
    try:
        base.mkdir(parents=True, exist_ok=True)
        if secret_path.exists():
            return secret_path.read_bytes()
        secret = secrets.token_bytes(32)
        # Atomic-ish create; if a concurrent driver won the race, read theirs.
        try:
            with open(secret_path, "xb") as fh:
                fh.write(secret)
        except FileExistsError:  # pragma: no cover - concurrent create
            return secret_path.read_bytes()
        return secret
    except OSError:  # pragma: no cover - unwritable workspace
        return b"loom-confirm-fallback-secret"


def make_confirm_token(plan_hash: str) -> str:
    """Mint a single-use, plan-hash-scoped, expiring confirm_token (§5.3).

    Shape: ``"<plan_hash>.<expiry_epoch>.<nonce>.<hmac>"`` where the HMAC (SHA-256)
    is taken over ``(plan_hash, expiry, nonce)`` with the per-workspace secret.
    Default 15-min expiry; the ``nonce`` is burned in the store on first
    validation so the token is single-use."""
    expiry = int(time.time()) + _CONFIRM_TOKEN_TTL_SEC
    nonce = secrets.token_hex(16)
    mac = _confirm_mac(plan_hash, expiry, nonce)
    return f"{plan_hash}.{expiry}.{nonce}.{mac}"


def _confirm_mac(plan_hash: str, expiry: int, nonce: str) -> str:
    msg = f"{plan_hash}.{expiry}.{nonce}".encode("utf-8")
    return hmac.new(_confirm_secret(), msg, hashlib.sha256).hexdigest()


def validate_confirm_token(token: Optional[str], plan_hash: str) -> bool:
    """Validate a confirm_token against a compiled-plan hash (§5.3, §9).

    Returns True iff the token is well-formed, its HMAC verifies, it is bound to
    THIS ``plan_hash``, it has not expired (15-min window), and its nonce has not
    already been burned. A successful validation BURNS the nonce in the store
    (single-use): a replay of the same token returns False."""
    if not token:
        return False
    parts = token.split(".")
    if len(parts) != 4:
        return False
    tok_plan, expiry_s, nonce, mac = parts
    # Plan-hash-scoped: a token for a different plan never validates.
    if not hmac.compare_digest(tok_plan, plan_hash):
        return False
    try:
        expiry = int(expiry_s)
    except ValueError:
        return False
    # Unforgeable: the HMAC must match (constant-time compare).
    if not hmac.compare_digest(mac, _confirm_mac(tok_plan, expiry, nonce)):
        return False
    # Unexpired (15-min window).
    if time.time() > expiry:
        return False
    # Single-use: burn the nonce in the store; a replay finds it already burned.
    return _burn_nonce(nonce, plan_hash)


def _burn_nonce(nonce: str, plan_hash: str) -> bool:
    """Atomically record a nonce as consumed; return True iff it was fresh.

    Uses an exclusive-create file under the workspace store as the burn ledger —
    the same local stand-in for an atomic metadata service the ObjectStore uses.
    If the directory is unwritable the token is accepted once (best-effort)."""
    import os

    burn_dir = Path(os.environ.get("LOOM_WORKSPACE") or os.getcwd()) / ".loom" / _CONFIRM_NONCE_KIND
    try:
        burn_dir.mkdir(parents=True, exist_ok=True)
        marker = burn_dir / nonce
        with open(marker, "xb") as fh:  # O_EXCL: fails iff already burned
            fh.write(plan_hash.encode("utf-8"))
        return True
    except FileExistsError:
        return False  # replay
    except OSError:  # pragma: no cover - unwritable workspace
        return True
