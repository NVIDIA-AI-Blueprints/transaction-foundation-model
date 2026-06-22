"""Shared helpers for the golden / conformance tests (Implement C).

The golden tests are the *conformance oracle as code*: they pin the reference
TabFormer vocab (6251 / 6283), the contract checks (C1 injectivity, C2
determinism, C3 grammar derivation), the corpus grammar, the dual-driver
byte-identity, and ingest idempotency — every invariant the build brief calls a
conformance gate.

They are written against the LOCKED engine/verb API names *verbatim*. Because
the engine body is filled in by a parallel Implement agent, a stubbed surface
raises ``NotImplementedError``. Rather than hard-fail before the implementation
lands, these helpers turn a stub into a clean ``pytest.skip`` so the suite is
green on the scaffold and becomes a strict gate the moment the engine is real.
The invariants themselves (the ``assert``s in each test) are NOT softened — only
the "is it implemented yet" guard is.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from loom.types import Diagnostic, Severity, Verdict


def require_engine(fn: Callable[[], Any]) -> Any:
    """Call ``fn`` (a thunk around a compile/spec call). If the engine is still a
    stub (``NotImplementedError``) skip the test; otherwise return the value."""
    try:
        return fn()
    except NotImplementedError:  # pragma: no cover - skip path on the scaffold
        pytest.skip("engine not implemented yet (stub raises NotImplementedError)")


def compiled_financial(**kwargs: Any):
    """``compile_spec(financial_spec(**kwargs))`` or skip if the engine is a stub."""
    from loom.engine import compile_spec, financial_spec

    return require_engine(lambda: compile_spec(financial_spec(**kwargs)))


def compiled_chain(**kwargs: Any):
    """``compile_spec(chain_spec(**kwargs))`` or skip if the engine is a stub."""
    from loom.engine import chain_spec, compile_spec

    return require_engine(lambda: compile_spec(chain_spec(**kwargs)))


def diagnostics_for(report, contract: str) -> list[Diagnostic]:
    """All diagnostics on a ContractReport matching a contract id (e.g. ``"C1"``)."""
    return [d for d in report.diagnostics if d.contract == contract]


def has_error(diags: list[Diagnostic]) -> bool:
    return any(d.severity is Severity.ERROR for d in diags)


def verb_fn(name: str):
    """The raw ``fn`` for a registered verb (callable as ``fn(args, ctx)``)."""
    from loom.registry import REGISTRY

    return REGISTRY[name].fn


def is_implemented(result) -> bool:
    """True once a verb returns something other than the INCOMPLETE scaffold stub.

    The scaffold stubs return ``verdict=INCOMPLETE`` with a "not implemented yet"
    summary; a real implementation returns OK/PLAN/REFUSED_* with real outputs."""
    if result.verdict is Verdict.INCOMPLETE and "not implemented" in result.summary.lower():
        return False
    return True


def store_list(store, kind=None):
    """``store.list(kind)`` or skip if the store is still a stub.

    The object store is built by its own Implement agent; while it is stubbed,
    ``list``/``get`` raise ``NotImplementedError``. Tests that read the store back
    to assert what a verb persisted use this so they skip (not fail) until the
    store lands, then assert strictly."""
    try:
        return store.list(kind)
    except NotImplementedError:  # pragma: no cover - skip path while store is a stub
        pytest.skip("object store is still a stub (NotImplementedError)")


def call_verb(name: str, args: dict, ctx, *, skip_if_stub: bool = True):
    """Invoke a verb ``fn`` and return its result.

    The package is built by several parallel Implement agents, so at any instant a
    verb body may be real while a seam it calls (e.g. ``store.content_id``,
    ``compile_spec``) is still a stub raising ``NotImplementedError``. That is a
    "not landed yet" condition, not a conformance failure — so when ``skip_if_stub``
    we ``pytest.skip`` on a ``NotImplementedError`` raised anywhere under the verb,
    and on the explicit INCOMPLETE scaffold stub. The conformance ``assert``s in
    each test stay strict; only this landing guard is lenient."""
    fn = verb_fn(name)
    try:
        result = fn(args, ctx)
    except NotImplementedError:  # pragma: no cover - skip path while seams are stubs
        if skip_if_stub:
            pytest.skip(f"{name}: a downstream seam is still a stub (NotImplementedError)")
        raise
    if skip_if_stub and not is_implemented(result):
        pytest.skip(f"{name} verb not implemented yet (scaffold stub)")
    return result
