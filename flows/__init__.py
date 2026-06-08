"""Metaflow flows used by Loom execution providers.

This package holds the *static* Metaflow ``FlowSpec`` classes Loom drives. There
is exactly one flow in v0.1 -- :class:`flows.eval_candidate.EvalCandidate` --
and it is run as a subprocess via ``metaflow.Runner`` by the Metaflow execution
provider (see :mod:`loom.providers.metaflow_exec`).

Design invariant (mirrors the repo CLAUDE.md): a *candidate* solution is never
turned into a new flow. There is ONE flow class, and each candidate enters it as
**data** (an ``IncludeFile`` parameter), so the flow definition is stable across
every evaluation.

Importing this package must not require Metaflow to be installed: the flow module
imports ``metaflow`` at its top level (a flow file genuinely needs it), so it is
imported lazily by the provider rather than eagerly here. We therefore keep this
package ``__init__`` import-light and expose only the path helper below.
"""

from __future__ import annotations

import os

#: Absolute path to the static evaluation flow file. The Metaflow ``Runner``
#: takes a flow *file path*, so the provider resolves the flow this way rather
#: than importing the flow class (which would require Metaflow at import time).
EVAL_CANDIDATE_FLOW_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "eval_candidate.py"
)

__all__ = ["EVAL_CANDIDATE_FLOW_PATH"]
