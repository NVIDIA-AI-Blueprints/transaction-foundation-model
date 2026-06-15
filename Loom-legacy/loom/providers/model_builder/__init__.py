"""Model-builder providers (the fourth Loom seam) and adapter registration.

Loom is ports-and-adapters. Alongside the *search* ("brain", AIDE), *execution*
("muscle", Metaflow/local), and *model* (LLM backend) ports, this package holds
the concrete adapters for the third heavy backend — the
:class:`~loom.providers.ModelBuilderProvider` port (training/serving). The ABC
itself, plus the ``OBJECTIVES``/``BUDGETS``/``MODES`` seam frozensets, live in
:mod:`loom.providers`; this package only carries the adapters:

* ``nemo`` -- the default ``model_builder_provider``: a lowering *compiler* that
  translates Loom DS-intent into a NeMo launch PLAN and gates the real GPU
  launch behind ``--launch`` (refusing cleanly when ``gpu_target`` is ``None``).
* ``local`` -- a torch-free CPU PPMI+TruncatedSVD adapter that is the testable /
  conformance default path, exercising ``tokenize -> pretrain -> embed ->
  finetune -> evaluate -> serve`` end-to-end with zero new heavy deps.

The two adapter modules (``local.py`` and ``nemo.py``) are written by later
agents and each self-registers via its ``@register_model_builder`` decorator on
import. They are imported at the *bottom*, each inside its own ``try/except``, so
a missing optional dependency (NeMo, torch, ...) — or the modules' absence while
core is being built — cannot break ``import loom.providers.model_builder`` or, by
extension, ``loom`` core.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Built-in adapter registration.
#
# Each import is guarded independently: a missing optional dependency for one
# adapter (or the module not yet existing) must NOT prevent the others (or core)
# from importing. Importing each module triggers its ``@register_model_builder``
# decorator side effects.
# ---------------------------------------------------------------------------

try:  # CPU PPMI+TruncatedSVD stand-in ("local") -- the torch-free default path.
    from . import local  # noqa: F401
except Exception:  # pragma: no cover - optional dependency guard
    pass

try:  # NeMo lowering compiler ("nemo") -- the default provider, gated launch.
    from . import nemo  # noqa: F401
except Exception:  # pragma: no cover - optional dependency guard
    pass
