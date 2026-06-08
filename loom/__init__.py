"""Loom: a general-purpose, domain-neutral automated ML engine.

Loom follows a ports-and-adapters ("providers") architecture, analogous to
Kubernetes pluggable runtimes. ``loom-core`` defines the provider *interfaces*
(the two seams: a search "brain" and an MLOps execution "muscle"); concrete
adapters (AIDE for search, Metaflow/local for execution) plug into those seams
and are selected purely by configuration.

This top-level package is intentionally dependency-light: importing ``loom``
must not pull in any optional/heavy third-party dependency (AIDE, Metaflow,
pandas, ...). Submodules import those lazily, inside functions/methods, so that
core types remain importable in any environment.
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
