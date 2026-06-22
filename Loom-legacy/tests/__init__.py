"""Loom test suite.

The pure-Python tests in this package (corpus + local execution + registry
resolution) run with no heavy optional dependencies installed. Tests that
exercise the AIDE or Metaflow adapters use :func:`pytest.importorskip` so they
are skipped — never errored — when those packages are absent.
"""
