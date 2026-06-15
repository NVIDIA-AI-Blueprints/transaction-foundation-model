"""Replay the examples/ walkthroughs as a regression eval bed.

Each ``examples/<NN-name>/run.sh`` is a self-contained, self-checking recipe: it
generates deterministic synthetic data, ingests it under a uniquely-named
dataset, runs a keyless verb sequence with ``--json``, and asserts the outcomes
inline (exiting nonzero on any regression). This test discovers those scripts
and replays each one, asserting exit 0 -- so a drift in the verb ``--json``
contract or a regressed outcome turns the example red here.

The whole module **skips cleanly** when the engine / cluster is not reachable
(the same posture as ``tests/test_doctor.py``): the examples talk to a real
Metaflow datastore, so on a box with no cluster there is nothing to assert. The
reachability gate reuses the read-only ``loom datasets --json`` smoke -- if it
returns one JSON object with ``status == "ok"`` the engine + datastore are live;
anything else (nonzero exit, unparseable output, error status, missing engine)
skips the module.

So a bare ``pytest`` works against a locally-running datastore, the harness
sources the cluster env file (``/tmp/loom-cluster-env.sh`` by default,
overridable via ``$LOOM_CLUSTER_ENV``) into the env it hands the probe and every
replay -- but it never overrides a var the caller already set, so a hand-sourced
env still wins. If the file is absent and nothing is reachable, the module skips.

Keyless only: every discovered example uses verbs that run WITHOUT a model key,
so this needs no secrets in CI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# --- Locations ---------------------------------------------------------------
# The repo root is the parent of tests/; examples/ live beside it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"

# The verb entrypoint the examples default to (no PATH / console-script needed).
# run.sh reads $LOOM; we pin the same venv module form for the reachability probe.
_PYTHON = sys.executable
_LOOM = [_PYTHON, "-m", "loom"]

# A generous per-example budget: each ingests + runs a short verb sequence as
# real Metaflow runs, which is slower than a unit test but bounded.
_PER_EXAMPLE_TIMEOUT_S = 300

# The cluster env the examples need (Metaflow datastore: minio + local metadata).
# Operators source this before running; the harness sources it too so `pytest`
# works without a hand-set env -- but it NEVER overrides vars already in the
# caller's environment (an explicitly-sourced env wins).
_CLUSTER_ENV_FILE = Path(os.environ.get("LOOM_CLUSTER_ENV", "/tmp/loom-cluster-env.sh"))


def _example_env() -> dict[str, str]:
    """The environment used to probe + replay the examples.

    Starts from the caller's environment, then fills in any datastore vars from
    the cluster env file (``/tmp/loom-cluster-env.sh`` by default, overridable
    via ``$LOOM_CLUSTER_ENV``) WITHOUT clobbering anything already set. This lets
    a bare ``pytest`` reach a locally-running datastore while still honoring an
    env the operator sourced by hand. If the file is absent we just return the
    caller's env (and the reachability probe will skip the module cleanly).
    """
    env = os.environ.copy()
    if not _CLUSTER_ENV_FILE.is_file():
        return env
    try:
        # Source the file in a clean shell and dump the resulting environment as
        # NUL-delimited KEY=VALUE pairs, so values with newlines stay intact.
        proc = subprocess.run(
            ["bash", "-c", f'set -a; . "{_CLUSTER_ENV_FILE}"; env -0'],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return env
    if proc.returncode != 0:
        return env
    for pair in proc.stdout.split("\0"):
        if not pair or "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        # Don't clobber what the caller explicitly set.
        env.setdefault(key, value)
    return env


# Prepared once: the env handed to both the reachability probe and every replay.
_EXAMPLE_ENV = _example_env()


def _discover_examples() -> list[Path]:
    """Return every ``examples/<NN-name>/run.sh`` (the ``_template`` is skipped).

    Discovery is by convention: a numeric-prefixed sibling directory of
    ``examples/`` that ships a ``run.sh``. ``_template/`` is excluded (it is the
    skeleton, not a runnable example).
    """
    if not _EXAMPLES_DIR.is_dir():
        return []
    scripts: list[Path] = []
    for child in sorted(_EXAMPLES_DIR.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        run_sh = child / "run.sh"
        if run_sh.is_file():
            scripts.append(run_sh)
    return scripts


def _engine_reachable() -> bool:
    """Return whether the Loom engine + Metaflow datastore are reachable.

    The doctor-style smoke: run ``loom datasets --json`` and accept only a clean
    one-object response with ``status == "ok"``. Any failure mode -- the engine
    not importable, a nonzero exit, unparseable stdout, or a non-ok status --
    means there is no cluster to replay the examples against, so the module
    skips. Inherits the caller's environment (the cluster env the operator
    sourced), exactly as ``run.sh`` does.
    """
    try:
        proc = subprocess.run(
            [*_LOOM, "datasets", "--json"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            env=_EXAMPLE_ENV,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if proc.returncode != 0:
        return False
    try:
        obj = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(obj, dict) and obj.get("status") == "ok"


# Module-level skip: no engine / cluster -> nothing to replay. Evaluated once.
pytestmark = pytest.mark.skipif(
    not _engine_reachable(),
    reason=(
        "Loom engine / Metaflow datastore not reachable (`loom datasets --json` "
        "did not return status=ok); skipping the examples regression bed. Source "
        "the cluster env (e.g. /tmp/loom-cluster-env.sh) to enable it."
    ),
)

_EXAMPLE_SCRIPTS = _discover_examples()


@pytest.mark.parametrize(
    "run_sh",
    _EXAMPLE_SCRIPTS,
    ids=[s.parent.name for s in _EXAMPLE_SCRIPTS],
)
def test_example_runs_and_self_asserts(run_sh: Path) -> None:
    """Replay one example's ``run.sh`` and assert it exits 0 (no regression).

    ``run.sh`` is ``set -euo pipefail`` and does its own ``--json`` assertions, so
    a clean exit 0 means every asserted verb outcome held. On a nonzero exit we
    surface the script's stdout + stderr (which carries the failing
    ``_assert_json`` message) so the regression is legible in the test log.
    """
    proc = subprocess.run(
        ["bash", str(run_sh)],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=_PER_EXAMPLE_TIMEOUT_S,
        env=_EXAMPLE_ENV,
    )
    assert proc.returncode == 0, (
        f"example {run_sh.parent.name} regressed (exit {proc.returncode}).\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


def test_examples_are_discovered() -> None:
    """Guard the discovery: every discovered example ships the canonical trio.

    SCAFFOLD posture: the six build agents populate ``examples/01..06`` after this
    skeleton lands, so an empty tree is not yet a failure -- it SKIPS (the suite
    must stay green during scaffolding). Once any example exists, it must carry
    the canonical layout (``run.sh`` + ``make_data.py`` + ``README.md``), which
    this asserts. The ``_template`` directory is intentionally NOT counted.
    """
    if not _EXAMPLE_SCRIPTS:
        pytest.skip(
            "no examples/<NN-name>/run.sh yet -- the build agents populate "
            "examples/01..06; _template/ is excluded by design."
        )
    for run_sh in _EXAMPLE_SCRIPTS:
        example = run_sh.parent
        assert (example / "make_data.py").is_file(), f"{example.name} missing make_data.py"
        assert (example / "README.md").is_file(), f"{example.name} missing README.md"
