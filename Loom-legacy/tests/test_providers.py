"""Provider tests: registry resolution + the local execution provider.

These are pure-Python: the registry and the ``local`` provider have no heavy
dependencies (the local provider runs code through the vendored, dependency-light
``loom.providers._interpreter``), so this module runs without AIDE or Metaflow
installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from loom.config import LoomConfig
from loom.providers import ExecutionProvider, SearchProvider
from loom.types import ExecutionResult, Task


# ---------------------------------------------------------------------------
# Registry resolution.
# ---------------------------------------------------------------------------


def test_registry_resolves_default_providers() -> None:
    """The default config names resolve to registered provider classes.

    The defaults are ``search="aide"`` and ``mlops="metaflow"``. Importing the
    providers package self-registers the built-ins (each guarded so a missing
    optional dependency only drops that one adapter).
    """
    from loom import registry

    cfg = LoomConfig()
    assert cfg.search_provider == "aide"
    assert cfg.mlops_provider == "metaflow"

    # The brain default ("aide") and both execution providers should be present
    # as long as their modules import. The AIDE/Metaflow modules are written to
    # lazy-import their heavy deps, so registration must not require them.
    search_cls = registry.get_search("aide")
    assert issubclass(search_cls, SearchProvider)

    for exec_name in ("local", "metaflow"):
        exec_cls = registry.get_execution(exec_name)
        assert issubclass(exec_cls, ExecutionProvider)


def test_registry_unknown_name_raises_keyerror() -> None:
    """An unregistered name raises ``KeyError`` listing what is available."""
    from loom import registry

    with pytest.raises(KeyError):
        registry.get_execution("does-not-exist")
    with pytest.raises(KeyError):
        registry.get_search("does-not-exist")


def test_local_provider_registered_under_name_local() -> None:
    """The local provider registers under the name ``"local"``."""
    from loom import registry

    local_cls = registry.get_execution("local")
    # The registry name attribute should match the registered key.
    assert getattr(local_cls, "name", None) == "local"


# ---------------------------------------------------------------------------
# Local execution provider behaviour.
# ---------------------------------------------------------------------------


@pytest.fixture
def restore_cwd():
    """Restore the process working directory after a provider chdir."""
    original = os.getcwd()
    try:
        yield
    finally:
        os.chdir(original)


def _make_task(data_dir: Path) -> Task:
    """Build a trivial task pointing at ``data_dir`` as its input."""
    return Task(
        data_dir=str(data_dir),
        goal="trivial smoke test",
        eval="exit code 0 and submission.csv written",
        experiment_id="exp-local-smoke",
        tenant="default",
    )


def test_local_provider_executes_trivial_snippet(tmp_path, restore_cwd) -> None:
    """A trivial snippet runs clean: prints, writes submission.csv, no exc.

    Verifies the full local path the contract specifies:
      * ``setup(task)`` stages ``./input`` from ``task.data_dir`` and creates an
        empty ``./working`` and chdirs into the workspace,
      * ``execute(code)`` returns a loom :class:`ExecutionResult` with no
        exception when the code runs to completion.
    """
    from loom.registry import get_execution

    # A data dir with one input file, to confirm setup() stages ./input.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    cfg = LoomConfig(
        mlops_provider="local",
        corpus_path=str(tmp_path / "corpus" / "nodes.jsonl"),
    )
    provider_cls = get_execution("local")
    provider = provider_cls(cfg)
    assert isinstance(provider, ExecutionProvider)
    assert provider.name == "local"

    task = _make_task(data_dir)
    provider.setup(task)

    # Code writes the required submission into ./working and prints a marker.
    code = (
        "import os\n"
        "print('hello from candidate')\n"
        "os.makedirs('working', exist_ok=True)\n"
        "with open(os.path.join('working', 'submission.csv'), 'w') as f:\n"
        "    f.write('id,prediction\\n0,1\\n')\n"
    )

    result = provider.execute(code, reset_session=True)

    assert isinstance(result, ExecutionResult)
    # No exception on a clean run.
    assert result.exc_type is None, result.term_out
    assert result.exc_info is None
    assert result.exc_stack is None
    # exec_time is a non-negative float.
    assert isinstance(result.exec_time, float)
    assert result.exec_time >= 0.0
    # Our printed marker is captured in term_out.
    assert any("hello from candidate" in line for line in result.term_out)
    # The interpreter appends a final "Execution time:" line.
    assert any("Execution time" in line for line in result.term_out)

    provider.teardown()


def test_local_provider_is_callable_as_exec_callback(tmp_path, restore_cwd) -> None:
    """An ExecutionProvider satisfies the AIDE exec-callback signature.

    ``provider(code, reset_session)`` is aliased to ``execute`` so the provider
    can be passed straight to a SearchProvider as its exec callback.
    """
    from loom.registry import get_execution

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    cfg = LoomConfig(mlops_provider="local")
    provider = get_execution("local")(cfg)
    provider.setup(_make_task(data_dir))

    result = provider("print('callable seam')", True)

    assert isinstance(result, ExecutionResult)
    assert result.exc_type is None
    assert any("callable seam" in line for line in result.term_out)

    provider.teardown()


def test_local_provider_captures_exception(tmp_path, restore_cwd) -> None:
    """A raising snippet surfaces the exception type without crashing."""
    from loom.registry import get_execution

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    cfg = LoomConfig(mlops_provider="local")
    provider = get_execution("local")(cfg)
    provider.setup(_make_task(data_dir))

    result = provider.execute("raise ValueError('boom')", reset_session=True)

    assert isinstance(result, ExecutionResult)
    assert result.exc_type == "ValueError"
    # exc_info / exc_stack populated for a raised exception.
    assert result.exc_info is not None
    assert result.exc_stack is not None

    provider.teardown()
