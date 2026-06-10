"""Tests for ``loom notebook`` — the GPU-backed remote-Jupyter launcher.

Mirrors ``test_modal_launcher.py``: the pure submission-spec build needs no
``modal`` and no account, and the live launch is exercised with a stubbed ``modal``
module so nothing touches the network or a GPU. The remote ``_serve`` body (which
starts JupyterLab + opens the tunnel) runs only on the GPU and is out of scope here;
what we verify is the routing/spec + that the launcher wires up the right Modal app
and image and refuses cleanly when ``modal`` is absent.
"""

from __future__ import annotations

import sys
import types

import pytest

from loom import notebook as nb
from loom.notebook import (
    DEFAULT_NOTEBOOK_PORT,
    NotebookSubmission,
    build_notebook_submission,
    launch_notebook,
    routes_to_modal,
)
from loom.providers.model_builder._modal_launcher import (
    DEFAULT_MODAL_APP,
    DEFAULT_NEMO_IMAGE,
    MODAL_GPU,
)


# ---------------------------------------------------------------------------
# Pure submission-spec construction (NO modal installed / no account).
# ---------------------------------------------------------------------------


def test_build_notebook_submission_spec_is_h100_nemo_image_port_and_datastore() -> None:
    """The pure spec carries GPU=H100, the NeMo NGC image, the Jupyter port, datastore on."""
    sub = build_notebook_submission("modal")
    assert isinstance(sub, NotebookSubmission)
    assert sub.gpu == MODAL_GPU == "H100"
    assert sub.image == DEFAULT_NEMO_IMAGE  # the same container training uses
    assert sub.app_name == DEFAULT_MODAL_APP
    assert sub.port == DEFAULT_NOTEBOOK_PORT == 8888
    assert sub.mount_datastore is True
    assert sub.timeout_seconds == 4 * 3600  # the interactive default


def test_build_notebook_submission_honors_env_image_override(monkeypatch) -> None:
    """``LOOM_NEMO_IMAGE`` overrides the container (same env seam as training)."""
    monkeypatch.setenv("LOOM_NEMO_IMAGE", "nvcr.io/nvidia/nemo:25.01")
    assert build_notebook_submission("modal").image == "nvcr.io/nvidia/nemo:25.01"


def test_build_notebook_submission_app_target_and_no_datastore() -> None:
    """``modal://<app>`` selects the app; ``mount_datastore=False`` is carried."""
    sub = build_notebook_submission("modal://team-burst", mount_datastore=False)
    assert sub.app_name == "team-burst"
    assert sub.mount_datastore is False
    assert sub.gpu_target == "modal://team-burst"


def test_notebook_timeout_env_override(monkeypatch) -> None:
    """``LOOM_NOTEBOOK_TIMEOUT`` overrides the session ceiling; bad values fall back."""
    monkeypatch.setenv("LOOM_NOTEBOOK_TIMEOUT", "7200")
    assert build_notebook_submission("modal").timeout_seconds == 7200
    monkeypatch.setenv("LOOM_NOTEBOOK_TIMEOUT", "not-a-number")
    assert build_notebook_submission("modal").timeout_seconds == 4 * 3600


def test_routes_to_modal() -> None:
    """Only ``modal`` / ``modal://app`` route to the launcher."""
    assert routes_to_modal("modal") is True
    assert routes_to_modal("modal://x") is True
    assert routes_to_modal("slurm") is False
    assert routes_to_modal("") is False


# ---------------------------------------------------------------------------
# Live launch with a stubbed modal (offline) — wires the right app + image.
# ---------------------------------------------------------------------------


class _FakeBuiltImage:
    def __init__(self, ref: str) -> None:
        self.ref = ref

    def pip_install(self, *_a, **_k):  # chainable, like modal.Image
        return self


class _FakeFunction:
    def remote(self, *_a, **_k):  # mimics modal's .remote(); the GPU body is a no-op here
        return None


class _FakeAppRun:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeApp:
    def __init__(self, name: str, sink: dict) -> None:
        self.name = name
        sink["app_name"] = name

    def function(self, **_kwargs):
        def _decorator(_fn):
            return _FakeFunction()

        return _decorator

    def run(self):
        return _FakeAppRun()


def _install_fake_modal(monkeypatch) -> dict:
    """Install a minimal fake ``modal`` so launch_notebook runs offline; returns a sink."""
    sink: dict = {}
    fake = types.ModuleType("modal")

    class _Image:
        @staticmethod
        def from_registry(ref):
            sink["image"] = ref
            return _FakeBuiltImage(ref)

    fake.Image = _Image
    fake.App = lambda name: _FakeApp(name, sink)
    fake.forward = lambda *_a, **_k: None  # not reached (lives in the remote body)
    monkeypatch.setitem(sys.modules, "modal", fake)
    return sink


def test_launch_notebook_wires_app_and_nemo_image(monkeypatch) -> None:
    """launch_notebook builds the Modal app + NeMo image and returns 0 (offline)."""
    sink = _install_fake_modal(monkeypatch)
    sub = build_notebook_submission("modal://team-burst")
    rc = launch_notebook(sub)
    assert rc == 0
    assert sink["app_name"] == "team-burst"
    assert sink["image"] == DEFAULT_NEMO_IMAGE  # the same container training uses


def test_launch_notebook_modal_absent_raises_actionable(monkeypatch) -> None:
    """With ``modal`` not importable, the launcher refuses with install/auth steps."""
    monkeypatch.setitem(sys.modules, "modal", None)
    with pytest.raises(RuntimeError) as exc:
        launch_notebook(build_notebook_submission("modal"))
    assert "pip install modal" in str(exc.value)


def test_datastore_env_forwarded_only_when_set(monkeypatch) -> None:
    """The forwarded datastore env carries only the vars actually set."""
    for k in nb._DATASTORE_ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("METAFLOW_DEFAULT_METADATA", "service")
    monkeypatch.setenv("METAFLOW_SERVICE_URL", "http://localhost:8080/")
    env = nb._datastore_env()
    assert env == {
        "METAFLOW_DEFAULT_METADATA": "service",
        "METAFLOW_SERVICE_URL": "http://localhost:8080/",
    }
