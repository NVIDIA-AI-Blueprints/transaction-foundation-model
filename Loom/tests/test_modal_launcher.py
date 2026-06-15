"""Tests for the Modal H100 GPU launcher behind the ``nemo`` ``LOOM_GPU_TARGET``.

These exercise the real ``_launch_and_track`` Modal path WITHOUT requiring the
optional ``modal`` package (or a Modal account) to be installed:

* :func:`build_modal_submission` is a pure function -- it produces the right
  submission spec (GPU=H100, the NeMo NGC container, the lowered training config,
  the timeout) with no ``modal`` installed, stubbing nothing.
* ``gpu_target="modal"`` + ``launch=True`` routes through the launcher: we stub a
  fake ``modal`` module so ``.remote()`` returns a fake checkpoint handle, and
  assert the ``nemo`` adapter snapshots it into a Metaflow artifact **pathspec**
  ArtifactRef (never a raw file / object-store URI).
* ``modal`` absent => a clean, actionable refusal (install + auth), no mutation.
* an unknown ``gpu_target`` => ``REFUSED_UNKNOWN_GPU_TARGET`` listing launchers.
* the existing ``None`` (``REFUSED_NO_GPU_TARGET``) and ``PLANNED`` gate paths
  still hold unchanged.
* no raw ``s3://`` / checkpoint-file literal leaks into the returned ArtifactRef.
"""

from __future__ import annotations

import sys
import types

import pytest

from loom.config import LoomConfig
from loom.providers.model_builder import _modal_launcher
from loom.providers.model_builder._modal_launcher import (
    DEFAULT_MODAL_APP,
    DEFAULT_NEMO_IMAGE,
    MODAL_GPU,
    ModalLaunchResult,
    build_modal_submission,
    parse_modal_target,
)
from loom.providers.model_builder.nemo import NemoModelBuilderProvider
from loom.types import ArtifactRef

# A representative lowered + costed plan, shaped exactly like
# ``NemoModelBuilderProvider._lower_to_nemo_config`` + ``_estimate_cost`` produce.
_PLAN = {
    "intent": {"objective": "next-event", "budget": "full", "sequences_ref": "IngestDataset/1"},
    "backend": "nemo",
    "recipe": "automodel-causal-lm:decoder/rope/gqa",
    "resources": {"resources": "gpu=8", "micro_batch": 32, "tensor_parallel": 4},
    "tokenizer": "gpu-field-tokenizer:hashed-compositional",
    "gpu_target": "modal",
    "cost": {"gpu_count": 8, "wall_clock_hours": 12.0, "est_usd": 288.0},
}


# ---------------------------------------------------------------------------
# Pure submission-spec construction (NO modal installed / no account).
# ---------------------------------------------------------------------------


def test_build_modal_submission_spec_is_h100_nemo_image_and_lowered_config() -> None:
    """The pure spec carries GPU=H100, the NeMo NGC container, the lowered config."""
    sub = build_modal_submission(_PLAN, "modal")
    assert sub.gpu == MODAL_GPU == "H100"
    assert sub.image == DEFAULT_NEMO_IMAGE  # the documented NGC pin (no env override)
    assert sub.app_name == DEFAULT_MODAL_APP
    # The lowered training config is shipped verbatim -- recipe/resources/tokenizer.
    assert sub.training_config["recipe"] == _PLAN["recipe"]
    assert sub.training_config["resources"] == _PLAN["resources"]
    assert sub.training_config["tokenizer"] == _PLAN["tokenizer"]
    assert sub.training_config["intent"]["objective"] == "next-event"
    # Timeout scales off the cost physics (12h -> 2x headroom), well above the floor.
    assert sub.timeout_seconds >= 12 * 2 * 3600


def test_build_modal_submission_honors_env_image_override(monkeypatch) -> None:
    """``LOOM_NEMO_IMAGE`` overrides the container at the point of use (env only)."""
    monkeypatch.setenv("LOOM_NEMO_IMAGE", "nvcr.io/nvidia/nemo:25.01")
    sub = build_modal_submission(_PLAN, "modal")
    assert sub.image == "nvcr.io/nvidia/nemo:25.01"


def test_parse_modal_target_app_forms() -> None:
    """``modal`` -> default app; ``modal://my-app`` -> ``my-app``."""
    assert parse_modal_target("modal") == DEFAULT_MODAL_APP
    assert parse_modal_target("modal://my-app") == "my-app"
    sub = build_modal_submission(_PLAN, "modal://team-burst")
    assert sub.app_name == "team-burst"


# ---------------------------------------------------------------------------
# Routing: gpu_target="modal" + launch=True drives the launcher (stubbed modal).
# ---------------------------------------------------------------------------


class _FakeFunction:
    """A stand-in for a Modal function: ``.remote()`` returns a fake result."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def remote(self, training_config):  # noqa: D401 - mimics modal's .remote()
        # The remote H100 job "ran" and produced a checkpoint handle + metrics.
        return self._payload


class _FakeAppRun:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeApp:
    """A stand-in for ``modal.App``: registers a function, runs as a context."""

    def __init__(self, name: str, payload: dict) -> None:
        self.name = name
        self._payload = payload

    def function(self, **_kwargs):
        def _decorator(_fn):
            return _FakeFunction(self._payload)

        return _decorator

    def run(self):
        return _FakeAppRun()


def _install_fake_modal(monkeypatch, payload: dict) -> None:
    """Install a minimal fake ``modal`` module so launch_on_modal runs offline."""
    fake = types.ModuleType("modal")

    class _Image:
        @staticmethod
        def from_registry(ref):
            return {"image": ref}

    fake.Image = _Image
    fake.App = lambda name: _FakeApp(name, payload)
    monkeypatch.setitem(sys.modules, "modal", fake)


def test_gpu_target_modal_launch_returns_backbone_pathspec_artifactref(monkeypatch) -> None:
    """``modal`` + launch=True => snapshotted backbone ArtifactRef (LAUNCHED, pathspec)."""
    _install_fake_modal(
        monkeypatch,
        {"checkpoint_handle": "modal-vol://run/ckpt", "metrics": {"final_loss": 0.42}},
    )
    nemo = NemoModelBuilderProvider(LoomConfig(gpu_target="modal"), launch=True)
    ref = nemo.pretrain("IngestDataset/1", "next-event", "full")

    assert isinstance(ref, ArtifactRef)
    assert ref.kind == "backbone"
    assert ref.error is None
    assert ref.summary.get("status") == "LAUNCHED"
    assert ref.summary.get("launcher") == "modal"
    assert ref.summary.get("gpu") == "H100"
    assert ref.summary.get("metrics") == {"final_loss": 0.42}
    # The backbone IS a Metaflow run/artifact pathspec (<FlowName>/<run_id>), never
    # the raw checkpoint handle the remote job produced.
    assert ref.pathspec is not None
    parts = [p for p in ref.pathspec.split("/") if p]
    assert len(parts) == 2, f"not a <FlowName>/<run_id> pathspec: {ref.pathspec!r}"


def test_modal_launch_via_modal_app_target(monkeypatch) -> None:
    """``modal://<app>`` routes too; the snapshot pathspec reflects the app."""
    _install_fake_modal(monkeypatch, {"checkpoint_handle": "h", "metrics": {}})
    nemo = NemoModelBuilderProvider(LoomConfig(gpu_target="modal://team-burst"), launch=True)
    ref = nemo.pretrain("IngestDataset/1", "next-event", "small")
    assert ref.summary.get("status") == "LAUNCHED"
    assert "team-burst" in ref.pathspec


# ---------------------------------------------------------------------------
# modal absent => a clean, actionable refusal (no mutation).
# ---------------------------------------------------------------------------


def test_modal_absent_yields_actionable_refusal(monkeypatch) -> None:
    """With ``modal`` not importable, the launcher refuses with install/auth steps."""
    # Force the lazy ``import modal`` to fail regardless of the host environment.
    monkeypatch.setitem(sys.modules, "modal", None)
    nemo = NemoModelBuilderProvider(LoomConfig(gpu_target="modal"), launch=True)
    ref = nemo.pretrain("IngestDataset/1", "next-event", "full")

    assert isinstance(ref, ArtifactRef)
    assert ref.pathspec is None  # no run produced -> no mutation
    assert ref.summary.get("status") == "REFUSED_MODAL_UNAVAILABLE"
    assert ref.error is not None
    msg = ref.error.lower()
    assert "pip install modal" in msg
    assert "modal token set" in msg or "modal setup" in msg


def test_require_modal_raises_actionable_runtimeerror(monkeypatch) -> None:
    """The launcher's lazy import gate raises an actionable RuntimeError when absent."""
    monkeypatch.setitem(sys.modules, "modal", None)
    with pytest.raises(RuntimeError) as exc:
        _modal_launcher._require_modal()
    assert "pip install modal" in str(exc.value)


# ---------------------------------------------------------------------------
# Unknown gpu_target => REFUSED_UNKNOWN_GPU_TARGET (no mutation).
# ---------------------------------------------------------------------------


def test_unknown_gpu_target_refuses_listing_launchers() -> None:
    """A non-modal target refuses up front, listing the supported launcher(s)."""
    nemo = NemoModelBuilderProvider(LoomConfig(gpu_target="slurm-cluster-x"), launch=True)
    ref = nemo.pretrain("IngestDataset/1", "next-event", "full")

    assert isinstance(ref, ArtifactRef)
    assert ref.pathspec is None  # no launch, no mutation
    assert ref.summary.get("status") == "REFUSED_UNKNOWN_GPU_TARGET"
    assert ref.error is not None
    assert "slurm-cluster-x" in ref.error
    assert "modal" in ref.error.lower()  # the supported launcher is named


# ---------------------------------------------------------------------------
# The existing gate paths still hold EXACTLY (None => refuse; target+!launch => plan).
# ---------------------------------------------------------------------------


def test_none_gpu_target_still_refuses_no_gpu_target() -> None:
    """``gpu_target=None`` still REFUSES up front (unchanged), never launching."""
    nemo = NemoModelBuilderProvider(LoomConfig(gpu_target=None), launch=True)
    ref = nemo.pretrain("IngestDataset/1", "next-event", "full")
    assert ref.pathspec is None
    assert ref.summary.get("status") == "REFUSED_NO_GPU_TARGET"
    assert ref.error is not None and "gpu" in ref.error.lower()


def test_modal_target_without_launch_still_plans() -> None:
    """A modal target with ``launch=False`` is still a staged PLAN (no mutation)."""
    nemo = NemoModelBuilderProvider(LoomConfig(gpu_target="modal"), launch=False)
    ref = nemo.pretrain("IngestDataset/1", "next-event", "full")
    assert ref.summary.get("status") == "PLANNED"
    assert ref.pathspec is not None
    assert ref.error is None


# ---------------------------------------------------------------------------
# No raw S3 / checkpoint-file literal leaks into the launched ArtifactRef.
# ---------------------------------------------------------------------------


def test_launched_artifactref_carries_no_raw_storage_or_checkpoint_literal(monkeypatch) -> None:
    """The LAUNCHED ref surfaces a pathspec + small summary -- no URI / file leak."""
    _install_fake_modal(
        monkeypatch,
        {"checkpoint_handle": "modal-vol://secret/path/model.ckpt", "metrics": {}},
    )
    nemo = NemoModelBuilderProvider(LoomConfig(gpu_target="modal"), launch=True)
    ref = nemo.pretrain("IngestDataset/1", "next-event", "full")

    blob = repr(ref).lower()
    assert "s3://" not in blob
    assert ".nemo" not in blob
    assert "modal-vol://" not in blob  # the opaque handle never leaks out
    # Defensive: the snapshot returned the pathspec, not the raw handle.
    assert ref.pathspec is not None and "://" not in ref.pathspec


def test_launch_result_dataclass_defaults() -> None:
    """ModalLaunchResult defaults are H100 + the default app (sanity)."""
    res = ModalLaunchResult(checkpoint_handle="h")
    assert res.gpu == "H100"
    assert res.app_name == DEFAULT_MODAL_APP
    assert res.metrics == {}
