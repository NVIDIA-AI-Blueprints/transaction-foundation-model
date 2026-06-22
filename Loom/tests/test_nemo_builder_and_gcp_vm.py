"""NeMo builder + gcp-vm executor — structural tests (ARCHITECTURE §5.1, §10 step 7).

NO GPU, NO VM bring-up, NO model turn (HARD CONSTRAINT #4): every assertion is
structural — the rendered YAML, the torchrun argv, the constructed gcloud/docker
command lines, the derived cost plan, and the JSONL→ProgressEvent mapping. The
executor runs in DRY-RUN (the default + ``LOOM_DRY_RUN``), so no subprocess fires.

The hard constraints these pin:
  #1 lazy import — ``import loom.adapters.nemo_builder`` (and the VM launcher
     module body) pulls in ZERO of nemo/torch/transformers/cudf.
  #2 the progress source is the net-new JSONL, NOT a recipe-stdout scrape; the
     stream carries the step-0 ``loss≈ln(vocab)`` canary.
  #5 the rendered YAML diff-equals the committed config except the 3 overrides.
"""

from __future__ import annotations

import json
import math
import os
import sys

import pandas as pd
import pytest
import yaml

import loom.adapters.gcp_vm_executor as gcp_mod  # noqa: F401 - self-registers
import loom.adapters.nemo_builder as nemo_mod  # noqa: F401 - self-registers
import loom.verbs  # noqa: F401 - registers pretrain
from loom.adapters.gcp_vm_executor import GcpVmExecutor
from loom.adapters.nemo_builder import (
    NeMoModelBuilder,
    build_torchrun_argv,
    config_overrides,
    render_config,
)
from loom.ports import (
    EXECUTORS,
    MODEL_BUILDERS,
    BudgetEnvelope,
    ComputeTarget,
    ModelSpec,
    Objective,
    PreparedCorpus,
)
from loom.registry import REGISTRY, VerbContext
from loom.store import ObjectStore
from loom.types import Status, Verdict

_TFM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_COMMITTED = os.path.join(_TFM_ROOT, "configs", "pretrain_financial_decoder.yaml")


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _corpus(vocab_size: int = 6251, train_uri: str = "gs://b/train/shard-*.arrow") -> PreparedCorpus:
    return PreparedCorpus(
        representation="event-sequence",
        representation_signature="vh-abc123",
        tensor_contract="clm/input_ids+labels/-100",
        train_uri=train_uri,
        val_uri=None,
        test_uri=None,
        manifest_uri="gs://b/manifest.json",
        seq_length=4096,
        pad_token_id=0,
        vocab_size=vocab_size,
        effective_tokens=1_000_000,
    )


def _model() -> ModelSpec:
    return ModelSpec(family="decoder-clm", arch={"_target_": "transformers.LlamaConfig"})


def _objective() -> Objective:
    return Objective(kind="next-token", requires_tensor_contract="clm/input_ids+labels/-100")


def _compute(gpus: int = 8, target: str = "a2-highgpu-8g") -> ComputeTarget:
    return ComputeTarget(launcher="gcp-vm", nproc_per_node=gpus, accelerator="gpu",
                         gpu_target=target, image="nvcr.io/nvidia/nemo:25.09.01")


def _budget(max_steps: int = 3000) -> BudgetEnvelope:
    return BudgetEnvelope(max_usd=312.0, max_wall_clock_min=240, max_steps=max_steps)


# ---------------------------------------------------------------------------
# HARD CONSTRAINT #1 — lazy import.
# ---------------------------------------------------------------------------


def test_no_banned_import_after_loading_adapters():
    banned = ("nemo", "nemo_automodel", "torch", "transformers", "cudf", "cuml", "torchdata")
    import loom.adapters._nemo_train_entry  # noqa: F401 - module body must be import-safe

    hit = sorted(
        m for m in sys.modules
        if any(m == b or m.startswith(b + ".") for b in banned)
    )
    assert hit == [], f"banned packages imported on the control plane: {hit}"


# ---------------------------------------------------------------------------
# HARD CONSTRAINT #5 — rendered YAML diff-equals committed except 3 overrides.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.path.exists(_COMMITTED), reason="committed config not in this checkout")
def test_rendered_yaml_diff_equals_committed_except_three_overrides():
    with open(_COMMITTED, "r", encoding="utf-8") as fh:
        committed = yaml.safe_load(fh)
    # vocab from-corpus DIFFERENT from the committed 6251 → all 3 overrides fire.
    corpus = _corpus(vocab_size=8192)
    rendered = render_config(corpus=corpus, model=_model(), budget=_budget())

    def flatten(d, prefix=""):
        out = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(flatten(v, key))
            else:
                out[key] = v
        return out

    cf, rf = flatten(committed), flatten(rendered)
    diffs = {k for k in set(cf) | set(rf) if cf.get(k) != rf.get(k)}
    assert diffs == {
        "model.config.vocab_size",
        "dataset.data_path",
        "validation_dataset.data_path",
        "step_scheduler.max_steps",
    }, f"unexpected diff vs committed config: {diffs}"
    # The _target_ map is carried through UNCHANGED (fidelity to the real recipe).
    assert rendered["model"]["_target_"] == "nemo_automodel.NeMoAutoModelForCausalLM.from_config"
    assert rendered["loss_fn"]["_target_"] == "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"
    assert rendered["checkpoint"]["save_consolidated"] is True
    assert rendered["checkpoint"]["model_save_format"] == "safetensors"


def test_config_overrides_reports_the_three_keys():
    ovr = config_overrides(corpus=_corpus(vocab_size=8192), model=_model(), budget=_budget())
    assert set(ovr) == {
        "model.config.vocab_size",
        "dataset.data_path",
        "validation_dataset.data_path",
        "step_scheduler.max_steps",
    }
    assert ovr["model.config.vocab_size"] == (6251, 8192)
    assert ovr["step_scheduler.max_steps"] == (30, 3000)


# ---------------------------------------------------------------------------
# torchrun argv — torchrun directly (NOT the automodel CLI), the wrapping launcher.
# ---------------------------------------------------------------------------


def test_torchrun_argv_shape():
    argv = build_torchrun_argv(
        config_path="/ws/r.yaml", corpus=_corpus(), budget=_budget(),
        compute=_compute(gpus=8), progress_jsonl="/ws/p.jsonl",
    )
    assert argv[0] == "torchrun"
    assert "--nproc-per-node=8" in argv
    assert argv[2].endswith("_nemo_train_entry.py"), "argv must wrap the adapter's OWN launcher"
    assert "train_decoder_model.py" not in " ".join(argv), "must NOT invoke the team's recipe script"
    # The dotted overrides (the documented real invocation).
    assert "--dataset.data_path" in argv and "--step_scheduler.max_steps" in argv
    # The launcher's own flags (the net-new JSONL source + the step-0 canary input).
    assert "--loom-progress-jsonl" in argv and "--loom-vocab-size" in argv


def test_argv_paths_are_container_relative_not_control_plane(monkeypatch, tmp_path):
    """Step-7 follow-up: every path baked into the argv must resolve INSIDE the
    NeMo container (CWD /workspace), NOT the control-plane run_dir. A control-plane
    path under the workspace root is prefix-swapped to /workspace/<rel>; a gs:// URI
    passes through untouched."""
    from loom.adapters.nemo_builder import to_container_path

    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    run_dir = os.path.join(str(tmp_path), ".loom", "runs", "nemo-x")
    config_path = os.path.join(run_dir, "rendered.yaml")
    progress = os.path.join(run_dir, "progress.jsonl")
    argv = build_torchrun_argv(
        config_path=config_path, corpus=_corpus(), budget=_budget(),
        compute=_compute(gpus=8), progress_jsonl=progress,
    )
    joined = " ".join(argv)
    # The control-plane absolute prefix must NOT survive into the container argv.
    assert str(tmp_path) not in joined, f"control-plane path leaked into argv: {joined}"
    # The -c config + the JSONL sink + the launcher are all under /workspace.
    ci = argv.index("-c")
    assert argv[ci + 1] == "/workspace/.loom/runs/nemo-x/rendered.yaml"
    pj = argv.index("--loom-progress-jsonl")
    assert argv[pj + 1] == "/workspace/.loom/runs/nemo-x/progress.jsonl"
    assert argv[2].startswith("/workspace/"), argv[2]
    # A gs:// shard URI passes through (resolves as-is inside the container).
    di = argv.index("--dataset.data_path")
    assert argv[di + 1] == "gs://b/train/shard-*.arrow"
    # The helper itself: workspace-root path → /workspace; outside → untouched.
    assert to_container_path(os.path.join(str(tmp_path), "src/clm_data.py")) == "/workspace/src/clm_data.py"
    assert to_container_path("/etc/passwd") == "/etc/passwd"
    assert to_container_path("gs://b/x") == "gs://b/x"


def test_argv_uses_executor_supplied_container_path():
    """The executor MAY override the translation via a duck-typed container_path
    hook (mirrors the existing checkpoint_uri cooperation)."""
    class _Ex:
        name = "fake"

        def container_path(self, p):
            return "/remote" + p

    argv = build_torchrun_argv(
        config_path="/ctl/r.yaml", corpus=_corpus(), budget=_budget(),
        compute=_compute(gpus=1), progress_jsonl="/ctl/p.jsonl", executor=_Ex(),
    )
    ci = argv.index("-c")
    assert argv[ci + 1] == "/remote/ctl/r.yaml"
    pj = argv.index("--loom-progress-jsonl")
    assert argv[pj + 1] == "/remote/ctl/p.jsonl"


# ---------------------------------------------------------------------------
# cost PLAN — DERIVED real $, never a label.
# ---------------------------------------------------------------------------


def test_plan_is_derived_real_dollars():
    builder = NeMoModelBuilder()
    plan = builder.plan(
        corpus=_corpus(), model=_model(), objective=_objective(),
        compute=_compute(gpus=8), budget=_budget(), executor=GcpVmExecutor(dry_run=True),
    )
    assert plan.derived is True, "the cost plan must be DERIVED, not a hardcoded label"
    assert plan.usd is not None and plan.usd > 0.0
    assert plan.params and plan.params > 1_000_000  # ~29M for the committed arch
    assert plan.tokens and plan.tokens == 3000 * 16 * 4096  # max_steps · gbs · seq
    assert plan.gpu_target == "a2-highgpu-8g"


# ---------------------------------------------------------------------------
# supports() — accept the TFM triple, reject MLM / a tensor-contract mismatch.
# ---------------------------------------------------------------------------


def test_supports_accepts_tfm_triple_rejects_mlm():
    b = NeMoModelBuilder()
    assert b.supports(model=_model(), objective=_objective(), corpus=_corpus()).supported is True
    mlm = Objective(kind="masked-lm", requires_tensor_contract="mlm/...")
    cap = b.supports(model=_model(), objective=mlm, corpus=_corpus())
    assert cap.supported is False and "MLM" in (cap.reason or "")
    # A vision family is rejected too.
    vis = ModelSpec(family="vision-mae", arch={})
    assert b.supports(model=vis, objective=_objective(), corpus=_corpus()).supported is False


# ---------------------------------------------------------------------------
# gcp-vm executor — DRY-RUN: build commands, NEVER execute (HARD CONSTRAINT #4).
# ---------------------------------------------------------------------------


def test_gcp_vm_executor_is_dry_runnable(monkeypatch):
    monkeypatch.setenv("LOOM_DRY_RUN", "1")
    ex = GcpVmExecutor()
    assert ex.dry_run is True
    assert ex.gpu_available() is True  # the single GPU VM advertises GPU capability
    argv = ["torchrun", "--nproc-per-node=8", "/p/_nemo_train_entry.py", "-c", "/ws/r.yaml"]
    handle = ex.submit(
        argv=argv, image="nvcr.io/nvidia/nemo:25.09.01", compute=_compute(),
        budget=_budget(), on_event=lambda e: None,
    )
    assert handle.status() == "succeeded"  # dry-run is born terminal
    cmds = [" ".join(c) for c in handle.commands]
    # The real mechanism: gpu-up, sync, container start, recipe exec.
    assert any("gcp-gpu-up.sh" in c for c in cmds)
    assert any("gcp-sync-workspace.sh" in c for c in cmds)
    assert any("docker run -d" in c and "--gpus all" in c for c in cmds)
    assert any("docker exec" in c and "torchrun" in c for c in cmds)
    # The wall-clock envelope is a hard `timeout` around the exec.
    assert any("timeout " in c for c in cmds)


def test_gcp_vm_kill_constructs_teardown(monkeypatch):
    monkeypatch.setenv("LOOM_DRY_RUN", "1")
    ex = GcpVmExecutor()
    h = ex.submit(argv=["torchrun", "x"], image=None, compute=_compute(),
                  budget=_budget(), on_event=lambda e: None)
    ex.kill(h.job_id)
    cmds = [" ".join(c) for c in h.commands]
    assert any("pkill -f torchrun" in c for c in cmds)
    assert any("gcp-gpu-down.sh" in c for c in cmds)


def test_gcp_vm_executor_implements_fetch_progress_and_container_path(monkeypatch):
    """Grounding finding #3 (major): the REAL gcp-vm executor must expose the
    duck-typed hooks NeMoTrainingHandle._refresh_progress / nemo_builder.to_container_path
    cooperate with — else the net-new JSONL the launcher writes ON THE VM is
    unreachable from the control plane (empty progress feed + final-loss on every
    live single-VM run). Structural + dry-run only: NO ssh, NO VM, NO spend."""
    monkeypatch.setenv("LOOM_DRY_RUN", "1")
    monkeypatch.setenv("GCP_INSTANCE", "tfm-gpu-notebook")
    monkeypatch.setenv("GCP_ZONE", "us-central1-f")
    monkeypatch.setenv("REMOTE_WORKSPACE", "/mnt/tfm/workspace")
    ex = GcpVmExecutor()

    # The hooks the handle/builder probe by name MUST exist on the real executor.
    assert callable(getattr(ex, "fetch_progress", None))
    assert callable(getattr(ex, "container_path", None))

    # container_path: an in-container /workspace path → /workspace (in-container
    # geometry is authoritative); a $REMOTE_WORKSPACE host path → its /workspace
    # prefix; a gs:// URI passes through.
    assert ex.container_path("/workspace/.loom/runs/x/progress.jsonl") \
        == "/workspace/.loom/runs/x/progress.jsonl"
    assert ex.container_path("/mnt/tfm/workspace/.loom/runs/x/rendered.yaml") \
        == "/workspace/.loom/runs/x/rendered.yaml"
    assert ex.container_path("gs://b/train/shard-*.arrow") == "gs://b/train/shard-*.arrow"

    # _host_path: the in-container path the launcher wrote (/workspace/<rel>) maps
    # to the backing host mount ($REMOTE_WORKSPACE/<rel>) that `cat` reads over ssh.
    assert ex._host_path("/workspace/.loom/runs/x/progress.jsonl") \
        == "/mnt/tfm/workspace/.loom/runs/x/progress.jsonl"

    # fetch_progress on a DRY-RUN executor is a guaranteed no-op (no ssh / no file
    # created) — the no-spend build never touches the VM (HARD CONSTRAINT #4).
    local = "/tmp/loom-fetch-progress-should-not-be-written.jsonl"
    if os.path.exists(local):
        os.remove(local)
    ex.fetch_progress("/workspace/.loom/runs/x/progress.jsonl", local)
    assert not os.path.exists(local), "dry-run fetch_progress must NOT touch the VM or write a file"


def test_nemo_handle_uses_real_executor_fetch_hook(tmp_path, monkeypatch):
    """End-to-end (handle + REAL gcp-vm executor): the handle's _refresh_progress
    finds the executor's fetch_progress hook and reads the VM-side JSONL into the
    control-plane path. We stub only the ssh transport (the dry-run gate is off so
    the hook runs, but `_run`/subprocess is replaced) — no real gcloud, no spend."""
    import math
    from loom.adapters._nemo_train_entry import _ProgressLog
    from loom.adapters.nemo_builder import NeMoTrainingHandle

    monkeypatch.setenv("REMOTE_WORKSPACE", str(tmp_path / "vm-mnt"))
    # The JSONL as it exists on the VM HOST mount ($REMOTE_WORKSPACE/.loom/...).
    host_file = str(tmp_path / "vm-mnt" / ".loom" / "runs" / "x" / "progress.jsonl")
    log = _ProgressLog(host_file)
    log.emit(step=0, loss=math.log(6251), phase="warmup", note="loss≈ln(vocab) (step-0 canary)")
    log.emit(step=7, loss=6.2, phase="consolidate", note="done")
    log.close()

    # A LIVE executor (dry_run=False) whose ssh transport is stubbed to a local cat:
    # this exercises the real _host_path mapping + the real fetch_progress body
    # without any gcloud/ssh/VM/spend.
    ex = GcpVmExecutor(dry_run=False)
    import subprocess as _sp

    def _fake_run(cmd, check, stdout):
        # The real cmd is `gcloud compute ssh ... --command 'cat <host_file>'`; the
        # host path is the last token after `cat `. Read it locally into stdout.
        joined = " ".join(cmd)
        assert "gcloud" in joined and "cat " in joined
        path = joined.split("cat ", 1)[1].strip().strip("'\"")
        with open(path, "r", encoding="utf-8") as r:
            stdout.write(r.read())
    monkeypatch.setattr(_sp, "run", _fake_run)

    ctl_path = str(tmp_path / "ctl" / "progress.jsonl")
    in_container = "/workspace/.loom/runs/x/progress.jsonl"

    class _Job:
        job_id = "live1"

        def status(self):
            return "succeeded"

    h = NeMoTrainingHandle(
        job_id="live1", _job=_Job(), _executor=ex,
        _progress_jsonl=ctl_path, _progress_jsonl_remote=in_container,
        _checkpoint_uri="gs://b/ckpt", _representation_signature="vh-abc123",
        _model_signature="msig-x", _budget=_budget(), _usd_per_gpu_hour=29.39,
        _gpus=8, _vocab_size=6251,
    )
    events = list(h.stream_events())
    assert os.path.exists(ctl_path), "the real fetch hook must populate the control-plane path"
    assert any(e.note and "ln(vocab)" in e.note for e in events)
    ckpt = h.result()
    assert ckpt.metrics.get("final_loss") == 6.2 and ckpt.metrics.get("step") == 7


# ---------------------------------------------------------------------------
# HARD CONSTRAINT #2 — the progress source is the net-new JSONL + the canary.
# ---------------------------------------------------------------------------


def test_jsonl_is_the_progress_source_with_step0_canary(tmp_path):
    from loom.adapters._nemo_train_entry import _ProgressLog
    from loom.adapters.nemo_builder import NeMoTrainingHandle

    p = str(tmp_path / "progress.jsonl")
    log = _ProgressLog(p)
    ln_vocab = math.log(6251)
    log.emit(step=0, loss=ln_vocab, phase="warmup", note=f"loss≈ln(vocab)={ln_vocab:.3f} (step-0 canary)")
    log.emit(step=1, loss=8.4, lr=2e-4, tokens=65536, phase="train")
    log.emit(step=2, loss=7.8, phase="consolidate", note="done")
    log.close()

    class _Job:
        job_id = "j1"

        def status(self):
            return "succeeded"

    h = NeMoTrainingHandle(
        job_id="j1", _job=_Job(), _executor=None, _progress_jsonl=p,
        _checkpoint_uri="gs://b/ckpt", _representation_signature="vh-abc123",
        _model_signature="msig-x", _budget=_budget(), _usd_per_gpu_hour=29.39,
        _gpus=8, _vocab_size=6251,
    )
    events = list(h.stream_events())
    assert len(events) >= 1, "the JSONL must yield ≥1 ProgressEvent (the launch-and-track feed)"
    canary = [e for e in events if e.note and "ln(vocab)" in e.note]
    assert canary and canary[0].step == 0
    assert abs(canary[0].loss - ln_vocab) < 1e-6
    ckpt = h.result()
    assert ckpt.fmt == "hf-safetensors-consolidated"
    assert ckpt.representation_signature == "vh-abc123"  # ECHOED
    assert ckpt.metrics.get("step0_canary") is not None
    assert ckpt.metrics.get("final_loss") == 7.8


def test_live_run_pulls_jsonl_back_from_vm(tmp_path):
    """Step-7 follow-up: on a live gcp-vm run the launcher wrote the JSONL ON THE
    VM (the in-container path), not on the control plane. The handle must ask the
    executor to fetch it back before reading — else the progress feed + final-loss
    are silently empty for every real run. We simulate the VM-side file + a fetch
    hook that copies it to the control-plane read path."""
    from loom.adapters._nemo_train_entry import _ProgressLog
    from loom.adapters.nemo_builder import NeMoTrainingHandle

    # The JSONL as it exists ON THE VM (the remote/in-container path).
    vm_path = str(tmp_path / "vm" / "progress.jsonl")
    log = _ProgressLog(vm_path)
    log.emit(step=0, loss=math.log(6251), phase="warmup", note="loss≈ln(vocab) (step-0 canary)")
    log.emit(step=1, loss=8.1, lr=2e-4, tokens=65536, phase="train")
    log.emit(step=42, loss=6.5, phase="consolidate", note="done")
    log.close()

    # The control-plane read path starts EMPTY (nothing synced yet).
    ctl_path = str(tmp_path / "ctl" / "progress.jsonl")
    assert not os.path.exists(ctl_path)

    fetched = {"n": 0}

    class _Ex:
        name = "gcp-vm-fake"

        def fetch_progress(self, remote, local):
            # The real executor runs `gcloud compute ssh ... cat <remote>` > local.
            fetched["n"] += 1
            os.makedirs(os.path.dirname(local), exist_ok=True)
            with open(remote, "r", encoding="utf-8") as r, open(local, "w", encoding="utf-8") as w:
                w.write(r.read())

        def kill(self, _):
            return None

    class _Job:
        job_id = "live1"

        def status(self):
            return "succeeded"

    # The remote path the launcher wrote to (in the real executor this is the
    # in-container /workspace/... path; the executor's gcloud-ssh fetch reads the
    # backing VM file). We point it at the simulated VM file the fake reads.
    h = NeMoTrainingHandle(
        job_id="live1", _job=_Job(), _executor=_Ex(),
        _progress_jsonl=ctl_path, _progress_jsonl_remote=vm_path,
        _checkpoint_uri="gs://b/ckpt", _representation_signature="vh-abc123",
        _model_signature="msig-x", _budget=_budget(), _usd_per_gpu_hour=29.39,
        _gpus=8, _vocab_size=6251,
    )
    # stream_events must trigger the fetch and then see the VM's records.
    events = list(h.stream_events())
    assert fetched["n"] >= 1, "the handle must pull the JSONL back from the VM"
    assert any(e.note and "ln(vocab)" in e.note for e in events)
    assert os.path.exists(ctl_path), "the fetch must populate the control-plane read path"
    # result() likewise refreshes + reads the terminal loss.
    ckpt = h.result()
    assert ckpt.metrics.get("final_loss") == 6.5
    assert ckpt.metrics.get("step") == 42


# ---------------------------------------------------------------------------
# The verb wiring — nemo + gcp-vm, dry-run, the gates (no GPU spend).
# ---------------------------------------------------------------------------


def _make_corpus(store: ObjectStore) -> str:
    df = pd.DataFrame(
        [
            ("0xa1", "2026-06-01 00:00:00", "DEXETH", "BUY", "WETH", 120.0),
            ("0xa1", "2026-06-01 00:05:00", "DEXETH", "SELL", "WETH", 118.5),
            ("0xb2", "2026-06-01 09:00:00", "DEXSOL", "BUY", "SOL", 42.0),
        ],
        columns=["wallet", "timestamp", "venue", "side", "item", "size_usd"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    ctx = VerbContext(store=store, driver="cli", interactive=True, extras={"dataframe": df})
    return str(REGISTRY["tokenize"].fn({"preset": "chain", "context_len": 256}, ctx).outputs[0])


def test_pretrain_nemo_without_gpu_target_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("LOOM_DRY_RUN", "1")
    monkeypatch.delenv("LOOM_GPU_TARGET", raising=False)
    store = ObjectStore(str(tmp_path))
    corpus = _make_corpus(store)
    res = REGISTRY["pretrain"].fn(
        {"in": corpus, "model_builder": "nemo"},
        VerbContext(store=store, driver="cli", interactive=True),
    )
    assert res.status is Status.REFUSED_NO_GPU_TARGET


def test_pretrain_nemo_defaults_to_gcp_vm_executor(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("LOOM_DRY_RUN", "1")
    store = ObjectStore(str(tmp_path))
    corpus = _make_corpus(store)
    res = REGISTRY["pretrain"].fn(
        {"in": corpus, "model_builder": "nemo", "gpu_target": "a2-highgpu-8g",
         "nproc_per_node": 8, "max_steps": 3000, "max_usd": 1000},
        VerbContext(store=store, driver="cli", interactive=True),
    )
    assert res.status is Status.PLAN
    assert res.data["executor"] == "gcp-vm"  # the §10 follow-up: nemo ⇒ gcp-vm
    assert res.cost_plan.derived is True and res.cost_plan.usd > 0


def test_pretrain_over_threshold_requires_typed_confirm(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("LOOM_DRY_RUN", "1")
    monkeypatch.setenv("LOOM_CONFIRM_USD_THRESHOLD", "0.01")  # force the typed gate
    store = ObjectStore(str(tmp_path))
    corpus = _make_corpus(store)
    base = {"in": corpus, "model_builder": "nemo", "gpu_target": "a2-highgpu-8g",
            "nproc_per_node": 8, "max_steps": 3000, "max_usd": 1000}
    plan = REGISTRY["pretrain"].fn(dict(base), VerbContext(store=store, driver="cli", interactive=True))
    assert plan.data["requires_typed_confirm"] is True
    # token alone (via ctx) — NOT enough above the threshold.
    r = REGISTRY["pretrain"].fn(
        dict(base, launch=True),
        VerbContext(store=store, driver="cli", interactive=True, confirm_token=plan.confirm_token),
    )
    assert r.status is Status.PLAN, "over-threshold launch must require a typed confirm"
    # token + --confirm → launches (dry-run checkpoint).
    r2 = REGISTRY["pretrain"].fn(
        dict(base, launch=True, confirm=True),
        VerbContext(store=store, driver="cli", interactive=True, confirm_token=r.confirm_token),
    )
    assert r2.status is Status.OK and r2.verdict is Verdict.PASS
    ckpt = store.get(str(r2.outputs[0]))
    assert ckpt.signatures["fmt"] == "hf-safetensors-consolidated"
    # The checkpoint uri is a durable VM-or-GCS location, NEVER a bare tempdir.
    uri = ckpt.extras["checkpoint"]["uri"]
    assert uri.startswith(("gs://", "vm://")), uri


def test_pretrain_stopped_at_budget_branch(tmp_path, monkeypatch):
    """A budget-stopped run → verdict INCOMPLETE + a resume token, status not OK-PASS."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    store = ObjectStore(str(tmp_path))
    corpus = _make_corpus(store)

    # A builder whose launch() returns a stopped_at_budget handle (no GPU, no NeMo).
    from loom.ports import CheckpointRef, ProgressEvent, register_model_builder

    class _StoppedHandle:
        job_id = "stopped-1"

        def stream_events(self):
            yield ProgressEvent(step=0, loss=8.7, usd_spent=0.0, usd_envelope=10.0,
                                phase="warmup", note="loss≈ln(vocab)")
            yield ProgressEvent(step=5, loss=8.0, usd_spent=9.9, usd_envelope=10.0, phase="train")

        def status(self):
            return "stopped_at_budget"

        def cancel(self):
            return None

        def result(self):
            c = store.get(corpus)
            return CheckpointRef(
                uri="vm://tfm/partial", fmt="hf-safetensors-consolidated",
                representation_signature=c.signatures["vocab_hash"],
                model_signature="msig-stopped", metrics={"step": 5, "final_loss": 8.0},
            )

    class _StoppedBuilder:
        name = "stopped-test"

        def supports(self, **_):
            from loom.ports import Capability
            return Capability(supported=True)

        def plan(self, **_):
            from loom.types import CostPlan
            return CostPlan(derived=True, usd=0.001, confidence="LOW")

        def launch(self, **_):
            return _StoppedHandle()

    register_model_builder(_StoppedBuilder())
    try:
        plan = REGISTRY["pretrain"].fn(
            {"in": corpus, "model_builder": "stopped-test"},
            VerbContext(store=store, driver="cli", interactive=True),
        )
        res = REGISTRY["pretrain"].fn(
            {"in": corpus, "model_builder": "stopped-test", "launch": True,
             "confirm_token": plan.confirm_token},
            VerbContext(store=store, driver="cli", interactive=True),
        )
        assert res.verdict is Verdict.INCOMPLETE
        assert res.data["stopped_at_budget"] is True
        assert res.data["resume_token"] and res.data["resume_token"].startswith("resume:")
        ckpt = store.get(str(res.outputs[0]))
        assert ckpt.extras["stopped_at_budget"] is True
        assert ckpt.verdict is Verdict.INCOMPLETE
    finally:
        MODEL_BUILDERS.pop("stopped-test", None)
