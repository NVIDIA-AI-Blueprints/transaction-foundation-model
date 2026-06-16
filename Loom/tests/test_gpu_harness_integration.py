"""GPU-harness INTEGRATION tests (Task D, ARCHITECTURE §5.1 + §10 step 7).

These exercise the NeMo model-builder adapter + the single-VM ``gcp-vm`` executor
+ the ``pretrain`` wiring as a *system*, end to end, with **NO GPU, NO GCP, NO
NeMo, NO model turn** (HARD CONSTRAINT #4). Every assertion is structural — a
fresh-process import set, the rendered-YAML diff, a parsed JSONL progress feed,
the dry-run command sequence the executor constructs, and the verb's gate
verdicts. Nothing here brings up a VM, runs a subprocess, or spends a cent.

The hard constraints these pin (the review checks them):
  #1 LAZY import — ``import loom`` + the two adapters pull in ZERO of
     {torch, transformers, nemo, nemo_automodel, cudf}. Verified in a FRESH
     interpreter (subprocess) so no earlier test's import can mask a leak.
  #2 the progress source is the net-new JSONL the launcher writes — a sample
     JSONL → an ordered list of ProgressEvent carrying the step-0 canary note;
     never a scrape of the recipe stdout.
  #4 the ``gcp-vm`` executor is DRY-RUNNABLE — it CONSTRUCTS the real gcp-script
     + docker + torchrun command lines without executing any of them.
  #5 the rendered NeMo YAML for the TFM triple diff-equals the committed
     ``configs/pretrain_financial_decoder.yaml`` except the 3 intended overrides.

Complements (does not duplicate) the unit-level ``test_nemo_builder_and_gcp_vm.py``:
those assert one method at a time; these drive the full ``pretrain`` gate +
fresh-interpreter import + a multi-record JSONL through ``stream_events``.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap

import pandas as pd
import pytest
import yaml

import loom.verbs  # noqa: F401 - registers pretrain + loads the adapters
from loom.adapters.gcp_vm_executor import GcpVmExecutor
from loom.adapters.local_executor import LocalExecutor
from loom.adapters.nemo_builder import (
    NeMoModelBuilder,
    NeMoTrainingHandle,
    render_config,
    render_config_yaml,
)
from loom.ports import (
    BudgetEnvelope,
    ComputeTarget,
    ModelSpec,
    Objective,
    PreparedCorpus,
)
from loom.registry import REGISTRY, VerbContext
from loom.store import ObjectStore
from loom.types import Status

_TFM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_COMMITTED = os.path.join(_TFM_ROOT, "configs", "pretrain_financial_decoder.yaml")

_BANNED = ("torch", "transformers", "nemo", "nemo_automodel", "cudf", "cuml", "torchdata")


# ---------------------------------------------------------------------------
# Fixtures — a PreparedCorpus for the TFM triple + a real Corpus DataObject.
# ---------------------------------------------------------------------------


def _corpus(vocab_size: int = 6251) -> PreparedCorpus:
    return PreparedCorpus(
        representation="event-sequence",
        representation_signature="vh-int-001",
        tensor_contract="clm/input_ids+labels/-100",
        train_uri="gs://b/loom-corpora/vh-int-001/train/shard-*.arrow",
        val_uri=None,
        test_uri=None,
        manifest_uri="gs://b/loom-corpora/vh-int-001/manifest.json",
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
    return ComputeTarget(
        launcher="gcp-vm", nproc_per_node=gpus, accelerator="gpu",
        gpu_target=target, image="nvcr.io/nvidia/nemo:25.09.01",
    )


def _budget(max_steps: int = 3000) -> BudgetEnvelope:
    return BudgetEnvelope(max_usd=312.0, max_wall_clock_min=240, max_steps=max_steps)


def _make_corpus(store: ObjectStore) -> str:
    """A real ``Corpus`` DataObject from the chain preset (CPU, no GPU/NeMo)."""
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
    res = REGISTRY["tokenize"].fn({"preset": "chain", "context_len": 256}, ctx)
    assert res.outputs, f"tokenize did not write a Corpus: {res.status}"
    return str(res.outputs[0])


# ===========================================================================
# HARD CONSTRAINT #1 — lazy import, verified in a FRESH interpreter.
# ===========================================================================


def test_lazy_import_invariant_in_a_fresh_interpreter():
    """``import loom`` + both GPU adapters (+ the VM launcher's module body) must
    pull in ZERO of {torch, transformers, nemo, nemo_automodel, cudf, ...}.

    Run in a SUBPROCESS so no module a previous in-process test imported can mask
    a leak — the cleanest integration-level statement of HARD CONSTRAINT #1: the
    loom package stays CPU-installable; NeMo/torch live ONLY in the VM container."""
    code = textwrap.dedent(
        """
        import sys
        import loom
        import loom.adapters.nemo_builder
        import loom.adapters.gcp_vm_executor
        import loom.adapters._nemo_train_entry  # module body must be import-safe too
        import loom.verbs  # the verb spine + adapter auto-load
        banned = %r
        hit = sorted(
            m for m in sys.modules
            if any(m == b or m.startswith(b + ".") for b in banned)
        )
        print(repr(hit))
        """
    ) % (_BANNED,)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=_TFM_ROOT,
    )
    assert proc.returncode == 0, f"fresh import failed:\n{proc.stderr}"
    hit = proc.stdout.strip()
    assert hit == "[]", f"banned packages imported on the control plane: {hit}"


# ===========================================================================
# HARD CONSTRAINT #5 — rendered YAML diff-equals committed except 3 overrides.
# ===========================================================================


@pytest.mark.skipif(not os.path.exists(_COMMITTED), reason="committed config not in this checkout")
def test_rendered_yaml_round_trips_and_diff_equals_committed():
    """Render → serialize → re-parse, then flat-diff vs the committed config. The
    ONLY differing keys are the three intended overrides; the entire ``_target_``
    map (NeMoAutoModel, LlamaConfig, the dataset file-path, FSDP2,
    MaskedCrossEntropy, consolidated-safetensors) is carried through verbatim."""
    with open(_COMMITTED, "r", encoding="utf-8") as fh:
        committed = yaml.safe_load(fh)

    corpus = _corpus(vocab_size=8192)  # != committed 6251 ⇒ the vocab override fires
    # Go through the YAML TEXT the adapter writes to the VM (round-trip fidelity).
    rendered_text = render_config_yaml(corpus=corpus, model=_model(), budget=_budget())
    rendered = yaml.safe_load(rendered_text)

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
    }, f"unexpected diff vs the committed recipe config: {diffs}"

    # The exact override values (fidelity to the real recipe everywhere else).
    assert rf["model.config.vocab_size"] == 8192
    assert rf["dataset.data_path"] == corpus.train_uri
    assert rf["validation_dataset.data_path"] == corpus.train_uri  # val falls back to train
    assert rf["step_scheduler.max_steps"] == 3000
    # No NeMo internals were touched.
    assert rf["model._target_"] == "nemo_automodel.NeMoAutoModelForCausalLM.from_config"
    assert rf["loss_fn._target_"] == "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"
    assert rf["checkpoint.save_consolidated"] is True


@pytest.mark.skipif(not os.path.exists(_COMMITTED), reason="committed config not in this checkout")
def test_vocab_none_keeps_committed_value_so_only_two_overrides_fire():
    """A continuous-representation corpus (``vocab_size=None``) must still render a
    VALID config: the vocab override is skipped (the committed 6251 stays), leaving
    only the two data_path + the max_steps overrides — never a crash, never a None
    written into ``model.config.vocab_size``."""
    corpus = PreparedCorpus(
        representation="event-sequence", representation_signature="vh-cont",
        tensor_contract="clm/input_ids+labels/-100",
        train_uri="gs://b/t/shard-*.arrow", val_uri=None, test_uri=None,
        manifest_uri="gs://b/m.json", seq_length=4096, pad_token_id=0,
        vocab_size=None, effective_tokens=1,
    )
    rendered = render_config(corpus=corpus, model=_model(), budget=_budget())
    assert rendered["model"]["config"]["vocab_size"] == 6251  # committed kept


# ===========================================================================
# HARD CONSTRAINT #2 — the progress parser: a sample JSONL → ordered events.
# ===========================================================================


def _drive_progress(tmp_path, records: list[dict]) -> NeMoTrainingHandle:
    """Write the given JSONL records exactly as the VM launcher would, then return
    a handle whose ``stream_events()`` tails that file (the integration path)."""
    p = str(tmp_path / "progress.jsonl")
    with open(p, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    class _Job:
        job_id = "j-int"

        def status(self):
            return "succeeded"

    return NeMoTrainingHandle(
        job_id="j-int", _job=_Job(), _executor=None, _progress_jsonl=p,
        _checkpoint_uri="gs://b/loom-checkpoints/run/ckpt",
        _representation_signature="vh-int-001", _model_signature="msig-int",
        _budget=_budget(), _usd_per_gpu_hour=29.39, _gpus=8, _vocab_size=6251,
    )


def test_progress_jsonl_parses_to_ordered_events_with_step0_canary(tmp_path):
    """A representative launcher JSONL (the step-0 ``loss≈ln(vocab)`` canary +
    two training steps + a consolidate record) parses to an ORDERED list of
    ProgressEvent. The canary is first, carries the note, and its loss is
    ln(vocab); the train losses thread through in step order; usd_spent is
    DERIVED from wall_clock × the VM $/hour (never a label)."""
    ln_vocab = math.log(6251)
    h = _drive_progress(tmp_path, [
        {"schema": "loom-nemo-progress/1", "step": 0, "loss": ln_vocab,
         "phase": "warmup", "wall_clock_min": 0.0,
         "note": f"loss≈ln(vocab)={ln_vocab:.3f} (step-0 canary)"},
        {"schema": "loom-nemo-progress/1", "step": 1, "loss": 8.40, "lr": 2e-4,
         "tokens": 65536, "phase": "train", "wall_clock_min": 0.5},
        {"schema": "loom-nemo-progress/1", "step": 2, "loss": 7.80,
         "phase": "train", "wall_clock_min": 1.0},
        {"schema": "loom-nemo-progress/1", "step": 2, "loss": 7.80,
         "phase": "consolidate", "wall_clock_min": 1.1, "note": "done"},
    ])
    events = list(h.stream_events())

    # The feed is non-empty and ordered by the JSONL it tailed.
    assert [e.step for e in events][:3] == [0, 1, 2]
    train_steps = [e.step for e in events if e.phase == "train"]
    assert train_steps == sorted(train_steps), "train events must thread in step order"

    canary = events[0]
    assert canary.step == 0 and canary.phase == "warmup"
    assert canary.note and "ln(vocab)" in canary.note
    assert abs(canary.loss - ln_vocab) < 1e-9

    # usd_spent is DERIVED from wall_clock_min × ($/gpu-hour × gpus)/60, not a label.
    step1 = next(e for e in events if e.step == 1 and e.phase == "train")
    expected = round(0.5 * (29.39 * 8) / 60.0, 4)
    assert step1.usd_spent == expected
    assert step1.usd_envelope == 312.0  # the binding cap surfaces on every event

    # result() pulls the final loss + the step-0 canary out of the same JSONL.
    ckpt = h.result()
    assert ckpt.fmt == "hf-safetensors-consolidated"
    assert ckpt.representation_signature == "vh-int-001"  # ECHOED from the corpus
    assert ckpt.metrics["final_loss"] == 7.80
    assert ckpt.metrics["step0_canary"] is not None


def test_progress_parser_tolerates_blank_and_malformed_lines(tmp_path):
    """The tailer must skip blanks/garbage (a partial line mid-fsync on the VM) and
    still yield the well-formed records — the feed never dies on a torn write."""
    p = str(tmp_path / "progress.jsonl")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write('{"step": 0, "loss": 8.7, "phase": "warmup", "note": "loss≈ln(vocab) canary"}\n')
        fh.write("{ this is not json\n")
        fh.write("   \n")
        fh.write('{"step": 1, "loss": 8.1, "phase": "train", "wall_clock_min": 0.2}\n')

    class _Job:
        job_id = "j2"

        def status(self):
            return "succeeded"

    h = NeMoTrainingHandle(
        job_id="j2", _job=_Job(), _executor=None, _progress_jsonl=p,
        _checkpoint_uri="vm://x/ckpt", _representation_signature="vh-int-001",
        _model_signature="m", _budget=_budget(), _usd_per_gpu_hour=3.67, _gpus=1,
        _vocab_size=6251,
    )
    events = list(h.stream_events())
    assert [e.step for e in events] == [0, 1], "only the well-formed records survive"


def test_progress_stream_empty_when_no_jsonl_yet(tmp_path):
    """Before the VM produces the JSONL (or on a dry-run that never launches), the
    stream yields nothing rather than raising — the widget feed degrades cleanly."""
    h = NeMoTrainingHandle(
        job_id="j3", _job=type("J", (), {"job_id": "j3", "status": lambda self: "succeeded"})(),
        _executor=None, _progress_jsonl=str(tmp_path / "does-not-exist.jsonl"),
        _checkpoint_uri="vm://x/ckpt", _representation_signature="vh-int-001",
        _model_signature="m", _budget=_budget(), _usd_per_gpu_hour=3.67, _gpus=1,
    )
    assert list(h.stream_events()) == []


# ===========================================================================
# HARD CONSTRAINT #4 — gcp-vm DRY-RUN: construct the real commands, never run them.
# ===========================================================================


def test_gcp_vm_dry_run_submits_a_real_command_sequence(monkeypatch):
    """``submit(argv=[torchrun, ...])`` in dry-run returns a command sequence that
    references the REAL gcp scripts (``gcp-gpu-up.sh`` + ``gcp-sync-workspace.sh``),
    the NeMo-container ``docker run -d --gpus all`` + ``docker exec``, and the
    submitted torchrun argv — all WRAPPED through ``gcloud compute ssh``, with NOT
    a single subprocess fired (the dry-run is born terminal ``succeeded``)."""
    monkeypatch.setenv("LOOM_DRY_RUN", "1")
    monkeypatch.delenv("LOOM_GCP_LIVE", raising=False)
    ex = GcpVmExecutor()
    assert ex.dry_run is True, "absence of LOOM_GCP_LIVE ⇒ dry-run by default"

    argv = ["torchrun", "--nproc-per-node=8", "/ws/_nemo_train_entry.py",
            "-c", "/ws/r.yaml", "--step_scheduler.max_steps", "3000"]
    handle = ex.submit(
        argv=argv, image="nvcr.io/nvidia/nemo:25.09.01",
        compute=_compute(), budget=_budget(), on_event=lambda e: None,
    )
    assert handle.status() == "succeeded"  # dry-run: born terminal, NOTHING ran
    assert handle.dry_run is True
    flat = [" ".join(c) for c in handle.commands]

    # The real single-VM mechanism, in order: up → sync → container start → exec.
    assert any(c.endswith("gcp-gpu-up.sh") for c in flat)
    assert any(c.endswith("gcp-sync-workspace.sh") for c in flat)
    assert any("docker run -d" in c and "--gpus all" in c and "sleep infinity" in c for c in flat)
    exec_cmd = next(c for c in flat if "docker exec" in c)
    # The submitted argv is what gets exec'd inside the container, wrapped in ssh.
    assert "gcloud compute ssh" in exec_cmd
    assert "torchrun" in exec_cmd and "_nemo_train_entry.py" in exec_cmd
    assert "train_decoder_model.py" not in exec_cmd, "must NOT invoke the team's recipe"
    # The binding wall-clock cap is a hard `timeout` around the exec.
    assert "timeout 14400" in exec_cmd  # 240 min × 60


def test_gcp_vm_dry_run_fires_no_subprocess(monkeypatch):
    """Belt-and-braces on HARD CONSTRAINT #4: assert ``subprocess.run`` is NEVER
    called on the whole submit→kill dry-run path (no VM, no spend)."""
    monkeypatch.setenv("LOOM_DRY_RUN", "1")
    calls: list = []
    import loom.adapters.gcp_vm_executor as gcp_mod
    monkeypatch.setattr(gcp_mod.subprocess, "run",
                        lambda *a, **k: calls.append((a, k)))
    ex = GcpVmExecutor()
    h = ex.submit(argv=["torchrun", "x"], image=None, compute=_compute(),
                  budget=_budget(), on_event=lambda e: None)
    ex.kill(h.job_id)
    assert calls == [], "dry-run must construct commands WITHOUT executing any subprocess"
    # kill still recorded the teardown commands for inspection.
    flat = [" ".join(c) for c in h.commands]
    assert any("pkill -f torchrun" in c for c in flat)
    assert any(c.endswith("gcp-gpu-down.sh") for c in flat)


def test_gpu_available_advertises_for_gcp_vm_but_not_for_local():
    """``gpu_available()`` gates ``REFUSED_NO_GPU_TARGET``: the single GPU VM
    ADVERTISES capability; the in-process ``local`` executor reports False (a nemo
    build through ``local`` is refused). Holds with no LOOM_* env set."""
    assert GcpVmExecutor(dry_run=True).gpu_available() is True
    assert LocalExecutor().gpu_available() is False


def test_checkpoint_uri_is_durable_never_a_tempdir(monkeypatch):
    """The executor resolves a DURABLE checkpoint uri (gs:// when a bucket is set,
    else vm://) — the §10 step-1-6 follow-up: the verb records a real VM-or-GCS
    location, never a bare local tempdir."""
    monkeypatch.setenv("LOOM_DRY_RUN", "1")
    monkeypatch.setenv("GCP_BUCKET", "lvl-tfm-artifacts")
    ex = GcpVmExecutor()
    uri = ex.checkpoint_uri("/ws/.loom/runs/nemo-abc-123", "models/decoder-demo/checkpoints/")
    assert uri.startswith("gs://lvl-tfm-artifacts/")
    monkeypatch.delenv("GCP_BUCKET", raising=False)
    ex2 = GcpVmExecutor()
    uri2 = ex2.checkpoint_uri("/ws/.loom/runs/nemo-abc-123", "models/decoder-demo/checkpoints/")
    assert uri2.startswith("vm://")


# ===========================================================================
# The GATE — pretrain --model-builder nemo, the no-target refusal + the PLAN.
# ===========================================================================


def test_pretrain_nemo_without_gpu_target_is_refused(tmp_path, monkeypatch):
    """``pretrain --model-builder nemo`` with NO --gpu-target / LOOM_GPU_TARGET →
    ``REFUSED_NO_GPU_TARGET`` (the GPU builder requires a machine type), and no
    Checkpoint is written."""
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
    assert store.list("Checkpoint") == []


def test_pretrain_nemo_with_dry_run_target_plans_a_real_derived_cost(tmp_path, monkeypatch):
    """With a dry-run GPU target, ``pretrain --model-builder nemo`` returns a PLAN
    whose ``cost_plan`` is DERIVED (``derived is True``) with a REAL estimate
    (``usd > 0``) — NOT the local rehearsal's ~$0. The executor resolves to the
    single-VM ``gcp-vm`` (the nemo default), and the PLAN exposes the real GPU $.
    No VM is brought up; the dry-run executor only plans."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("LOOM_DRY_RUN", "1")
    store = ObjectStore(str(tmp_path))
    corpus = _make_corpus(store)
    res = REGISTRY["pretrain"].fn(
        {"in": corpus, "model_builder": "nemo", "gpu_target": "a2-highgpu-8g",
         "nproc_per_node": 8, "max_steps": 3000, "max_usd": 10_000},
        VerbContext(store=store, driver="cli", interactive=True),
    )
    assert res.status is Status.PLAN
    assert res.data["executor"] == "gcp-vm", "nemo ⇒ the single-VM gcp-vm executor"
    assert res.cost_plan.derived is True
    assert res.cost_plan.usd is not None and res.cost_plan.usd > 0.0
    # A REAL GPU estimate, not the ~$0 CPU rehearsal (sanity floor for an A100 run).
    assert res.cost_plan.usd > 0.01, "a real GPU $ estimate, not the local ~$0"
    assert res.cost_plan.params and res.cost_plan.params > 1_000_000
    # tokens = max_steps · global_batch_size · seq_len. The verb resolves seq_len
    # from the corpus (here the tokenize context_len=256), not the RoPE max.
    assert res.cost_plan.tokens == 3000 * 16 * res.cost_plan.seq_len
    assert res.cost_plan.seq_len == 256, "seq_len is the corpus context_len, from-corpus"
    assert res.cost_plan.gpu_target == "a2-highgpu-8g"
    assert store.list("Checkpoint") == [], "a PLAN writes no Checkpoint"


def test_pretrain_nemo_derived_cost_dominates_local(tmp_path, monkeypatch):
    """The nemo (GPU) derived cost is orders of magnitude above the local CPU
    rehearsal's ~$0 — proving the PLAN is a real per-adapter estimate, not a shared
    label. Same corpus, two builders, two genuinely different derived numbers."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("LOOM_DRY_RUN", "1")
    store = ObjectStore(str(tmp_path))
    corpus = _make_corpus(store)

    local = REGISTRY["pretrain"].fn(
        {"in": corpus, "model_builder": "local"},
        VerbContext(store=store, driver="cli", interactive=True),
    )
    nemo = REGISTRY["pretrain"].fn(
        {"in": corpus, "model_builder": "nemo", "gpu_target": "a2-highgpu-8g",
         "nproc_per_node": 8, "max_steps": 3000, "max_usd": 10_000},
        VerbContext(store=store, driver="cli", interactive=True),
    )
    assert local.cost_plan.derived is True and nemo.cost_plan.derived is True
    assert nemo.cost_plan.usd > local.cost_plan.usd * 100, (
        f"nemo ${nemo.cost_plan.usd} should dwarf local ${local.cost_plan.usd}"
    )


def test_pretrain_nemo_explicit_local_executor_is_refused_no_gpu(tmp_path, monkeypatch):
    """Forcing ``--execution local`` under a nemo build → ``REFUSED_NO_GPU_TARGET``:
    even WITH a gpu_target, the in-process ``local`` executor cannot reach a GPU, so
    the gate refuses rather than silently planning a CPU run for a GPU builder."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("LOOM_DRY_RUN", "1")
    store = ObjectStore(str(tmp_path))
    corpus = _make_corpus(store)
    res = REGISTRY["pretrain"].fn(
        {"in": corpus, "model_builder": "nemo", "execution": "local",
         "gpu_target": "a2-highgpu-8g", "max_usd": 10_000},
        VerbContext(store=store, driver="cli", interactive=True),
    )
    assert res.status is Status.REFUSED_NO_GPU_TARGET
    assert res.data["executor"] == "local"


def test_pretrain_nemo_falls_back_to_local_when_gpu_target_set_via_env(tmp_path, monkeypatch):
    """``LOOM_GPU_TARGET`` (not just the explicit flag) satisfies the GPU gate — the
    machine type echoes through the env, and the nemo PLAN derives off it."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("LOOM_DRY_RUN", "1")
    monkeypatch.setenv("LOOM_GPU_TARGET", "a2-highgpu-1g")
    store = ObjectStore(str(tmp_path))
    corpus = _make_corpus(store)
    res = REGISTRY["pretrain"].fn(
        {"in": corpus, "model_builder": "nemo", "max_usd": 10_000},
        VerbContext(store=store, driver="cli", interactive=True),
    )
    assert res.status is Status.PLAN
    assert res.cost_plan.gpu_target == "a2-highgpu-1g"
    assert res.cost_plan.derived is True and res.cost_plan.usd > 0.0


# ===========================================================================
# supports() integration — the nemo builder, through the registry, accepts the
# TFM triple and rejects an incompatible objective with a usable reason.
# ===========================================================================


def test_nemo_builder_supports_through_registry():
    from loom.ports import MODEL_BUILDERS

    builder = MODEL_BUILDERS["nemo"]
    assert isinstance(builder, NeMoModelBuilder)
    ok = builder.supports(model=_model(), objective=_objective(), corpus=_corpus())
    assert ok.supported is True

    mlm = Objective(kind="masked-lm", requires_tensor_contract="mlm/...")
    bad = builder.supports(model=_model(), objective=mlm, corpus=_corpus())
    assert bad.supported is False and bad.reason and "MLM" in bad.reason
