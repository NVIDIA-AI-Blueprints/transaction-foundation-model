"""``nemo`` — the model-builder port adapter #1 (ARCHITECTURE §5.1, §10 step 7).

A decoder-CLM builder over NeMo AutoModel. This file is a **YAML-renderer +
``torchrun`` argv-builder + process supervisor** — everything NeMo-specific stays
here; none of it leaks into ``loom/ports.py``. It maps onto the REAL recipe:

  * the config ``_target_`` map verified in
    ``configs/pretrain_financial_decoder.yaml`` (``NeMoAutoModelForCausalLM.from_config``
    + ``transformers.LlamaConfig`` + the ``src/clm_data.py`` dataset ``_target_`` +
    ``FSDP2Manager`` + ``MaskedCrossEntropy`` + ``save_consolidated: true``
    safetensors), and
  * the verified 4-line recipe shell of ``scripts/train_decoder_model.py``
    (``parse_args_and_load_config`` → ``TrainFinetuneRecipeForNextTokenPrediction``
    → ``.setup()`` → ``.run_train_validation_loop()``), which the adapter's OWN
    wrapping launcher (``_nemo_train_entry.py``) re-implements + instruments with
    a JSONL progress log.

The launch is ``torchrun --nproc-per-node=N <launcher> -c <rendered.yaml>
--dataset.data_path … --validation_dataset.data_path … --step_scheduler.max_steps
…`` (ARCHITECTURE §5.1; ``torchrun`` directly, NOT the ``automodel`` CLI — the
config's documented gotcha, lines 47-48). The adapter does NOT run the recipe
itself: it hands the argv to the chosen :class:`~loom.ports.Executor`
(``gcp-vm``), which is the only thing that knows *where* the work runs.

================================  HARD CONSTRAINT #1  ==========================
LAZY NeMo/torch import. ``import loom`` and ``import loom.adapters.nemo_builder``
pull in ZERO of nemo / nemo_automodel / torch / transformers / cudf. This module
imports only stdlib + ``yaml`` + the loom ports/types at module scope. The
NeMo/torch code lives ONLY inside the wrapping launcher (which runs in the NeMo
container on the VM) and is referenced here by FILE PATH, never imported. There is
no ``import torch`` / ``import nemo_automodel`` anywhere in this file — not even
inside a function — because this file never runs the model; it only renders the
YAML + builds the argv + tails the JSONL the VM produces.
===============================================================================

This module imports nothing from NeMo/torch/transformers/RAPIDS/BigQuery.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import yaml

from ..ports import (
    BudgetEnvelope,
    Capability,
    CheckpointRef,
    ComputeTarget,
    Executor,
    JobHandle,
    ModelSpec,
    Objective,
    PreparedCorpus,
    ProgressEvent,
    TrainingHandle,
    register_model_builder,
)
from ..types import CostPlan

# The git-ish code identity baked into ``model_signature`` (the replay anchor, §3).
_CODE_SHA = "nemo-builder/0.1.0"

# The C4 tensor-contract the next-token CLM objective requires; the corpus
# produces it, this builder + the objective consume it (the narrow waist, §7).
_CLM_TENSOR_CONTRACT = "clm/input_ids+labels/-100"

# The committed reference config the rendered YAML must diff-equal except the 3
# intended overrides (HARD CONSTRAINT #5). Resolved relative to the TFM repo root
# (this Loom package lives at ``<repo>/Loom/loom/adapters/``).
_TFM_ROOT = Path(__file__).resolve().parents[3]
_REFERENCE_CONFIG = _TFM_ROOT / "configs" / "pretrain_financial_decoder.yaml"

# The recipe the wrapping launcher re-implements + instruments. The adapter ships
# its OWN launcher; it never edits or invokes the team's
# ``scripts/train_decoder_model.py`` (HARD CONSTRAINT #3).
_WRAPPING_LAUNCHER = Path(__file__).resolve().parent / "_nemo_train_entry.py"

# The container-internal path of the team's custom dataset file the YAML's
# ``dataset._target_`` references (``src/clm_data.py:build_financial_clm_dataset``).
# The executor mounts the synced workspace at the container CWD, so this stays a
# repo-relative file-path ``_target_`` exactly as the committed config has it.
_DATASET_TARGET = "src/clm_data.py:build_financial_clm_dataset"

# Where the synced workspace is mounted INSIDE the NeMo container. The executor's
# ``docker run -v $REMOTE_WORKSPACE:/workspace -w /workspace`` (matches
# ``scripts/gcp-jupyter.sh:59`` + ci.yml) puts the tar-synced TFM repo tree here,
# and the recipe is exec'd from this CWD. So EVERY path baked into the torchrun
# argv (the ``-c`` config, the ``--loom-progress-jsonl`` sink, the dotted
# ``--dataset.data_path``) must resolve INSIDE the container, NOT on the CPU
# control plane where the run_dir physically lives. The control-plane run_dir
# (``$LOOM_WORKSPACE/.loom/runs/<run>/…``) is the SAME tree the executor tar-syncs
# to ``$REMOTE_WORKSPACE`` and mounts here, so the translation is a single
# prefix-swap: ``<workspace-root>/<rel>`` → ``/workspace/<rel>`` (the step-7
# follow-up fix for the absolute-control-plane-path-in-argv bug). The executor MAY
# override the translation (a duck-typed ``container_path`` hook, probed below);
# absent that, this prefix-swap is the correct default for the single-VM mount.
_CONTAINER_WORKSPACE = "/workspace"

# A coarse on-demand GPU $/hour table for the cost PLAN (single-VM machine types
# in scripts/gcp-lib.sh). Approximate public on-demand list prices; the plan is a
# DERIVED estimate (CostPlan.derived=True), never a label, and carries its
# confidence. Override per-deployment via LOOM_GPU_USD_PER_HOUR.
_GPU_USD_PER_HOUR = {
    "a2-highgpu-1g": 3.67,    # 1× A100 40GB
    "a2-highgpu-2g": 7.35,
    "a2-highgpu-4g": 14.69,
    "a2-highgpu-8g": 29.39,
    "a2-ultragpu-1g": 5.07,   # 1× A100 80GB
    "a2-ultragpu-2g": 10.14,
    "a2-ultragpu-4g": 20.28,
    "a2-ultragpu-8g": 40.55,
    "g2-standard-24": 2.00,   # 2× L4
}
_DEFAULT_GPU_USD_PER_HOUR = 3.67  # fall back to a2-highgpu-1g


# ---------------------------------------------------------------------------
# YAML rendering — start from the committed config, apply exactly the 3 overrides.
# ---------------------------------------------------------------------------


def _reference_config() -> dict:
    """Load the committed reference config (the fidelity anchor). Falls back to an
    in-code copy of the verified structure if the repo file is not reachable (e.g.
    the loom package installed standalone), so rendering never hard-depends on the
    TFM checkout — but the on-repo path is preferred so a config edit by the team
    flows through (minus our 3 overrides)."""
    try:
        with open(_REFERENCE_CONFIG, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        if isinstance(cfg, dict):
            return cfg
    except (OSError, yaml.YAMLError):  # pragma: no cover - standalone install
        pass
    return copy.deepcopy(_FALLBACK_REFERENCE_CONFIG)


def render_config(
    *,
    corpus: PreparedCorpus,
    model: ModelSpec,
    budget: BudgetEnvelope,
) -> dict:
    """Render the NeMo config dict for the TFM triple.

    It is the committed ``pretrain_financial_decoder.yaml`` with EXACTLY the three
    intended overrides applied (HARD CONSTRAINT #5):

      1. ``model.config.vocab_size``  ← ``corpus.vocab_size`` (from-corpus; the
         committed config hardcodes ``6251`` — C1→arch, ARCHITECTURE §4).
      2. ``dataset.data_path`` and ``validation_dataset.data_path`` ← the corpus
         shard URIs (committed: ``null``).
      3. ``step_scheduler.max_steps`` ← ``budget.max_steps`` (committed: ``30``).

    Everything else (the ``_target_`` map, FSDP2, the loss_fn, the optimizer, the
    consolidated-safetensors checkpoint block) is carried through UNCHANGED. The
    parsed result therefore diff-equals the committed config on every key but the
    three above — see :func:`config_overrides` for the exact diff."""
    cfg = _reference_config()

    # Override 1: vocab_size from the corpus (None ⇒ keep the committed value so a
    # continuous-rep corpus that carries no vocab still renders a valid config).
    if corpus.vocab_size is not None:
        cfg.setdefault("model", {}).setdefault("config", {})["vocab_size"] = int(corpus.vocab_size)

    # Override 2: the corpus shard URIs (train + val). val falls back to train when
    # the corpus has no separate val split (this local slice; the C6 split is step 8).
    train_uri = corpus.train_uri
    val_uri = corpus.val_uri or corpus.train_uri
    cfg.setdefault("dataset", {})["data_path"] = train_uri
    cfg.setdefault("validation_dataset", {})["data_path"] = val_uri

    # Override 3: max_steps from the binding budget envelope (when set).
    if budget.max_steps is not None:
        cfg.setdefault("step_scheduler", {})["max_steps"] = int(budget.max_steps)

    return cfg


def config_overrides(
    *, corpus: PreparedCorpus, model: ModelSpec, budget: BudgetEnvelope
) -> dict[str, Any]:
    """The exact set of keys the rendered config changes vs the committed one — the
    machine-checkable form of HARD CONSTRAINT #5 (used by the structural diff test).
    Keys are dotted config paths; values are ``(committed, rendered)`` pairs."""
    ref = _reference_config()
    rendered = render_config(corpus=corpus, model=model, budget=budget)
    out: dict[str, Any] = {}
    for dotted in ("model.config.vocab_size", "dataset.data_path",
                   "validation_dataset.data_path", "step_scheduler.max_steps"):
        a = _dig(ref, dotted)
        b = _dig(rendered, dotted)
        if a != b:
            out[dotted] = (a, b)
    return out


def _dig(d: dict, dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def render_config_yaml(
    *, corpus: PreparedCorpus, model: ModelSpec, budget: BudgetEnvelope
) -> str:
    """The rendered config serialized to YAML text (what gets written to the VM)."""
    return yaml.safe_dump(
        render_config(corpus=corpus, model=model, budget=budget),
        sort_keys=False, default_flow_style=False,
    )


# ---------------------------------------------------------------------------
# Parameter counting + cost derivation (CPU-cheap; from the rendered arch).
# ---------------------------------------------------------------------------


def _llama_param_count(arch: dict) -> int:
    """Approximate a LlamaConfig decoder param count from the rendered arch dict.

    Standard decoder accounting (embeddings + per-layer attn + GQA KV + SwiGLU MLP
    + norms), enough for a DERIVED cost estimate (not a label). GQA is honored via
    ``num_key_value_heads``; SwiGLU via the 3-matrix MLP (gate/up/down)."""
    v = int(arch.get("vocab_size") or 0)
    h = int(arch.get("hidden_size") or 0)
    layers = int(arch.get("num_hidden_layers") or 0)
    n_heads = int(arch.get("num_attention_heads") or 0) or 1
    n_kv = int(arch.get("num_key_value_heads") or n_heads) or n_heads
    inter = int(arch.get("intermediate_size") or (4 * h))
    head_dim = h // n_heads if n_heads else h
    if h == 0 or layers == 0:
        return 0
    # Attention: q proj (h×h) + k,v proj (h × n_kv·head_dim) each + o proj (h×h).
    attn = h * h + 2 * (h * n_kv * head_dim) + h * h
    # SwiGLU MLP: gate (h×inter) + up (h×inter) + down (inter×h).
    mlp = 3 * h * inter
    # RMSNorm: 2 per layer (input + post-attn), h params each.
    norm = 2 * h
    per_layer = attn + mlp + norm
    embed = v * h
    final_norm = h
    tie = bool(arch.get("tie_word_embeddings", False))
    lm_head = 0 if tie else v * h
    return int(embed + layers * per_layer + final_norm + lm_head)


def _gpu_usd_per_hour(machine_type: Optional[str]) -> float:
    env = os.environ.get("LOOM_GPU_USD_PER_HOUR")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    if machine_type and machine_type in _GPU_USD_PER_HOUR:
        return _GPU_USD_PER_HOUR[machine_type]
    return _DEFAULT_GPU_USD_PER_HOUR


def _derive_cost(
    *, corpus: PreparedCorpus, model: ModelSpec, compute: ComputeTarget, budget: BudgetEnvelope
) -> CostPlan:
    """The DERIVED real-$ plan (ARCHITECTURE §3/§5.1): wall-clock from the
    6·N·D FLOPs rule over the budgeted token throughput, priced at the VM's GPU
    $/hour. NEVER a hardcoded number (CostPlan.derived=True).

    tokens  = max_steps · global_batch_size · seq_len   (the budgeted token count)
    params  = N from the rendered LlamaConfig arch
    flops   ≈ 6 · N · tokens                              (fwd+bwd training FLOPs)
    hours   = flops / (gpus · per_gpu_flops · mfu) / 3600
    usd     = hours · gpus · $/gpu-hour
    """
    arch = render_config(corpus=corpus, model=model, budget=budget)["model"]["config"]
    params = _llama_param_count(arch)
    seq_len = int(arch.get("max_position_embeddings") or corpus.seq_length or 4096)
    # Use the corpus seq_length for the token count (the trained context), not the
    # RoPE max_position_embeddings (which only bounds extrapolation).
    train_seq = int(corpus.seq_length or seq_len)

    ref_cfg = _reference_config()
    gbs = int(_dig(ref_cfg, "step_scheduler.global_batch_size") or 16)
    max_steps = int(budget.max_steps or _dig(ref_cfg, "step_scheduler.max_steps") or 0)
    tokens = max_steps * gbs * train_seq

    gpus = max(1, int(compute.nproc_per_node) * max(1, int(compute.nnodes)))
    # A100-class realized throughput assumption: ~150 TFLOP/s effective per GPU
    # (≈ 0.5 MFU on a 312 TFLOP/s bf16 A100). A coarse, documented constant.
    per_gpu_flops = 150e12
    flops = 6.0 * float(params) * float(tokens)
    gpu_seconds = flops / (gpus * per_gpu_flops) if (gpus and per_gpu_flops) else 0.0
    hours = gpu_seconds / 3600.0
    usd_per_gpu_hour = _gpu_usd_per_hour(compute.gpu_target)
    usd = round(hours * gpus * usd_per_gpu_hour, 4)

    return CostPlan(
        derived=True,
        usd=usd,
        confidence="LOW",  # a coarse 6ND·MFU estimate — honest about its precision
        tokens=int(tokens),
        params=int(params),
        seq_len=train_seq,
        gpu_target=compute.gpu_target,
        envelope={
            "max_usd": budget.max_usd,
            "max_steps": budget.max_steps,
            "max_wall_clock_min": budget.max_wall_clock_min,
        },
        inputs={
            "builder": "nemo",
            "machine_type": compute.gpu_target,
            "gpus": gpus,
            "global_batch_size": gbs,
            "max_steps": max_steps,
            "usd_per_gpu_hour": usd_per_gpu_hour,
            "est_wall_clock_min": round(hours * 60.0, 2),
            "per_gpu_flops_assumed": per_gpu_flops,
            "image": compute.image,
            "representation": corpus.representation,
            "tensor_contract": corpus.tensor_contract,
        },
    )


def _model_signature(model: ModelSpec, objective: Objective, arch: dict) -> str:
    """``hash{arch, objective, code_sha}`` — the replay anchor (§3)."""
    h = hashlib.sha256()
    h.update(json.dumps(arch, sort_keys=True, default=str).encode("utf-8"))
    h.update(b"\x00")
    h.update(model.family.encode("utf-8"))
    h.update(b"\x00")
    h.update(objective.kind.encode("utf-8"))
    h.update(b"\x00")
    h.update(_CODE_SHA.encode("utf-8"))
    return "msig-" + h.hexdigest()[:32]


# ---------------------------------------------------------------------------
# Container-path translation — the control-plane run_dir → in-container path.
#
# The argv runs INSIDE the NeMo container (CWD ``/workspace`` = the mounted synced
# workspace). So every absolute control-plane path that goes into it must be
# rewritten to its in-container location, or torchrun fails to load the config /
# write the JSONL on a live run (the step-7 follow-up bug). The executor MAY supply
# the mapping (duck-typed ``container_path`` hook); otherwise we prefix-swap the
# workspace root for ``/workspace``.
# ---------------------------------------------------------------------------


def _workspace_roots() -> list[str]:
    """The control-plane roots of the tree the executor tar-syncs onto the VM, in
    precedence order. The executor syncs the WHOLE TFM repo root (``-C "$ROOT"`` in
    ``gcp-sync-workspace.sh``) to ``$REMOTE_WORKSPACE`` and mounts it at
    ``/workspace`` — so the launcher (which ships inside the loom package, under the
    repo) AND the team's ``src/clm_data.py`` are both there. The verb additionally
    anchors the run_dir at ``LOOM_WORKSPACE``; in production that IS the repo root
    (``os.getcwd()``), but we accept both so a separately-rooted run_dir still
    translates. A path is translated against the FIRST root that contains it."""
    roots: list[str] = []
    ws = os.environ.get("LOOM_WORKSPACE")
    if ws:
        roots.append(os.path.abspath(ws))
    roots.append(str(_TFM_ROOT))
    # Dedup while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def to_container_path(host_path: str, executor: Optional[Executor] = None) -> str:
    """Translate a control-plane path → its in-container path (CWD ``/workspace``).

    Single-VM mount geometry: the executor tar-syncs the workspace root to
    ``$REMOTE_WORKSPACE`` and ``docker run -v $REMOTE_WORKSPACE:/workspace``, so a
    control-plane ``<root>/<rel>`` is ``/workspace/<rel>`` inside the container. An
    already-cloud/container-absolute path (``gs://…`` / under ``/workspace`` /
    ``/mnt/tfm/…``) is passed through untouched; a path under NEITHER synced root is
    left as-is (it won't be on the VM, so a live run surfaces the real error rather
    than this silently mangling an unrelated path).

    The executor MAY override via a duck-typed ``container_path(host_path)`` hook
    (mirrors the existing ``checkpoint_uri`` cooperation); absent that, the
    prefix-swap below is the correct default for the single-VM executor."""
    # Executor-supplied translation wins (live executors that mount elsewhere).
    hook = getattr(executor, "container_path", None) if executor is not None else None
    if callable(hook):
        try:
            return str(hook(host_path))
        except Exception:  # noqa: BLE001 - fall back to the prefix-swap
            pass
    if not host_path:
        return host_path
    # Cloud URIs and paths ALREADY inside the container resolve as-is.
    if "://" in host_path or host_path == _CONTAINER_WORKSPACE \
            or host_path.startswith(_CONTAINER_WORKSPACE + "/"):
        return host_path
    # A RELATIVE path is already expressed relative to the workspace root (the
    # recipe's CWD is /workspace), so it maps straight under /workspace — NOT
    # abspath'd against the control-plane CWD (that would graft an unrelated prefix).
    if not os.path.isabs(host_path):
        return _CONTAINER_WORKSPACE + "/" + host_path.replace(os.sep, "/").lstrip("/")
    # An ABSOLUTE control-plane path under a synced root → prefix-swap to /workspace.
    for root in _workspace_roots():
        try:
            rel = os.path.relpath(host_path, root)
        except ValueError:  # pragma: no cover - different drives (Windows); skip
            continue
        if not rel.startswith(".."):
            return _CONTAINER_WORKSPACE + "/" + rel.replace(os.sep, "/")
    return host_path


# ---------------------------------------------------------------------------
# argv construction — torchrun over the wrapping launcher (NOT train_decoder_model).
# ---------------------------------------------------------------------------


def build_torchrun_argv(
    *,
    config_path: str,
    corpus: PreparedCorpus,
    budget: BudgetEnvelope,
    compute: ComputeTarget,
    progress_jsonl: str,
    executor: Optional[Executor] = None,
) -> list[str]:
    """The exact ``torchrun`` argv the executor submits (ARCHITECTURE §5.1).

    ``torchrun --nproc-per-node=N <wrapping launcher> -c <rendered.yaml>
    --dataset.data_path … --validation_dataset.data_path … --step_scheduler.max_steps …``
    plus the launcher's two own flags (``--loom-progress-jsonl`` / ``--loom-vocab-size``).

    Uses ``torchrun`` directly, NOT the ``automodel`` CLI — the config's documented
    gotcha (``configs/…yaml:47-48``: the automodel CLI misparses ``--nproc-per-node``
    as a recipe override). The launcher forwards the dotted ``--dataset.*`` /
    ``--step_scheduler.*`` overrides to NeMo's own parser unchanged.

    Every path baked in (the ``-c`` config, the ``--loom-progress-jsonl`` sink) is
    translated to its IN-CONTAINER location via :func:`to_container_path`, because
    this argv is exec'd inside the NeMo container at ``/workspace`` — NOT on the CPU
    control plane where the run_dir physically lives (the step-7 follow-up fix). The
    dataset ``data_path`` is the corpus shard URI; a ``gs://`` shard resolves as-is,
    a control-plane-local shard is translated too."""
    train_uri = to_container_path(str(corpus.train_uri), executor)
    val_uri = to_container_path(str(corpus.val_uri or corpus.train_uri), executor)
    argv = [
        "torchrun",
        f"--nproc-per-node={int(compute.nproc_per_node)}",
    ]
    if int(compute.nnodes) > 1:
        argv.append(f"--nnodes={int(compute.nnodes)}")
    argv += [
        # The launcher itself lives under the synced workspace (it ships inside the
        # loom package, which the executor syncs), so it too needs translation.
        to_container_path(str(_WRAPPING_LAUNCHER), executor),
        "-c", to_container_path(config_path, executor),
        "--dataset.data_path", train_uri,
        "--validation_dataset.data_path", val_uri,
    ]
    if budget.max_steps is not None:
        argv += ["--step_scheduler.max_steps", str(int(budget.max_steps))]
    # The launcher's own flags (the net-new JSONL source + the step-0 canary). The
    # JSONL path is the IN-CONTAINER path the launcher writes to (the control-plane
    # handle reads it back via the executor's fetch hook — see NeMoTrainingHandle).
    argv += ["--loom-progress-jsonl", to_container_path(progress_jsonl, executor)]
    if corpus.vocab_size is not None:
        argv += ["--loom-vocab-size", str(int(corpus.vocab_size))]
    return argv


# ---------------------------------------------------------------------------
# The TrainingHandle — tails the JSONL the launcher writes → ProgressEvents.
# ---------------------------------------------------------------------------


@dataclass
class NeMoTrainingHandle:
    """A launch-and-track handle over a ``gcp-vm`` job.

    ``stream_events()`` reads the **net-new JSONL** the wrapping launcher writes
    (NOT the recipe's stdout) and converts each record → a :class:`ProgressEvent`,
    interpolating ``usd_spent`` from the elapsed wall-clock × the VM $/hour so the
    budget telemetry is live. The step-0 ``loss≈ln(vocab)`` canary record (emitted
    by the launcher) flows through as the first event's ``note``."""

    job_id: str
    _job: JobHandle
    _executor: Executor
    # The CONTROL-PLANE path the handle READS the JSONL from (local/dry-run: the
    # launcher writes here directly; live gcp-vm: the executor fetch hook refreshes
    # it from the VM before each read — see _refresh_progress).
    _progress_jsonl: str
    _checkpoint_uri: str
    _representation_signature: str
    _model_signature: str
    _budget: BudgetEnvelope
    _usd_per_gpu_hour: float
    _gpus: int
    _vocab_size: Optional[int] = None
    # The IN-CONTAINER/VM path the launcher actually WROTE the JSONL to (what went
    # into the argv). On a live run the file lives here on the VM, not on the control
    # plane, so the handle asks the executor to pull it back (the step-7 follow-up
    # fix: without this the live progress feed + final-loss are silently empty).
    _progress_jsonl_remote: Optional[str] = None
    _events: list[ProgressEvent] = field(default_factory=list)
    _dry_run: bool = False

    def _refresh_progress(self) -> None:
        """Pull the JSONL back from the VM to the control-plane read path, if the
        executor cooperates (duck-typed ``fetch_progress(remote, local)`` hook,
        mirroring the existing ``checkpoint_uri`` cooperation). No-op for the
        local/dry-run executor (the launcher already wrote to the read path) and a
        no-op when no remote path / no hook exists — so reading degrades to the
        control-plane file, never raising."""
        remote = self._progress_jsonl_remote
        if not remote or self._executor is None:
            return
        fetch = getattr(self._executor, "fetch_progress", None)
        if not callable(fetch):
            return
        try:
            fetch(remote, self._progress_jsonl)
        except Exception:  # noqa: BLE001 - best-effort; degrade to the local file
            pass

    def stream_events(self) -> Iterator[ProgressEvent]:
        """Tail the launcher's JSONL → ProgressEvents (deduped on step).

        Each JSONL record is ``{step, loss, lr, tokens, phase, wall_clock_min,
        note, ...}``. ``usd_spent`` is derived from ``wall_clock_min`` × the VM's
        GPU $/hour (× gpus); ``usd_envelope`` is the binding cap. On a live gcp-vm
        run the launcher wrote the JSONL ON THE VM, so we first ask the executor to
        pull it back to the control-plane read path (``_refresh_progress``); on a
        DRY-RUN the JSONL never appears, so this yields whatever was pre-seeded
        (none) — the structural-verification path asserts the argv + YAML, not a
        live stream."""
        usd_envelope = float(self._budget.max_usd)
        seen_steps: set[int] = set()
        self._refresh_progress()
        path = Path(self._progress_jsonl)
        if not path.exists():
            # No JSONL (dry-run, or the VM has not produced it yet). Nothing to yield.
            yield from self._events
            return
        per_min = (self._usd_per_gpu_hour * self._gpus) / 60.0
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except ValueError:
                    continue
                step = int(rec.get("step", -1))
                phase = rec.get("phase", "train")
                # Dedup repeated poller records for the same step (keep the first,
                # which carries the loss when the hook fired).
                key = step
                if phase == "train" and key in seen_steps:
                    continue
                if phase == "train":
                    seen_steps.add(key)
                wall = float(rec.get("wall_clock_min", 0.0) or 0.0)
                ev = ProgressEvent(
                    step=step,
                    loss=rec.get("loss"),
                    usd_spent=round(wall * per_min, 4),
                    usd_envelope=usd_envelope,
                    gpu_pct=None,
                    wall_clock_min=wall,
                    phase=phase,
                    note=rec.get("note"),
                )
                yield ev

    def status(self) -> str:
        return self._job.status()

    def cancel(self) -> None:
        self._executor.kill(self._job.job_id)

    def result(self) -> CheckpointRef:
        """Resolve the ``save_consolidated: true`` + ``model_save_format:
        safetensors`` output dir into a portable :class:`CheckpointRef` (C5).

        ``fmt="hf-safetensors-consolidated"`` — downstream ``embed`` loads it with
        vanilla ``AutoModelForCausalLM.from_pretrained``, zero NeMo dependency. The
        ``representation_signature`` is ECHOED from the corpus (the §3/§7 pairing
        invariant); ``model_signature`` is this builder's replay anchor. Final loss
        is read from the last train JSONL record when present."""
        metrics: dict[str, Any] = {"dry_run": self._dry_run}
        last_loss: Optional[float] = None
        last_step: Optional[int] = None
        # Pull the final JSONL back from the VM (live gcp-vm) before reading the
        # terminal loss / step-0 canary — no-op for local/dry-run.
        self._refresh_progress()
        path = Path(self._progress_jsonl)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    for raw in fh:
                        raw = raw.strip()
                        if not raw:
                            continue
                        rec = json.loads(raw)
                        if rec.get("loss") is not None:
                            last_loss = float(rec["loss"])
                        if rec.get("step") is not None:
                            last_step = int(rec["step"])
                        note = rec.get("note") or ""
                        if note.startswith("loss≈ln(vocab)"):
                            metrics["step0_canary"] = rec.get("loss")
            except (OSError, ValueError):  # pragma: no cover
                pass
        if last_loss is not None:
            metrics["final_loss"] = last_loss
        if last_step is not None:
            metrics["step"] = last_step
        metrics["final_status"] = self.status()
        return CheckpointRef(
            uri=self._checkpoint_uri,
            fmt="hf-safetensors-consolidated",
            representation_signature=self._representation_signature,
            model_signature=self._model_signature,
            metrics=metrics,
        )


# ---------------------------------------------------------------------------
# The ModelBuilder.
# ---------------------------------------------------------------------------


@dataclass
class NeMoModelBuilder:
    """ModelBuilder #1 — ``name="nemo"``. Decoder-CLM via NeMo AutoModel.

    A YAML-renderer + ``torchrun`` argv-builder + JSONL-tailing supervisor. It owns
    every NeMo-specific concern and imports NONE of nemo/torch (the recipe runs on
    the VM, inside the container, via the wrapping launcher)."""

    name: str = "nemo"

    # -- supports() ------------------------------------------------------

    def supports(
        self, *, model: ModelSpec, objective: Objective, corpus: PreparedCorpus
    ) -> Capability:
        """Accept ``family="decoder-clm"`` + ``objective.kind="next-token"`` +
        ``corpus.tensor_contract="clm/input_ids+labels/-100"``. Reject MLM (no MLM
        recipe) and any tensor-contract mismatch — surfaced as the
        ``REFUSED_CONTRACT`` reason (ARCHITECTURE §5.1)."""
        if model.family not in ("decoder-clm",):
            return Capability(
                supported=False,
                reason=(f"nemo builder supports family 'decoder-clm'; got "
                        f"{model.family!r} (no recipe for it)."),
            )
        if objective.kind not in ("next-token",):
            return Capability(
                supported=False,
                reason=(f"nemo builder supports objective 'next-token' "
                        f"(TrainFinetuneRecipeForNextTokenPrediction); got "
                        f"{objective.kind!r} — there is no MLM recipe."),
            )
        # The tensor-contract handshake: the corpus must produce what the objective
        # requires, and it must be the CLM contract NeMo's MaskedCrossEntropy honors.
        if corpus.tensor_contract != objective.requires_tensor_contract:
            return Capability(
                supported=False,
                reason=(f"tensor-contract mismatch: corpus produces "
                        f"{corpus.tensor_contract!r} but objective requires "
                        f"{objective.requires_tensor_contract!r}."),
            )
        if corpus.tensor_contract != _CLM_TENSOR_CONTRACT:
            return Capability(
                supported=False,
                reason=(f"nemo builder consumes {_CLM_TENSOR_CONTRACT!r} "
                        f"(MaskedCrossEntropy + -100 labels); got "
                        f"{corpus.tensor_contract!r}."),
            )
        return Capability(supported=True, reason=None)

    # -- plan() ----------------------------------------------------------

    def plan(
        self,
        *,
        corpus: PreparedCorpus,
        model: ModelSpec,
        objective: Objective,
        compute: ComputeTarget,
        budget: BudgetEnvelope,
        executor: Executor,
    ) -> CostPlan:
        """Render the YAML in memory, count params from the rendered arch, and
        derive the real-$ plan (tokens × params × GPU-$/h). NO launch."""
        return _derive_cost(corpus=corpus, model=model, compute=compute, budget=budget)

    # -- launch() --------------------------------------------------------

    def launch(
        self,
        *,
        corpus: PreparedCorpus,
        model: ModelSpec,
        objective: Objective,
        compute: ComputeTarget,
        budget: BudgetEnvelope,
        executor: Executor,
    ) -> TrainingHandle:
        """Materialize the rendered YAML, build the ``torchrun`` argv, and submit it
        through the executor (``gcp-vm``). Returns a :class:`NeMoTrainingHandle`
        whose ``stream_events()`` tails the launcher's JSONL.

        The rendered YAML + the JSONL progress path are written under the executor's
        run workspace (``corpus.extras['run_dir']`` when the verb threads one, else a
        workspace temp dir); the executor mounts/syncs them onto the VM. On a DRY-RUN
        executor (``LOOM_DRY_RUN`` / ``executor.dry_run``) NOTHING is launched — the
        argv + YAML are constructed and the handle is born terminal with no JSONL."""
        run_dir = self._run_dir(corpus)
        config_path = os.path.join(run_dir, "pretrain_financial_decoder.rendered.yaml")
        progress_jsonl = os.path.join(run_dir, "progress.jsonl")
        os.makedirs(run_dir, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as fh:
            fh.write(render_config_yaml(corpus=corpus, model=model, budget=budget))

        argv = build_torchrun_argv(
            config_path=config_path, corpus=corpus, budget=budget,
            compute=compute, progress_jsonl=progress_jsonl, executor=executor,
        )
        # The IN-CONTAINER path the launcher writes the JSONL to (what's in the
        # argv). The handle reads the control-plane ``progress_jsonl`` locally, and
        # for a live run asks the executor to fetch this remote path back into it.
        progress_jsonl_remote = to_container_path(progress_jsonl, executor)

        # The checkpoint dir the consolidated safetensors lands in (C5). Read off
        # the rendered config; the executor maps it to a VM-or-GCS uri it records.
        rendered = render_config(corpus=corpus, model=model, budget=budget)
        ckpt_dir = _dig(rendered, "checkpoint.checkpoint_dir") or "checkpoints"
        # The executor resolves the FINAL durable uri (gs:// or vm:/path) the
        # checkpoint is fetched from; default to the in-run-dir path for the local/
        # dry-run case so the verb records a real (non-tempdir) uri.
        checkpoint_uri = self._checkpoint_uri(executor, run_dir, str(ckpt_dir))

        collected: list[ProgressEvent] = []

        def _on_event(ev: ProgressEvent) -> None:
            collected.append(ev)

        job = executor.submit(
            argv=argv, image=compute.image, compute=compute, budget=budget,
            on_event=_on_event,
        )

        usd_per_gpu_hour = _gpu_usd_per_hour(compute.gpu_target)
        gpus = max(1, int(compute.nproc_per_node) * max(1, int(compute.nnodes)))
        rendered_arch = rendered["model"]["config"]
        return NeMoTrainingHandle(
            job_id=job.job_id,
            _job=job,
            _executor=executor,
            _progress_jsonl=progress_jsonl,
            _checkpoint_uri=checkpoint_uri,
            _representation_signature=corpus.representation_signature,  # ECHOED (§3/§7)
            _model_signature=_model_signature(model, objective, rendered_arch),
            _budget=budget,
            _usd_per_gpu_hour=usd_per_gpu_hour,
            _gpus=gpus,
            _vocab_size=corpus.vocab_size,
            _progress_jsonl_remote=progress_jsonl_remote,
            _events=collected,
            _dry_run=bool(getattr(executor, "dry_run", False)),
        )

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _run_dir(corpus: PreparedCorpus) -> str:
        """The per-run workspace the rendered YAML + JSONL are written under.

        Prefers a verb-threaded ``corpus.extras['run_dir']`` (store-anchored, the
        non-tempdir fix); else a workspace ``.loom/runs/<sig>`` dir; never a bare
        ``mkdtemp`` so the artifacts are addressable + the JSONL tail-able."""
        explicit = (corpus.extras or {}).get("run_dir")
        if explicit:
            return str(explicit)
        ws = os.environ.get("LOOM_WORKSPACE") or os.getcwd()
        sig = (corpus.representation_signature or "run")[:16]
        return os.path.join(ws, ".loom", "runs", f"nemo-{sig}-{int(time.time())}")

    @staticmethod
    def _checkpoint_uri(executor: Executor, run_dir: str, ckpt_dir: str) -> str:
        """Resolve the durable checkpoint uri the executor will fetch from.

        A ``gcp-vm`` executor exposes ``checkpoint_uri(run_dir, ckpt_dir)`` returning
        a ``gs://`` or ``vm:`` uri (the artifact bucket / VM path); fall back to the
        in-run-dir local path so the verb always records a REAL uri (never a tempdir,
        the §10 step-1-6 follow-up fix)."""
        resolver = getattr(executor, "checkpoint_uri", None)
        if callable(resolver):
            try:
                return str(resolver(run_dir, ckpt_dir))
            except Exception:  # noqa: BLE001 - fall back to the local path
                pass
        return os.path.join(run_dir, ckpt_dir)


# A minimal in-code copy of the verified committed config, used ONLY when the TFM
# repo file is unreachable (standalone install). Kept faithful to the committed
# structure so the diff-equal property holds in that case too.
_FALLBACK_REFERENCE_CONFIG: dict = {
    "model": {
        "_target_": "nemo_automodel.NeMoAutoModelForCausalLM.from_config",
        "config": {
            "_target_": "transformers.LlamaConfig",
            "vocab_size": 6251, "hidden_size": 512, "num_hidden_layers": 8,
            "num_attention_heads": 8, "num_key_value_heads": 2,
            "intermediate_size": 1408, "max_position_embeddings": 8192,
            "rope_theta": 500000.0, "hidden_act": "silu", "rms_norm_eps": 1.0e-5,
            "attention_dropout": 0.0, "tie_word_embeddings": False,
            "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
        },
    },
    "dataset": {"_target_": _DATASET_TARGET, "data_path": None,
                "merchant_hash_size": 2000, "seq_length": 4096},
    "dataloader": {"_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                   "collate_fn": "torch.utils.data.default_collate"},
    "validation_dataset": {"_target_": _DATASET_TARGET, "data_path": None,
                           "merchant_hash_size": 2000, "seq_length": 4096},
    "validation_dataloader": {"_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                              "collate_fn": "torch.utils.data.default_collate"},
    "step_scheduler": {"global_batch_size": 16, "local_batch_size": 16,
                       "ckpt_every_steps": 15, "val_every_steps": 15,
                       "num_epochs": 20, "max_steps": 30},
    "dist_env": {"backend": "nccl", "timeout_minutes": 5},
    "distributed": {"_target_": "nemo_automodel.components.distributed.fsdp2.FSDP2Manager",
                    "dp_size": "none", "dp_replicate_size": 1, "tp_size": 1,
                    "cp_size": 1, "sequence_parallel": False},
    "loss_fn": {"_target_": "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"},
    "optimizer": {"_target_": "torch.optim.AdamW", "betas": [0.9, 0.95],
                  "lr": 0.0002, "weight_decay": 0.077},
    "lr_scheduler": {"lr_decay_style": "cosine", "lr_warmup_steps": 10, "min_lr": 6.5e-6},
    "rng": {"_target_": "nemo_automodel.components.training.rng.StatefulRNG",
            "seed": 42, "ranked": True},
    "checkpoint": {"enabled": True, "checkpoint_dir": "models/decoder-demo/checkpoints/",
                   "model_save_format": "safetensors", "save_consolidated": True,
                   "model_repo_id": "custom-financial-decoder"},
}


# ARCHITECTURE §2.4 / §10 step 7: one-line registration under the registry key.
register_model_builder(NeMoModelBuilder())

__all__ = [
    "NeMoModelBuilder",
    "NeMoTrainingHandle",
    "render_config",
    "render_config_yaml",
    "config_overrides",
    "build_torchrun_argv",
    "to_container_path",
]
