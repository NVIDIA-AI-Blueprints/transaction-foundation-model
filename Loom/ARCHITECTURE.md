> **Status:** DRAFT, 2026-06-16. **Loom = a generic ports-and-adapters FM-training harness; TFM is one bundled triple.**

# Loom — a generic ports-and-adapters foundation-model training harness

**Thesis.** Loom is a typed DAG of stages — `ingest → prepare → pretrain → embed → evaluate → report` — over content-addressed, lineage-carrying data-objects. The harness is the constant (the verb spine, the typed result envelope, the lineage store, the gated/budget-enveloped launch, the telemetry capture, the checkpoint↔representation signature). The variable is three ports a campaign spec selects: a **data-representation port** (raw events → corpus), a **model-builder port** (corpus → checkpoint), and a thin **executor port** (where heavy work runs). An "FM kind" is a `(data-representation adapter, model-builder adapter, objective)` triple; **TFM is the bundled default triple** `event-sequence` + `nemo` + `next-token`. The load-bearing decision is that the contract is the *typed artifact between* two stages — a `PreparedCorpus`'s `signatures` dict + payload format, never a Python type — which is why streaming (I1) and a second adapter both land behind a contract the harness already validates, without touching the harness.

---

## 0. The model in one paragraph

Three discipline rules govern every choice below:
- **No port nobody implements.** Each port ships with a real second adapter on the roadmap; a would-be port with a single forever-adapter stays a harness constant (datastore, eval logic) or a value-object field (the budget envelope). The agent's reasoning LLM is explicitly **not** a Loom port.
- **The v0.1 core is untouched.** `registry.py`, `types.py`, `store.py`, `tools.py` are not edited; all new value-objects live in a fresh `loom/ports.py`. The extension points those four files already expose — a free-string `Diagnostic.contract`, open `signatures`/`kind`/`extras` dicts, the auto-gate on `LAUNCH_AND_TRACK` — are exactly what the ports use. (Two pre-existing stubs in `tools.py` get *implemented*, not rewritten.)
- **NeMo maps on as an adapter, never up into the harness.** Nothing in `loom/ports.py` names NeMo, FSDP2, torchrun, safetensors, RAPIDS, or BigQuery.

---

## 1. The harness core (the constant)

Committed and load-bearing; verified against the real code. A new adapter on either side inherits them unchanged.

- **The narrow waist — one typed declaration, two faces.** Each verb is declared once via `@register(...)` (`loom/registry.py`); from it Loom generates the human CLI (`loom <verb>`) and the agent tool (`loom.<verb>`). `loom <verb> --json` is byte-identical to `dispatch("loom.<verb>", …).to_json()`. This is *the* invariant (`types.py` header is marked `LOCKED`).
- **The result envelope** (`loom/types.py:VerbResult`, stable key order): `verb, status, verdict, tier, capability_mode, summary, outputs[], diagnostics[], data{}, experiment, cost_plan, confirm_token`. A `Diagnostic` is `{contract, severity, message, fix, data}` and **`contract` is a free-form string** (`types.py:138` — `"C1"|"C3"|"EDA"|…`, not an enum). This is the key extensibility hook: a port mints its own contract names with zero core change. `cost_plan` and `confirm_token` exist today as inert placeholders.
- **Tier is a property of the verb, not a flag** (`Tier{READ_ONLY, WORKSPACE_WRITE, EXPENSIVE, IRREVERSIBLE}`) plus `CapabilityMode{NONE, SEARCHABLE, LAUNCH_AND_TRACK}`. `tool_schema` already derives `gated = tier is IRREVERSIBLE or capability_mode is LAUNCH_AND_TRACK` and emits `disable_model_invocation` (`tools.py:31-44`). **So `pretrain` auto-gates the agent the instant it registers — no core edit.**
- **The `REFUSED_*` family already exists** (`types.py:39-46`, verified, complete list): `REFUSED_NO_METRIC`, `REFUSED_NO_BASELINE`, `REFUSED_NO_GPU_TARGET`, `REFUSED_AGENT_CANNOT_LAUNCH`, `REFUSED_NONINTERACTIVE_LAUNCH`, `REFUSED_SPEND_CAP`, `REFUSED_STALE`, `REFUSED_CONTRACT`. The launch verb returns the relevant members directly.
- **`CostPlan` is already GPU-shaped** (`types.py:108-126`): `derived, usd, confidence, tokens, params, seq_len, gpu_target, envelope, inputs`. Its docstring: *"must never be a hardcoded number once a GPU verb populates it."* Adapters fill it; the harness puts it on the envelope unchanged.
- **Lineage / content-addressed data-objects.** `loom/store.py:ObjectStore.{new_ref, put, get, find_by_content, content_id}`. `DataObject.kind` is a free string and `signatures`/`extras` are open dicts — nothing in the store knows what a vocab is; the corpus's `vocab_hash` lives in `signatures` *purely by convention*. The store today is strictly **local-filesystem** (`.loom/objects`, `payload_path` on disk, `open(payload_file)`); its docstring names itself the v0.2 Metaflow-datastore swap point. It cannot reach `gs://` at all today (see §6 / I1).
- **The gated, budget-enveloped launch.** Cost is *derived from the compiled plan*, never a label. Approval is a **binding `{max_steps, max_usd, max_wall_clock_min}` envelope** the orchestrator hard-kills at, not a one-time "go." The agent gets `status:PLAN` + a single-use, plan-hash-scoped, 15-min-expiry `confirm_token` and must make a second call but **cannot mint a launch** (`REFUSED_AGENT_CANNOT_LAUNCH`). The only real enforcement is the Pi `tool_call {block}` hook keyed on tool identity — `disable_model_invocation` is Loom-internal metadata, never a tool-schema lock.
- **Telemetry / moat capture.** Every verb appends a typed, redacted rollout; `owned_by=general` only trains. Harness-level — every adapter's runs feed it.
- **`make_confirm_token`/`validate_confirm_token`** are the only two core functions `pretrain` forces into existence (currently `NotImplementedError` stubs, `tools.py:89-101`). Implemented *once* at the harness level when `pretrain` lands: HMAC over `(plan_hash, expiry, nonce)`, single-use (nonce burned in the store), 15-min expiry, plan-hash-scoped.

---

## 2. The two ports + the executor (real interfaces)

New module `loom/ports.py` — pure `typing.Protocol` + frozen dataclasses, importing only `loom.types`. **It imports nothing from NeMo, torch, RAPIDS, or BigQuery.** Standalone placement (not appended to the LOCKED `types.py`) keeps the envelope module's blast radius zero.

### 2.0 Harness-level value objects (the inter-stage artifact contracts)

```python
# loom/ports.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Optional, Protocol, runtime_checkable
from .types import CostPlan, Diagnostic   # reuse the EXISTING envelope value-objects

Signature = str   # a content hash; the retrain trigger (generalizes today's vocab_hash)

@dataclass(frozen=True)
class SourceRef:                       # raw input to `prepare`; cloud-resident, NEVER local
    uri: str                           # bq://project.dataset.table  |  gs://bucket/prefix/
    snapshot: dict = field(default_factory=dict)  # {max_event_date|date_prefix_range} — provenance anchor

@dataclass(frozen=True)
class PreparedCorpus:                  # prepare → pretrain handoff. OPAQUE to the model-builder.
    representation: str                # registry name, e.g. "event-sequence"
    representation_signature: Signature # generalizes today's vocab_hash (the retrain trigger)
    tensor_contract: str               # "clm/input_ids+labels/-100" (C4) | "mlm/..." | "vision/patches"
    train_uri: str                     # gs://.../train/shard-*.arrow  (NOT an in-RAM list — I1)
    val_uri: Optional[str]
    test_uri: Optional[str]
    manifest_uri: str                  # counts, snapshot range, split definition
    seq_length: int
    pad_token_id: int
    vocab_size: Optional[int]          # None for continuous representations; feeds model arch (C1→arch)
    effective_tokens: int              # measured NON-redundant tokens → feeds the D/N param-scaling gate
    provenance: dict = field(default_factory=dict)   # snapshot range / MAX(event_date) — separate from C2
    extras: dict = field(default_factory=dict)

@dataclass(frozen=True)
class ModelSpec:                       # architecture, framework-neutral
    family: str                        # "decoder-clm" | "encoder-mlm" | "vision-mae"  (picks objective family)
    arch: dict                         # {"_target_": "transformers.LlamaConfig", "vocab_size": ..., ...}
    init: str = "from_scratch"         # | "from_pretrained:<hub_id_or_uri>"

@dataclass(frozen=True)
class Objective:                       # the third leg of an FM kind — a plain record, NOT a port
    kind: str                          # "next-token" | "masked-lm" | "contrastive"
    requires_tensor_contract: str      # the C4 string the corpus MUST satisfy
    loss_target: Optional[str] = None  # advisory; adapter validates support (NeMo CLM ⇒ MaskedCrossEntropy)

@dataclass(frozen=True)
class ComputeTarget:                   # consumed by the executor; NOT itself a port
    launcher: str                      # "torchrun" | "local" | "metaflow-gcp" | "modal"
    nproc_per_node: int = 1
    nnodes: int = 1
    accelerator: Literal["gpu", "cpu"] = "gpu"
    gpu_target: Optional[str] = None   # echoes LOOM_GPU_TARGET; None ⇒ REFUSED_NO_GPU_TARGET
    image: Optional[str] = None        # echoes LOOM_NEMO_IMAGE
    parallelism: dict = field(default_factory=dict)   # {dp,tp,cp,sequence_parallel} — adapter maps to its knobs

@dataclass(frozen=True)
class BudgetEnvelope:                  # the BINDING approval object — orchestrator hard-kills at these
    max_usd: float
    max_wall_clock_min: Optional[int] = None
    max_steps: Optional[int] = None

@dataclass(frozen=True)
class Capability:                      # supports() return — declare/reject an FM-kind combo
    supported: bool
    reason: Optional[str] = None       # why a combo is rejected → a REFUSED_CONTRACT diagnostic

@dataclass(frozen=True)
class CheckpointRef:                   # pretrain → embed handoff
    uri: str                           # consolidated HF safetensors dir
    fmt: str                           # "hf-safetensors-consolidated"  (C5)
    representation_signature: Signature # ECHOED from PreparedCorpus → the harness-level pairing invariant
    model_signature: Signature         # hash of {arch, objective, code_sha} — the replay anchor
    metrics: dict = field(default_factory=dict)   # final/last loss, step, val, step0_canary

@dataclass(frozen=True)
class ProgressEvent:                   # what feeds the Pi widgets
    step: int
    loss: Optional[float]
    usd_spent: float
    usd_envelope: float
    gpu_pct: Optional[float] = None
    wall_clock_min: float = 0.0
    phase: str = "train"               # "warmup"|"train"|"val"|"consolidate"
    note: Optional[str] = None         # e.g. step-0 canary "loss≈ln(vocab)"
```

`SourceRef`, `PreparedCorpus`, `CheckpointRef`, `BudgetEnvelope`, `Signature` are **harness-level** — they are the narrow waist *between* the ports. `CostPlan` is **reused from `types.py`**, not redefined: its GPU fields are already shaped.

### 2.1 The data-representation port

```python
@runtime_checkable
class DataRepresentation(Protocol):
    name: str                                                  # registry key; "event-sequence" is adapter #1
    produces_tensor_contract: str                              # "clm/input_ids+labels/-100" (the C4 it emits)
    def build_spec(self, args: dict) -> Any: ...               # was tokenize._build_spec
    def compile(self, spec: Any, *, context_len: int) -> Any: ...   # was engine.compile_spec (data-free, deterministic)
    def contracts(self, compiled: Any) -> list[Diagnostic]: ...      # PORT-LOCAL C1/C2/C3 cards
    def representation_passed(self, compiled: Any) -> bool: ...      # [FIX] the contract-NAME-AGNOSTIC write-gate
    def plan(self, *, spec: Any, source: SourceRef,
             executor: "Executor") -> CostPlan: ...            # CPU-cheap, or BigQuery-bytes estimate
    def materialize(self, *, compiled: Any, source: SourceRef,
                    executor: "Executor") -> PreparedCorpus: ...# the I1 scale-out (§6) — writes SHARDS
    def signatures(self, compiled: Any) -> dict: ...           # the harness handoff dict (today: tokenize.py:399-407)
```

`compile()` returns the abstraction of today's `CompiledTokenizer`; every representation exposes a stable `representation_signature` (today's `vocab_hash`) and declares its `tensor_contract`. `contracts()` returns `Diagnostic(contract=…)` cards under *its own* names. **`representation_passed(compiled)` is the one method that generalizes the corpus write-gate** (the major fix, detailed in §8): it returns `True` iff the compiled object carries **no ERROR-severity Diagnostic**, regardless of contract string. The default implementation any adapter inherits is literally `not any(d.severity is Severity.ERROR for d in self.contracts(compiled))`; the event-sequence adapter satisfies it for free because `ContractReport.add` already flips `passed=False` on every ERROR (`api.py:276-280`), and C1 injectivity, C1 density, and C3 grammar all emit `Diagnostic(severity=ERROR)` (`contracts.py:116,137,206`). `materialize()` is the only method that touches scale — it runs the harness-owned executor (§6) and returns a `PreparedCorpus` pointing at sharded object storage, never an in-RAM list.

### 2.2 The model-builder port

```python
@runtime_checkable
class TrainingHandle(Protocol):                                # the launch-and-track capability surface
    job_id: str
    def stream_events(self) -> Iterator[ProgressEvent]: ...    # → loom top --json widget feed (source = §5.1)
    def status(self) -> Literal["pending","running","succeeded","failed","killed","stopped_at_budget"]: ...
    def cancel(self) -> None: ...
    def result(self) -> CheckpointRef: ...                     # blocks until terminal

@runtime_checkable
class ModelBuilder(Protocol):
    name: str                                                  # "nemo" (#1) | "local" (CPU rehearsal, #2)
    def supports(self, *, model: ModelSpec, objective: Objective,
                 corpus: PreparedCorpus) -> Capability: ...    # declare/reject an FM-kind combo BEFORE spend
    def plan(self, *, corpus: PreparedCorpus, model: ModelSpec, objective: Objective,
             compute: ComputeTarget, budget: BudgetEnvelope, executor: "Executor") -> CostPlan: ...   # NO launch
    def launch(self, *, corpus: PreparedCorpus, model: ModelSpec, objective: Objective,
               compute: ComputeTarget, budget: BudgetEnvelope, executor: "Executor") -> TrainingHandle: ...
```

### 2.3 The executor port (the scale-out seam)

The single graft that fixes the leak both reviewing personas flagged in the minimal designs: **the GPU launch and the corpus-build fan-out share one swappable seam.** Without it, `launcher: local → metaflow-gcp → modal` selection and the budget hard-kill would be re-implemented inside every model-builder *and* every representation's `materialize`. With it, the executor is the *only* place that knows how work is submitted, fanned out, and killed; `LOOM_GPU_TARGET` becomes a real boundary instead of an env string read in N adapters.

```python
@runtime_checkable
class Executor(Protocol):
    name: str                                                  # "local" | "metaflow-gcp" | "modal"
    def gpu_available(self) -> bool: ...                       # gates REFUSED_NO_GPU_TARGET
    def submit(self, *, argv: list[str], image: Optional[str], compute: ComputeTarget,
               budget: BudgetEnvelope, on_event) -> "JobHandle": ...   # runs `torchrun … train_decoder_model.py`
    def foreach(self, *, fn, shards: list[str], compute: ComputeTarget) -> list[str]: ...  # the Tier-B fan-out (§6)
    def kill(self, job_id: str) -> None: ...                   # the binding-envelope hard-kill
```

`ModelBuilder.launch()` calls `executor.submit(...)`; `DataRepresentation.materialize()` calls `executor.foreach(...)`. Both flow through the same budget/kill contract — which is why gated-launch behavior is identical regardless of adapter.

The datastore is deliberately **not** a port: `loom/store.py` is already the documented v0.2 Metaflow swap point and has exactly one forever-implementer per deployment, so it stays a concrete class. **[FIX]** The I1 streaming reader needs a `store.open_shards(uri) -> ShardReader` (lazy/mmap streaming read), but this is **not** a trivially-additive local method: the current `ObjectStore` is local-filesystem only (`.loom/objects`, `payload_path` on disk, `open(...)`) and cannot read `gs://...` at all. `open_shards` therefore **lands together with the v0.2 cloud datastore backend (the Metaflow-datastore swap)**, when the store gains real `gs://` read capability — not as a one-line method stub on today's local store. The eval logic (C6) likewise stays in `baseline.py`/`evaluate` as harness code, not a port.

### 2.4 The registries

Three module-globals mirroring the proven `REGISTRY: dict[str, Verb]` pattern:

```python
REPRESENTATIONS: dict[str, DataRepresentation] = {}
MODEL_BUILDERS:  dict[str, ModelBuilder] = {}
EXECUTORS:       dict[str, Executor] = {}
def register_representation(r): REPRESENTATIONS[r.name] = r; return r
def register_model_builder(b):  MODEL_BUILDERS[b.name] = b; return b
def register_executor(e):       EXECUTORS[e.name] = e; return e
```

---

## 3. The adapter contract

An adapter implements one Protocol plus a one-line registration. It touches **none** of the harness mechanics directly — it fills value-objects and yields events; the harness renders the envelope, runs the gate, writes lineage. Six obligations:

| Obligation | How the adapter expresses it | Backing seam (verified) |
|---|---|---|
| **Declare tier / capability** | *Not the adapter's* — the **verb** declares it. `prepare` registers `tier=WORKSPACE_WRITE`; `pretrain` registers `tier=EXPENSIVE, capability_mode=LAUNCH_AND_TRACK`. | `registry.py:register(...)`; the moment `pretrain` registers, `tools.py:31-44` auto-sets `disable_model_invocation` and arms the `REFUSED_*` machinery |
| **Emit the cost PLAN** | `ModelBuilder.plan()` renders config in-memory (no launch), reads `max_steps`/`global_batch_size`/param-count/`ComputeTarget`, returns `CostPlan(derived=True, usd=6·params·tokens·…, params=…, tokens=…, seq_len=…, gpu_target=…, envelope=…)`. | `types.py:CostPlan` (all fields already present) |
| **Gated launch handshake** | `pretrain` returns `status=PLAN` + a `confirm_token` when `usd` exceeds threshold; on the confirmed second call it calls `launch()`. | `tools.py:make/validate_confirm_token` (the two stubs `pretrain` forces real) |
| **Launch-and-track** | `launch()` returns a `TrainingHandle`; the harness wires `stream_events()` into the widget feed and `executor.kill()`s at the `BudgetEnvelope`. Terminal `stopped_at_budget` ⇒ `DataObject` status `STOPPED_AT_BUDGET` + resume token. | binding envelope; `Verdict.INCOMPLETE` already exists for partial checkpoints |
| **Checkpoint↔representation signature** | `launch().result()` returns `CheckpointRef.representation_signature` **echoed verbatim from `PreparedCorpus.representation_signature`**, plus its own `model_signature = hash{arch, objective, code_sha}`. `embed`/`evaluate` assert equality before any forward pass; mismatch ⇒ `REFUSED_CONTRACT`. | `store.py:DataObject.signatures` (open dict); `types.py:Status.REFUSED_CONTRACT` already exists |
| **Progress for Pi widgets** | `TrainingHandle.stream_events()` yields `ProgressEvent{step, loss, usd_spent, usd_envelope, gpu_pct, phase, note}`. The first event carries the **step-0 loss canary** `loss ≈ ln(vocab_size)` in `note` — a harness-level sanity narration wired into the port's stream, *not* into NeMo. **[FIX] The event SOURCE is net-new adapter work, not a pre-existing wire** (§5.1). The surfacing path (`loom top`, the sparkline/curve widget) is **also wholly net-new** (§10, step 7b). | `store.py` (the run record persisted per event) |

Any contract finding — port-local or harness-level — is a `Diagnostic(contract="C1"|"C5"|…)` card in `diagnostics[]`. The free-string `contract` field is the universal extension mechanism; a new adapter mints its own contract names with zero core change.

---

## 4. The campaign spec — selecting the FM kind

A campaign spec is the *single declaration* that names one adapter per port; the harness renders it to the human CLI flags and the agent tool args identically (the narrow waist). An **FM kind is this whole block** — a path through the stage graph.

```yaml
# campaign: tfm-dex-phase1.yaml   — an FM KIND is this whole block
fm_kind: tfm                       # human label; expands to the triple below
representation:
  adapter: event-sequence          # → REPRESENTATIONS["event-sequence"]   (data-representation port #1)
  preset: chain                     # adapter-local arg (today's spec.preset, engine/spec.py:338)
  source: bq://level-mark-437714-b1.mbd_recs.cross_chain_interactions   # pathspec, NEVER local
  snapshot: { max_event_date: auto }   # records the provenance anchor (separate from C2)
objective:
  kind: next-token                 # requires tensor_contract clm/input_ids+labels/-100
model:
  builder: nemo                    # → MODEL_BUILDERS["nemo"]              (model-builder port #1)
  family: decoder-clm
  arch: { _target_: transformers.LlamaConfig, hidden_size: 512, num_hidden_layers: 8, vocab_size: from-corpus }
execution:
  adapter: metaflow-gcp            # → EXECUTORS["metaflow-gcp"]; echoes LOOM_GPU_TARGET
  compute: { launcher: torchrun, nproc_per_node: 8, parallelism: { dp: auto, tp: 1, cp: 1 } }
budget: { max_usd: 312, max_wall_clock_min: 240, max_steps: 3000 }
```

**Resolution flow** in the `pretrain` verb (a deterministic harness-owned planner — no LLM in the hot path):

```python
corpus  = store.get(corpus_ref)                      # → PreparedCorpus from its signatures + payload
repr_   = REPRESENTATIONS[spec.representation]
builder = MODEL_BUILDERS[spec.model_builder]
execu   = EXECUTORS[spec.execution]
obj     = Objective(spec.objective, requires_tensor_contract=…, loss_target=…)
# 1. validate the FM kind BEFORE any spend:
cap = builder.supports(model=model_spec, objective=obj, corpus=corpus)
if not cap.supported:                                # e.g. local + masked-lm, or tensor-contract mismatch
    return REFUSED_CONTRACT with cap.reason as a Diagnostic card
# 2. plan → PLAN+stop;  3. human/agent confirm;  4. launch under the binding envelope.
```

`vocab_size: from-corpus` is resolved by reading `corpus.vocab_size` (originally `Corpus.signatures["vocab_size"]`) — the model arch is parameterized by the representation's hand-counted vocab (**C1**), threading the two ports *without the builder importing `TokenizerSpec`*. This closes the gap the real config exposes (it hardcodes `vocab_size: 6251`).

`--model-builder {local,nemo}` is the CLI shorthand for `model.builder`. **`tokenize` survives as an alias** of `prepare` bound to `representation="event-sequence"` — TFM out of the box, and the byte-identical dual-driver contract is preserved.

---

## 5. The NeMo AutoModel reference adapter (#1) + event-sequence representation (#1) = TFM

### 5.1 `loom/adapters/nemo_builder.py` — ModelBuilder #1: a YAML-renderer + argv-builder + process supervisor

Mapped to the **real** `configs/pretrain_financial_decoder.yaml` and `scripts/train_decoder_model.py` (4-line recipe shell, verified). Everything NeMo-specific stays in this file; none leaks into the port.

- **`supports()`** → accepts `family="decoder-clm"`, `objective.kind="next-token"`, `corpus.tensor_contract="clm/input_ids+labels/-100"`; **rejects MLM** (no MLM recipe) with a reason. Also enforces the tensor-contract handshake: `corpus.tensor_contract == objective.requires_tensor_contract`, else `Capability(False, …)` → `REFUSED_CONTRACT`.
- **`plan()`** → renders the YAML *in memory*, reads `step_scheduler.max_steps` + `global_batch_size`, computes params from `model.arch` (`transformers.LlamaConfig`), multiplies by `ComputeTarget` GPU type/count → `CostPlan(derived=True, …)`. No launch. This arms the Pi gated-launch card.
- **`launch()`** → materializes the YAML, builds the dotted-override argv, and calls `executor.submit(argv=…)`:
  ```
  torchrun --nproc-per-node={compute.nproc_per_node} scripts/train_decoder_model.py \
     -c <rendered.yaml> \
     --dataset.data_path            {corpus.train_uri} \
     --validation_dataset.data_path {corpus.val_uri} \
     --step_scheduler.max_steps     {budget.max_steps}
  ```
  The recipe driver is the verified 4-liner: `parse_args_and_load_config()` → `TrainFinetuneRecipeForNextTokenPrediction(cfg)` → `.setup()` → `.run_train_validation_loop()`. The adapter honors the documented gotcha (`configs/…yaml:47-48`): use `torchrun` directly; the `automodel` CLI misparses `--nproc-per-node`.
- **The `_target_` map it owns** (verified in the config):

  | Block | `_target_` | Role |
  |---|---|---|
  | `model` | `nemo_automodel.NeMoAutoModelForCausalLM.from_config` | builds an HF decoder from-scratch |
  | `model.config` | `transformers.LlamaConfig` | architecture = a swappable HF config |
  | `dataset` / `validation_dataset` | file-path `src/clm_data.py:build_financial_clm_dataset`, `data_path: null` | the custom corpus→tensors builder (the I1 seam, §6) |
  | `dataloader` | `torchdata.stateful_dataloader.StatefulDataLoader` | resumable loading |
  | `distributed` | `nemo_automodel.components.distributed.fsdp2.FSDP2Manager` (`dp_size: none`→infer, `tp_size:1`) | sharding; maps from `ComputeTarget.parallelism` |
  | `loss_fn` | `nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy` | honors `-100` labels |
  | `checkpoint` | `model_save_format: safetensors`, `save_consolidated: true` | **C5 portability** |

- **`result()`** → resolves the `save_consolidated: true` + `model_save_format: safetensors` output dir into `CheckpointRef(fmt="hf-safetensors-consolidated", representation_signature=<echoed>, model_signature=hash{arch,objective,code_sha})`. Downstream `embed` loads it with vanilla `AutoModelForCausalLM.from_pretrained` — zero NeMo dependency (C5).
- **`stream_events()` — [FIX] the progress source is net-new, not a scrape of unverified stdout.** Verified: `scripts/train_decoder_model.py` prints only a config header (arch/hidden/layers/vocab/max_steps); there is **zero loss output in the visible 4-line driver**. Per-step loss is emitted from *inside* NeMo's `run_train_validation_loop()`, whose stdout/log grammar is a NeMo-AutoModel internal **not pinned or verified anywhere in this repo**. Because the progress stream is load-bearing (it is the entire launch-and-track widget feed + the budget telemetry), the adapter must NOT rest on a free-form stdout grammar. The progress feed is built as net-new adapter work, using a robust source in this priority order: **(a)** wrap the recipe with a thin logging callback/hook if NeMo-AutoModel exposes one; else **(b)** have the adapter itself write a **structured JSONL** (step, loss, lr, tokens) from a wrapping training loop / checkpoint-dir tailer that it owns; else **(c)** parse a *pinned* log format only after verifying it against one real run. The step-0 `loss ≈ ln(vocab_size)` canary and the loss extraction are **part of this net-new adapter work**, not a pre-existing wire.

Scaling is a config edit (`dp/tp/cp`), not a code change — the config is already multi-node-shaped.

### 5.2 `loom/adapters/event_sequence.py` — DataRepresentation #1

The existing `loom/engine/` package **is** this adapter, wrapped behind the Protocol with near-zero change. The methods delegate 1:1 to functions that already exist and are exported from `loom/engine/__init__.py`:

| Port method | Existing implementation (verified) |
|---|---|
| `build_spec(args)` | `tokenize.py:_build_spec` (financial/chain presets) |
| `compile(spec, context_len)` | `engine.compile_spec` → `CompiledTokenizer` (config-only, deterministic, emits `vocab_hash`) |
| `contracts(compiled)` | `engine/contracts.py` C1/C2/C3 → `Diagnostic(contract="C1"…)` |
| `representation_passed(compiled)` | `compiled.report.passed` — already `False` iff any ERROR fired (`api.py:276-280`); the inherited ERROR-scan default returns the identical verdict |
| `materialize(compiled, source, executor)` | generalizes `engine.materialize_corpus_lines` (the `spec.preset=="chain"` switch at `engine/spec.py:338` *becomes* the registry lookup) — now writes **sharded Arrow** to GCS (§6) |
| `signatures(compiled)` | exactly the dict at `tokenize.py:399-407`, with `vocab_hash`→`representation_signature`, `encode_path`→`representation` |
| `produces_tensor_contract` | `"clm/input_ids+labels/-100"` |

`CompiledTokenizer.vocab_hash` **is** `representation_signature`. The dataset's `{input_ids, labels}` / `-100` obligation (verified in `src/clm_data.py:90-98`) is the `tensor_contract` string — the narrow waist both adapters agree on but neither owns.

---

## 6. Cloud / streaming data path (I1 — the hard gate)

The corpus is **never** a local file. Sources are cloud-resident and large: the `cross_chain_interactions` mart (BigQuery), the `embed-pipeline-datasets` GCS Datasets tier (NDJSON, append-only, retention forever), public BigQuery DEX exports, and MBD (950M). I1 — replacing the everything-in-RAM Python list in `src/clm_data.py:load_corpus_and_tokenize` (verified: `sequences = []; … sequences.append(…)`, lines 123-130) — is a **hard gate**, not a follow-up: it OOMs the scaled/MBD runs.

The architectural claim: **the data-representation port's output contract is `(sharded integer corpus on object storage, manifest)`, produced by a harness-owned scale-out backend (the executor), and consumed via the unchanged NeMo `_target_` seam swapped from in-RAM-list to streaming-shard reader.** This makes I1 a property of the *port-and-executor boundary*, not a patch.

**Two-tier pushdown inside `EventSequenceRepresentation.materialize()`:**

- **Tier A — BigQuery pushdown (heavy reduction in SQL).** Entity-grouping, temporal ordering, the **C6** temporal∘wallet-disjoint split, continuous→bin discretization (`amt_bin = sum(amt >= threshold)` → pure SQL `CASE`/`>=`), and calendar extraction (`ts.dt.hour/dayofweek/month` → SQL) all run against `cross_chain_interactions` (or GCS external tables). No raw-row egress, no single-box sort. Output: partitioned, pre-bucketed, pre-ordered Parquet on GCS.
- **Tier B — RAPIDS-on-GPU via `executor.foreach()` over shards.** The genuinely GPU-shaped steps (hash tokenization `hash_values()`, the spec-compiler interleave to token-ID streams, chunking into ≤`seq_length` sequences) run as a fan-out over the Parquet shards — each task a short-lived on-demand GPU worker reading **one shard** with `cudf.read_parquet` and writing one corpus shard; it never holds the whole corpus in GPU memory.

> **Scope honestly:** Tier-B `executor.foreach` over per-shard GPU workers is **net-new infrastructure.** Today's `scripts/gcp-gpu-up.sh` drives a single named VM; Metaflow is not wired into the active training repo. The `metaflow-gcp` executor adapter is the thing that generalizes the one-shared-VM pattern into a per-shard fan-out — build it as new infra, not as a rename of the existing scripts. The `local` executor (in-process `foreach`) and a single-VM executor are the cheaper intermediate adapters that exercise the seam first.

**The corpus artifact (the I1 unlock):**
```
gs://embed-pipeline-datasets/loom-corpora/<repr-signature>/<snapshot-range>/{train,val,test}/shard-NNNN.arrow
+ manifest.json
```
`manifest.json` carries `vocab_size` (C1), `representation_signature`, `tokens_per_event`/`chunk_size` (C3), the snapshot/`MAX(event_date)` range (provenance), the C6 split definition, shard/row/token counts, and the **measured non-redundant `effective_tokens`** (the D/N param-scaling gate — never scale params on calendar; DEX swaps are low-entropy). These become `PreparedCorpus` fields and the `DataObject.signatures`/`extras`.

**Feeding NeMo (surgical, inside the existing seam):** replace the in-RAM `build_financial_clm_dataset`/`load_corpus_and_tokenize` with a **streaming/memory-mapped Arrow-shard reader** behind the *same* `build_*_clm_dataset(**kwargs)` signature; `--dataset.data_path` points at the GCS shard prefix (already `null` in the config, awaiting an override). The `_target_` mechanism, `torchrun`, FSDP2, and HF-consolidated safetensors are **all unchanged** — NeMo only ever sees an iterable yielding `{input_ids, labels}`, sharded so each rank streams its own subset. The `gs://` reads themselves require the **cloud datastore backend** (`store.open_shards`, §2.3) — i.e. I1 ships with the v0.2 datastore swap, not against today's local store. **The model-builder port and the NeMo adapter need zero change to gain streaming** — the test that the seam sits in the right place.

---

## 7. The contract split (port-local vs harness-level)

| Contract | Where it lives | Why |
|---|---|---|
| **C1** vocab injectivity + density (the MONTH_12/CARD_0 collision) | **data-representation port-local** | meaningless without a discrete vocab; in `engine/contracts.py`, emitted as `Diagnostic(contract="C1", severity=ERROR)` |
| **C2** config-only determinism / fitted-artifact | **data-representation port-local** | `compile_spec` is data-free; the fitted state lives on the compiled object. *Distinct* from provenance (below) |
| **C3** corpus grammar `<bos> txn (<sep> txn)* <eos>`, `tokens_per_event`, `chunk_size = context_len//(tokens_per_txn+1)` | **data-representation port-local** | CLM grammar of *this* representation; an MLM/vision rep declares its own analogues |
| tensor-contract *realization* (NeMo's `MaskedCrossEntropy` honoring `-100`), FSDP2/torchrun mechanics, the `_target_` surface, the `automodel`-vs-`torchrun` gotcha | **model-builder port-local** | NeMo-specific realization |
| **C4** the `{input_ids, labels}/-100` tensor contract, as the `tensor_contract` *interface string* | **harness-level (the narrow waist)** | the representation *produces* it, the builder+objective *consume* it — neither owns it alone; lives on `PreparedCorpus.tensor_contract` |
| **the corpus write-gate** (refuse to persist a corpus when contracts fail) | **harness-level mechanism, port-local verdict** | **[FIX]** the *decision* is generic — `prepare` refuses on `not repr.representation_passed(compiled)` (any ERROR-severity Diagnostic), never on a named contract; *which* contracts can fail is the representation's business |
| **C5** checkpoint portability (`hf-safetensors-consolidated`) | **harness-level** | a 2nd builder (DeepSpeed) must also emit it ⇒ proves it isn't NeMo's; `embed` asserts `from_pretrained` loads |
| **C6** eval hygiene (temporal ∘ entity-disjoint split + row-IDs) | **harness-level** (in `baseline.py`/`evaluate`, downstream of both ports; produced once in BigQuery before any GPU spend) | representation-independent; already partly in `baseline.py` as a cheap leave-one-last-out analogue, not the full split |
| **checkpoint↔representation signature pairing** | **harness-level invariant** | `representation_signature` threaded `PreparedCorpus → CheckpointRef`; mismatch ⇒ `REFUSED_CONTRACT`. Hardens the gap the docs flag as missing today |
| **provenance / snapshot anchor** (for live cloud sources) | **harness-level** | the GCS date-prefix range / `MAX(event_date)` baked into `PreparedCorpus.provenance` and the data-object name |
| lineage, budget envelope, cost PLAN, launch-and-track, telemetry, dual-driver | **harness-level / constant** | the brief's "harness is the constant" |

The mechanism for all port-local contracts is identical and already in the core: a free-string `Diagnostic.contract` (`types.py:138`) + open `signatures`/`kind`/`extras` (`store.py`). **C2 is kept honest:** the port-local C2 is strictly *config-only determinism of the tokenizer*; the live-source provenance anchor is a *separate* harness-level concern. Conflating them (treating provenance as "C2-equiv" inside the port) would misplace the snapshot boundary, which must travel with every corpus regardless of representation.

---

## 8. How the v0.1 engine refactors onto this

**Untouched (verified clean — the constant):**
- `loom/registry.py` — verb declaration + dispatch; gains sibling registries in `loom/ports.py`, the `register` mechanism unchanged.
- `loom/types.py` — the LOCKED envelope, `CostPlan` (GPU fields already present), all `REFUSED_*` enums, `Diagnostic.contract` free string, `Tier`/`CapabilityMode`. **No edit** — new value-objects live in `loom/ports.py`.
- `loom/store.py` — `DataObject` (free `kind`, open `signatures`/`extras`), `ObjectStore.{new_ref,put,get,find_by_content,content_id}`. **No edit in the no-GPU slice.** `open_shards`/`gs://` reads land later, *with the v0.2 cloud datastore backend* (§2.3), not as a local one-liner.
- `loom/tools.py` — the `make/validate_confirm_token` stubs get *implemented* (not rewritten) when `pretrain` lands.
- `loom/verbs/ingest.py`, `loom/eda.py` — already sniff columns + scan identity leakage; never compile a spec. **No change.**
- `loom/verbs/baseline.py` — C6 hold-out logic stays here.

**Lifted behind a port (code essentially verbatim):**
- `loom/engine/{api,spec,strategies,contracts}.py` → wrapped as `EventSequenceRepresentation` in `loom/adapters/event_sequence.py` (delegations only). The `spec.preset=="chain"` switch (`engine/spec.py:338`) becomes the `REPRESENTATIONS` lookup — the proto-registry made real.

**Recast (the seam) — and the ONE piece that is *generalized*, not merely delegated:**
- `loom/verbs/tokenize.py` → generic `loom/verbs/prepare.py`. Three of the four representation touchpoints (`_build_spec`, `compile_spec`, `_materialize_corpus`) become straight `repr.build_spec/compile/materialize` delegations, and the surrounding `content_id` dedupe, `new_ref`/`put`, the `signatures={…}` block (`tokenize.py:399-407`), and the `VerbResult` assembly are reused verbatim.
- **[FIX] The write-REFUSAL is the one piece that must be generalized.** Today `tokenize.py:254-295` decides the refusal by **naming contracts and reading tokenizer-shaped attributes**: `if not (compiled.report.injective and compiled.report.dense)` (C1) and `any(d.contract == "C3" and d.severity is Severity.ERROR …)` (C3). A second representation (encoder-MLM, vision-patches) has no C1 injectivity / C3 grammar and no `report.injective/.dense` attribute at all — so this verb body would either wrongly let a failing MLM corpus through, or force a harness edit per new representation, breaking the "second adapter slots in without touching the harness" test. The generic `prepare.py` therefore replaces both hardcoded blocks with a **single contract-name-agnostic gate**:
  ```python
  diags = repr.contracts(compiled)                 # the port's own cards, whatever they are named
  if not repr.representation_passed(compiled):     # == no ERROR-severity Diagnostic; the port decides
      return VerbResult(status=REFUSED_CONTRACT, verdict=FAIL,
                        tier=WORKSPACE_WRITE, diagnostics=diags, …)   # no Corpus written
  ```
  This is **behavior-preserving for event-sequence**: `representation_passed` is `compiled.report.passed`, and `ContractReport.add` already sets `passed=False` on every ERROR — and C1 injectivity (`contracts.py:116`), C1 density (`:137`), and C3 grammar (`:206`) all emit `Diagnostic(severity=ERROR)`, so the exact same corpora are refused. But the verb no longer knows the strings "C1"/"C3" or the `report.injective/.dense` shape — an MLM rep folds its own masking-ratio / span-grammar failures into ERROR-severity Diagnostics and is gated identically. `tokenize` is re-registered as an **alias** bound to `representation="event-sequence"` (the registry already supports `aliases`, `registry.py:66,80,104`).

**Net-new (does not exist anywhere under `loom/` — verified: `verbs/` holds only `tokenize|ingest|baseline`, `_VERB_MODULES = ("tokenize","ingest","baseline")`):**
- `loom/ports.py` (the three Protocols + value-objects + three registries).
- `loom/adapters/{event_sequence,local_builder,nemo_builder}.py` and the executor adapters.
- The `pretrain`, `embed`, `evaluate`, `report` verbs (add to `_VERB_MODULES`). `embed` stays a **distinct** budgeted GPU verb — deliberately split from `pretrain`, never re-merged into one "train" call.
- **The `loom top` verb + the progress/sparkline Pi widget** — **[FIX] wholly net-new**, the same honesty class as Tier-B `foreach`. No `top` verb exists (`_VERB_MODULES` has none) and the only built widget under `cli/dist/widgets` is `contract-diff` — there is no progress/sparkline widget. `loom top` is a new verb to add to `_VERB_MODULES` and the live curve is a new Pi widget (with a mandatory ANSI/braille sparkline fallback when images are unavailable), not an existing feed.

**Net churn to the harness: zero edits to the four core files in the no-GPU slice**; one new types module, three adapter wrappers, one verb rename-with-alias *whose only generalized line is the ERROR-scan refusal*, two stub implementations, and one new `top` verb + widget. The `gs://` datastore capability is a separate, later v0.2 change.

---

## 9. One plausible SECOND adapter per port (the generality proof)

| Port | Adapter #1 (TFM) | Adapter #2 (proof) | Why it needs **zero harness change** |
|---|---|---|---|
| **data-representation** | `event-sequence` (tokenized CLM) | **`masked-sequence`** (encoder-MLM) — declares `tensor_contract="mlm/input_ids+attention_mask+masked_labels"`, its own C1-analogue (vocab) and C3-analogue (masking ratio / span grammar) under fresh `Diagnostic.contract` names, folding failures into ERROR-severity so `representation_passed` gates it without the verb naming a contract. A **`vision-patches`** adapter is equally addable — no MCC, no `<bos> txn` grammar; emits `"vision/patches+labels"`, does no Tier-A SQL, patches images in its own Tier-B fan-out. | Same `PreparedCorpus`/manifest/signature contract; mints its own contract strings (free field); the generic ERROR-scan refusal + executor fan-out + lineage are untouched. |
| **model-builder** | `nemo` (FSDP2/torchrun/HF-safetensors) | **`local` (PPMI+SVD, CPU rehearsal)** — torch-free, `plan()` returns ~$0, `launch()` runs a real-but-trivial loop on the same C4 shards, **still emits `CheckpointRef{fmt:"hf-safetensors-consolidated"}`** so C5 round-trips. A `deepspeed` variant slots in identically, mapping `ComputeTarget.parallelism` to ZeRO stages. | Proves C5 is harness-level, not NeMo's; the port names neither FSDP2 nor torchrun. |
| **executor** | `metaflow-gcp` (on-demand GPU fan-out) | **`local`** (in-process `foreach`, single box) / **`modal`** — `gpu_available()`, `submit()` run `torchrun` locally or on a Modal image. | Already a committed env-seam (`LOOM_GPU_TARGET`); same `JobHandle`/budget-kill contract. |
| **objective** *(record, not port)* | `next-token` | **`masked-lm`** — `requires_tensor_contract="mlm/…"`. Forces a different representation and a builder whose `supports()` accepts MLM. | Objective-as-first-class-argument (not hardwired) is exactly what makes the MLM FM kind addable. |

The common requirement that proves the seam is in the right place: every adapter takes a `PreparedCorpus` it does **not** interpret beyond `tensor_contract`, a framework-neutral `ModelSpec`/`Objective`, a `ComputeTarget` whose `parallelism` it maps to its own knobs, an `Executor` it submits through, and must return a `CostPlan` (pre-launch) + a portable `CheckpointRef`. **Nothing in `loom/ports.py` names NeMo, FSDP2, torchrun, RAPIDS, BigQuery, safetensors, or vocab.**

**The single cheapest generality proof = the `local` CPU-rehearsal model-builder.** With `ComputeTarget.accelerator="cpu"`, `nproc_per_node=1`, tiny `max_steps`, it exercises the entire model-builder port + the `LAUNCH_AND_TRACK` auto-gate + the `confirm_token` round-trip + the C5 checkpoint round-trip with **no GPU and no CI fixture** — directly addressing the GPU-only and no-CI sharp edges and proving the abstraction is not NeMo-shaped.

---

## 10. The concrete NEXT BUILD SLICE

In dependency order. Steps 1-6 require **no GPU and no NeMo** and fully exercise both ports + the executor + the gate + the C5/signature invariant; steps 7-8 attach NeMo and MBD-scale streaming behind the unchanged `_target_` seam.

1. **`loom/ports.py`** — the three Protocols (`DataRepresentation` incl. `representation_passed`, `ModelBuilder`, `Executor`), the value-objects (`SourceRef`, `PreparedCorpus`, `ModelSpec`, `Objective`, `ComputeTarget`, `BudgetEnvelope`, `CheckpointRef`, `ProgressEvent`, `TrainingHandle`, `Capability`), and the three registries. Pure types, no behavior, no core edit. *(Test: `runtime_checkable` isinstance checks; nothing imports NeMo/torch/RAPIDS.)*
2. **`loom/adapters/event_sequence.py`** — wrap `loom/engine/` to satisfy `DataRepresentation` (delegations only); `representation_passed` returns `compiled.report.passed`. *(Test: identical `vocab_hash` and `Diagnostic`s to today's `tokenize`.)*
3. **`loom/verbs/prepare.py`** — generalize `tokenize.py` to the `REPRESENTATIONS[…]` lookup; keep all store/envelope plumbing; **replace the hardcoded C1/C3 refusal with the single `not repr.representation_passed(compiled)` ERROR-scan gate** (the §8 fix); re-register `tokenize` as the `representation="event-sequence"` alias. Add a `local` executor (in-process `foreach`). *(Test: `loom tokenize --json` byte-identical to before, incl. the C1-collision and C3-grammar REFUSED_CONTRACT envelopes; `dispatch("loom.tokenize",…)` byte-identical to the CLI — the locked dual-driver invariant.)*
4. **`loom/adapters/local_builder.py`** — ModelBuilder #2 (PPMI+SVD, CPU, ~$0), emitting a tiny real consolidated HF-safetensors so C5 round-trips. **Build this before NeMo:** it is the first thing that exercises the whole port + gate + C5, GPU-free, and becomes the CI oracle.
5. **Implement `make_confirm_token`/`validate_confirm_token`** in `tools.py` — HMAC over `(plan_hash, expiry, nonce)`, single-use nonce burned in the store, 15-min expiry, plan-hash-scoped. Wire the Pi `tool_call {block}` hook keyed on tool identity.
6. **`loom/verbs/pretrain.py`** — register `tier=EXPENSIVE, capability_mode=LAUNCH_AND_TRACK` (⇒ `disable_model_invocation` auto-true), add to `_VERB_MODULES`. Resolve the campaign triple → `builder.supports()` (reject incoherent combos / tensor-contract / signature mismatch with `REFUSED_CONTRACT`) → `builder.plan()` → `status=PLAN` + `confirm_token` if over threshold → on confirmed second call, `builder.launch()` via the executor → write `Checkpoint` `DataObject` with `signatures.representation_signature` echoed (the C5/C1 cross-port match) + `model_signature`. Honor `REFUSED_NO_GPU_TARGET`/`REFUSED_SPEND_CAP`/`REFUSED_AGENT_CANNOT_LAUNCH`. *(Test: with `--model-builder local`, full PLAN→confirm→checkpoint round-trip, no GPU.)*
7. **`loom/adapters/nemo_builder.py` + `metaflow-gcp` executor** — the YAML-renderer + dotted-override argv-builder + `torchrun` supervisor over the real config/launcher; `result()` resolves the consolidated safetensors. **(7a) Progress source = net-new:** `stream_events()` is fed from a NeMo logging callback/hook if one exists, else a structured JSONL the adapter writes itself, else a *pinned* log format verified against one real run — **not** a scrape of the unverified recipe stdout (the visible driver prints no per-step loss). The step-0 `loss≈ln(vocab)` canary is emitted by this adapter code. **(7b) `loom top` verb + progress widget = net-new:** add `top` to `_VERB_MODULES` and build the live-curve Pi widget (sparkline fallback). *(Test: rendered YAML diff-equals the committed `pretrain_financial_decoder.yaml` for the TFM triple; cost PLAN derived, not labeled; the JSONL/callback emits ≥1 `ProgressEvent` on a 2-step smoke run.)*
8. **I1 streaming reader + v0.2 cloud datastore** — `event_sequence.materialize()` runs the Tier-A BigQuery pushdown → Tier-B `executor.foreach` over shards, emitting sharded Arrow on GCS; swap `clm_data.py`'s in-RAM loader for a memory-mapped/streaming Arrow-shard reader behind the same `build_*_clm_dataset(**kwargs)` signature. **`store.open_shards()` lands here, together with the cloud datastore backend that can read `gs://`** (not a local one-liner). *(Test: a multi-shard GCS corpus trains under `local` without loading all shards into RAM; harness + NeMo adapter unchanged — the seam test.)*

**Critical path to first real GPU run:** 1→2→3→4 gives a free, CPU-rehearsed, end-to-end DAG + the contract tests (incl. the generalized refusal proven byte-identical on the C1/C3 cases); 5→6 lands the gated `pretrain`; 7 attaches real NeMo on the existing single-`corpus.json` payload; 8 unlocks D1/MBD scale (and the cloud datastore) — and lands *inside* the representation adapter + executor without touching anything built in steps 1-7, the final proof the seam sits in the right place.

---

### Key file map (all absolute)

- **Harness core (untouched in the no-GPU slice):** `/Users/anub/Work/transaction-foundation-model/Loom/loom/{registry,types,store,eda}.py`, `/Users/anub/Work/transaction-foundation-model/Loom/loom/verbs/{ingest,baseline}.py`
- **Confirm-token stubs to implement:** `/Users/anub/Work/transaction-foundation-model/Loom/loom/tools.py:89-101`
- **Lift behind the representation port:** `/Users/anub/Work/transaction-foundation-model/Loom/loom/engine/{api,spec,strategies,contracts}.py` (preset switch `engine/spec.py:338`; ERROR-severity C1/C3 cards `contracts.py:116,137,206`; `ContractReport.add` flips `passed` on ERROR `engine/api.py:276-280`)
- **Recast seam (`tokenize` → `prepare`), incl. the GENERALIZED refusal:** `/Users/anub/Work/transaction-foundation-model/Loom/loom/verbs/tokenize.py` (the hardcoded C1/C3 refusal to replace: `254-295`; `_build_spec:65`, `_materialize_corpus:126`, `_contract_diagnostics:139`, `signatures={…}` block `399-407`) and `loom/verbs/__init__.py` (`_VERB_MODULES`)
- **Net-new:** `/Users/anub/Work/transaction-foundation-model/Loom/loom/ports.py`, `/Users/anub/Work/transaction-foundation-model/Loom/loom/adapters/{event_sequence,local_builder,nemo_builder}.py`, `loom/verbs/{pretrain,top}.py`, the progress/sparkline Pi widget
- **NeMo integration reality:** `/Users/anub/Work/transaction-foundation-model/configs/pretrain_financial_decoder.yaml` (the `_target_` map; `torchrun` gotcha `47-48`; `save_consolidated:true` `155-156`; hardcoded `vocab_size:6251` closed by `from-corpus`), `/Users/anub/Work/transaction-foundation-model/scripts/train_decoder_model.py` (the 4-line recipe launcher; **prints only a config header — no per-step loss**, hence the net-new progress source), `/Users/anub/Work/transaction-foundation-model/src/clm_data.py` (the in-RAM I1 site, `load_corpus_and_tokenize:123-130`; the `{input_ids,labels}/-100` C4 contract `90-98`)
