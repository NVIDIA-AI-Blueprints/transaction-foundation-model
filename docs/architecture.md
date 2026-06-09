# Loom architecture

Loom is **an agentic CLI for the full data-science lifecycle** —
**not** an automated ML engine. It's a catalog of `/loom-*` verbs (`connect · eda ·
features · pipeline · train · optimize · validate · viz · report · deploy · ops ·
collab`) spanning the whole lifecycle; ML *modeling* (the `optimize` verb, AIDE's
search) is the ~3% "brain", and the other **97%** — data access, EDA, features,
pipelines, training, validation, viz, reporting, deployment, ops, collaboration —
is the product. It is built as **ports and adapters** ("providers"), the same way
Kubernetes treats container runtimes as pluggable: `loom-core` defines the provider
*interfaces* (the ports); concrete *adapters* implement them and are chosen purely
by configuration. Adding a new brain, MLOps backend, or model-builder is a new
adapter class plus a one-line registration — never a core change.

This document covers the provider model, the **four seams** (search, execution,
model, model-builder), the two execution paths (the search loop and the lifecycle
flows), the request lifecycle, and how to add a provider.

## The big picture

```
                        ┌──────────────────────── controller.run_loom ───────────────────────┐
                        │                                                                     │
  Task ────────────────▶│  exec_cls   = get_execution(config.mlops_provider)   # registry     │
  LoomConfig ──────────▶│  search_cls = get_search(config.search_provider)     # by name      │
                        │  execution  = exec_cls(config);  search = search_cls(config)         │
                        │  corpus     = Corpus(config)                                         │
                        │  execution.setup(task)                                               │
                        │  result = search.run(task, execute=execution,                        │
                        │                      on_node=corpus.record, budget=config.budget)    │
                        │  execution.teardown()                                                │
                        └──────────┬──────────────────────────────────────┬───────────────────┘
                                   │                                       │
                          SearchProvider  (brain)              ExecutionProvider  (muscle)
                             default: aide                  default: metaflow │ local (dev)
                                   │                                       │
                propose code ─▶ execute(code, reset) ─▶ ExecutionResult ─▶ score ─▶ NodeRecord
                                                                                       │
                                                                                       ▼
                                                                            Corpus (JSONL, owned_by)
```

The controller (`loom/controller.py`) is the only orchestration seam. It resolves
provider **classes by name** from the registry and never imports a concrete
adapter directly — that indirection is what makes Loom pluggable.

**Two execution paths share these ports.** The diagram above is the **search /
`optimize`** path (`controller.run_loom` — the AIDE brain proposing candidates).
The other **lifecycle verbs** (`eda`, `features`, `validate`, `train`, `deploy`, …)
run a **static Metaflow flow per verb through the MLOps interface**
(`ExecutionProvider.run_flow(flow_path, params, tags) -> RunResult`), each producing
a versioned **Metaflow run + an `@card`** plus a typed summary — never an in-process
candidate. `train` additionally resolves the **model-builder** port (below) inside
its flow. Both paths stay backend-swappable because the verb speaks the *interface*,
never a concrete backend.

## The four ports (seams)

The search, execution, and model-builder ports are abstract base classes in
`loom/providers/__init__.py`; the model (LLM-backend) port lives in
`loom/providers/model/__init__.py`. Every provider is constructed uniformly as
`Provider(config)` so the controller (and the AIDE adapter, for the model port)
can wire them symmetrically.

### SearchProvider — the brain

Proposes, executes, scores, and records candidate solutions. One abstract method:

```python
class SearchProvider(ABC):
    name: str = "search"

    @abstractmethod
    def run(
        self,
        task: Task,
        execute: ExecCallback,
        on_node: Optional[OnNode] = None,
        budget: Optional[object] = None,
    ) -> SearchResult: ...
```

A search provider runs its own loop: propose code → run it via the supplied
`execute` callback → score the result → emit one `NodeRecord` per finished node
through `on_node` → return the best solution as a `SearchResult`. The default
adapter is `aide`.

### ExecutionProvider — the muscle (MLOps)

Runs candidate code and returns an `ExecutionResult`. It is also **callable**,
which is the crucial cross-seam trick (see below):

```python
class ExecutionProvider(ABC):
    name: str = "execution"

    @abstractmethod
    def execute(self, code: str, reset_session: bool = True) -> ExecutionResult: ...

    __call__ = execute          # an ExecutionProvider *is* an ExecCallback

    def setup(self, task: Task) -> None: ...     # stage ./input + empty ./working, set cwd
    def teardown(self) -> None: ...
    def runs(self, experiment_id: str) -> list[dict]: ...   # leaderboard, default []
```

Implementations stage a workspace (populate `./input` from `task.data_dir`,
create an empty `./working`, set the current working directory), execute code,
and optionally expose a leaderboard via `runs()`. Built-in adapters are
`metaflow` (default) and `local` (a Metaflow-free dev path).

### ModelProvider — the LLM backend

The third port. It decides *which model the brain talks to and how it is
authenticated*, configuring AIDE's **existing** LLM backends via config/env (plus
an optional runtime dispatch override) **without forking AIDE**. It lives in
`loom/providers/model/__init__.py`:

```python
@dataclass(frozen=True)
class ModelRoute:
    model_name: str
    base_url: str | None = None
    key_env: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)
    judge_capable: bool = True
    kind: str = "api"                 # api | openai-compat | cli-bridge
    attribution_headers: dict | None = None


class ModelProvider(ABC):
    name: str = "model"

    @abstractmethod
    def resolve(self, role: str) -> ModelRoute: ...        # role in {"code", "feedback"}
    def prepare_env(self, env: MutableMapping[str, str]) -> None: ...
    def preflight(self, role: str) -> list[str]: ...
    def install_dispatch_override(self, aide_backend_module) -> None: return None
```

A model provider is resolved **per role** — `"code"` writes solutions, `"feedback"`
is AIDE's judge (`submit_review`, which **always** uses tool calling). `resolve`
returns a `ModelRoute`; `prepare_env` materializes that route's knobs
(`OPENAI_BASE_URL`, the key var AIDE reads) into the environment **before the first
query** — AIDE memoizes its clients with funcy `@once`, so env must be set up
front. `preflight` returns human hints for missing creds/login; `install_dispatch_override`
is a no-op except for `cli-bridge` kinds.

Why this matters: **AIDE routes purely by model name.** Its `determine_provider`
sends `claude-*` to the native Anthropic backend, `gpt-*`/`o<N>` to the OpenAI
backend, `gemini-*` to Gemini; otherwise, if `OPENAI_BASE_URL` is set, to the
OpenAI backend in chat-completions mode against that URL; else to the dedicated
OpenRouter backend. Two sharp edges the providers route around:

- The dedicated **OpenRouter** backend raises `NotImplementedError` for any
  `func_spec`, and the judge always passes one — so the `openrouter` provider
  forces the OpenAI-compatible path (set `OPENAI_BASE_URL`, copy the key) rather
  than letting AIDE reach that backend. `judge_capable` reflects a best-effort
  check of OpenRouter's `?supported_parameters=tools` listing.
- Real **OpenAI** model names make the OpenAI backend hardcode `api.openai.com`
  and *ignore* `OPENAI_BASE_URL` — so `openai-api` deliberately leaves it unset.

Built-in adapters: `anthropic-api` (default), `openai-api`, `openrouter`, `nim`,
`openai-compat`, and the two `cli-bridge` adapters `claude-subscription` /
`codex-subscription`.

### The cli-bridge kind and the dispatch override

A `cli-bridge` provider has no API key — it drives the user's **own** local CLI
(`claude -p` / `codex exec`). It can't use any of AIDE's vendor backends, so at run
setup it installs a `loom_query` into `aide.backend.provider_to_query_func` for the
provider key its sentinel model name routes to (`claude-code-subscription` →
`anthropic`; `codex-mini-latest` → `openai`). The helper in
`loom/providers/model/_dispatch.py` first runs a **signature smoke test** (assert
`aide.backend.query`, `provider_to_query_func` dict, and `determine_provider` all
exist) so an upstream reshape fails loudly here rather than mid-run, then binds the
wrapper. The wrapper honors AIDE's `(output, req_time, in_tokens, out_tokens, info)`
return contract; for `func_spec` (judge) calls it returns the parsed dict matching
`func_spec.json_schema` — coerced from the CLI's text (Claude) or forced via
`codex exec --output-schema` (Codex). This module never edits files under `aide/`;
it patches the in-memory dispatch table.

### ModelBuilderProvider — the model-builder (training)

The fourth port: the heavy **training/serving** backend behind `/loom-train`,
parallel to the search and execution ports (ABC in `loom/providers/__init__.py`,
adapters in `loom/providers/model_builder/`). It is stated in **Loom DS-intent
vocabulary, never backend nouns** — the adapter is a *compiler that lowers intent →
backend config*, and every backend noun (`Megatron`, `.nemo`, `@resources(gpu=8)`)
lives inside the adapter and nowhere else.

```python
OBJECTIVES = frozenset({"next-event", "masked-field", "contrastive"})
BUDGETS    = frozenset({"probe", "small", "full"})
MODES      = frozenset({"batch", "online"})

class ModelBuilderProvider(ABC):
    name: str = "model_builder"

    @abstractmethod
    def manifest(self) -> CapabilityManifest: ...                 # the only required method
    def tokenize(self, sequences_ref, scheme)            -> ArtifactRef: ...  # searchable
    def pretrain(self, sequences_ref, objective, budget) -> ArtifactRef: ...  # launch-and-track
    def finetune(self, backbone_ref, task_ref, recipe)   -> ArtifactRef: ...  # searchable
    def embed(self, backbone_ref, data_ref)              -> ArtifactRef: ...  # searchable
    def evaluate(self, model_ref, holdout_ref, metric)   -> Scores: ...       # searchable
    def serve(self, model_ref, mode)                     -> ArtifactRef: ...
```

I/O are **Metaflow artifact pathspecs** (via `loom.dataio` / the Client API), never
`.nemo` files or raw object storage — a backbone/tokenizer/embeddings/model is only
ever a pathspec, so the rest of the lifecycle composes with it. Each capability
declares a **`mode`** in `manifest()`: `pretrain` is `launch-and-track` (heavy GPU —
AIDE must **not** tree-search it; only `/loom-train` invokes it), while
`tokenize/finetune/embed/evaluate` are `searchable` (cheap scalars `/loom-optimize`
may search). This encodes the AIDE-vs-builder division of labor as a *provider fact*,
not user knowledge — and it is asserted by the golden conformance suite.

`manifest()` is the only required method; a backend overrides only the capabilities
it supports and declares the rest unsupported, so a capability gap is **refused up
front with a clear message** rather than failing deep in a GPU job. Built-in
adapters:

- **`nemo`** (default) — a lowering compiler. It maps Loom intent → NeMo config
  (`pretrain(objective="next-event")` → an AutoModel causal-LM recipe;
  `budget="full"` → `@resources(gpu=8)` + Megatron parallelism), estimates cost
  (GPU-count · hours · $), and gates the real GPU launch behind `--launch`. With no
  `gpu_target` configured it returns a clean `REFUSED_NO_GPU_TARGET` (never
  launches); with a target but `launch=False` it returns a staged `PLANNED` plan —
  the same posture as `deploy --apply`.

  **GPU launch targets (`LOOM_GPU_TARGET`).** When the gate ALLOWS (`--launch`
  with a target set), the lowered plan is routed by launcher: **`modal`** (or
  `modal://<app>`) — the v0.2 default — submits the NeMo NGC container training to
  an **on-demand H100 via Modal** (the laptop stays the control plane; the GPU
  burst is ephemeral), then snapshots the produced checkpoint **back as a Metaflow
  artifact pathspec** (never a checkpoint file or object-store URI). `modal` is an
  optional, lazily-imported dep — absent, the launcher refuses with an actionable
  install/auth message. Any other target → `REFUSED_UNKNOWN_GPU_TARGET` (listing
  the supported launchers). The container image is read from `LOOM_NEMO_IMAGE` at
  the point of use (env only, never committed).
- **`local`** — a **torch-free CPU stand-in** (PPMI + TruncatedSVD sequence
  embeddings) that actually builds a backbone/embeddings end-to-end, deterministically
  and in seconds, with zero new dependencies. It is the conformance/CI/dev default
  path. An optional torch GRU fidelity mode hides behind the `model-local` extra and
  falls back to the numpy path when torch is absent.

It **hides vocabulary, not physics**: `budget="full"` still means real GPU-hours,
surfaced at the expensive/mutate approval gate — Loom hides *Megatron parallelism*,
never *this costs $X and takes Y hours*.

## The cross-seam contract

Two design decisions let any brain drive any muscle with zero glue:

1. **`ExecutionProvider` is an exec-callback.** The search side expects a
   callable `ExecCallback = Callable[[str, bool], ExecutionResult]`, i.e.
   `(code, reset_session) -> ExecutionResult`. Because `ExecutionProvider.__call__`
   aliases `execute`, the chosen execution provider *is* a valid exec-callback and
   is passed straight into `search.run(..., execute=execution, ...)`. Do not break
   this aliasing.

2. **`ExecutionResult` is field-identical to AIDE's.** The locked type in
   `loom/types.py` matches `aide.interpreter.ExecutionResult` exactly:

   ```python
   @dataclass
   class ExecutionResult:
       term_out: list[str]
       exec_time: float
       exc_type: str | None
       exc_info: dict | None = None
       exc_stack: list[tuple] | None = None
   ```

   Field parity means the AIDE adapter can convert between the two types with a
   straight `aide.interpreter.ExecutionResult(**dataclasses.asdict(result))` — no
   bespoke mapping. If AIDE's type ever changes shape, this dataclass must be
   updated to match it verbatim.

## Core types (`loom/types.py`)

- **`ExecutionResult`** — the locked, AIDE-parity result above.
- **`Task(data_dir, goal, eval, experiment_id, tenant="default", dataset_ref=None)`**
  — one experiment: what to achieve, how it's scored, the grouping id, the tenant,
  and **where the input data is**, via two fields the active provider picks
  between:
  - `dataset_ref` — a Metaflow **pathspec** (e.g. `"IngestDataset/123"`)
    identifying the ingested **data object**. The `metaflow` provider reads it via
    the Client API (`loom.dataio`); see [The data model](#the-data-model-input-is-a-metaflow-data-object) below.
  - `data_dir` — a plain local directory, used by the `local` (Metaflow-free) dev
    provider.
- **`SearchResult(best_code, best_metric, journal_path, tree_path, node_count)`**
  — the outcome of a search.
- **`NodeRecord(...)`** — one finished search node, persisted by the corpus
  (carries `code`, `term_out`, `metric`, model/token routing metadata, and the
  `tenant` / `owned_by` IP tags).
- **`RunResult(pathspec, successful, card_path, summary, error)`** — the outcome of
  a **lifecycle flow** run through `ExecutionProvider.run_flow` (a Metaflow run + an
  `@card` + a typed summary); the mandated artifact of every `/loom-*` lifecycle verb.
- **Model-builder types** (the `ModelBuilderProvider` contract):
  - **`ArtifactRef(pathspec, kind, summary, error)`** — a model-builder output: a
    Metaflow run pathspec + a small JSON-able summary; `kind` ∈
    `backbone | tokenizer | embeddings | model | endpoint`. **Never a `.nemo` file
    or a storage URI** — bulk stays in Metaflow as named artifacts.
  - **`Scores(metric, value, detail)`** — the scalar(s) `evaluate` returns (a derived
    number + small detail like `baseline`/`lift`/`n_holdout`, never rows).
  - **`Capability(name, mode, supported, notes)`** — one declared capability and its
    AIDE-search `mode` (`searchable` | `launch-and-track`); stand-ins carry an honest
    `notes` ("don't over-sell").
  - **`CapabilityManifest(backend, capabilities)`** — what `manifest()` returns;
    `supports(name)` / `mode_of(name)` drive the up-front capability negotiation.

## The data model: input is a Metaflow data object

Loom's input data is a **Metaflow data object — a Metaflow Artifact** — referenced
by **pathspec** and read **only through the Metaflow Client API**
(`metaflow.Run(pathspec).data.<artifact>`). This is load-bearing:

- **One external→Metaflow boundary.** `loom ingest` runs the
  `IngestDataset(FlowSpec)` flow (`flows/ingest_dataset.py`) once to turn a local
  dir/CSV into Metaflow artifacts — `train` / `test` (DataFrames) and a `schema`
  dict — and prints the run's pathspec. That pathspec is the `dataset_ref` carried
  on the `Task`.
- **One data-load path.** `loom/dataio.py` is the *only* place Loom reads that
  data: `resolve_run(dataset_ref)` returns a `metaflow.Run`;
  `materialize_dataset(dataset_ref, dest_dir)` writes `train.csv` / `test.csv` into
  a host-local dir; `dataset_schema(dataset_ref)` returns the `schema` dict. All
  three go through the Client API and lazy-import `metaflow` inside the function.
- **The datastore is opaque.** Where the artifacts physically live — **local or
  object storage (S3/minio)** — is an implementation detail **Metaflow owns**,
  configured solely in the Metaflow profile/environment (`METAFLOW_PROFILE` /
  `METAFLOW_*`). **Loom code never touches that storage directly**: no
  object-storage SDK, no bucket-URI literals, no raw-URI Metaflow datastore
  handle anywhere in `loom/` or `flows/` (a test scans the source to keep it so).
  Loom is agnostic — it only ever sees artifacts.

Both execution paths converge on the same on-disk workspace shape (`./input` with
`train.csv`/`test.csv`, an empty `./working`): the `metaflow` provider materializes
the data object into `./input` via the Client API, while the `local` provider
copies `data_dir` into `./input`. So a candidate solution and AIDE's data-preview
read `./input/...` identically regardless of where the bytes came from.

## Configuration (`loom/config.py`)

`LoomConfig` is the single source of truth for provider selection, model routing,
and budget. It is loaded with precedence: dataclass defaults < optional YAML <
environment / `.env` < explicit overrides. Key fields:

- `search_provider` (default `aide`), `mlops_provider` (default `metaflow`).
- `code_provider` / `feedback_provider` (both default `anthropic-api`) — the model
  ("LLM backend") provider per role (env `LOOM_CODE_PROVIDER` /
  `LOOM_FEEDBACK_PROVIDER`; flags `--code-provider` / `--feedback-provider`, or
  `--model-provider` for both).
- `metaflow_profile` (env `METAFLOW_PROFILE`) — lets a tenant point Loom at their
  own Metaflow endpoint (BYO perimeter).
- `code_model` / `feedback_model` — the model names. `nim_base_url` (env
  `OPENAI_BASE_URL`) and `model_base_url` (env `LOOM_MODEL_BASE_URL`) are the
  OpenAI-compatible endpoints used by the `nim` and `openai-compat` providers. The
  matching API key (`NVIDIA_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  `OPENROUTER_API_KEY`) is read from the environment **at the point of use** and
  never stored on the config — model providers only move or pass through what the
  user already set.
- `budget` — `BudgetConfig(steps, num_drafts, debug_prob, max_debug_depth)`.
- `corpus_path`, `tenant`, `owned_by`.

**Secrets are never hardcoded or stored on the config object.**

## The corpus and the IP boundary (`loom/corpus.py`)

Every `NodeRecord` a search provider emits is appended to a JSONL corpus at
`config.corpus_path`. The corpus enforces a single, generic, multi-tenant IP
boundary:

- A record's `owned_by` is the IP owner. The sentinel `"general"` means the
  record is not owned by any specific tenant and may be used across tenants; any
  other value tags it as tenant-owned.
- `Corpus.general()` returns only the `"general"` records — the slice a
  cross-tenant "moat" model is allowed to train on. Tenant-owned records stay
  isolated.

The boundary is domain-neutral: the corpus knows nothing about any customer or
vertical, only the generic `owned_by` tag. Secrets are never persisted — a
`NodeRecord` has no field for key material.

Telemetry (`loom/telemetry/`) keeps the same posture and two **separate** planes:
the **training corpus** (complete, append-only, transcript-style — every
trajectory in full, no sampling) vs **ops telemetry** (the optional, sampled OTel
mirror behind `LOOM_TELEMETRY_OTEL_OPS`) — kept separate so the corpus never
flows through a sampling/aggregating backend.

## Request lifecycle

0. (metaflow path) `loom ingest --source <path>` runs `IngestDataset` once via
   `metaflow.Runner`, producing a data object whose pathspec is printed. The user
   passes it as `loom run --dataset <pathspec>`.
1. The CLI (`loom run …`) builds a `Task` (with `dataset_ref` for the metaflow
   path, or `data_dir` for the local path) and a `LoomConfig`, then calls
   `controller.run_loom(task, config)`. A pre-flight requires the metaflow provider
   to have a `--dataset` (or, as a fallback, a local `--data`) and guides the user
   to `loom ingest` otherwise.
2. The controller resolves the execution and search provider **classes** from the
   registry by their configured names, instantiates each from the config, and
   creates a `Corpus`.
3. `execution.setup(task)` records the input reference (the metaflow provider
   carries `task.dataset_ref`; the local provider stages `./input` from
   `task.data_dir` and sets cwd). For the metaflow path the AIDE adapter
   materializes the data object to a host-local dir via the Client API
   (`loom.dataio`) and points `cfg.data_dir` at it, so AIDE's data-preview reflects
   the real dataset.
4. `search.run(task, execute=execution, on_node=corpus.record, budget=…)` runs
   the loop: propose → `execute(code)` → score → `on_node(NodeRecord)` for each
   finished node. Inside the AIDE adapter's `_build_cfg`, the code and feedback
   `ModelProvider`s are resolved, their model names set on
   `cfg.agent.code/feedback.model`, `prepare_env(os.environ)` is called for each
   (before the first query), and any `cli-bridge` provider installs its dispatch
   override into `aide.backend` — all once at run setup. The CLI has already
   pre-flighted credentials/login and the judge's tool capability.
5. `execution.teardown()` is called in a `finally`, then the `SearchResult` is
   returned. The CLI prints the best metric, artifact paths, and a leaderboard.

## Built-in adapters (v0.1)

| Name | Port | Module | Notes |
| --- | --- | --- | --- |
| `aide` | search | `loom/providers/aide_search.py` | Drives AIDE's agent/journal loop (pinned by SHA `40dcf28`). Does **not** edit AIDE; converts results via field parity. |
| `metaflow` | execution | `loom/providers/metaflow_exec.py` + `flows/eval_candidate.py` | Runs each candidate through **one static** `EvalCandidate(FlowSpec)` via `metaflow.Runner`; the candidate enters as **data** (`IncludeFile`), never as a generated flow. Input is a Metaflow data object referenced by `task.dataset_ref` (read via the Client API). BYO Metaflow endpoint. |
| `local` | execution | `loom/providers/local_exec.py` | Metaflow-free dev path; runs candidates in-process via the vendored interpreter. Input is a local `task.data_dir`. |
| `anthropic-api` | model | `loom/providers/model/anthropic_api.py` | **Default.** Native Claude via AIDE's Anthropic backend; reads `ANTHROPIC_API_KEY`. Judge-capable. |
| `openai-api` | model | `loom/providers/model/openai_api.py` | Native OpenAI (`gpt-*`/`o<N>`); reads `OPENAI_API_KEY`. Leaves `OPENAI_BASE_URL` unset (real OpenAI ignores it). |
| `openrouter` | model | `loom/providers/model/openrouter.py` | OpenRouter via the OpenAI-compatible path; copies `OPENROUTER_API_KEY` → `OPENAI_API_KEY`. Per-slug judge check. |
| `nim` | model | `loom/providers/model/nim.py` | NVIDIA NIM via the OpenAI-compatible path; copies `NVIDIA_API_KEY` → `OPENAI_API_KEY`. |
| `openai-compat` | model | `loom/providers/model/openai_compat.py` | Generic self-host (LiteLLM/vLLM/Ollama); `OPENAI_BASE_URL` from `model_base_url`, `OPENAI_API_KEY` passthrough. |
| `claude-subscription` | model | `loom/providers/model/claude_subscription.py` | `cli-bridge`: drives the user's local `claude -p`. No key; checks CLI + login. |
| `codex-subscription` | model | `loom/providers/model/codex_subscription.py` | `cli-bridge`: drives the user's local `codex exec` (judge via `--output-schema`). No key; checks CLI + `~/.codex/auth.json`. |
| `nemo` | model-builder | `loom/providers/model_builder/nemo.py` | **Default.** Lowers Loom intent → NeMo config; estimates cost; gates the real GPU launch behind `--launch` (refuses cleanly with no `gpu_target`). All NeMo nouns confined here. |
| `local` | model-builder | `loom/providers/model_builder/local.py` | Torch-free CPU stand-in (PPMI + TruncatedSVD); builds a backbone/embeddings end-to-end, deterministic, zero new deps. The conformance/CI/dev default. |

The vendored interpreter (`loom/providers/_interpreter.py`) is a dependency-light
port of AIDE's interpreter that produces a Loom `ExecutionResult`. It is shared by
the `local` path and the Metaflow `evaluate` step, so neither hard-depends on AIDE
internals.

### Why one static Metaflow flow

`EvalCandidate(FlowSpec)` (`start → evaluate → validate → end`) is defined once.
Each candidate solution enters it as **data** (an `IncludeFile` candidate plus
`Parameter`s for goal/eval/timeout/seed/`dataset_ref`), never as a freshly
generated flow per candidate. This keeps the flow definition stable, cacheable,
and inspectable across every evaluation. The `start` step materializes the input
data object into `./input` through the Client API (`loom.dataio.materialize_dataset`)
— never by touching the datastore directly. The separate `IngestDataset(FlowSpec)`
(`start → end`) is the one place outside data crosses into Metaflow. Use standard
Metaflow APIs only.

The **lifecycle verbs** follow the same discipline: each is its own static
`FlowSpec` (`flows/eda.py`, `features.py`, `validate.py`, `pipeline.py`, `deploy.py`,
`ops.py`, `collab.py`, `report.py`, `viz.py`, `train.py`) run through
`ExecutionProvider.run_flow` and returning a `RunResult` (a Metaflow run + an
`@card`). The chosen capability/objective/budget enter as **`Parameter`s (data)** —
never a generated flow per request — so the topology stays static, cacheable, and
inspectable. `train.py` resolves the configured model-builder provider inside its
`build` step.

## How to add a provider

No core edits are needed — write one adapter and register it.

### A new execution provider (MLOps backend)

```python
from loom.config import LoomConfig
from loom.providers import ExecutionProvider
from loom.registry import register_execution
from loom.types import ExecutionResult, Task


@register_execution("my-runtime")
class MyRuntimeExecution(ExecutionProvider):
    name = "my-runtime"

    def __init__(self, config: LoomConfig) -> None:
        self.config = config
        # Lazy-import any heavy SDK *inside* methods, not at module top level,
        # so `import loom.providers` still succeeds without it installed.

    def setup(self, task: Task) -> None:
        ...  # stage ./input from task.data_dir, make empty ./working, set cwd

    def execute(self, code: str, reset_session: bool = True) -> ExecutionResult:
        ...  # run code, return a loom ExecutionResult (the 5 locked fields)

    def teardown(self) -> None:
        ...

    def runs(self, experiment_id: str) -> list[dict]:
        return []  # optional leaderboard
```

`__call__` is inherited and already aliases `execute`, so the new provider is
immediately usable as an exec-callback. Add the registering import (guarded by its
own `try/except`) to the bottom of `loom/providers/__init__.py`, then select it
with `--mlops my-runtime` / `LOOM_MLOPS_PROVIDER=my-runtime`.

### A new search provider (brain)

```python
from loom.config import LoomConfig
from loom.providers import ExecCallback, OnNode, SearchProvider
from loom.registry import register_search
from loom.types import SearchResult, Task


@register_search("my-brain")
class MyBrainSearch(SearchProvider):
    name = "my-brain"

    def __init__(self, config: LoomConfig) -> None:
        self.config = config

    def run(self, task, execute: ExecCallback, on_node=None, budget=None) -> SearchResult:
        # propose -> result = execute(code, reset_session=True) -> score
        # -> if on_node: on_node(NodeRecord(...))  for each finished node
        # -> return SearchResult(best_code, best_metric, ...)
        ...
```

Select it with `--search my-brain` / `LOOM_SEARCH_PROVIDER=my-brain`.

### A new model provider (LLM backend)

```python
from typing import MutableMapping

from loom.config import LoomConfig
from loom.providers.model import ModelProvider, ModelRoute
from loom.registry import register_model


@register_model("my-llm")
class MyLlmModelProvider(ModelProvider):
    name = "my-llm"

    def __init__(self, config: LoomConfig) -> None:
        self.config = config

    def resolve(self, role: str) -> ModelRoute:
        model = self.config.feedback_model if role == "feedback" else self.config.code_model
        return ModelRoute(model_name=model, base_url=..., key_env="MY_API_KEY",
                          judge_capable=True, kind="openai-compat")

    def prepare_env(self, env: MutableMapping[str, str]) -> None:
        env["OPENAI_BASE_URL"] = ...        # if OpenAI-compatible; never invent secrets
        if (key := env.get("MY_API_KEY")):
            env["OPENAI_API_KEY"] = key     # copy into the var AIDE's backend reads

    def preflight(self, role: str) -> list[str]:
        return [] if env_looks_ok() else ["MY_API_KEY is not set ..."]
```

Add the registering import (guarded) to the bottom of
`loom/providers/model/__init__.py`, then select it with `--code-provider my-llm` /
`--feedback-provider my-llm` (or `--model-provider my-llm` for both) /
`LOOM_CODE_PROVIDER=my-llm`. If your backend has no API key (it drives a local
CLI), use `kind="cli-bridge"` and override `install_dispatch_override` using
`loom/providers/model/_dispatch.py`.

### A new model-builder provider (training backend)

```python
from loom.config import LoomConfig
from loom.providers import ModelBuilderProvider, OBJECTIVES, BUDGETS
from loom.registry import register_model_builder
from loom.types import ArtifactRef, Capability, CapabilityManifest, Scores


@register_model_builder("my-trainer")
class MyTrainer(ModelBuilderProvider):
    name = "my-trainer"

    def __init__(self, config: LoomConfig) -> None:
        self.config = config        # read gpu_target / model_builder_base_url from it

    def manifest(self) -> CapabilityManifest:
        return CapabilityManifest(backend="my-trainer", capabilities={
            "pretrain": Capability("pretrain", "launch-and-track", supported=True,
                                   notes="...honest capability / stand-in note..."),
            "embed":    Capability("embed",    "searchable",       supported=True),
            # ...declare every capability; unsupported ones are refused up front
        })

    def pretrain(self, sequences_ref, objective, budget) -> ArtifactRef:
        assert objective in OBJECTIVES and budget in BUDGETS     # reject backend nouns at the seam
        ...  # I/O are Metaflow artifact pathspecs (loom.dataio) — never .nemo / raw S3
```

It must pass the **golden conformance suite** (`tests/test_model_builder_conformance.py`,
parametrized over every registered backend): the `tokenize→pretrain→embed→finetune→
evaluate` round-trip, valid artifact pathspecs (no file/`s3://`/`.nemo`), comparable
scores, manifest honesty, correct `mode`s, and up-front capability-gap refusal. Add
the guarded registering import to `loom/providers/model_builder/__init__.py`, then
select it with `LOOM_MODEL_BUILDER_PROVIDER=my-trainer`.

### Rules for any new provider

- **Construct from config**: `Provider(config)`; read endpoints/profiles from it,
  read secrets from the environment at the point of use, never hardcode them.
- **Lazy-import** heavy/optional dependencies inside methods so core import stays
  light, and guard the registering import in `loom/providers/__init__.py`.
- **Honor the contracts**: an `ExecutionProvider` must return the 5-field
  `ExecutionResult` (and `run_flow` a `RunResult`); a `SearchProvider` must emit
  `NodeRecord`s via `on_node` and return a `SearchResult`; a `ModelBuilderProvider`
  must return `ArtifactRef`/`Scores` (pathspecs, never files) and pass the
  conformance suite.
- **Stay domain-neutral**: never fit a provider to a specific customer, dataset,
  or vertical.
