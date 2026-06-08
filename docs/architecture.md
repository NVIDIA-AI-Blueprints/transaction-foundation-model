# Loom architecture

Loom is a **general-purpose, domain-neutral automated ML engine** built as
**ports and adapters** ("providers"), the same way Kubernetes treats container
runtimes as pluggable. `loom-core` defines the provider *interfaces* (the ports);
concrete *adapters* implement them and are chosen purely by configuration. Adding
a new brain or a new MLOps backend is a new adapter class plus a one-line
registration — never a core change.

This document covers the provider model, the three seams, the request
lifecycle, and how to add a provider.

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

## The three ports (seams)

The search and execution ports are abstract base classes in
`loom/providers/__init__.py`; the model port lives in
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
- **`Task(data_dir, goal, eval, experiment_id, tenant="default")`** — one
  experiment: where the data is, what to achieve, how it's scored, the grouping
  id, and the tenant.
- **`SearchResult(best_code, best_metric, journal_path, tree_path, node_count)`**
  — the outcome of a search.
- **`NodeRecord(...)`** — one finished search node, persisted by the corpus
  (carries `code`, `term_out`, `metric`, model/token routing metadata, and the
  `tenant` / `owned_by` IP tags).

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

## Request lifecycle

1. The CLI (`loom run …`) builds a `Task` and a `LoomConfig` and calls
   `controller.run_loom(task, config)`.
2. The controller resolves the execution and search provider **classes** from the
   registry by their configured names, instantiates each from the config, and
   creates a `Corpus`.
3. `execution.setup(task)` stages the workspace (`./input` from `task.data_dir`,
   empty `./working`, cwd set).
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
| `metaflow` | execution | `loom/providers/metaflow_exec.py` + `flows/eval_candidate.py` | Runs each candidate through **one static** `EvalCandidate(FlowSpec)` via `metaflow.Runner`; the candidate enters as **data** (`IncludeFile`), never as a generated flow. BYO Metaflow endpoint. |
| `local` | execution | `loom/providers/local_exec.py` | Metaflow-free dev path; runs candidates in-process via the vendored interpreter. |
| `anthropic-api` | model | `loom/providers/model/anthropic_api.py` | **Default.** Native Claude via AIDE's Anthropic backend; reads `ANTHROPIC_API_KEY`. Judge-capable. |
| `openai-api` | model | `loom/providers/model/openai_api.py` | Native OpenAI (`gpt-*`/`o<N>`); reads `OPENAI_API_KEY`. Leaves `OPENAI_BASE_URL` unset (real OpenAI ignores it). |
| `openrouter` | model | `loom/providers/model/openrouter.py` | OpenRouter via the OpenAI-compatible path; copies `OPENROUTER_API_KEY` → `OPENAI_API_KEY`. Per-slug judge check. |
| `nim` | model | `loom/providers/model/nim.py` | NVIDIA NIM via the OpenAI-compatible path; copies `NVIDIA_API_KEY` → `OPENAI_API_KEY`. |
| `openai-compat` | model | `loom/providers/model/openai_compat.py` | Generic self-host (LiteLLM/vLLM/Ollama); `OPENAI_BASE_URL` from `model_base_url`, `OPENAI_API_KEY` passthrough. |
| `claude-subscription` | model | `loom/providers/model/claude_subscription.py` | `cli-bridge`: drives the user's local `claude -p`. No key; checks CLI + login. |
| `codex-subscription` | model | `loom/providers/model/codex_subscription.py` | `cli-bridge`: drives the user's local `codex exec` (judge via `--output-schema`). No key; checks CLI + `~/.codex/auth.json`. |

The vendored interpreter (`loom/providers/_interpreter.py`) is a dependency-light
port of AIDE's interpreter that produces a Loom `ExecutionResult`. It is shared by
the `local` path and the Metaflow `evaluate` step, so neither hard-depends on AIDE
internals.

### Why one static Metaflow flow

`EvalCandidate(FlowSpec)` (`start → evaluate → validate → end`) is defined once.
Each candidate solution enters it as **data** (an `IncludeFile` candidate plus
`Parameter`s for goal/eval/timeout/seed/input_ref), never as a freshly generated
flow per candidate. This keeps the flow definition stable, cacheable, and
inspectable across every evaluation. Use standard Metaflow APIs only.

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

### Rules for any new provider

- **Construct from config**: `Provider(config)`; read endpoints/profiles from it,
  read secrets from the environment at the point of use, never hardcode them.
- **Lazy-import** heavy/optional dependencies inside methods so core import stays
  light, and guard the registering import in `loom/providers/__init__.py`.
- **Honor the contracts**: an `ExecutionProvider` must return the 5-field
  `ExecutionResult`; a `SearchProvider` must emit `NodeRecord`s via `on_node` and
  return a `SearchResult`.
- **Stay domain-neutral**: never fit a provider to a specific customer, dataset,
  or vertical.
