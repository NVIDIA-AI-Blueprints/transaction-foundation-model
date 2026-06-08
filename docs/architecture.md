# Loom architecture

Loom is a **general-purpose, domain-neutral automated ML engine** built as
**ports and adapters** ("providers"), the same way Kubernetes treats container
runtimes as pluggable. `loom-core` defines the provider *interfaces* (the ports);
concrete *adapters* implement them and are chosen purely by configuration. Adding
a new brain or a new MLOps backend is a new adapter class plus a one-line
registration — never a core change.

This document covers the provider model, the two seams in v0.1, the request
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

## The two ports (seams)

Both ports are abstract base classes in `loom/providers/__init__.py`. Every
provider is constructed uniformly as `Provider(config)` so the controller can
wire them symmetrically.

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
- `metaflow_profile` (env `METAFLOW_PROFILE`) — lets a tenant point Loom at their
  own Metaflow endpoint (BYO perimeter).
- `code_model` / `feedback_model` / `nim_base_url` (env `OPENAI_BASE_URL`) — model
  routing. The matching API key (`NVIDIA_API_KEY`, `ANTHROPIC_API_KEY`) is read
  from the environment **at the point of use** and never stored on the config.
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
   finished node.
5. `execution.teardown()` is called in a `finally`, then the `SearchResult` is
   returned. The CLI prints the best metric, artifact paths, and a leaderboard.

## Built-in adapters (v0.1)

| Name | Port | Module | Notes |
| --- | --- | --- | --- |
| `aide` | search | `loom/providers/aide_search.py` | Drives AIDE's agent/journal loop (pinned by SHA `40dcf28`). Does **not** edit AIDE; converts results via field parity. |
| `metaflow` | execution | `loom/providers/metaflow_exec.py` + `flows/eval_candidate.py` | Runs each candidate through **one static** `EvalCandidate(FlowSpec)` via `metaflow.Runner`; the candidate enters as **data** (`IncludeFile`), never as a generated flow. BYO Metaflow endpoint. |
| `local` | execution | `loom/providers/local_exec.py` | Metaflow-free dev path; runs candidates in-process via the vendored interpreter. |

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
