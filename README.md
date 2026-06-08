# Loom

Loom is a **general-purpose, domain-neutral automated ML engine**. You hand it a
dataset, a goal, and an evaluation metric; Loom searches for solution code that
maximizes that metric, runs each candidate in a real execution environment, and
returns the best solution together with a leaderboard of everything it tried.

Loom is **not** tied to any task type, data shape, or vertical. The metric is the
spec: anything you can describe as "here is the data, here is the goal, here is
how a solution is scored" is a valid Loom run.

## Architecture: ports and adapters

Loom is built as **ports and adapters** ("providers"), the same way Kubernetes
treats container runtimes as pluggable. `loom-core` defines provider
*interfaces* (the ports); concrete *adapters* implement them and are selected
purely by configuration. There are two seams in v0.1:

| Seam | Port | Role | Built-in adapters |
| --- | --- | --- | --- |
| **Search** ("the brain") | `SearchProvider` | Proposes, scores, and records candidate solutions | `aide` (default) |
| **Execution** ("the muscle") | `ExecutionProvider` | Runs candidate code and returns an `ExecutionResult` | `metaflow` (default), `local` (dev) |

The two are wired together by a single seam: an `ExecutionProvider` is itself
*callable* with the search-side exec-callback signature
`(code, reset_session) -> ExecutionResult`, so any execution provider can be
handed straight to any search provider. The search provider proposes code, calls
the execution provider to run it, scores the result, and emits one `NodeRecord`
per finished node.

Providers are resolved by name through a small registry
(`loom/registry.py`): config picks `loom.search.provider` (default `aide`) and
`loom.mlops.provider` (default `metaflow`), and the controller looks those names
up. Adding a new brain or a new MLOps backend means writing one adapter class
and registering it — no core changes.

```
              ┌───────────────── controller ─────────────────┐
              │                                               │
   Task ─────▶│  search = get_search(cfg.search_provider)     │
   Config ───▶│  exec   = get_execution(cfg.mlops_provider)   │
              │                                               │
              │  search.run(task, execute=exec,               │
              │             on_node=corpus.record, budget)    │
              │        │                ▲                     │
              │        ▼                │                     │
              │   propose code ──▶ exec(code) ──▶ score ──▶ NodeRecord
              └───────────────────────────────────────────────┘
                          │                         │
                   SearchProvider            ExecutionProvider
                     (aide)              (metaflow | local)
```

## Components

- **`loom/types.py`** — locked core types: `ExecutionResult` (field-identical to
  AIDE's), `Task`, `SearchResult`, `NodeRecord`.
- **`loom/config.py`** — `LoomConfig`, loaded from defaults < optional YAML <
  environment / `.env` < overrides. Secrets are never stored on it.
- **`loom/registry.py`** — name → provider-class registry.
- **`loom/providers/`** — the two ports plus the built-in adapters:
  - `aide_search.py` — drives AIDE's agent/journal loop as the search brain.
  - `metaflow_exec.py` — runs each candidate through one static Metaflow flow
    (`flows/eval_candidate.py`, a top-level package) where the candidate enters
    as *data*, never as generated flow code.
  - `local_exec.py` — a Metaflow-free local execution path for fast iteration.
  - `_interpreter.py` — a dependency-light interpreter shared by the local path
    and the Metaflow flow.
- **`loom/corpus.py`** — appends `NodeRecord`s to a JSONL corpus, with a
  multi-tenant IP boundary (`owned_by`).
- **`loom/controller.py`** — `run_loom(task, config)` wires it all together.
- **`loom/cli.py`** — the `loom` command-line entrypoint.
- **`skills/`** — Claude Code skill-pack (`loom-optimize`, `loom-eda`) that
  drives Loom conversationally.

## Quickstart

### Install

```bash
pip install -e .
```

Configuration and secrets come from the environment (and an optional `.env`).
Nothing sensitive is ever hardcoded:

```bash
# Model routing (NVIDIA NIM, OpenAI-compatible) — or use Claude.
export OPENAI_BASE_URL="https://integrate.api.nvidia.com/v1"
export NVIDIA_API_KEY="..."          # consumed by the adapter at call time
# export ANTHROPIC_API_KEY="..."     # if routing to Claude instead

# Metaflow (only needed for the metaflow provider) — bring your own perimeter.
export METAFLOW_PROFILE="my-profile"
```

### Run locally (fast dev path, no Metaflow)

`local` execution runs candidates in-process via the vendored interpreter:

```bash
loom run \
  --data ./data/my_task \
  --goal "Predict the target column from the provided features." \
  --metric "Maximize ROC AUC on a held-out split." \
  --steps 10 \
  --mlops local
```

### Run on Metaflow (default execution)

`metaflow` execution runs each candidate through the static `EvalCandidate` flow
on your Metaflow endpoint. Data stays inside your perimeter:

```bash
loom run \
  --data ./data/my_task \
  --goal "Predict the target column from the provided features." \
  --metric "Maximize ROC AUC on a held-out split." \
  --steps 20 \
  --mlops metaflow
```

The CLI prints the best metric, the artifact paths (journal, tree, best code),
and a short leaderboard of the runs it explored.

### Drive it from Claude Code

The `skills/` pack lets you run Loom conversationally — `loom-eda` to profile a
dataset, then `loom-optimize` to plan a run, confirm cost/data at an approval
gate, launch the `loom run` CLI, and narrate the result. See
[`skills/README.md`](skills/README.md).

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — the ports-and-adapters model,
  the two seams in detail, and how to add a provider.
- [`CLAUDE.md`](CLAUDE.md) — repository invariants.
