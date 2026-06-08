# Loom

**Loom is an agentic data-science engine.** Hand it a dataset, a goal, and an
evaluation metric — Loom writes solution code, runs each candidate for real,
scores it against your metric, and hands back the best solution plus a
leaderboard of everything it tried.

> The **metric is the spec.** Anything you can state as *"here's the data,
> here's the goal, here's how a solution is scored"* is a valid Loom run.
> Loom is **domain-neutral** — not tied to any task type, data shape, or vertical.

Under the hood Loom is **ports & adapters** (like Kubernetes' pluggable
runtimes): a **search** provider is the brain (default: [AIDE](https://github.com/WecoAI/aideml)'s
tree search) and an **execution** provider is the muscle (default: a real
[Metaflow](https://metaflow.org) flow; a Metaflow-free `local` path for quick trials).

---

## Quickstart

You need **Python 3.10–3.12** (any of conda, Homebrew, pyenv, or the
[python.org](https://www.python.org/downloads/) installer works) and an **LLM
API key** — Loom's brain writes code with an LLM. The default model is Claude, so
the quickest path is an `ANTHROPIC_API_KEY`; see [other providers](#llm-providers)
for OpenAI / NVIDIA NIM. Forget the key and `loom run` tells you exactly which one
to set.

```bash
# 1. Get the code
git clone git@github.com:ZKAI-Network/Loom.git
cd Loom

# 2. Install into a clean virtual environment
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # first install pulls AIDE's ML stack — give it a few minutes

# 3. Point Loom at an LLM (Claude is the default)
export ANTHROPIC_API_KEY="sk-ant-..."

# 4. Make the bundled demo dataset (generic synthetic tabular task — no download)
python tasks/generic_demo/prepare_data.py

# 5. Run Loom — the `local` provider needs no Metaflow, no GPU, no cloud
loom run \
  --data tasks/generic_demo/input \
  --goal  "Train on train.csv and predict the target for each row in test.csv; write ./working/submission.csv with columns id,target (probability of class 1)." \
  --metric "Maximize ROC-AUC between the predicted probability and the true target on a held-out split." \
  --steps 4 \
  --mlops local
```

You'll see Loom propose candidate solutions, run each one, and finish with
something like:

```
Loom run complete.
  best metric : 0.93xx
  nodes       : 4
  journal     : .../logs/<run>/journal.json
  tree        : .../logs/<run>/tree_plot.html
  best code   : 42 line(s) found

Leaderboard (top 4):
   1. metric=0.93xx  node=...  [improve]
   2. metric=0.91xx  node=...  [draft]
   ...
```

Open the **tree** HTML to see the search, or the **journal** for every candidate.
Start with a small `--steps` (each step is an LLM call + a real code execution);
raise it once you've seen a run work.

---

## What just happened

```
   your task ──▶  SEARCH provider (aide)  ──proposes code──▶  EXECUTION provider (local)
   data+goal       draft / debug / improve                    runs it, returns result
   +metric              ▲                                          │
                        └─────────── score (your metric) ◀─────────┘
                                  keep the best · record every node
```

The search brain proposed a solution, the execution muscle ran it in an isolated
workspace (your data in `./input`, the solution writes `./working/submission.csv`),
the result was scored against your metric, and the loop kept improving the best
one. Every candidate is recorded; the winner is returned.

---

## Run it on your own data

Loom expects a **data directory** it can stage into a workspace. The bundled demo
shows the convention:

```
your_task/
  train.csv             # labelled rows the solution learns from
  test.csv              # rows the solution must predict
  sample_submission.csv # the exact output shape expected
```

A solution reads from `./input/` and writes its predictions to
`./working/submission.csv`. Then point Loom at it:

```bash
loom run \
  --data ./your_task \
  --goal  "<one sentence: what a solution should achieve>" \
  --metric "<one sentence: how it's scored, with direction — e.g. 'Maximize ROC-AUC' / 'Minimize RMSE'>" \
  --steps 10 --mlops local
```

The metric sentence is the most important input — state the optimization
**direction** unambiguously, because Loom optimizes exactly what you ask for.

---

## Local vs. Metaflow

| | `--mlops local` | `--mlops metaflow` (default) |
|---|---|---|
| Setup | none | a Metaflow endpoint |
| Best for | quick trials, dev, CI | scale, reproducibility, audit |
| Each candidate | runs in-process | runs as a versioned Metaflow run |
| Data | local | stays in **your** perimeter (BYO endpoint) |

Switch to Metaflow once the local path works. Loom runs each candidate through
**one static flow** (`flows/eval_candidate.py`) — the candidate enters as *data*,
never as generated flow code — against whatever Metaflow endpoint your
`METAFLOW_PROFILE` points at, so your data never leaves your environment:

```bash
export METAFLOW_PROFILE="my-profile"
loom run --data ./your_task --goal "..." --metric "..." --steps 20 --mlops metaflow
```

---

## Drive it conversationally (Claude Code)

Prefer to talk to it? The [`skills/`](skills/) pack turns Loom into a
[Claude Code](https://claude.com/claude-code) workflow: `/loom-eda` profiles a
dataset, then `/loom-optimize` pins down the metric, shows a plan, asks you to
approve cost/data, runs `loom run`, and narrates the result. See
[`skills/README.md`](skills/README.md).

---

## Configuration

Everything is configured by flags and environment variables (and an optional
`.env` or `--config YAML`). **Secrets are only ever read from the environment —
never put a key on the command line.**

<a name="llm-providers"></a>**LLM providers** (the brain needs one):

```bash
# Claude (default — models claude-sonnet-4-5)
export ANTHROPIC_API_KEY="..."

# …or any OpenAI-compatible endpoint, including NVIDIA NIM
export OPENAI_BASE_URL="https://integrate.api.nvidia.com/v1"
export OPENAI_API_KEY="..."          # your endpoint/NIM key (read by the OpenAI-compatible client)
export LOOM_CODE_MODEL="..."         # a model your endpoint serves
export LOOM_FEEDBACK_MODEL="..."
```

**Common settings** (all optional; flags override env which overrides defaults):

| Env var | Flag | Default | Meaning |
|---|---|---|---|
| `LOOM_SEARCH_PROVIDER` | `--search` | `aide` | search ("brain") provider |
| `LOOM_MLOPS_PROVIDER` | `--mlops` | `metaflow` | execution ("muscle") provider |
| `LOOM_BUDGET_STEPS` | `--steps` | `10` | number of search steps |
| `LOOM_CODE_MODEL` | — | `claude-sonnet-4-5` | model that writes solutions |
| `LOOM_FEEDBACK_MODEL` | — | `claude-sonnet-4-5` | model that reviews/scores |
| `METAFLOW_PROFILE` | — | (Metaflow default) | your Metaflow endpoint |
| `LOOM_CORPUS_PATH` | — | `corpus/nodes.jsonl` | where node records are logged |
| — | `--experiment-id` | `loom-<uuid>` | stable id to group a run |

`loom run --help` lists every flag.

---

## How it's built

Loom-core defines provider **interfaces** (ports); concrete **adapters** plug in
and are picked by config — add a new brain or MLOps backend by writing one class
and registering it, no core changes.

| Seam | Port | Built-in adapters |
|---|---|---|
| Search ("brain") | `SearchProvider` | `aide` |
| Execution ("muscle") | `ExecutionProvider` | `metaflow`, `local` |

```
loom/
  types.py        ExecutionResult · Task · SearchResult · NodeRecord  (the locked contract)
  config.py       LoomConfig (env / .env / YAML)
  registry.py     name → provider class
  providers/      aide_search.py · metaflow_exec.py · local_exec.py · _interpreter.py
  controller.py   run_loom(task, config) — wires it together
  corpus.py       append-only JSONL of every node (multi-tenant IP boundary)
  cli.py          the `loom` command
flows/eval_candidate.py   the one static Metaflow flow (candidate = data)
skills/           Claude Code skill-pack (loom-optimize, loom-eda)
tasks/generic_demo/       the bundled smoke-test task
```

Full design and "how to add a provider": [`docs/architecture.md`](docs/architecture.md).
Repository invariants: [`CLAUDE.md`](CLAUDE.md).

---

## Tests

```bash
pip install -e ".[dev]"   # or: pip install pytest
pytest
```

Pure-Python tests (provider registry, corpus, the `local` execution path) run
without an LLM key or Metaflow; tests that need AIDE or Metaflow skip cleanly if
those aren't installed.

---

## Status

**v0.1.** The `local` provider is the easy on-ramp and needs only an LLM key. The
`metaflow` provider needs a Metaflow endpoint. AIDE is pinned by commit for
reproducibility. Loom is intentionally general-purpose — point it at any dataset
with a measurable goal.
