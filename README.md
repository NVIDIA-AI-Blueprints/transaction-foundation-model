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

## The data model: your input is a Metaflow data object

Loom's input data is a **Metaflow data object — a Metaflow Artifact** — referenced
by **pathspec** (e.g. `IngestDataset/123`) and read **only through the Metaflow
Client API** (`metaflow.Run(pathspec).data.<artifact>`). The datastore behind it —
**local or object storage (S3/minio)** — is an **opaque implementation detail
Metaflow owns**: you configure it once in your **Metaflow profile/environment**,
and **Loom never talks to that storage directly** (no bucket URIs in Loom code).
Whether the bytes live on your laptop or in a bucket, Loom only ever sees
artifacts.

There is exactly **one boundary** where outside data crosses into Metaflow:
`loom ingest`. After that, the dataset is a versioned, profile-backed object you
reference by pathspec — the same object whether Loom runs locally now or in a
cluster later.

```bash
# 1. Ingest once — turns a local dir/CSV into a Metaflow data object and prints
#    its pathspec (the dataset_ref). Storage is whatever your Metaflow profile says.
loom ingest --source ./your_task --name my-dataset
#   ...
#   dataset_ref : IngestDataset/123

# 1b. See what's ingested — lists data objects via the Client API
#     (pathspec · name · nrows/schema). Read-only.
loom datasets

# 1c. Profile the data object (read-only) — schema, missingness, target balance,
#     correlations, and leakage flags, emitted as a Metaflow run + an @card.
loom eda --dataset IngestDataset/123 --target target

# 1d. Visualize the data object (read-only) — distributions, correlation heatmap,
#     target-vs-feature — emitted as @card images.
loom viz --dataset IngestDataset/123 --target target

# 2. Run against the data object by pathspec (the metaflow provider reads it via
#    the Client API; your data stays wherever your Metaflow datastore keeps it).
loom run \
  --dataset IngestDataset/123 \
  --goal  "<one sentence: what a solution should achieve>" \
  --metric "<one sentence: how it's scored, with direction>" \
  --steps 10 --mlops metaflow
```

`loom datasets`, `loom eda`, `loom features`, `loom pipeline`, `loom train`,
`loom validate`, `loom deploy`, `loom ops`, `loom collab`, `loom report`, and
`loom viz` are **lifecycle commands** that run *through Loom's MLOps interface*
(`ExecutionProvider.run_flow`) rather than the candidate-search path — every
lifecycle command's output is a **Metaflow run + an `@card`** plus a typed JSON
summary with a `VERDICT`/status line. Each verb declares an **approval tier**:

- **read-only** (never prompts): `eda`, `report`, `viz`, **`ops`**.
- **workspace-write** (light/auto, network off; reads read-only, writes only its own
  workspace): `validate`, **`features`** (writes a NEW data object).
- **workspace-write → escalates to expensive** at its train/optimize stage:
  **`pipeline`** (the costly optimize stage is held to a declared budget; gate before
  it).
- **expensive / mutate → escalates to irreversible/external** at the real GPU launch
  (always gated, never model-auto-invoked): **`train`** — builds a model through the
  third heavy backend (the `ModelBuilderProvider` seam) stated in DS-intent vocabulary
  (objective/budget/backbone/metric); the cost PLAN (hours/$/GPU-count) is surfaced at
  the gate, and the real heavy GPU launch is behind `--launch` (**OFF by default**),
  refusing cleanly with no GPU target.
- **irreversible / external** (always gated, the real external action **OFF by
  default**, never model-auto-invoked): **`deploy`** (the real apply is behind
  `--apply`) and **`collab`** (the off-box send is behind `--send`).

Lifecycle flows need the `metaflow` MLOps provider (the `local` dev path runs
candidate code only, not lifecycle flows).

After (or instead of) a search run, the **lifecycle verbs** engineer features, run
the whole pipeline, evaluate, deploy, monitor, report on, share, and visualize what
Loom did:

```bash
# Build engineered features into a NEW data object (workspace-write). Compose with
# loom-eda via --from: the EDA-flagged leakage columns are DROPPED before building.
# The new FeaturesFlow/<id> pathspec is a --dataset for every downstream verb.
loom features --dataset IngestDataset/123 --target target [--from EdaFlow/7] [--recipe minimal|full]

# Run the whole lifecycle as ONE gated run: profile -> features -> a bounded
# candidate/optimize step -> validate. Each stage asserts the prior stage's VERDICT
# (leakage blocks features; a sub-threshold validate marks the run FAIL). The
# optimize stage is the bounded EXPENSIVE step — workspace-write that escalates.
loom pipeline --dataset IngestDataset/123 --goal "<one sentence>" --target target

# Build the model the lifecycle needs through the model-builder seam — EXPENSIVE/MUTATE.
# Stated in DS-intent vocabulary (objective/budget/backbone/metric); the adapter hides
# all backend vocabulary (you never name NeMo/a GPU-count/a checkpoint). pretrain is
# launch-and-track (AIDE never tree-searches it — use loom-optimize for cheap scalars).
# The cost PLAN (hours/$/GPU-count) is surfaced at the gate; the real GPU launch is OFF
# by default behind --launch and refuses cleanly with no GPU target. The torch-free CPU
# `local` adapter actually builds a backbone + IngestDataset-shaped embeddings (a
# first-class --dataset for the downstream verbs); the default `nemo` adapter plans it.
loom train --dataset IngestDataset/123 --objective next-event --budget probe   # PLAN / CPU build, NO GPU launch
loom train --dataset IngestDataset/123 --capability embed --backbone TrainFlow/12   # embed via a frozen backbone
loom train --dataset IngestDataset/123 --objective next-event --budget full --launch  # real GPU launch (needs a gpu_target)

# Rigorously validate a baseline/solution (CV + a sealed holdout + calibration +
# fairness when --sensitive is given + leakage flags) — emits a run + @card with a
# VERDICT that gates promotion. Workspace-write (own workspace), never prompts.
loom validate --dataset IngestDataset/123 --target target [--solution EvalCandidate/42] [--sensitive region]

# Promote a validated solution — IRREVERSIBLE/EXTERNAL. Deploy ASSERTS the upstream
# validate VERDICT==PASS (no leakage, holdout clears the floor) before it will
# deploy; a sub-threshold/REVIEW/FAIL/leaky validate BLOCKS it (the cross-verb exit
# gate). --apply is OFF by default: the default produces a deployment PLAN + a staged
# registry manifest with NO external mutation. The real apply runs ONLY when the gate
# ALLOWED *and* --apply is passed. The deploy target is env/config-driven.
loom deploy --validate ValidateFlow/12            # PLAN only (staged register, no mutation)
loom deploy --validate ValidateFlow/12 --apply    # real external action (gate must ALLOW)

# Monitor runs + data objects (read-only): run health, the leaderboard, and a simple
# data-object DRIFT check vs a reference. Never prompts.
loom ops --flow ValidateFlow                                    # run health for a flow
loom ops --experiment loom-abc123                               # runs + leaderboard
loom ops --dataset IngestDataset/200 --reference IngestDataset/123   # data drift check

# Assemble a sanitized, shareable bundle of a run (report/card + lineage manifest).
# --send is OFF by default: the default BUILDS the bundle only (no data leaves the
# box). The off-box send (--send) pushes to an env/config-driven sink
# (LOOM_COLLAB_WEBHOOK / LOOM_COLLAB_OUTBOX, never a hardcoded target) and is the
# irreversible/external action — always gated.
loom collab --run ValidateFlow/12            # build the bundle only, NO send
loom collab --run ValidateFlow/12 --send     # push off-box to the env/config sink

# Assemble an experiment's runs + metrics + lineage into a model-card (read-only).
loom report --experiment loom-abc123          # or: --runs EvalCandidate/1,EvalCandidate/2

# Plot a data object or a run's results as @card images (read-only).
loom viz --dataset IngestDataset/123 --kind all     # or: --run EvalCandidate/42
```

**Composition + exit gates.** Verbs compose by handing a Metaflow object to the next
with machine-checkable gates: `loom eda` leakage flags → `loom features` (via
`--from`, dropped); a `loom features` data object → `loom pipeline`/`loom validate`
(via `--dataset`); a `loom train` backbone → `loom train --capability embed` (via
`--backbone`), and its `IngestDataset`-shaped embeddings → `loom validate`/`loom
optimize` (via `--dataset`); a `loom validate` `VERDICT==PASS` → `loom deploy` (via
`--validate` — a sub-threshold validate **BLOCKS** the deploy); a run → `loom collab`
(via `--run`). The safe-by-default verbs gate their costly action: `loom train`
plans / builds on CPU unless you pass `--launch` (and a GPU target must be set),
`loom deploy` produces a plan unless you pass `--apply` (and the gate must ALLOW), and
`loom collab` builds only unless you pass `--send`.

---

## Run it on your own data

A directory source is expected to contain `train.csv` (plus optional `test.csv` /
`sample_submission.csv`); a single `.csv` is train/test-split for you. The bundled
demo shows the directory convention:

```
your_task/
  train.csv             # labelled rows the solution learns from
  test.csv              # rows the solution must predict
  sample_submission.csv # the exact output shape expected
```

A solution reads from `./input/` and writes its predictions to
`./working/submission.csv`. There are two ways to point Loom at data:

- **`--mlops metaflow` (default):** `loom ingest` your data once into a Metaflow
  data object, then `loom run --dataset <pathspec>` (see the section above). The
  data is read through the Metaflow Client API; the datastore is opaque to Loom.
- **`--mlops local` (Metaflow-free dev path):** skip ingest and point `--data` at a
  local directory — handy for quick trials without a Metaflow profile:

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
| Setup | none | a Metaflow profile |
| Best for | quick trials, dev, CI | scale, reproducibility, audit |
| Each candidate | runs in-process | runs as a versioned Metaflow run |
| Input | a local `--data` dir | a **Metaflow data object** (`--dataset <pathspec>` from `loom ingest`) |
| Data | local | a Metaflow artifact; storage (local or S3/minio) stays in **your** perimeter and is opaque to Loom |

Switch to Metaflow once the local path works. First `loom ingest` your data into a
data object (the one external→Metaflow boundary), then Loom runs each candidate
through **one static flow** (`flows/eval_candidate.py`) — the candidate enters as
*data*, never as generated flow code — reading the dataset through the Metaflow
**Client API** against whatever endpoint your `METAFLOW_PROFILE` points at. Loom
never touches the datastore directly, so your data never leaves your environment:

```bash
export METAFLOW_PROFILE="my-profile"
loom ingest --source ./your_task --name my-dataset      # -> dataset_ref : IngestDataset/123
loom run --dataset IngestDataset/123 --goal "..." --metric "..." --steps 20 --mlops metaflow
```

---

## Drive it conversationally (Claude Code)

Prefer to talk to it? The [`skills/`](skills/) pack turns Loom into a
[Claude Code](https://claude.com/claude-code) workflow: `/loom-eda` profiles a
dataset, `/loom-optimize` pins down the metric, shows a plan, asks you to approve
cost/data, runs `loom run`, and narrates the result — then `/loom-validate` checks a
candidate against a sealed holdout, `/loom-viz` plots the data or the search, and
`/loom-report` writes up the experiment. See [`skills/README.md`](skills/README.md).

---

## Configuration

Everything is configured by flags and environment variables (and an optional
`.env` or `--config YAML`). **Secrets are only ever read from the environment —
never put a key on the command line.**

<a name="llm-providers"></a>**LLM providers** (the brain needs one). The default is native Claude:

```bash
# Claude (default — models claude-sonnet-4-5)
export ANTHROPIC_API_KEY="..."

# …or any OpenAI-compatible endpoint, including NVIDIA NIM
export OPENAI_BASE_URL="https://integrate.api.nvidia.com/v1"
export OPENAI_API_KEY="..."          # your endpoint/NIM key (read by the OpenAI-compatible client)
export LOOM_CODE_MODEL="..."         # a model your endpoint serves
export LOOM_FEEDBACK_MODEL="..."
```

OpenAI, OpenRouter, NVIDIA NIM, generic OpenAI-compatible self-hosts, and even
your own logged-in Claude/Codex CLI are all selectable per role — see
[**Model providers**](#model-providers) below.

**Common settings** (all optional; flags override env which overrides defaults):

| Env var | Flag | Default | Meaning |
|---|---|---|---|
| `LOOM_SEARCH_PROVIDER` | `--search` | `aide` | search ("brain") provider |
| `LOOM_MLOPS_PROVIDER` | `--mlops` | `metaflow` | execution ("muscle") provider |
| `LOOM_CODE_PROVIDER` | `--code-provider` | `anthropic-api` | model ("LLM backend") provider for the code role |
| `LOOM_FEEDBACK_PROVIDER` | `--feedback-provider` | `anthropic-api` | model provider for the feedback/judge role |
| — | `--model-provider` | — | shorthand setting **both** roles at once |
| `LOOM_BUDGET_STEPS` | `--steps` | `10` | number of search steps |
| `LOOM_CODE_MODEL` | — | `claude-sonnet-4-5` | model that writes solutions |
| `LOOM_FEEDBACK_MODEL` | — | `claude-sonnet-4-5` | model that reviews/scores |
| `METAFLOW_PROFILE` | — | (Metaflow default) | your Metaflow endpoint |
| `LOOM_CORPUS_PATH` | — | `corpus/nodes.jsonl` | where node records are logged |
| — | `--experiment-id` | `loom-<uuid>` | stable id to group a run |

`loom run --help` lists every flag.

---

## Model providers

The **model provider** is Loom's third port (alongside search and execution): it
decides *which model the brain talks to and how it is authenticated*. You pick one
per role — `--code-provider` (writes solutions) and `--feedback-provider` (the
judge that reviews/scores each run) — or set both at once with `--model-provider`.
Each provider configures [AIDE](https://github.com/WecoAI/aideml)'s **existing**
LLM backends via config/env; Loom never forks AIDE and **never handles your
secrets** — providers only move or pass through the env vars you already set.

> **The judge needs tools.** AIDE's feedback step (`submit_review`) always uses
> tool/function calling. If your feedback route can't do tool calls, `loom run`
> fails fast (exit 2) and tells you to pick a tool-capable feedback model.

Select a provider per role with `--code-provider` / `--feedback-provider`
(`--model-provider` sets both), or via `LOOM_CODE_PROVIDER` / `LOOM_FEEDBACK_PROVIDER`
in the environment. Flags override env; env overrides the `anthropic-api` default.

| Provider | Auth | How to select | Judge-capable |
|---|---|---|---|
| `anthropic-api` | `ANTHROPIC_API_KEY` (honors `ANTHROPIC_BASE_URL`) | **default** — or `--model-provider anthropic-api` / `LOOM_CODE_PROVIDER=anthropic-api` | yes (native Claude tool use) |
| `openai-api` | `OPENAI_API_KEY` (real OpenAI ignores `OPENAI_BASE_URL`) | `--model-provider openai-api`; set `LOOM_CODE_MODEL`/`LOOM_FEEDBACK_MODEL` to a `gpt-*`/`o<N>` model | yes (Responses-API tools) |
| `openrouter` | `OPENROUTER_API_KEY` (copied into `OPENAI_API_KEY`) | `--model-provider openrouter`; model is a `provider/model` slug, e.g. `anthropic/claude-sonnet-4.5` | per-slug — best-effort tool-support check; pick a tool-capable feedback slug |
| `nim` | `NVIDIA_API_KEY` (copied into `OPENAI_API_KEY`) | `--model-provider nim`; endpoint via `OPENAI_BASE_URL` (`nim_base_url`), default `integrate.api.nvidia.com` | configurable (default yes) |
| `openai-compat` | `OPENAI_API_KEY` (passthrough; may be a dummy local token) | `--model-provider openai-compat`; endpoint via `LOOM_MODEL_BASE_URL` (LiteLLM / vLLM / Ollama / gateway) | configurable (default yes) |
| `claude-subscription` | none — your local `claude` CLI's own login | `--model-provider claude-subscription` | yes (CLI text coerced to the judge's JSON) |
| `codex-subscription` | none — your local `codex` CLI's own login (`~/.codex/auth.json`) | `--model-provider codex-subscription` | yes (`codex exec --output-schema`) |
| `loom-proxy` | `LOOM_API_KEY` (the gateway holds the real vendor key server-side) | `--model-provider loom-proxy` (opt-in; not yet the default — the gateway isn't hosted) | yes (Anthropic passthrough — native tool use) |

```bash
# Native OpenAI for both roles
loom run --data ./task --goal "..." --metric "..." \
  --model-provider openai-api
export OPENAI_API_KEY="sk-..."; export LOOM_CODE_MODEL="gpt-4o"; export LOOM_FEEDBACK_MODEL="gpt-4o"

# OpenRouter (pick a tool-capable slug for the FEEDBACK role)
export OPENROUTER_API_KEY="sk-or-..."
export LOOM_CODE_MODEL="anthropic/claude-sonnet-4.5"
export LOOM_FEEDBACK_MODEL="anthropic/claude-sonnet-4.5"   # must support tools
loom run --data ./task --goal "..." --metric "..." --model-provider openrouter

# Mix roles: cheap local model writes code, Claude judges
export OPENAI_BASE_URL="http://localhost:11434/v1"   # Ollama, say
loom run --data ./task --goal "..." --metric "..." \
  --code-provider openai-compat --feedback-provider anthropic-api
```

### OpenRouter wiring

OpenRouter speaks an OpenAI-compatible API, but AIDE's *dedicated* OpenRouter
backend can't do tool calling — and the judge always needs it. So the `openrouter`
provider points AIDE's **OpenAI-compatible** path at OpenRouter instead: it sets
`OPENAI_BASE_URL=https://openrouter.ai/api/v1` and copies your `OPENROUTER_API_KEY`
into `OPENAI_API_KEY` (the var that client reads), then routes on a non-reserved
slug like `anthropic/claude-sonnet-4.5`. For the feedback role it best-effort
checks OpenRouter's
[`?supported_parameters=tools`](https://openrouter.ai/models?supported_parameters=tools)
listing and warns (or fails fast) if your feedback slug can't run the judge. If
the check can't run (offline), it assumes capable and warns.

### Use your Claude / Codex subscription

`claude-subscription` and `codex-subscription` are **CLI bridges**: instead of an
API key, Loom drives the `claude` (Claude Code) or `codex` CLI **you have already
installed and logged in on your own machine**, by installing a small dispatch
shim into AIDE's backend at run setup.

**Honest caveats — read before relying on these:**

- **It drives your own local CLI.** Loom shells out to the binary on your `PATH`
  (`claude -p …` / `codex exec …`) and reads the answer back. It **never sees,
  stores, or transmits your subscription credentials** — `claude` owns its login
  (`CLAUDE_CODE_OAUTH_TOKEN` or `~/.claude`), and `codex` reuses `~/.codex/auth.json`
  (a **full-account credential** for your ChatGPT/Codex account — treat that file
  as sensitive). Login is something **you** start, not Loom.
- **Metering is separate (as of 2026-06-15).** The `claude` CLI draws on a
  **separate Agent SDK / subscription credit pool**, metered independently from a
  raw `ANTHROPIC_API_KEY`. Codex subscription use is subject to a **rolling-window
  usage cap**. A heavy AIDE run is many LLM calls (every draft/debug/improve step,
  plus a judge call per executed node) and can exhaust either quickly — for
  sustained or large runs, an **API key** (`anthropic-api` / `openai-api`) is often
  the better fit. These bridges are best for light, interactive use of a
  subscription you already have.
- **You're responsible for your CLI's terms.** Running your own authenticated CLI
  to do your own work is the intended use of that CLI; usage remains subject to
  that tool's own terms.

```bash
# Prereqs: the CLI is installed, on PATH, and logged in.
claude /login            # or: claude setup-token
loom run --data ./task --goal "..." --metric "..." --model-provider claude-subscription

# Codex (human-started login first)
loom run --data ./task --goal "..." --metric "..." --model-provider codex-subscription
```

`loom run` pre-flights both: a missing binary or login signal yields an actionable
hint before any work starts.

### The Loom gateway (`loom-proxy`) — and the privacy tiers

`loom-proxy` routes your LLM calls through **Loom's own gateway** instead of
calling the vendor directly. You authenticate with a Loom-issued `LOOM_API_KEY`;
the gateway holds the real vendor key **server-side** (you never see it), injects
Loom's system prompt, forwards to the real Anthropic Messages API (v0 is a 1:1
**Anthropic passthrough**, so native tool use and the judge work with no
translation), and **logs every call centrally** — request system + messages,
response text + token usage, model, latency, and optional tenant/owner tags — to
one JSONL corpus. That central capture is Loom's **data flywheel**: the fuel for
distilling small, Loom-owned models over time.

> **Status: opt-in, not the default — the gateway is not hosted yet.** Select it
> explicitly with `--model-provider loom-proxy`. It becomes the **default once
> hosted** — it adds no egress beyond using Claude (see below), it just lets Loom log the prompt traces that fuel the moat.

**Run the gateway yourself (the server side):**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # the REAL vendor key — stays on the server
export LOOM_API_KEY="loom-..."          # the key callers must present (allowlist)
loom proxy serve --host 127.0.0.1 --port 8088
#   POST http://127.0.0.1:8088/v1/messages   (Anthropic Messages passthrough)
#   GET  http://127.0.0.1:8088/healthz

# point a run at it (in another shell, with only the LOOM key):
export LOOM_API_KEY="loom-..."          # callers never need the vendor key
loom run --data ./task --goal "..." --metric "..." --model-provider loom-proxy
```

`LOOM_API_BASE` (default `http://127.0.0.1:8088`) is where the `loom-proxy`
provider points AIDE's Anthropic backend; `LOOM_PROXY_LOG_PATH` (default
`learnings/proxy_calls.jsonl`) is the central call log. Set `LOOM_API_KEYS`
(comma-separated) on the server for a multi-key allowlist.

**What actually leaves your machine — the honest version:**

- **Your bulk data never goes to the LLM, in *either* mode.** Datasets/transactions live as **Metaflow data objects in your own datastore/perimeter**; candidate code processes them there. The LLM only ever sees small *derived* context — a schema, a data-preview, code, metrics — never the data itself. That is the real "stays local" guarantee, and it holds for **every** provider. Keep it true with **prompt hygiene**: never stuff raw rows into a prompt (AIDE already injects a *small* data-preview, not the data).
- **LLM prompts go to a third-party LLM either way.** Using Claude at all sends those small prompts to Anthropic. `loom-proxy` sits in that *same* path — **no incremental egress** ("Claude or Us — no difference").

So picking a provider is **not** a bulk-data-privacy choice — it's a *data-collection* choice: **does Loom also log your prompt traces** (and thereby fuel the moat)?

| Provider | Prompts go to | Loom logs them? |
|---|---|---|
| **`loom-proxy`** (default *once hosted*) | Anthropic, via Loom's gateway | **yes** — every call to the moat corpus, tagged by tenant/owner |
| **BYO-key** (`anthropic-api`/`openai-api`/`openrouter`/`nim`/`openai-compat`/`claude-subscription`/`codex-subscription`) | straight to your vendor/endpoint | **no** — Loom's gateway sees nothing |

A tenant that doesn't want Loom collecting its prompt traces uses a BYO-key provider; its **bulk data is protected the same way in both modes** (it's in Metaflow, not in the prompt). The moat trains **only on `owned_by: general` data**; tenant-tagged rows (`x-loom-owned-by: <tenant>`) are isolated from the cross-tenant set (the same IP boundary [corpus](loom/corpus.py) enforces). No keys are ever logged — the gateway records request/response content + metadata, never key material.

---

## How it's built

Loom-core defines provider **interfaces** (ports); concrete **adapters** plug in
and are picked by config — add a new brain or MLOps backend by writing one class
and registering it, no core changes.

| Seam | Port | Built-in adapters |
|---|---|---|
| Search ("brain") | `SearchProvider` | `aide` |
| Execution ("muscle") | `ExecutionProvider` | `metaflow`, `local` |
| Model ("LLM backend") | `ModelProvider` | `anthropic-api`, `openai-api`, `openrouter`, `nim`, `openai-compat`, `claude-subscription`, `codex-subscription`, `loom-proxy` |

```
loom/
  types.py        ExecutionResult · RunResult · Task (data_dir + dataset_ref) · SearchResult · NodeRecord  (the locked contract)
  config.py       LoomConfig (env / .env / YAML)
  registry.py     name → provider class
  dataio.py       materialize_dataset / dataset_schema — the Client-API data door (never touches S3)
  providers/      aide_search.py · metaflow_exec.py · local_exec.py · _interpreter.py
  controller.py   run_loom(task, config) — wires it together
  corpus.py       append-only JSONL of every node (multi-tenant IP boundary)
  proxy/          the Loom gateway — Anthropic passthrough that logs every call (the moat capture)
  cli.py          the `loom` command (`loom ingest`, `loom datasets`, `loom eda`, `loom features`, `loom pipeline`, `loom validate`, `loom deploy`, `loom ops`, `loom collab`, `loom report`, `loom viz`, `loom run`, `loom proxy serve`)
flows/ingest_dataset.py   the external→Metaflow boundary (dir/CSV → data object)
flows/eda.py              read-only EDA/profiling flow (lifecycle command → run + @card)
flows/features.py         feature-engineering flow — WRITES a NEW data object (FeaturesFlow/<id>) the downstream verbs consume (→ run + @card)
flows/pipeline.py         end-to-end lifecycle flow — profile→features→optimize→validate in one gated run, per-stage VERDICT gates (→ run + @card)
flows/validate.py         rigorous validation flow — CV + sealed holdout + calibration + fairness + leakage (→ run + @card)
flows/deploy.py           gated deployment flow — asserts validate VERDICT==PASS, produces a PLAN/manifest; real apply behind --apply, OFF by default (→ run + @card)
flows/ops.py              read-only ops/monitoring flow — run health + leaderboard + data-object drift (→ run + @card)
flows/collab.py           collaboration flow — sanitized shareable bundle + lineage manifest; off-box send behind --send, OFF by default (→ run + @card)
flows/report.py           read-only experiment-report flow — runs + metrics + lineage → model-card (→ run + @card)
flows/viz.py              read-only visualization flow — standard plots → @card images
flows/eval_candidate.py   the one static evaluation flow (candidate = data)
skills/           Claude Code skill-pack (loom-connect, loom-eda, loom-features, loom-pipeline, loom-validate, loom-deploy, loom-ops, loom-collab, loom-report, loom-viz, loom-optimize)
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
