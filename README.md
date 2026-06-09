# Loom

**Loom is an agentic CLI for data science.** It's a catalog of
`/loom-*` verbs — `connect · eda · features · pipeline · train · optimize ·
validate · viz · report · deploy · ops · collab` — spanning the **whole**
data-science lifecycle, each an agentic skill (a [Claude Code](https://claude.com/claude-code)
skill today; `loom <verb>` in the binary later). ML *modeling* is the ~3%
"brain"; the product is the other **97%** — data access, EDA, features,
pipelines, training, validation, viz, reporting, deployment, ops, collaboration.

Loom is **ports & adapters** (like Kubernetes' pluggable runtimes): three
swappable interfaces, sensible defaults, and every verb runs *through an
interface* — never a concrete backend — so any layer is drop-in replaceable.

| Interface | Role | Default lego |
|---|---|---|
| **Search** ("brain") | the `/loom-optimize` ML-iteration engine (one slice) | [AIDE](https://github.com/WecoAI/aideml) tree-search |
| **MLOps** ("muscle") | data objects · flows · runs · `@card` · deploy/ops | [Metaflow](https://metaflow.org) (a Metaflow-free `local` path for quick trials) |
| **Model-builder** ("training") | `/loom-train` — pretrain · finetune · embed · serve | **NeMo** (a torch-free CPU `local` stand-in for dev) |

> The **metric is the spec.** Anything you can state as *"here's the data,
> here's the goal, here's how a solution is scored"* is a valid run. Loom is
> **domain-neutral** — not tied to any task type, data shape, or vertical.

---

## The lifecycle at a glance

Every verb is a Claude Code skill now (`/loom-eda`) and the same `loom eda` in the
binary later. Each runs *through* Loom's MLOps interface, produces a **Metaflow
run + an `@card`** plus a typed JSON summary with a `VERDICT`/status line, and
declares an **approval tier** enforced beneath the model.

| Verb | What it does | Tier |
|---|---|---|
| `connect` / `ingest` / `datasets` | bring data in as a Metaflow data object; list what's ingested | read-only / boundary |
| `eda` | profile a data object — schema, missingness, target balance, leakage flags | read-only |
| `viz` | plots from a data object or a run — distributions, correlation, leaderboard | read-only |
| `features` | engineer features into a **new** data object (leakage-aware) | workspace-write |
| `optimize` (`loom run`) | the AIDE search — propose/run/score candidate solutions to your metric | workspace-write |
| `train` | pretrain a backbone / build embeddings via the model-builder (NeMo / CPU) | **expensive · always-gate** |
| `validate` | rigorous eval — CV + sealed holdout + calibration + fairness + leakage | workspace-write |
| `pipeline` | the whole lifecycle as one gated run — profile → features → optimize → validate | workspace-write → expensive |
| `report` | an experiment's runs + metrics + lineage → a model-card | read-only |
| `deploy` | promote a **validated** solution — gated on `validate==PASS` | **irreversible/external** |
| `ops` | run health + leaderboard + data-object drift | read-only |
| `collab` | a sanitized, shareable bundle (card + lineage) | workspace-write (send is external) |

**Approval matrix** (enforced by the client/hook layer, not by prompt text):

| Tier | Examples | Gate |
|---|---|---|
| read-only | `eda`, `viz`, `report`, `ops` | never prompts |
| workspace-write | `features`, `validate`, `optimize`, `collab` (build) | light/auto, network off by default |
| expensive / mutate | `train` (GPU), `pipeline`'s optimize stage | **always gate** — cost/rows shown |
| irreversible / external | `deploy --apply`, `collab --send`, the real `train --launch` | **always gate, never model-auto-invoked** |

**Composition = artifact hand-off + machine-checkable exit gates:** `eda` leakage
flags block `features`; a `features` data object feeds `pipeline`/`validate`; a
`validate` `VERDICT==PASS` is required by `deploy` (a sub-threshold validate
**BLOCKS** it). The two irreversible verbs are **safe by default** — `deploy`
plans unless `--apply` (and the gate must ALLOW); `collab` builds unless `--send`.

---

## Quickstart — the 60-second smoke test (no infra)

You need **Python 3.10–3.12** and an **LLM API key** — the brain writes code with
an LLM. The default model is Claude, so the quickest path is an `ANTHROPIC_API_KEY`
(see [Model providers](#model-providers) for OpenAI / NVIDIA NIM / your Claude or
Codex subscription). Forget the key and the CLI tells you exactly which to set.

The `local` MLOps path needs **no Metaflow, no GPU, no cloud** — perfect for a
first "does it work":

```bash
# 1. Get the code
git clone git@github.com:ZKAI-Network/Loom.git && cd Loom

# 2. Install into a clean virtualenv (first install pulls AIDE's ML stack — a few minutes)
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 3. Point Loom at an LLM (Claude is the default)
export ANTHROPIC_API_KEY="sk-ant-..."

# 4. Make the bundled demo dataset (generic synthetic tabular task — no download)
python tasks/generic_demo/prepare_data.py

# 5. Run the AIDE search — the engine behind /loom-optimize
loom run \
  --data tasks/generic_demo/input \
  --goal  "Train on train.csv and predict the target for each row in test.csv; write ./working/submission.csv with columns id,target (probability of class 1)." \
  --metric "Maximize ROC-AUC between the predicted probability and the true target on a held-out split." \
  --steps 4 \
  --mlops local
```

You'll see Loom propose candidates, run each for real, and finish with a
leaderboard:

```
Loom run complete.
  best metric : 0.93xx
  nodes       : 4
  journal     : .../logs/<run>/journal.json
  tree        : .../logs/<run>/tree_plot.html

Leaderboard (top 4):
   1. metric=0.93xx  node=...  [improve]
   2. metric=0.91xx  node=...  [draft]
```

Open the **tree** HTML to see the search, or the **journal** for every candidate.
Start small (`--steps` = LLM calls + real executions); raise it once a run works.

---

## Two ways to run

| | `--mlops local` | `--mlops metaflow` (default) |
|---|---|---|
| Setup | none (just an LLM key) | a Metaflow profile + datastore ([below](#set-up-the-metaflow-datastore)) |
| Covers | the `loom run` search only | **the whole lifecycle** (`eda`, `features`, `train`, `validate`, `deploy`, …) |
| Each candidate | runs in-process | runs as a versioned Metaflow run |
| Input | a local `--data` dir | a **Metaflow data object** (`--dataset <pathspec>` from `loom ingest`) |
| Data | local | a Metaflow artifact; the datastore (local or S3/minio) stays in **your** perimeter, opaque to Loom |

The `local` path is the on-ramp. **The lifecycle verbs need `--mlops metaflow`** (a
Metaflow datastore) — they run *flows*, not in-process candidates.

---

## Interactive CLI (the `loom` REPL)

Run `loom` with **no subcommand** (or the explicit alias `loom chat`) to drop into
a branded interactive shell — a thin loop over the *same* verbs, with
a themed render layer, interactive approval gates, and a streaming search. It is a
shell over the lifecycle, not an "automated ML engine": every line routes through
the exact same parser and handlers the one-shot `loom <verb>` commands use, so the
REPL can never drift from the CLI.

```bash
loom            # no subcommand -> the REPL
loom chat       # the same thing, explicitly
loom --no-ui    # plain, color-stripped output (also via LOOM_NO_UI=1) — for CI/pipes
```

**Slash-commands.** Type a verb with or without a leading `/`; tab-completion
offers every verb plus a small meta set:

- every lifecycle verb — `/eda`, `/datasets`, `/viz`, `/features`, `/run`,
  `/validate`, `/pipeline`, `/deploy`, `/train`, `/ops`, `/report`, `/collab`,
  `/ingest`, `/telemetry`, `/skillopt`;
- meta — `/help` (the verb table), `/status` (the banner + active providers),
  `/doctor` (the stack health check), `/clear`, `/exit` (and `/quit`).

A bare natural-language line isn't free-form chat — it returns a hint listing the
verbs. Quoted free-text args survive (`/run --goal "predict churn" --metric auc`).

**Streaming + approval UX.** A running verb is wrapped in a spinner; `/run` and
`/pipeline` drive the AIDE search and stream the leaderboard as it fills, falling
back to a spinner + the final rendered leaderboard when streaming isn't reachable.
The approval gate is interactive and enforces the
[§1 cost/data matrix](skills/CONVENTIONS.md) — read-only
verbs never prompt; workspace-write runs with a one-line auto note; the
**expensive / irreversible** real actions (`deploy --apply`, `train --launch`,
`collab --send`) show the cost/operation and require a deny-first `y/N` confirm
before the handler runs. `Ctrl-C` cancels the current action (not the REPL);
`Ctrl-D` or `/exit` quits cleanly.

**Keyless by default.** The read-only / lifecycle verbs (`datasets`, `eda`, `viz`,
`report`, `ops`, `validate`, `features`, `train` local, `telemetry`, `doctor`) work
**without an API key**. Only the LLM verbs (`run` / `pipeline`, which drive the AIDE
search brain) need one — and a missing key yields an **actionable line** (`set
ANTHROPIC_API_KEY …` or pick a `--model-provider`), never a traceback.

A short transcript (no key, no infra needed):

```text
 ██╗      ██████╗  ██████╗  ███╗   ███╗
 ██║     ██╔═══██╗██╔═══██╗ ████╗ ████║
 ██║     ██║   ██║██║   ██║ ██╔████╔██║
 ██║     ██║   ██║██║   ██║ ██║╚██╔╝██║
 ███████╗╚██████╔╝╚██████╔╝ ██║ ╚═╝ ██║
 ╚══════╝ ╚═════╝  ╚═════╝  ╚═╝     ╚═╝
  an agentic CLI for data science
  v0.1.0

  search        : aide
  mlops         : metaflow
  model-builder : nemo
  model         : anthropic-api

  type /help for the verbs, /exit to quit.
loom> /eda --dataset IngestDataset/123 --target is_fraud
  running eda...
╭─ EDA profile ────────────────────────────────────────────╮
│ rows x cols : 50000 x 31                                  │
│ target      : is_fraud                                    │
│ leakage     : none detected                               │
╰──────────────────────────────────────────────────────────╯
✓ eda ok
loom> /deploy --validate ValidateFlow/41 --apply
╭─ Approval required: IRREVERSIBLE/EXTERNAL ────────────────╮
│ deploy to the external registry                           │
│ This is the IRREVERSIBLE/EXTERNAL tier — it always gates. │
│ The model proposes; only you can fire this.               │
╰──────────────────────────────────────────────────────────╯
Proceed with deploy to the external registry? [y/N]: n
  IRREVERSIBLE/EXTERNAL: BLOCKED — deploy to the external registry not approved.
loom> /exit
  bye.
```

The interactive UI deps (`rich` + `prompt_toolkit`) are imported **lazily** from
the REPL path, so a stripped environment still runs every one-shot subcommand even
without them installed.

---

## The data model: your input is a Metaflow data object

Loom's input is a **Metaflow data object — a Metaflow Artifact** — referenced by
**pathspec** (e.g. `IngestDataset/123`) and read **only through the Metaflow
Client API**. The datastore behind it — local or object storage (S3/minio) — is
an **opaque detail Metaflow owns**: you configure it once in your Metaflow
profile/environment, and **Loom never talks to that storage directly** (no bucket
URIs in Loom code). There is exactly **one boundary** where outside data crosses
in: `loom ingest`. After that the dataset is a versioned, profile-backed object
you reference by pathspec — the same object whether Loom runs locally now or in a
cluster later.

```bash
# Ingest once — a local dir/CSV becomes a Metaflow data object; prints the dataset_ref.
loom ingest --source ./your_task --name my-dataset      # -> dataset_ref : IngestDataset/123

# See what's ingested (read-only, via the Client API).
loom datasets
```

A directory source contains `train.csv` (plus optional `test.csv` /
`sample_submission.csv`); a single `.csv` is split for you. Solutions read from
`./input/` and write predictions to `./working/submission.csv`.

---

## The lifecycle commands

Each runs through Loom's MLOps interface (`ExecutionProvider.run_flow`) and returns
a Metaflow run + an `@card` + a typed summary. The noun is a single `--dataset`
pathspec; a small meta flag set (`--from`, `--target`, …) does the rest.

```bash
# Profile a data object — schema, missingness, target balance, correlations, leakage flags.
loom eda --dataset IngestDataset/123 --target target

# Plot a data object or a run's results as @card images.
loom viz --dataset IngestDataset/123 --kind all        # or: --run EvalCandidate/42

# Build engineered features into a NEW data object (workspace-write). Compose with eda
# via --from: EDA-flagged leakage columns are DROPPED. The new FeaturesFlow/<id> pathspec
# is a --dataset for every downstream verb.
loom features --dataset IngestDataset/123 --target target [--from EdaFlow/7] [--recipe minimal|full]

# The AIDE search (the engine behind /loom-optimize): propose/run/score candidates.
loom run --dataset IngestDataset/123 --goal "<one sentence>" --metric "<one sentence, with direction>" --steps 10

# Rigorously validate a baseline/solution — CV + a sealed holdout + calibration +
# fairness (with --sensitive) + leakage flags. Emits a VERDICT that gates promotion.
loom validate --dataset IngestDataset/123 --target target [--solution EvalCandidate/42] [--sensitive region]

# Run the whole lifecycle as ONE gated run: profile -> features -> a bounded optimize
# step -> validate. Each stage asserts the prior VERDICT (leakage blocks features;
# a sub-threshold validate marks the run FAIL).
loom pipeline --dataset IngestDataset/123 --goal "<one sentence>" --target target

# Promote a validated solution — IRREVERSIBLE/EXTERNAL. Deploy ASSERTS the upstream
# validate VERDICT==PASS; a sub-threshold/FAIL/leaky validate BLOCKS it (cross-verb gate).
# --apply is OFF by default: the default is a PLAN + staged manifest, NO external mutation.
loom deploy --validate ValidateFlow/12            # PLAN only (no mutation)
loom deploy --validate ValidateFlow/12 --apply    # real action (gate must ALLOW)

# Monitor (read-only): run health, the leaderboard, and a data-object DRIFT check.
loom ops --flow ValidateFlow
loom ops --dataset IngestDataset/200 --reference IngestDataset/123

# Assemble a sanitized, shareable bundle (report/card + lineage). --send is OFF by
# default (builds only, nothing leaves the box); --send pushes to an env/config sink.
loom collab --run ValidateFlow/12                 # build only, NO send

# An experiment's runs + metrics + lineage -> a model-card (read-only).
loom report --experiment loom-abc123              # or: --runs EvalCandidate/1,EvalCandidate/2
```

### `loom train` — build a model via the model-builder (NeMo)

`loom train` runs the **`/loom-train`** verb against the **model-builder interface**
(third port, default **NeMo**). You state intent in DS vocabulary —
`--objective {next-event|masked-field|contrastive}`, `--budget {probe|small|full}`,
`--backbone`, `--metric` — and the adapter compiles that to backend config; no
NeMo/Megatron/`.nemo` nouns ever surface. `pretrain` is **launch-and-track** (AIDE
never tree-searches it; for cheap scalars like heads/tokenization use `loom run`).

`train` is the **expensive/mutate, always-gate** tier: the `@card` shows the cost
PLAN (GPU-count · hours · $) for your budget, and the real heavy GPU launch is
**OFF by default behind `--launch`**.

```bash
# Default backend is `nemo`. With no GPU target it REFUSES cleanly (never launches) —
# it shows the cost plan and tells you what to set. This is the safe default.
loom train --dataset IngestDataset/123 --objective next-event --budget full
#   cost (gate) : budget=full: 8 GPU x 12 h = 96 GPU-hours (~$288)
#   STATUS      : REFUSED_NO_GPU_TARGET   (set LOOM_GPU_TARGET, or use the CPU stand-in below)

# The torch-free CPU `local` stand-in actually builds a backbone end-to-end — no GPU,
# sub-2s, deterministic (a real PPMI+SVD embedding model; great for dev + CI).
LOOM_MODEL_BUILDER_PROVIDER=local \
  loom train --dataset IngestDataset/123 --objective next-event --budget probe
#   STATUS      : BUILT   (-> a backbone pathspec + @card)
```

For the real GPU pretrain (v0.2), point `LOOM_GPU_TARGET` at an on-demand GPU
(Modal / GCP / AWS) — your laptop stays the control plane. The model-builder port,
the gate, the cost-surface, and the `local` stand-in are documented in
[`docs/architecture.md`](docs/architecture.md).

---

<a name="set-up-the-metaflow-datastore"></a>
## Set up the Metaflow datastore (to test the lifecycle verbs)

The lifecycle verbs need `--mlops metaflow` + a datastore. **The one-command path
is built:** the [`/loom-setup-metaflow`](skills/loom-setup-metaflow/SKILL.md) skill
drives the idempotent [`scripts/setup_metaflow_minikube.sh`](scripts/setup_metaflow_minikube.sh)
(detect-before-install the prereqs, start the cluster, apply minio, write
`.env.metaflow`), then verifies with the read-only `loom doctor` (PASS/WARN/FAIL per
check + a one-line VERDICT; exits 0 iff nothing FAILs). The skill **always gates**
before installing/starting anything (it is local-dev + reversible via
`minikube delete`).

```bash
# One command (gated): stand up the local datastore, then verify it.
bash scripts/setup_metaflow_minikube.sh
source .env.metaflow && loom doctor        # read-only health check — must end VERDICT: PASS
```

Or run the verified recipe by hand (minikube + minio as an S3-compatible datastore —
runs on a laptop, no cloud, no GPU):

```bash
# 1. A local cluster + an S3-compatible store (minio). (Docker driver via colima on macOS.)
minikube start
kubectl create namespace loom
kubectl apply -n loom -f skills/loom-setup-metaflow/manifests/minio.yaml
kubectl port-forward -n loom svc/minio 9000:9000 9001:9001 &     # keep this alive

# 2. Point Metaflow at it (S3 datastore on minio, local metadata). Local-dev creds only.
export METAFLOW_DEFAULT_DATASTORE=s3
export METAFLOW_DATASTORE_SYSROOT_S3=s3://metaflow/metaflow
export METAFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin123
export METAFLOW_DEFAULT_METADATA=local
export METAFLOW_USER="$(whoami)"
aws --endpoint-url http://localhost:9000 s3 mb s3://metaflow   # create the bucket once

# 3. Now ingest + run the lifecycle end-to-end:
loom ingest --source ./your_task --name my-dataset            # -> IngestDataset/123
loom datasets
loom eda --dataset IngestDataset/123 --target target
loom validate --dataset IngestDataset/123 --target target
LOOM_MODEL_BUILDER_PROVIDER=local loom train --dataset IngestDataset/123 --objective next-event --budget probe
```

A `train` fixture with a planted signal wants per-account event sequences
(`account · t · event · … · label`); generic tabular works for `eda`/`features`/
`validate`/`run`.

---

## Examples & the eval bed

[`examples/`](examples/) is six per-use-case walkthroughs that **double as a
regression eval bed** — the core lifecycle, leakage detection, the sequence
model-builder (CPU), gated deploy, ops/drift, and telemetry distillation. Each
is self-contained and **self-checking**: a `run.sh` generates deterministic
domain-neutral synthetic data, ingests it under a unique dataset name, runs a
**keyless** verb sequence with `--json`, and asserts the outcomes inline
(exiting nonzero on any drift in the `--json` contract).

```bash
source /tmp/loom-cluster-env.sh                       # the datastore env (above)
bash examples/01-tabular-classification/run.sh        # one example
for d in examples/[0-9]*/; do bash "$d/run.sh"; done  # the whole bed
```

`tests/test_examples.py` replays every `run.sh` and asserts exit 0, so a
regressed verb outcome turns red in CI; it **skips cleanly** when no datastore is
reachable. Start at [`examples/README.md`](examples/README.md) for the index and
the keyless-vs-key-gated note.

---

## Drive it conversationally (Claude Code)

Prefer to talk to it? The [`skills/`](skills/) pack turns Loom into a Claude Code
workflow — one verb table, both surfaces. `/loom-setup-metaflow` stands up the local
datastore (and `loom doctor` health-checks it), `/loom-connect` brings data in,
`/loom-eda` profiles it, `/loom-optimize` pins the metric and runs the search,
`/loom-train` builds a backbone, `/loom-validate` checks a sealed holdout,
`/loom-pipeline` chains the lifecycle, `/loom-deploy` gates promotion,
`/loom-viz` / `/loom-report` / `/loom-collab` visualize, write up, and share. Or
let **`/loom-auto`** drive the whole happy path for you — it orchestrates the verbs
end-to-end (eda → features → optimize → validate → report), threading each artifact
and asserting each VERDICT, gating only at the expensive optimize step and **never**
auto-firing deploy/collab-send. Each verb plans, gates on cost/data, calls the
interface, and narrates a lineage-grounded result. See
[`skills/README.md`](skills/README.md).

---

## Configuration

Everything is configured by flags + environment variables (and an optional `.env`
or `--config YAML`). **Secrets are only ever read from the environment — never put
a key on the command line.**

| Env var | Flag | Default | Meaning |
|---|---|---|---|
| `LOOM_SEARCH_PROVIDER` | `--search` | `aide` | search ("brain") provider |
| `LOOM_MLOPS_PROVIDER` | `--mlops` | `metaflow` | execution ("muscle") provider |
| `LOOM_MODEL_BUILDER_PROVIDER` | — | `nemo` | model-builder ("training") provider — `nemo` or `local` |
| `LOOM_GPU_TARGET` | — | (none) | GPU target for `train --launch`; unset ⇒ a clean refusal, never a launch |
| `LOOM_CODE_PROVIDER` | `--code-provider` | `anthropic-api` | model ("LLM backend") provider for the code role |
| `LOOM_FEEDBACK_PROVIDER` | `--feedback-provider` | `anthropic-api` | model provider for the feedback/judge role |
| — | `--model-provider` | — | shorthand setting **both** roles at once |
| `LOOM_BUDGET_STEPS` | `--steps` | `10` | number of search steps |
| `LOOM_CODE_MODEL` | — | `claude-sonnet-4-5` | model that writes solutions |
| `LOOM_FEEDBACK_MODEL` | — | `claude-sonnet-4-5` | model that reviews/scores |
| `METAFLOW_PROFILE` | — | (Metaflow default) | your Metaflow endpoint |

`loom <verb> --help` lists every flag.

---

## Model providers

The **model provider** is Loom's LLM-backend port: *which model the brain talks to
and how it's authenticated*. Pick one per role — `--code-provider` (writes
solutions) and `--feedback-provider` (the judge that scores) — or set both with
`--model-provider`. Each configures [AIDE](https://github.com/WecoAI/aideml)'s
**existing** LLM backends via config/env; Loom never forks AIDE and **never handles
your secrets**.

> **The judge needs tools.** AIDE's feedback step always uses tool/function
> calling. If your feedback route can't, `loom run` fails fast (exit 2) and tells
> you to pick a tool-capable feedback model.

| Provider | Auth | Judge-capable |
|---|---|---|
| `anthropic-api` | `ANTHROPIC_API_KEY` (honors `ANTHROPIC_BASE_URL`) | yes (native Claude tool use) |
| `openai-api` | `OPENAI_API_KEY` | yes (Responses-API tools) |
| `openrouter` | `OPENROUTER_API_KEY` → `OPENAI_API_KEY`; model is a `provider/model` slug | per-slug — pick a tool-capable feedback slug |
| `nim` | `NVIDIA_API_KEY`; endpoint via `OPENAI_BASE_URL` (default `integrate.api.nvidia.com`) | configurable (default yes) |
| `openai-compat` | `OPENAI_API_KEY`; endpoint via `LOOM_MODEL_BASE_URL` (LiteLLM / vLLM / Ollama) | configurable (default yes) |
| `claude-subscription` | none — your local `claude` CLI login | yes (CLI text coerced to JSON) |
| `codex-subscription` | none — your local `codex` CLI login (`~/.codex/auth.json`) | yes (`codex exec --output-schema`) |
| `loom-proxy` | `LOOM_API_KEY` (the gateway holds the vendor key server-side) | yes (Anthropic passthrough) |

```bash
# Native OpenAI for both roles
export OPENAI_API_KEY="sk-..."; export LOOM_CODE_MODEL="gpt-4o"; export LOOM_FEEDBACK_MODEL="gpt-4o"
loom run --data ./task --goal "..." --metric "..." --model-provider openai-api

# NVIDIA NIM (a real NeMo touchpoint, zero GPU infra — "NeMo inside Loom" on day one)
export NVIDIA_API_KEY="nvapi-..."
loom run --data ./task --goal "..." --metric "..." --model-provider nim

# Mix roles: a cheap local model writes code, Claude judges
export OPENAI_BASE_URL="http://localhost:11434/v1"   # Ollama, say
loom run --data ./task --goal "..." --metric "..." --code-provider openai-compat --feedback-provider anthropic-api
```

**OpenRouter** speaks an OpenAI-compatible API but AIDE's *dedicated* OpenRouter
backend can't do tool calling, so the `openrouter` provider points AIDE's
**OpenAI-compatible** path at OpenRouter (`OPENAI_BASE_URL=…/api/v1`, key copied
into `OPENAI_API_KEY`) and best-effort checks the feedback slug's tool support.

**Subscriptions** (`claude-subscription` / `codex-subscription`) are **CLI
bridges**: Loom drives the `claude` / `codex` CLI **you already installed and
logged in**, never seeing your credentials. Metering is a **separate
subscription/rolling-window pool** — a heavy AIDE run is many calls and can exhaust
it; for sustained runs an API key is the better fit. `loom run` pre-flights both
(a missing binary/login → an actionable hint).

---

## The moat — the data flywheel & LOOM-DS-1

Loom's long game is a **data flywheel**: usage produces a trace corpus, the corpus
improves Loom, better Loom drives more usage. The **primary capture seam is the
Loom gateway** (the `loom-proxy` provider + `loom proxy serve`) — a thin
**Anthropic-passthrough** that holds the vendor key server-side, injects Loom's
prompt, forwards to Anthropic, and **logs every call centrally** (request +
response + usage + model + latency + tenant/owner tags) to one JSONL corpus.

```bash
# Server side: the gateway holds the REAL vendor key; callers present only a LOOM_API_KEY.
export ANTHROPIC_API_KEY="sk-ant-..."     # stays on the server
export LOOM_API_KEY="loom-..."            # the key callers must present (allowlist)
loom proxy serve --host 127.0.0.1 --port 8088

# Client side (another shell, only the LOOM key):
export LOOM_API_KEY="loom-..."
loom run --data ./task --goal "..." --metric "..." --model-provider loom-proxy
```

That central capture feeds two loops:

- **Now — text-space (zero GPU), wired in v0.2:** **HiveMind + SkillOpt** is the
  built inner loop — `loom skillopt` (the `loom-skillopt` verb). HiveMind captures
  each `/loom-*` verb's `learnings/rollouts.jsonl` corpus (`owned_by=general` only —
  the IP boundary; tenant rows are excluded from the cross-tenant moat), then
  SkillOpt's deterministic, LLM-free scorer grades the incumbent `SKILL.md` + any
  candidate on Loom's mixed metric (HARD = the 7-point acceptance contract; SOFT =
  corpus failure-mode coverage) and applies a **held-out, never-worse promotion gate**
  — the exact parallel of `loom-deploy`'s exit gate: the best hard-valid candidate is
  promoted only if it beats the incumbent by a margin, so a contract violator or a
  regression **can never deploy a worse skill**. **Safe by default:** it PROPOSES a
  `SKILL.candidate.md` sidecar + the gate VERDICT + a diff; the in-place overwrite is
  behind `--apply` and runs ONLY when the gate PROMOTED (mirroring `loom deploy
  --apply`). Each `/loom-*` command IS a `SKILL.md` — the *trainable artifact*.
- **Later — weights:** distill **LOOM-DS-1**, an open-weights, Loom-owned
  **data-science model** fine-tuned (via NeMo) on the accumulated teacher traces —
  *the way Claude has a coding model, Loom has a data-science one.* The outer loop,
  same corpus.

**Privacy — collection is a choice, not a hidden default:**

- **Your bulk data never goes to the LLM, in *either* mode.** Datasets/transactions
  live as Metaflow data objects in **your** datastore/perimeter; candidate code
  processes them there. The LLM only ever sees small *derived* context (schema,
  preview, code, metrics) — keep it that way with **prompt hygiene**.
- **LLM prompts go to a third-party LLM either way.** Using Claude at all sends those
  small prompts to Anthropic; `loom-proxy` sits in that *same* path — **no
  incremental egress** ("Claude or Us — no difference").

| Provider | Prompts go to | Loom logs them? |
|---|---|---|
| **`loom-proxy`** (default *once hosted*) | Anthropic, via Loom's gateway | **yes** — to the moat corpus, tagged by tenant/owner |
| **BYO-key** (everything else) | straight to your vendor/endpoint | **no** — the gateway sees nothing |

A tenant that doesn't want its traces collected uses a BYO-key provider; its bulk
data is protected the same way in both modes. The moat trains **only on
`owned_by: general` data**; tenant-tagged rows are isolated, and **no keys are ever
logged**. `loom-proxy` is **opt-in today** (the gateway isn't hosted yet) and
becomes the default once hosted. `LOOM_API_BASE` (default `http://127.0.0.1:8088`)
and `LOOM_PROXY_LOG_PATH` (default `learnings/proxy_calls.jsonl`) configure it.

### Hosting the gateway (flip `loom-proxy` to default)

The hosted gateway is the **primary distillation-capture seam** — it's the one
place every routed call is logged centrally, so it feeds the telemetry layer (the
`trajectory_id`-correlated `llm_request` events) and, through it, LOOM-DS-1.
`deploy/gateway/` containerizes `loom proxy serve` (Dockerfile + Compose + a k8s
Deployment/Service); host it, set `ANTHROPIC_API_KEY` (the real vendor key, stays
server-side) and `LOOM_API_KEYS` (the caller allowlist) as **runtime secrets**
(never baked into the image), and point clients at it with `LOOM_API_BASE`. See
[`deploy/gateway/README.md`](deploy/gateway/README.md).

Once hosted, `loom-proxy` becomes the **default** model provider automatically:
when a **hosted** gateway is detected — `LOOM_API_BASE` set to a **non-loopback**
URL, or `LOOM_PROXY_DEFAULT` truthy — and the user hasn't explicitly chosen a
provider, Loom defaults `code_provider` / `feedback_provider` to `loom-proxy`
instead of `anthropic-api`. Precedence: an explicit `--code/feedback/model-provider`
(or `LOOM_CODE/FEEDBACK_PROVIDER`) **always wins**; else a hosted gateway ⇒
`loom-proxy`; else `anthropic-api`. A loopback base (`127.0.0.1` / `localhost`, the
default included) is local dev, **not** hosting, so it never flips the default.

### Telemetry & distillation — trajectories → LOOM-DS-1

**Telemetry here is COMPLETE training-data collection for LOOM-DS-1, NOT
observability.** No sampling, no batch-drop, no TTL, no aggregation — **every
trajectory, in full** (modeled on a transcript, not a metrics pipeline). The
corpus is the **complete append-only JSONL** → a **versioned Metaflow data
object** (`loom telemetry export --to-dataset`) for durable scale. OTel is an
**optional ops-dashboards mirror ONLY** (`LOOM_TELEMETRY_OTEL_OPS`) — **do NOT
send the training corpus to a sampling observability/metrics backend** (they
sample + aggregate + expire — even first-party analytics pipelines do).

The capture above scatters a run's signal across three stores; the **telemetry
layer** (`loom/telemetry/`, modeled on Claude Code's append-only session
**transcript**, not its sampling analytics plane) stitches them into
**distillation-grade trajectories** — the training examples LOOM-DS-1 distills from.
It does **not** re-log; it **correlates**. Three signal types feed one trajectory:

- **Telemetry events** (`telemetry/events.jsonl`) — sequenced, content-redacted
  events (`trajectory.start`, `llm_request`, `trajectory.end`) carrying standard
  attributes + a **monotonic `event.sequence`** for ordering.
- **Proxy LLM I/O** (`learnings/proxy_calls.jsonl`) — the request/response the
  gateway already logs, now tagged with the trajectory.
- **Command rollouts** (`learnings/rollouts.jsonl`) — the per-run outcome/metric.

The model is CC's **interaction-root** one: every user-request → response cycle is a
single root *trajectory* (the controller opens it with the run's `experiment_id` and
closes it with the outcome + duration), under which the LLM calls and tool/exec steps
are correlated by one stable **`trajectory_id`** written onto every signal.
`assemble_trajectory` is the pure JOIN that re-materializes one ordered trajectory
(`context → [llm_request → response → tool/exec → observation]* → outcome{reward}`).

**Content is redacted by default** to `<REDACTED:kind>` unless `LOOM_LOG_CONTENT`
is set — only schema/preview/metrics enter the corpus, never raw rows, never keys.
The **IP boundary** is the same as the corpus/learnings: the distillation export
trains **only on `owned_by: general`** trajectories; a tenant-owned one is
excluded.

The corpus capture (`LOOM_TELEMETRY`) is **decoupled** from the **optional OTel
ops mirror**: that mirror is a *separate* explicit opt-in, gated by
`LOOM_TELEMETRY_OTEL_OPS` **in addition to** `OTEL_*_EXPORTER` (console/otlp),
lazily imported and a clean no-op (never a hard dep) when the SDK is absent.
Enabling capture never implies the ops mirror. ⚠ The ops mirror is **ops-only,
NOT the corpus** — never route the training corpus to a sampling
observability/metrics backend; they sample + aggregate + expire. The complete
append-only JSONL corpus is always available and never flows through an exporter.

```bash
loom telemetry status                       # read-only: counts, the general/tenant split, ops-mirror state, paths
loom telemetry export                        # assemble trajectories -> telemetry/loom-ds-1.jsonl (general-only, redacted)
loom telemetry export --to-dataset loom-ds-1 # ALSO ingest the corpus as a VERSIONED Metaflow data object (durable, lossless) -> prints its pathspec
loom telemetry export --owned-by general --with-content   # opt in to raw content (off by default)
loom telemetry trace --trajectory loom-abc123             # show one assembled trajectory
```

`loom telemetry export` is the bridge to the **LOOM-DS-1** corpus: each kept
trajectory becomes one reward-weighted SFT/teacher example (`context`,
`teacher_output`, `tools_trajectory`, `reward`/`weight`). The `--out` JSONL is
always written; **`--to-dataset NAME`** additionally ingests that same corpus
through the same `IngestDataset` seam `loom ingest` uses, so it becomes a
**durable, content-addressed, versioned, lossless Metaflow data object** — a
first-class `dataset_ref` for scale (needs `--mlops metaflow`). Telemetry is
**off unless `LOOM_TELEMETRY` is set**; `LOOM_TELEMETRY_PATH` /
`LOOM_TRAJECTORIES_PATH` (defaults under `telemetry/`) and
`LOOM_TELEMETRY_INCLUDE_SESSION_ID` (cardinality) configure it. This is the
full-lifecycle data discipline — *capture → correlate → distill* — not an
automated black box.

---

## How it's built

Loom-core defines provider **interfaces** (ports); concrete **adapters** plug in and
are picked by config — add a new backend by writing one class and registering it,
no core changes.

| Seam | Port | Built-in adapters |
|---|---|---|
| Search ("brain") | `SearchProvider` | `aide` |
| Execution ("muscle") | `ExecutionProvider` | `metaflow`, `local` |
| Model-builder ("training") | `ModelBuilderProvider` | `nemo`, `local` |
| Model ("LLM backend") | `ModelProvider` | `anthropic-api`, `openai-api`, `openrouter`, `nim`, `openai-compat`, `claude-subscription`, `codex-subscription`, `loom-proxy` |

```
loom/
  types.py        ExecutionResult · RunResult · Task · SearchResult · NodeRecord ·
                  ArtifactRef · Scores · Capability · CapabilityManifest   (the locked contract)
  config.py       LoomConfig (env / .env / YAML)
  registry.py     name -> provider class  (search · execution · model · model-builder)
  dataio.py       materialize_dataset / dataset_schema — the Client-API data door (never touches S3)
  providers/      aide_search.py · metaflow_exec.py · local_exec.py · model/ · model_builder/{local,nemo}.py
  controller.py   run_loom(task, config) — wires it together
  corpus.py       append-only JSONL of every node (multi-tenant IP boundary)
  proxy/          the Loom gateway — Anthropic passthrough that logs every call (the moat capture)
  cli.py          the `loom` command (doctor · ingest · datasets · eda · viz · features · run ·
                  train · validate · pipeline · deploy · ops · collab · report · proxy serve ·
                  skillopt · telemetry)
  telemetry/      distillation-grade trajectory capture — events · attributes · the interaction-root
                  trajectory model + the JOIN · the LOOM-DS-1 distill bridge · the optional OTel sink
  hivemind.py     capture: learnings traces -> a per-verb VerbCorpus digest (the flywheel's left half)
  skillopt.py     the deterministic SKILL.md scorer + the never-worse promotion GATE (the moat's heart)
flows/
  ingest_dataset.py  the external->Metaflow boundary (dir/CSV -> data object)
  eval_candidate.py  the one static evaluation flow (candidate = data)
  eda · viz · features · validate · pipeline · deploy · ops · collab · report · train .py
                     one static lifecycle flow per verb -> a Metaflow run + @card
skills/           the Claude Code skill-pack — one /loom-* skill per verb (incl. loom-train, loom-skillopt)
tasks/generic_demo/   the bundled smoke-test task
```

Full design + "how to add a provider": [`docs/architecture.md`](docs/architecture.md).
Repository invariants: [`CLAUDE.md`](CLAUDE.md).

---

## Tests

```bash
pip install -e ".[dev]"   # or: pip install pytest
pytest                    # 411 passed, 1 skipped
```

Pure-Python tests (registry, corpus, the `local` paths, the model-builder
**conformance suite**) run without an LLM key or Metaflow; tests that need AIDE or
Metaflow skip cleanly if those aren't installed. The conformance suite runs
torch-free; the optional torch fidelity mode is behind the `model-local` extra.

---

## Status

**v0.1 + the v0.2 text-space moat.** The whole lifecycle verb-catalog is built behind
three swappable ports — AIDE (search), Metaflow (orchestrate), NeMo (train). The
`local` MLOps + `local` model-builder paths run on a laptop with no GPU and only an
LLM key; the real GPU pretrain is launch-and-track and deferred to v0.2 (the seam,
gate, and cost-surface are in place — it refuses cleanly with no GPU target). The
**self-improvement loop's text-space half is now wired** — HiveMind capture +
SkillOpt's never-worse held-out gate (`loom skillopt`) close the flywheel on the
`learnings.jsonl` corpus (deterministic, LLM-free, safe-by-default `--apply`); the
weights-space outer loop (distilling LOOM-DS-1 via NeMo) stays on the v0.2+ roadmap.
AIDE is pinned by commit for reproducibility. Loom is intentionally general-purpose —
point it at any dataset with a measurable goal.
