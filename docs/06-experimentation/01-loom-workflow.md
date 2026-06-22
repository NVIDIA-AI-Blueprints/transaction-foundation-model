# The Loom Workflow: Disciplined Experimentation on This Model

**[Loom](https://github.com/ZKAI-Network/loom)** is ZKAI's agentic CLI for the data-science lifecycle. This page explains what it is, the five concepts you need, and — concretely — how each kind of experiment on this repo runs through it.

> Loom evolves quickly; treat this page as the *usage contract* and `loom /help` + the repo's `README.md`/`INSTALL.md` as ground truth for flags.

## 1. What Loom is (and the philosophy you're buying into)

Loom turns "I have data, a goal, and a metric" into an orchestrated, *gated*, fully-tracked sequence of lifecycle stages:

```
ingest → eda → features → optimize (search) → train → validate → report → deploy → ops
```

Three design principles, quoted from its docs because they shape daily use:

1. **"The metric is the spec."** Every run starts by declaring a dataset, a one-sentence goal, and a one-sentence measurable metric. No metric, no experiment.
2. **Ports and adapters.** Four swappable provider interfaces: the *search brain* (default **AIDE** tree-search), the *execution/MLOps backend* (default **Metaflow**; `local` for dev), the *LLM provider* (Claude by default), and — most relevant to us — the *model-builder* (default **NeMo AutoModel**, i.e. exactly [this repo's training stack](../02-concepts/06-gpu-stack.md)). Backends change by config, not code.
3. **Safety by default.** Read-only verbs never prompt; expensive verbs always show a cost PLAN and gate; irreversible verbs (`deploy --apply`, `collab --send`, real GPU launches) require explicit human confirmation and are never auto-fired by the model.

## 2. The five concepts

| Concept | What it is | Why you care |
|---|---|---|
| **Verbs** | `loom eda`, `loom features`, `loom train`, `loom validate`, … — each one CLI command + one `/loom-*` skill | the vocabulary of every experiment |
| **Data objects & pathspecs** | ingested datasets and run outputs are versioned Metaflow artifacts addressed like `IngestDataset/123`, `FeaturesFlow/456` | artifacts thread between stages by reference — no loose CSVs as truth |
| **VERDICTs & gates** | stages emit machine-checkable verdicts (`PASS`/`REVIEW`/`FAIL`, leakage flags); downstream stages assert upstream verdicts | a leaky feature **blocks** the feature stage; a failed validation **blocks** deploy |
| **Capability modes** | model-builder capabilities are `searchable` (cheap scalars — the AIDE brain may tree-search them) or `launch-and-track` (expensive — **never** auto-searched; gated launch) | pretraining is launch-and-track; pooling/head/hyperparams are searchable — this distinction is the whole GPU-safety story |
| **Learnings corpus** | every run appends a typed, content-redacted record (`learnings/rollouts.jsonl`); `loom telemetry export` distills trajectories | experiments compound: our runs train Loom's own DS model (LOOM-DS-1) |

## 3. Setup in two modes

```bash
git clone git@github.com:ZKAI-Network/Loom.git && cd Loom
python3.12 -m venv .venv && source .venv/bin/activate && pip install -e .
export ANTHROPIC_API_KEY="sk-ant-..."        # only search/agentic verbs need a key
```

- **`--mlops local`** — no infrastructure; in-process candidates; perfect for iterating on a laptop/dev box. Covers `loom run` (the AIDE search engine).
- **`--mlops metaflow`** — the full lifecycle with versioned runs and artifacts. One-time local datastore: `bash scripts/setup_metaflow_minikube.sh` then `source .env.metaflow`; or point at your own Metaflow deployment. `loom doctor` must end `VERDICT: PASS`.

Rule of thumb we follow: **prototype `local`, promote to `metaflow` the moment a result might be worth citing** — only Metaflow runs have the lineage to back a claim.

## 4. How this repo's experiments map onto Loom

### Pattern A — cheap, searchable experiments (no retraining)

For ideas that don't touch the checkpoint — pooling variants ([R6](../03-learning-path/level-400-design-contracts-and-extensions.md#4-extension-recipes)), PCA dimensionality, XGBoost feature sets, graph-feature fusion ([G1](../05-research/02-improvement-ideas.md#g--graph--relational-context)) — let the AIDE brain search against the declared metric:

```bash
# One-time: get the eval table (embeddings ∥ raw features ∥ label) into a data object
loom ingest --source ./data/eval_frames/combined_test --name tfm-eval-2026q2
loom eda --dataset IngestDataset/101 --target is_fraud        # read-only; flags leakage

loom run \
  --dataset IngestDataset/101 \
  --goal   "Maximize fraud detection quality using foundation-model embeddings + raw features." \
  --metric "Maximize average precision (AUPRC) on the temporally held-out test split." \
  --steps 15 \
  --mlops metaflow
# → leaderboard, journal, best-candidate run (a pathspec you can validate & cite)
```

The EDA leakage gate has teeth here: identity-flavored columns (the `CUST_*` family — [sharp edge #1](../03-learning-path/level-400-design-contracts-and-extensions.md#3-sharp-edges-read-before-deploying-or-publishing-numbers)) and target-correlated artifacts get flagged *before* they contaminate a "win".

### Pattern B — pretraining experiments (launch-and-track)

For ideas that change the model ([T1 time-delta](../05-research/02-improvement-ideas.md#t1--turn-on-and-then-improve-time-encoding), [T2 drop-CUST](../05-research/02-improvement-ideas.md#t2--drop-the-cust-token-and-ablate-card-the-deployability-ablation), [A1 context](../05-research/02-improvement-ideas.md#a--architecture--scale), new corpora): `loom train` is the **model-builder seam**. You speak data-science intent (*objective*, *budget*, *capability*); the NeMo adapter lowers it to the same NeMo AutoModel machinery as [`scripts/train_decoder_model.py`](../../scripts/train_decoder_model.py):

```bash
# 1. Ingest the (new-tokenizer) corpus as a versioned data object
loom ingest --source ./data/decoder_corpus_t1 --name tfm-corpus-t1-timedelta

# 2. Plan the pretrain — note: NO real launch yet, this prints the cost PLAN
loom train \
  --dataset IngestDataset/201 \
  --objective next-event \
  --budget small \
  --metric downstream-fraud-ap
#   cost (gate): budget=small: 4 GPU × 6 h ≈ 24 GPU-hours
#   STATUS     : REFUSED_NO_GPU_TARGET     ← refuses cleanly without a GPU target

# 3. Launch for real — explicit target + explicit --launch (the irreversible-tier gate)
export LOOM_GPU_TARGET=modal                      # or your cluster adapter
export LOOM_NEMO_IMAGE=nvcr.io/nvidia/nemo:25.09.01
loom train --dataset IngestDataset/201 --objective next-event --budget small \
           --metric downstream-fraud-ap --launch
# → TrainFlow/310: checkpoint tracked as an ArtifactRef with full lineage

# 4. Embed with the frozen backbone (cheap; searchable tier)
loom train --dataset IngestDataset/201 --capability embed \
           --backbone TrainFlow/310 --budget probe
# → EmbeddingsFlow/311 — a first-class data object

# 5. Validate against the baseline embeddings on identical splits
loom validate --dataset EmbeddingsFlow/311 --target is_fraud
# → VERDICT: PASS / REVIEW / FAIL, CV + sealed-holdout numbers
```

Budget tiers quote real money before anything runs (probe ≈ 1 GPU × 2 h; small ≈ 4 × 6 h; full ≈ 8 × 12 h — the full tier matching how this repo's shipped checkpoint was trained). **AIDE never tree-searches a pretrain** — `launch-and-track` mode structurally forbids spawning twenty 8-GPU jobs because the search brain felt inspired.

There's also a torch-free CPU model-builder (`LOOM_MODEL_BUILDER_PROVIDER=local`, PPMI+SVD embeddings, seconds, deterministic) — useful for **rehearsing the whole flow end-to-end** before any GPU is involved, and as a humbling classical baseline.

### Pattern C — new-data pipelines (chains, MBD)

The [data-section recipe](../04-data/08-from-raw-data-to-training-run.md) slots into the lifecycle verbs: `ingest` the Parquet export → `eda` (schema/coverage/freshness sanity — chain exports *do* arrive broken) → corpus generation → Pattern B. For repeated chain pulls, the chained pipeline (`loom pipeline` / the `/loom-auto` meta-skill: profile → features → bounded optimize → validate, each stage asserting the previous VERDICT) is the repeatable unit.

### The full loop, end to end

```
idea (research backlog) ──► branch + one-line hypothesis
      │ loom ingest / eda            (data object + leakage gate)
      │ loom train [--launch]        (cost-gated pretrain → checkpoint ref)
      │ loom train --capability embed (embeddings ref)
      │ loom run / validate          (search downstream; VERDICT vs baseline)
      │ loom report --experiment <id> (model card: runs + metrics + lineage)
      ▼ research KB write-back       (result, +/-, linked pathspecs)
```

`--experiment <id>` (e.g. `tfm-t1-timedelta`) threads every run of one investigation together; `loom report` then assembles the comparison and lineage automatically. `loom ops --dataset <ref> --reference <baseline-ref>` adds drift checks — the operational answer to the literature's [unsolved temporal-drift problem](../05-research/01-literature-review.md#7-other-findings-that-should-steer-our-roadmap).

## 5. The approval matrix (what gates when)

| Tier | Our verbs | Behavior |
|---|---|---|
| read-only | `eda`, `viz`, `report`, `ops`, `datasets`, `doctor`, `telemetry status` | never prompts; no key needed |
| workspace-write | `features`, `validate` | runs freely inside the workspace; light gates on data scope |
| expensive | `run`/`optimize` (search), `train --launch` planning | always shows cost/data PLAN first |
| irreversible / external | `train --launch` (real GPUs), `deploy --apply`, `collab --send` | explicit human confirmation; never model-auto-invoked |

Internalize the bottom row: **nothing in this workflow can spend cluster money or publish an artifact without a human pressing the button.** That property is why we route experiments through Loom rather than ad-hoc scripts.

## 6. House rules when using Loom on this repo

1. **One hypothesis per `--experiment` ID**, named for it: `tfm-t2-drop-cust`, `tfm-a1-ctx8k`, `tfm-d2-eth-corpus`.
2. **The metric sentence comes from the backlog item** — write it before the first command (default: AP on the temporal test split; the [multi-task battery](../05-research/02-improvement-ideas.md#e1--build-the-multi-task-behavioral-benchmark) as it lands).
3. **Baseline runs live in the same experiment** — re-run the unmodified pipeline under the same ID so the report contains its own control.
4. **Respect the contracts when staging data** — corpus + tokenizer-config + `vocab_size` travel together ([C1](../03-learning-path/level-400-design-contracts-and-extensions.md#1-the-contracts)); ingest them as one data object so the lineage can't lie.
5. **PASS before promote** — embeddings/checkpoints used in customer-facing demos must trace to a `validate` VERDICT of PASS.
6. **Write the ending** — link the `loom report` card in the research-KB entry, including for negative results.
