> **Status:** DRAFT proposal — for review by Anub & Yassine. **Last updated:** 2026-06-15.
> Supersedes the generic-DS Loom (archived in `../Loom-legacy/`). This is a from-scratch rebuild.

# Loom — an agentic tool for training a SOTA foundation model

## 1. What Loom is now (and is not)

**Loom is a tool whose single job is to help a team train a state-of-the-art foundation model.**
It is built **general** (any event-sequence foundation model) but its **first and driving use
case is this repo's transaction foundation model (TFM)**.

It is **not** a generic data-science workbench. The previous Loom tried to cover the whole DS
lifecycle (`eda · features · viz · report · collab · deploy · ops …`) with model training as one
port among many. That surface is dropped. The new Loom's surface *is* the FM-training loop and the
experiments that make a transaction FM SOTA.

This narrowing is consistent with how Loom is actually used here: Yassine's
[usage contract](../docs/06-experimentation/01-loom-workflow.md) and the
[experiment backlog](../docs/05-research/02-improvement-ideas.md) already revolve around
`ingest → tokenize → pretrain → embed → evaluate`. We make that loop the whole product and go deep.

## 2. Principles (kept from the usage contract)

1. **The metric is the spec.** Every campaign declares a dataset, a one-sentence goal, and a
   one-sentence measurable metric. No metric, no experiment.
2. **Ports & adapters.** Swappable provider interfaces (below) — backends change by config, not code.
3. **Safety by default.** Read-only verbs never prompt; expensive verbs print a cost PLAN and gate;
   irreversible verbs (real GPU launches, deploys) require an explicit human button and are never
   model-auto-fired.
4. **Contracts are load-bearing.** The six contracts in
   [Level 400](../docs/03-learning-path/level-400-design-contracts-and-extensions.md) (C1 vocab,
   C2 determinism, C3 corpus grammar, C4 `{input_ids,labels}`/`-100`, C5 HF safetensors, C6 temporal
   split + row-IDs) are enforced by Loom, not left to discipline. Most silent failures in FM training
   are a broken contract; Loom makes them loud.

## 3. The spine: the FM-training pipeline as a typed, gated DAG

Each stage produces a versioned **data object** (Metaflow artifact, addressed by pathspec) carrying
its lineage. Stages emit **VERDICTs** (`PASS`/`REVIEW`/`FAIL`); downstream stages assert upstream
verdicts.

```
ingest ─► tokenize ─► (eda/leakage gate) ─► pretrain[--launch] ─► embed ─► evaluate ─► report
  data      corpus+        schema/coverage     checkpoint           emb       multi-task   model
  object    vocab          /freshness          (ArtifactRef)        object    VERDICT       card
```

| Stage | What it does | Capability mode / gate | Backend |
|---|---|---|---|
| **ingest** | register a dataset as a versioned object: local Parquet, BigQuery chain export, GCS NDJSON, or the [embed-datasets-catalog](../docs/04-data/09-zkai-internal-datasets.md). Records source + date-range + freshness (reproducibility). | read-only | data port |
| **tokenize** | **the heart.** A *tokenizer-spec* → corpus + vocab. Agent proposes field→token-strategy (rule-based per [the raw-data→training-run recipe](../docs/04-data/08-from-raw-data-to-training-run.md)), human approves, Loom codegens the `TokenizerPipeline` subclass, computes `vocab_size`, and **checks C1/C2/C3** (deterministic vocab, grammar, `chunk_size = ctx // (tokens+1)`). | workspace-write | tokenizer |
| **eda** | schema/coverage/freshness sanity + **leakage gate** — flags identity tokens (`CUST_*` sharp edge), target-correlated artifacts, train/val leakage. | read-only | — |
| **pretrain** | **launch-and-track.** Lowers DS intent (`--objective next-event --budget small --metric downstream-fraud-ap`) to NeMo AutoModel (YAML + `torchrun` + FSDP2). Prints cost PLAN; `REFUSED_NO_GPU_TARGET` without a target; real run needs explicit `--launch` + `LOOM_GPU_TARGET`. Checkpoint saved HF-safetensors (C5) with the **tokenizer signature attached** (C1/E2 guardrail). | expensive → irreversible | model-builder |
| **embed** | frozen-backbone embedding extraction, last-token pooling, **row-IDs preserved** (C6). | searchable (cheap) | model-builder |
| **evaluate** | the [multi-task behavioral benchmark (E1)](../docs/05-research/02-improvement-ideas.md#e1): fraud AUPRC/AUROC, next-merchant/next-item Prec@K, amount sMAPE, credit linear-probe, segmentation, few-shot transfer, temporal drift (3/6/12mo). Downstream heads (XGBoost HPO, PCA dim, pooling) are **searchable** — AIDE may tree-search them. VERDICT vs baseline. | searchable | search + model-builder |
| **report** | model card per `--experiment <id>` (runs + metrics + lineage; negative results too). | read-only | — |
| **ops** | drift checks vs a reference (the literature's [unsolved drift problem](../docs/05-research/01-literature-review.md#7-other-findings); rolling-LoRA refresh trigger). | read-only | execution |

**Capability modes are the GPU-safety story:** `searchable` = cheap scalars the AIDE brain may
tree-search; `launch-and-track` = expensive GPU pretrains that are **never** auto-searched. AIDE can
never spawn twenty 8-GPU jobs.

## 4. Architecture — ports & adapters (narrowed)

- **model-builder** *(now the core)* — NeMo AutoModel (default); peer adapters DeepSpeed / FSDP /
  Megatron / Accelerate (roadmap); a **torch-free CPU rehearsal builder** (PPMI+SVD) for dry-running
  the whole flow with zero GPU.
- **execution / MLOps** — Metaflow (lineage, artifacts, runs); `local` for dev.
- **search brain** — AIDE, **scoped to `searchable` tiers only** (downstream heads, pooling, PCA,
  tokenizer-config search *within* contracts). Never pretrains.
- **data** *(new first-class port)* — dataset adapters encoding the data-doc knowledge: BigQuery
  per-chain query templates (EVM/Solana/Stellar/Celo), GCS NDJSON, the embed-datasets-catalog
  (REST + MCP), local Parquet. Knows partition-column cost traps and freshness assertions.
- **LLM provider** — Claude (anthropic) default; own-login subscription options.

## 5. What's genuinely new vs. old Loom (the FM-training depth)

1. **Tokenizer-spec as a first-class, contract-checked artifact**, with agentic field→strategy
   proposal + codegen + vocab accounting. Getting tokenization right *is* the game
   ("no universal tokenization exists" — lit review). This is the centerpiece.
2. **Capability-mode-aware GPU cost governance** (the launch-and-track gate + budget tiers
   probe/small/full quoting real GPU-hours).
3. **Multi-task behavioral eval (E1)** as `evaluate`, replacing single-metric fraud AP.
4. **The experiment backlog as the unit of work** — `--experiment tfm-t1-timedelta` threads runs;
   `report` assembles the comparison and lineage. Loom speaks T/O/A/E/G/D/I.
5. **Tokenizer-signature guardrails (E2)** — persist tokenizer config + vocab hash beside
   checkpoints; assert at load. Closes the silent C1-mismatch failure.

## 6. Proposed v0.1 (TFM-first, smallest useful)

Drive the first-wave experiments **E2 → T1 → T2** on TabFormer (the shipped recipe), using the
repo's existing `src/` (tokenizer pipeline, `clm_data.py`, `scripts/train_decoder_model.py`,
`decoder_inference.py`, notebook-05 eval) **as the backend Loom orchestrates** — those files are the
contract ground-truth, so Loom wraps them rather than reimplementing.

v0.1 verbs: `ingest · tokenize · eda · pretrain(--launch gated) · embed · evaluate · report`, with
`--experiment` threading, contract + leakage guardrails, `local` + Metaflow execution, and a Modal
GPU target. **Reuse from `Loom-legacy/`:** the NeMo adapter, Modal GPU launcher, Metaflow executor
seam, and the learnings/telemetry capture — these are FM-training-relevant and already proven.

**v0.2:** data adapters (D5 internal trade streams via the embed-catalog; D2 chain corpora via
BigQuery templates), the full E1 benchmark, A1 context-length scaling.

## 7. Open questions for review

1. **Primary driver — human or agent?** Is Loom mainly a CLI a data scientist runs, or is it meant
   to be driven by an agent (Claude/Codex) too? (Affects whether the surface is CLI-first or an
   SDK/tool-API the agent calls.) *Lean: both — CLI verbs that double as agent tools, as in the old
   design.*
2. **Wrap vs. reimplement** the repo's `src/` tokenizer + training code for v0.1. *Lean: wrap — the
   contracts live there and Yassine's docs treat `src/` as ground truth.*
3. **How much port generality now?** *Lean: keep model-builder + execution + data + search ports;
   drop the broad lifecycle verbs; scope AIDE to searchable only.*
4. **Reuse vs rewrite** the legacy NeMo/Modal/Metaflow adapters. *Lean: reuse as reference, port
   forward the proven seams.*
