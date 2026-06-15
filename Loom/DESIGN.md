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

## 6. Decisions (resolved 2026-06-15)

1. **Driver: both.** CLI verbs that a human runs *and* that double as tools an agent (Claude/Codex)
   can call. Every verb is a typed command with a machine-readable result; the CLI is one front-end,
   the agent-tool schema another, over the same engine.
2. **Engine: clean reimplementation.** Loom gets its **own** engine (tokenizer, training
   orchestration, eval) built from scratch for the general FM-training tool — *not* a wrapper around
   this repo's `src/`. The repo's `src/` and `Loom-legacy/` are **reference**, not backend.
3. **Contract conformance is how we de-risk the rewrite.** Because we're reimplementing, the C1–C6
   contracts could drift. The repo's `src/` is the **conformance oracle**: golden tests assert
   Loom's tokenizer produces the *identical* vocab, vocab-size, and corpus grammar as
   `src/tokenizer/financial_pipeline.py`, and that pretrain's first-step loss ≈ ln(vocab). This *is*
   experiment **E2** (tokenizer-signature guardrails + golden tests) — so the first slice both ships
   value and pins the contracts.
4. **Ports:** keep model-builder (NeMo) + execution (Metaflow) + data + search (AIDE, searchable
   only); drop the broad lifecycle verbs. Mine the legacy NeMo/Modal/Metaflow seams as reference and
   re-implement clean.

## 7. v0.1 build plan (TFM-first, smallest useful)

Target the first-wave experiments **E2 → T1 → T2** on TabFormer. Build sequence, each slice landing
a runnable verb + tests:

1. **Skeleton** — package layout, `pyproject`, the engine/CLI/agent-tool split, the port interfaces
   (empty), result/VERDICT types, `--experiment` threading.
2. **E2 slice (contracts first)** — the tokenizer engine + contract checker (C1–C3) + the **golden
   tests against `src/` as oracle**. Verbs: `tokenize`, `eda` (leakage gate). No GPU.
3. **pretrain (launch-and-track)** — NeMo adapter + cost PLAN + `REFUSED_NO_GPU_TARGET` + `--launch`
   gate + Modal GPU target; tokenizer signature attached to the checkpoint (C5/E2). The torch-free
   CPU rehearsal builder lands here for GPU-free end-to-end dry runs.
4. **embed + evaluate** — frozen-backbone extraction (C6 row-IDs) + a first `evaluate` (fraud AUPRC
   vs baseline; the E1 multi-task battery grows column-by-column from here).
5. **report** — model card per experiment (runs + metrics + lineage).

Proposed layout:

```
Loom/
  pyproject.toml
  loom/
    engine/        # tokenizer, contracts, corpus, eval — the clean reimplementation
    ports/         # model_builder/ (nemo, local-cpu), execution/ (metaflow, local), search/ (aide), data/, llm/
    verbs/         # ingest, tokenize, eda, pretrain, embed, evaluate, report — each = CLI cmd + agent tool
    cli.py         # CLI front-end over verbs
    tools.py       # agent-tool schema over the same verbs
  tests/
    golden/        # conformance oracle vs ../src and ../Loom-legacy
```

**v0.2:** data adapters (D5 internal trade streams via the embed-catalog; D2 chain corpora via
BigQuery templates), the full E1 benchmark, A1 context-length scaling.
