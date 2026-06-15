# Transaction Foundation Model — Documentation

Welcome. These docs teach you — a data scientist who may never have touched a foundation model — everything you need to understand, run, extend, and do research with this repository.

The repo demonstrates one idea, end to end:

> **Treat a customer's transaction history like a sentence, pretrain a small language model on millions of those sentences, and reuse what it learns (as embeddings) to improve downstream tasks like fraud detection.**

You don't need to know what a "decoder-only transformer" or "causal language modeling" is yet. That's what these docs are for.

---

## How these docs are organized

The documentation follows a **level scaffolding** (think university course numbering): each level revisits the same system one layer deeper. Around that core path sit reference and how-to sections you can dip into when needed.

| Section | What it gives you | Read it when |
|---------|-------------------|--------------|
| [01 — Getting Started](01-getting-started/01-orientation.md) | Orientation, environment setup, glossary | First contact with the repo |
| [02 — Concepts](02-concepts/README.md) | Primers on everything you might not know yet (foundation models, tokenization, NeMo, RAPIDS, …) | Whenever a term in the learning path is unfamiliar |
| [03 — Learning Path](03-learning-path/README.md) | **The core curriculum**: Levels 100 → 500, from big picture to line-by-line code anatomy | In order, at your own pace |
| [04 — Data](04-data/README.md) | Dataset catalog + step-by-step guides for adding new data — public blockchain data from BigQuery (EVM, Solana, Stellar, Celo) and ZKAI's internal Embed pipeline datasets | When you want to train on new/your own data |
| [05 — Research](05-research/README.md) | Literature review, concrete improvement ideas, and the sequenced [dataset & training roadmap](05-research/03-dataset-and-training-roadmap.md) | When you move from "using" to "researching" |
| [06 — Experimentation](06-experimentation/README.md) | How we run disciplined experiments with [Loom](https://github.com/ZKAI-Network/loom) | Before you launch your first training run variant |
| [Backlog](backlog/README.md) | Step-by-step execution specs for the near-term roadmap phases (Phase 0, Phase 1, …) | When you're ready to *build* the next phase |

## Recommended paths

**"I just arrived"** (half a day)
1. [Orientation](01-getting-started/01-orientation.md) — what this repo is and what you'll build
2. [Level 100](03-learning-path/level-100-the-big-picture.md) — the big picture
3. [Environment setup](01-getting-started/02-environment-setup.md) — get the notebooks running
4. If you are on macOS/Conductor, use the [GCP GPU notebook runtime](../infra/gcp-notebook/README.md)
5. Run notebook `01_dataset_baseline.ipynb`

**"I want to actually understand it"** (2–3 days)
1. The five [Concepts primers](02-concepts/README.md) that are new to you
2. [Level 200](03-learning-path/level-200-the-building-blocks.md) — the building blocks
3. [Level 300](03-learning-path/level-300-the-pipeline-in-code.md) — the pipeline in code, alongside notebooks 02–05
4. [Level 400](03-learning-path/level-400-design-contracts-and-extensions.md) — contracts, caveats, extension points

**"I'm about to modify the pipeline or port it to new data"** (add half a day)
1. [Level 500](03-learning-path/level-500-the-code-anatomy.md) — the notebook flow as a production DAG, every tokenizer class and method, the GPU parallelization model, and the porting surface

**"I'm here to do research / train on new data"**
1. Everything above, then:
2. [Research section](05-research/README.md) — what the literature says and where this model can improve
3. [Data section](04-data/README.md) — bring in new datasets (e.g., on-chain transaction data)
4. [Dataset & training roadmap](05-research/03-dataset-and-training-roadmap.md) — the sequenced, gated build plan (small-first, blockchain-led, two product lineages)
5. [Backlog](backlog/README.md) — when you're ready to build it: executable Phase 0 / Phase 1 specs
6. [Experimentation with Loom](06-experimentation/01-loom-workflow.md) — run changes as disciplined, tracked experiments

## Documentation conventions

- **Levels build on each other.** A claim explained at Level 100 in one sentence gets a code-level explanation at Level 300. Repetition is intentional — it's how the scaffolding works.
- **Every code claim is linked.** We cite real files like [`src/tokenizer/financial_pipeline.py`](../src/tokenizer/financial_pipeline.py) so you can verify and explore.
- **Data samples are real.** Token examples come from the actual TabFormer schema used by the notebooks.
- **Boxes flag the unfamiliar:**
  > 🧠 **New concept?** Links a primer in [02-concepts](02-concepts/README.md) — read it, then come back.

## The one-paragraph summary (if you read nothing else)

Raw credit-card transactions (amount, merchant, time, location, …) are converted by a deterministic, GPU-accelerated tokenizer into short "sentences" of ~12 domain tokens each (`AMT_1 MERCH_667 CAT_RETAIL …`). Per-customer histories of ~315 transactions become sequences of ~4,096 tokens. A small (~29M parameter) Llama-style decoder is pretrained on ~64K such sequences with next-token prediction — no fraud labels involved. The trained model then converts any transaction history into a 512-dimensional embedding vector. XGBoost models using **raw features + these embeddings** beat raw-feature-only baselines on fraud detection, especially in average precision — the metric that matters when fraud is 1-in-1000. The same recipe applies to any event-sequence data: card payments, bank transfers, or on-chain transactions.
