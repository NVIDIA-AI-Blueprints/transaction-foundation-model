# Public Datasets Catalog

Datasets our [research KB](https://github.com/ZKAI-Network/research) has vetted as relevant for transaction-foundation-model work, beyond [TabFormer](01-tabformer.md). For each: what it is, what it's uniquely good for, and the integration path into this repo. (Blockchain sources get their own [dedicated guides](03-bigquery-blockchain-primer.md).)

---

## MBD — Multimodal Banking Dataset

*The scale upgrade.* ([arXiv:2409.17587](https://arxiv.org/abs/2409.17587), KDD 2025; on HuggingFace: `ai-lab/MBD` / detached `MBD-mini`)

| | |
|---|---|
| Scale | **~950M transactions**, ~1B geo events, ~5M dialog events; ~2.2M corporate clients; ~2 years |
| Nature | **Real** bank data, industrially anonymized (hashed categoricals, quantized amounts, shifted time) |
| Labels | 4 bank-product purchase-propensity targets (multi-label, temporal) |
| License/access | public research download via HuggingFace |

**Good for:** the 40× data-scaling experiment ([idea D1](../05-research/02-improvement-ideas.md#d--data-each-feeds-a2-how-tos-live-in-the-data-section)); multi-task evaluation (its benchmark ships with temporal protocol); multimodal fusion research (transactions + geo + dialog embeddings).

**Integration notes:** events arrive as `(client_id, event_time, event_type, amount, …)` with anonymized categorical codes — the fields map almost 1:1 onto our [universal recipe](08-from-raw-data-to-training-run.md): hash/fixed-vocab the codes, bin the amounts, calendar-encode the times. Its size demands the [streaming-loader idea I1](../05-research/02-improvement-ideas.md#i--infrastructure) before full-corpus training; start with `MBD-mini`.

## PersonaLedger

*The controllable synthetic.* (Capital One, ICLR 2026, [arXiv:2601.03149](https://arxiv.org/abs/2601.03149); HuggingFace: `capitalone/PersonaLedger`)

| | |
|---|---|
| Scale | ~30M transactions, ~23K LLM-generated personas |
| Nature | synthetic, but **rule-grounded** — generation rules double as ground truth |
| Labels | benchmark tasks derived from the generating rules |

**Good for:** controlled ablations (you *know* the causal structure, so you can test whether the model recovers it); probing what pretraining objectives capture; cheap iteration before burning GPU-hours on MBD-scale runs.

## Elliptic 1 & 2 — Bitcoin illicit-activity graphs

*The labeled adversarial graphs.* (Elliptic1: [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set); Elliptic2: [arXiv:2404.19109](https://arxiv.org/abs/2404.19109), KDD 2024)

| | | |
|---|---|---|
| Elliptic1 | 204K tx-nodes, 234K edges | node-level licit/illicit labels |
| Elliptic2 | 49M nodes, 196M edges, 122K labeled **subgraphs** | money-laundering *patterns* (peeling chains, smurfing) |

**Good for:** evaluating the relational ideas ([G1/G2](../05-research/02-improvement-ideas.md#g--graph--relational-context)) on ground truth; AML-style tasks our sequence model is *expected* to fail at (per PRAGMA's −47% lesson) — an honest stress test. Note these are graph datasets with engineered node features, not raw event sequences; they complement rather than replace a chain corpus.

## EWE-1 artifacts — open-weights Ethereum FM

*(sistemalabs, 2026 — [GitHub](https://github.com/0xideas/ewe-1-inference), [HuggingFace](https://huggingface.co/sistemalabs))*

Not a dataset but the first **open-weights** blockchain foundation model (35M/110M/500M params, pretrained on 1.1B Ethereum transactions). Useful to us as: a baseline to compare any model we pretrain on [EVM data](04-guide-evm-bigquery.md); a reference for feature/vocabulary design (31 features, 64-txn lookback, monthly vocab refresh); and an embedding source to benchmark our own embeddings against on shared tasks.

## Event-sequence benchmark suites

For when we adopt the [multi-task evaluation protocol](../05-research/02-improvement-ideas.md#e1--build-the-multi-task-behavioral-benchmark):

- **EBES** ([arXiv:2410.03399](https://arxiv.org/abs/2410.03399), KDD 2025) — standardized event-sequence benchmark with temporal protocols.
- **HORIZON** ([arXiv:2604.17259](https://arxiv.org/abs/2604.17259)) — 54M users / 486M interactions; behavioral-FM benchmark.
- **PyTorch-Lifestream** (IJCAI 2025) — tooling ecosystem with CoLES-style baselines worth reusing.

*(None include graph structure — the gap our chain corpora can fill.)*

## Quick chooser

| You want to… | Use |
|---|---|
| Learn the pipeline / reproduce the blueprint | [TabFormer](01-tabformer.md) |
| Scale pretraining on real payments data | **MBD** |
| Controlled ablations with known ground truth | PersonaLedger |
| Real counterparty graphs + organic adversarial labels | [Chain data via BigQuery](03-bigquery-blockchain-primer.md) (+ Elliptic for labeled AML) |
| A published model to benchmark against | EWE-1 |
| Standardized multi-task eval | EBES / HORIZON / MBD benchmark |

Whatever you pick, the integration path is the same: [from raw data to training run](08-from-raw-data-to-training-run.md).
