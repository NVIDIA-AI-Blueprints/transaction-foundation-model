# Data: Datasets and How to Feed Them In

The model is only as interesting as its corpus. This section covers **what data exists** (the included TabFormer dataset, other public research datasets, public blockchain data, and [ZKAI's internal Embed pipeline datasets](09-zkai-internal-datasets.md)) and **exactly how to turn any of it into a training run** for this repo's pipeline.

## Section map

| Page | What it covers |
|------|----------------|
| [01 — TabFormer](01-tabformer.md) | The included dataset: schema, quirks, download |
| [02 — Public datasets catalog](02-public-datasets-catalog.md) | MBD, PersonaLedger, Elliptic 1/2, and friends — what each is good for |
| [03 — BigQuery blockchain primer](03-bigquery-blockchain-primer.md) | **Start here for on-chain data**: account setup, the free tier, cost control, the dataset landscape, the gotchas |
| [04 — EVM chains guide](04-guide-evm-bigquery.md) | Step-by-step: Ethereum + Polygon/Arbitrum/Optimism/Avalanche/… → wallet-sequence corpus |
| [05 — Solana guide](05-guide-solana-bigquery.md) | Step-by-step for the SVM world (and its petabyte-scale sharp edges) |
| [06 — Stellar guide](06-guide-stellar-hubble.md) | Step-by-step for SDF's Hubble dataset — the most payment-like chain data |
| [07 — Celo & other chains](07-guide-celo-and-other-chains.md) | Celo's special situation post-L2 migration; NEAR, Polkadot, Cardano, Bitcoin-family, XRP |
| [08 — From raw data to training run](08-from-raw-data-to-training-run.md) | **The universal recipe**: schema mapping → tokenizer steps → corpus → config → train → evaluate |
| [09 — ZKAI internal datasets](09-zkai-internal-datasets.md) | 🔒 Our Embed pipeline catalog: DEX/Hyperliquid/Polymarket/Kalshi trade streams + the cross-chain mart — **the likely starting point for next-trade prediction work** |

## Why blockchain data, specifically?

Our research program ([literature review](../05-research/01-literature-review.md)) identifies on-chain data as the strategic complement to card/bank data:

1. **It's the only large-scale, *real*, *public* transaction data.** Bank data is locked behind privacy law (the industry's entire moat is data access); synthetic data is synthetic. Chains publish every transaction, forever, for free.
2. **It has true counterparty structure.** TabFormer has users and merchants; chains have `from_address → to_address` — an explicit graph. That's exactly what the literature says sequence models are missing (PRAGMA's −47% AML failure; RiskSEA's graph-fusion gains).
3. **It has organic labels.** Phishing lists, sanctioned addresses, exploit post-mortems, Elliptic's annotations — real adversarial behavior, not simulated.
4. **It's an open benchmark substrate.** Anyone can reproduce a corpus from a public SQL query — which makes results publishable and comparable (cf. EWE-1, the first open-weights blockchain FM).
5. **The transfer question is open.** *No published blockchain↔fiat transfer study exists* — testing whether chain-pretrained representations help card fraud (and vice versa) is [our flagged original-contribution opportunity](../05-research/02-improvement-ideas.md#d3--the-blockchainfiat-transfer-study-the-original-contribution-bet).

## The shape of every data journey here

Whatever the source, the path into the model is always the same five artifacts (details: [guide 08](08-from-raw-data-to-training-run.md)):

```
source data ──► entity-ordered event table ──► tokenizer pipeline ──► corpus .txt ──► YAML + train ──► checkpoint
   (SQL/files)    (one row per event,            (a subclass of          (<bos> … <eos>    (vocab_size       (+ embeddings,
                   sorted by entity, time)        TokenizerPipeline)      lines)            updated!)         evaluation)
```

Keep [Level 400's contracts](../03-learning-path/level-400-design-contracts-and-extensions.md) open while you work — every new dataset renegotiates the vocabulary contract (C1), and the temporal-split discipline (C6) applies to chains exactly as to cards.

> **Freshness note.** The BigQuery facts in these guides (dataset IDs, schemas, prices, partition columns) were verified against primary sources in **June 2026**, and each guide includes the one-liner to re-verify live. Public datasets do silently stall (it has happened to Solana and Polygon) — always check `MAX(block_timestamp)` before building a corpus.
