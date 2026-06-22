# Celo (a Special Case) & Other Chains

## Part 1 — Celo

### Why Celo is on our shortlist at all

Celo is a **mobile-first payments chain**: stablecoin-denominated transfers (cUSD/cEUR/cREAL), sub-cent fees, large real-payment volume in emerging markets (MiniPay et al.). Behaviorally, its ledger is closer to *consumer payments* than almost any other chain — squarely relevant to our transaction-modeling and alternative-credit research threads (see `wiki/credit-scoring.md` and the Tala strategy work in the [research KB](https://github.com/ZKAI-Network/research)).

### The catch: Celo changed what it *is*

On **2025-03-26, at block 31,056,500**, Celo migrated from a standalone L1 to an **OP-Stack Ethereum L2** ([migration notice](https://docs.celo.org/cel2/notices/l2-migration)). Data-wise that means:

- **Schema:** still EVM — everything in the [EVM guide](04-guide-evm-bigquery.md) applies conceptually (addresses, transactions, token transfers; stablecoin payments are ERC-20 transfers of the cUSD/cEUR contracts).
- **Pipelines:** the historical BigQuery ETL was built for the *L1* client, so its post-migration continuation is in doubt (verified June 2026: no evidence either way — check live, below).

### The BigQuery situation, verified honestly (June 2026)

| Claim you might encounter | Status |
|---|---|
| `goog_blockchain_celo_mainnet_us` | **Does not exist.** Celo is not in Google's managed-dataset list. |
| `bigquery-public-data.crypto_celo` | Does not exist under that project. |
| **`nansen-public-data.crypto_celo`** | **Real** — the community dataset (Nansen-maintained [celo-etl](https://github.com/nansen-ai/celo-etl)), standard ethereum-etl schema (`blocks`, `transactions`, `token_transfers`, `logs`, …). **Post-L2-migration freshness unverified — assume possibly frozen at block 31,056,500 until you check.** |
| A BigQuery page on docs.celo.org | None today; Celo's docs point to indexers (The Graph, SubQuery, Envio, Goldrush) and Dune. |

**First command of any Celo work:**

```sql
SELECT MAX(block_timestamp) AS latest, MAX(number) AS latest_block
FROM `nansen-public-data.crypto_celo.blocks`;
-- latest_block ≥ 31,056,500 and a recent timestamp ⇒ post-migration data exists.
-- Otherwise the dataset is frozen at the L1 era.
```

### Three workable paths

1. **Pre-migration history corpus (available now).** Years of L1 Celo (2020 → 2025-03) in `nansen-public-data.crypto_celo` — a *complete, immutable* dataset of real stablecoin payments. For pretraining, frozen history is not a defect; mind only that your [temporal-split discipline](../03-learning-path/level-400-design-contracts-and-extensions.md#1-the-contracts) keeps evaluation inside the covered era. Use the [EVM guide's queries](04-guide-evm-bigquery.md) with the dataset ID swapped, with one Celo twist: **payments are mostly `token_transfers` rows** (cUSD `0x765de816845861e75a25fca122bb6898b8b1282a`, cEUR `0x10c892a6ec43a53e45d0b916b4b7d383b1b78c0f`), not native-value `transactions`.
2. **Post-migration via indexers (current data, not BigQuery).** Dune (`celo` tables), Goldsky/Envio/SubQuery streams, or Blockscout's API/DB dumps. Export to Parquet and rejoin the [universal recipe](08-from-raw-data-to-training-run.md) — only the extraction layer changes.
3. **Self-ETL (full control).** Celo-as-OP-Stack means standard Ethereum tooling works: an archive node + [cryo](https://github.com/paradigmxyz/cryo) or ethereum-etl dumps straight to Parquet. The heaviest option; justified once Celo becomes a *primary* corpus rather than an experiment.

**Recommendation:** start with path 1 (it's one dataset-ID swap away from the EVM guide and entirely sufficient for pretraining + the transfer study); adopt path 2 only when your downstream task needs post-2025 behavior.

---

## Part 2 — Other chains with public BigQuery data

Quick-reference for the rest of the verified landscape ([primer §4](03-bigquery-blockchain-primer.md#4-the-landscape-whats-queryable-as-of-june-2026) has the full table). For each: the dataset, the data-model gotcha, and the corpus hint.

### NEAR
`bigquery-public-data.crypto_near_mainnet_us` — foundation-maintained, near-real-time ([docs](https://docs.near.org/data-infrastructure/big-query)). Tables: `blocks`, `chunks`, `transactions`, `execution_outcomes`, `receipt_details`, `receipt_origin`, `receipt_actions`, `account_changes`. **Gotcha:** NEAR's async runtime means a "transaction" fans out into *receipts*; model **`receipt_actions`** (action kind ≈ the MCC analog) rather than transactions. Human-readable account IDs (`alice.near`) hash like any categorical.

### Polkadot / Substrate
`bigquery-public-data.crypto_polkadot` (tables suffixed by parachain ID: `blocks0`, `transfers0`, …) plus the richer `substrate-etl` project (Polkaholic), e.g. `polkadot.xcmtransfers` day-partitioned on `origination_ts`. **Gotcha:** per-parachain table sharding; inventory with `bq ls` first. Corpus hint: `transfers0` (relay-chain balance transfers) is event-shaped; XCM transfers add a cross-chain dimension no other dataset here has.

### Bitcoin family (UTXO chains)
`bigquery-public-data.crypto_bitcoin` (+ `crypto_litecoin`, `crypto_dogecoin`, `crypto_bitcoin_cash`, `crypto_dash`, `crypto_zcash`). **Gotchas:** partition column is **`block_timestamp_month`** (filtering `block_timestamp` alone scans everything!), and the **UTXO model has no stable account** — "entity" requires address-clustering heuristics (or accept address-level sequences, knowing wallets rotate addresses). That's why account-model chains are better first corpora; Bitcoin matters mainly because [Elliptic's labels](02-public-datasets-catalog.md#elliptic-1--2--bitcoin-illicit-activity-graphs) live there. Tables: `blocks`, `transactions` (with nested `inputs`/`outputs` arrays — `UNNEST` required).

### Cardano
`iog-data-analytics.cardano_mainnet` — IOG-maintained, db-sync relational model, ~2-hour cadence, beta. UTXO (eUTXO) — same entity caveat as Bitcoin.

### XRP Ledger
`xrpledgerdata.fullhistory` — community full history. Account-based and payments-heavy (it's a payments network), so structurally friendly; verify maintenance status before depending on it.

### MultiversX
`bigquery-public-data.crypto_multiversx_mainnet_eu` — **EU region**: you can't join it with the US datasets, and your billing/processing location follows the data.

### Not in BigQuery (June 2026)
**Base**, **zkSync Era** (academic Parquet dump only: [arXiv:2407.18699](https://arxiv.org/abs/2407.18699)). Use Dune/Goldsky/indexers or self-ETL.

## Choosing, in one table

| You want… | Pick |
|---|---|
| The canonical first chain corpus | [Ethereum](04-guide-evm-bigquery.md) |
| Payments-shaped, multi-currency, gentle | [Stellar](06-guide-stellar-hubble.md) |
| Scale + retail bursts + SVM diversity | [Solana](05-guide-solana-bigquery.md) |
| Emerging-market stablecoin payments | Celo (path 1 above) |
| Labeled AML graphs | Bitcoin + Elliptic |
| Architecture-diverse generalization test | + NEAR or Polkadot |

Whatever you choose: [from raw data to training run](08-from-raw-data-to-training-run.md) is the same next step.
