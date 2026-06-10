# BigQuery Blockchain Primer: Setup, Costs, and the Dataset Landscape

**Read this before any chain-specific guide.** It gets you from zero to safely querying petabyte-scale public blockchain datasets, without an accidental $90 query.

> Facts below (IDs, prices, partition columns) verified against primary sources **June 2026**; each chain guide includes live re-verification commands. 🧠 New to BigQuery entirely? It's Google Cloud's serverless SQL warehouse: you write SQL in a browser or CLI against tables Google hosts; you pay (only) for the bytes your queries scan.

## 1. Why BigQuery for chain data

Full blockchain histories are awkward to self-host (an Ethereum archive node is days of sync and terabytes; Solana is petabytes). Google and several foundations maintain **public, queryable copies** of major chains' full histories in BigQuery, updated continuously. For corpus-building — "give me 18 months of ordered events for N million accounts" — a SQL warehouse is exactly the right tool, and the data is free to store (the host pays storage; you pay queries).

## 2. Account setup (10 minutes)

**Option A — Sandbox (no credit card):** go to [console.cloud.google.com/bigquery](https://console.cloud.google.com/bigquery), sign in, create a project. The sandbox gives the same free quotas as a billed account (below) with one big caveat: **anything you create — tables, saved results — auto-expires after 60 days**. Fine for exploration; not for corpus storage.

**Option B — Billing enabled (recommended for real work):** attach a billing account. You keep the free monthly tier; you can persist tables, use `EXPORT DATA`, and raise quotas.

**CLI (used throughout the guides):**

```bash
# Install the Google Cloud SDK, then:
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
bq query --use_legacy_sql=false 'SELECT 1'   # smoke test
```

## 3. The cost model (and how to never get surprised)

| Fact | Value (US multi-region, on-demand) |
|---|---|
| Query price | **$6.25 / TiB scanned** |
| Free query tier | **first 1 TiB / month** |
| Free storage | 10 GiB (your own tables) |
| Cached re-runs | free |
| Failed queries | not billed |
| Minimum per query | 10 MB per table referenced |

The catch: these datasets are *huge*. Ethereum's transactions table alone is multi-TB; a careless `SELECT *` on Solana has measured at ~3.6 TiB (≈ $90) for one query. Your 1 TiB free tier evaporates fast. The defense is mechanical — make these five habits reflexive:

1. **Always filter on the partition column.** Every large chain table is time-partitioned; a partition filter prunes the bytes *before* scanning. The column differs by dataset (the #1 gotcha):

   | Dataset family | Partition column | Granularity |
   |---|---|---|
   | `crypto_ethereum` (legacy/community) | `block_timestamp` | DAY |
   | `goog_blockchain_*` (Google-managed) | `block_timestamp` | **MONTH** |
   | `crypto_bitcoin` | **`block_timestamp_month`** (a separate column!) | MONTH |
   | `crypto_solana_mainnet_us` | `block_timestamp` | DAY (+ filter **mandatory** on big tables) |
   | Stellar `crypto_stellar` history tables | **`batch_run_date`** | DAY |

   Bitcoin example of why this matters: filtering `block_timestamp_month = '2020-05-01'` scans ~161 MiB; the same query filtered only on `block_timestamp` scans ~412 GiB.

2. **Never `SELECT *`.** BigQuery is columnar — you pay per column read. Name the fields.
3. **Dry-run first.** The console shows "This query will process X" top-right before you run; in CLI:
   ```bash
   bq query --use_legacy_sql=false --dry_run 'SELECT from_address FROM `bigquery-public-data.crypto_ethereum.transactions` WHERE DATE(block_timestamp) = "2026-05-01"'
   ```
4. **Set a hard ceiling.** `--maximum_bytes_billed=200000000000` (200 GB) makes over-budget queries *fail* instead of bill. You can also set per-user daily quotas project-wide.
5. **Stage, then iterate.** First query writes a filtered slice into *your* dataset (`CREATE TABLE my_ds.eth_slice AS SELECT …`); all iteration happens on the small copy.

## 4. The landscape: what's queryable, as of June 2026

**Google-managed datasets** (project `bigquery-public-data`, dataset pattern `goog_blockchain_<chain>_<network>_us`) — uniform schema across chains: `blocks`, `transactions`, `receipts`, `logs`, `decoded_events` (+ on Ethereum: `traces`, `token_transfers`, `accounts`, `accounts_state`). MONTHLY partitions on `block_timestamp`; addresses lowercase. Ethereum is GA; **the rest are Preview (no SLA)**:

| Chain | Dataset ID |
|---|---|
| Ethereum | `goog_blockchain_ethereum_mainnet_us` |
| Polygon | `goog_blockchain_polygon_mainnet_us` |
| Arbitrum One | `goog_blockchain_arbitrum_one_us` |
| Optimism | `goog_blockchain_optimism_mainnet_us` |
| Avalanche C-Chain | `goog_blockchain_avalanche_contract_chain_us` |
| Fantom Opera | `goog_blockchain_fantom_opera_us` |
| Tron | `goog_blockchain_tron_mainnet_us` |
| Cronos | `goog_blockchain_cronos_mainnet_us` *(ID pattern-inferred — verify with `bq show`)* |

**Community datasets (Blockchain ETL lineage)** — older, battle-tested, slightly different schema (receipts denormalized into `transactions`; DAY partitions):

| Chain | Dataset ID |
|---|---|
| Ethereum | `bigquery-public-data.crypto_ethereum` |
| Bitcoin | `bigquery-public-data.crypto_bitcoin` (also Litecoin, Dogecoin, Bitcoin Cash, Dash, Zcash as `crypto_<name>`) |
| Ethereum Classic | `bigquery-public-data.crypto_ethereum_classic` |
| Polygon (legacy) | `public-data-finance.crypto_polygon` |
| Celo | `nansen-public-data.crypto_celo` — **see the [Celo guide](07-guide-celo-and-other-chains.md) for important caveats** |
| Tezos, Zilliqa, IoTeX, Theta, Band, Beacon chain | `public-data-finance.crypto_*` |

**Foundation/other-party datasets:**

| Chain | Dataset ID | Maintainer |
|---|---|---|
| **Solana** | `bigquery-public-data.crypto_solana_mainnet_us` | BCW for the community — [guide 05](05-guide-solana-bigquery.md) |
| **Stellar** | `crypto-stellar.crypto_stellar` (+ `crypto_stellar_dbt`) | Stellar Development Foundation — [guide 06](06-guide-stellar-hubble.md) |
| NEAR | `bigquery-public-data.crypto_near_mainnet_us` | NEAR Foundation |
| Polkadot | `bigquery-public-data.crypto_polkadot`; `substrate-etl` (Polkaholic) | community |
| Cardano | `iog-data-analytics.cardano_mainnet` | IOG |
| XRP Ledger | `xrpledgerdata.fullhistory` | community |
| MultiversX | `bigquery-public-data.crypto_multiversx_mainnet_eu` (**EU region!**) | foundation |

**Not in BigQuery** (as of June 2026): **Base** and **zkSync Era** have no public BigQuery datasets (zkSync has an academic Parquet dump, [arXiv:2407.18699](https://arxiv.org/abs/2407.18699)); for these, use Dune/Goldsky/indexers or run your own ETL.

## 5. Universal gotchas (each guide repeats its own)

1. **Datasets stall silently.** Solana halted 2025-03-31→04-06; Polygon's Google-managed dataset stopped updating around 2025-03-08 for a period. **Start every corpus build with a freshness assertion:**
   ```sql
   SELECT MAX(block_timestamp) FROM `bigquery-public-data.crypto_ethereum.transactions`;
   ```
2. **Two Ethereums.** Legacy `crypto_ethereum` (DAY partitions, receipts inside `transactions`) vs Google-managed (`goog_…`, MONTHLY partitions, separate `receipts`, plus `decoded_events`). Pick one per project and note it in your corpus metadata.
3. **Solana table names contain spaces and capitals** — `` `…crypto_solana_mainnet_us.Token Transfers` `` — backticks mandatory.
4. **Region matters.** Everything listed is US multi-region except MultiversX (EU). Cross-region joins fail; keep your own working dataset in `US`.
5. **Preview ≠ GA.** Most Google-managed chains are Pre-GA: no SLA, schemas can change.
6. **Numbers are big.** Wei values exceed INT64; amount columns are NUMERIC/BIGNUMERIC or strings. Convert with care (`SAFE_CAST`, divide by 1e18 *after* casting to FLOAT64/NUMERIC).
7. **Sandbox artifacts expire in 60 days** — export corpora before they evaporate.

## 6. Getting query results out (for the tokenizer)

Three patterns, in increasing scale:

```python
# A. Small (≤ a few GB): straight to pandas
import pandas_gbq
df = pandas_gbq.read_gbq(QUERY, project_id=PROJECT)
```

```sql
-- B. Medium/large: EXPORT DATA → GCS as Parquet (then gsutil/gcloud storage cp)
EXPORT DATA OPTIONS(
  uri='gs://YOUR_BUCKET/eth_corpus/part-*.parquet',
  format='PARQUET', overwrite=true
) AS
SELECT ...;
```

```bash
# C. Existing staged table → GCS
bq extract --destination_format=PARQUET my_ds.eth_slice 'gs://YOUR_BUCKET/eth_slice/part-*.parquet'
```

Parquet is the right interchange format — cuDF reads it natively (`cudf.read_parquet`), exactly like the TabFormer splits in notebook 02.

**Next:** pick your chain — [EVM](04-guide-evm-bigquery.md) · [Solana](05-guide-solana-bigquery.md) · [Stellar](06-guide-stellar-hubble.md) · [Celo & others](07-guide-celo-and-other-chains.md) — then [turn the export into a training run](08-from-raw-data-to-training-run.md).
