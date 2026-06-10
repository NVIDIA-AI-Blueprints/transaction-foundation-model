# Step-by-Step: Solana (SVM) Data from BigQuery

**Goal:** per-wallet, time-ordered token-movement events from Solana, exported to Parquet for the [tokenizer recipe](08-from-raw-data-to-training-run.md).
**Prereqs:** [the BigQuery primer](03-bigquery-blockchain-primer.md). Solana is where the cost habits stop being optional: this dataset exceeds **a petabyte**.

## 1. The dataset (and its famous quirk)

- **Dataset:** `bigquery-public-data.crypto_solana_mainnet_us` (maintained for the community by BCW via the Rust `solana-etl`; ~2–5 min ingest lag).
- **Tables:** `Accounts`, `Block Rewards`, `Blocks`, `Instructions`, `Token Transfers`, `Tokens`, `Transactions`.
- **The quirk:** table names are **capitalized and contain spaces** — backtick-quote the *whole* path, every time:

```sql
SELECT COUNT(*) FROM `bigquery-public-data.crypto_solana_mainnet_us.Token Transfers`
WHERE block_timestamp >= TIMESTAMP('2026-05-01') AND block_timestamp < TIMESTAMP('2026-05-02');
```

- **Partitioning & enforcement:** large tables are **day-partitioned on `block_timestamp`**, clustered on high-cardinality keys (signatures, accounts), and the biggest **enforce a mandatory partition filter** — an unbounded query is rejected rather than billed. Good: it protects you. The community's measured example of why: one query cost ~3.58 TiB (≈ $90) unpruned vs ≈ $7 with partition+cluster pruning.

## 2. Freshness check

```sql
SELECT MAX(block_timestamp) AS latest
FROM `bigquery-public-data.crypto_solana_mainnet_us.Blocks`
WHERE block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY);
```

(Even the freshness probe needs a partition filter — note the 1-day bound.) This dataset **has stalled before** — 2025-03-31 → 2025-04-06 — so assert, don't assume.

## 3. Understand the SVM data model (30 seconds)

Solana's model differs from EVM in ways that change what "an event" is:

- A **transaction** is a bundle of **instructions**, each targeting a **program** (smart contract). The program ID is the closest analog of EVM's method selector / TabFormer's MCC — "what kind of action".
- Token balances live in **token accounts** (per wallet × mint); the ETL conveniently extracts movement events into **`Token Transfers`** (source, destination, mint, value) so you don't have to decode instructions yourself.
- Throughput is huge (hundreds of millions of transactions *per day*, many of them validator votes) — **never** model "all transactions"; model a cohort.

For a first corpus, **`Token Transfers` is the right table**: it's already event-shaped (who → whom, how much, which asset, when), like a payments ledger.

> **Schema discipline:** column layouts here are community-maintained — print the live schema before writing real queries, and adjust the column names below if they've drifted:
> ```bash
> bq show --schema --format=prettyjson 'bigquery-public-data:crypto_solana_mainnet_us.Token Transfers'
> ```
> Expect fields like `block_timestamp`, `tx_signature`, `source`, `destination`, `mint`, `value`, `decimals`, `transfer_type`.

## 4. Step 1 — Stage a wallet cohort

Same logic as [EVM step 1](04-guide-evm-bigquery.md#4-step-1--stage-a-wallet-cohort): activity-banded, deterministically sampled. **Keep windows short** — Solana's per-day volume is ~two orders of magnitude above Ethereum's; two weeks of Solana ≈ a quarter of Ethereum:

```sql
CREATE OR REPLACE TABLE `YOUR_PROJECT.chain_corpus.sol_cohort` AS
SELECT source AS address, COUNT(*) AS n_sent
FROM `bigquery-public-data.crypto_solana_mainnet_us.Token Transfers`
WHERE block_timestamp >= TIMESTAMP('2026-05-01')
  AND block_timestamp <  TIMESTAMP('2026-05-15')
GROUP BY source
HAVING COUNT(*) BETWEEN 30 AND 3000
   AND MOD(ABS(FARM_FINGERPRINT(source)), 50) = 0;     -- deterministic 2% sample
```

Dry-run first; if the estimate is uncomfortable, shrink the window before shrinking the sample rate (the partition filter is what controls cost).

## 5. Step 2 — Build the event stream

```sql
CREATE OR REPLACE TABLE `YOUR_PROJECT.chain_corpus.sol_events` AS
WITH cohort AS (SELECT address FROM `YOUR_PROJECT.chain_corpus.sol_cohort`)

SELECT
  c.address                                  AS entity,
  'OUT'                                      AS direction,
  t.destination                              AS counterparty,
  SAFE_CAST(t.value AS FLOAT64)
    / POW(10, COALESCE(t.decimals, 0))       AS amount,       -- per-mint decimals, unlike EVM!
  t.mint                                     AS asset,        -- token identifier (the "currency")
  t.transfer_type                            AS event_type,
  t.block_timestamp                          AS ts
FROM `bigquery-public-data.crypto_solana_mainnet_us.Token Transfers` t
JOIN cohort c ON t.source = c.address
WHERE t.block_timestamp >= TIMESTAMP('2026-05-01')
  AND t.block_timestamp <  TIMESTAMP('2026-05-15')

UNION ALL

SELECT
  c.address, 'IN', t.source,
  SAFE_CAST(t.value AS FLOAT64) / POW(10, COALESCE(t.decimals, 0)),
  t.mint, t.transfer_type, t.block_timestamp
FROM `bigquery-public-data.crypto_solana_mainnet_us.Token Transfers` t
JOIN cohort c ON t.destination = c.address
WHERE t.block_timestamp >= TIMESTAMP('2026-05-01')
  AND t.block_timestamp <  TIMESTAMP('2026-05-15');
```

**Optional enrichment — program-call behavior.** What *programs* a wallet touches is a strong behavioral signal (DEX user vs NFT trader vs staker). After a schema check on `Instructions`, a per-day `(signer, program_id, count)` aggregate joined into your events adds an SVM-native field with no EVM analog. Mark it as a v2 enrichment; `Token Transfers` alone trains a perfectly good first model.

## 6. Step 3 — Export

Identical to [EVM step 3](04-guide-evm-bigquery.md#6-step-3--export-to-parquet): `EXPORT DATA` → GCS Parquet → `gcloud storage cp` → `cudf.read_parquet`.

## 7. Mapping to tokens (preview)

| Column | Token (strategy) | Note |
|---|---|---|
| `amount` | `AMT_*` (log bins) | decimals already applied |
| `counterparty` | `CTPY_*` (hash) | beware mega-hubs: DEX pools/CEX hot wallets dominate — consider a `HUB` token for top-K counterparties |
| `asset` (mint) | `TOK_*` (mapping top-K mints + hash tail) | SOL/USDC/USDT cover most volume |
| `event_type` | fixed vocab | transfer / mint / burn |
| `direction` | `DIR_IN`/`DIR_OUT` | |
| `ts` | `HOUR_* DOW_* MONTH_*` + `TDIF_*` | time-deltas matter even more here — Solana behavior is bursty at the seconds scale |

Then: [the universal training recipe](08-from-raw-data-to-training-run.md).

## Solana-specific gotchas, recapped

1. Backticks around table names **with their spaces** — most linters and ORMs choke; raw SQL strings only.
2. Mandatory partition filters on big tables — structurally bound every query (and every CTE branch).
3. Volume: start with **days**, not months; scale the window after you've seen real scan sizes.
4. Decimals are **per-mint** (`POW(10, decimals)`), unlike EVM's uniform 1e18 for native value.
5. Vote/system noise: `Token Transfers` already filters most of it — another reason to prefer it over raw `Transactions` for a first corpus.
6. The dataset has stalled before (2025-03-31 incident) — freshness-check every run.
