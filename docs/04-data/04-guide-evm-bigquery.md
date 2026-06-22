# Step-by-Step: EVM Chain Data from BigQuery (Ethereum → any EVM chain)

**Goal:** a Parquet export of per-wallet, time-ordered transaction events from an EVM chain, ready for the [tokenizer recipe](08-from-raw-data-to-training-run.md).
**Prereqs:** [the BigQuery primer](03-bigquery-blockchain-primer.md) (account, cost habits).
**Budget:** the full walkthrough below stays in the low-hundreds of GB scanned — well inside the free monthly TiB if you keep the time windows as written.

EVM = Ethereum and its schema-identical family (Polygon, Arbitrum, Optimism, Avalanche C-chain, Fantom, Cronos, plus Celo with [caveats](07-guide-celo-and-other-chains.md)). Learn the recipe on Ethereum once; every other EVM chain is a dataset-ID swap (§7).

## 1. Choose your Ethereum dataset

| | Legacy `bigquery-public-data.crypto_ethereum` | Google-managed `…goog_blockchain_ethereum_mainnet_us` |
|---|---|---|
| Partitioning | **DAY** on `block_timestamp` → finer cost pruning | MONTH on `block_timestamp` |
| Receipts | denormalized into `transactions` (`receipt_status`) | separate `receipts` table |
| Extras | `token_transfers`, `traces`, `contracts`, `tokens`, `balances` | `decoded_events`, `accounts_state` |
| Maturity | community-maintained since 2018 | GA, uniform schema across chains |

**For corpus work, this guide uses the legacy dataset** (day partitions + `token_transfers` + single-table status). The Google-managed one is the better choice when you need decoded contract events.

## 2. Freshness check (always, before anything)

```sql
SELECT MAX(block_timestamp) AS latest
FROM `bigquery-public-data.crypto_ethereum.transactions`;
```

Expect within minutes–hours of now (documented lag ≈ 4 min). If it's days stale, stop — datasets do stall ([primer §5](03-bigquery-blockchain-primer.md#5-universal-gotchas-each-guide-repeats-its-own)).

## 3. Know the table you'll live in

`crypto_ethereum.transactions` — one row per native transaction; the columns that matter for sequence modeling:

| Column | Type | Use for us |
|---|---|---|
| `from_address`, `to_address` | STRING (hex) | **entity & counterparty** — the explicit graph TabFormer lacks |
| `value` | NUMERIC (wei) | amount (÷1e18 = ETH) |
| `block_timestamp` | TIMESTAMP | event time **and partition column** |
| `input` | STRING (hex calldata) | `SUBSTR(input,1,10)` = 4-byte **method selector** ≈ "what action" (the MCC analog) |
| `receipt_status` | INT64 | 1 = success, 0 = reverted |
| `gas_price`, `receipt_gas_used` | INT64 | fee behavior (urgency signal) |
| `nonce` | INT64 | sender's lifetime tx index (a free per-entity sequence check) |

ERC-20/721 movements live separately in `token_transfers` (`token_address`, `from_address`, `to_address`, `value` STRING — uint256 exceeds INT64).

## 4. Step 1 — Stage a wallet cohort

Sequence modeling wants entities with *histories*, so we pick wallets by activity band. Deterministic sampling (`FARM_FINGERPRINT`, not `RAND()`) makes the corpus reproducible — cite the query + window and anyone can rebuild it.

```sql
-- ≈ scans only from_address over 3 months of partitions
CREATE SCHEMA IF NOT EXISTS `YOUR_PROJECT.chain_corpus`;

CREATE OR REPLACE TABLE `YOUR_PROJECT.chain_corpus.eth_cohort` AS
SELECT from_address AS address, COUNT(*) AS n_sent
FROM `bigquery-public-data.crypto_ethereum.transactions`
WHERE block_timestamp >= TIMESTAMP('2026-01-01')
  AND block_timestamp <  TIMESTAMP('2026-04-01')
GROUP BY from_address
HAVING COUNT(*) BETWEEN 50 AND 5000          -- active humans/contracts, not bots/exchanges
   AND MOD(ABS(FARM_FINGERPRINT(from_address)), 20) = 0;   -- deterministic 5% sample
```

Tune the `HAVING` band to your question: 50–5,000 sent txns/quarter ≈ "real users and small contracts"; the >5,000 tail is exchanges, routers, MEV bots — interesting, but they'll dominate token statistics if you let them.

## 5. Step 2 — Build the event stream (both directions)

A wallet's behavior = what it *sends* and what it *receives*. Union both, tagged with `direction` — this is the `direction` field from the [universal schema](../05-research/01-literature-review.md#3-tokenization--schema-findings-the-part-most-transferable-to-us):

```sql
CREATE OR REPLACE TABLE `YOUR_PROJECT.chain_corpus.eth_events` AS
WITH cohort AS (SELECT address FROM `YOUR_PROJECT.chain_corpus.eth_cohort`)

SELECT
  c.address                                   AS entity,
  'OUT'                                       AS direction,
  t.to_address                                AS counterparty,
  SAFE_CAST(t.value AS FLOAT64) / 1e18        AS amount_native,   -- fine for log-binning; NOT for accounting
  SUBSTR(t.input, 1, 10)                      AS method_id,       -- '0x' = plain transfer
  t.receipt_status                            AS status,
  t.gas_price                                 AS gas_price,
  t.block_timestamp                           AS ts
FROM `bigquery-public-data.crypto_ethereum.transactions` t
JOIN cohort c ON t.from_address = c.address
WHERE t.block_timestamp >= TIMESTAMP('2026-01-01')
  AND t.block_timestamp <  TIMESTAMP('2026-04-01')

UNION ALL

SELECT
  c.address, 'IN', t.from_address,
  SAFE_CAST(t.value AS FLOAT64) / 1e18,
  SUBSTR(t.input, 1, 10), t.receipt_status, t.gas_price, t.block_timestamp
FROM `bigquery-public-data.crypto_ethereum.transactions` t
JOIN cohort c ON t.to_address = c.address
WHERE t.block_timestamp >= TIMESTAMP('2026-01-01')
  AND t.block_timestamp <  TIMESTAMP('2026-04-01');
```

Notes that save real money/pain:

- **Both subqueries keep the partition filter** — the join does *not* prune partitions for you.
- No `ORDER BY` here: ordering giant results is expensive, and the [repo's `preprocess()` sorts by entity/time anyway](../03-learning-path/level-300-the-pipeline-in-code.md#stage-a--preprocessing-raw-columns--pipeline-ready-columns) (cuDF, free).
- Optional third branch: ERC-20 movements from `token_transfers` (add `token_address` as an `asset` column; `value` needs `SAFE_CAST(value AS FLOAT64)` and per-token decimals from the `tokens` table — or skip decimals and log-bin the raw integer, which is what we do for a first corpus).

Sanity-check the result:

```sql
SELECT COUNT(*) n_events, COUNT(DISTINCT entity) n_entities,
       APPROX_QUANTILES(cnt, 4) events_per_entity_quartiles
FROM (SELECT entity, COUNT(*) cnt FROM `YOUR_PROJECT.chain_corpus.eth_events` GROUP BY entity), UNNEST([cnt]) cnt;
```

## 6. Step 3 — Export to Parquet

```sql
EXPORT DATA OPTIONS(
  uri = 'gs://YOUR_BUCKET/eth_corpus_2026q1/part-*.parquet',
  format = 'PARQUET', overwrite = true
) AS
SELECT * FROM `YOUR_PROJECT.chain_corpus.eth_events`;
```

```bash
gcloud storage cp -r gs://YOUR_BUCKET/eth_corpus_2026q1 ./data/chain/eth/
```

From here it's `cudf.read_parquet("data/chain/eth/*.parquet")` and the [universal tokenizer recipe](08-from-raw-data-to-training-run.md), whose worked example is exactly this export. Preview of the field mapping:

| Export column | Token (strategy) | TabFormer analog |
|---|---|---|
| `amount_native` | `AMT_*` (log bins) | Amount |
| `counterparty` | `CTPY_*` (hash, 2–10K buckets) | Merchant Name |
| `method_id` | `METHOD_*` (mapping over top selectors + default) | MCC |
| `direction` | `DIR_IN`/`DIR_OUT` (fixed) | — (new!) |
| `status` | `OK`/`REVERT` (fixed) | — (new!) |
| `gas_price` | `GAS_*` (quantile bins) | — (new!) |
| `ts` | `HOUR_* DOW_* MONTH_*` + `TDIF_*` | Time fields |

## 7. Any other EVM chain = swap the dataset ID

Same query, two edits — the dataset ID, and (for Google-managed datasets) remember partitions are **MONTHLY**, so align windows to month boundaries:

```sql
-- e.g. Polygon
FROM `bigquery-public-data.goog_blockchain_polygon_mainnet_us.transactions` t
WHERE t.block_timestamp >= TIMESTAMP('2026-01-01')   -- month-aligned
  AND t.block_timestamp <  TIMESTAMP('2026-04-01')
```

Schema deltas vs legacy: `receipt_status` lives in the separate `receipts` table (join on `transaction_hash`, or skip status for a first corpus); addresses are already lowercase. Dataset IDs for Arbitrum/Optimism/Avalanche/Fantom/Tron/Cronos: [primer §4](03-bigquery-blockchain-primer.md#4-the-landscape-whats-queryable-as-of-june-2026).

**Multi-chain corpora:** add a literal `'ETH' AS chain` / `'POLYGON' AS chain` column per source and a `CHAIN_*` token — one model over many chains is exactly the cross-domain setting our [research program](../05-research/02-improvement-ideas.md#d--data-each-feeds-a2-how-tos-live-in-the-data-section) cares about.

## 8. Downstream labels (for the evaluation side)

Pretraining needs no labels, but [evaluation](../05-research/02-improvement-ideas.md#e1--build-the-multi-task-behavioral-benchmark) does. Public EVM label sources to join against your cohort: OFAC's sanctioned-address list (machine-readable mirrors on GitHub), Etherscan's public label cloud (phishing/exploit tags; manual export), CryptoScamDB, post-mortem address lists from major exploits, and the labels shipped with [Elliptic2](02-public-datasets-catalog.md#elliptic-1--2--bitcoin-illicit-activity-graphs)-style academic releases. Keep label timestamps in mind — labels assigned *after* your evaluation window are still fair; behavior *after* the window is not ([temporal discipline, C6](../03-learning-path/level-400-design-contracts-and-extensions.md#1-the-contracts)).

**Next:** [Solana](05-guide-solana-bigquery.md) · [Stellar](06-guide-stellar-hubble.md) · [straight to the training recipe](08-from-raw-data-to-training-run.md).
