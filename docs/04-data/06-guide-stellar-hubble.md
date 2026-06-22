# Step-by-Step: Stellar Data via Hubble (BigQuery)

**Goal:** per-account, time-ordered **payment** events from Stellar, exported for the [tokenizer recipe](08-from-raw-data-to-training-run.md).
**Prereqs:** [the BigQuery primer](03-bigquery-blockchain-primer.md).

## Why Stellar is special for *this* project

Stellar is the most *payments-shaped* public chain: account-based, asset-denominated (USDC and anchored fiat tokens, not just a native coin), built for remittances and anchors. Of all public ledgers, its event stream looks most like the card/bank data this repo was designed around — typed operations, named currencies, human-scale accounts — making it the gentlest on-chain starting point, and an ideal middle rung for the [blockchain↔fiat transfer study](../05-research/02-improvement-ideas.md#d3--the-blockchainfiat-transfer-study-the-original-contribution-bet).

## 1. The datasets: Hubble

Maintained by the Stellar Development Foundation ("**Hubble**"; docs: [developers.stellar.org → Data → Hubble](https://developers.stellar.org/docs/data/analytics/hubble)). SDF hosts and pays storage; you pay queries.

| Dataset | Contents |
|---|---|
| `crypto-stellar.crypto_stellar` | raw/core history: `history_ledgers`, `history_transactions`, `history_operations`, `history_effects`, `history_trades`, `history_assets`, `history_contract_events`, plus state (`accounts`, `account_signers`, `trust_lines`, `claimable_balances`) |
| `crypto-stellar.crypto_stellar_dbt` | curated dbt marts: **`enriched_history_operations`** (operations pre-joined with transaction + ledger context — *use this*), `accounts_current`, `trust_lines_current`, `offers_current`, `liquidity_pools_current`, fee stats |

Three Hubble-specific quirks:

1. **It won't appear in the BigQuery Explorer search** — it lives in SDF's project, not `bigquery-public-data`. Open it by ID ("+ Add → Star a project by name → `crypto-stellar`") or via the docs' console links.
2. **Partition/cluster scheme:** history tables are partitioned on **`batch_run_date`** (the ETL batch date — *not* the ledger close time) and `history_operations` is clustered on `transaction_id, source_account, type`. Filter `batch_run_date` for cost, `closed_at` for semantics — typically both, with the batch window padded a day on each side.
3. **Freshness:** "intraday batches; no same-day guarantee." Build corpora up to *yesterday*, not *now*.

```bash
# See what actually exists today (the enriched table has historically moved between the two datasets):
bq ls --project_id=crypto-stellar crypto_stellar | head -40
bq ls --project_id=crypto-stellar crypto_stellar_dbt | head -40
bq show --schema --format=prettyjson crypto-stellar:crypto_stellar_dbt.enriched_history_operations | head -80
```

## 2. Freshness check

```sql
SELECT MAX(closed_at) AS latest
FROM `crypto-stellar.crypto_stellar_dbt.enriched_history_operations`
WHERE batch_run_date >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 DAY);
```

## 3. Understand Stellar operations (1 minute)

A Stellar transaction contains **operations**; each has an integer `type` / string `type_string`. The ones that matter for a payments corpus:

| `type_string` | Meaning |
|---|---|
| `payment` | direct asset transfer A → B (the core event) |
| `path_payment_strict_send` / `_receive` | cross-asset payment through the DEX (sender pays X, receiver gets Y) |
| `create_account` | first funding of a new account |
| `manage_buy_offer` / `manage_sell_offer` | DEX order activity (trading behavior, not payments) |
| `change_trust` | opt-in to hold an asset (a "new currency relationship" event) |

Amounts come with `asset_code` (`USDC`, `XLM` for native, anchored fiat codes) — a real multi-currency field, which TabFormer never had.

## 4. Step 1+2 — Cohort and event stream in one pass

Stellar's volume is modest enough (vs Solana) to do both steps in one statement over a quarter. As always: verify column names against the live schema first (§1).

```sql
CREATE OR REPLACE TABLE `YOUR_PROJECT.chain_corpus.xlm_events` AS
WITH ops AS (
  SELECT
    op_source_account                       AS sender,      -- check schema: source_account / op_source_account
    `to`                                    AS receiver,    -- flattened detail column for payments
    SAFE_CAST(amount AS FLOAT64)            AS amount,
    COALESCE(asset_code, 'XLM')             AS asset,       -- native XLM has no code
    type_string                             AS event_type,
    closed_at                               AS ts
  FROM `crypto-stellar.crypto_stellar_dbt.enriched_history_operations`
  WHERE batch_run_date >= TIMESTAMP('2025-12-31')           -- partition pruning (padded)
    AND batch_run_date <  TIMESTAMP('2026-04-02')
    AND closed_at      >= TIMESTAMP('2026-01-01')           -- semantic window
    AND closed_at      <  TIMESTAMP('2026-04-01')
    AND type_string IN ('payment', 'path_payment_strict_send', 'path_payment_strict_receive')
),
cohort AS (
  SELECT sender AS address FROM ops
  GROUP BY sender
  HAVING COUNT(*) BETWEEN 20 AND 5000
     AND MOD(ABS(FARM_FINGERPRINT(sender)), 10) = 0         -- deterministic 10% sample
)
SELECT c.address AS entity, 'OUT' AS direction, o.receiver AS counterparty,
       o.amount, o.asset, o.event_type, o.ts
FROM ops o JOIN cohort c ON o.sender = c.address
UNION ALL
SELECT c.address, 'IN', o.sender, o.amount, o.asset, o.event_type, o.ts
FROM ops o JOIN cohort c ON o.receiver = c.address;
```

Then [export to Parquet exactly as in the EVM guide](04-guide-evm-bigquery.md#6-step-3--export-to-parquet).

## 5. Mapping to tokens (preview)

| Column | Token (strategy) | Note |
|---|---|---|
| `amount` | `AMT_*` (log bins) | bin per-asset or in a reference unit — mixing XLM and USDC magnitudes in one binning is a classic mistake |
| `counterparty` | `CTPY_*` (hash) | anchors/exchanges are mega-hubs; consider top-K `HUB_*` tokens |
| `asset` | `CUR_*` (mapping top assets + default) | the multi-currency field — new vs TabFormer |
| `event_type` | `OP_*` (fixed: payment / path-send / path-receive) | the MCC analog |
| `direction` | `DIR_IN`/`DIR_OUT` | |
| `ts` | `HOUR_* DOW_* MONTH_*` + `TDIF_*` | remittance rhythms (paydays!) are exactly what the model should find |

Continue with [the universal training recipe](08-from-raw-data-to-training-run.md).

## Stellar gotchas, recapped

1. Two datasets (`crypto_stellar` vs `crypto_stellar_dbt`) and the enriched table has lived in both — `bq ls` before you write SQL.
2. Partition column is `batch_run_date` (ETL time), not the semantic `closed_at` — filter both, pad the batch window.
3. Native XLM has `asset_code = NULL` — `COALESCE` it.
4. Path payments have *two* amounts (source-paid and destination-received: `source_amount` vs `amount`) — decide which leg you're modeling.
5. No same-day data guarantee — end corpora at T-1.
6. Trading ops (`manage_*_offer`) dwarf payments in row count; the `type_string` filter isn't optional.
