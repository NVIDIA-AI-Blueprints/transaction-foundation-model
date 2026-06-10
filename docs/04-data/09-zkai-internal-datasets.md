# ZKAI Internal Datasets: The Embed Pipeline Catalog

**⏱ ~15 min · Level: 200–300 · Prerequisites: [the universal recipe](08-from-raw-data-to-training-run.md)**

> **Internal access required.** Everything on this page lives in ZKAI's GCP/Postgres infrastructure and needs team credentials. If you're an external reader: treat this as a worked example of cataloging proprietary event streams for FM training — the public-data path starts at the [BigQuery primer](03-bigquery-blockchain-primer.md).

This is **where our first production-facing training runs will start**: existing customers want **next-trade prediction for wallets**, and these pipelines already produce continuous, enriched, wallet-keyed trade streams across DEXs, perps, and prediction markets — exactly the `(entity = wallet, event = trade)` shape the model consumes. No BigQuery export step, no third-party freshness risk: the data lands in our own buckets minutes after it happens on-chain.

## 1. What the catalog is

[`ZKAI-Network/embed-datasets-catalog`](https://github.com/ZKAI-Network/embed-datasets-catalog) is the agent-discoverable source of truth for every data product the Embed pipelines emit. Three things to know about its design:

1. **One YAML manifest per data product** under `catalog/manifests/<pipeline>/`, with stable pipeline-qualified IDs like `dex-trades-pipeline.gcs.eth_dex_trades`. The producing infrastructure lives in [`embed-iac`](https://github.com/ZKAI-Network/embed-iac); manifests point back to it via `upstream_refs` and `lineage`.
2. **Manifests deliberately contain no columns.** Schemas are read *live* from Postgres/BigQuery (and GCS prefixes are listed live) through the catalog's REST + MCP service — so the docs can't drift from reality. To see a schema, ask the service, not the YAML.
3. **It's queryable by agents.** The Cloud Run service exposes `GET /data-products`, `/search?q=…`, `/data-products/{id}/peek?n=10`, `POST /query` (read-only SQL), and an MCP endpoint with the matching tools (`list/describe/search/query/peek_data_product`). This is the intended discovery path for Loom and for Claude-driven exploration.

```bash
# Discovery workflow (any HTTP client; creds = GCP ADC + PG* env vars per the repo README)
curl $CATALOG_URL/search?q=dex+trades
curl $CATALOG_URL/data-products/dex-trades-pipeline.gcs.eth_dex_trades        # manifest + live listing
curl $CATALOG_URL/data-products/dex-trades-pipeline.farcaster_mbd.dex_wallets/peek?n=10
```

## 2. The three storage tiers — build only on one of them

Every pipeline lands data in up to three places. **The retention policies make the choice for you:**

| Tier | Example bucket | Retention | Use for training? |
|------|----------------|-----------|-------------------|
| **Raw** (QuickNode uploads, pre-processing input) | `dex-trades-pipeline-raw-trades-ethereum` | **30 days** | ❌ never — it expires under you |
| **Archive** (raw Pub/Sub message dumps) | `dex-trades-pipeline-archive-dex-wallets` | **90 days** | ❌ debugging/replay only |
| **Datasets** (final enriched topics → GCS sink) | `embed-pipeline-datasets/<pipeline>/<topic>/` | **forever** | ✅ this is the corpus substrate |

The datasets tier receives newline-delimited JSON, **date-partitioned**, flushed every 5 minutes or 100 MB — i.e., a continuously growing, replayable event log. (Reproducibility note: a corpus built from it must record the date-prefix range it covered — that range is this tier's equivalent of [C2's determinism contract](../03-learning-path/level-400-design-contracts-and-extensions.md#1-the-contracts).)

## 3. The data products, by pipeline

### DEX trades (`dex-trades-pipeline`) — the next-trade-prediction substrate

| Product (catalog ID) | What it is | Where |
|---|---|---|
| `…gcs.eth_dex_trades` | Fully **enriched Ethereum DEX trades**, explicitly landed "for ML training" | `embed-pipeline-datasets/dex-trades-pipeline/eth-dex-trades/` |
| `…gcs.base_dex_trades` | Same, **Base** | `…/base-dex-trades/` |
| `…gcs.solana_dex_trades` | Same, **Solana** | `…/solana-dex-trades/` |
| `…farcaster_mbd.dex_wallets` | **Aggregated per-wallet statistics** across all three chains (Postgres, `ds_enrich.farcaster_mbd.dex_wallets`) | live serving table |

The enriched trade streams come out of the `dex-wallet-enrich` service — trades already joined with wallet and token context. Three chains, one schema family ⇒ a **multi-chain corpus with no schema reconciliation work**, something the public-data guides ([04](04-guide-evm-bigquery.md)/[05](05-guide-solana-bigquery.md)) have to earn the hard way. The `dex_wallets` table is the natural source of wallet-level features for the downstream head and for cohort selection (e.g., "wallets with ≥ 50 trades").

### Hyperliquid (`hyperliquid-pipeline`) — perps behavior

`hyperliquid_activity` (trade/activity events), `hyperliquid_asset_updates`, `hyperliquid_wallet_roles` — all NDJSON under `hyperliquid-pipeline-datasets/hyperliquid-trades-pipeline/…`, documented as the "replayable analytical source-of-truth" (Postgres is serving storage). Derivatives behavior — leverage, liquidations, funding-driven activity — is a *behaviorally distinct regime* from spot DEX trading; treat it as a domain for [few-shot transfer evaluation](../05-research/02-improvement-ideas.md#e1--build-the-multi-task-behavioral-benchmark), not just more rows.

### Polymarket (`polymarket-pipeline`) — prediction-market bets

`filtered_trades`, `enriched_markets`, `price_changes`, `wallet_new` / `wallet_update` (NDJSON under `embed-pipeline-datasets/polymarket-trades-pipeline/…`), plus the `farcaster_mbd.polymarket_bets` Postgres table. One operational quirk worth knowing: when sampling mode is enabled, raw ingestion switches from realtime Eventarc to a scheduled replay job — check the manifest's lineage notes before assuming the stream is complete realtime coverage.

### Kalshi (`kalshi-pipeline`) — regulated prediction markets

`trades_enriched`, `markets_enriched`, `wallets_update` under `embed-pipeline-datasets/kalshi-trades-pipeline/…`. Kalshi ingestion shares the Solana raw bucket with the DEX pipeline — cataloged once under the DEX pipeline, so don't double-count it.

### ⭐ The cross-chain mart (`mbd-dataform`) — start here

`mbd-dataform.bq.cross_chain_interactions` → BigQuery table `level-mark-437714-b1.mbd_recs.cross_chain_interactions`: the **canonical cross-chain wallet-item interaction mart spanning DEX, Polymarket, Kalshi, and Hyperliquid**, materialized by the shared [Dataform repo](https://github.com/ZKAI-Network/mbd-dataform). It's already the unified `(wallet, item, protocol, event_date, event_hour_utc, …)` event table that step 1 of the [universal recipe](08-from-raw-data-to-training-run.md) asks you to build — someone built it for you.

```sql
-- Orient yourself (from the manifest's curated sample queries):
SELECT protocol, COUNT(*) AS row_count
FROM `level-mark-437714-b1.mbd_recs.cross_chain_interactions`
GROUP BY protocol ORDER BY protocol;
```

**Two caveats**, straight from the manifest: freshness is **manual on demand** (Dataform runs are not scheduled — check `MAX(event_date)` and trigger a run if stale), and only **marts** are stable consumer contracts (staging tables are queryable but may change shape).

For completeness, the catalog also registers `public.bigquery-public-data.crypto_ethereum` — the same public dataset the [EVM guide](04-guide-evm-bigquery.md) builds on — so raw-chain joins stay one `query_data_product` call away.

## 4. From catalog to next-trade prediction

Here's the payoff: **next-trade prediction doesn't require inventing a new objective — it *is* [causal language modeling](../02-concepts/03-causal-language-modeling.md) on this data.** Predicting the next trade's tokens (venue, side, asset, size) for a wallet is literally the pretraining task; the customer deliverable falls out of the loss function.

Following the [universal recipe](08-from-raw-data-to-training-run.md), with the recipe's chain example adapted:

1. **Entity & event** — entity = `wallet_address`; event = one trade/interaction. Use the cross-chain mart (one `SELECT … ORDER BY wallet, timestamp`) or a single chain's enriched DEX stream for the first run.
2. **Schema → tokens** (one line per field, à la the [EVM mapping](04-guide-evm-bigquery.md)):

   | Field | Strategy | Tokens |
   |---|---|---|
   | protocol/venue | fixed vocab | `VENUE_DEXETH`, `VENUE_HL`, `VENUE_POLY`, `VENUE_KALSHI`, … |
   | side | fixed | `SIDE_BUY` / `SIDE_SELL` |
   | item (token/market) | **hash, ~5,000 buckets** | `ITEM_3041` — the high-cardinality field; upgrade path is [T4 tiered vocab](../05-research/02-improvement-ideas.md#t4--tiered-merchant-vocabulary) with top-K assets first-class |
   | size (USD) | log-bins | `AMT_4` |
   | inter-trade gap | **time-delta bins — non-negotiable here** | `TDIF_12` — burst-trading vs dormancy *is* the signal in trading data ([T1](../05-research/02-improvement-ideas.md#t1--turn-on-and-then-improve-time-encoding)) |
   | calendar | fixed | `HOUR_*`, `DOW_*` |

   **No wallet-identity token.** Apply the [T2 lesson](../05-research/02-improvement-ideas.md#t2--drop-the-cust-token-and-ablate-card-the-deployability-ablation) from day one: customers will ask about wallets the model has never seen, so identity must come from history, not from an ID embedding — and evaluate on a **wallet-disjoint split**.
3. **Evaluate the thing the customer buys** — next-item Prec@1 / Recall@K and next-side accuracy on temporally held-out trades: exactly the next-merchant task in the [E1 benchmark](../05-research/02-improvement-ideas.md#e1--build-the-multi-task-behavioral-benchmark), measured by ranking the model's next-token distribution over the `ITEM_*` (and `SIDE_*`) slice of the vocabulary. Wallet embeddings for recommendations/segmentation come free from the same checkpoint via [last-token pooling](../02-concepts/05-embeddings.md).
4. **Run it through [Loom](../06-experimentation/01-loom-workflow.md)** as Pattern C → Pattern B: `ingest` the export (record the mart snapshot date / GCS date-range in the data object name), `eda`, corpus, cost-gated `train --objective next-event`, `validate` against an obvious baseline (popularity / repeat-last-item — wallets are habitual; **beat the heuristic before celebrating**).

The multi-protocol structure also sets up the in-house version of the [transfer-learning bet](../05-research/02-improvement-ideas.md#d3--the-blockchainfiat-transfer-study-the-original-contribution-bet): does DEX-pretraining transfer to Hyperliquid wallets? To Polymarket bettors? Same evaluation harness, zero new data work.

## 5. Checklist before your first corpus

- [ ] Catalog service reachable; `describe_data_product` returns **live schema** for your chosen product (manifests won't tell you columns)
- [ ] Reading from the **datasets tier** (permanent), never raw (30 d) or archive (90 d)
- [ ] Cross-chain mart: checked `MAX(event_date)`; triggered a Dataform run if stale
- [ ] Snapshot boundary recorded (date-prefix range or mart snapshot) in the corpus/data-object name
- [ ] Wallet-disjoint split + temporal split both in place ([C6](../03-learning-path/level-400-design-contracts-and-extensions.md#1-the-contracts))
- [ ] Popularity / repeat-last-item baseline computed before any GPU spend
