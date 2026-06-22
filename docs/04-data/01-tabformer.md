# TabFormer: The Included Dataset

The dataset every notebook uses. Knowing its quirks saves you from misreading results.

## Facts

| Property | Value |
|---|---|
| Source | IBM, released with the TabFormer paper ([arXiv:2011.01843](https://arxiv.org/abs/2011.01843), ICASSP 2021) |
| File | `card_transaction.v1.csv` (~2.2 GB, inside `transactions.tgz`) |
| Download | [IBM Box](https://ibm.ent.box.com/v/tabformer-data/folder/130747715605) — notebook 01's first cell automates it |
| Rows | ~24.4M card transactions |
| Entities | 2,000 users × up to 10 cards each |
| Time span | 2002 – 2019 |
| Label | `Is Fraud?` — ~0.12% positive (~30K frauds) |
| Nature | **Synthetic** — generated to mimic real card behavior |

## Schema

```
User, Card, Year, Month, Day, Time, Amount, Use Chip, Merchant Name,
Merchant City, Merchant State, Zip, MCC, Errors?, Is Fraud?
```

Field quirks the pipeline handles (see [`preprocess()`](../../src/tokenizer/financial_pipeline.py) and [Level 300 Stage A](../03-learning-path/level-300-the-pipeline-in-code.md)):

- **`Amount`** is a string with a `$` prefix (and occasionally commas): `"$42.75"`.
- **`Merchant Name`** is a long opaque integer rendered as a string (`"3527213246127876953"`) — synthetic IDs, not names; treated as a categorical to hash.
- **`Time`** is `"HH:MM"`; combined with Year/Month/Day into a timestamp. There are no seconds.
- **`Zip`** arrives as a float-string (`"95113.0"`); online transactions have no ZIP/state → defaults (`000`, `XX`/`ONLINE`).
- **`MCC`** is the standard merchant category code (e.g. 5411 grocery); ~110 distinct values appear.
- **`Errors?`** (e.g. "Insufficient Balance") is unused by the tokenizer — a free experiment for the curious.

## How each notebook consumes it

| Notebook | What it takes | What it leaves |
|---|---|---|
| 01 | raw CSV | temporal 80/10/10 splits (`data/TabFormer/temporal_split/*.parquet`) + 100K stratified `val_eval`/`test_eval` subsets |
| 02 | the splits | tokenized corpora (`data/decoder_corpus/*.txt`) |
| 04 | splits + checkpoint | embeddings for a balanced 1M train sample + the eval subsets |
| 05 | embeddings + splits | the three-model comparison |

## What being synthetic means for conclusions

TabFormer is ideal for a blueprint: free, sizable, clean licensing, plausible structure. But keep in mind, especially for anything customer-facing:

- **Patterns are generator artifacts.** The fraud process is simulated; absolute metric values don't transfer to any real portfolio. *Relative* statements ("combined features beat raw") are the meaningful kind.
- **The world is closed.** 2,000 users for 17 years, no new-customer influx — which flatters identity-flavored tokens like `CUST_*` ([Level 400 sharp edge #1](../03-learning-path/level-400-design-contracts-and-extensions.md#3-sharp-edges-read-before-deploying-or-publishing-numbers)).
- **Counterparty structure is thin.** Users transact with merchants only; there's no user↔user network, so graph-style ideas ([G1/G2](../05-research/02-improvement-ideas.md#g--graph--relational-context)) can't be properly exercised here — one more reason for the [blockchain corpora](03-bigquery-blockchain-primer.md).

When you outgrow it: the [public datasets catalog](02-public-datasets-catalog.md) (MBD's 950M *real* transactions is the natural next step) and the [chain guides](03-bigquery-blockchain-primer.md).
