# Level 100: The Big Picture

*10 minutes. No prerequisites. By the end you can explain this repo to a colleague over coffee.*

## The problem

A fraud model looks at a transaction and asks: *is this one bad?* Classic ML answers with hand-crafted features — amount, merchant type, hour, maybe some rolling aggregates someone engineered in 2019. It works, but it's brittle, slow to adapt, and mostly blind to the thing fraud analysts actually use: **the story so far**. A $300 electronics purchase at 2am means one thing for a card that does this monthly, another for a card that's never left grocery stores in Iowa.

Stories are sequences. And the best sequence-understanding technology we have is the language model.

## The idea

This repo asks: *what if we treated a customer's transaction history like text, and pretrained a language model on it?*

A raw transaction:

```
$42.75 | Walmart | groceries (MCC 5411) | Wed 09:32 | card 0 | San Jose, CA | user 1001
```

becomes a 12-token "sentence":

```
AMT_1 MERCH_667 CAT_RETAIL MCC_5411 HOUR_09 DOW_2 MONTH_01 CARD_0 CHIP_CHIP ZIP3_951 STATE_CA CUST_1001
```

A customer's history becomes a "paragraph" of ~315 such sentences. A small language model (~29M parameters — tiny by ChatGPT standards, right-sized for this vocabulary) is then trained on one deceptively simple game: **given the history so far, predict the next token.** Played millions of times across 19.5M transactions, the game forces the model to learn spending rhythms, merchant relationships, and what "normal" looks like per card — without ever seeing a fraud label.

Then comes the payoff move: instead of asking the model to *predict*, we read out its **internal summary** of a history — 512 numbers called an *embedding* — and hand those numbers to a perfectly ordinary XGBoost fraud model as extra features.

## Does it work?

That's notebook 05's experiment, and the answer is the most instructive part:

| XGBoost variant | Features | Result |
|---|---|---|
| Baseline | 13 raw tabular features | strong ROC-AUC already |
| Embeddings only | 64-d compressed embeddings | *worse* than raw — embeddings lose fine detail |
| **Combined** | raw + embeddings | **clear winner on Average Precision** |

Average Precision is the metric that matters at ~0.1% fraud rates — it measures how clean the *top* of your alert queue is. The embeddings sharpen exactly that: they add the sequential context ("is this normal *for this card*?") that no per-row feature can express. Lesson: **foundation-model embeddings complement classic features; they don't replace them.**

## The five-notebook tour

The whole system is five notebooks run in order (reusable code lives in [`src/`](../../src)):

1. **[`01_dataset_baseline.ipynb`](../../01_dataset_baseline.ipynb)** — load the **TabFormer** dataset (24.4M synthetic card transactions, 2,000 users, 2002–2019, 0.12% fraud), split by *time* (train on past, test on future — anything else cheats), train the raw-feature XGBoost baseline.
2. **[`02_seq_preproc_tokenization.ipynb`](../../02_seq_preproc_tokenization.ipynb)** — build the transaction tokenizer, show why GPT's text tokenizer would be ~3× wasteful and semantically destructive, write the training corpus.
3. **[`03_foundation_model_training.ipynb`](../../03_foundation_model_training.ipynb)** — pretrain the decoder with NVIDIA NeMo AutoModel. (Runs a 2-minute demo; the real checkpoint — ~3,000 steps on 8× A100 — ships with the repo via Git LFS.)
4. **[`04_inference_embedding_extraction.ipynb`](../../04_inference_embedding_extraction.ipynb)** — extract 512-d embeddings for ~1.2M transactions; visualize the space with UMAP.
5. **[`05_xgboost_fraud_detection.ipynb`](../../05_xgboost_fraud_detection.ipynb)** — the three-way comparison above.

Everything runs on NVIDIA GPUs inside one container ([setup guide](../01-getting-started/02-environment-setup.md)): RAPIDS for data processing, NeMo for training.

## Why you (a data scientist) should care

- **It's learned feature engineering.** The pretrained model converts "design 200 behavioral aggregates by hand" into "extract one 512-d vector." The same vector serves fraud today, churn or credit risk tomorrow — pretrain once, reuse everywhere.
- **The pattern is portable.** Nothing here is credit-card-specific: any *event sequence with structured fields* — bank transfers, app events, **on-chain transactions** — can be tokenized the same way. That portability is the premise of our [data](../04-data/README.md) and [research](../05-research/README.md) programs; the industry versions of this idea (Visa, Revolut, Nubank) are documented in the [literature review](../05-research/01-literature-review.md).
- **It's honest about limits.** This is a teaching blueprint, not a product: synthetic data, a deliberately small model, and at least one known leakage-flavored token kept for pedagogy (see [Level 400](level-400-design-contracts-and-extensions.md)).

## The sentence to remember

> **Transaction histories are sentences in a behavioral language; pretraining a small language model on them yields embeddings that make ordinary fraud models meaningfully better.**

**Next:** [Level 200 — The Building Blocks](level-200-the-building-blocks.md), where the pipeline becomes six concrete components.
