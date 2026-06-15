# Orientation: What This Repo Is (and Isn't)

**Audience:** a data scientist comfortable with pandas, scikit-learn, and XGBoost. No foundation-model experience assumed.
**Time:** ~10 minutes.

## What you're looking at

This repository is an **end-to-end blueprint** for building a *transaction foundation model*: a small language model pretrained not on text, but on **sequences of financial transactions**. It was published as an NVIDIA developer example and is the foundation we build on for research and training of similar models on transaction data — including, eventually, on-chain (blockchain) transaction data.

The entire workflow lives in five numbered notebooks at the repo root, supported by a small library in [`src/`](../../src):

| Stage | Notebook | What happens | Reusable code |
|-------|----------|--------------|---------------|
| 1. Baseline | [`01_dataset_baseline.py`](../../01_dataset_baseline.py) | Load 24M credit-card transactions, split by time, train a classic XGBoost fraud model | — |
| 2. Tokenize | [`02_seq_preproc_tokenization.py`](../../02_seq_preproc_tokenization.py) | Convert each transaction into ~12 domain tokens; stitch per-customer histories into "sentences" | [`src/tokenizer/`](../../src/tokenizer) |
| 3. Pretrain | [`03_foundation_model_training.py`](../../03_foundation_model_training.py) | Train a ~29M-parameter decoder model to predict the next token | [`scripts/train_decoder_model.py`](../../scripts/train_decoder_model.py), [`src/clm_data.py`](../../src/clm_data.py), [`configs/`](../../configs) |
| 4. Embed | [`04_inference_embedding_extraction.py`](../../04_inference_embedding_extraction.py) | Use the trained model to turn transaction histories into 512-d vectors | [`src/decoder_inference.py`](../../src/decoder_inference.py) |
| 5. Evaluate | [`05_xgboost_fraud_detection.py`](../../05_xgboost_fraud_detection.py) | Show that raw features + embeddings beat raw features alone | — |

## The idea in one picture

```
  One transaction (a row in a table)
  ┌──────────────────────────────────────────────────────────────┐
  │ $42.75 | Walmart | MCC 5411 | 09:32 Wed | Card 0 | CA | u1001│
  └──────────────────────────────────────────────────────────────┘
                              │  tokenizer (deterministic rules)
                              ▼
  AMT_1 MERCH_667 CAT_RETAIL MCC_5411 HOUR_09 DOW_2 MONTH_01
  CARD_0 CHIP_CHIP ZIP3_951 STATE_CA CUST_1001        (12 tokens)
                              │  group by customer+card, chronologically
                              ▼
  <bos> txn₁ <sep> txn₂ <sep> … <sep> txn₃₁₅ <eos>    (~4,096 tokens)
                              │  pretrain: "predict the next token"
                              ▼
        ~29M-parameter decoder-only transformer (Llama-style)
                              │  read out internal representation
                              ▼
            512-dimensional embedding per history
                              │  feed as features
                              ▼
              XGBoost fraud detector (improved AP)
```

If that picture raises questions — *what's a token? why predict the next one? what's a decoder?* — good. Those are exactly the questions the [Concepts](../02-concepts/README.md) section and the [Learning Path](../03-learning-path/README.md) answer, step by step.

## What this repo is

- **A teaching blueprint.** Each notebook is heavily annotated and runs end to end on a single GPU (the real pretrained checkpoint, trained ~3,000 steps on 8× A100, ships via Git LFS so you don't have to retrain).
- **A starting point for research.** The tokenizer, data pipeline, and training config are deliberately modular so you can swap datasets, token schemes, and architectures. The [Research](../05-research/README.md) and [Data](../04-data/README.md) sections describe where we're taking it.
- **A pattern, not a product.** The pattern — *deterministic domain tokenization → causal pretraining → embedding extraction → downstream evaluation* — applies to card payments, bank transfers, and on-chain transactions alike.

## What this repo is not

- **Not a production fraud system.** There are no services, no tests, no monitoring. Notably, the tokenizer includes a `CUST_*` (customer ID) token that is useful on this synthetic benchmark but would be a leakage/generalization concern in production (see [Level 400](../03-learning-path/level-400-design-contracts-and-extensions.md)).
- **Not a large language model.** ~29M parameters is ~10,000× smaller than frontier LLMs. Foundation-model *technique* doesn't require foundation-model *scale* — and for a vocabulary of only ~6,251 tokens, small is appropriate.
- **Not CPU-friendly.** The pipeline assumes NVIDIA GPUs throughout (RAPIDS for data, NeMo for training). See [Environment Setup](02-environment-setup.md).

## Where to go next

1. [Environment Setup](02-environment-setup.md) — get the container and notebooks running.
2. [Level 100: The Big Picture](../03-learning-path/level-100-the-big-picture.md) — the 10-minute conceptual tour.
3. Keep the [Glossary](03-glossary.md) open in a tab. Every acronym you'll meet is in there.
