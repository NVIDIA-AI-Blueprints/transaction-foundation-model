# Primer 1: What Is a Foundation Model?

**You know:** supervised learning — features in, labels out.
**You'll learn:** what "pretraining a foundation model" means, why it works without labels, and why anyone bothers.

## The problem with labels

Fraud labels are scarce (~0.12% of TabFormer rows), expensive, delayed (chargebacks arrive weeks later), and task-specific. A model trained only on "is this fraud?" learns nothing reusable for churn, credit risk, or segmentation — you start from scratch for each task.

Meanwhile, the *unlabeled* data is enormous and rich. Every transaction sequence encodes spending rhythms, merchant loyalties, geographic habits, salary cycles. Classic ML throws that structure away unless you hand-engineer features for it.

## The foundation-model bet

A **foundation model** is a model pretrained on large unlabeled data with a *generic* objective, so that its internal representations transfer to many downstream tasks.

The bet has three parts:

1. **A generic objective forces general understanding.** This repo's objective is *next-token prediction*: given a customer's history so far, predict the next transaction's tokens. To get good at this, the model can't memorize answers — it must implicitly learn things like "this customer grocery-shops Saturday mornings", "gas stations follow highway-adjacent merchants", "small online charges at 3am are unusual for this card". Nobody labels any of that. The prediction task *extracts* it.

2. **Representations are the product, not predictions.** After pretraining we mostly discard the model's actual next-token guesses. What we keep is the **hidden state** — the internal vector the model computed in order to make its guess. That vector is a dense summary of "everything relevant about this history" ([Primer 5](05-embeddings.md) covers extraction).

3. **One pretraining, many tasks.** The same embeddings can feed a fraud detector today and a churn model tomorrow. The expensive part (pretraining) amortizes.

## "Foundation model" ≠ "huge model"

The famous foundation models (GPT, Claude, Llama) have billions of parameters because *natural language* needs them: ~100K-token vocabularies and all of human knowledge. This repo's model is **~29M parameters** with a **~6,251-token vocabulary** — about the size of GPT-2's smallest cousin's pinky finger — and that's appropriate: the "language" of one card dataset is vastly simpler than English. What makes it a foundation model is the *training pattern* (generic pretraining → transfer), not the parameter count.

Industry context, for calibration: Visa's TREASURE and Revolut's PRAGMA apply the same pattern at 10M–1B parameters over billions of real transactions; results at that scale (+111% anomaly detection for TREASURE) are why this area is hot. See the [literature review](../05-research/01-literature-review.md).

## Where this sits in the model-family zoo

You'll meet three pretraining styles in the literature:

| Style | Objective | Famous example | Used here? |
|-------|-----------|----------------|------------|
| **Autoregressive (causal)** | Predict next token, left-to-right | GPT, Llama | ✅ — see [Primer 3](03-causal-language-modeling.md) |
| **Masked (bidirectional)** | Hide random tokens, predict them from both sides | BERT; IBM's original TabFormer model | ❌ (a research direction) |
| **Contrastive** | Pull representations of "same entity" views together | SimCLR, CoLES | ❌ (a research direction) |

This repo chose autoregressive for good reasons: every token position gives a training signal (efficient), it matches how transaction streams actually arrive (past → future), and the tooling (NeMo AutoModel, HuggingFace causal-LM APIs) is mature. Hybrid objectives are a promising upgrade path — see [improvement ideas](../05-research/02-improvement-ideas.md).

## The two-stage pattern in this repo, concretely

```
Stage 1 — pretrain (notebooks 02–03, no fraud labels touched):
  19.5M transactions → ~64K sequences (~263M tokens)
  → 29M-param decoder trained to predict next token
  → checkpoint: models/decoder-foundation-model/

Stage 2 — transfer (notebooks 04–05, fraud labels used only by XGBoost):
  same transactions → 512-d embeddings from the frozen checkpoint
  → XGBoost on [13 raw features + 64-d PCA(embeddings)]
  → higher average precision than raw features alone
```

Two details worth noticing:

- **The fraud label is never shown to the foundation model.** If embeddings still help detect fraud (they do), the model must have learned behavioral structure that *correlates* with fraud — that's transfer, not memorization.
- **The model is frozen in stage 2.** We don't even fine-tune it; we just read vectors out. This is the simplest possible transfer mechanism — and the cheapest to serve.

## Why this beats hand-engineered features (and why it doesn't, alone)

Notebook 05's punchline is instructive on both sides:

- **Embeddings alone < raw features alone.** A 64-d PCA of a 512-d summary loses fine-grained signal ("amount was exactly $9,999.99") that a tree on raw features nails.
- **Raw + embeddings > raw alone, decisively, on average precision.** The embeddings contribute what raw per-row features *cannot express*: sequential context. "Is this transaction normal *for this card's history*?" is invisible to a row-wise model and is precisely what a sequence model encodes.

The general lesson: foundation-model embeddings are a **complement** to your existing features, not a replacement — at least at this model scale.

## Key takeaways

- A foundation model = generic pretraining on unlabeled data + reuse across downstream tasks.
- The objective is a *pretext*; the representations are the product.
- Scale is contextual: 29M parameters is a perfectly real foundation model for a 6K-token transaction language.
- Embeddings complement, not replace, classic features.

**Next:** [Primer 2 — Tokens, tokenizers, vocabularies](02-tokenization-and-vocabularies.md), because before any of this works, transactions must become tokens.
