# Concepts: Primers for the Unfamiliar

This section assumes you're a competent data scientist — you know pandas, scikit-learn, train/test splits, gradient boosting, and you've *heard of* HuggingFace — but that foundation models, language modeling internals, and the NVIDIA GPU stack are new territory.

Each primer is self-contained, ~10 minutes, and grounded in this repo's actual code. Read the ones you need; skip what you know. The [Learning Path](../03-learning-path/README.md) links back here whenever it uses one of these concepts.

| # | Primer | Read it if you're asking… |
|---|--------|--------------------------|
| 1 | [What is a foundation model?](01-foundation-models.md) | "Why pretrain at all? Why not just train XGBoost on labels like always?" |
| 2 | [Tokens, tokenizers, vocabularies](02-tokenization-and-vocabularies.md) | "What exactly is a token, and why do we make our own instead of using GPT's?" |
| 3 | [Causal language modeling](03-causal-language-modeling.md) | "What does the model actually optimize? Where do labels come from with no labels?" |
| 4 | [Decoder-only transformers](04-decoder-architecture.md) | "What's inside the model? What do hidden size / layers / heads / RoPE / GQA mean?" |
| 5 | [Embeddings](05-embeddings.md) | "How does a next-token predictor produce features for XGBoost?" |
| 6 | [The GPU stack](06-gpu-stack.md) | "What are NeMo, RAPIDS, cuDF, NGC, FSDP2 — and what does each do for me?" |

## The mental bridge from classic ML

If you've only done supervised tabular ML, here is the single biggest shift to internalize:

**Classic supervised ML** (what you know):

```
features (hand-engineered) + labels  →  model  →  predictions
```

One model, one task, labels required, features designed by you.

**The foundation-model pattern** (what this repo does):

```
Stage 1 (pretraining, no labels):
    raw sequences  →  generic objective ("predict next token")  →  pretrained model

Stage 2 (downstream, your labels):
    pretrained model(your data) → embeddings → + raw features → XGBoost → predictions
```

Two stages. The first is expensive, label-free, and done once. The second is cheap, uses your labels, and can be repeated for *many* tasks (fraud, churn, segmentation…) off the same pretrained model. The pretrained model functions as **learned feature engineering**: instead of you writing "number of transactions in the last 7 days" by hand, the model learns what matters about a history by being forced to predict its continuation, millions of times.

Everything else in these primers is detail in service of that pattern.
