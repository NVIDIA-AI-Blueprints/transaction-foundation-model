# Primer 3: Causal Language Modeling

**You know:** supervised learning needs (X, y) pairs.
**You'll learn:** how next-token prediction manufactures millions of (X, y) pairs from unlabeled sequences, what the loss means, and how this repo wires it up.

## The objective

**Causal language modeling (CLM)** trains a model to answer one question, at every position of every sequence:

> Given everything so far, what comes next?

Take a tokenized history:

```
<bos> AMT_1 MERCH_667 CAT_RETAIL MCC_5411 HOUR_09 DOW_2 ...
```

The model is trained so that:

- given `<bos>`, it predicts `AMT_1`
- given `<bos> AMT_1`, it predicts `MERCH_667`
- given `<bos> AMT_1 MERCH_667`, it predicts `CAT_RETAIL`
- … and so on, for all ~4,096 positions.

"**Causal**" means the model may only use *earlier* positions to predict — no peeking forward. That constraint is enforced by the architecture itself (the attention mask, [Primer 4](04-decoder-architecture.md)), so a single forward pass over a 4,096-token sequence yields ~4,096 training examples at once. That density is why this repo's ~64K sequences provide ~263M token-level training signals.

## Where the "labels" come from

Here's the implementation detail that confuses everyone the first time. Look at [`src/clm_data.py`](../../src/clm_data.py):

```python
input_ids = np.full(self.seq_length, self.pad_token_id, dtype=np.int64)
input_ids[: len(tokens)] = tokens

labels = np.full(self.seq_length, -100, dtype=np.int64)
labels[: len(tokens)] = tokens

return {
    "input_ids": torch.from_numpy(input_ids),
    "labels": torch.from_numpy(labels),
}
```

**The labels are the inputs.** Identical, unshifted. This works because HuggingFace-convention causal-LM forwards do the shift internally: the prediction at position *i* is compared against the label at position *i + 1*. You give the model the sequence; the framework constructs the "predict-the-next" alignment.

Two more conventions hiding in that snippet:

- **`-100` = "don't score this position."** Padding positions get label `-100`, which cross-entropy losses ignore by convention (NeMo's `MaskedCrossEntropy` here, [config](../../configs/pretrain_financial_decoder.yaml)). The model never gets rewarded for predicting `<pad>`.
- **Fixed-length tensors.** Every sequence is truncated/padded to `seq_length: 4096` so batches stack into rectangular tensors.

That ~60-line file is the *entire* labeled-data manufacturing process. No annotation, no labeling vendor — the sequence is its own supervision. This is what "self-supervised" means.

## What the loss number means

The model's output at each position is a probability distribution over all 6,251 vocabulary tokens. The **cross-entropy loss** is the average negative log-probability assigned to the true next token:

- **Random guessing**: every token gets probability 1/6251 → loss = ln(6251) ≈ **8.74**. This is where an untrained model starts (notebook 03 quotes it as the sanity baseline).
- **Loss 5.0** → the model is as good as choosing uniformly among e⁵ ≈ 148 tokens. (`perplexity = exp(loss)`)
- **Loss 2.0** → effectively choosing among ~7. For structured fields this is plausible: given `HOUR_09` was just emitted, `DOW_*` is next, and a card's day-of-week distribution is narrow.

Watch loss with structure in mind: the 12 fields differ wildly in predictability. `CUST_*` after 11 other tokens of the same customer's transaction is nearly free; `MERCH_*` is genuinely hard. Falling loss means the model is mastering, in rough order: the field *grammar* (positions cycle AMT→MERCH→…→CUST), the marginal distributions (most transactions are small), then the conditional behavioral patterns (this card buys gas Mondays) — which is the part we actually want.

## Why prediction ⇒ understanding (the whole point)

To predict the next transaction's tokens well, the model is *forced* to maintain an internal summary of the history: who this customer behaves like, what time-of-day it is in their rhythm, what merchants cluster together. Those summaries are the hidden states — and they, not the predictions, are what we harvest as embeddings ([Primer 5](05-embeddings.md)).

A worthwhile subtlety for transaction data: the *order of fields within* a transaction is fixed by the tokenizer (AMT before MERCH before HOUR…), so some "predictions" are really within-row inference ("given amount and merchant, what hour?") while crossing a `<sep>` is true *next-event* prediction ("given 100 transactions, what does this card do next?"). Both produce useful gradients; the mixture is a design choice you can revisit (field order is an [ablation idea](../05-research/02-improvement-ideas.md)).

## CLM vs MLM, in one minute

BERT-style **masked language modeling** hides ~15% of tokens and predicts them using *both* directions. Comparison for our setting:

| | CLM (this repo) | MLM (e.g. IBM TabFormer's BERT, PRAGMA) |
|---|---|---|
| Signal per pass | every position | only masked positions (~15%) |
| Sees future context | no | yes |
| Natural for | streaming/next-event prediction, generation | bidirectional encoding of a complete window |
| Embedding extraction | last-token pooling natural | CLS/mean pooling natural |

Neither dominates universally; production systems increasingly use **hybrid objectives** (Revolut's PRAGMA combines three masking granularities; see the [literature review](../05-research/01-literature-review.md)). This repo uses pure CLM for simplicity and tooling maturity.

## How it's wired in this repo

The training stack is intentionally thin — three pieces:

1. **Dataset** — [`src/clm_data.py`](../../src/clm_data.py) reads corpus lines, encodes with the deterministic tokenizer, yields `{input_ids, labels}` (shown above).
2. **Config** — [`configs/pretrain_financial_decoder.yaml`](../../configs/pretrain_financial_decoder.yaml) names that dataset builder via `_target_`, defines the model, optimizer (AdamW, lr 2e-4, cosine decay), batch size (16 sequences ≈ 65K tokens/step), and checkpointing.
3. **Launcher** — [`scripts/train_decoder_model.py`](../../scripts/train_decoder_model.py) hands the config to NeMo AutoModel's `TrainFinetuneRecipeForNextTokenPrediction`, which owns the training loop, distribution, and checkpoint writing ([Primer 6](06-gpu-stack.md)).

There is no hand-written training loop anywhere in the repo. The domain lives in the tokenizer and dataset; the engineering lives in NeMo.

## Key takeaways

- CLM converts unlabeled sequences into dense supervision: every position is a training example; `labels = input_ids`, shift handled internally, `-100` masks padding.
- Loss is interpretable: starts at ln(vocab) ≈ 8.74, and falling loss tracks the model learning grammar → marginals → behavior.
- Prediction is a pretext; the forced internal summaries (hidden states) are the product.
- CLM vs MLM vs hybrids is a live research axis — this repo is the clean CLM baseline.

**Next:** [Primer 4 — Decoder-only transformers](04-decoder-architecture.md): the machine that does the predicting.
