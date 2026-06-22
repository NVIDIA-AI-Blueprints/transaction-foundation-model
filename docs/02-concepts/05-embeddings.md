# Primer 5: Embeddings — Turning a Predictor into a Feature Factory

**You know:** feature vectors, PCA, cosine similarity.
**You'll learn:** what hidden states are, how pooling turns them into one vector per history, why last-token pooling is the default for decoder models, and how the vectors flow into XGBoost.

## Hidden states: the model's working memory

Run the trained model over a tokenized sequence and, at every layer, it computes a 512-d vector *per token position* — the **hidden states**. The final layer's hidden state at position *i* is, by construction, the model's best summary of *everything up to and including position i* — because that summary is exactly what it uses to predict position *i+1*.

For prediction, that summary gets projected to 6,251 logits and we argmax. For **embedding extraction**, we skip the projection and keep the summary itself:

```python
# src/decoder_inference.py — the crucial flag
outputs = self.model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_hidden_states=True,     # ← give me the internals, not just logits
)
hidden_states = outputs.hidden_states[-1]   # final layer: (batch, seq_len, 512)
```

That's the entire trick. Any HuggingFace causal LM exposes it; nothing about it is specific to transactions.

## Pooling: many positions → one vector

A sequence gives `(seq_len, 512)` hidden states; downstream models want one fixed-size vector. [`HuggingFaceDecoderInference._pool_embeddings`](../../src/decoder_inference.py) implements the two standard options:

**Last-token pooling (the default here):** take the hidden state at the final *non-padding* position.

```python
seq_lengths = attention_mask.sum(dim=1) - 1          # index of last real token
embeddings = hidden_states[batch_indices, seq_lengths, :]   # (batch, 512)
```

Why it's right for *causal* models: the causal mask means position *i* has only seen tokens ≤ *i* — so **only the last position has seen the whole sequence**. Earlier positions are summaries of prefixes. Decoder-based embedding models in the text world (the code docstring cites NV-Embed-v2, E5-Mistral) standardized on exactly this.

**Mean pooling (available, non-default):** average all non-pad positions. For *bidirectional* encoders (BERT) it's the natural choice since every position sees everything; for causal models it averages progressively-informed prefix summaries — sometimes a useful regularizer, usually slightly worse. It's one flag away if you want to A/B it (`pooling="mean"`).

> **Pitfall worth internalizing:** with last-token pooling, padding *placement* matters. Pad on the right and compute the last *real* index via the attention mask (as the code does). Naively taking `hidden_states[:, -1, :]` on right-padded batches reads the summary of `<pad>` tokens — a classic silent bug.

## What gets embedded in this repo (and the alignment trick)

Notebook 04 embeds **one sequence per transaction**: for each target transaction, the tokens of the customer's history up to and including it (truncated to the context window) are encoded, and the last-token embedding becomes *that transaction's* 512-d feature vector. Three artifacts are saved per split:

```
data/embeddings/
├── train_embeddings.npy   # (n, 512) float vectors
├── train_labels.npy       # fraud labels, aligned by row
└── train_row_ids.npy      # ← the unglamorous hero
```

The `row_ids` matter because the tokenizer's `preprocess()` **sorts** the frame by `(user, card, time)` — a different order than the raw parquet. Notebook 05 uses these IDs to realign embeddings with raw features row-by-row. If you ever extend the pipeline, preserve this pattern; misalignment here would silently destroy the comparison (features from one transaction, label from another).

Scale facts, for expectations: train = 1M-row *balanced* sample (~2.5% fraud), val/test = 100K *stratified* samples (~0.1% fraud, the realistic rate) — matching the XGBoost setup in notebook 05. Batched extraction with pinned memory ([`extract_embeddings_batched`](../../src/decoder_inference.py)) makes this minutes, not hours, on an A100.

## What the vectors are good for

The same 512 numbers support several workflows:

1. **Features for a downstream model** (this repo's main path). Notebook 05 compresses 512-d → **64-d with PCA** first — XGBoost on hundreds of dense, correlated columns overfits and slows; PCA keeps most variance in a tree-friendlier shape. Then it trains on `[13 raw features ∥ 64 PCA dims]`.
2. **Similarity & retrieval.** Cosine-near histories behave alike — useful for "find accounts like this confirmed-fraud account."
3. **Clustering / segmentation.** Behavioral segments from k-means on embeddings, no labels needed.
4. **Visualization & debugging.** Notebook 04 projects to 2D/3D with UMAP; fraud forming visible (sub)structures is qualitative evidence pretraining captured something real.

## Why frozen embeddings transfer (and their limits)

The pretraining objective never saw a fraud label, so why do its vectors help? Because predicting behavior requires *modeling normality* — and fraud is, largely, a deviation from an account's normal. The embedding encodes "what this history looks like, in behavior-space"; XGBoost learns which regions of that space correlate with fraud.

The honest limits (notebook 05 shows both):

- **Embeddings alone underperform raw features.** Binning/hash/PCA all discard precision a tree on raw values exploits.
- **Raw + embeddings wins on Average Precision** — embeddings contribute the sequential context raw rows cannot express. Complement, not replacement.

A stronger-but-costlier alternative to frozen embeddings is **fine-tuning** the whole model with a classification head — more accurate in much of the literature, but you lose the "one frozen model, many cheap tasks" economics. It's a natural [next experiment](../05-research/02-improvement-ideas.md).

## Key takeaways

- Embeddings = final-layer hidden states + pooling; `output_hidden_states=True` is the only API you need.
- Last-token pooling is principled for causal models — only the last position has seen everything; respect the attention mask.
- Save row IDs alongside embeddings; alignment bugs are silent and fatal.
- Frozen embeddings are cheap, reusable, and complementary to raw features; fine-tuning is the heavier upgrade path.

**Next:** [Primer 6 — The GPU stack](06-gpu-stack.md): the NVIDIA machinery that makes all of this run.
