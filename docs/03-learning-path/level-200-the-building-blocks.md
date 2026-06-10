# Level 200: The Building Blocks

*30 minutes. Assumes [Level 100](level-100-the-big-picture.md). By the end you can draw the system from memory and explain what each block consumes and produces.*

The system is six blocks in a line. Each block's **output is the next block's input** — internalize the handoffs and you understand the architecture:

```
[1] Dataset ──► [2] Tokenizer ──► [3] Corpus ──► [4] Pretraining ──► [5] Embeddings ──► [6] Downstream
 24.4M rows      12 tokens/txn     ~64K seqs      29M-param           512-d vector       XGBoost
 temporal        deterministic     of ~4096       decoder, next-      per transaction    raw vs +emb
 splits          vocab ~6,251      tokens         token objective     history            comparison
```

---

## Block 1 — Dataset and the temporal split

**Input:** `card_transaction.v1.csv` (TabFormer). **Output:** `train/val/test.parquet`.

TabFormer is IBM's synthetic credit-card dataset: ~24.4M transactions, 2,000 users (10 cards each), 2002–2019, fraud rate ~0.12%. Columns:

```
User, Card, Year, Month, Day, Time, Amount, Use Chip, Merchant Name,
Merchant City, Merchant State, Zip, MCC, Errors?, Is Fraud?
```

The non-obvious decision is the **split**: 80/10/10 by *date* (cumulative transaction count), not by random rows. Two reasons, both about honesty:

1. **Deployment realism** — in production you always train on the past and score the future; random splits leak future statistics into training and overstate performance.
2. **Sequence integrity** — block 3 builds chronological per-customer sequences; random row splits would scatter one customer's timeline across train *and* test.

Everything downstream inherits these splits — the foundation model trains on the train split's corpus only, and the final comparison evaluates on the most recent 10%.

> Details and download instructions: [TabFormer dataset page](../04-data/01-tabformer.md).

## Block 2 — The tokenizer

**Input:** transaction rows. **Output:** 12 token strings per row, from a fixed vocabulary of ~6,251.

Each field gets its own small tokenizer step, chosen by data shape — this trio of strategies is the whole design space you need:

| Strategy | Used for | Example | The trade it makes |
|---|---|---|---|
| **Fixed vocab** (range of ints) | hour, weekday, month, card, ZIP3, customer, amount bins | `HOUR_09`, `AMT_1` | resolution ↔ vocabulary size |
| **Mapping** (known values / ranges) | MCC, industry (from MCC ranges), state, chip type | `MCC_5411`, `CAT_RETAIL` | unknowns collapse to a default |
| **Hashing** (unbounded categoricals) | merchant name → 2,000 buckets | `MERCH_667` | collisions ↔ bounded vocab |

Worth pausing on the **amount**: `$42.75 → AMT_1` ($10–50 bin). We *deliberately throw away precision* so the model sees a small alphabet of magnitudes — distributions over 7 bins are learnable from this data; distributions over every cent are not. (The exact-amount signal isn't lost to the *system*: XGBoost still gets the raw value in block 6.)

The vocabulary is **deterministic** — rebuilt identically from configuration anywhere, nothing fitted to ship — and **contracted**: the model is built with `vocab_size: 6251`, so tokenizer and checkpoint are a matched pair.

> Deep dive: [Primer 2 — Tokens, tokenizers, vocabularies](../02-concepts/02-tokenization-and-vocabularies.md).

## Block 3 — The corpus

**Input:** token strings + customer/card grouping. **Output:** text files, one training sequence per line.

Single transactions are vocabulary; **sequences are the curriculum**. Transactions are grouped by `(user, card)`, sorted chronologically, and chunked ~315 per sequence:

```
<bos> AMT_1 MERCH_667 … CUST_1001 <sep> AMT_0 MERCH_44 … CUST_1001 <sep> … <eos>
```

The magic number 315: each transaction is 12 tokens + 1 `<sep>` ≈ 13, and 315 × 13 ≈ 4,096 — the training context window. So one training example ≈ **months of one card's behavior**. Five special tokens give sequences their grammar: `<bos>`/`<eos>` (boundaries), `<sep>` (transaction delimiter), `<pad>` (filler, ignored by the loss), `<unk>` (out-of-vocabulary safety).

Output: `data/decoder_corpus/{train,val,test}_corpus.txt` — ~64K training sequences, ~263M tokens. Plain text you can `head`: legibility is a feature; you can *read* what the model reads.

## Block 4 — Pretraining

**Input:** the corpus. **Output:** a trained checkpoint in HuggingFace format.

The model is a **decoder-only transformer** (Llama-style recipe at custom scale): hidden size 512, 8 layers, 8 attention heads (2 KV heads via GQA), context 4,096 (RoPE supports 8,192), ~29M parameters — built from `transformers.LlamaConfig` with **random init**, no pretrained weights anywhere.

The objective is **causal language modeling**: at every one of the ~4,096 positions, predict the next token from everything before it. Labels are manufactured from the data itself (`labels = input_ids`; the framework handles the shift; padding masked with `-100`). One number to carry around: random guessing gives loss ln(6,251) ≈ **8.7**; watching the curve fall from there is watching the model learn grammar → marginals → behavior.

Training is delegated to **NeMo AutoModel**: the repo provides a YAML config and ~10 lines of launcher; NeMo owns the loop, FSDP2 distribution, and checkpointing. The same command scales 1 → 8 GPUs by changing `--nproc-per-node`. Crucially the checkpoint is saved in **HuggingFace format**, so downstream blocks need no NeMo at all.

> Deep dives: [Primer 3 — Causal LM](../02-concepts/03-causal-language-modeling.md), [Primer 4 — Decoder architecture](../02-concepts/04-decoder-architecture.md), [Primer 6 — GPU stack](../02-concepts/06-gpu-stack.md).

## Block 5 — Embedding extraction

**Input:** checkpoint + tokenized histories. **Output:** one 512-d vector per transaction.

For each transaction, encode the card's history up to and including it, run the frozen model with `output_hidden_states=True`, and take the final layer's hidden state at the **last non-pad position** — the only position that has "seen" the whole history, hence the natural summary for a causal model (*last-token pooling*).

Three aligned artifacts are saved per split — embeddings, labels, and **row IDs** (the tokenizer re-sorts data, so explicit IDs are what keep vector *i* attached to the right transaction in block 6):

```
train_embeddings.npy  (1M × 512)   train_labels.npy   train_row_ids.npy
val/test: 100K stratified samples at the realistic ~0.1% fraud rate
```

UMAP projections of these vectors (notebook 04) are the qualitative checkpoint: fraud visibly concentrating in regions of behavior-space is your first evidence the pretraining learned something real.

> Deep dive: [Primer 5 — Embeddings](../02-concepts/05-embeddings.md).

## Block 6 — Downstream evaluation

**Input:** embeddings + raw features + labels. **Output:** the verdict.

Three XGBoost models, independently HPO-tuned (Optuna), all GPU-trained, evaluated on the held-out *future* test split:

| Model | Features | ROC-AUC | Average Precision |
|---|---|---|---|
| Baseline | 13 raw | high (near-saturated) | good |
| Embeddings-only | 64-d PCA of 512 | lower | lower |
| **Combined** | 13 raw + 64-d PCA | high | **best, clearly** |

Two evaluation choices worth copying into any project:

- **Average Precision as headline metric.** At 0.1% prevalence, ROC-AUC barely moves when the top of the ranking improves; AP is exactly the top-of-queue quality fraud ops care about.
- **Balanced training, realistic evaluation.** XGBoost trains on a balanced ~2.5%-fraud sample (trees learn poorly at 1:1000) but is evaluated at the true ~0.1% rate.

And the result's shape — embeddings lose alone, win combined — is the level's takeaway restated: the foundation model contributes *sequential context*, the raw features contribute *precision*; the system wants both.

---

## The handoff table (commit this to memory)

| # | Block | Consumes | Produces | Lives in |
|---|-------|----------|----------|----------|
| 1 | Dataset | raw CSV | temporal parquet splits | notebook 01 |
| 2 | Tokenizer | rows | 12 tokens/row, vocab 6,251 | [`src/tokenizer/`](../../src/tokenizer) |
| 3 | Corpus | tokens + grouping | `*_corpus.txt`, ~4,096-token lines | notebook 02 → [`pipeline.to_corpus_lines`](../../src/tokenizer/pipeline.py) |
| 4 | Pretraining | corpus | HF checkpoint | [`configs/`](../../configs) + [`scripts/`](../../scripts) + NeMo |
| 5 | Embeddings | checkpoint + histories | `.npy` vectors + row IDs | [`src/decoder_inference.py`](../../src/decoder_inference.py), notebook 04 |
| 6 | Evaluation | vectors + raw + labels | metrics comparison | notebook 05 |

**Next:** [Level 300 — The Pipeline in Code](level-300-the-pipeline-in-code.md): the same six blocks, now at the level of actual functions, files, and tensors.
