# Glossary

Every term you'll meet in these docs, grouped by theme. Skim now, return often. Terms with a dedicated primer link to it.

## Foundation models & training

- **Foundation model (FM)** — A model pretrained on large amounts of unlabeled data with a generic objective, intended to be *reused* across many downstream tasks (via embeddings, fine-tuning, or prompting). See [the primer](../02-concepts/01-foundation-models.md).
- **Pretraining** — The initial, expensive, label-free training phase. Here: next-token prediction over transaction sequences.
- **Fine-tuning** — Continuing training of a pretrained model on a (usually labeled, smaller) task-specific dataset. This repo *doesn't* fine-tune — it extracts embeddings instead; fine-tuning is a research direction ([improvement ideas](../05-research/02-improvement-ideas.md)).
- **Self-supervised learning (SSL)** — Training where the supervision signal is derived from the data itself (e.g., "predict the next token", "fill in the masked value") rather than human labels.
- **Causal language modeling (CLM)** — The "predict the next token from everything before it" objective. *Causal* = can only look backward in the sequence. See [the primer](../02-concepts/03-causal-language-modeling.md).
- **Masked language modeling (MLM)** — Alternative objective (used by BERT-family models): hide random tokens, predict them from both directions. Not used here, but common in the literature (e.g., IBM's original TabFormer model, Revolut's PRAGMA).
- **Transfer learning** — Reusing what a model learned on one task/dataset for another. Embedding extraction is transfer learning's simplest form.
- **Downstream task** — The task you actually care about (here: fraud detection), as opposed to the pretraining objective.
- **Checkpoint** — Saved model weights (+ config) at some training step. Stored here in HuggingFace format under [`models/decoder-foundation-model/`](../../models/decoder-foundation-model).
- **Epoch / step / batch** — One pass over the dataset / one optimizer update / the set of examples in one update. The config's `global_batch_size: 16` means 16 sequences (~65K tokens) per step.
- **Learning-rate schedule / warmup** — How the optimizer step size changes over training. This repo: AdamW, cosine decay, 10 warmup steps ([config](../../configs/pretrain_financial_decoder.yaml)).
- **Loss (cross-entropy)** — The number training minimizes; for CLM, how surprised the model is by each true next token. Random guessing over a 6,251-token vocabulary gives loss ln(6251) ≈ 8.74.
- **Perplexity** — exp(loss); "how many tokens is the model effectively choosing between". Lower is better.

## Tokens & sequences

- **Token** — The atomic unit a language model reads/predicts. Here, tokens are *domain-defined*, like `AMT_3` ("amount in $100–500") or `MERCH_1498` ("merchant hash bucket 1498"). See [the primer](../02-concepts/02-tokenization-and-vocabularies.md).
- **Tokenizer** — The deterministic mapping from raw data to tokens. This repo's is built from modular per-field steps ([`src/tokenizer/`](../../src/tokenizer)).
- **Vocabulary** — The fixed set of all tokens the model knows; here ~6,251 including 5 special tokens.
- **Special tokens** — Structural markers: `<pad>` (id 0), `<bos>` begin-sequence (1), `<eos>` end-sequence (2), `<sep>` between transactions (3), `<unk>` unknown (4).
- **BPE (Byte-Pair Encoding)** — The subword tokenization used by GPT-style text models. Notebook 02 shows why BPE is a poor fit for transactions (a 19-digit merchant ID explodes into ~10 meaningless fragments).
- **Sequence / context window** — The maximum number of tokens the model sees at once. Trained at 4,096 here (~315 transactions); the architecture supports 8,192 via RoPE.
- **Corpus** — The training text files, one sequence per line: `data/decoder_corpus/*.txt`.
- **Padding** — Filling short sequences up to fixed length with `<pad>`; padded positions are excluded from loss via the label value `-100`.

## Model architecture

- **Transformer** — The neural architecture underlying modern language models; processes all positions in parallel through *attention* layers. See [the primer](../02-concepts/04-decoder-architecture.md).
- **Decoder-only** — The GPT/Llama-style transformer variant where each position can only attend to earlier positions — exactly matching the next-token objective.
- **Attention / self-attention** — The mechanism by which each token gathers information from other tokens, with learned weighting.
- **Hidden state / hidden size** — The vector each layer computes per token position; here 512-dimensional (hence 512-d embeddings).
- **Llama** — Meta's open decoder architecture family. This repo borrows the *architecture config* (not the weights!) at toy scale via `transformers.LlamaConfig`.
- **RoPE (Rotary Position Embeddings)** — How the model knows token *positions*; allows some extrapolation beyond the trained length (trained 4,096 → supports 8,192).
- **GQA (Grouped-Query Attention)** — Efficiency trick: 8 query heads share 2 key/value heads (4:1), cutting memory at little quality cost.
- **SwiGLU / RMSNorm** — The modern activation function and normalization used by Llama-style models (vs GELU/LayerNorm in GPT-2).
- **safetensors** — HuggingFace's safe, fast weight-file format used by the checkpoint.

## Embeddings & evaluation

- **Embedding** — A fixed-size vector representing variable-size input; here, 512 floats summarizing a transaction history. See [the primer](../02-concepts/05-embeddings.md).
- **Pooling (last-token / mean)** — How per-position hidden states collapse to one vector. Last-token pooling (default here) takes the hidden state at the final non-pad position ([`src/decoder_inference.py`](../../src/decoder_inference.py)).
- **PCA** — Linear dimensionality reduction; notebook 05 compresses 512-d → 64-d embeddings before XGBoost.
- **UMAP** — Nonlinear 2D/3D projection used in notebook 04 to *visualize* the embedding space.
- **AUROC (ROC-AUC)** — Probability a random fraud scores above a random non-fraud. Near-saturated on this dataset; insufficient alone.
- **Average Precision (AP / AUPRC)** — Area under precision-recall; the headline metric at ~0.1% fraud prevalence, where precision in the top of the ranking is what matters operationally.
- **Temporal split** — Train on the past, validate/test on the future (80/10/10 by date here) — the only honest split for time-ordered data.
- **Leakage** — Any way information from evaluation data (or the future, or the label) sneaks into training features. Watch for it constantly; see [Level 400's sharp edges](../03-learning-path/level-400-design-contracts-and-extensions.md#3-sharp-edges-read-before-deploying-or-publishing-numbers).
- **HPO (hyperparameter optimization)** — Automated search over model settings; notebook 05 uses **Optuna**-tuned XGBoost parameters.

## The GPU stack (NVIDIA)

> Full primer: [The GPU Stack](../02-concepts/06-gpu-stack.md)

- **NeMo / NeMo AutoModel** — NVIDIA's training framework; AutoModel is the part that trains any HuggingFace-compatible model with distributed training, checkpointing, and recipes driven by YAML configs.
- **HuggingFace Transformers** — The de-facto standard model-zoo library; this repo trains with NeMo but *saves* HuggingFace-format checkpoints so anything (including plain `transformers`) can load them.
- **RAPIDS** — NVIDIA's GPU data-science suite. **cuDF** = GPU pandas, **cuML** = GPU scikit-learn, **CuPy** = GPU NumPy.
- **NGC** — NVIDIA GPU Cloud, the registry hosting the NeMo container image (`nvcr.io/nvidia/nemo:25.09.01`).
- **FSDP2 (Fully Sharded Data Parallel)** — PyTorch's strategy for splitting model state across GPUs; how NeMo AutoModel scales from 1 GPU to multi-node.
- **torchrun** — PyTorch's launcher for multi-GPU/multi-node processes (`torchrun --nproc-per-node=8 …`).
- **Git LFS (Large File Storage)** — Git extension storing big binaries (the model checkpoint) outside normal git history.
- **`_target_` (in YAML)** — NeMo AutoModel's way to point config at a Python class or function to instantiate, e.g. `_target_: src/clm_data.py:build_financial_clm_dataset`.

## Data & domain

- **TabFormer** — IBM's synthetic credit-card dataset: ~24.4M transactions, 2,000 users, 2002–2019, ~0.12% fraud. See [the dataset page](../04-data/01-tabformer.md).
- **MCC (Merchant Category Code)** — Standard 4-digit code for merchant type (5411 = grocery stores).
- **ZIP3** — First 3 digits of a US ZIP code; a coarse, privacy-friendlier region.
- **EVM (Ethereum Virtual Machine)** — The execution environment of Ethereum and many compatible chains (Polygon, Arbitrum, Optimism, Celo, …). EVM chains share a common data schema (blocks, transactions, logs, traces).
- **SVM (Solana Virtual Machine)** — Solana's execution environment; different data model (instructions within transactions, token balances).
- **BigQuery** — Google Cloud's serverless SQL warehouse; hosts public datasets of full blockchain histories. See [the data guides](../04-data/README.md).
- **Hubble** — The Stellar Development Foundation's public analytics dataset on BigQuery.

## Experimentation

- **Loom** — [ZKAI-Network/loom](https://github.com/ZKAI-Network/loom): the agentic CLI we use to run the experiment lifecycle (ingest → EDA → features → search → train → validate → deploy) with cost gates and lineage. See [the Loom guide](../06-experimentation/01-loom-workflow.md).
- **Metaflow** — The MLOps backend Loom uses for versioned runs and artifacts.
- **AIDE** — The tree-search "brain" Loom uses to propose and refine candidate solutions against a declared metric.
- **Run / artifact / lineage** — A tracked execution / its stored outputs / the recorded chain of what produced what.
