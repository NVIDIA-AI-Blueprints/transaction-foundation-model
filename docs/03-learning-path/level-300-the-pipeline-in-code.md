# Level 300: The Pipeline in Code

*60 minutes, best spent with the notebooks open. Assumes [Level 200](level-200-the-building-blocks.md). By the end you can navigate every file in `src/`, trace a transaction from CSV to embedding, and launch training yourself.*

We follow one transaction all the way through:

```python
{"Amount": "$42.75", "Merchant Name": "3527213246127876953", "MCC": 5411,
 "Year": 2025, "Month": 1, "Day": 15, "Time": "09:32", "Card": 0,
 "Use Chip": "Chip Transaction", "Zip": "95113.0", "Merchant State": "CA", "User": 1001}
```

---

## Stage A — Preprocessing: raw columns → pipeline-ready columns

**Code:** [`FinancialTokenizerPipeline.preprocess()`](../../src/tokenizer/financial_pipeline.py) (a `@staticmethod`, ~70 lines, pure cuDF).

`preprocess` normalizes the raw frame and derives the exact intermediate columns the tokenizer steps expect. Representative moves:

```python
df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

# Amount "$42.75" → float → cumulative-threshold bin 0..6
amt = df["amount"].astype(str).str.replace("$", "", regex=False)
amt_f = amt.astype(float)
df["amt_val"] = ((amt_f >= 10).astype("int32") + (amt_f >= 50).astype("int32")
               + (amt_f >= 100).astype("int32") + (amt_f >= 500).astype("int32")
               + (amt_f >= 1000).astype("int32") + (amt_f >= 5000).astype("int32"))

# Merchant: uppercase, strip punctuation, GPU hash
merch_clean = (df["merchant_name"].astype(str).str.upper()
                 .str.replace(r"[^A-Z0-9\s\-]", "", regex=True))
df["merch_hash"] = merch_clean.hash_values()

# Timestamp → hour / day-of-week / month
dt = cudf.to_datetime(date_str, format="%Y-%m-%d %H:%M")
df["hour"], df["dow"], df["month"] = dt.dt.hour, dt.dt.dayofweek, dt.dt.month
```

Three details that reward attention:

1. **The amount binning is a sum of boolean comparisons** — branch-free and vectorized on GPU. `$42.75` clears only the `>=10` threshold → `amt_val = 1` → eventually `AMT_1`. The thresholds `[0,10,50,100,500,1000,5000]` are `AMOUNT_THRESHOLDS` at the top of the file.
2. **Defensive normalization everywhere**: `Zip "95113.0" → "951"` (strip float artifact, take 3 digits, zero-pad), missing states → `"XX"`, `User` clipped to `[0, 2999]`. Real-world schemas are dirty; the tokenizer contract is clean.
3. **The chronological sort happens here** — `df.sort_values(["user", "card", "time_full"])` — plus a per-card `time_delta_s` (seconds since previous transaction) computed via `groupby(...).diff()`. The sort is what makes Stage C's sequences temporal; the delta feeds the optional `TimeDeltaTokenizer`.

Our row exits as: `amt_val=1, merch_hash=<uint64>, mcc_int=5411, mcc_str="5411", hour=9, dow=2, month=1, card=0, chip_upper="CHIP TRANSACTION", zip3=951, state_clean="CA", cust=1001`.

## Stage B — Tokenization: columns → token strings

**Code:** [`TokenizerPipeline`](../../src/tokenizer/pipeline.py) (the engine) + [`FinancialTokenizerPipeline._configure_steps()`](../../src/tokenizer/financial_pipeline.py) (the 12-step recipe) + the step classes ([`fixed_vocab.py`](../../src/tokenizer/fixed_vocab.py), [`mapping.py`](../../src/tokenizer/mapping.py), [`categorical_hash.py`](../../src/tokenizer/categorical_hash.py), [`numerical.py`](../../src/tokenizer/numerical.py), [`timedelta.py`](../../src/tokenizer/timedelta.py), all subclassing [`base.py`](../../src/tokenizer/base.py)).

The pipeline is a registry of named steps bound to columns:

```python
self.add_step("amt_val", FixedVocabTokenizer(prefix="AMT", min_val=0, max_val=6))
self.add_step("merch_hash", CategoricalHashTokenizer(vocab_limit=2000, special_token="MERCH"))
self.add_step("mcc_int", MappingTokenizer(prefix="CAT", ranges=INDUSTRY_RANGES, default="GENERAL"))
...  # 12 steps total; +TimeDeltaTokenizer if include_time_delta=True
```

The lifecycle is sklearn-shaped — `fit` → `transform` — with two pipeline-level responsibilities:

- **`fit(df)`** calls each step's `build_vocab()`, then assigns each step an **offset** into the global ID space (specials occupy 0–4, then steps in registration order) and assembles the global `vocab` / `id_to_token` dicts. When there are ≥5 steps, fitting runs on parallel **CUDA streams** (`_fit_parallel`) — independent per-column work overlapped on GPU.
- **`transform(df)`** runs each step's `tokenize()` and returns a **DataFrame of token strings**, one column per step:

```python
token_df = pipeline.fit(processed).transform(processed)
# columns: amt_val, merch_hash, mcc_int, mcc_str, hour, dow, month, card, ...
# row 0:   AMT_1   MERCH_667  CAT_RETAIL MCC_5411 HOUR_09 DOW_2 MONTH_01 CARD_0 ...
```

Each step is trivially small — `FixedVocabTokenizer.tokenize` is two lines (`clip` then `map`); `MappingTokenizer` handles three config shapes (value list, dict, range table); `CategoricalHashTokenizer` is `hash % vocab_limit`. **This is the main extension surface**: a new field = a new step (or a config tweak), nothing else changes. Steps also implement `get_state()/from_state()` for serialization.

> Notebook 02 §1.2–1.6 runs exactly this, including the GPT-2 comparison and edge cases ($8,500 → `AMT_6`; online transactions → `STATE_ONLINE`-ish defaults).

## Stage C — Corpus generation: token strings → training text

**Code:** [`TokenizerPipeline.to_corpus_lines()`](../../src/tokenizer/pipeline.py).

```python
lines = pipeline.to_corpus_lines(
    token_df,                      # output of transform()
    df_meta=processed,             # carries the grouping columns
    group_cols=["user", "card"],   # one stream per card
    chunk_size=315,                # ≈ 4096 / 13 tokens-per-txn
)
```

Internally: concatenate the 12 token columns per row (`str.cat(sep=" ")`), `groupby(group_cols).cumcount() // chunk_size` to assign chunk IDs, aggregate each chunk's rows to a list, then format:

```python
"<bos> " + " <sep> ".join(txn_list) + " <eos>"
```

Notebook 02 §2 applies this per split and writes `data/decoder_corpus/{train,val,test}_corpus.txt`. Inspect your work — this is the model's entire sensory world:

```bash
head -c 400 data/decoder_corpus/train_corpus.txt
```

## Stage D — The training dataset: text lines → tensors

**Code:** [`src/clm_data.py`](../../src/clm_data.py) — the file NeMo's YAML points at.

Two layers. `load_corpus_and_tokenize()` instantiates the **deterministic** tokenizer interface and encodes each line:

```python
tokenizer = FinancialTabularTokenizer(merchant_hash_size=2000, ...)
token_ids = tokenizer.encode(line, max_length=seq_length)   # pads to 4096
```

[`FinancialTabularTokenizer`](../../src/tokenizer/financial_tokenizer.py) is worth understanding: it wraps the pipeline but builds the **vocab purely from configuration** (`_build_vocab_from_pipeline` calls each step's `build_vocab()` with no data). That's why the training job — and any future inference service — reconstructs the *identical* 6,251-entry vocab with zero fitted artifacts. It exposes the LM-standard API: `encode/decode/vocab_size/pad_token_id/...`.

Then `FinancialCLMDataset.__getitem__` produces the training dict ([explained in Primer 3](../02-concepts/03-causal-language-modeling.md)):

```python
input_ids[: len(tokens)] = tokens          # pad with pad_token_id
labels[: len(tokens)] = tokens             # pad positions stay -100
return {"input_ids": ..., "labels": ...}   # shift happens inside the model
```

The YAML hook is the file-path `_target_`:

```yaml
dataset:
  _target_: src/clm_data.py:build_financial_clm_dataset
  data_path: null          # ← you pass this on the CLI
  merchant_hash_size: 2000 # must match the corpus's tokenizer config
  seq_length: 4096
```

NeMo resolves the path, calls the function with the YAML keys as kwargs, and wraps the result in a `StatefulDataLoader`. Note `build_financial_clm_dataset(..., **kwargs)` swallows unknown keys — forward-compatible with new config fields.

## Stage E — Training: config + launcher + NeMo

**Code:** [`configs/pretrain_financial_decoder.yaml`](../../configs/pretrain_financial_decoder.yaml) + [`scripts/train_decoder_model.py`](../../scripts/train_decoder_model.py).

The config's sections, mapped to what they control (the model block is dissected in [Primer 4](../02-concepts/04-decoder-architecture.md)):

| YAML block | Controls | Demo values worth knowing |
|---|---|---|
| `model` | architecture (LlamaConfig, from scratch) | vocab 6251, hidden 512, 8 layers |
| `dataset` / `validation_dataset` | the Stage-D builder | `seq_length: 4096` |
| `step_scheduler` | batch & duration | `global_batch_size: 16`, **`max_steps: 30`** ← demo! |
| `distributed` | FSDP2 manager | `tp_size: 1`, dp inferred |
| `loss_fn` | `MaskedCrossEntropy` | honors the `-100` labels |
| `optimizer` / `lr_scheduler` | AdamW, lr 2e-4, cosine, 10 warmup steps | |
| `checkpoint` | HF-format consolidated safetensors | → `models/decoder-demo/checkpoints/` |

The launcher is deliberately boring — parse config, print a banner, run the recipe:

```python
cfg = parse_args_and_load_config()
recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
recipe.setup()
recipe.run_train_validation_loop()
```

Launch commands (also in the config header):

```bash
# single GPU (sanity check)
python scripts/train_decoder_model.py -c configs/pretrain_financial_decoder.yaml \
    --dataset.data_path data/decoder_corpus/train_corpus.txt

# 8 GPUs — how the shipped checkpoint was made (~3,000 steps)
torchrun --nproc-per-node=8 scripts/train_decoder_model.py \
    -c configs/pretrain_financial_decoder.yaml \
    --dataset.data_path data/decoder_corpus/train_corpus.txt \
    --validation_dataset.data_path data/decoder_corpus/val_corpus.txt \
    --step_scheduler.max_steps 3000
```

Expectations for the 30-step demo (notebook 03): loss starts near ln(6251) ≈ 8.7 and should reach ~5–6 in ~2 minutes. The demo writes to `models/decoder-demo/`; notebooks 04–05 read the **LFS-shipped** `models/decoder-foundation-model/` instead — keep the two separate in your head.

## Stage F — Embedding extraction: checkpoint → vectors

**Code:** [`src/decoder_inference.py`](../../src/decoder_inference.py); driven by notebook 04.

Loading needs only HuggingFace (NeMo's job ended at the checkpoint):

```python
infer = HuggingFaceDecoderInference(
    model_path="models/decoder-foundation-model",
    tokenizer=FinancialTabularTokenizer(merchant_hash_size=2000),
    pooling="last_token",
)
```

The forward path is three moves — mask, hidden states, pool ([mechanics in Primer 5](../02-concepts/05-embeddings.md)):

```python
attention_mask = (input_ids != tokenizer.pad_token_id).long()
outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
hidden = outputs.hidden_states[-1]                      # (B, T, 512)
seq_lengths = attention_mask.sum(dim=1) - 1             # last real index per row
emb = hidden[torch.arange(B), seq_lengths, :]           # (B, 512)
```

For the 1.2M-row extraction, `extract_embeddings_batched()` adds throughput engineering you can reuse anywhere: **pinned host memory** (`.pin_memory()`) for async H2D copies, accumulation in a preallocated **GPU buffer** with a single D2H copy at the end, `batch_size=4096`, and a tqdm progress bar.

## Stage G — The comparison: vectors + raw → verdict

**Code:** notebook 05 (orchestration only — no `src/` involvement).

The mechanics that make the comparison *fair* are the teachable part:

1. **PCA 512 → 64** on embeddings (fit on train only) to de-noise and tame dimensionality for trees.
2. **Alignment via `*_row_ids.npy`** — raw features are re-ordered to match the embedding rows exactly (the tokenizer's sort changed row order; this is the guard).
3. **Identical splits everywhere** — the same 1M balanced train sample (deterministic seed) and the same 100K stratified val/test parquets from notebook 01.
4. **Independent Optuna HPO per variant** — baseline, embeddings-only, combined each get their own tuned hyperparameters; otherwise the comparison favors whichever variant the shared settings happened to suit. Early stopping on val AUC; no `scale_pos_weight` (the ~1:39 balanced train ratio needs no reweighting).
5. **Report both ROC-AUC and AP** on the untouched future test split; AP carries the conclusion.

---

## The whole trace, one screen

```
CSV row ($42.75, Walmart, 5411, 2025-01-15 09:32, CA, user 1001)
 │ preprocess()                 financial_pipeline.py   amt_val=1, hour=9, dow=2, zip3=951…
 │ fit()/transform()            pipeline.py + steps     "AMT_1 MERCH_667 … CUST_1001"
 │ to_corpus_lines()            pipeline.py             "<bos> … <sep> … <eos>"  (×315 txns)
 │ encode() → __getitem__       clm_data.py             {input_ids[4096], labels[4096]}
 │ recipe.run_…loop()           NeMo AutoModel + YAML   checkpoint (HF safetensors)
 │ extract_embeddings_batched() decoder_inference.py    (N, 512) float32 + row_ids
 ▼ PCA + XGBoost                notebook 05             AP: combined > raw > embeddings-only
```

**Next:** [Level 400 — Design Contracts & Extensions](level-400-design-contracts-and-extensions.md): the invariants that hold this together, the known sharp edges, and how to modify each piece without breaking the others.
