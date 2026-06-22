# Level 500: The Code Anatomy — Notebooks, Classes, Parallelism, and the Port Surface

*Half a day, with the source open in one pane and this page in the other. Assumes [Level 400](level-400-design-contracts-and-extensions.md). This is the deepest data-side level: after it, the notebooks and `src/tokenizer/` should read like code you could have written — and rewritten for a different dataset.*

**Scope.** Everything from raw file to training-ready tensors: the notebook flow, the tokenizer classes method by method, the GPU parallelization model, and exactly which pieces are *engine* (reusable) versus *template* (TabFormer-specific). Model training internals (NeMo recipe, FSDP, optimization) are deliberately out of scope — they get their own deep dive later.

**What Level 500 adds over 300/400.** Level 300 told you *where* each stage lives; Level 400 told you *which contracts* hold it together. Level 500 reads the code the way a reviewer would: every method's job, every GPU↔CPU boundary, every place the notebooks quietly rely on a design property — plus three findings you can only get from reading the source (a vocabulary ID collision, an inert masking path, and the exact chunk arithmetic). The point is not trivia: this repo is a **template built around one synthetic IBM dataset (TabFormer)**, and you can only re-instantiate a template safely once you understand why each line is the way it is.

Contents:

1. [The notebook flow — and the production pipeline hiding in it](#part-1--the-notebook-flow-and-the-production-pipeline-hiding-in-it)
2. [The class anatomy: `src/tokenizer/`, method by method](#part-2--the-class-anatomy-srctokenizer-method-by-method)
3. [The parallelization model: how to think in GPU](#part-3--the-parallelization-model-how-to-think-in-gpu)
4. [The port surface: rebuilding it for your dataset](#part-4--the-port-surface-rebuilding-it-for-your-dataset)

---

## Part 1 — The notebook flow, and the production pipeline hiding in it

### 1.1 The artifact DAG

The five notebooks are not a monolith — they are five jobs that communicate **only through files on disk**. Draw the artifacts and the production pipeline appears:

```
data/TabFormer/raw/card_transaction.v1.csv          (~24M rows; downloaded from IBM Box)
   │
   │  NB01 §1–2   day-granular temporal cutoffs (80/10/10)
   ▼
data/TabFormer/temporal_split/{train,val,test}.parquet          (raw columns, unmodified)
   │                          {val_eval,test_eval}.parquet      (100K stratified eval subsets)
   │
   │  NB02 §2     preprocess → fit → transform → to_corpus_lines   (independently per split)
   ▼
data/decoder_corpus/{train,val,test}_corpus.txt                 (~64K lines for train)
   │
   │  NB03        subprocess → scripts/train_decoder_model.py      (training: later level)
   ▼
models/decoder-demo/            (30-step toy)     models/decoder-foundation-model/  (shipped, LFS)
   │
   │  NB04        per-transaction encode → extract_embeddings_batched
   ▼
data/embeddings/{split}_embeddings.npy  +  {split}_labels.npy  +  {split}_row_ids.npy
   │
   │  NB05        row-id join + PCA(64) + OrdinalEncoder → 3× XGBoost
   ▼
data/outputs/*.png  +  the AP/AUC verdict
```

Three structural facts worth internalizing:

1. **Every notebook starts by asserting its inputs exist** (`assert p.exists(), "... run notebook 01 first"`) and **guards its outputs** (`if corpus_path.exists(): ... continue`, `if not TGZ_PATH.exists(): urlretrieve(...)`). That is a hand-rolled `make`: re-running a notebook is cheap and idempotent, and any notebook can be re-run alone if its upstream artifacts are present.
2. **No Python state crosses a notebook boundary.** Each notebook re-imports, re-reads parquet, and (in 02 and 04) re-instantiates the tokenizer pipeline from scratch. This works because of contract C2 — configuration is the only state — and it is precisely what makes the notebook→script conversion mechanical.
3. **One step already ships as a production script**: training. Notebook 03 is a thin `subprocess` wrapper around [`scripts/train_decoder_model.py`](../../scripts/train_decoder_model.py) + a YAML. That is the pattern the other steps would follow.

The same DAG as a production pipeline (names invented, mapping exact):

| Pipeline step | Notebook source | Inputs | Outputs | Parallelizable over |
|---|---|---|---|---|
| `ingest.py` | NB01 §0 | URL | raw CSV | — |
| `make_splits.py` | NB01 §1–2 | raw CSV | 3 + 2 parquets | — (one global pass) |
| `build_corpus.py --split X` | NB02 §2 | one parquet | one corpus .txt | splits, and entity shards within a split (§3.1) |
| `train.py` | NB03 → existing script | corpus + YAML | HF checkpoint | GPUs (torchrun) |
| `extract_embeddings.py --split X` | NB04 §2 | checkpoint + parquet | npy triplet | splits, batches |
| `evaluate.py` | NB05 | npy + parquets | metrics, plots | model variants |

What stops you from running this as-is in Airflow today is exactly one thing: **configuration lives in cell constants, duplicated**. `MERCHANT_HASH_SIZE = 2000` is retyped in NB02, NB04, [`src/clm_data.py`](../../src/clm_data.py)'s defaults, and [`configs/pretrain_financial_decoder.yaml`](../../configs/pretrain_financial_decoder.yaml); `CHUNK_SIZE = 315`, `MAX_LENGTH = 128`, the seed-42 balanced-sample recipe (NB01, NB04, NB05 — see §1.5) are similarly scattered. They agree today by discipline, not by mechanism. The first production hardening is one config object (or one YAML anchor) that every step reads — that single change turns contract C1 from "remember to keep them in sync" into "cannot diverge."

### 1.2 Notebook 01, cell by cell: establishing the data contract

NB01's deliverables are the **temporal split parquets** (everything downstream), the **eval subsets** (NB04/05), and a tuned **raw-feature XGBoost baseline** (the number NB05 must beat). The cells, in execution order, with the design intent of each:

**Download guard.** `urlretrieve` from IBM Box → untar → `card_transaction.v1.csv`. Both steps are skipped if their target exists.

**Load and date assembly.** `cudf.read_csv` puts ~24M rows on the GPU in seconds. The date column is assembled by *string concatenation* (`Year + '-' + Month.zfill(2) + '-' + Day.zfill(2)`) then `cudf.to_datetime` — all GPU kernels, no Python row loop. Notice what is *not* used: the `Time` column. Splits are day-granular on purpose (next cell).

**`find_cutoff_date(gdf, target_ratio)` — the split logic.** Groupby date → daily counts → sort → `cumsum` → first date where cumulative count ≥ target. Two deliberate properties: (a) the cutoff is found by *row mass*, not by calendar arithmetic, so an 80/10/10 split stays 80/10/10 even if transaction volume grows over the years; (b) cutoffs land on day boundaries, so all of a day's transactions stay on one side — no intra-day leakage ambiguity. Only the one-row cutoff result moves to host (`.head(1).to_pandas()`); the boolean masks that apply the cutoffs run on GPU.

**Parquet write — before feature engineering.** The split frames are written with their **raw columns** (only the helper `date` dropped). The `Hour`/`Amount`-as-float/`_target` engineering happens *after* the write, in-memory, for the baseline only. This ordering is load-bearing: NB02 and NB04 re-read these parquets and run their own `preprocess()`. The parquets are the **single source of raw truth**; every consumer derives its own view. If NB01 had saved engineered columns, the tokenizer's and the baseline's notions of "the data" could silently diverge.

**The `to_pandas()` boundary.** XGBoost itself trains on GPU (`device='cuda'`), but the sklearn pieces (`OrdinalEncoder`, `train_test_split`) are host libraries — so the frames cross to pandas once, after all heavy row-level work is done. This is the recurring seam pattern: **GPU for the millions-of-rows stages, host for the model-fitting stages that need the CPU ecosystem** (see §3.2).

**`create_balanced_train_sample` — the 1M training sample.** All fraud rows capped at 10% of the budget (TabFormer train has ~24K fraud → cap inactive → ~2.4% fraud), the rest sampled from normals, `np.random.seed(42)`, concatenate, shuffle. Memorize the *shape* of this recipe — seed, fraud-first, `min()` caps, concat order, shuffle — because NB04 and NB05 **re-derive this exact sample independently** rather than reading it from disk (§1.5).

**`stratified_subsample` + the eval-parquet save.** Val/test eval sets are 100K stratified samples (preserving the ~0.1% fraud rate — you *evaluate* at deployment prevalence, you *train* at balanced prevalence). The save is the first appearance of **row-identity bookkeeping**, the theme of contract C6: `X_val.index` (positional indices surviving `train_test_split`) → `raw_full.iloc[subset_idx]` → `val_eval.parquet`. It works because `val_df` came from `to_pandas()` with a fresh `RangeIndex` that matches parquet row order. Fragile if you reorder anything in between — which is exactly why NB04 switches to explicit `__row_id__` columns (§1.5).

### 1.3 Notebook 02, cell by cell: the tokenizer exercised, then the corpus

NB02 has two halves: a **pedagogical** half (§1.1–1.6: one transaction through the machinery, GPT-2 comparison, edge cases) and a **production** half (§2: the corpus loop). The pedagogical half is also a free test suite — every claim in it is executable.

**The single-row demo (§1.2) quietly proves the determinism contract.** It builds a one-row cuDF frame, runs `preprocess → fit → transform`, and prints 12 tokens. Stop and ask: how can you *fit* a tokenizer on **one row**? Because in the default configuration, `fit()` learns nothing from data — every step's `build_vocab()` constructs its vocabulary from constructor arguments alone (§2.3). `fit()` on one row and `fit()` on 19.5M rows produce bit-identical vocabularies. The demo is the proof-by-execution of contract C2.

**The GPT-2 comparison (§1.3–1.5)** tokenizes the same transaction with GPT-2 BPE (~50K vocab → ~40 tokens) versus the domain tokenizer (12 tokens), then converts the ratio into the only currency that matters: **transactions of history per 4,096-token context window** (~315 vs ~90). This is the design argument for building a tokenizer at all, stated as arithmetic.

**The edge-case helper (§1.6)** (`quick_tokenize(overrides)`) re-runs single rows with one field perturbed: `$8,500 → AMT_6`, online transaction → `STATE_ONLINE`, `$0.50 → AMT_0`. Note it calls `pipeline.transform` with the *already-fitted* pipeline — transform is reusable; fit was needed once.

**The corpus loop (§2) — read it as the production job it almost is:**

```python
for split_name, parquet_path, corpus_path in splits:
    if corpus_path.exists(): continue                  # idempotency guard
    gdf = cudf.read_parquet(str(parquet_path))
    pip = FinancialTokenizerPipeline(merchant_hash_size=MERCHANT_HASH_SIZE)
    gdf_proc = pip.preprocess(gdf)
    pip.fit(gdf_proc)                                  # ← legal ONLY because nothing is learned
    token_df = pip.transform(gdf_proc)
    corpus_lines = pip.to_corpus_lines(token_df, gdf_proc, group_cols, chunk_size=CHUNK_SIZE)
    # write lines
```

Four things to internalize:

1. **A fresh pipeline is constructed and `fit()` per split.** Under the default (deterministic) config this is merely tidy. Switch on a *fitted* strategy — `amount_strategy="quantile"` — and this same loop becomes a **two-headed bug**: each split learns its own bin boundaries (train/val/test token semantics silently diverge), and fitting on val/test is information leakage. The correct fitted-strategy shape is: fit on train once, persist with `get_state()`, restore with `from_state()` for the other splits (§2.2). The notebook doesn't do this because it doesn't need to — but the loop's structure *assumes* determinism, and the assumption is invisible until you change a constructor flag.
2. **The `group_cols` sniffing loop** (`for col_name in ["user", "User", "cust"]: ...`) is a template-adaptation hook: it discovers the entity columns by name across schema variants. After `preprocess()` lower-cases column names, it resolves to `["user", "card"]` — one token stream per *card*, not per customer. (A user with 4 cards contributes 4 independent behavioral stories. Defensible — card behavior is coherent; spending patterns differ across cards — but it's a choice you revisit per dataset: for wallets, `group_cols=["entity"]`.)
3. **The chunk arithmetic is exact, not approximate.** A chunk of *n* transactions costs `12n` field tokens + `(n−1)` separators + `<bos>` + `<eos>` = `13n + 1` tokens. The largest *n* with `13n + 1 ≤ 4096` is **n = 315, which gives exactly 4096**. Full chunks waste zero context. The general formula for porting: `chunk_size = (seq_length − 1) // (tokens_per_txn + 1)`. Level 300's "⌊4096/13⌋ ≈ 315" is the right intuition; this is the exact form — and it matters because an off-by-one here silently truncates every sequence's last transaction at training time (contract C3).
4. **Only the tail chunk of each (user, card) stream is short.** `cumcount() // chunk_size` assigns chunk IDs; every chunk except the last per group has exactly 315 transactions. Short tails become short corpus lines, which become padded training sequences — see the `-100` finding in §2.5 for where that padding actually lands in the loss.

### 1.4 Notebooks 03–05 in one screen (flow only — training is a later level)

**NB03** asserts the corpora exist, then `subprocess.run`s the training script with `--checkpoint.checkpoint_dir models/decoder-demo/checkpoints` and 30 steps — a smoke test that produces *mush* weights, plus a printed `torchrun` command for the real 8-GPU/3,000-step run. The flow-level point: training was *already* factored out of the notebook into [`scripts/train_decoder_model.py`](../../scripts/train_decoder_model.py) + YAML. Demo writes `decoder-demo/`; NB04/05 read the LFS-shipped `decoder-foundation-model/` (Level 400, sharp edge #2).

**NB04** is the subtlest of the five. Per split it: reads the parquet → extracts labels **before** `preprocess()` (which renames and re-orders columns) → for train, **re-derives the seed-42 balanced 1M sample** → injects `gdf["__row_id__"] = np.arange(len(gdf))` → `preprocess()` (re-sorts!) → reads back `row_ids = gdf["__row_id__"]` (now in post-sort order) and re-indexes labels by them → `transform()` → **`pipeline.encode(token_df, max_length=128)`** → batched GPU inference → saves the `{embeddings, labels, row_ids}` triplet.

Two facts here change how you should think about the whole system:

- **The embeddings are per-transaction, not per-history.** `pipeline.encode()` is the *row-level* encoder: each transaction becomes `<bos> + 12 field tokens + <eos>` padded to 128 — there is **no `<sep>`, no history** in the inference input. The model was *pretrained* on 4,096-token histories but is *applied* to 14-real-token sequences; what survives into the embedding is what pretraining baked into the token representations, not the customer's recent context. (Level 400, sharp edge #3, now with the mechanism visible.) Extracting *history-aware* embeddings — encode the user's last N transactions, pool at the final position — is a high-value experiment the current code makes easy: the corpus-style path already exists in `to_corpus_lines` + `FinancialTabularTokenizer.encode`.
- **The `__row_id__` pattern is the canonical alignment fix.** NB01's "index survives `to_pandas()`" trick is implicit; NB04's explicit ID column survives *any* reordering, including `preprocess()`'s sort. When you port, copy NB04's pattern, not NB01's.

**NB05** loads the npy triplets, re-derives (seed 42 again) the same balanced sample from `train.parquet`, and aligns raw features to embeddings with `…loc[balanced_idx].reset_index().iloc[train_row_ids]` — the RNG recipe gets the same 1M rows, the row-ids re-create the post-sort order. Then PCA(512→64) fit on train only, ordinal-encode raw, and three XGBoost variants (raw / embeddings / combined) with the comparison hygiene Level 300 §G covers.

Note the coupling: NB04 and NB05 each contain a copy of the NB01 sampling recipe, and correctness depends on all three never diverging (same seed, same 10% cap, same concat order, same shuffle). In a production refactor this becomes one artifact: `make_splits.py` writes `balanced_train_idx.npy` once and everyone reads it. (The repo's choice is defensible pedagogy — each notebook stays runnable standalone — but recognize it as a contract C6 instance enforced by copy-paste.)

---

## Part 2 — The class anatomy: `src/tokenizer/`, method by method

### 2.0 The architecture in one diagram

```
                         BaseTokenizer (base.py)                        ← the step contract
                         build_vocab() · tokenize() · get_state()/from_state()
                                      ▲
        ┌──────────────┬──────────────┼──────────────┬──────────────┐
  FixedVocabTokenizer  MappingTokenizer  CategoricalHashTokenizer  NumericalTokenizerOptBin  TimeDeltaTokenizer
  (bounded ints)       (finite sets/ranges)  (unbounded categoricals)  (fitted bins, cuML)   (log time bins)
                                      ▲  registered into
                         TokenizerPipeline (pipeline.py)               ← the ENGINE (dataset-agnostic)
                         add_step · fit · transform · encode · to_corpus_lines · global vocab/offsets
                                      ▲  subclassed by
                         FinancialTokenizerPipeline (financial_pipeline.py)   ← the TEMPLATE (TabFormer)
                         _configure_steps() = the 12 steps · preprocess() = raw → step inputs
                                      ▲  wrapped by
                         FinancialTabularTokenizer (financial_tokenizer.py)   ← the LM-facing ADAPTER
                         encode/decode/vocab_size/pad_token_id — what clm_data.py and NeMo consume
```

Hold this split in your head for Part 4: **engine** files you will never modify when porting; **template** files you will rewrite entirely; the **adapter** you copy and re-point. The notebooks orchestrate; they own no tokenization logic.

### 2.1 `BaseTokenizer` — a two-method contract plus a persistence protocol

Every step is: configuration in `__init__`, **data only ever flows through `build_vocab()` and `tokenize()`**. The base class owns three mechanisms:

- **`_idx_to_token` is the single source of truth.** A step's vocabulary is one dict `{local_index: token_string}`. The reverse map (`vocab` property) is derived lazily by inversion; `vocab_size` is its length. Subclasses never manage both directions.
- **A nested-vocab convention you should know exists**: if `_idx_to_token` values are themselves dicts (`{sub_key: {idx: token}}`), `vocab_size` sums the sub-dicts and the pipeline assigns each sub-vocab its own offset (`f"{tok_id}.{sub}"`). **No shipped step uses this** — it's an affordance for one step emitting multiple token families (e.g. a single step producing both `AMT_*` and `AMTSIGN_*`). You'll see the `isinstance(..., dict)` branches it causes throughout `pipeline.py`; now you know what they're for.
- **The persistence protocol**: `get_state()` returns `{class, init_params, vocab_state, fitted_state}`; `from_state()` reconstructs without re-fitting. Look at who actually *has* fitted state: `NumericalTokenizerOptBin` (the cuML discretizer's learned boundaries) and `TimeDeltaTokenizer` (its precomputed bin edges). The deterministic steps return `{}` — for them, `init_params` *is* the state. **The shipped flow never calls `get_state()`** — determinism makes persistence unnecessary. The moment you enable a fitted strategy, this protocol stops being dead code and becomes mandatory infrastructure (fit on train → `get_state()` → ship the JSON next to the corpus → `from_state()` everywhere else).

### 2.2 The five step classes — reference card, then the four traps

| Class | Field shape | `build_vocab` | `tokenize` mechanics | Vocab size | OOV behavior |
|---|---|---|---|---|---|
| [`FixedVocabTokenizer`](../../src/tokenizer/fixed_vocab.py) | bounded ints (HOUR, DOW, MONTH, CARD, ZIP3, CUST) | enumerate `[min_val, max_val]` — no data | `astype(int32).clip(min,max).map(dict)` — GPU | `max−min+1` | **silent saturation** to boundary token |
| [`MappingTokenizer`](../../src/tokenizer/mapping.py) | known finite sets (CHIP, STATE, MCC) or int ranges (CAT) | from `mapping`/`values`/`ranges` config | direct: `to_pandas().map(dict)` — **host**; range: numpy masked fill — **host** | `len(labels)+default` | default label (`XX`, `UNK`, `-1`, `GENERAL`) |
| [`CategoricalHashTokenizer`](../../src/tokenizer/categorical_hash.py) | unbounded categoricals (MERCH) | enumerate `0..limit−1` — no data | `column % vocab_limit → map` — GPU; **expects pre-hashed ints** | `vocab_limit` | none possible — every hash lands in a bucket |
| [`NumericalTokenizerOptBin`](../../src/tokenizer/numerical.py) | continuous values | names bins **and fits cuML `KBinsDiscretizer`** ← the one data-driven `build_vocab` | `builder.transform → map` — GPU | `num_bins` | clipped into edge bins by discretizer |
| [`TimeDeltaTokenizer`](../../src/tokenizer/timedelta.py) | inter-event seconds | enumerate bins; boundaries precomputed in `__init__` | `clip(0, 10y) → log1p → cp.digitize` — GPU (CuPy) | `num_bins` | clamped to horizon |

The four traps the reference card can't show:

**Trap 1 — `FixedVocabTokenizer` never produces `<unk>`.** Out-of-range values are `clip`ped: user 5,000 becomes `CUST_2999`, hour 25 becomes `HOUR_23`. Garbage in your integer column doesn't crash and doesn't surface — it impersonates a legitimate boundary value. The flip side: in the entire default pipeline, **no step can emit an out-of-vocabulary token string** (fixed vocab clips, mapping defaults, hash mods). Therefore if you ever see `<unk>` in encoded output, it is not "a rare value" — it is a **config-mismatch smoke alarm**: the encoding vocab was built with different constructor arguments than the corpus (wrong `merchant_hash_size`, missing step, reordered steps). Treat any nonzero `<unk>` count as a contract C1 violation, full stop.

**Trap 2 — `MappingTokenizer` has a hidden fourth mode that breaks determinism.** The three documented modes (direct dict, passthrough `values` list, integer `ranges`) all build the vocab from configuration. But construct it with *neither* `mapping`, `values`, nor `ranges`, and `build_vocab(column_data)` falls into discovery mode: it enumerates the **unique values found in the data**. That silently converts contract C2 from "deterministic" to "fitted" — re-fit on a different split and the same merchant state can get a different token ID. The financial template never triggers this; a porter passing `MappingTokenizer(prefix="VENUE")` with no value list *will*, and NB02-style per-split fitting then produces three incompatible vocabularies. Always pass an explicit `values=[...]` (the chain example in [the universal recipe](../04-data/08-from-raw-data-to-training-run.md) does this correctly: `values=["IN", "OUT"]`).

**Trap 3 — `CategoricalHashTokenizer` doesn't hash.** Read `tokenize()`: it's `column_data % self.vocab_limit`. The hashing happened upstream, in `preprocess()`: `df["merch_hash"] = merch_clean.hash_values()` — cuDF's GPU MurmurHash3 over the *cleaned* merchant string. Two consequences. First, the bucket assignment is `murmur3(clean_name) % 2000` — fully deterministic with no fitted dictionary, which is *why the merchant "vocabulary" also needs no data* (the hash trick is determinism's answer to unbounded cardinality). Second, the **string cleaning is part of the vocabulary**: change the regex in `preprocess()` (`[^A-Z0-9\s\-]`) and the same merchant hashes to a different bucket — an invisible contract C1 break that no vocab-size check will catch. The cleaning regex is as load-bearing as the hash size.

**Trap 4 — `amount_strategy="quantile"` re-bins already-binned data.** `preprocess()` *always* computes `amt_val` as the fixed-threshold ordinal (the sum-of-comparisons, values 0–6). With `amount_strategy="quantile"`, `_configure_steps()` attaches `NumericalTokenizerOptBin` to that same `amt_val` column — so the cuML discretizer fits its quantile bins **on the ordinals 0–6, not on raw dollar amounts**. You get up to 7 distinguishable "quantile" bins no matter what `amount_bins` says, with boundaries that are artifacts of the threshold bins. To use a data-driven amount strategy for real you must also change `preprocess()` to put the raw float in the column (`df["amt_val"] = amt_f`) — and then you've taken on the fitted-artifact burden of Trap 2/§2.1 as well. As shipped, the flag is wired but the data path for it is not. (This is the general lesson: `preprocess()` and `_configure_steps()` are **one design, split across two methods** — every step's input column has a producing block in `preprocess()`, and they must be changed together.)

### 2.3 `TokenizerPipeline` — the engine, method by method

[`pipeline.py`](../../src/tokenizer/pipeline.py) is ~350 lines and dataset-agnostic. Its job: turn an ordered set of steps into (a) one global vocabulary and (b) three output formats.

**`add_step(column_name, tokenizer)`.** The step's registry key (`tok_id`) is the column name itself (multi-column steps join with `_` — another unused-but-present affordance; `column_specs` would hand a multi-column step a DataFrame instead of a Series). Consequence of keying by column: **renaming a column in `preprocess()` renames the step**, which renames its offset entry, which relabels part of the global ID space. Column names are part of contract C1.

**`fit(df)` → `_fit_sequential` / `_fit_parallel`.** Both do the same two-phase accounting; the parallel variant (taken when `len(steps) ≥ stream_threshold=5`, so always for the 12-step template) launches each step's `build_vocab` on its own CUDA stream first, barriers, then packs offsets (§3.1 dissects the streams). Phase one: each step builds its local vocab (and the missing-column check raises here — porting errors surface at `fit`, loudly, as `ValueError: Columns {...} not found`). Phase two: walk `tokenizer_order`, assign each step `vocab_offset[tok_id] = current_offset`, advance by `vocab_size`. **Registration order is therefore the memory layout of the vocabulary** — same steps in a different order = same token strings with different IDs = a checkpoint-incompatible tokenizer that produces zero errors (contract C1's mechanism, seen from inside).

**`_build_global_vocab()`.** Specials first (`<pad>`=0, `<bos>`=1, `<eos>`=2, `<sep>`=3, `<unk>`=4 — insertion order of the `SPECIAL_TOKENS` dict), then for each step: `gid = int(local_idx) + offset`. The full layout for the default template — worth printing and pinning above your desk, because *this table is the embedding matrix's row map*:

| Step (column) | Class / prefix | Tokens | Global IDs |
|---|---|---:|---|
| specials | — | 5 | 0–4 |
| `amt_val` | FixedVocab `AMT` | 7 | 5–11 |
| `merch_hash` | Hash `MERCH` | 2,000 | 12–2011 |
| `mcc_int` | Mapping(ranges) `CAT` | 14 | 2012–2025 |
| `mcc_str` | Mapping(values) `MCC` | 110 | 2026–2135 |
| `hour` | FixedVocab `HOUR` | 24 | 2136–2159 |
| `dow` | FixedVocab `DOW` | 7 | 2160–2166 |
| `month` | FixedVocab `MONTH` | 12 | **2168–2179** ⚠️ |
| `card` | FixedVocab `CARD` | 10 | **2179–2188** ⚠️ |
| `chip_upper` | Mapping(dict) `CHIP` | 4 | 2189–2192 |
| `zip3` | FixedVocab `ZIP3` | 1,000 | 2193–3192 |
| `state_clean` | Mapping(values) `STATE` | 58 | 3193–3250 |
| `cust` | FixedVocab `CUST` | 3,000 | 3251–6250 |
| **Total** | | **6,251 strings** | **6,250 distinct IDs** |

⚠️ **Finding: `MONTH_12` and `CARD_0` share global ID 2179, and ID 2167 is an unused hole.** Root cause, visible in two lines: `FixedVocabTokenizer.build_vocab` uses the *raw values* as local indices (`{i: f"MONTH_{i:02d}" for i in range(1, 13)}` — keys 1–12, because `min_val=1`), while the offset packer advances by *count* (`vocab_size` = 12). So MONTH occupies `offset+1 … offset+12` instead of `offset+0 … offset+11`, overrunning by one into CARD's `offset+0`. Every step with `min_val > 0` shifts its IDs up by `min_val`; MONTH is the only such step in the template. Verify it yourself in the container:

```python
from src.tokenizer import FinancialTabularTokenizer
t = FinancialTabularTokenizer()
print(t.vocab["MONTH_12"], t.vocab["CARD_0"])     # 2179 2179  ← same embedding row
print(len(t.vocab), len(t.id_to_token))           # 6251 6250  ← one ID double-booked
print([i for i in range(6251) if i not in t.id_to_token])   # [2167]  ← the hole
```

Consequences, precisely: the model **cannot distinguish December from card 0** at the embedding level (they share a row — though the rigid 12-token rhythm means position usually disambiguates them in context, which is why training visibly works anyway); `decode(2179)` always prints `CARD_0` even when the encoded token was `MONTH_12`; embedding row 2167 is allocated and never trained; nothing crashes, nothing warns. And the reason to **document rather than hot-fix**: the shipped LFS checkpoint was *trained with this layout* — "fixing" the tokenizer (e.g. `gid = local_idx − min_val + offset`) shifts all twelve MONTH IDs and breaks compatibility with the existing model (contract C1 cuts both ways). Fix it when you next retrain from scratch; until then, the correct move is a **golden test that pins the defect**: `assert len(t.id_to_token) == len(t.vocab) - 1` today, flipped to `==` after the retrain. For porters this is an active landmine, not history: any `FixedVocabTokenizer(min_val=1)` in *your* pipeline collides its top token with the next step's first ID (the chain example in [guide 08](../04-data/08-from-raw-data-to-training-run.md) has `MONTH, min_val=1` directly before `TDIF` — same bug, `MONTH_12 ≡ TDIF_0`). Until the engine is fixed, the porting rule is mechanical: **all `FixedVocabTokenizer` steps use `min_val=0`** (shift in `preprocess()` if needed: `df["month0"] = dt.dt.month - 1`).

**`transform(df)`.** Same stream choreography as fit; each step's token Series lands in `local_results[tok_id]`, then `cudf.concat(parts, axis=1)`. The concat aligns **by index** — which is why every step's `tokenize` carefully preserves the input index (`cudf.Series(result, index=column_data.index)`), including the two host-roundtrip paths in `MappingTokenizer`. Output: one DataFrame, one column per step, one row per transaction, all strings. Column order = `tokenizer_order` = the within-transaction token order in the corpus.

**`encode(token_df, max_length=4096, add_special=True)` — the row-level path.** Builds `(n_rows, max_length)` int64: `<bos>`, one ID per *column* (so 12 IDs), `<eos>`, pad. Each **row is a transaction**; there is no grouping and no `<sep>`. Per column it does `to_pandas().map(vocab).fillna(unk)` — a host roundtrip per column (acceptable: 12 columns; see §3.2) — and the `fillna(unk)` is where the Trap-1 smoke alarm would ring. Silent truncation if `max_length` is too small (`if col_offset >= max_length: break` drops columns, then the `<eos>` write is skipped) — with 12 steps you need ≥ 14; NB04's 128 leaves room for added steps. This is what NB04 calls — re-read §1.4's "per-transaction embeddings" with this method in mind.

**`to_corpus_lines(token_df, df_meta, group_cols, chunk_size=315)` — the sequence-level path.** Five moves: (1) `str.cat` the 12 token columns with spaces — GPU, one string per transaction; (2) take `group_cols` from `df_meta` — **`df_meta` must be the same frame you transformed**, same index, same (sorted!) row order, or the group labels pair with the wrong token strings; (3) `cumcount() // chunk_size` → `_chunk_id` — position within the group, which is *temporal* position only because `preprocess()` sorted by time (the sort is doing silent work here); (4) `groupby(group_cols + ["_chunk_id"]).agg(list)` — GPU list-aggregation; (5) `.to_pandas()`, then a Python `"<bos> " + " <sep> ".join(txns) + " <eos>"` per chunk — the one host loop, over ~64K chunks, not 19.5M rows. Returns a Python list of strings; the notebook writes them as lines. (At 100× scale, this list and the text file are what you'd replace first — §3.2.)

**`fit_transform(df)`** is the two-call composition, nothing more.

### 2.4 `FinancialTokenizerPipeline` — the template

Two methods *are* the template; everything else is inherited.

**`_configure_steps()`** registers the 12 steps of §2.3's table, in that order, plus two opt-ins (`amount_strategy`, `include_time_delta`). The constants above it (`KNOWN_MCCS` — the 110 MCC codes present in TabFormer, `INDUSTRY_RANGES` — the 13 ISO-style MCC bands, `CHIP_MAPPING`, `ALL_STATES` — 50 states + DC + territories + `XX` + `ONLINE`, `AMOUNT_THRESHOLDS`) are **the dataset's domain knowledge, lifted into configuration** — exactly the part you replace wholesale when porting. Note the redundant-by-design encodings: `mcc_int → CAT_*` (13 coarse industries) *and* `mcc_str → MCC_*` (110 fine codes) tokenize the *same source field* at two granularities — a hierarchy the model can exploit (rare MCC, common industry), bought for 124 vocab entries and 2 of the 12 tokens. That's a pattern, not an accident: when one field carries both a coarse and a fine signal, spend two tokens.

**`preprocess(df)`** — static, pure cuDF, the **other half of every step's definition** (Trap 4). Block by block, each with its porting lesson:

| Block | Code essence | What to notice |
|---|---|---|
| column normalize | `strip().replace(" ","_").lower()` | downstream names (`group_cols`, step keys) depend on this |
| amount → `amt_val` | sum of 6 boolean `(amt_f >= t)` casts | branch-free GPU binning; **duplicates** `AMOUNT_THRESHOLDS` semantics in code — change one, change both |
| merchant → `merch_hash` | regex-clean → `hash_values()` | the regex is part of the vocab (Trap 3) |
| MCC → `mcc_int`, `mcc_str` | `fillna(-1)`, int and str views | `-1` is the explicit missing token, present in `KNOWN_MCCS` |
| timestamp → `hour/dow/month` | string-assemble → `to_datetime` → `.dt` accessors | `Time` has minutes only; `fillna("00:00")` defends missing |
| `card`, `cust` | `astype(int).clip(0, 9)` / `.clip(0, 2999)` | clip = Trap 1: user 3000+ *becomes* user 2999 |
| zip → `zip3` | `fillna("00000")`, strip `.0` artifact, first 3 digits | defensive against float-typed ZIPs — real-world dirt handled at the door |
| state → `state_clean` | `fillna("XX").upper().strip()`, empty→`XX` | `ONLINE` is a legitimate "state" in TabFormer |
| **the sort** | `sort_values(["user","card","time_full"])` | **row order changes here** — the reason `__row_id__`/row-ID joins exist (C6); also what makes `cumcount` temporal |
| time delta | `groupby(["user","card"]).diff()` → seconds, `fillna(0).clip(0)` | first event of each card = delta 0; computed even when the TDIF step is off (cheap, harmless) |

Read the table column-wise and the porting recipe writes itself: every block answers "which raw columns exist, how dirty are they, what does the step need." That's why [guide 08](../04-data/08-from-raw-data-to-training-run.md)'s chain example is recognizably the same function with different field names.

### 2.5 `FinancialTabularTokenizer` + `clm_data.py` — the LM-facing adapter and the training boundary

**Why the adapter exists.** Training and inference code (NeMo's dataset builder, `HuggingFaceDecoderInference`) speak the HuggingFace tokenizer dialect: `encode(text) → List[int]`, `decode`, `vocab_size`, `pad_token_id`. The pipeline speaks cuDF. [`FinancialTabularTokenizer`](../../src/tokenizer/financial_tokenizer.py) bridges by **constructing the pipeline only to harvest its vocabulary** — `_build_vocab_from_pipeline()` calls every step's `build_vocab()` *with no data* (the deepest consequence of determinism: the full 6,251-entry vocab materializes from constructor args on any machine, GPU dataframes never touched at vocab-build time) — then re-implements the offset packing locally. Note that duplication: `_build_vocab_from_pipeline` is a copy of `_fit_sequential`'s accounting, so an engine fix (e.g. the `min_val` bug) must land in **both** places or training and corpus vocabularies silently diverge — the nastiest possible C1 break, since both sides are "fixed."

**`encode(text, max_length)`** is pure Python: `text.split()` → truncate → **pad with `"<pad>"` strings to `max_length`** → `vocab.get(token, unk)` per token. Host CPU, per corpus line, paid once at dataset-load time. `decode` drops pads and joins. Plain and slow-ish by design — this path processes 64K lines, not 19.5M rows.

**[`clm_data.py`](../../src/clm_data.py)** is the corpus→tensor boundary (the last data-side code; training consumes its output). `load_corpus_and_tokenize` instantiates the adapter, encodes every line, and wraps the list in `FinancialCLMDataset`, whose `__getitem__` builds the `{input_ids, labels}` dict with `-100` at unused label positions (contract C4). Also note the `sys.path.insert` at module top: NeMo's file-path `_target_` resolution imports this file *outside* the package, so the tokenizer import has to be self-arranged — keep that hack when you copy the file.

⚠️ **Finding: the `-100` masking is inert in the shipped flow.** Trace the lengths: `encode(line, max_length=seq_length)` **pre-pads every sequence to exactly `seq_length`** with pad IDs. So in `__getitem__`, `len(tokens) == seq_length` always, both `input_ids[: len(tokens)] = tokens` and `labels[: len(tokens)] = tokens` overwrite their full buffers, and **no position ever keeps the `-100` fill** — pad positions carry `pad_id` (0) as their label instead. Consequence: `MaskedCrossEntropy` masks nothing, and the model is *trained to predict `<pad>` after `<eos>`* on every short sequence (only tail chunks are short — a few percent of positions on TabFormer; the model learns "emit `<pad>` forever after `<eos>`" almost immediately, so the practical cost is a small amount of wasted compute and a slightly flattered loss). The two-line fix if you care (`encode` without `max_length`, let `__getitem__` do the padding — its code already handles it correctly); the reason to care more on *your* dataset: if your entities are short-history (many short sequences), the pad fraction stops being negligible and starts diluting the loss signal. A `assert (labels == -100).any()` on one batch is the golden test.

**The two encode paths, side by side** (Level 400 sharp edge #3, now fully mechanical):

| | Training path | NB04 inference path |
|---|---|---|
| Code | `FinancialTabularTokenizer.encode` (str → ids) | `TokenizerPipeline.encode` (DataFrame → ids) |
| Input unit | one **corpus line** = up to 315 transactions | one **transaction** = one row |
| Sequence | `<bos> 12tok (<sep> 12tok)×314 <eos>` = 4096 | `<bos> 12tok <eos>` + 114 pads = 128 |
| History context | up to 315 transactions | **none** |
| Where it runs | host CPU at dataset load | host map per column, GPU model after |

Same vocabulary, same model — different *meaning* of the resulting hidden states. Any embedding experiment must state which path produced its vectors.

---

## Part 3 — The parallelization model: how to think in GPU

This is the section to internalize rather than memorize: the *reasoning pattern* NVIDIA's engineers applied, because it's the pattern you reuse when your dataset is 10× bigger.

### 3.1 Three nested layers of parallelism

**Layer 1 — rows: vectorize inside each kernel (where ~all the speed lives).** There is no Python loop over transactions anywhere in the hot path. Every operation you've seen — `str.replace`, `hash_values`, boolean-sum binning, `to_datetime`, `clip`, `map(dict)` (a device-side lookup join), `%`, `digitize`, `groupby().diff()`, `cumcount`, `agg(list)` — is one or a few libcudf/CuPy kernels sweeping millions of rows with thousands of GPU threads. The unit of *work* is the column; the unit of *vectorization* is the row. When you write your own `preprocess()`, the discipline is: **if you're about to write `for` or `.apply(lambda …)`, find the column-level verb instead** — the 19.5M-row stages of this repo run in seconds entirely because every verb is columnar.

**Layer 2 — steps: overlap independent columns with CUDA streams (opportunistic).** `fit` and `transform` both use the same choreography when `len(steps) ≥ stream_threshold` (5):

```python
streams = [cp.cuda.Stream(non_blocking=True) for _ in self.tokenizer_order]
for stream, tok_id in zip(streams, self.tokenizer_order):
    tokenizer.stream = stream          # swap in (steps hold an optional stream attr)
    with stream:                       # route launches issued under this context
        local_results[tok_id] = tokenizer.tokenize(col_data)
    tokenizer.stream = original        # swap back
cp.cuda.Stream.null.synchronize()      # sync point before results are consumed
```

Why this is *correct*: steps read disjoint columns and write disjoint results, so there are no cross-stream data hazards, and results are only consumed after the synchronize. Why it's *opportunistic* rather than guaranteed speedup: a CUDA stream is a queue — kernels in different streams *may* execute concurrently when resources allow. CuPy ops (the `TimeDelta` math, hash bucketing arithmetic) reliably honor the active stream context; cuDF's internal kernels may or may not issue on it depending on the library build, and any step that detours through pandas (the `MappingTokenizer` host paths) leaves the GPU entirely. So read the design's intent precisely: **correctness never depends on the streams; they exist to overlap many small independent kernels and hide launch latency where the libraries cooperate.** That's also the explanation for `stream_threshold=5` — below a handful of steps, stream setup/swap overhead isn't worth the possible overlap — and for why the per-step work is deliberately small and independent: twelve ~tens-of-milliseconds kernels are exactly the workload where overlap pays, whereas one giant GPU-saturating kernel would gain nothing from streams. When you add steps, keep them column-pure (no cross-column reads inside a step) and the stream machinery extends to them for free.

**Layer 3 — data: shard by entity, because determinism removed the only global dependency.** Here is the deepest design insight in the repo, and it's invisible until you look for it. A *learned* vocabulary (BPE, frequency-ranked merchants, quantile bins) requires a **global pass over all data before any shard can be encoded** — corpus statistics are a synchronization barrier. This pipeline's vocabulary is derived from configuration, so that barrier **does not exist**: any process on any machine can construct the identical vocab and start encoding immediately. The only remaining whole-dataset operations are the temporal cutoff search (one cheap groupby in NB01) and the `groupby(user, card)` for sequence assembly — and the latter *partitions perfectly by entity*. Which means the scale-out story is embarrassingly parallel: NB02's per-split loop is already 3 independent jobs; within a split, shard rows by `hash(user) % N` (each shard gets complete entities), run the identical `preprocess → fit → transform → to_corpus_lines` per shard on its own GPU (the natural dask-cudf or one-process-per-shard layout), and concatenate corpus files at the end. **No coordination, no shared state, byte-identical vocab everywhere.** Determinism (contract C2) isn't just a reproducibility nicety — it is the parallelization strategy.

### 3.2 The GPU↔host seams — where the data leaves the device, and why that's fine

Honest accounting of every host transfer in the data path:

| Seam | Where | Data size at the seam | Verdict |
|---|---|---|---|
| `MappingTokenizer._tokenize_direct` | `to_pandas().map(dict)` per mapping step | full column (~19.5M strings, ×3 steps) | the real Layer-1 exception: a per-row host map; fine at this scale (seconds), first thing to GPU-ify (cudf `merge`/categoricals) at 10× |
| `MappingTokenizer._tokenize_range` | `.values.get()` → numpy masked fill | full int column ×1 step | same family; 13 vectorized numpy passes, cheap |
| `pipeline.encode` | `to_pandas().map(vocab)` per column | eval subsets (NB04: 1.2M rows × 12 cols) | bounded by inference workload anyway |
| `to_corpus_lines` tail | grouped lists → pandas → Python join | ~64K chunks (not 19.5M rows!) | the aggregation already shrank the data 300×; host is fine |
| corpus file write / read | Python text I/O | GB-scale for train (~62K lines × ~32 KB) | replace with parquet at scale |
| `clm_data` encode | pure-Python split+lookup per line | 64K lines, once at load | the everything-in-RAM ceiling (L400 #5) |
| NB01/NB05 `to_pandas()` | before sklearn/XGBoost-input prep | 1M–19.5M rows, once | host ML ecosystem boundary; deliberate |

The pattern behind the verdicts is **Amdahl reasoning at the data level**: GPU work happens where the row count is huge (per-transaction transforms); host work is tolerated where prior aggregation already shrank the data (per-sequence, per-chunk, per-column-map). When you port, hold that line: it's fine to do host work *after* a groupby that reduced 19.5M to 64K; it's a 100× slowdown to do host work *before* it. The repo's two deviations (the Mapping host maps) survive only because TabFormer has exactly four mapping steps and 19.5M rows is small for a modern GPU — they're marked "first to fix" above for a reason.

### 3.3 What executes where — the whole DAG, device by device

| Stage | Device | Parallel axis | Sync points |
|---|---|---|---|
| CSV/parquet read | GPU (cuDF I/O) | — | — |
| `preprocess()` | GPU | rows (kernels) | the sort (global op) |
| `fit()` | GPU, trivial work | steps (streams) | null-stream sync |
| `transform()` | GPU (+ Mapping host seams) | rows × steps | null-stream sync |
| `to_corpus_lines()` | GPU until agg, then host | groups | the groupby |
| corpus write | host | splits | — |
| `clm_data` load/encode | host CPU | lines (single process today) | — |
| training | GPU(s) | torchrun DP | NeMo's business (later level) |
| NB04 encode+inference | host map → GPU model | batches | per-batch H2D/D2H |

---

## Part 4 — The port surface: rebuilding it for your dataset

[Guide 08](../04-data/08-from-raw-data-to-training-run.md) is the step-by-step recipe (follow it when you actually port — including to the [ZKAI Embed datasets](../04-data/09-zkai-internal-datasets.md), where the field mapping is already drafted for next-trade prediction). This section is the recipe's *code-level justification*: which files each decision touches, and the pitfalls only visible from Part 2.

### 4.1 File-by-file port classification

| File | Class | When porting to a new dataset |
|---|---|---|
| `base.py`, `pipeline.py` | **engine** | don't touch (except to fix the `min_val` offset bug — in which case fix `financial_tokenizer.py`'s copy of the packing too, and retrain) |
| `fixed_vocab.py`, `mapping.py`, `categorical_hash.py`, `numerical.py`, `timedelta.py` | **engine** (step library) | reuse as-is; add new step classes here only for genuinely new field *shapes* (rare — the five cover almost everything) |
| `financial_pipeline.py` | **template** | rewrite wholesale: your constants, your `_configure_steps()`, your `preprocess()` — keep the *structure* (it is the documentation) |
| `financial_tokenizer.py` | **adapter** | copy, re-point the pipeline constructor, rename; ~10 changed lines |
| `clm_data.py` | **adapter** | copy, swap the tokenizer import and entry-point name; the `{input_ids, labels}` shape (C4) needs nothing |
| notebooks 01–02 | **orchestration** | become your `make_splits.py` / `build_corpus.py` (§1.1's table) |
| `configs/*.yaml` | **config** | copy; change `vocab_size`, `_target_`, paths (guide 08 step 6) |

### 4.2 The six decisions, tied to the code they touch

1. **Entity & event** → `group_cols` in `to_corpus_lines` and the sort keys in `preprocess()`. (TabFormer: `(user, card)`; wallets: `entity`. The grouping *is* the identity model — remember the template also puts identity *in the vocab* as `CUST_*`, which you will almost certainly not copy: Level 400 sharp edge #1, guide 09's "no wallet-identity token" rule.)
2. **Fields → strategies** → one `add_step` per field in `_configure_steps()` *plus its producing block* in `preprocess()` — always changed as a pair (Trap 4). Use §2.2's reference card as the decision table.
3. **Tokens per event** → `chunk_size = (seq_length − 1) // (tokens_per_event + 1)` — exact, not approximate (§1.3). Recount every time you add/remove a step (C3).
4. **Vocab arithmetic** → sum the per-step sizes by hand, then `assert YourTabularTokenizer().vocab_size == <hand_sum>` — and also `assert len(t.id_to_token) == len(t.vocab)` to catch `min_val`-style collisions (§2.3) before they're trained into a checkpoint.
5. **Determinism stance** → all-deterministic steps (explicit `values=`, fixed thresholds, hash buckets) keeps the per-split/per-shard `fit()` pattern legal and the Layer-3 sharding free. Any fitted step (quantile bins, discovery-mode mappings) → fit on train only, persist via `get_state()`, distribute the state file, and accept that your corpus job now has a global dependency.
6. **Inference encode path** → decide *per-transaction* (NB04 path: simple, no history) vs *per-history* (corpus-style: what the model was actually pretrained on) **before** running embedding experiments, and record it next to the results (§2.5's table).

### 4.3 The porting pitfalls, collected (all derived in Part 2)

- `FixedVocabTokenizer(min_val>0)` → ID collision with the next step (§2.3). Use `min_val=0`, shift in `preprocess()`.
- `MappingTokenizer` with no `values`/`mapping`/`ranges` → silent data-driven vocab, breaks C2 (Trap 2). Always enumerate.
- New step without its `preprocess()` block → loud `ValueError` at fit (good). Right column, wrong dtype/range → **silent clip saturation** (Trap 1) — assert value ranges in `preprocess()` instead.
- Changed string-cleaning before `hash_values()` → silently re-buckets every categorical (Trap 3). The regex is vocab.
- Reordered/renamed steps or columns → same tokens, different IDs (C1). Pin with a golden vocab-layout test.
- Forgot to recompute `chunk_size` → every sequence truncates mid-event (C3).
- Short-history entities + the pre-padding `encode` → pad tokens trained as targets at scale (§2.5). Fix the two lines or accept the dilution knowingly.
- Re-sorted rows + positional joins → misaligned labels (C6). Copy NB04's `__row_id__` pattern verbatim.
- Duplicated constants across notebook/scripts → centralize before the first real run (§1.1).

**The golden tests worth writing before any retrain** (an afternoon, repays itself the first time anything drifts): exact token output for 3 fixed sample rows; `vocab_size` + a hash of the sorted `(token, id)` pairs; `len(id_to_token) == len(vocab)` (collision sentinel); one corpus line's token count and `<sep>` count; `(labels == -100).any()` on one training batch; `<unk>` count == 0 on a real split.

---

## The Level-500 summary

> The notebooks are a production DAG communicating through disk artifacts, runnable as five scripts the moment configuration is centralized. The tokenizer is a deterministic engine (`pipeline.py` + five step classes) wearing a TabFormer costume (`financial_pipeline.py`): vocabulary *constructed* from config, never learned — which is simultaneously the reproducibility story, the no-artifact deployment story, and the embarrassingly-parallel scale-out story. The costume swaps via two methods (`preprocess`, `_configure_steps`) that form one design split across two places. And because you read the source: you know `MONTH_12` and `CARD_0` share embedding row 2179, that the `-100` mask never fires, that the embeddings NB05 evaluates saw exactly one transaction each, and that 13 × 315 + 1 = 4096 — exactly.

**Next:** port it — [the universal recipe](../04-data/08-from-raw-data-to-training-run.md) with [your dataset](../04-data/README.md), tracked [with Loom](../06-experimentation/01-loom-workflow.md). The model-training deep dive (NeMo recipe, distributed internals, scaling) is the planned companion to this level.
