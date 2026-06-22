import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <!--
    SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: Apache-2.0

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Notebook 02: Sequential Preprocessing and Tokenization

    This notebook builds a **custom domain-specific tokenizer** for financial transactions and generates tokenized corpora for foundation model pretraining. Instead of treating each transaction as an independent row, we serialize a customer's transaction history into a sequence of domain-specific tokens so a language model can learn sequential patterns (spending velocity, geographic shifts, recurring merchants).

    1. **Custom tokenization** -- why domain-specific tokens outperform general-purpose BPE tokenizers for financial data.
    2. **Corpus generation** -- convert the temporal splits from notebook 01 into text sequences for foundation model pretraining.
    """)
    return


@app.cell
def _():
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(".").resolve()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    TEMPORAL_SPLIT_DIR = PROJECT_ROOT / "data" / "TabFormer" / "temporal_split"
    TRAIN_DATA = TEMPORAL_SPLIT_DIR / "train.parquet"
    VAL_DATA   = TEMPORAL_SPLIT_DIR / "val.parquet"
    TEST_DATA  = TEMPORAL_SPLIT_DIR / "test.parquet"

    for p in [TRAIN_DATA, VAL_DATA, TEST_DATA]:
        assert p.exists(), f"Missing temporal split: {p}  (run notebook 01 first)"

    CORPUS_DIR = PROJECT_ROOT / "data" / "decoder_corpus"
    CORPUS_PATH      = CORPUS_DIR / "train_corpus.txt"
    VAL_CORPUS_PATH  = CORPUS_DIR / "val_corpus.txt"
    TEST_CORPUS_PATH = CORPUS_DIR / "test_corpus.txt"

    from src.tokenizer import FinancialTokenizerPipeline, FinancialTabularTokenizer

    print(f"Project Root      : {PROJECT_ROOT}")
    print(f"Temporal Splits   : {TEMPORAL_SPLIT_DIR}")
    print(f"  Train           : {TRAIN_DATA}")
    print(f"  Val             : {VAL_DATA}")
    print(f"  Test            : {TEST_DATA}")
    print(f"Output Corpora    : {CORPUS_DIR}")
    print(f"  Train Corpus    : {CORPUS_PATH}")
    print(f"  Val Corpus      : {VAL_CORPUS_PATH}")
    print(f"  Test Corpus     : {TEST_CORPUS_PATH}")
    return (
        CORPUS_DIR,
        CORPUS_PATH,
        FinancialTabularTokenizer,
        FinancialTokenizerPipeline,
        TEST_CORPUS_PATH,
        TEST_DATA,
        TRAIN_DATA,
        VAL_CORPUS_PATH,
        VAL_DATA,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. Tokenization Deep Dive

    Standard LLM tokenizers are optimized for natural language. For tabular financial data, they often fail to capture semantic meaning efficiently:
    - **Amounts**: `$123.45` might be split into `['$', '123', '.', '45']`, losing magnitude information.
    - **Merchants**: `Walmart #8832` might be split into subwords, losing the entity identity.
    - **Timestamps**: Dates are split into arbitrary numbers.

    Our **Financial Tokenizer** uses domain-specific logic:
    - **Amount Binning**: Maps amounts to log-scale bins (e.g., `AMT_3` = $100-500).
    - **Merchant Hashing**: Hashes merchant names to a fixed vocabulary (e.g., `MERCH_1234`).
    - **Temporal Encoding**: Encodes time as `HOUR_XX`, `DOW_X`, `MONTH_XX`.
    - **Location Decomposition**: Splits location into `ZIP3_xxx` (region) + `STATE_xx`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.1 Initialize Tokenizer

    We initialize our custom tokenizer and load a standard GPT-2 BPE tokenizer for comparison.
    """)
    return


@app.cell
def _(FinancialTabularTokenizer, FinancialTokenizerPipeline):
    import torch
    import cudf
    from transformers import AutoTokenizer

    pipeline = FinancialTokenizerPipeline(merchant_hash_size=2000)
    fin_tokenizer = FinancialTabularTokenizer(merchant_hash_size=2000)

    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"Financial Vocab Size: {fin_tokenizer.get_vocab_size()}")
    print(f"  Pipeline steps    : {len(pipeline.tokenizer_order)}")
    print(f"  Step names        : {pipeline.tokenizer_order}")
    print(f"GPT-2 Vocab Size    : {gpt2_tokenizer.vocab_size}")
    return cudf, gpt2_tokenizer, pipeline


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2 Sample Transaction

    Tokenize a single transaction using the TabFormer schema to see the pipeline output.
    """)
    return


@app.cell
def _(FinancialTokenizerPipeline, cudf, pipeline):
    # Build a single-row cuDF DataFrame matching the TabFormer schema
    sample_df = cudf.DataFrame({'Amount': ['$42.75'], 'Merchant Name': ['3527213246127876953'], 'MCC': [5411], 'Year': [2025], 'Month': [1], 'Day': [15], 'Time': ['09:32'], 'Card': [0], 'Use Chip': ['Chip Transaction'], 'Zip': ['95113.0'], 'Merchant State': ['CA'], 'User': [1001]})
    processed = FinancialTokenizerPipeline.preprocess(sample_df)
    pipeline.fit(processed)
    _token_df = pipeline.transform(processed)
    print('Pipeline token output (one column per field):')
    print(_token_df.to_pandas().iloc[0].to_dict())
    token_cols = list(_token_df.columns)
    txn_str = ' '.join((_token_df[c].to_pandas().iloc[0] for c in token_cols))
    financial_text = f'<bos> {txn_str} <eos>'
    # Run the full pipeline: preprocess → fit → transform
    # Show per-field tokens
    # Assemble into the text format used by the decoder training corpus
    print(f'\nFull token sequence:\n{financial_text}')
    sample_tokenized = True
    return financial_text, sample_tokenized


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.3 Tokenizer Comparison

    How would a general-purpose BPE tokenizer handle the same transaction? We compare our Financial Tokenizer against GPT-2 applied to a raw serialization of the tabular fields. The token count directly determines how many transactions fit into the model's fixed context window.
    """)
    return


@app.cell
def _(financial_text, gpt2_tokenizer):
    fin_tokens = financial_text.split()

    raw_tabular_text = (
        "$42.75, 3527213246127876953, 5411, "
        "2025-01-15, 09:32, 0, Chip Transaction, "
        "95113, CA, 1001"
    )

    print("=" * 70)
    print("TOKENIZER COMPARISON: Same Transaction, Two Approaches")
    print("=" * 70)

    print(f"\n--- Financial Tokenizer (domain-specific pipeline) ---")
    print(f"Input : raw tabular fields → binning + hashing + encoding")
    print(f"Count : {len(fin_tokens)} tokens")
    print(f"Tokens: {fin_tokens}\n")

    gpt2_tokens = gpt2_tokenizer.tokenize(raw_tabular_text)
    print(f"--- GPT-2 BPE Tokenizer (on raw tabular text) ---")
    print(f"Input : {raw_tabular_text}")
    print(f"Count : {len(gpt2_tokens)} tokens")
    print(f"Tokens: {gpt2_tokens}")
    return fin_tokens, gpt2_tokens


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.4 Quantitative Summary

    The token count difference has a dramatic impact on what the model can learn.
    """)
    return


@app.cell
def _(fin_tokens, gpt2_tokens):
    fin_count = len(fin_tokens)
    gpt2_count = len(gpt2_tokens)

    CONTEXT_WINDOW = 4096
    TOKENS_PER_TXN_SEP = 1

    fin_per_txn = fin_count - 2  # subtract <bos>/<eos> (per-sequence overhead, not per-txn)
    fin_txns_per_seq = CONTEXT_WINDOW // (fin_per_txn + TOKENS_PER_TXN_SEP)
    gpt2_txns_per_seq = CONTEXT_WINDOW // (gpt2_count + TOKENS_PER_TXN_SEP)

    print("=" * 70)
    print("TOKENIZER COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Tokenizer':<20} {'Tokens/Txn':>12} {'Compression':>14} {'Txns in 4096':>15}")
    print("-" * 70)
    print(f"{'Financial':<20} {fin_per_txn:>12} {'1x (baseline)':>14} {f'~{fin_txns_per_seq}':>15}")
    print(f"{'GPT-2 (BPE)':<20} {gpt2_count:>12} {f'{gpt2_count/fin_per_txn:.1f}x more':>14} {f'~{gpt2_txns_per_seq}':>15}")
    print("-" * 70)
    print(f"\nWith a 4096-token context window:")
    print(f"  Financial tokenizer: ~{fin_txns_per_seq} transactions of temporal context")
    print(f"  GPT-2 BPE tokenizer: ~{gpt2_txns_per_seq} transactions")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.5 Analysis: Why This Matters

    The Financial Tokenizer maps each field to **one semantically meaningful token**, while BPE tokenizers fragment raw values into subwords:

    | Concept | Financial Tokenizer | GPT-2 BPE on raw text | Problem with BPE |
    |---------|--------------------|-----------------------|------------------|
    | Amount | `AMT_1` (1 token) | `$42`, `.`, `75` (3+ tokens) | Magnitude is lost -- `$42` and `$4,200` produce similar fragments |
    | Merchant ID | `MERCH_667` (1 token) | `352`, `72`, `13`, ... (10+ tokens) | A 19-digit number explodes into meaningless digit subwords |
    | Time of day | `HOUR_09` (1 token) | `09`, `:`, `32` (3 tokens) | Hour and minute are separated; temporal grouping must be learned |
    | Location | `ZIP3_951` + `STATE_CA` (2 tokens) | `95`, `113`, `,`, `CA` (4+ tokens) | Full ZIP is exposed; no privacy-preserving truncation |

    With a 4096-token context window, standard BPE tokenizers fit \~80-130 transactions per sequence. Our financial tokenizer fits **\~315 transactions** (12 tokens each), giving the model access to months of spending history. The tokenization also provides a **privacy layer**: merchant names are hashed, amounts are binned, and ZIP codes are truncated to 3 digits.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.6 Edge Cases
    """)
    return


@app.cell
def _(FinancialTokenizerPipeline, cudf, pipeline, sample_tokenized):
    _ = sample_tokenized

    def quick_tokenize(overrides: dict) -> str:
        """Helper: tokenize a single transaction with field overrides."""
        row = {
            "Amount": ["$42.75"], "Merchant Name": ["3527213246127876953"],
            "MCC": [5411], "Year": [2025], "Month": [1], "Day": [15],
            "Time": ["09:32"], "Card": [0], "Use Chip": ["Chip Transaction"],
            "Zip": ["95113.0"], "Merchant State": ["CA"], "User": [1001],
        }
        row.update({k: [v] for k, v in overrides.items()})
        gdf = cudf.DataFrame(row)
        gdf = FinancialTokenizerPipeline.preprocess(gdf)
        tdf = pipeline.transform(gdf)
        return " ".join(tdf[c].to_pandas().iloc[0] for c in tdf.columns)

    print("Large amount ($8,500):")
    tokens = quick_tokenize({"Amount": "$8500.00"})
    print(f"  {tokens.split()[0]}  (AMT_6 = $5,000+)")

    print("\nOnline transaction (no physical location):")
    tokens = quick_tokenize({"Use Chip": "Online Transaction", "Zip": "00000", "Merchant State": ""})
    print(f"  Tokens: {tokens}")

    print("\nSmall amount ($0.50):")
    tokens = quick_tokenize({"Amount": "$0.50"})
    print(f"  {tokens.split()[0]}  (AMT_0 = $0-10)")

    print("\nThe tokenizer maps all amounts to 7 log-scale bins, preserving magnitude")
    print("while avoiding the sparsity of exact dollar amounts.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Generate Tokenized Corpora (Train / Val / Test)

    Convert each temporal split into text sequences for causal language modeling. Transactions are grouped by (user, card), chunked into sequences of \~315 transactions, and formatted as:

    ```
    <bos> AMT_x MERCH_x ... CUST_x <sep> AMT_x MERCH_x ... CUST_x <sep> ... <eos>
    ```

    Each sequence is \~4096 tokens, matching the foundation model training context window.
    """)
    return


@app.cell
def _(
    CORPUS_DIR,
    CORPUS_PATH,
    FinancialTokenizerPipeline,
    TEST_CORPUS_PATH,
    TEST_DATA,
    TRAIN_DATA,
    VAL_CORPUS_PATH,
    VAL_DATA,
    cudf,
):
    import time
    MERCHANT_HASH_SIZE = 2000
    CHUNK_SIZE = 315
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    splits = [('train', TRAIN_DATA, CORPUS_PATH), ('val', VAL_DATA, VAL_CORPUS_PATH), ('test', TEST_DATA, TEST_CORPUS_PATH)]
    for split_name, parquet_path, corpus_path in splits:
        if corpus_path.exists():
            _n_lines = sum((1 for _ in open(corpus_path)))
            print(f'[{split_name}] Corpus already exists: {_n_lines:,} sequences')
            continue
        print(f"\n{'=' * 60}")
        print(f'Generating decoder corpus for {split_name}')
        print(f"{'=' * 60}")
        t0 = time.time()
        gdf = cudf.read_parquet(str(parquet_path))
        print(f'  Loaded {len(gdf):,} rows in {time.time() - t0:.1f}s')
        pip = FinancialTokenizerPipeline(merchant_hash_size=MERCHANT_HASH_SIZE)
        gdf_proc = pip.preprocess(gdf)
        pip.fit(gdf_proc)
        _token_df = pip.transform(gdf_proc)
        group_cols = []
        for col_name in ['user', 'User', 'cust']:
            if col_name in gdf_proc.columns:
                group_cols.append(col_name)
                break
        for col_name in ['card', 'Card', 'card_id']:
            if col_name in gdf_proc.columns:
                group_cols.append(col_name)
                break
        if not group_cols:
            group_cols = [gdf_proc.columns[0]]
        print(f'  Grouping by: {group_cols}')
        print(f'  Chunk size: {CHUNK_SIZE} transactions (~{CHUNK_SIZE * 13} tokens)')
        corpus_lines = pip.to_corpus_lines(_token_df, gdf_proc, group_cols, chunk_size=CHUNK_SIZE)
        with open(corpus_path, 'w') as _f:
            for _line in corpus_lines:
                _f.write(_line + '\n')
        elapsed = time.time() - t0
        print(f'  Generated {len(corpus_lines):,} sequences in {elapsed:.1f}s')
        print(f'  Saved to: {corpus_path}')
        sample = corpus_lines[0]
        n_tokens = len(sample.split())
        n_seps = sample.count('<sep>')
        print(f'  Sample: {n_tokens} tokens, {n_seps + 1} transactions')
        print(f'  Preview: {sample[:200]}...')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Verify Output
    """)
    return


@app.cell
def _(CORPUS_PATH, TEST_CORPUS_PATH, VAL_CORPUS_PATH):
    import os
    print('Corpus Summary:')
    print('=' * 60)
    for name, path in [('Train', CORPUS_PATH), ('Val', VAL_CORPUS_PATH), ('Test', TEST_CORPUS_PATH)]:
        _n_lines = sum((1 for _ in open(path)))
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f'  {name:6s}: {_n_lines:>8,} sequences  ({size_mb:>7.1f} MB)  {path.name}')
    print(f'\nFirst 3 lines of {CORPUS_PATH.name}:')
    with open(CORPUS_PATH, 'r') as _f:
        for _ in range(3):
            _line = _f.readline().strip()
            print(f'  {_line[:120]}...')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Summary

    The custom Financial Tokenizer produces **3-4x fewer tokens** per transaction than GPT-2 BPE (12 tokens vs 30-50+), enabling \~315 transactions per 4096-token sequence. Each token captures a complete financial concept rather than arbitrary subwords.

    **Outputs:**
    - `data/decoder_corpus/train_corpus.txt` -- training corpus (\~64K sequences)
    - `data/decoder_corpus/val_corpus.txt` -- validation corpus (\~10K sequences)
    - `data/decoder_corpus/test_corpus.txt` -- test corpus (\~11K sequences)

    Continue to [03_foundation_model_training.py](./03_foundation_model_training.py).
    """)
    return


if __name__ == "__main__":
    app.run()
