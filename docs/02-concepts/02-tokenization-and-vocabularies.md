# Primer 2: Tokens, Tokenizers, Vocabularies

**You know:** one-hot encoding, ordinal encoding, binning.
**You'll learn:** what a token is to a language model, why this repo builds a custom tokenizer instead of borrowing GPT's, and how a ~6,251-token vocabulary is constructed deterministically.

## What a token is

A language model never sees text, dollars, or merchant names. It sees a sequence of **integers**, each an index into a fixed list called the **vocabulary**. A *token* is one entry in that list; a *tokenizer* is the function mapping raw input → token sequence.

For text models the tokens are word fragments ("trans", "action"). But nothing says tokens must come from text. A token is just a discrete symbol — so we are free to *design* a symbol alphabet for transactions. That's the key creative move in this repo.

If you've ever binned a continuous variable and one-hot encoded it, you already understand 80% of this primer: **a domain token is a named bin.** `AMT_3` *is* "amount ∈ [$100, $500)" — the difference is that instead of becoming a sparse indicator column, it becomes a symbol in a sequence, and the model learns its meaning from context.

## One transaction → 12 tokens

The repo's tokenizer ([`src/tokenizer/financial_pipeline.py`](../../src/tokenizer/financial_pipeline.py)) turns each TabFormer row into exactly 12 tokens, one per field:

| Raw field | Example value | Token | Strategy |
|-----------|---------------|-------|----------|
| Amount | `$42.75` | `AMT_1` | 7 fixed dollar bins: 0, 10, 50, 100, 500, 1000, 5000 |
| Merchant Name | `3527213246127876953` | `MERCH_667` | hash into 2,000 buckets |
| MCC → industry | `5411` | `CAT_RETAIL` | range lookup (5000–5599 ⇒ RETAIL) |
| MCC code | `5411` | `MCC_5411` | mapping over 110 known codes, default `-1` |
| Time → hour | `09:32` | `HOUR_09` | fixed vocab 0–23 |
| Date → weekday | `2025-01-15` | `DOW_2` | fixed vocab 0–6 (Mon=0; Jan 15 2025 is a Wednesday) |
| Date → month | `2025-01-15` | `MONTH_01` | fixed vocab 1–12 |
| Card | `0` | `CARD_0` | fixed vocab 0–9 |
| Use Chip | `Chip Transaction` | `CHIP_CHIP` | mapping {SWIPE, CHIP, ONLINE}, default UNK |
| Zip | `95113` | `ZIP3_951` | first 3 digits, fixed vocab 000–999 |
| Merchant State | `CA` | `STATE_CA` | mapping over 58 states/territories + `ONLINE`/`XX` |
| User | `1001` | `CUST_1001` | fixed vocab 0–2999 |

So the row becomes the "sentence":

```
AMT_1 MERCH_667 CAT_RETAIL MCC_5411 HOUR_09 DOW_2 MONTH_01 CARD_0 CHIP_CHIP ZIP3_951 STATE_CA CUST_1001
```

Run it yourself — notebook 02 §1.2 does exactly this on a sample row.

## Three tokenizer strategies (and when each is right)

The pipeline is assembled from modular steps, each a subclass of [`BaseTokenizer`](../../src/tokenizer/base.py). Three strategies cover the 12 fields:

### 1. Fixed vocabulary — for bounded integers
[`FixedVocabTokenizer`](../../src/tokenizer/fixed_vocab.py): the vocab is fully determined by configuration — every integer in `[min_val, max_val]` gets a token, no data needed.

```python
self.add_step("hour", FixedVocabTokenizer(
    prefix="HOUR", min_val=0, max_val=23, pad_width=2,
))
```

Used for HOUR, DOW, MONTH, CARD, ZIP3, CUST, and (in the default "fixed" strategy) the 7 amount bins.

### 2. Mapping — for known categorical sets
[`MappingTokenizer`](../../src/tokenizer/mapping.py): an explicit value list, dict, or *range table* plus a default for everything else. The elegant case is industry category, derived from MCC by range:

```python
INDUSTRY_RANGES = [
    (0, 1499, "AGRICULTURAL"),
    (3000, 3299, "AIRLINES"),
    (5000, 5599, "RETAIL"),
    ...
]
self.add_step("mcc_int", MappingTokenizer(
    prefix="CAT", ranges=INDUSTRY_RANGES, default="GENERAL",
))
```

Note this gives the model a **two-level hierarchy**: `CAT_RETAIL` (coarse) *and* `MCC_5411` (fine) for the same underlying value. Rare MCCs still get a meaningful coarse token.

### 3. Hashing — for unbounded categorical sets
[`CategoricalHashTokenizer`](../../src/tokenizer/categorical_hash.py): there are ~100K distinct merchant names, most appearing rarely. A vocab entry per merchant would explode the embedding table and starve rare merchants of signal. Instead, hash every name into 2,000 buckets:

```python
self.add_step("merch_hash", CategoricalHashTokenizer(
    vocab_limit=2000, special_token="MERCH",
))
```

The trade: hash collisions (two merchants share `MERCH_667`) in exchange for a bounded vocabulary and signal-sharing. The same trick powers ad-click models ("the hashing trick") — and improving it (e.g., frequency-aware vocabularies, probabilistic hash embeddings) is an [open research idea](../05-research/02-improvement-ideas.md).

## Why not just use GPT-2's tokenizer?

Notebook 02 §1.3 runs the comparison. The same transaction serialized as text —
`"$42.75, 3527213246127876953, 5411, 2025-01-15, 09:32, 0, Chip Transaction, 95113, CA, 1001"` — costs **~40 BPE tokens** vs **12 domain tokens**, and the BPE fragments are semantically destructive:

| Concept | Domain token | GPT-2 BPE | What breaks |
|---|---|---|---|
| Amount | `AMT_1` | `$42`, `.`, `75` | $42 and $4,200 look similar; magnitude lost |
| Merchant | `MERCH_667` | `352`, `72`, `13`, … (10+) | 19-digit ID becomes digit confetti |
| Hour | `HOUR_09` | `09`, `:`, `32` | model must *learn* what a clock is |

Consequences of 12 vs ~40 tokens per transaction:

1. **Context capacity.** In a 4,096-token window: ~315 transactions of history (domain tokens) vs ~80–130 (BPE). More history = more behavioral context per prediction.
2. **Embedding-table economics.** Vocab 6,251 × hidden 512 ≈ 3.2M embedding parameters. GPT-2's 50,257-token vocab would cost 25.7M — more than the rest of this 29M-parameter model combined.
3. **A privacy layer for free.** Merchants are hashed, amounts binned, ZIPs truncated to 3 digits. The corpus never contains raw values.

## How the global vocabulary is assembled

Each step owns a small local vocab; [`TokenizerPipeline`](../../src/tokenizer/pipeline.py) concatenates them with **offsets** into one global ID space:

```
ids 0–4        <pad> <bos> <eos> <sep> <unk>     (5 special tokens)
ids 5–11       AMT_0 … AMT_6                     (7)
ids 12–2011    MERCH_0 … MERCH_1999              (2000)
ids 2012–2025  CAT_*                             (14)
…and so on through CUST_2999                     = 6,251 total
```

Two properties matter enormously in practice:

- **Deterministic.** Every vocab is derived from *configuration*, not data frequency. [`FinancialTabularTokenizer`](../../src/tokenizer/financial_tokenizer.py) rebuilds the identical vocab anywhere — training job, inference service, your laptop — with no fitted artifact to ship. (Contrast with BPE, which must be *trained* and version-pinned.)
- **Contracted.** The model config pins `vocab_size: 6251`. Change `merchant_hash_size` and the vocab size changes → the checkpoint no longer matches → you retrain. Tokenizer and model are a matched pair; see [Level 400](../03-learning-path/level-400-design-contracts-and-extensions.md).

It also exposes the standard LM-tokenizer API so the rest of the stack feels like HuggingFace:

```python
tok = FinancialTabularTokenizer(merchant_hash_size=2000)
tok.vocab_size      # 6251
tok.encode("<bos> AMT_1 MERCH_667 ... <eos>")   # → [1, 6, 679, ...]
tok.decode([1, 6, 679])                          # → "<bos> AMT_1 MERCH_667"
tok.pad_token_id, tok.bos_token_id, tok.eos_token_id   # 0, 1, 2
```

## From tokens to sequences

Single transactions aren't useful sequences. [`to_corpus_lines`](../../src/tokenizer/pipeline.py) groups transactions by `(user, card)`, sorts chronologically (the sort happens in `preprocess()`), and chunks into ~315-transaction sequences:

```
<bos> AMT_1 MERCH_667 … CUST_1001 <sep> AMT_0 MERCH_44 … CUST_1001 <sep> … <eos>
```

Why 315? Each transaction is 12 tokens + 1 `<sep>` ≈ 13; 315 × 13 ≈ 4,095 ≈ the 4,096-token training window. The format and chunk size are part of the data contract.

## Key takeaways

- Tokens are designed symbols; for tabular data you get to *choose* the alphabet.
- Three strategies — fixed vocab, mapping (incl. ranges/hierarchy), hashing — cover almost any schema; they're composable steps in a pipeline.
- Domain tokenization beats BPE for transactions on context capacity, parameter economics, semantic integrity, and privacy.
- The vocabulary is deterministic and **contractually coupled** to the model checkpoint.

**Next:** [Primer 3 — Causal language modeling](03-causal-language-modeling.md): what the model actually learns from these sequences.
