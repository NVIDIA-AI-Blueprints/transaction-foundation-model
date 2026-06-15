# From Raw Data to Training Run: The Universal Recipe

This is the bridge between *any* event dataset — a [BigQuery chain export](03-bigquery-blockchain-primer.md), [MBD](02-public-datasets-catalog.md#mbd--multimodal-banking-dataset), your own internal data — and a pretrained model. It generalizes what notebooks 01–03 do for TabFormer into eight explicit steps, with a fully worked example: the [Ethereum export from the EVM guide](04-guide-evm-bigquery.md).

**Prerequisites:** [Level 300](../03-learning-path/level-300-the-pipeline-in-code.md) (you'll recognize every pattern) and [Level 400](../03-learning-path/level-400-design-contracts-and-extensions.md) (you're about to renegotiate contracts C1–C3 deliberately).

---

## Step 0 — Decide what an "entity" and an "event" are

Everything downstream follows from two choices:

- **Entity** (the sequence owner — TabFormer: `(user, card)`; chains: the wallet/account; banking: the customer). One entity = one behavioral story = one stream of sequences.
- **Event** (one row — a transaction, a transfer, an operation) with a timestamp and a handful of describable fields.

Sanity targets, from this repo's working point: entities with **tens-to-thousands of events** each; **millions-to-tens-of-millions of events** total for a first corpus (TabFormer: 19.5M events → ~64K sequences → ~263M tokens for a 29M-param model — a healthy ratio; scale both together per the [scaling guidance](../05-research/01-literature-review.md)).

## Step 1 — Map fields to the universal schema

Fill this table for your dataset (the [research KB's universal schema](../05-research/01-literature-review.md#3-tokenization--schema-findings-the-part-most-transferable-to-us)):

| Universal field | TabFormer | Ethereum export (our example) | Yours |
|---|---|---|---|
| `entity_id` | User + Card | `entity` (wallet) | |
| `counterparty_id` | Merchant Name | `counterparty` | |
| `timestamp` | Year/Month/Day/Time | `ts` | |
| `amount` | Amount ($) | `amount_native` (ETH) | |
| `category` (action kind) | MCC | `method_id` (4-byte selector) | |
| `direction` | — (always outbound) | `direction` IN/OUT | |
| domain extras | Use Chip, Zip, State | `status`, `gas_price` | |

A field earns a token if it's **(a) present on ~every event, (b) discretizable into a vocab that won't starve** (rule of thumb: average ≥ ~1K occurrences per token in your corpus), and **(c) plausibly behavior-bearing**. Drop free-text and near-unique fields (or hash them); resist the urge to tokenize everything — each extra token per event costs context capacity (Step 3).

## Step 2 — Choose a strategy per field

The [three strategies from Primer 2](../02-concepts/02-tokenization-and-vocabularies.md#three-tokenizer-strategies-and-when-each-is-right), as a decision table:

| Field shape | Strategy | Repo class | Sizing rule |
|---|---|---|---|
| bounded int (hour, month, small bins) | fixed vocab | [`FixedVocabTokenizer`](../../src/tokenizer/fixed_vocab.py) | exactly `max−min+1` tokens |
| known finite set / ranges | mapping (+ default!) | [`MappingTokenizer`](../../src/tokenizer/mapping.py) | enumerate + 1 default |
| unbounded categorical (counterparties) | hash | [`CategoricalHashTokenizer`](../../src/tokenizer/categorical_hash.py) | buckets ≈ corpus_events / 5K–50K; expect collisions |
| continuous (amounts, fees) | threshold bins (deterministic) or quantile (fitted!) | preprocess + `FixedVocabTokenizer`, or [`NumericalTokenizerOptBin`](../../src/tokenizer/numerical.py) | 7–32 bins, log-spaced for heavy tails |
| inter-event time | log bins | [`TimeDeltaTokenizer`](../../src/tokenizer/timedelta.py) | 32 bins covers seconds→months |

Two rules carried over from [Level 400](../03-learning-path/level-400-design-contracts-and-extensions.md): prefer **deterministic** strategies (C2 — no fitted artifact to version); and think hard before adding an **identity token** like `CUST_*` (sharp edge #1 — for chains, the entity ID stays *out* of the token stream; identity lives in the grouping, not the vocabulary).

## Step 3 — Write the pipeline subclass

Mirror [`financial_pipeline.py`](../../src/tokenizer/financial_pipeline.py): a `preprocess()` staticmethod (raw → clean columns) plus `_configure_steps()` (columns → tokens). The Ethereum example in full — note how *every* pattern is copied from the financial pipeline:

```python
# src/tokenizer/chain_pipeline.py
import cudf
from .pipeline import TokenizerPipeline
from .fixed_vocab import FixedVocabTokenizer
from .mapping import MappingTokenizer
from .categorical_hash import CategoricalHashTokenizer
from .timedelta import TimeDeltaTokenizer

# log-spaced native-amount thresholds (ETH): 0, 0.001, 0.01, 0.1, 1, 10, 100, 1000
AMOUNT_THRESHOLDS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
GAS_GWEI_THRESHOLDS = [1.0, 5.0, 15.0, 30.0, 60.0, 120.0]          # gas-price bins (gwei)

class ChainTokenizerPipeline(TokenizerPipeline):
    """EVM wallet-event tokenizer: 9 tokens/event (+1 with time delta).

    AMT DIR CTPY MTH ST GAS HOUR DOW MONTH [TDIF]
    """

    def __init__(self, counterparty_hash_size: int = 5000,
                 method_hash_size: int = 256,
                 include_time_delta: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.counterparty_hash_size = counterparty_hash_size
        self.method_hash_size = method_hash_size
        self.include_time_delta = include_time_delta
        self._configure_steps()

    def _configure_steps(self):
        self.add_step("amt_bin", FixedVocabTokenizer(prefix="AMT", min_val=0, max_val=7))
        self.add_step("direction", MappingTokenizer(prefix="DIR", values=["IN", "OUT"], default="OUT"))
        self.add_step("ctpy_hash", CategoricalHashTokenizer(
            vocab_limit=self.counterparty_hash_size, special_token="CTPY"))
        self.add_step("method_hash", CategoricalHashTokenizer(
            vocab_limit=self.method_hash_size, special_token="MTH"))
        self.add_step("status", FixedVocabTokenizer(prefix="ST", min_val=0, max_val=1))
        self.add_step("gas_bin", FixedVocabTokenizer(prefix="GAS", min_val=0, max_val=6))
        self.add_step("hour", FixedVocabTokenizer(prefix="HOUR", min_val=0, max_val=23, pad_width=2))
        self.add_step("dow", FixedVocabTokenizer(prefix="DOW", min_val=0, max_val=6))
        self.add_step("month", FixedVocabTokenizer(prefix="MONTH", min_val=1, max_val=12, pad_width=2))
        if self.include_time_delta:
            self.add_step("time_delta_s", TimeDeltaTokenizer(num_bins=32, special_token="TDIF"))

    @staticmethod
    def preprocess(df: cudf.DataFrame) -> cudf.DataFrame:
        """Parquet export (entity, direction, counterparty, amount_native,
        method_id, status, gas_price, ts) → pipeline-ready columns."""
        amt = df["amount_native"].fillna(0.0).astype("float64")
        df["amt_bin"] = sum(
            (amt >= t).astype("int32") for t in AMOUNT_THRESHOLDS    # 0 ⇒ dust, 7 ⇒ ≥1000 ETH
        )
        df["ctpy_hash"] = df["counterparty"].fillna("0x0").astype(str).hash_values()
        df["method_hash"] = df["method_id"].fillna("0x").astype(str).hash_values()
        df["status"] = df["status"].fillna(1).astype("int32").clip(0, 1)

        gwei = (df["gas_price"].fillna(0).astype("float64") / 1e9)
        df["gas_bin"] = sum((gwei >= t).astype("int32") for t in GAS_GWEI_THRESHOLDS)

        ts = cudf.to_datetime(df["ts"])
        df["hour"], df["dow"], df["month"] = ts.dt.hour, ts.dt.dayofweek, ts.dt.month

        df["time_full"] = ts
        df = df.sort_values(["entity", "time_full"])                 # chronological — C6!
        td = df.groupby("entity")["time_full"].diff()
        df["time_delta_s"] = td.dt.total_seconds().fillna(0).clip(0)
        return df.reset_index(drop=True)
```

> ⚠️ **Before you copy the `MONTH` step:** as shipped, `FixedVocabTokenizer` with `min_val > 0` shifts its global IDs up by `min_val`, colliding its top token with the next step's first token — here `MONTH_12` would share an ID with `TDIF_0`, exactly the [`MONTH_12 ≡ CARD_0` defect in the financial pipeline](../03-learning-path/level-400-design-contracts-and-extensions.md#3-sharp-edges-read-before-deploying-or-publishing-numbers) (sharp edge #9). Since you're training from scratch anyway, use `min_val=0` everywhere and shift in `preprocess()` (`df["month"] = ts.dt.month - 1`, tokens `MONTH_00`–`MONTH_11`), or fix the engine first — mechanism in [Level 500](../03-learning-path/level-500-the-code-anatomy.md#23-tokenizerpipeline--the-engine-method-by-method). Then assert `len(tok.id_to_token) == len(tok.vocab)` as your collision sentinel.

**Vocabulary accounting** (you need this number twice below): 5 specials + 8 AMT + 2 DIR + 5000 CTPY + 256 MTH + 2 ST + 7 GAS + 24 HOUR + 7 DOW + 12 MONTH + 32 TDIF = **5,355**.

**Tokens per event:** 10 (+1 `<sep>`) → `chunk_size = 4096 // 11 ≈ 372` events per sequence.

## Step 4 — The LM-facing tokenizer + dataset builder

Two small files, both near-copies of existing ones (the deliberate pattern — see [`financial_tokenizer.py`](../../src/tokenizer/financial_tokenizer.py) and [`clm_data.py`](../../src/clm_data.py)):

1. **`src/tokenizer/chain_tokenizer.py`** — duplicate `financial_tokenizer.py`, swap the pipeline construction for `ChainTokenizerPipeline(...)` and the constructor params. Everything else (`_build_vocab_from_pipeline`, `encode/decode`, special-token IDs) is already pipeline-agnostic. Verify: `ChainTabularTokenizer().vocab_size == 5355`.
2. **`src/chain_clm_data.py`** — duplicate `clm_data.py`, import your tokenizer, rename the entry point `build_chain_clm_dataset`. The `{input_ids, labels}` contract (C4) needs zero changes.

## Step 5 — Generate splits and corpus

Mirror notebooks 01–02 (a ~40-line script):

```python
import cudf
from src.tokenizer.chain_pipeline import ChainTokenizerPipeline

df = cudf.read_parquet("data/chain/eth/*.parquet")
df = ChainTokenizerPipeline.preprocess(df)

# Temporal 80/10/10 by event time — same discipline as notebook 01 (contract C6)
q80, q90 = df["time_full"].quantile([0.8, 0.9]).to_pandas()
splits = {"train": df[df.time_full < q80],
          "val":   df[(df.time_full >= q80) & (df.time_full < q90)],
          "test":  df[df.time_full >= q90]}

pipe = ChainTokenizerPipeline()
pipe.fit(splits["train"])                      # offsets/vocab are config-determined anyway
for name, part in splits.items():
    tokens = pipe.transform(part)
    lines = pipe.to_corpus_lines(tokens, part, group_cols=["entity"], chunk_size=372)
    with open(f"data/chain_corpus/{name}_corpus.txt", "w") as f:
        f.write("\n".join(lines))
    print(name, len(lines), "sequences")
```

**Inspect before training** — `head -c 500 data/chain_corpus/train_corpus.txt` should read like:

```
<bos> AMT_2 DIR_OUT CTPY_3417 MTH_201 ST_1 GAS_2 HOUR_14 DOW_1 MONTH_02 TDIF_18 <sep> AMT_0 DIR_IN ...
```

Eyeball-checks that catch 90% of bugs: tokens cycle in step order; `<sep>` between events; no literal `nan`/`None` tokens; line token-counts ≤ 4,096.

## Step 6 — Config: copy, then change three things

```bash
cp configs/pretrain_financial_decoder.yaml configs/pretrain_chain_decoder.yaml
```

```yaml
model:
  config:
    vocab_size: 5355                                   # ① your Step-3 number — contract C1
dataset:
  _target_: src/chain_clm_data.py:build_chain_clm_dataset   # ② your builder
  data_path: null
  seq_length: 4096
validation_dataset:
  _target_: src/chain_clm_data.py:build_chain_clm_dataset   # ②
  data_path: null
  seq_length: 4096
step_scheduler:
  max_steps: 3000                                      # ③ a real run, not the 30-step demo
```

Architecture knobs can stay as-is for a first run — 29M params is a sane starting point for a first chain corpus of this size.

## Step 7 — Train, with sanity gates

```bash
torchrun --nproc-per-node=8 scripts/train_decoder_model.py \
    -c configs/pretrain_chain_decoder.yaml \
    --dataset.data_path data/chain_corpus/train_corpus.txt \
    --validation_dataset.data_path data/chain_corpus/val_corpus.txt
```

Gates, in order ([Primer 3](../02-concepts/03-causal-language-modeling.md) explains each):

1. Banner prints your vocab (5,355) and architecture — wrong number = stop, contract C1.
2. First-step loss ≈ ln(5355) ≈ **8.59**. Materially higher → encoding bug; materially lower → duplicate/degenerate corpus lines.
3. Loss falls fast for ~100 steps (grammar), then grinds (behavior). Plateau at grammar level (~5–6) → fields may be unpredictable noise; revisit Step 1 choices.
4. Val loss tracks train loss; divergence at this scale usually means train/val split leakage, not overfitting.

## Step 8 — Embeddings and evaluation

[`HuggingFaceDecoderInference`](../../src/decoder_inference.py) is already domain-agnostic — pass your tokenizer:

```python
from src.decoder_inference import HuggingFaceDecoderInference
from src.tokenizer.chain_tokenizer import ChainTabularTokenizer

infer = HuggingFaceDecoderInference(
    model_path="models/chain-decoder/checkpoints/...",   # your consolidated HF checkpoint
    tokenizer=ChainTabularTokenizer(), pooling="last_token")
emb = infer.extract_embeddings_batched(padded_ids)        # (N, 512)
```

Evaluate per the [multi-task protocol](../05-research/02-improvement-ideas.md#e1--build-the-multi-task-behavioral-benchmark): **next-counterparty prediction** needs no labels at all; phishing/sanctions [label joins](04-guide-evm-bigquery.md#8-downstream-labels-for-the-evaluation-side) give a fraud-style task; and remember **row-ID alignment** when joining embeddings back to features (contract C6 — `preprocess()` re-sorted your rows!).

---

## The checklist (print me)

- [ ] Entity & event defined; corpus size sanity (≥10⁶ events, entities with ≥ tens of events)
- [ ] Field table filled; every field earns its token (coverage, vocab occupancy, behavior-bearing)
- [ ] Strategy per field; deterministic unless consciously fitted (C2); no identity tokens without a reason (T2!)
- [ ] `tokens_per_event` counted → `chunk_size = 4096 // (tokens_per_event + 1)` (C3)
- [ ] Vocab size computed by hand **and** asserted via `tokenizer.vocab_size` (C1)
- [ ] Temporal splits before corpus generation (C6)
- [ ] Corpus eyeballed (`head`), line lengths checked
- [ ] YAML: `vocab_size`, `_target_`, `max_steps`
- [ ] First-step loss ≈ ln(vocab)
- [ ] Embeddings joined on row IDs, never position (C6)
- [ ] Run tracked as an experiment — [Loom](../06-experimentation/01-loom-workflow.md) — and findings recorded in the [research KB](https://github.com/ZKAI-Network/research)
