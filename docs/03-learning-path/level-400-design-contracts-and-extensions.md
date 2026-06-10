# Level 400: Design Contracts & Extensions

*45 minutes. Assumes [Level 300](level-300-the-pipeline-in-code.md). This is the level for people who will **change** the system — it documents the invariants that hold it together, the rationale behind contestable decisions, the known sharp edges, and safe extension recipes.*

A useful frame: this repo is **three contracts glued together** —

```
data contract  ──►  tokenizer contract  ──►  training/inference contract
(schema, splits)    (vocab, sequence format)  (tensors, checkpoint format)
```

Break a contract invisibly and the system keeps running while producing garbage. This page is the contract registry.

---

## 1. The contracts

### C1 — Tokenizer ↔ model: the vocabulary is load-bearing

The model is built with `vocab_size: 6251` and special-token IDs `pad=0, bos=1, eos=2 (sep=3, unk=4 by vocab order)`; the tokenizer generates exactly that vocabulary **from configuration**. Consequences:

- **Any change to tokenizer configuration changes the vocab** — `merchant_hash_size`, amount strategy/bins, `include_time_delta`, adding/removing steps, even reordering steps (offsets shift, so the same token gets a different ID). Each of those means: update `model.config.vocab_size`, **retrain from scratch**, and never mix the new tokenizer with the old checkpoint.
- The shipped checkpoint ⇔ `FinancialTabularTokenizer(merchant_hash_size=2000, amount_strategy="fixed", include_time_delta=False)`. Treat that tuple as the checkpoint's signature. (A worthwhile hardening task: persist the tokenizer config + vocab hash *next to* the checkpoint and assert at load time. Today nothing stops a silent mismatch — IDs index the embedding table either way.)

### C2 — Determinism: configuration is the only state

The vocab must be reconstructible *bit-identically* anywhere — training job, notebook, future serving — from config alone. This is why every step's `build_vocab()` works without data, and why the data-driven options (`amount_strategy="quantile"|"kmeans"`) are **off by default**: they would introduce a fitted artifact you'd have to version and ship. If you enable them, you take on that artifact-management burden ([`get_state()/from_state()`](../../src/tokenizer/base.py) exists for exactly this).

### C3 — Corpus format: `<bos> txn (<sep> txn)* <eos>`, 12 tokens per txn

[`clm_data.py`](../../src/clm_data.py) assumes whitespace-tokenizable lines in this grammar; `to_corpus_lines` produces them. The `chunk_size=315` is **derived**, not free: ⌊4096 / (12+1)⌋ ≈ 315. Add a 13th token per transaction (e.g. time delta) and the right chunk size becomes ⌊4096/14⌋ ≈ 292 — forget, and every sequence silently truncates mid-transaction at training time.

### C4 — Dataset ↔ NeMo: `{input_ids, labels}` with `-100` masking

NeMo's recipe + `MaskedCrossEntropy` expect HuggingFace-convention batches: equal-length `input_ids`/`labels`, ignored positions at `-100`, shift internal to the model. Any replacement dataset (new corpus, new domain) only has to honor this dict shape — that's the *whole* integration surface, resolved via the YAML's file-path `_target_`.

### C5 — Checkpoint: HuggingFace consolidated safetensors

`save_consolidated: true` is what makes notebooks 04–05 (and any future serving stack) NeMo-free. If you touch checkpointing config, preserve this — the repo's portability story depends on `AutoModelForCausalLM.from_pretrained` *just working*.

### C6 — Evaluation hygiene: temporal splits + row IDs

Two invariants make the final numbers meaningful:

- **Temporal discipline**: pretraining corpus from the train split only; XGBoost early-stops on val; test is untouched future. When you add data ([guide](../04-data/08-from-raw-data-to-training-run.md)), preserve "train on past, evaluate on future."
- **Row-ID alignment**: the tokenizer *re-sorts* rows; `*_row_ids.npy` is the only thing keeping embedding *i*, label *i*, and raw-feature row *i* the same transaction. Every new embedding consumer must join on row IDs, never on positional order.

---

## 2. Contestable design decisions (and why they were made)

| Decision | Rationale | The legitimate alternative |
|---|---|---|
| **Decoder + CLM** (vs BERT-style MLM) | every position trains; matches streaming arrival; mature tooling; natural last-token embeddings | MLM/hybrid objectives — strong literature results ([review](../05-research/01-literature-review.md)) |
| **12 flat tokens per transaction** | simple, legible, fixed cost | field-type-aware embeddings, learned field fusion (PRAGMA-style); fewer tokens/txn |
| **Fixed amount bins** (7 thresholds) | deterministic (C2), interpretable | quantile/kmeans bins (more uniform occupancy) at the cost of a fitted artifact |
| **Merchant hash = 2,000 buckets** | bounds vocab & embedding table; rare-merchant signal sharing | larger table (fewer collisions), frequency-ranked vocab, or learned merchant embeddings |
| **`CUST_*` token included** | helps the model condition on "who" without per-user memory; boosts benchmark | drop it for deployability — see sharp edge below |
| **Frozen embeddings** (vs fine-tuning) | one pretrain, many cheap downstream tasks; simple serving | task fine-tuning typically wins accuracy; costs per-task training/serving |
| **29M params** | matched to ~263M tokens and 6K vocab; trains in hours | scale up *with* data, not instead of it (Chinchilla-style intuition) |
| **`<sep>` between transactions** | explicit event boundary; trivial parsing | rely on the fixed 12-token rhythm; spend the saved tokens on history |

These rows are, deliberately, an experiment menu — several reappear as [improvement ideas](../05-research/02-improvement-ideas.md) with literature backing.

## 3. Sharp edges (read before deploying or publishing numbers)

1. **`CUST_*` is identity-flavored.** With users 0–2999 in-vocabulary, the model can learn *per-customer* signatures. On a closed-world synthetic benchmark that's fine (and partly the point: personalization); in production it raises (a) cold-start failure for unseen customers (`<unk>`), (b) memorization/privacy questions, (c) optimistic offline numbers if your eval users overlap training users (here they do — splits are temporal, not user-disjoint). Any real deployment should ablate CUST (and arguably CARD): retokenize without it, retrain, re-evaluate. A **user-disjoint evaluation split** is the cheap diagnostic.
2. **Demo checkpoint ≠ shipped checkpoint.** Notebook 03 writes a 30-step toy to `models/decoder-demo/`; notebooks 04–05 read the LFS-shipped ~3,000-step `models/decoder-foundation-model/`. Pointing 04 at the demo gives mush embeddings and a confusing afternoon.
3. **Two encode paths exist.** Corpus-line encoding for training ([`financial_tokenizer.encode`](../../src/tokenizer/financial_tokenizer.py) — string split + vocab lookup) and DataFrame encoding for batch inference ([`pipeline.encode`](../../src/tokenizer/pipeline.py) — per-*transaction* `<bos> tokens <eos>`, no `<sep>`-joined history). They produce different sequence shapes; know which one your experiment is using before comparing embeddings.
4. **Merchant collisions are by-construction.** ~100K names → 2,000 buckets means ~50 merchants/bucket. The model cannot distinguish colliding merchants, ever. If a merchant-level signal matters to your task, raise the bucket count (→ C1: retrain) or change the strategy.
5. **Everything-in-RAM dataset.** `load_corpus_and_tokenize` materializes all encoded sequences as Python lists — fine at ~64K×4096, a wall at 100× that. Streaming/memory-mapped datasets are the known fix when you scale data ([ideas](../05-research/02-improvement-ideas.md)).
6. **Balanced-train, stratified-eval protocol.** XGBoost trains at ~2.5% fraud but is *evaluated* at ~0.1%. Sound — but quote test-set numbers only, and never compare against papers using different prevalence without noting it.
7. **No tests, no CI.** The repo is a blueprint. Before research forks diverge, pin behavior with golden tests: token output for a fixed sample row, vocab size/hash stability, corpus line format, embedding shape. Cheap insurance against silent contract breaks (C1–C3 especially).
8. **GPU-only path.** cuDF imports at module load; there is no CPU fallback for the tokenizer pipeline. Plan dev workflows accordingly (small GPU instance > laptop).

## 4. Extension recipes

Each recipe lists the touch points and which contracts it stresses.

**R1 — Resize the merchant table (e.g. 2,000 → 10,000).**
`FinancialTokenizerPipeline(merchant_hash_size=10000)` everywhere it's constructed (notebook 02, `configs/*.yaml` dataset block, notebook 04) → vocab grows by 8,000 → set `model.config.vocab_size` accordingly → regenerate corpus, retrain, re-extract. *Stresses C1/C2.*

**R2 — Enable the time-delta token.**
`include_time_delta=True` (adds `TDIF_*`, 32 log-bins) → 13 tokens/txn → `chunk_size ≈ 292` in corpus generation → vocab +32 → `vocab_size` update → retrain. The hooks are already in `preprocess()` (`time_delta_s`) and [`timedelta.py`](../../src/tokenizer/timedelta.py); literature says inter-event time is one of the highest-value signals you can add. *Stresses C1/C3.*

**R3 — Add a brand-new field** (e.g. channel, currency, on-chain method ID).
Derive the column in `preprocess()` → choose a strategy (fixed/mapping/hash — [Primer 2](../02-concepts/02-tokenization-and-vocabularies.md)) → `add_step(...)` → recount tokens/txn → R2's checklist. For a new *dataset*, follow the dedicated [from-raw-data-to-training-run guide](../04-data/08-from-raw-data-to-training-run.md).

**R4 — Swap the architecture.**
Replace `model.config._target_` with any HF decoder config (MistralConfig, Qwen2Config, GPT2Config…) and matching dims; keep `vocab_size` and special IDs (C1). Nothing else changes — that's the AutoModel value proposition. Same recipe scales width/depth/context (remember: scale data with parameters).

**R5 — Longer context.**
`dataset.seq_length` 4096 → 8192 (RoPE ceiling: `max_position_embeddings`), `chunk_size` ≈ 630 → ~630 transactions of history. Costs: attention compute grows ~quadratically; batch size may need to drop. Literature (nuFormer at 2,048 *transactions*) suggests real headroom here.

**R6 — Different pooling / layers for embeddings.**
`pooling="mean"` is one constructor flag; richer variants (mid-layer hidden states, concat of last-N tokens, per-transaction pooling at `<sep>` positions) are small edits in [`_pool_embeddings`](../../src/decoder_inference.py). Cheap experiments — no retraining — and ideal first [Loom runs](../06-experimentation/01-loom-workflow.md).

**R7 — Fine-tune instead of freeze.**
Add a classification head over the pooled state and fine-tune (full or LoRA) on fraud labels — NeMo AutoModel's finetune recipes cover this; the dataset must then emit labels per sequence. Expect accuracy ↑, task-coupling ↑, serving cost ↑. Compare against the frozen baseline *on the same splits* before adopting.

---

## The Level-400 summary

> The system is held together by six contracts — vocabulary, determinism, corpus grammar, tensor interface, checkpoint format, evaluation hygiene. Most "mysterious" failures are a contract silently broken; most extensions are a contract consciously renegotiated (then: regenerate corpus, retrain, re-extract). Know the contracts and the repo is yours to reshape.

**Next stops:** [Research — what the literature suggests trying](../05-research/README.md) · [Data — feeding new datasets in](../04-data/README.md) · [Experimentation — running it all with Loom](../06-experimentation/01-loom-workflow.md).
