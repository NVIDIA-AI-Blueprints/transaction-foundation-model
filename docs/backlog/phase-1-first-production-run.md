# Phase 1 — Minimal First Production Run (Crypto Next-Trade)

**Backlog: [D5](../05-research/02-improvement-ideas.md#d5--next-trade-prediction-on-zkai-internal-trade-streams--customer-driven) (+ [T1](../05-research/02-improvement-ideas.md#t1--turn-on-and-then-improve-time-encoding), [T2](../05-research/02-improvement-ideas.md#t2--drop-the-cust-token-and-ablate-card-the-deployability-ablation), [E1](../05-research/02-improvement-ideas.md#e1--build-the-multi-task-behavioral-benchmark)) · Contracts: [C1](../03-learning-path/level-400-design-contracts-and-extensions.md#1-the-contracts) / C3 / C6 · Effort: M · GPU**

> **Why this phase is the first thing to build.** "Next-trade prediction *is* [causal language
> modeling](../02-concepts/03-causal-language-modeling.md) on this data — the customer deliverable
> falls out of the loss function" ([doc 09 §4](../04-data/09-zkai-internal-datasets.md#4-from-catalog-to-next-trade-prediction)).
> Existing customers asked for exactly this, and it requires no new objective. It is the
> [roadmap](../05-research/03-dataset-and-training-roadmap.md)'s Lineage A first deliverable and ships
> a real Loom-sprint product **regardless of whether any later phase succeeds.**

**Prerequisite:** [Phase 0](phase-0-guardrails.md) gate fully green. The committed `vocab_size` /
`chunk_size`, the leak-free split, and the baseline table are inputs here.

Run the whole phase as one Loom experiment: `--experiment tfm-d5-dex-nexttrade` (baseline runs live
under the same ID — [house rule](../06-experimentation/01-loom-workflow.md#6-house-rules-when-using-loom-on-this-repo)).

---

## Step 1.1 — Write the tokenizer pipeline (`src/tokenizer/chain_pipeline.py`)

**What:** A `TokenizerPipeline` subclass for the Phase-0 DEX field set.
**How:** Mirror [`src/tokenizer/financial_pipeline.py`](../../src/tokenizer/financial_pipeline.py) exactly — a `preprocess()` staticmethod (mart rows → clean columns; **sort by `["wallet", "timestamp"]`** for C6) plus `_configure_steps()` (the Phase-0 table). The full pattern is the worked Ethereum example in [universal recipe Step 3](../04-data/08-from-raw-data-to-training-run.md#step-3--write-the-pipeline-subclass) — copy it, swap the fields:
- log-bin `size_usd` with deterministic thresholds (no fitted artifact — [C2](../03-learning-path/level-400-design-contracts-and-extensions.md#1-the-contracts));
- hash `item` via [`CategoricalHashTokenizer`](../../src/tokenizer/categorical_hash.py) (confirmed data-free / C2-clean), with the **T4 top-K assets first-class** for the eval denominator;
- `include_time_delta=True` → [`TimeDeltaTokenizer`](../../src/tokenizer/timedelta.py) computes the inter-trade gap inside `preprocess()` after the sort (**T1**);
- **no `cust`/identity step (T2).**

**Done when:** `ChainTokenizerPipeline.preprocess()` runs on a mart sample and emits the Phase-0 columns; no `nan`/`None` tokens.

## Step 1.2 — Write the LM tokenizer + dataset builder

**What:** `src/tokenizer/chain_tokenizer.py` and `src/chain_clm_data.py`.
**How:** Near-copies ([universal recipe Step 4](../04-data/08-from-raw-data-to-training-run.md#step-4--the-lm-facing-tokenizer--dataset-builder)):
- `chain_tokenizer.py` ← [`financial_tokenizer.py`](../../src/tokenizer/financial_tokenizer.py); swap in `ChainTokenizerPipeline`. Assert `ChainTabularTokenizer().vocab_size == ` the Phase-0 hand-count.
- `chain_clm_data.py` ← [`clm_data.py`](../../src/clm_data.py); rename entry point `build_chain_clm_dataset`. The `{input_ids, labels}` contract ([C4](../03-learning-path/level-400-design-contracts-and-extensions.md#1-the-contracts)) is unchanged.

> **Flag for [I1](../05-research/02-improvement-ideas.md#i1--streaming-corpus-loading):** `clm_data.py` loads the whole corpus into RAM. Fine for Phase 1; **must be replaced with streaming shards before Phase 2 scale-up.** Leave a `# TODO(I1)` so it's a schedule item, not a surprise.

**Done when:** the vocab-size assertion passes; a tiny corpus round-trips through `encode`/`decode`.

## Step 1.3 — Build the corpus (internal mart + public augmentation)

**What:** Generate `train/val/test` corpora from the Phase-0 split.
**Why augment:** internal volume is **early** — the mart alone likely falls short of the repo's healthy ratio (~19.5M events → ~263M tokens for 29M params). Add **public [BigQuery EVM/Solana DEX exports](../04-data/04-guide-evm-bigquery.md) (D2)**; they share the `(wallet, trade)` shape, so they extend the corpus with no schema-reconciliation work.
**How:** mirror [universal recipe Step 5](../04-data/08-from-raw-data-to-training-run.md#step-5--generate-splits-and-corpus) — `group_cols=["wallet"]`, the Phase-0 `chunk_size`, write one sequence per line. **Inspect before training:** `head -c 500 train_corpus.txt` should show tokens cycling in step order, `<sep>` between trades, no literal `nan`, lines ≤ 4096 tokens.
**Done when:** three corpus files exist, eyeballed; line counts and token-budget sane; ingested as a versioned Loom data object with the Phase-0 provenance name.

## Step 1.4 — Config (`configs/pretrain_chain_decoder.yaml`)

**What:** Training config.
**How:** `cp configs/pretrain_financial_decoder.yaml configs/pretrain_chain_decoder.yaml` and change the [three things](../04-data/08-from-raw-data-to-training-run.md#step-6--config-copy-then-change-three-things):
```yaml
model: { config: { vocab_size: <Phase-0 hand-count> } }   # ① C1
dataset:            { _target_: src/chain_clm_data.py:build_chain_clm_dataset, seq_length: 4096 }  # ②
validation_dataset: { _target_: src/chain_clm_data.py:build_chain_clm_dataset, seq_length: 4096 }  # ②
step_scheduler:     { max_steps: 3000 }                    # ③ a real run, not the 30-step demo
```
**Keep the 29M architecture unchanged.** Do **not** scale params on run #1 ([A2](../05-research/02-improvement-ideas.md#a2--widthdepth-scaling-sweep-with-data-scaling): never scale params into a thin token budget).
**Done when:** config validates; `vocab_size` matches the tokenizer assertion.

## Step 1.5 — Train, with sanity gates (Loom Pattern B)

**What:** Pretrain via [Loom `train --launch`](../06-experimentation/01-loom-workflow.md#pattern-b--pretraining-experiments-launch-and-track) (cost-gated, human-confirmed). Under the hood this is [`scripts/train_decoder_model.py`](../../scripts/train_decoder_model.py) with FSDP2 (`torchrun --nproc-per-node=N`).
**Sanity gates, in order** ([universal recipe Step 7](../04-data/08-from-raw-data-to-training-run.md#step-7--train-with-sanity-gates)):
1. Banner prints the hand-counted vocab + architecture — **wrong number = stop (C1).**
2. First-step loss ≈ **ln(vocab)** (≈ 8.5 for ~5k vocab). Materially higher → encoding bug; lower → degenerate/duplicate corpus.
3. Loss falls fast for ~100 steps (grammar) then grinds (behavior). Plateau at grammar level (~5–6) → fields may be noise; revisit Phase-0 field choices.
4. **Val tracks train** — divergence at this scale means split leakage, not overfitting (re-check Phase 0).
**Done when:** all four gates green; checkpoint tracked as a Loom `TrainFlow/<id>` artifact with lineage. On-demand cloud means the run scales freely — but it is `launch-and-track` (never auto-searched), human-confirmed.

## Step 1.6 — Extract embeddings

**What:** Wallet embeddings for the recommendation/segmentation deliverable.
**How:** [`src/decoder_inference.py`](../../src/decoder_inference.py) is already domain-agnostic — pass `ChainTabularTokenizer()`, `pooling="last_token"`. **Join embeddings back to features on `row_ids`, never on position** ([C6](../03-learning-path/level-400-design-contracts-and-extensions.md#1-the-contracts) — `preprocess()` re-sorted the rows). In Loom: `train --capability embed --backbone TrainFlow/<id>`.
**Done when:** a `(N, 512)` embedding array per split, row-ID-aligned, saved as an `EmbeddingsFlow/<id>` data object.

## Step 1.7 — Evaluate against the baseline panel (the real gate)

**What:** Score the model vs the Phase-0 baselines on the **wallet-disjoint, temporally-held-out** split.
**Why a panel, not one number** (the design panel's sharpest fix): a single next-item metric over a 5,000-bucket hash is gameable in both directions — repeat-last is near-unbeatable for habitual wallets, and collisions inflate/deflate Prec@1. So the gate rests on **clean metrics**:

| Metric | Baseline | Role | How |
|---|---|---|---|
| **next-side accuracy** | majority class | **primary** (hash-free) | rank the masked `SIDE_*` logit slice |
| **next-amount sMAPE / bin-acc** | last-amount | **primary** | over the `AMT_*` slice |
| next-item Prec@1 / Recall@K | popularity **and** repeat-last | supporting | over `ITEM_*`, reported **separately** for **de-hashed top-K** (T4) and the hashed tail |

Pre-register the go/no-go margins on the two **primaries**. Commit the constrained-decoding/masking protocol (rank only the relevant vocab slice) — don't just assert the metric. Report everything **sliced by activity tier** (Phase 0.4).
**How:** [Loom `validate`](../06-experimentation/01-loom-workflow.md#pattern-b--pretraining-experiments-launch-and-track) against the baseline on identical splits → `VERDICT: PASS/REVIEW/FAIL`.
**Done when:** the panel is computed and the `loom report` card is assembled under `tfm-d5-dex-nexttrade`.

## Step 1.8 — Ship the deliverable

**What:** The customer-facing artifacts: a **next-trade ranker** (top-K next `ITEM`/`SIDE` for a wallet's history) and a **wallet-embedding API** (the 512-d vector for recs/segmentation), demoed on held-out wallets.
**Done when:** both run end-to-end on unseen wallets; the checkpoint/embeddings trace to a `validate` VERDICT of **PASS** ([house rule 5](../06-experimentation/01-loom-workflow.md#6-house-rules-when-using-loom-on-this-repo)).

---

## Advance gate (Phase 1 → Phase 2)

- [ ] Tokenizer/dataset/config built as pattern-copies; `vocab_size` asserted == Phase-0 hand-count (C1)
- [ ] Corpus eyeballed; public-DEX augmentation in; line budgets ≤ 4096 (C3); ingested with provenance name
- [ ] Training sanity gates 1–4 green (banner vocab; first-step loss ≈ ln(vocab); grammar→behavior; val tracks train)
- [ ] Embeddings extracted, **row-ID-aligned** (C6)
- [ ] **Beat popularity/repeat-last on the two pre-registered primaries** (next-side, next-amount) on the **wallet-disjoint** split; supporting item metrics reported de-hashed + tail, sliced by activity tier
- [ ] `validate` VERDICT = PASS; `loom report` card linked in the [research KB](https://github.com/ZKAI-Network/research)
- [ ] Next-trade ranker + wallet-embedding API demoed on unseen wallets

When green, the curriculum may grow — proceed to Phase 2 (multi-chain, then multi-protocol as separate
eval domains; see the [roadmap](../05-research/03-dataset-and-training-roadmap.md#3-the-phased-plan)).
Remember: **no parameter scale-up until measured non-redundant token supply justifies it, and no
multi-billion-token run before [I1](../05-research/02-improvement-ideas.md#i1--streaming-corpus-loading) streaming exists.**
