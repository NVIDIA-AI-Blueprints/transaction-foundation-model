# Improvement Ideas: A Ranked Experiment Backlog

Every idea below is (a) motivated by the [literature review](01-literature-review.md), (b) mapped to the exact files it touches in this repo, and (c) sized. **Effort**: S = hours, M = days (usually includes a retrain), L = weeks. **Contracts** refers to [Level 400](../03-learning-path/level-400-design-contracts-and-extensions.md) — any idea touching C1 (vocabulary) implies *regenerate corpus → retrain → re-extract*.

Run anything here as a tracked experiment via [Loom](../06-experimentation/01-loom-workflow.md); record verdicts back to the [research KB](https://github.com/ZKAI-Network/research).

**Suggested first wave** (high value-per-effort, in order): E2 → T1 → T2 → E1 → O1 → A1 → D1 → G1.

---

## T — Tokenization & input representation

### T1 — Turn on (and then improve) time encoding
**What:** Enable the existing-but-dormant inter-transaction time-delta token (`include_time_delta=True`, 32 log-bins → `TDIF_*`), and as a follow-up, evaluate relative-time encodings beyond bins.
**Why:** Relative inter-event time + periodic calendar features is the cross-paper standard ([arXiv:2404.02047](https://arxiv.org/abs/2404.02047); used by TREASURE, PRAGMA, MBD). Burst-vs-lull dynamics are core fraud signal that our current calendar-only encoding (HOUR/DOW/MONTH) can't express — two transactions 30 seconds apart vs 3 days apart look identical today.
**How:** [`financial_pipeline.py`](../../src/tokenizer/financial_pipeline.py) already computes `time_delta_s` in `preprocess()`; [`timedelta.py`](../../src/tokenizer/timedelta.py) exists. Follow [Level 400 R2](../03-learning-path/level-400-design-contracts-and-extensions.md#4-extension-recipes): 13 tokens/txn → `chunk_size≈292`, vocab +32 → `vocab_size`, retrain.
**Effort:** M. **Contracts:** C1, C3.

### T2 — Drop the CUST token (and ablate CARD): the deployability ablation
**What:** Retokenize without the customer-identity token; compare fraud AP and embedding quality; add a **user-disjoint** evaluation split.
**Why:** Identity tokens risk memorization, cold-start failure on unseen users, and optimistic offline metrics ([Level 400 sharp edge #1](../03-learning-path/level-400-design-contracts-and-extensions.md#3-sharp-edges-read-before-deploying-or-publishing-numbers)). Production systems represent users via *history*, not via an ID embedding. This is the single most important credibility experiment before any customer-facing claim.
**How:** Remove the `cust` step in `_configure_steps()`; vocab −3,000; retrain; in notebook 01, add a split holding out entire users.
**Effort:** M. **Contracts:** C1, C6.

### T3 — Log-normalized amounts / data-driven bins
**What:** Replace 7 fixed dollar thresholds with (a) more, log-spaced bins, (b) the existing `amount_strategy="quantile"` (cuML) path, or (c) the literature-standard `sign(a)·log(1+|a|)` continuous embedding (requires a small model change: a numeric side-channel per token).
**Why:** `sign·log1p` is the near-universal convention ([arXiv:2404.02047](https://arxiv.org/abs/2404.02047)); 7 bins is coarse (a $150 and a $450 purchase share `AMT_3`).
**How:** (a)/(b) are config + [`numerical.py`](../../src/tokenizer/numerical.py) (note (b) introduces a fitted artifact — breaks C2's zero-artifact property; ship the binner state via `get_state()`); (c) touches model code — bigger lift.
**Effort:** S–M (a/b), L (c). **Contracts:** C1, C2.

### T4 — Tiered merchant vocabulary
**What:** Replace the flat 2,000-bucket hash with: top-K merchants by frequency as first-class tokens (Tier 1), multi-hash embeddings for the tail (Tier 2), default/behavioral fallback for unseen (Tier 3). Periodic vocabulary refresh (the EWE-1 pattern).
**Why:** The KB's recommended high-cardinality strategy; ~50 merchants/bucket collisions destroy merchant-level signal. Probabilistic hash embeddings ([arXiv:2511.20893](https://arxiv.org/abs/2511.20893)) and MPZCH ([arXiv:2602.17050](https://arxiv.org/abs/2602.17050)) are the reference upgrades.
**How:** New tokenizer step (frequency table = fitted state — C2 burden as in T3-b); for multi-hash, model-side embedding changes.
**Effort:** M (Tier 1 only) – L (full). **Contracts:** C1, C2.

### T5 — Field-order & field-dropout ablations
**What:** (a) Permute the 12-field order (e.g., move `CUST`/`CARD` first so behavioral tokens are conditioned on identity, or coarse→fine); (b) random field dropout during corpus generation for robustness to missing fields.
**Why:** Within-transaction order determines what each next-token prediction *means* ([Primer 3](../02-concepts/03-causal-language-modeling.md)); nobody has published a principled ordering. Cheap, publishable, and informative about what the model actually uses.
**How:** Reorder `add_step()` calls (offsets shift — full retrain per variant); dropout in `to_corpus_lines`.
**Effort:** M per variant. **Contracts:** C1, C3.

## O — Objectives

### O1 — Hybrid objective: add masked / event-level prediction
**What:** Mix CLM with (a) span/field masking within transactions and/or (b) *event-level* objectives (predict the whole next transaction's fields jointly, not token-by-token).
**Why:** Hybrid SSL outperforms single objectives (contrastive ≈ global structure, generative ≈ local patterns — [arXiv:2408.09995](https://arxiv.org/abs/2408.09995); theory in [ICML 2024](https://proceedings.mlr.press/v235/zhang24m)); PRAGMA's three-tier masking (15% token / 10% event / 10% key) is the production reference `[checked]`, with masking *rates* an open ablation.
**How:** Custom loss/collator beyond `MaskedCrossEntropy` — extend [`clm_data.py`](../../src/clm_data.py) to emit masked variants; NeMo AutoModel accepts custom `loss_fn` targets.
**Effort:** L.

### O2 — Contrastive auxiliary (CoLES-style)
**What:** Add a sequence-level contrastive term: sub-slices of the *same* card's history are positives, other cards negatives ([CoLES, arXiv:2002.08232](https://arxiv.org/abs/2002.08232)).
**Why:** Directly optimizes the thing we actually use (sequence embeddings) rather than relying on next-token pressure to produce them; pairs naturally with T2 (identity comes from history, not a CUST token).
**Effort:** L.

### O3 — JEPA-style latent prediction (exploratory)
**What:** Predict *latent* representations of future transaction windows instead of exact tokens (T-JEPA [arXiv:2410.05016](https://arxiv.org/abs/2410.05016), Graph-JEPA [arXiv:2309.16014](https://arxiv.org/abs/2309.16014)).
**Why:** The KB's open question #1-adjacent bet: latent objectives may capture relational/structural signal token-level prediction misses. High risk, high novelty.
**Effort:** L (research project).

## A — Architecture & scale

### A1 — Context-length study: 315 → 630 → 1,260 transactions
**What:** Train at `seq_length` 8,192 (the RoPE ceiling; `chunk_size≈630`) and, with config changes, beyond.
**Why:** TREASURE validates 512-txn contexts; nuFormer 2,048; the scaling-laws paper flags context length as *the* open axis ([arXiv:2606.05257](https://arxiv.org/abs/2606.05257)). We're at 315 — likely below the value frontier. ([Level 400 R5](../03-learning-path/level-400-design-contracts-and-extensions.md#4-extension-recipes).)
**Effort:** M (8K) – L (beyond). **Contracts:** C3.

### A2 — Width/depth scaling sweep with data scaling
**What:** 29M → 60M → 120M params *together with* more data (D1/D2) — guided by behavioral-FM scaling laws (D/N from ~340 down to ~36 as compute grows `[checked]`).
**Why:** Don't scale parameters into a fixed 263M-token corpus; the literature says data-first at our compute level.
**How:** Pure config (R4); the FSDP2 stack already supports it.
**Effort:** M per point (compute-bound).

### A3 — Architecture A/Bs the config makes nearly free
**What:** (a) `tie_word_embeddings: true` (−3.2M params — meaningful at 29M); (b) GPT-2 vs Llama recipe; (c) a Mamba/SSM baseline at matched params *if* tooling permits.
**Why:** (a)/(b) are cheap hygiene experiments; (c) addresses the KB's open question #2 (no Mamba-vs-transformer evidence at transaction scale).
**Effort:** S (a,b) / L (c).

### A4 — Field-fusion input layer (PRAGMA-style, longer-term)
**What:** Replace "12 tokens in a row" with per-field encoders fused into **one vector per transaction**, then a transformer over transactions.
**Why:** Type-specific encoding is the top-ranked tokenization strategy in the KB; sequence length drops 12× (315 txns → 315 positions), buying massive context headroom at equal compute.
**Cost:** Departs from the plain-HF-decoder contract (C4/C5) — custom modeling code. This is the "v2 architecture" candidate.
**Effort:** L.

## E — Evaluation (do these *first*; they make every other idea measurable)

### E1 — Build the multi-task behavioral benchmark
**What:** Extend notebook 05 into a harness reporting, per the KB protocol: fraud AUPRC/AUROC, **next-merchant Prec@1/Rec@K**, **amount sMAPE**, linear-probe credit-style score, clustering quality, **few-shot transfer** (10–500 labels), **temporal robustness** (test windows 3/6/12 months past training cutoff).
**Why:** Single-metric evaluation can't distinguish most ideas above; this is the KB's synthesized standard (PRAGMA/TREASURE/FDB) and our future regression suite.
**Effort:** M–L (incremental — each task is a column, start with next-merchant which needs zero new labels).

### E2 — User-disjoint split + tokenizer-signature guardrails
**What:** (a) Add a user-held-out split to notebook 01; (b) persist tokenizer config + vocab hash beside checkpoints and assert at load; (c) golden tests for token output / vocab size / corpus grammar.
**Why:** Closes the three silent-failure paths documented in Level 400 (identity leakage, C1 mismatch, contract drift). Cheapest credibility per hour available in this backlog.
**Effort:** S.

### E3 — Calibration & ops-curve reporting
**What:** Add precision@k / alert-budget curves and probability calibration (reliability diagrams) to notebook 05.
**Why:** AP summarizes ranking; fraud ops live at fixed alert budgets; credit applications need calibrated probabilities (and, eventually, fairness auditing — a literature-wide gap we should fill, per `wiki/credit-scoring.md`).
**Effort:** S.

## G — Graph & relational context

### G1 — Counterparty/graph features, late-fusion
**What:** Compute cheap graph features (degree, PageRank-ish centrality, community ID over a user↔merchant or address↔address graph; node2vec where feasible) and concatenate with embeddings for XGBoost.
**Why:** The RiskSEA result — behavioral F1 0.718 → 0.851 with graph fusion `[unverified]` — and PRAGMA's −47% AML failure both say sequence models need relational context. Late fusion is the integration pattern with the best evidence-to-effort ratio.
**How:** No model changes — a feature-engineering stage in notebooks 04/05. On blockchain data ([guides](../04-data/README.md)) the graph is explicit (`from_address → to_address`), making this idea *easier* on-chain than on TabFormer (whose graph is only user↔merchant).
**Effort:** M.

### G2 — Graph-context tokens (mid-term)
**What:** Quantize a transaction's k-hop neighborhood embedding into vocabulary tokens (the universal schema's `graph_context` field) so the sequence model *sees* relational context inline.
**Why:** KB open question #5 (optimal integration pattern); ATH motifs ([arXiv:2001.05233](https://arxiv.org/abs/2001.05233)) offer a transferable, hand-crafted starting alphabet for laundering-shaped topology.
**Effort:** L.

## D — Data (each feeds A2; how-tos live in the [Data section](../04-data/README.md))

### D1 — Pretrain on MBD (950M real transactions)
The scale jump (40×) most likely to move every metric; multimodal extensions optional. [Catalog entry](../04-data/02-public-datasets-catalog.md). **Effort:** M–L.

### D2 — Blockchain corpora from BigQuery (EVM / Solana / Stellar)
Real, public, counterparty-rich event streams at arbitrary scale; the substrate for G1/G2 and D3. Follow the [step-by-step guides](../04-data/03-bigquery-blockchain-primer.md). **Effort:** M (first corpus).

### D3 — The blockchain↔fiat transfer study (the original-contribution bet)
Pretrain on chain data, evaluate transfer to TabFormer fraud (and reverse). *No published study exists* `[checked — absence]` — a publishable result either way, and the thesis behind this whole docs section. Requires D2 + a [shared schema mapping](../04-data/08-from-raw-data-to-training-run.md) + E1 to measure. **Effort:** L.

### D4 — Drift study on real data
With D1/D2 in place, measure 3/6/12-month degradation (E1's temporal axis) and trial rolling-LoRA refresh ([arXiv:2505.17902](https://arxiv.org/abs/2505.17902)) vs full retrain. Addresses the literature's unsolved-drift gap. **Effort:** L.

## I — Infrastructure

### I1 — Streaming corpus loading
Replace the everything-in-RAM list in [`clm_data.py`](../../src/clm_data.py) with memory-mapped/streaming encoding (pre-tokenized `.npy`/Arrow shards). Prerequisite for D1/D2 at scale. **Effort:** M.

### I2 — LoRA fine-tuning path
Wire NeMo AutoModel's PEFT/LoRA recipes to the checkpoint for task adaptation — PRAGMA evidence: matches full training at 2–4% parameter overhead `[checked]`. Gives every downstream task a stronger ceiling than frozen embeddings (compare honestly via E1). **Effort:** M.

---

## Idea → evidence → effort, at a glance

| ID | Idea | Key evidence | Effort | Retrain? |
|----|------|--------------|--------|----------|
| E2 | disjoint split + guardrails | Level 400 sharp edges | S | no |
| T1 | time-delta token | universal schema convention | M | yes |
| T2 | drop CUST | deployability; memorization risk | M | yes |
| E1 | multi-task benchmark | PRAGMA/TREASURE protocol | M–L | no |
| O1 | hybrid masking | PRAGMA 3-tier; hybrid > single | L | yes |
| A1 | longer context | TREASURE 512 / nuFormer 2,048 | M–L | yes |
| T3 | log-amounts | sign·log1p standard | S–M | yes |
| T4 | tiered merchants | EWE-1 pattern; PHE/MPZCH | M–L | yes |
| G1 | graph late-fusion | RiskSEA +0.13 F1; PRAGMA AML −47% | M | no |
| D1 | MBD scale-up | 950M real txns, open | M–L | yes |
| D2 | chain corpora | EWE-1; open data moat | M | yes |
| D3 | chain↔fiat transfer | no published study — novel | L | yes |
| I2 | LoRA path | PRAGMA: LoRA ≥ full | M | no (adapter) |

*Maintain this table as experiments land; move findings (including negative results) to the [research KB](https://github.com/ZKAI-Network/research).*
