# Dataset & Training Roadmap

**⏱ ~20 min · Level: 300–400 · Prerequisites: [Improvement Ideas](02-improvement-ideas.md), [the universal recipe](../04-data/08-from-raw-data-to-training-run.md), [Level 400 contracts](../03-learning-path/level-400-design-contracts-and-extensions.md)**

The [literature review](01-literature-review.md) says *what the field proved*; the [improvement ideas](02-improvement-ideas.md) are *a menu of experiments*. This page is the **sequence**: how ZKAI turns that menu into a gated build that produces a shippable product on the [Loom 12-week sprint](https://zkailabs.com/loom) timeline while growing the strongest open blockchain behavioral foundation model as a durable asset.

It exists because three real questions had no single answer in the menu:

1. **Sequencing** — start with a small dataset, or concatenate *all* on-chain data at once?
2. **Synthetic banking data** — add it to the corpus, or not?
3. **Alternative data** (anonymized grocery-delivery / digital-services receipts) — helpful, or dilutive?

This roadmap decides all three, then phases the work. The detailed, executable steps for the near-term phases live in [`docs/backlog/`](../backlog/README.md).

> **How this was produced.** A three-strategy adversarial design panel (specialist-first vs. scale-first-concatenation vs. universal-multidomain), each design critiqued against this repo's own contracts and the literature, then reconciled against four founder decisions: customers are **mixed** (crypto-native *and* fiat-consumer), compute is **on-demand cloud**, the internal Embed pipelines are **live but early** (limited volume today), and the [blockchain↔fiat transfer study (D3)](02-improvement-ideas.md#d3--the-blockchainfiat-transfer-study-the-original-contribution-bet) is **deferred** until the product track has paying traction.

---

## 1. The three forks — decided

| Fork | Decision | Why (one line) |
|---|---|---|
| **Small vs. concatenate-all-onchain** | **Start small, grow in gated stages.** One domain (DEX-only slice) first, then chains, then protocols — each behind a beat-the-baseline gate. | Big-bang concatenation is an irreversible [C1 vocabulary](../03-learning-path/level-400-design-contracts-and-extensions.md#1-the-contracts) commitment and makes the [Step-7 sanity gates](../04-data/08-from-raw-data-to-training-run.md#step-7--train-with-sanity-gates) (first-step loss ≈ ln(vocab); grammar plateau) undiagnosable across mixed regimes. You can't debug an encoding bug across five behavioral regimes at once. |
| **Synthetic banking data** | **Not in the corpus.** Use [PersonaLedger](../04-data/02-public-datasets-catalog.md#personaledger) only as a *rule-recovery eval rig*. | Its value is that you *know* its causal rules — that makes it an instrument for testing whether the objective recovers them, not training fuel. Mixing simulated behavior into the embedding space biases representations in ways the rule-recovery probe can't detect on real distributions. |
| **Alternative data (grocery / receipts)** | **No by default.** Admit only as a capped, per-customer **LoRA adapter** ([I2](02-improvement-ideas.md#i2--lora-fine-tuning-path)) on a frozen base, value-gated — else drop. | Lowest-quality, most-distant source. It belongs on a *specific* customer's adapter, never blended into the foundation corpus. Coverage breadth must not dilute the real-data edge. |

**The meta-decision that ties them together:** ZKAI builds the strongest **blockchain** behavioral FM as its durable asset, and serves fiat fintechs via a **parallel fiat lineage + LoRA adapters** — *not* by pre-committing the foundation vocabulary to a blockchain↔fiat transfer the literature explicitly rates as unproven ([§5](01-literature-review.md#5-blockchain-native-models-the-bridge-to-our-data-program), open question #1). Transfer (D3) is a parallel research bet, never a product dependency.

---

## 2. Two product lineages, one codebase

Because the first customers are mixed, the codebase runs **two lineages off the same recipe** — same tokenizer framework, same training stack, same [eval harness (E1)](02-improvement-ideas.md#e1--build-the-multi-task-behavioral-benchmark), same [guardrails (E2)](02-improvement-ideas.md#e2--user-disjoint-split--tokenizer-signature-guardrails). They differ only in corpus and vocabulary, and **each keeps its own checkpoints** (we do *not* blend behavioral regimes into one softmax).

| | **Lineage A — Crypto** (the differentiated bet) | **Lineage B — Fiat-consumer** (the proven blueprint) |
|---|---|---|
| Serves | Crypto-native customers directly | Fiat fintechs (neobank / card / BNPL) |
| Corpus | Internal DEX mart + public [BigQuery DEX exports (D2)](../04-data/03-bigquery-blockchain-primer.md) | Customer's own anonymized txns; later a shared [MBD (D1)](../04-data/02-public-datasets-catalog.md#mbd--multimodal-banking-dataset) base |
| Deliverable | Next-trade ranker + wallet-embedding API (CLM *is* the task) | Embeddings → the customer's task (fraud / churn / segmentation), à la notebooks 04–05 |
| Provenance | The repo's endorsed [first production run (D5)](02-improvement-ideas.md#d5--next-trade-prediction-on-zkai-internal-trade-streams--customer-driven) | The repo already demonstrates this on TabFormer (card data) |
| Transfer dependency | None | **None** — trains on in-domain data; does not rely on D3 |

D3 is later the *bridge between the two lineages*. Because they're built as matched-recipe siblings, the matched-compute paired experiment D3 needs comes nearly for free.

---

## 3. The phased plan

Each phase has an **advance gate** — you do not proceed until it is met. Near-term phases (0, 1, 1F) have step-by-step specs in [`docs/backlog/`](../backlog/README.md).

| Phase | Goal | Backlog IDs | Contracts | Effort | Detailed spec |
|---|---|---|---|---|---|
| **0** | Guardrails before any GPU spend | E2 | C1/C2/C3/C6 | S | [phase-0-guardrails](../backlog/phase-0-guardrails.md) |
| **1** | Minimal first production run (crypto next-trade) | D5, T1, T2, E1 | C1/C3/C6 | M | [phase-1-first-production-run](../backlog/phase-1-first-production-run.md) |
| **1F** | Fiat first run (parallel, for fiat customers) | (blueprint) | C1/C6 | M | *(backlog, after Phase 1)* |
| **2** | Multi-chain, then multi-protocol as transfer *domains* | E1, D2, A2 | C1/C6 | M | *(backlog)* |
| **3** | Infra unlock (I1) + the consumer bridge (I2) | I1, I2 | — | M | *(backlog)* |
| **4** | Research bet — D3 transfer + scale (**deferred, parallel**) | D3, D1, G1 | C1/C6 | L | *(deferred)* |

### Phase 0 — Guardrails before any GPU spend
The highest value-per-hour item in the entire backlog. Every panel stance's worst failure was a silent contract break or a leaky split producing a confident-but-wrong green light. Persist + assert the tokenizer identity (config + vocab hash, golden tests); **hand-count the vocabulary integer and commit it**; build the **wallet-disjoint ([T2](02-improvement-ideas.md#t2--drop-the-cust-token-and-ablate-card-the-deployability-ablation)) composed with temporal ([C6](../03-learning-path/level-400-design-contracts-and-extensions.md#1-the-contracts))** split correctly (they do *not* compose for free); compute the popularity / repeat-last baseline *before* any GPU spend; record the data snapshot range as the [C2](../03-learning-path/level-400-design-contracts-and-extensions.md#1-the-contracts)-equivalent provenance anchor.

### Phase 1 — Minimal first production run (the first thing to build)
The doc-endorsed [D5](02-improvement-ideas.md#d5--next-trade-prediction-on-zkai-internal-trade-streams--customer-driven) run. Train a 29M decoder on the **DEX-only slice** of the [`cross_chain_interactions` mart](../04-data/09-zkai-internal-datasets.md#-the-cross-chain-mart-mbd-dataform--start-here), **augmented with public BigQuery DEX exports** (internal volume is early), with **time-delta tokens on (T1)** and **no wallet-identity token (T2)**. Evaluate with a **panel of clean metrics** — next-side accuracy and next-amount sMAPE as the pre-registered primaries, de-hashed top-K next-item Prec@K (pull [T4](02-improvement-ideas.md#t4--tiered-merchant-vocabulary) forward for the eval) as supporting — against popularity / repeat-last on the wallet-disjoint split. **Do not filter to high-activity wallets** (≥50 trades skews to bots/MEV/whales — no fintech consumer looks like that); include ≥ tens-of-trades wallets and report by activity tier.

### Phase 1F — Fiat first run (parallel)
For a fiat customer in a sprint: run the *same recipe* on their own anonymized transactions (or TabFormer / MBD-mini as a stand-in) — tokenizer adapted to their schema → pretrain a small FM → embeddings → their downstream task. This is notebooks 01–05 generalized; **no transfer bet required.** A shared fiat base on full MBD (D1) comes later, gated on I1.

### Phase 2 — Multi-chain, then multi-protocol as transfer *domains*
**2a** add base + solana DEX (same schema family ⇒ near-zero reconciliation — a corpus-size bump within one regime). **2b** bring Hyperliquid / Polymarket / Kalshi in as **separate E1 eval domains with their own corpora/checkpoints** — *not* blended into one vocab (`SIDE_BUY/SELL` is meaningless for a perp funding event or a market resolution; forcing it is a C1 silent-corruption path). Run the in-house DEX→perps→prediction-markets transfer matrix as a cheap, zero-new-data D3 warm-up. **Split on the global wallet** (the same human trades across venues; the mart is keyed `(wallet, item, protocol)`). Scale params (29M→60M) **only when measured non-redundant token supply justifies it** by the [D/N math](01-literature-review.md#2-where-this-repo-sits-in-the-design-space) — never on calendar.

### Phase 3 — Infra unlock + the consumer bridge
**[I1](02-improvement-ideas.md#i1--streaming-corpus-loading) streaming loader is a hard gate, not a follow-up** — the in-RAM list in [`src/clm_data.py`](../../src/clm_data.py) walls at ~100× the current corpus and will OOM the scaled/MBD runs. **[I2](02-improvement-ideas.md#i2--lora-fine-tuning-path) LoRA** is how Loom serves a fiat-first customer who brings their own data, and the only home for alt-data (capped, customer-specific, value-gated). *Honesty guardrail:* LoRA adapts *within* distribution — sell it as "we adapt a strong behavioral base to your data," not as proven blockchain→fiat transfer.

### Phase 4 — Research bet (deferred, parallel, never gates revenue)
Resourced only after the product track has paying traction. [D3](02-improvement-ideas.md#d3--the-blockchainfiat-transfer-study-the-original-contribution-bet) run as **matched-compute paired models** (chain-pretrained-then-fiat-adapted vs. [MBD](../04-data/02-public-datasets-catalog.md#mbd--multimodal-banking-dataset)-from-scratch at matched params/tokens/context) on a domain-disjoint fiat task — be willing to publish a negative result. [G1](02-improvement-ideas.md#g1--counterpartygraph-features-late-fusion) graph late-fusion held until an AML/relational customer task appears. Add SR 11-7 / disparate-impact auditing ([E3](02-improvement-ideas.md#e3--calibration--ops-curve-reporting)) before any credit-adjacent fiat engagement.

---

## 4. Biggest risks & how the roadmap mitigates them

| Risk | Severity | Mitigation |
|---|---|---|
| Promotion gate measured on a collision-corrupted hashed-item field — stalls the curriculum or gives false confidence | **Highest** | Gate is a **panel**: clean next-side + next-amount primary; de-hashed top-K item (T4 pulled forward) supporting. |
| loom's 12-week consumer promise mis-sold on the months-long, may-fail D3 result | **High** | **Two lineages + D3 deferred.** Crypto ships Phase 1; fiat ships Phase 1F on in-domain data. Neither needs transfer. |
| Internal data early/thin → 29M model under-fed, embeddings weak | **Medium** | **Augment with public BigQuery DEX (D2)** to clear the repo's ~263M-token reference; keep params at 29M until tokens justify more. |
| Silent corpus corruption (wrong chunk_size; cross-venue schema mash; split leakage) | **High** | Phase 0 hand-counts vocab + commits the integer; venues stay **separate checkpoints**; **global-wallet + temporal** split with a golden test; provenance in the artifact name. |
| Scaling params into a thin/low-entropy token budget | **Medium** | Param scale-up **gated on measured non-redundant tokens** (D/N math), never on calendar. DEX swaps are low-entropy — count effective tokens, don't assume. |
| OOM on scale-up (in-RAM loader) | **Medium (schedule)** | **I1 streaming is a hard gate** before MBD / concatenated runs. |

---

## 5. How it threads through Loom

Every phase runs as a tracked [Loom](../06-experimentation/01-loom-workflow.md) experiment, one hypothesis per `--experiment` ID:

- **Phase 0** is mostly [Pattern C](../06-experimentation/01-loom-workflow.md#pattern-c--new-data-pipelines-chains-mbd) `ingest` + `eda` (the EDA leakage gate catches split leakage and identity columns) plus golden tests in CI — no GPU.
- **Phase 1 / 1F / 2** pretrains are [Pattern B](../06-experimentation/01-loom-workflow.md#pattern-b--pretraining-experiments-launch-and-track) `train --launch` (cost-gated, human-confirmed), then `train --capability embed`, then `validate` against the baseline on identical splits. Suggested IDs: `tfm-d5-dex-nexttrade`, `tfm-1f-fiat-base`, `tfm-d2-multichain`.
- **PASS before promote:** any embedding/checkpoint used in a customer-facing sprint demo must trace to a `validate` VERDICT of PASS, and the `loom report` card goes back to the [research KB](https://github.com/ZKAI-Network/research) — including negative results.

---

## 6. What to build first (one paragraph)

Do **[Phase 0](../backlog/phase-0-guardrails.md)** (E2 guardrails + hand-counted vocab + global-wallet/temporal split + popularity baseline), then train a **29M CLM on the DEX-only slice of `cross_chain_interactions`** (augmented with public BigQuery DEX) with **T1 time-delta on, no identity token (T2)**, evaluated by **next-side accuracy + next-amount sMAPE + de-hashed top-K next-item Prec@1** on a wallet-disjoint, temporally-held-out split against popularity/repeat-last. That is [Phase 1](../backlog/phase-1-first-production-run.md) — the minimal first production run (D5). It ships a real Loom-sprint product, and every later phase is gated on beating those clean baselines.
