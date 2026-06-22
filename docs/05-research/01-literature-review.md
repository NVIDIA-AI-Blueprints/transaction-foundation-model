# Literature Review: Foundation Models for Transactions

*Distilled from the [ZKAI-Network/research](https://github.com/ZKAI-Network/research) knowledge base (`wiki/transaction-intelligence.md`, `wiki/fm-finance.md`, and the long-form brief `raw/transaction-intelligence/foundation-transaction-models.md`; last synthesis 2026-06). Claims carry the KB's verification tags: `[checked]` / `[inferred]` / `[unverified]`.*

This page answers three questions: **what has the field proven**, **where does this repo sit**, and **what does the literature imply we should do next** (the actionable half lives in [Improvement Ideas](02-improvement-ideas.md)).

---

## 1. The headline: this pattern works at production scale

Between 2024 and 2026, transaction foundation models (FTMs) moved from research to deployed systems at major payment companies — all using some variant of this repo's recipe (self-supervised pretraining on event sequences → reusable representations → downstream tasks):

| System | Org | Scale | Architecture | Headline result | Status |
|---|---|---|---|---|---|
| **TREASURE** ([arXiv:2511.19693](https://arxiv.org/abs/2511.19693), KDD 2026) | Visa | 6B txns, 30M cardholders | 3-layer decoder, ctx = 512 txns | **+111%** anomaly detection, +104% recommendation vs production | `[checked]` |
| **PRAGMA** ([arXiv:2604.08649](https://arxiv.org/abs/2604.08649)) | Revolut + NVIDIA | 24B events / 207B tokens, 26M users | Encoder family, 10M–1B params, masked modeling | **+130.2%** PR-AUC credit scoring; **−47.1%** F₀.₅ on AML | `[checked]` |
| **TransactionGPT** ([arXiv:2511.08939](https://arxiv.org/abs/2511.08939)) | Visa | billions | 3D-Transformer (feature × metadata × temporal) | +22% anomaly detection; 300× faster, 92% smaller than Llama2-7B | `[unverified]` |
| **nuFormer** ([arXiv:2507.23267](https://arxiv.org/abs/2507.23267)) | Nubank | ~100B events | Encoder + DCNv2, next-token, ctx up to **2,048 txns** | production gains | `[checked]` (scale) |
| **PANTHER** ([arXiv:2510.10102](https://arxiv.org/abs/2510.10102)) | WeChat Pay | production | structured tokenization | +25.6% HitRate@1 next-txn, +38.6% fraud recall | `[unverified]` |
| **Foundation Purchasing Model** ([arXiv:2401.01641](https://arxiv.org/abs/2401.01641), ICAIF 2023) | 180 banks | 5.1B txns | GRU + NPPR objective | up to +140% fraud value detection; **cross-bank transfer** | `[unverified]` |
| **ARGUS** ([arXiv:2507.15994](https://arxiv.org/abs/2507.15994), KDD 2026) | Yandex | billions | transformer, next-item + feedback | scaling laws hold to 1B params | `[checked]` (existence) |

Three structural observations from the KB worth internalizing:

1. **None of these are open-source.** "No industrial event-sequence FM is open-source. The competitive moat is data access, not model architecture" (`wiki/fm-finance.md`, `[checked]`). This makes open blueprints like this repo — and open data, especially [blockchain data](../04-data/README.md) — strategically interesting: the architecture knowledge is public; the differentiator is what you train on.
2. **Transformers won.** Every production deployment chose transformers over RNNs (TREASURE's ablations show consistent superiority `[checked]`); Mamba/SSMs remain theoretically attractive (O(n) in sequence length) but unvalidated at billion-transaction scale `[checked — absence]`.
3. **Gains are smooth, not emergent.** HSTU ([arXiv:2402.17152](https://arxiv.org/abs/2402.17152), Meta) and ARGUS observe clean power-law scaling without phase transitions — budget expectations accordingly.

## 2. Where this repo sits in the design space

| Design axis | This repo | The field |
|---|---|---|
| Objective | pure causal LM (next token) | TREASURE/TransactionGPT: autoregressive; PRAGMA: 3-tier masked (15% token / 10% event / 10% key); growing evidence **hybrids win** — contrastive captures global, generative captures local patterns ([arXiv:2408.09995](https://arxiv.org/abs/2408.09995) `[unverified]`; ICML 2024 theory [Look Ahead or Look Around](https://proceedings.mlr.press/v235/zhang24m) `[unverified]`) |
| Tokenization | 12 flat domain tokens/txn, hash merchants, fixed amount bins | "No universal tokenization exists" `[checked]`; ranked strategies: type-specific field encoders (PRAGMA/TREASURE) > text serialization > hierarchical/3D fusion > LLM alignment (LATTE, [arXiv:2508.10021](https://arxiv.org/abs/2508.10021)). Amounts: `sign(a)·log(1+|a|)` is the cross-paper norm ([arXiv:2404.02047](https://arxiv.org/abs/2404.02047) `[checked]`) |
| Context | 315 txns (4,096 tokens) | 512 txns is the production-validated default (TREASURE); nuFormer pushes 2,048; context-length scaling explicitly flagged as an open axis by the scaling-laws study ([arXiv:2606.05257](https://arxiv.org/abs/2606.05257) `[checked]`) |
| Scale | 29M params / 263M tokens | production: 10M–1B params on 10⁹–10¹¹ events; scaling-law guidance: data-heavy ratios early (D/N ≈ 340 at small compute → ≈ 36 at large) `[checked]` |
| Adaptation | frozen embeddings → XGBoost | linear probes (cheap eval), **LoRA fine-tuning matches or exceeds full training** across PRAGMA's six banking tasks at 2–4% parameter overhead `[checked]` |
| Entities | single-user sequences | the known blind spot — see §4 |

The honest read: this repo is a **clean, minimal instance of the production-validated recipe** — its simplifications (pure CLM, flat tokens, frozen embeddings, small scale) are pedagogically motivated and each one maps to a documented upgrade path in the literature.

## 3. Tokenization & schema findings (the part most transferable to us)

The KB's long-form brief proposes a **universal transaction schema** synthesized from the surveyed systems — directly relevant when we [bring new datasets](../04-data/08-from-raw-data-to-training-run.md):

```
Core fields (all domains):     entity_id, counterparty_id, timestamp,
                               amount (sign·log1p-normalized), category, direction
Extended (domain-specific):    channel, geo_hash, graph_context, metadata_text
```

Field-specific conventions with multi-paper support `[checked]`:

- **Amounts:** `sign(a) × log(1+|a|)` continuous normalization or log-scale bins (we use fixed dollar bins — coarser but deterministic).
- **Timestamps:** *relative inter-event time* + periodic calendar encodings (hour/day-of-week/month). We encode the calendar part; inter-event time exists but is off by default — the literature says it's high-value (see [idea T1](02-improvement-ideas.md#t1--turn-on-and-then-improve-time-encoding)).
- **High-cardinality entities (merchants, addresses):** tiered strategies — top-K learned embeddings for the head (80–95% of volume), multi-hash embeddings for the tail, behavioral-feature fallback for cold-start; monthly vocabulary refresh (the EWE-1 pattern). Probabilistic hash embeddings ([arXiv:2511.20893](https://arxiv.org/abs/2511.20893)) and Meta's zero-collision MPZCH ([arXiv:2602.17050](https://arxiv.org/abs/2602.17050)) are the modern upgrades to our flat 2,000-bucket hash.
- **Counterparty is a core field.** Note that TabFormer has no counterparty (merchants aren't payers); blockchain data has true `from → to` structure — one reason it's an interesting pretraining domain for us.

## 4. The known failure mode: relational blindness

The single most important negative result in the literature: **PRAGMA underperforms its production baseline on AML by −47.1% F₀.₅** despite +130% on credit scoring — because money-laundering patterns (smurfing, layering, mule networks) live in the **cross-user transaction graph**, which a single-user sequence model cannot see (§3.4.5 `[checked]`; the specific figure `[unverified]` at table level).

Supporting graph-side evidence:

- **Elliptic2** ([arXiv:2404.19109](https://arxiv.org/abs/2404.19109), KDD 2024): laundering manifests as identifiable *subgraph* patterns (peeling chains, smurfing) in Bitcoin; subgraph classification reaches ROC-AUC 0.889 `[unverified]`.
- **RiskSEA** ([arXiv:2410.02160](https://arxiv.org/abs/2410.02160), Coinbase, IEEE ICBC 2025): on Ethereum's 266M-address graph, behavioral sequence features F1 = 0.718 → **0.851 when fused with node2vec graph topology** `[unverified]`. Fusion, not replacement.
- **ATH motifs** ([arXiv:2001.05233](https://arxiv.org/abs/2001.05233), IEEE TKDE 2021): a compact, transferable formalism (topology/amount/temporal attribute vectors) for mixing-service detection.

Implication for this repo: our embeddings should be expected to *help* fraud-style anomaly tasks and *underperform* on network-crime tasks until we add counterparty/graph context — that's [idea G1/G2](02-improvement-ideas.md#g--graph--relational-context).

## 5. Blockchain-native models (the bridge to our data program)

A young but directly relevant line of work — on-chain data is public, graph-structured, and label-rich, making it the natural open benchmark substrate `[checked]`:

| Model | What it is | Why it matters to us |
|---|---|---|
| **BERT4ETH** ([arXiv:2303.18138](https://arxiv.org/abs/2303.18138), WWW 2023, open) | BERT-style MLM over Ethereum wallet activity sequences | proof that LM-style pretraining on wallet sequences detects phishing/de-anonymization; tiny (hidden=64) |
| **BlockFound / BlockScan** ([arXiv:2410.04039](https://arxiv.org/abs/2410.04039)) | modular tokenizer for multi-modal blockchain txns + MLM | closest published analog of *our tokenizer pipeline* applied to chain data |
| **EWE-1** (sistemalabs, 2026, [open-weights](https://huggingface.co/sistemalabs)) | causal transformer on **1.1B Ethereum transactions**; 35M/110M/500M params; 31 features × 64-txn lookback | the first open-weights blockchain FM: 85–90% within-wallet cosine similarity, 10–20% error reduction on phishing `[unverified]`; its monthly-vocab-refresh and feature design are directly reusable |
| **RiskSEA / Elliptic2** (above) | graph methods over chains | the graph complement |

And the KB's flagged **original-research opportunity**: *"No published blockchain-to-fiat transfer study exists"* `[checked — absence]`. Nobody has shown whether representations pretrained on public chain data transfer to card/bank fraud (or vice versa). Structural feasibility is argued (shared graph topology, temporal patterns, power-law degree/amount distributions); this repo + the [blockchain data guides](../04-data/README.md) are precisely the equipment needed to test it.

## 6. Datasets & benchmarks

Public datasets the KB endorses for this line of work (full how-to in the [Data section](../04-data/02-public-datasets-catalog.md)):

| Dataset | Scale | Nature | Notes |
|---|---|---|---|
| **TabFormer** ([arXiv:2011.01843](https://arxiv.org/abs/2011.01843)) | 24M txns | synthetic card | what this repo uses |
| **MBD** ([arXiv:2409.17587](https://arxiv.org/abs/2409.17587), KDD 2025) | **950M txns**, +geo, +dialog | real, anonymized corporate banking | largest open multimodal financial dataset; HuggingFace |
| **PersonaLedger** ([arXiv:2601.03149](https://arxiv.org/abs/2601.03149), Capital One, ICLR 2026) | 30M txns, 23K personas | LLM-generated synthetic | rule-grounded benchmarks |
| **Elliptic1/2** | 204K nodes / 49M nodes | Bitcoin graph | ground-truth illicit labels |
| **BigQuery public chain data** | petabytes | real, permissionless | EVM + Solana + Stellar + more; our [step-by-step guides](../04-data/03-bigquery-blockchain-primer.md) |

First-generation **event-sequence benchmarks** — none include graph structure `[checked]`, which the KB calls the biggest benchmark gap: HORIZON ([arXiv:2604.17259](https://arxiv.org/abs/2604.17259)), EBES ([arXiv:2410.03399](https://arxiv.org/abs/2410.03399), KDD 2025), MBD's benchmark suite, plus the PyTorch-Lifestream ecosystem (IJCAI 2025).

The KB's recommended **multi-task evaluation protocol** (synthesized from PRAGMA/TREASURE/Amazon FDB) is the standard we should converge to instead of single-metric fraud AP: fraud (AUPRC), next-merchant prediction (Prec@1/Rec@K), amount prediction (sMAPE), credit-scoring linear probe, unsupervised segmentation, few-shot transfer (10–500 labels), temporal robustness (3/6/12-month drift), AML precision @ low FPR. See [idea E1](02-improvement-ideas.md#e1--build-the-multi-task-behavioral-benchmark).

## 7. Other findings that should steer our roadmap

- **Temporal drift is unsolved.** No FTM paper evaluates beyond a few months of deployment `[checked — absence]`. Candidate mitigations from adjacent fields: meta-learned adaptation (DoubleAdapt, [arXiv:2306.09862](https://arxiv.org/abs/2306.09862)), rolling LoRA + ADWIN drift triggers ([arXiv:2505.17902](https://arxiv.org/abs/2505.17902)), TabPFN dual-memory streaming ([arXiv:2502.16840](https://arxiv.org/abs/2502.16840)). TREASURE names in-context drift adaptation a key direction `[checked]`.
- **Compute planning numbers** `[inferred]`: ~300M params on ~10B tokens ≈ 500–1,000 A100 GPU-hours (at the 10–20% MFU typical of embedding-heavy workloads); FSDP2 is the recommended default training stack at 300M–10B scale — conveniently, already this repo's stack.
- **JEPA-style latent prediction** (T-JEPA for tabular, [arXiv:2410.05016](https://arxiv.org/abs/2410.05016); Graph-JEPA [arXiv:2309.16014](https://arxiv.org/abs/2309.16014)) is the KB's candidate answer to "can latent objectives close the relational gap?" — open question, plausible thesis topic.
- **Governance & fairness are literature-wide gaps** `[checked — absence]`: no surveyed FTM paper addresses SR 11-7-style model governance or disparate-impact auditing. Anything we ship toward credit use cases must add this ourselves (`wiki/credit-scoring.md` covers the alternative-data evidence base and its proxy-discrimination risks).
- **Tabular FMs are converging on the same space** — TabPFN v2 (Nature 2025), Mitra (NeurIPS 2025): worth tracking as potential downstream heads or baselines.

## 8. Open questions (verbatim from the KB, lightly trimmed)

1. Blockchain-to-fiat transfer — what's the right shared representation space? *(original-contribution opportunity)*
2. Can Mamba/SSMs close the gap with transformers beyond 2K-transaction contexts?
3. Is there a principled way to set masking rates (PRAGMA's 15/10/10 lacks ablation)?
4. Cross-institution transfer under privacy constraints — federated/DP variants unexplored.
5. Optimal graph-integration pattern for AML: features-as-input vs attention-within-graph vs late fusion?
6. How do fine-tuned FTMs satisfy model-risk-management frameworks?
7. Fairness: what does disparate-impact auditing look like for behavioral embeddings?
8. How does the compute-optimal data/model ratio shift with context length?

**Next:** [Improvement Ideas](02-improvement-ideas.md) — this review converted into a ranked experiment backlog for this repo.
