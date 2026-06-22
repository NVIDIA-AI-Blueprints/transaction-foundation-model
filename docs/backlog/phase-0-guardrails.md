# Phase 0 — Guardrails Before Any GPU Spend

**Backlog: [E2](../05-research/02-improvement-ideas.md#e2--user-disjoint-split--tokenizer-signature-guardrails) · Contracts: [C1](../03-learning-path/level-400-design-contracts-and-extensions.md#1-the-contracts) / C2 / C3 / C6 · Effort: S (hours) · No GPU**

> **Why this phase exists.** Every strategy the design panel considered had the same worst-case
> failure: a silent contract break or a leaky split that produces a confident-but-wrong green light —
> a "win" that evaporates on the first honest external eval (a real held-out customer cohort). This
> phase is the cheapest insurance in the entire roadmap. **Nothing in Phase 1 touches a GPU until
> every gate here is green.**

The deliverable is *not* a model. It is: a frozen, asserted tokenizer identity; a hand-counted
vocabulary integer; a verified leak-free split; a baseline number to beat; and a reproducible data
snapshot. Each is a golden test that later phases assert against.

---

## Step 0.1 — Verify data access and freshness

**What:** Confirm you can read the [`cross_chain_interactions` mart](../04-data/09-zkai-internal-datasets.md#-the-cross-chain-mart-mbd-dataform--start-here) and its live schema.
**Why:** Manifests carry *no columns* — schemas are read live ([doc 09 §1](../04-data/09-zkai-internal-datasets.md#1-what-the-catalog-is)). And the mart's freshness is **manual on demand** ([doc 09 §3](../04-data/09-zkai-internal-datasets.md#-the-cross-chain-mart-mbd-dataform--start-here)): a stale Dataform run means a stale corpus.
**How:**
```bash
# catalog service reachable; live schema for the DEX product
curl $CATALOG_URL/data-products/mbd-dataform.bq.cross_chain_interactions
# orient: row counts by protocol, and the freshness check
bq query --use_legacy_sql=false '
  SELECT protocol, COUNT(*) n, MAX(event_date) max_date
  FROM `level-mark-437714-b1.mbd_recs.cross_chain_interactions`
  GROUP BY protocol ORDER BY protocol'
```
**Done when:** `describe_data_product` returns a live schema; you have row counts per protocol and a `MAX(event_date)`; if stale, you triggered a Dataform run. **Read from the datasets tier only** (permanent) — never the raw (30 d) or archive (90 d) buckets.

## Step 0.2 — Pin the entity, event, and field set (the Phase-1 schema)

**What:** Commit the exact `(entity, event)` definition and the per-field tokenizer strategy.
**Why:** [Universal recipe Step 0–2](../04-data/08-from-raw-data-to-training-run.md#step-0--decide-what-an-entity-and-an-event-are) — everything downstream follows from this, and a field added later is a full retrain ([C1](../03-learning-path/level-400-design-contracts-and-extensions.md#1-the-contracts)).
- **Entity** = `wallet_address`. **Event** = one DEX trade/interaction.
- **Field set (candidate)** — DEX-only, from [doc 09 §4](../04-data/09-zkai-internal-datasets.md#4-from-catalog-to-next-trade-prediction):

| Field | Strategy | Repo class | Notes |
|---|---|---|---|
| venue | fixed vocab | [`FixedVocabTokenizer`](../../src/tokenizer/fixed_vocab.py) | `VENUE_DEXETH/DEXBASE/DEXSOL` (Phase 1 = DEX family only) |
| side | mapping (+default) | [`MappingTokenizer`](../../src/tokenizer/mapping.py) | `SIDE_BUY / SIDE_SELL` |
| item (token/market) | hash | [`CategoricalHashTokenizer`](../../src/tokenizer/categorical_hash.py) | ~5,000 buckets; **T4 top-K upgrade point** (Step 0.3) |
| size (USD) | log-bins | preprocess + `FixedVocabTokenizer` | deterministic thresholds (no fitted artifact — [C2](../03-learning-path/level-400-design-contracts-and-extensions.md#1-the-contracts)) |
| inter-trade gap | log bins | [`TimeDeltaTokenizer`](../../src/tokenizer/timedelta.py) | **T1 — non-negotiable; burst-vs-dormancy is the signal** |
| hour | fixed vocab | `FixedVocabTokenizer` | `HOUR_00..23` |
| day-of-week | fixed vocab | `FixedVocabTokenizer` | `DOW_0..6` |
| **— no wallet-identity token —** | — | — | **T2 from day one**: identity comes from history, not an ID embedding |

**Done when:** the field table is filled, every field [earns its token](../04-data/08-from-raw-data-to-training-run.md#step-1--map-fields-to-the-universal-schema) (coverage, ≥~1K occurrences/token, behavior-bearing), and no identity token is present.

## Step 0.3 — Hand-count the vocabulary and commit the integer

**What:** Compute the exact `vocab_size` by hand and the resulting `chunk_size`. Commit both as constants.
**Why:** This is the [C1/C3](../03-learning-path/level-400-design-contracts-and-extensions.md#1-the-contracts) step two of the three panel stances skipped. Waving at "~10 tokens/event" is not allowed: `chunk_size = 4096 // (tokens_per_event + 1)`, so a miscount silently truncates mid-trade — the exact failure C3 guards.
**How (worked example for the candidate set):**
```
specials  5   (<pad> <bos> <eos> <sep> <unk>)
VENUE     3
SIDE      2   (+ no default needed if closed set; else +1)
ITEM   5000
AMT       8   (log-bins)
TDIF     32
HOUR     24
DOW       7
────────────
vocab ≈ 5081      tokens/event = 7 (venue,side,item,amt,tdif,hour,dow)
chunk_size = 4096 // (7 + 1) = 512 trades/sequence
```
The numbers above are *illustrative* — the doc-08 chain example lands at 5,355 for a comparable set. **Your committed integers come from the actual field choices**, and they are asserted in Step 0.5.
> **Pull [T4](../05-research/02-improvement-ideas.md#t4--tiered-merchant-vocabulary) forward for the eval (recommended):** promote the top-K assets by frequency to first-class `ITEM_*` tokens so the Phase-1 eval denominator is not collision-corrupted. The frequency table is a *fitted artifact* — version it via the pipeline's `get_state()` so [C2](../03-learning-path/level-400-design-contracts-and-extensions.md#1-the-contracts) determinism is preserved by configuration.
**Done when:** `vocab_size` and `chunk_size` are committed constants with the arithmetic written down.

## Step 0.4 — Build the split once, correctly (wallet-disjoint × temporal)

**What:** Construct the train/val/test split that is **both** wallet-disjoint ([T2](../05-research/02-improvement-ideas.md#t2--drop-the-cust-token-and-ablate-card-the-deployability-ablation)) **and** temporally honest ([C6](../03-learning-path/level-400-design-contracts-and-extensions.md#1-the-contracts)).
**Why:** These do **not** compose for free — a wallet's trades straddle any global time boundary. Get this wrong and the gate metric is optimistic (the worst failure: a confident green light on a leaky split).
**How (the construction that satisfies both):**
1. **Partition wallets disjointly** into train/val/test (deterministic — e.g. by a stable hash of `wallet_address`, ~80/10/10). No wallet appears in two splits → eval measures generalization to **unseen wallets**.
2. **Train corpus** = full chronologically-ordered histories of *train* wallets.
3. **Eval (val/test)** = for each held-out wallet, feed the model its history *prefix* and score the **chronologically-next** trade(s). Prediction is always forward → no within-wallet future leakage.
4. **Cohort rule:** include wallets with **≥ tens of trades** (the recipe's "tens-to-thousands of events per entity" target). **Do NOT filter to ≥50 trades** — that skews the corpus to bots/MEV/whales, a population no fintech consumer resembles. Instead **tag each wallet's activity tier** and report metrics sliced by tier.

> **Honest caveat (defer, don't ignore):** because train wallets' data can extend later in time than a test wallet's scored trade, the model can learn market-wide future facts (e.g. a token that later pumped) via *other* wallets. That is a second-order leak for next-trade ranking; the strict global-temporal holdout belongs to the [E1 temporal-robustness axis](../05-research/02-improvement-ideas.md#e1--build-the-multi-task-behavioral-benchmark) (test windows 3/6/12 months past the training cutoff), not Phase 1's primary gate.

**Done when:** a golden test asserts (a) the wallet sets are pairwise disjoint, (b) within every eval wallet the scored trades are strictly after the prefix, (c) the partition is deterministic across runs.

## Step 0.5 — E2 guardrails: persist + assert tokenizer identity

**What:** Make the tokenizer self-verifying so a config drift can't silently produce a different corpus.
**Why:** Closes the [C1 mismatch and contract-drift](../03-learning-path/level-400-design-contracts-and-extensions.md#3-sharp-edges-read-before-deploying-or-publishing-numbers) silent-failure paths.
**How:**
- Persist the tokenizer config + a vocab hash beside every checkpoint; **assert at load** (refuse to embed with a mismatched tokenizer).
- Golden tests (run in CI, no GPU):
  - a fixed sample trade row → expected token string;
  - `tokenizer.vocab_size == ` the Step-0.3 hand-count;
  - a corpus line obeys `<bos> trade (<sep> trade)* <eos>` and `len ≤ 4096`;
  - the split invariants from Step 0.4.
**Done when:** all golden tests pass and the load-time assertion is wired.

## Step 0.6 — Compute the baseline (before any GPU spend)

**What:** Compute the **popularity** and **repeat-last-item** baselines on the eval slice, for every metric the Phase-1 panel uses.
**Why:** [Doc 09 checklist](../04-data/09-zkai-internal-datasets.md#5-checklist-before-your-first-corpus) — *beat the heuristic before celebrating.* Wallets are habitual; repeat-last is a strong baseline. Computing it now means the Phase-1 gate is a real comparison, not a vibe.
**How:** on the held-out wallets, score: next-side = majority class; next-item = popularity rank **and** repeat-last; next-amount = last-amount. Record per metric and per activity tier.
**Done when:** baseline table saved as a Loom data object / artifact, referenced by the Phase-1 experiment ID.

## Step 0.7 — Record the data provenance anchor

**What:** Put the data snapshot boundary into the corpus / data-object name.
**Why:** The datasets tier is a continuously-growing log flushed every 5 min ([doc 09 §2](../04-data/09-zkai-internal-datasets.md#2-the-three-storage-tiers--build-only-on-one-of-them)); the date-prefix range is the [C2](../03-learning-path/level-400-design-contracts-and-extensions.md#1-the-contracts)-equivalent — the only thing that makes the corpus reproducible.
**Done when:** the corpus artifact name encodes the GCS date-prefix range (and/or the mart `MAX(event_date)` snapshot), and it's `loom ingest`-ed as one data object so the lineage can't lie.

---

## Advance gate (Phase 0 → Phase 1)

- [ ] Catalog reachable; live schema confirmed; reading from the **datasets tier**; mart freshness checked
- [ ] Entity/event/field set pinned; every field earns its token; **no identity token** (T2)
- [ ] `vocab_size` and `chunk_size` **hand-counted and committed** (C1/C3)
- [ ] Split is **wallet-disjoint × temporal**, verified leak-free by golden test (T2/C6)
- [ ] Tokenizer config + vocab hash persisted and **asserted at load**; golden tests green (E2)
- [ ] **Popularity + repeat-last baselines computed** and saved (before any GPU)
- [ ] Data snapshot range recorded in the corpus artifact name (C2 equivalent)

When every box is checked, proceed to [Phase 1](phase-1-first-production-run.md).
