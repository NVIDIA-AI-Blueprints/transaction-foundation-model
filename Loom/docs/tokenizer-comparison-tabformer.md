# Hand-coded vs. Loom-built tokenizer — TabFormer

> **Status:** current. **Last updated:** 2026-06-18.
> A comparative analysis of two tokenizers for the **same dataset** (IBM TabFormer, 24,386,901
> credit-card transactions): the repo's **hand-coded** `FinancialTabularTokenizer` (`src/tokenizer/`)
> vs. the tokenizer **Loom designed automatically** (`loom propose`, learned over the full corpus
> via the streaming path). Numbers are real and reproducible — hand-coded from `configs`/`docs`
> (and Loom's byte-identical `--preset financial`), Loom-built from `loom ingest --streaming on`
> → `loom propose` on the full 2.2 GB file.

---

## The two numbers

| | **Hand-coded** (`FinancialTabularTokenizer`) | **Loom-built** (`loom propose`) |
|---|---|---|
| **vocab_size** | **6,251** | **5,410** |
| tokens / event | 12 | 15 |
| how it was made | ~1,507 lines of bespoke code, expert-authored | 3 commands (`ingest → propose → tokenize`), auto-designed from EDA |
| identity column | **included** (`CUST`, 3,000 tokens) | **excluded** (entity, T2 leakage) |
| contracts | none (shipped a silent `MONTH_12`/`CARD_0` collision on id 2179) | C1/C2/C3 checked — the collision is *impossible* (refused with a named diff) |

> Loom can also *reproduce* the hand-coded design exactly — `loom tokenize --preset financial` → **6,251**,
> collision-free. So this is hand-coded vs. **Loom's own from-scratch proposal**, not a tooling limit.

---

## Where the vocabulary goes (the key insight)

Both tokenizers spend most of their vocabulary on **high-cardinality counterparty/geography** — but they
allocate the rest very differently:

**Hand-coded — 6,251 tokens, 12 fields:**

| Field | Strategy | Tokens | % |
|---|---|---:|---:|
| `CUST` (customer identity) | FixedVocab 0–2999 | 3,000 | 48% |
| `MERCH` (merchant) | Hash | 2,000 | 32% |
| `ZIP3` (zip prefix) | FixedVocab 0–999 | 1,000 | 16% |
| `MCC` | Mapping | 110 | |
| `STATE` | Mapping | 58 | |
| `HOUR` / `MONTH` / `DOW` | calendar | 24 / 12 / 7 | |
| `CAT` (MCC→industry group) | MappingRange | 14 | |
| `CARD` | FixedVocab | 10 | |
| `AMT` | log-bins | 7 | |
| `CHIP` | Mapping | 4 | |
| specials | | 5 | |

→ **80% of the hand-coded vocab is identity + merchant** (`CUST` + `MERCH`). All the "semantic" fields
(amount, mcc, state, industry, card, chip, calendar) total ~246 tokens — **4%**.

**Loom-built — 5,410 tokens, 15 fields:**

| Field | Strategy | Tokens | % |
|---|---|---:|---:|
| `merchant_city` | Hash (corpus-scaled) | 2,439 | 45% |
| `zip` | Hash (corpus-scaled) | 2,439 | 45% |
| `merchant_state` | Mapping (all 223 states) | 224 | |
| `mcc` | Mapping | 110 | |
| `time_gap` | TimeDelta | 32 | |
| `day` / `year` / `month` | FixedVocab | 31 / 30 / 12 | |
| `time_hour` / `time_month` / `time_dow` | calendar | 24 / 12 / 7 | |
| `errors` | Mapping | 24 | |
| `card` | FixedVocab | 9 | |
| `amount` | log-bins | 8 | |
| `use_chip` | Mapping | 4 | |
| specials | | 5 | |
| **excluded** | `User` (entity) · `Merchant Name` (free-text, >100k distinct) · `Is Fraud?` (target) | — | |

→ **0% identity** (the wallet/customer is the sequence *owner*, not a feature) and **90% counterparty/geo
hashing** (`merchant_city` + `zip`). It keeps the *complete* `STATE` mapping (all 223) and adds calendar +
gap + `errors`.

---

## The trade-offs, axis by axis

### 1. Identity — `CUST` (3,000 tokens, **48% of the hand-coded vocab**)
- **Hand-coded includes a per-customer token.** It lets the model memorise per-customer behaviour in TabFormer's *closed* 2,000-customer world — useful for the fraud task it was built for. But it is **identity leakage**: there is no token for a customer the model never saw (→ `UNK`), so it cannot transfer, and the TabFormer docs themselves flag `CUST_*` as "flattering identity-flavoured tokens" (Level-400 sharp-edge #1).
- **Loom excludes the entity (T2).** The per-customer *sequence* is the unit of learning, not a per-customer token. This is leakage-safe and transfers to new customers, at the cost of any deliberate per-customer signal (which is arguably memorisation, not generalisation).
- **Trade-off:** identity tokens trade transfer for in-distribution memorisation. For a *foundation* model meant to generalise, Loom's exclusion is the more defensible default; the hand-coded `CUST` suits a closed-world supervised task.

### 2. Hash sizing — fixed vs. corpus-scaled
- **Hand-coded:** `MERCH` fixed at 2,000 buckets, `ZIP3` at 1,000 — hyperparameters the author chose. Portable and reproducible (the vocab is the same regardless of data volume), but the bucket count is arbitrary w.r.t. the real cardinality.
- **Loom:** buckets = `corpus_events / 10,000` (clamped) → **2,439** at 24.4M rows. Right-sized to the corpus (more data → more buckets → fewer collisions), but the vocab size now *depends on corpus volume* — a sample yields a smaller vocab, so the locked vocab must be learned at full scale (this is exactly why the streaming path matters).
- **Trade-off:** principled-but-data-coupled vs. simple-but-arbitrary. Neither is wrong; Loom's is automatic.

### 3. Domain structure — expert grouping vs. flat categories
- **Hand-coded encodes domain knowledge:** `MCC → CAT` (14 *industry ranges*, grouping ~110 codes into industries) and `ZIP → ZIP3` (the first-3-digit *geographic prefix*, 1,000). These are insightful structures — industry semantics, geographic locality.
- **Loom (auto) is flat:** `mcc → mapping` (110, one token per code) and `zip → hash` (2,439, no geographic prefix). It does **not invent** the industry grouping or the ZIP-prefix — those are domain insights, not inferable from cardinality.
- **Trade-off:** this is where hand-coding adds genuine value the generic proposer can't. *Loom can express both* (the field-map supports range-mappings and prefix strategies) — `propose` just doesn't suggest them by default. You'd **edit** the proposed spec to add `ZIP3`/industry-range if you want that structure.

### 4. High-cardinality — hash vs. drop
- **Hand-coded hashes** `MERCH` (2,000) — keeps the merchant/counterparty signal, with collisions.
- **Loom drops `Merchant Name`** (100,343 distinct, over the 100k free-text cutoff) by default — conservative, avoids a noisy ultra-high-card field, but loses that signal unless re-included as `hash` via the edit loop. (Loom *did* keep `merchant_city` as a coarser hashed proxy.)
- **Trade-off:** Loom's free-text drop is over-conservative for a *meaningful* counterparty — you'd edit it back. It matters little here (TabFormer has no counterparty *network* — users transact only with merchants), but it matters a lot on-chain (next section).

### 5. Effort, auditability, safety
- **Hand-coded:** ~1,507 lines across 7 files, weeks of expert work; shipped a *silent* `MONTH_12`/`CARD_0` collision (both → id 2179) found only by reading the code.
- **Loom:** 3 commands, minutes; a reason per field; an editable spec; and C1/C2/C3 contracts that make the collision **impossible** (it refuses with a named diff, writes nothing).

---

## So which is "better"?

Neither dominates — they sit at two ends of a spectrum, and the right answer is to **use both**:

- **Loom's auto-proposal is the fast, leakage-safe, contract-checked *starting point*.** Best for a new/unknown schema, an honest baseline, and avoiding identity leakage + silent collisions for free.
- **The expert's hand-coded design is the domain-tuned *endpoint*.** Best for the ~5% of domain structure that actually matters (industry grouping, geographic prefixes, deliberate identity/counterparty decisions, fixed portable sizing).

The intended workflow is the union: **`propose` → review/edit the field-map (inject the domain structure) → `tokenize`.** Loom does the grunt work and the safety; the human adds the domain insight — instead of hand-writing 1,500 lines and hoping there's no collision. And the final vocab should be **learned on the full corpus** (streaming), because *both* the mapping value-sets *and* the hash sizing depend on it.

---

## Forward: carrying this to the cross-chain bet / swap / predict dataset

The next build is a tokenizer for cross-chain user actions (bet / swap / predict). The lessons above map directly — and the existing **`chain` preset** (`SIDE` 3, `ITEM` 5,000 hash, `SIZE` 8 bins, `GAP` 32, `VENUE` 3, `HOUR`/`DOW`; **no wallet-identity token** — 5,082 vocab) is the on-chain template.

What carries over:

1. **Exclude the wallet identity (T2).** Loom does this by default; the `chain` preset already omits it ("a wallet-identity hash would leak identity"). The per-wallet *sequence* is the unit.
2. **KEEP the counterparty / pool / market — do not let it drop.** Unlike TabFormer (no counterparty network), on-chain data has *true* counterparty structure (`from→to`, pools, prediction markets) — this is core signal. Expect `propose` to flag a high-card counterparty as free-text and drop it; **edit it back to `hash`** (or mapping top-N + hash tail). This is exactly trade-off #4, and here it's load-bearing.
3. **Asset / token → hash** (thousands of assets) — like `ITEM`. The action verb (**bet/swap/predict**) → a small `mapping` (high signal, like `SIDE`). The **chain** → a small `mapping`.
4. **Amount / size → log-bins**, but watch the numeric format: crypto amounts span a huge dynamic range and arrive as **wei integers / scientific notation / decimal strings** — verify they're detected as continuous (the currency/numeric-string + code-name guards apply), not hashed or mis-binned.
5. **Use the streaming path.** The corpus is large and a *sample would miss assets and counterparties* — the same failure as the 10 missing `merchant_state`s on TabFormer. Learn the complete value-sets from the full corpus.

Recommended sequence: `loom ingest --streaming on` → `loom propose` → **edit** (keep counterparty, size the asset hash, confirm amount→log-bins, pick the action mapping) → `loom tokenize` → `loom embed`/`evaluate`. Start from the `chain` preset's shape as the reference.
