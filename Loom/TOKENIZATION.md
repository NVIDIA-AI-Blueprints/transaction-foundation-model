# Loom — Tokenization User Manual

> **Status:** current (engine verbs `ingest` · `propose` · `tokenize` are built and runnable; `embed`/`evaluate` validate them locally). **Last updated:** 2026-06-17
> **Scope:** how to build a custom, contract-checked tokenizer for **any** foundation model with Loom — finance, music, genomics, sensor logs, anything with records. Every command and output below is real and reproducible from a clean checkout.
> Related: [`README.md`](./README.md) (overview) · [`ARCHITECTURE.md`](./ARCHITECTURE.md) (how the harness is built) · [`GPU-RUNBOOK.md`](./GPU-RUNBOOK.md) (training, later).

---

## 1. Why tokenization is the thing to get right first

A foundation model never sees your raw data. It sees **tokens** — a finite vocabulary of integers, plus the grammar that says how they're laid out in a sequence. *How you turn records into tokens* decides what the model can possibly learn. Get it wrong (collisions, leakage, a vocabulary that doesn't fit the data, a fitted artifact you can't reproduce) and you find out only after a GPU run that cost real money.

Tokenization is normally weeks of bespoke pipeline code, re-written per dataset, hard to audit. **Loom turns it into three commands** that produce a versioned, **contract-checked** token corpus in under a second, on a laptop, for $0:

```
loom ingest    →    loom propose    →    loom tokenize
(register data)     (AI-grade spec)      (compile to a checked Corpus)
```

Two principles run through it:

- **Domain-agnostic.** Loom doesn't know what a transaction is. It reasons about *columns* — categorical, continuous, timestamp, high-cardinality, identifier, sequence-over-an-alphabet — and picks a tokenization **strategy** per column. The same three verbs tokenize a music-listening log, a DNA dataset, and a payments table. (A finance preset ships in the box, but it's just one example, not the point.)
- **Design local, train at scale.** The tokenizer is **deterministic and config-only** — the vocabulary is a pure function of the spec, not fitted to the rows. So a vocab you design and validate on a laptop **sample** is *bit-identical* on the full cloud-scale corpus. You design here; you train later, elsewhere, unchanged.

---

## 2. Setup

From a checkout of the `Loom/` package:

```bash
cd Loom
python3.12 -m venv .venv
source .venv/bin/activate        # activates the engine CLI `loom`
pip install -e .
loom --help                      # should print the verb list
```

> **Which `loom`?** With the venv active, `loom` is the **engine CLI** (you, a human, running verbs) — that's what this manual uses. There's also a global `loom` (the Pi agent runtime) that lets a Claude/Codex agent call the *identical* verbs as tools (see §10). Same verbs, two drivers. If you ever get the wrong one, `python -m loom <verb>` always runs the engine.

Everything is CPU-only (pandas/numpy). No GPU, no cloud, no API key.

---

## 3. The mental model (30 seconds)

- **Data-objects + pathspecs.** Every verb reads and writes a versioned, content-addressed object referenced by a stable pathspec: `IngestDataset/1`, `TokenizerSpec/1`, `Corpus/1`. Re-running with the same inputs is a free no-op (idempotent). Objects live under `.loom/objects/`.
- **Three tokenization verbs:**
  - `ingest` — register a raw file as a dataset (schema sniff + a leakage/identity scan).
  - `propose` — analyze the columns and emit an **editable** tokenizer spec (which column → which strategy, and *why*).
  - `tokenize` — compile a spec into a **Corpus** (the token vocabulary + grammar), checking contracts **C1/C2/C3** and refusing — with a named diff, not a stack trace — if anything is wrong.
- **Compile before you spend.** The contracts catch silently-wrong tokenizers *before* a Corpus is written, so a bad spec can never reach a training run.

---

## 4. Quickstart — tokenize a dataset in three commands

We'll build a tokenizer for a **music-listening foundation model** — sequences of "user played a track" events. Nothing here is finance-specific; it's just a dataset with the column shapes you meet everywhere.

**Make the sample** (3,000 listening events; reproducible):

```python
# listens.py
import csv, random, datetime as dt
random.seed(7)
genres  = ['pop','rock','jazz','hiphop','classical','electronic']
artists = [f'artist_{i:02d}' for i in range(40)]
tracks  = [f'trk_{i:04d}'   for i in range(200)]
cities  = [f'city_{i:03d}'  for i in range(800)]
base = dt.datetime(2026, 3, 1, 8, 0, 0)
rows = [['user_id','track','artist','genre','city','ts','minutes_played']]
for _ in range(3000):
    rows.append([
        f'u{random.randint(0,49):03d}',
        random.choice(tracks), random.choice(artists),
        random.choice(genres), random.choice(cities),
        (base + dt.timedelta(minutes=random.randint(0, 7*24*60))).isoformat(),
        round(random.uniform(0.3, 5.0), 2),
    ])
csv.writer(open('listens.csv','w',newline='')).writerows(rows)
```

```bash
python listens.py
```

### 4.1 `ingest` — register the data

```bash
loom ingest --in listens.csv --entity user_id --event listen --name music-listens
```

```
⚠ ingest  status=OK  verdict=REVIEW
  tier=workspace-write
  IngestDataset/1 'music-listens' rows=3000 eda VERDICT: REVIEW
  → IngestDataset/1
  ⚠ EDA: column 'user_id' looks identity-like (id-shaped column name): 50/3000 distinct (2% unique)
      fix: if 'user_id' is the grouping entity, pass it as --entity (it is then never tokenized as a feature, T2); otherwise drop it before tokenizing
```

`--entity user_id` declares the **sequence owner** (the model learns one sequence per user). `--event listen` names the row semantics. The leakage scan flags identity-shaped columns up front — here it's just confirming `user_id` is the entity (which we already declared, so it won't be tokenized as a feature). `REVIEW` means "look at these notes," not "error."

### 4.2 `propose` — get an editable spec, with reasons

```bash
loom propose --in IngestDataset/1
```

```
⚠ propose  status=OK  verdict=REVIEW
  tier=workspace-write
  TokenizerSpec/1 verdict=REVIEW  9 fields tokenized, 1 excluded  vocab≈593  tokens/event=9  chunk_size=409  (edit TokenizerSpec/1 then `loom tokenize --spec TokenizerSpec/1`)
  → TokenizerSpec/1
  · PROPOSE: field 'track' (col 'track') → mapping → 201 tokens: categorical, 200 distinct (< 500): Mapping over the observed values + 1 default (size 201)
  · PROPOSE: field 'artist' (col 'artist') → mapping → 41 tokens: categorical, 40 distinct (< 500): Mapping over the observed values + 1 default (size 41)
  · PROPOSE: field 'genre' (col 'genre') → mapping → 7 tokens: categorical, 6 distinct (< 500): Mapping over the observed values + 1 default (size 7)
  · PROPOSE: field 'city' (col 'city') → hash → 256 tokens: high-cardinality, 779 distinct (~merchant/counterparty): Hash into 256 buckets — expect collisions
  · PROPOSE: field 'ts_hour' (col 'ts') → calendar → 24 tokens: timestamp 'ts': calendar HOUR via FixedVocab(0,23) — calendar seasonality, config-only (C2-clean)
  · PROPOSE: field 'ts_dow' (col 'ts') → calendar → 7 tokens: timestamp 'ts': calendar DOW via FixedVocab(0,6)
  · PROPOSE: field 'ts_month' (col 'ts') → calendar → 12 tokens: timestamp 'ts': calendar MONTH (MONTH_00..MONTH_11)
  · PROPOSE: field 'ts_gap' (col 'ts') → timedelta → 32 tokens: timestamp 'ts': inter-event TimeDelta (32 log-bins, seconds→months) — the gap between events
  · PROPOSE: field 'minutes_played' (col 'minutes_played') → amount → 8 tokens: continuous float: 8 log-spaced deterministic threshold bins (no fitted artifact, C2-clean)
  ⚠ EDA: EXCLUDED 'user_id' (entity): column 'user_id' looks identity-like ...
```

In one pass, Loom read seven raw columns and produced a sensible, fully-explained tokenizer:

| Column | → Strategy | Tokens | Why |
|---|---|---|---|
| `track` | `mapping` | 201 | categorical, < 500 distinct → one token per value (+ a default) |
| `artist` | `mapping` | 41 | categorical, < 500 distinct |
| `genre` | `mapping` | 7 | categorical, < 500 distinct |
| `city` | `hash` | 256 | high-cardinality (779 distinct) → hash buckets (collisions expected) |
| `ts` → `ts_hour/dow/month` | `calendar` | 24+7+12 | timestamp → cyclical calendar parts |
| `ts` → `ts_gap` | `timedelta` | 32 | inter-event gap → log-spaced time bins |
| `minutes_played` | `amount` | 8 | continuous float → log-spaced magnitude bins |
| `user_id` | **excluded** | — | the entity (sequence owner) — never a feature (prevents identity leakage) |

`vocab≈593`, `tokens/event=9`, `chunk_size=409`. The spec is saved as `TokenizerSpec/1` and as an editable YAML you can open and change (§6).

### 4.3 `tokenize` — compile to a contract-checked Corpus

```bash
loom tokenize --in IngestDataset/1 --spec TokenizerSpec/1
```

```
✓ tokenize  status=OK  verdict=PASS
  tier=workspace-write
  Corpus/1 verdict=PASS vocab=593 tokens/txn=9 chunk_size=409 sig=sha256:090945f3363…
  → Corpus/1
  · C2: C2 determinism OK — vocab is config-only, no fitted artifact.
  · C3: C3: chunk_size 409 = 4096 // (9 + 1) — derived & announced.
```

That's a real tokenizer: a **593-token vocabulary**, a content hash (`sig`) that identifies it forever, `9` tokens per event, and a `409`-event chunk that fits a 4096-token context. The `C2`/`C3` lines are the contracts confirming the vocab is reproducible and the grammar is derived correctly. **You went from a CSV to a checked tokenizer in three commands.**

---

## 5. The strategies Loom knows (and when it picks them)

`propose` is a column classifier. Each strategy turns one column into a block of tokens:

| Strategy | Picked when | What it emits | Example |
|---|---|---|---|
| `mapping` | categorical, **< 500** distinct | one token per observed value (+ a default) | `genre`, `artist`, `track` |
| `hash` | categorical, **high cardinality** | N hash buckets (collisions expected, bounded vocab) | `city` → 256 |
| `amount` | continuous **float** | log-spaced magnitude bins (config-only, no fitted binner) | `minutes_played` → 8 bins |
| `calendar` | a **timestamp** column | cyclical parts: hour / day-of-week / month | `ts_hour/dow/month` |
| `timedelta` | a timestamp column | the **inter-event gap**, log-binned (seconds→months) | `ts_gap` → 32 |
| `fixedvocab` | a bounded **integer range** | one token per value in `[min, max]` | a 0–9 rating |
| `kmer` | a **sequence over an alphabet** | overlapping k-mers (e.g. DNA codons) | §8 |
| *(excluded)* | the **entity**, the **target**, or an id-shaped / near-unique column | nothing — kept out of the vocab | `user_id` |

Auto-rules worth knowing (they explain every choice above, and are honest about edge cases):

- **Identifiers are excluded, not tokenized.** A column whose *name* looks id-like (`*_id`, `key`, …) **or** whose values are near-unique is treated as an identifier and excluded — tokenizing a near-unique column produces garbage. If it's your sequence owner, pass it as `--entity`; if it's a feature you do want (e.g. an item id), re-include it as `hash` (§6).
- **The entity and target are always excluded** from the vocabulary — the entity to avoid identity leakage, the target to avoid label leakage.
- **`float` → `amount` (bins); integer/string categoricals → `mapping` (< 500) or `hash`.** A currency-formatted string like `"$12.50"` is detected as continuous and binned, not hashed. An *integer* column is treated as categorical by cardinality — if you want a large integer quantity treated as continuous, store it as a float or switch it to `amount` (§6).
- **Timestamps fan out** into calendar parts **plus** the inter-event gap automatically.

---

## 6. You're in control — edit the spec, re-tokenize

`propose` proposes; you decide. The spec is a plain YAML file (`loom-fieldmap/1`) you can read, diff, and edit. Here's the shape (abbreviated — one entry per strategy):

```yaml
# loom-fieldmap/1 — Loom tokenizer spec (edit me, then `loom tokenize --spec <this file>`)
version: loom-fieldmap/1
entity: user_id        # EXCLUDED from the vocab (the sequence owner)
event: listen
target: null           # if set, EXCLUDED as leakage
context_len: 4096
fields:
  - {name: genre,  source: genre,  strategy: mapping, values: [pop, rock, jazz, ...], default: UNK}
  - {name: city,   source: city,   strategy: hash,    buckets: 256}
  - {name: amt,    source: minutes_played, strategy: amount,   bins: 8}
  - {name: hour,   source: ts,      strategy: calendar, part: hour}      # part: hour|dow|month
  - {name: gap,    source: ts,      strategy: timedelta, bins: 32, max_years: 10.0}
  - {name: rating, source: stars,   strategy: fixedvocab, min: 0, max: 9}
  - {name: kmer,   source: sequence, strategy: kmer, k: 3, alphabet: ACGT, stride: 1}
```

**Field-map keys per strategy:** `mapping` → `values` + `default`; `hash` → `buckets`; `amount` → `bins`; `calendar` → `part`; `timedelta` → `bins` (+ `max_years`); `fixedvocab` → `min`/`max`; `kmer` → `k`/`alphabet`/`stride`. (Advanced: an explicit `prefix:` is honored verbatim — useful, but two fields with the same prefix will deliberately trip the C1 contract; see §7.)

**Example edit** — `city` was hashed into 256 buckets, but say you want to *cap* the high-cardinality `track` column instead of giving it 201 mapping tokens. Change its one line to `strategy: hash, buckets: 64` and re-tokenize:

```bash
loom tokenize --in IngestDataset/1 --spec my_edited_spec.yaml
```

```
✓ tokenize  status=OK  verdict=PASS
  Corpus/2 verdict=PASS vocab=456 tokens/txn=9 chunk_size=409 sig=sha256:7516674f765…
  · C2: C2 determinism OK — vocab is config-only, no fitted artifact.
  · C3: C3: chunk_size 409 = 4096 // (9 + 1) — derived & announced.
```

Vocabulary dropped from **593 → 456** and you got a *new* `Corpus/2` with a new content hash. The loop is `propose → edit → tokenize`, and it's instant. (`--spec` accepts either a `TokenizerSpec/n` pathspec or a YAML/JSON file path.)

---

## 7. The safety net — contracts (C1 / C2 / C3)

Every `tokenize` checks three contracts and **refuses to write a Corpus** if any fails:

- **C1 — injective + dense.** Every (field, value) maps to a *unique* token id, and the ids form a gap-free `0..vocab_size-1` range. No two things ever collide onto one id.
- **C2 — determinism.** The vocabulary is config-only — no artifact fitted to the rows — so it's reproducible and scale-invariant.
- **C3 — grammar.** `chunk_size = context_len // (tokens_per_event + 1)` is derived and announced (the `+1` is the event separator).

Watch C1 catch a real mistake. Two fields are given the same token prefix `DUP`, so their default token `DUP_UNK` would land on two different ids:

```bash
loom tokenize --in IngestDataset/1 --spec collision.yaml
```

```
✗ tokenize  status=REFUSED_CONTRACT  verdict=FAIL
  REFUSED_CONTRACT: C1 injectivity/density failed for preset 'custom' — no Corpus written
  (the named diff explains the collision; reordering shifts every id ⇒ vocab_hash changes ⇒ retrain required).
  ✗ C1: C1 injectivity FAIL: token 'DUP_UNK' is assigned to more than one id (last in step 'artist').
      fix: two steps emit the same token string; rename the prefix on one step so every (step,value) maps to a unique token.
```

No stack trace, no silently-broken vocabulary, **no Corpus written** — a named diagnostic that tells you exactly what collided and how to fix it. This is the difference between catching a tokenizer bug in a millisecond on your laptop and discovering it after a multi-hour training run.

---

## 8. Any domain — the same three verbs

Loom isn't built around any one data type. A new modality is a **field-map**, not new code. Here's **genomics**: tokenize DNA into 3-mers (codons) over the `{A,C,G,T}` alphabet — a completely different shape from the music log, through the identical `tokenize` verb.

```python
# dna.py — 60 reads of length 40 over ACGT
import csv, random
random.seed(1)
rows = [['seq_id','sequence']]
for i in range(60):
    rows.append([f's{i:03d}', ''.join(random.choice('ACGT') for _ in range(40))])
csv.writer(open('dna.csv','w',newline='')).writerows(rows)
```

```yaml
# dna_kmer.yaml
version: loom-fieldmap/1
entity: seq_id
event: read
context_len: 4096
fields:
  - {name: kmer, source: sequence, strategy: kmer, k: 3, stride: 1, alphabet: ACGT}
```

```bash
python dna.py
loom ingest   --in dna.csv --entity seq_id --event read --name dna-reads
loom tokenize --in IngestDataset/2 --spec dna_kmer.yaml
```

```
✓ tokenize  status=OK  verdict=PASS
  Corpus/2 verdict=PASS vocab=69 tokens/txn=1 chunk_size=2048 sig=sha256:8f45ca18774…
  · C2: C2 determinism OK — vocab is config-only, no fitted artifact.
  · C3: C3: chunk_size 2048 = 4096 // (1 + 1) — derived & announced.
```

A **69-token** vocabulary: 64 codons (`4³`) + 5 special tokens, each k-mer a token, the read expanded into a sequence of them. The same machinery — strategies, the field-map, C1/C2/C3 — extends to speech (codec tokens), music (MIDI/codec events), or any records you can describe as fields. **Finance, music, and DNA all run through `ingest → propose → tokenize`.**

---

## 9. Batteries included — presets

For common schemas, skip authoring entirely and compile a ready-made tokenizer with one flag:

```bash
loom tokenize --in IngestDataset/1 --preset financial   # → a 6,251-token transaction tokenizer
loom tokenize --in IngestDataset/1 --preset chain        # → an on-chain / DEX tokenizer
```

The `financial` preset compiles a complete, contract-checked **6,251-token** vocabulary (amount bins, merchant hash, MCC industry ranges, calendar, card, ZIP, state, …) — the same kind of spec `propose` builds, just pre-written and frozen. Presets are the fast path; `propose` is the path for *your* schema.

---

## 10. The other driver — let an agent do it

Every verb has two faces from a single definition: the human CLI (`loom tokenize …`) and an **agent tool** (`loom.tokenize(…)`) returning a byte-identical result envelope. With `--json` you see exactly what the agent sees:

```bash
loom tokenize --in IngestDataset/1 --spec TokenizerSpec/1 --json
# {"verb":"tokenize","status":"OK","verdict":"PASS","data":{"vocab_size":593,"vocab_hash":"sha256:…","tokens_per_txn":9,"chunk_size":409, ...}, "diagnostics":[...]}
```

Run the global `loom` (the Pi agent runtime) and a Claude/Codex agent drives the **same** verbs — it can read your data, call `loom.propose`, reason about the field-map, edit it, and call `loom.tokenize` for you — with the identical contracts and refusals. You can hand-author specs or let the agent; the engine, the checks, and the result are the same.

---

## 11. Validate before you train

A tokenizer that *compiles* isn't necessarily a tokenizer worth training on. Loom can validate it **locally, on CPU, for ~$0**, before you spend a GPU-hour:

```bash
loom embed    --in Corpus/1          # fit a quick PPMI-SVD embedding (CPU, deterministic)
loom evaluate --in Corpus/1          # score it: does it beat the trivial baselines a model must beat?
```

`evaluate` returns a `PASS` / `REVIEW` / `FAIL` verdict — checking both that the vocabulary *fits the data* (coverage, dead tokens) and that a cheap embedding of it *captures real structure* (it has to beat the repeat-last-item baseline on a held-out split). A `FAIL` comes with a refine plan naming the exact field-map knob to change — closing the loop back to §6. **Design and validate on a laptop; train only what's earned it.**

---

## 12. Demo script (≈4 minutes)

A clean recording sequence. Prep: a checkout with the venv active, and the `listens.py` + `dna.py` generators ready.

1. **The hook (15s).** "A foundation model only sees tokens. Designing that tokenizer is normally weeks of code per dataset. Watch me build one for a music model — then for DNA — in a couple of minutes, on this laptop, with correctness checked for free."
2. **Ingest (20s).** `python listens.py` then `loom ingest …`. Point at the leakage scan: "it already flagged the identity column."
3. **Propose (45s).** `loom propose --in IngestDataset/1`. Walk the table: "It read my columns and chose — mapping for the categoricals, *hash* for the high-cardinality city, *log-bins* for the continuous minutes, *calendar + gap* for the timestamp, and it **excluded** the user as the sequence owner so identity can't leak. With reasons."
4. **Tokenize (20s).** `loom tokenize … --spec TokenizerSpec/1`. "593 tokens, a content hash, contracts C2/C3 green. That's a real tokenizer."
5. **Control (30s).** Edit one line (`track → hash, buckets: 64`), re-tokenize: "593 → 456. I'm in control; it's instant."
6. **The safety net (30s).** Run `collision.yaml`: "I made a mistake on purpose — two fields collide. It refuses, names the exact collision, and writes *nothing*. That's a bug caught in a millisecond instead of after a GPU run."
7. **Any domain (40s).** `dna.py` → `loom tokenize --spec dna_kmer.yaml`: "Same three verbs, totally different data — DNA into 64 codons. Loom doesn't know what a transaction or a gene is; it reasons about column shapes."
8. **Close (20s).** "Design on a laptop on a sample; the tokenizer is deterministic, so it's bit-identical at cloud scale. Validate it with `loom evaluate` before you ever spend a GPU-hour. That's Loom."

---

## Appendix — command & field-map reference

**Verbs (tokenization path):**

| Verb | Key flags | Produces |
|---|---|---|
| `loom ingest` | `--in <path>` · `--entity <col>` · `--event <label>` · `--target <col>` · `--name <str>` | `IngestDataset/n` |
| `loom propose` | `--in IngestDataset/n` · `--entity/--event/--target` (override) · `--context-len` | `TokenizerSpec/n` (+ editable YAML) |
| `loom tokenize` | `--in IngestDataset/n` · `--spec <TokenizerSpec/n \| file.yaml>` · `--preset financial\|chain` · `--context-len` · `--include-time-delta` | `Corpus/n` |

Add `--json` to any verb for the raw (agent-identical) envelope; `-q` to print only the output pathspec.

**Field-map (`loom-fieldmap/1`) per-strategy keys:**

| `strategy` | required keys | optional |
|---|---|---|
| `mapping` | `values: [...]` | `default` (e.g. `UNK`) |
| `hash` | `buckets: <int>` | — |
| `amount` | `bins: <int>` | — |
| `calendar` | `part: hour\|dow\|month` | — |
| `timedelta` | `bins: <int>` | `max_years` (default 10.0) |
| `fixedvocab` | `min: <int>`, `max: <int>` | `pad` |
| `kmer` | `k: <int>`, `alphabet: <str>` | `stride` (default 1) |

Common fields on every entry: `name`, `source` (the raw column), `strategy`. Top-level: `version`, `entity` (excluded), `event`, `target` (excluded), `context_len`, `fields: [...]`. An explicit `prefix:` overrides the auto-derived token prefix (and is honored verbatim — use with care; see §7).

**Data-objects:** `IngestDataset/n` (registered raw data) → `TokenizerSpec/n` (the editable spec) → `Corpus/n` (the compiled, contract-checked token vocabulary + grammar). Pathspec numbers increment per object kind as you create them.
