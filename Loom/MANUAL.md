# Loom — User Manual (v0.1, Phase-0)

> **Status:** v0.1 Phase-0 slice — built and runnable. **Last updated:** 2026-06-15
> **Full vision:** [`DESIGN.md`](./DESIGN.md) · **Engineering notes:** [`NOTES.md`](./NOTES.md) · **Package overview:** [`README.md`](./README.md)

---

## 1. What Loom is

Loom is an **agent harness for training SOTA foundation models** on sequential transaction data. Its core idea is a **typed-contract narrow waist**: every operation is a *verb* you declare once and then drive two ways — as a human CLI command (`loom <verb> …`) and as an agent tool (`loom.<verb>(…)`) — sharing one **result envelope**. Verbs consume and produce content-addressed **data-objects** referenced by stable pathspecs (`Corpus/1`, `IngestDataset/1`, `Baseline/1`). The design principle is **compile before you spend**: each verb checks typed *contracts* (C1/C2/C3…) and refuses, with a named diff instead of a stack trace, before any expensive or silently-wrong work happens. The top-level help puts it plainly: *"Loom — typed verbs you compile before you spend. A human and a Claude/Codex agent drive the identical verbs."*

For the complete lifecycle vision (the model/eval half — `pretrain`/`embed`/`evaluate`/`report`, GPU/NeMo, AIDE search, cost gating), read [`DESIGN.md`](./DESIGN.md). This manual documents **only what is built and runs in v0.1**.

---

## 2. Status & scope

```
┌──────────────────────────────────────────────────────────────────────┐
│ Loom v0.1 — Phase-0 slice                                              │
│                                                                        │
│ BUILT & RUNNABLE (this manual):                                        │
│   • 3 verbs: tokenize · ingest · baseline                              │
│   • CPU only — pandas/numpy. ZERO GPU, zero ML deps. Every verb <1s.   │
│   • Local content-addressed object store under .loom/objects/          │
│   • Dual driver: CLI command == agent tool, byte-identical results     │
│                                                                        │
│ NOT BUILT YET (roadmap — see §9 and DESIGN.md, do NOT expect these):   │
│   • Verbs: pretrain · embed · evaluate · report (the model/eval half)  │
│   • Real GPU / NeMo / real training                                    │
│   • Metaflow execution, AIDE tree-search, live TUI                     │
│   • Launch/cost gating, spend caps, PLAN/confirm round-trips           │
│   • Inspector commands (ls/show/cat/inspect) — none exist              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Install

Requires **Python ≥ 3.10** (developed on 3.12). Loom is an **editable install** of the `loom` package living inside the transaction-foundation-model repo.

```bash
# From the TFM repo, editable-install the package:
pip install -e ./Loom

# Runtime deps are minimal (pulled automatically): numpy>=1.23, pandas>=1.5,
# PyYAML>=6.0, packaging>=21.0.  No GPU / ML deps.
```

Verify the install:

```bash
loom --help
python -c "import loom; print(loom.__version__)"   # -> 0.1.0
```

The CLI entry point is `loom` (`[project.scripts]: loom = "loom.cli:main"`). You can also run it as a module:

```bash
python -m loom --help
```

In this environment the installed binary and venv python are:

```
/Users/anub/Work/transaction-foundation-model/Loom/.venv/bin/loom
/Users/anub/Work/transaction-foundation-model/Loom/.venv/bin/python
```

---

## 4. Core concepts

### 4.1 Verbs

A verb is a single typed declaration that becomes both a CLI command and an agent tool. v0.1 registers **exactly three** verbs — no aliases, no hidden utilities:

| Verb | One line |
|---|---|
| `tokenize` | compile a declarative tokenizer spec to a Corpus (C1/C2/C3 checked, <1s, no GPU) |
| `ingest` | register a dataset as a versioned, content-addressed object (schema sniff + EDA leakage gate) |
| `baseline` | compute popularity + repeat-last-item baselines (the control a model must beat) |

### 4.2 Data-objects & pathspecs

Every verb output is a **data-object** addressed as `Type/<n>` (e.g. `Corpus/1`, `IngestDataset/2`, `Baseline/1`). The `<n>` is minted by an atomic per-kind counter that starts at 1. Objects are **content-addressed**: the same inputs + same spec produce the same `content_id`, and re-running yields the *same* pathspec — no duplicate twin is written. Objects live on disk under `.loom/objects/` (see §7). You thread objects between verbs by passing the pathspec to `--in` (e.g. `loom baseline --in IngestDataset/1`).

### 4.3 The result envelope & VERDICTs

Every verb returns one stable **result envelope**. The human view is a pretty card on stderr; the machine view (`--json`) is the same data. Two fields drive everything:

- **`status`** — `OK` / `REFUSED_CONTRACT` / `FAIL` (Phase-0 verbs emit these three).
- **`verdict`** — `PASS` / `REVIEW` / `FAIL` / `INCOMPLETE`.

Process **exit code**: `1` if `verdict==FAIL` or `status==FAIL`; `2` if `status` starts with `REFUSED_` *and* verdict isn't FAIL; else `0`. **Caveat:** a contract refusal sets both `status=REFUSED_CONTRACT` *and* `verdict=FAIL`, and the FAIL check wins — so a contract refusal exits **1**, not 2.

Full envelope key reference is in §6.2.

### 4.4 Contracts in plain language

Loom's job is to catch silently-wrong tokenization *before* you train. The contracts the verbs actually check:

- **C1 — injectivity + density.** Every `(step, value)` maps to a *unique* id, and the vocab is *dense* (`0..vocab_size-1`, no gaps, no overlaps). This is the reference bug Loom exists to kill: the original tokenizer keyed ids by raw value, so `MONTH_12` and `CARD_0` both resolved to id 2179 and id 2167 was dead — training ran and produced a plausible-but-garbage loss (see [`NOTES.md`](./NOTES.md) §"The reference bug Loom fixes"). On any collision, the verb **refuses to write** and emits a named diff.
- **C2 — determinism.** Vocab is config-only ⇒ deterministic ⇒ INFO PASS. If you pick a fitted `--amount-strategy` (`quantile`/`kmeans`) the vocab depends on a fitted artifact, so C2 downgrades to a WARNING (state must persist). It never blocks the write in v0.1.
- **C3 — grammar.** `chunk_size = context_len // (tokens_per_txn + 1)`, derived and announced on every run. A `chunk_size < 1` is a hard C3 refusal.
- **C6 — eval split / ordering** (baseline): entity sort + leave-one-last-out. `--eval-split` is recorded but only `temporal` is implemented.
- **EDA — leakage scan** (ingest): advisory only → `REVIEW` verdict, never a hard FAIL.

### 4.5 The dual-driver model

`loom <verb> …` and the agent tool `loom.<verb>(…)` run the **identical** verb function and produce the **same** envelope. Three global flags shape the output (accepted both **before** and **after** the verb):

```
loom [--json] [--experiment EXPERIMENT] [-q] <verb> …
```

- `--json` — print the raw result envelope (byte-identical to the agent tool result).
- `--experiment EXPERIMENT` — the join key threading runs together. **Not part of the content address** — the same spec dedupes across experiments.
- `-q, --quiet` — print *only* the output pathspec on stdout (the human card goes to stderr).

```bash
loom tokenize --preset financial -q          # stdout is exactly: Corpus/1
loom --experiment exp-42 tokenize --preset financial   # threads exp-42 onto the run
```

---

## 5. The verbs

### 5.1 `tokenize`

**What it does:** compiles a declarative tokenizer field-spec into a **Corpus** object — deriving `vocab_size`, `vocab_hash`, `tokens_per_txn`, and `chunk_size`, and checking contracts C1 (injective+dense), C2 (determinism), C3 (chunk+grammar). Runs in <1s on CPU. If you pass a readable `--in IngestDataset/<n>`, it also materializes corpus lines; with no input, the Corpus carries the compiled vocab + signature only.

**Flags:**

| Flag | Meaning | Default |
|---|---|---|
| `--in IN` | input `IngestDataset/<n>` pathspec; optional (vocab is config-only, so a missing dataset is non-fatal) | — |
| `--preset PRESET` | `financial` \| `chain` | `financial` |
| `--include-time-delta` | add the TDIF time-delta field (financial only) | off |
| `--merchant-hash-size N` | merchant hash buckets (financial); for `chain` this drives `item_hash_size` (default 5000) | 2000 |
| `--amount-strategy` | `fixed` \| `quantile` \| `kmeans`; quantile/kmeans flips C2 to a fitted-artifact WARNING | `fixed` |
| `--drop-step DROP_STEP` | drop one named step (e.g. `cust`) — **single step only** | — |
| `--reorder-step REORDER_STEP` | **surface-only / NO-OP in v0.1** — accepted but never read; passing it changes nothing | — |
| `--no-identity-token` | chain only: keep wallet identity out of vocab (chain default is already identity-OFF) | — |
| `--eval-split EVAL_SPLIT` | `temporal` \| `entity-disjoint` — accepted but **not consumed by tokenize** in this slice | — |
| `--context-len CONTEXT_LEN` | model context length | 4096 |
| `--confirm-token CONFIRM_TOKEN` | agent second-call confirm token — **inert in v0.1** (no verb is gated) | — |

> There is **no** `--schema` CLI flag (it is read internally as a synonym for `--preset` but cannot be passed). Use `--preset`.

#### Example A — financial PASS (the happy path)

```bash
loom tokenize --preset financial
```

```
✓ tokenize  status=OK  verdict=PASS
  tier=workspace-write
  Corpus/1 verdict=PASS vocab=6251 tokens/txn=12 chunk_size=315 sig=sha256:ba0e0daa6c1…
  → Corpus/1
  · C2: C2 determinism OK — vocab is config-only, no fitted artifact.
  · C3: C3: chunk_size 315 = 4096 // (12 + 1) — derived & announced.
  · C3: no input rows materialized (dataset missing/empty) — Corpus carries the compiled vocab + signature only.
      fix: pass an IngestDataset/<n> with readable rows to emit corpus lines.
```

The financial preset's 12 steps: `amt, merch, cat, mcc, hour, dow, month, card, chip, zip3, state, cust`. Adding the time-delta field shifts the numbers (vocab 6283, chunk 292):

```bash
loom tokenize --preset financial --include-time-delta
# → Corpus/N  vocab=6283 tokens/txn=13 chunk_size=292
```

Dropping a step shrinks the vocab (here `cust` → vocab 3251, chunk 341):

```bash
loom tokenize --preset financial --drop-step cust
# → Corpus/N  vocab=3251 tokens/txn=11 chunk_size=341
```

#### Example B — the chain preset

```bash
loom tokenize --preset chain
```

```
✓ tokenize  status=OK  verdict=PASS
  tier=workspace-write
  Corpus/2 verdict=PASS vocab=5082 tokens/txn=7 chunk_size=512 sig=sha256:aef29f09b02…
  → Corpus/2
  · C2: C2 determinism OK — vocab is config-only, no fitted artifact.
  · C3: C3: chunk_size 512 = 4096 // (7 + 1) — derived & announced.
  · C3: no input rows materialized (dataset missing/empty) — Corpus carries the compiled vocab + signature only.
      fix: pass an IngestDataset/<n> with readable rows to emit corpus lines.
```

Chain step_names: `venue, side, item, size, gap, hour, dow`.

#### Example C — a C1 contract violation (the teaching moment)

This is *the* reason Loom exists. The reference tokenizer once let **`MONTH_12` and `CARD_0` collide on the same id 2179** (id 2167 left dead) — training ran and produced a garbage-but-plausible loss. Loom's C1 check refuses to write a Corpus on any such collision, surfacing a named diff instead of a silent failure or a stack trace.

No CLI flag can trigger a C1 failure in v0.1 (the presets are always valid and `--reorder-step` is a no-op), so to *see* the check fire you hand-build a colliding spec via the engine API:

```python
from loom.engine.api import TokenizerSpec, FieldStep, FixedVocab
from loom.engine import compile_spec

spec = TokenizerSpec(steps=(
    FieldStep("a", "a", FixedVocab("DUP", 0, 0, 0)),
    FieldStep("b", "b", FixedVocab("DUP", 0, 0, 0)),   # same token "DUP_0"
), preset="handbuilt")
c = compile_spec(spec)
```

The compiled report fails C1 with a named diagnostic:

```
injective: False  dense: True  passed: False
[C1/error] C1 injectivity FAIL: token 'DUP_0' is assigned to more than one id (last in step 'b').
   fix: two steps emit the same token string; rename the prefix on one step so every (step,value) maps to a unique token.
   data: {'token': 'DUP_0', 'id': 6, 'step': 'b'}
[C2/info] C2 determinism OK — vocab is config-only, no fitted artifact.
[C3/info] C3: chunk_size 1365 = 4096 // (2 + 1) — derived & announced.
```

Driven through the verb layer, the verb **refuses and writes nothing**:

```
status: REFUSED_CONTRACT  verdict: FAIL  exit_code: 1
outputs: []
summary: REFUSED_CONTRACT: C1 injectivity/density failed for preset 'handbuilt' — no Corpus written
         (the named diff explains the collision; reordering shifts every id ⇒ vocab_hash changes ⇒ retrain required).
wrote_corpus: False
```

After the refusal `.loom/` is never even created — proof that a contract failure lands no object. **The fix** is exactly what the diff says: rename the colliding prefix on one step so every `(step, value)` maps to a unique token (the real-world analogue of fixing the MONTH/CARD overlap with 0-based contiguous id blocks).

---

### 5.2 `ingest`

**What it does:** registers a source dataset as a versioned, **content-addressed** `IngestDataset` object. It sniffs the schema, runs an advisory **EDA leakage scan**, and is **idempotent** — the same source + same spec dedupes to the same pathspec (content-addressed by `source_fingerprint + spec_hash`). Reads `.parquet`/`.pq` (needs a parquet engine installed), `.csv`/`.csv.gz`/`.tsv`, either a single file or a directory.

**Flags:**

| Flag | Meaning |
|---|---|
| `--in IN` | source path/URI (single file or directory); also accepts the positional `input` |
| `--name NAME` | human name for the dataset |
| `--entity ENTITY` | grouping entity column (e.g. `wallet`, `cust`) |
| `--event EVENT` | event row semantics (e.g. `trade`, `txn`) |
| `--target TARGET` | label column; enables target-leakage correlation/determinism scan |
| `--force` | re-pull a moving source as a new object (bypasses idempotent dedupe) |
| `--confirm-token` | inert in v0.1 |

#### Example — ingest a CSV, then re-run it (idempotency)

The examples in §5.2–§5.3 and §7 all use this exact 5-row demo CSV (a tiny synthetic TabFormer-shaped frame; the numbers below are reproducible from it byte-for-byte):

```bash
cat > txns.csv <<'CSV'
cust,card,amount,mcc,merchant,chip,zip,state,datetime
0,0,$12.50,5411,WHOLE FOODS #123,Swipe Transaction,94107,CA,2026-01-02 09:15:00
0,1,$4.00,5814,BLUE BOTTLE,Chip Transaction,94110,CA,2026-01-02 13:40:00
1,0,$1500.00,4111,BART,Online Transaction,94612,CA,2026-02-14 18:05:00
1,0,$0.99,5942,AMAZON,Online Transaction,10001,NY,2026-03-30 23:59:00
2,2,$87.20,5912,CVS PHARMACY,Swipe Transaction,60601,IL,2026-12-25 07:30:00
CSV

loom ingest --in txns.csv --name demo --entity cust --event txn
```

```
⚠ ingest  status=OK  verdict=REVIEW
  tier=workspace-write
  IngestDataset/1 'demo' rows=5 eda VERDICT: REVIEW
  → IngestDataset/1
  ⚠ EDA: column 'cust' looks identity-like (id-shaped column name): 3/5 distinct (60% unique)
      fix: if 'cust' is the grouping entity, pass it as --entity (it is then never tokenized as a feature, T2); otherwise drop it before tokenizing
  ⚠ EDA: column 'amount' looks identity-like (near-unique values (looks like a row/entity key)): 5/5 distinct (100% unique)
      fix: if 'amount' is the grouping entity, pass it as --entity (it is then never tokenized as a feature, T2); otherwise drop it before tokenizing
  ⚠ EDA: column 'mcc' looks identity-like (near-unique values (looks like a row/entity key)): 5/5 distinct (100% unique)
      fix: if 'mcc' is the grouping entity, pass it as --entity (it is then never tokenized as a feature, T2); otherwise drop it before tokenizing
  ⚠ EDA: column 'merchant' looks identity-like (near-unique values (looks like a row/entity key)): 5/5 distinct (100% unique)
      fix: if 'merchant' is the grouping entity, pass it as --entity (it is then never tokenized as a feature, T2); otherwise drop it before tokenizing
  ⚠ EDA: column 'zip' looks identity-like (near-unique values (looks like a row/entity key)): 5/5 distinct (100% unique)
      fix: if 'zip' is the grouping entity, pass it as --entity (it is then never tokenized as a feature, T2); otherwise drop it before tokenizing
  ⚠ EDA: column 'datetime' looks identity-like (near-unique values (looks like a row/entity key)): 5/5 distinct (100% unique)
      fix: if 'datetime' is the grouping entity, pass it as --entity (it is then never tokenized as a feature, T2); otherwise drop it before tokenizing
```

Notes on what you're seeing:
- The EDA verdict is **REVIEW** — *advisory*, never a hard FAIL. The scan flags id-shaped column **names** (`cust`) and **near-unique** columns (`amount`, `mcc`, `merchant`, `zip`, `datetime` — all at 100% unique on these 5 rows). It flags `cust` *even though* you passed it as `--entity`, because the heuristic is purely column-shape based. (On a real-scale dataset the near-unique flags on `mcc`/`zip` would clear; with only 5 rows almost every column looks near-unique — which is itself a useful reminder that the scan is advisory.)
- **Idempotency:** run the *identical* command again and you get the same `IngestDataset/1` — no twin object is written. Use `--force` only to deliberately re-pull a moving source as a new object.

---

### 5.3 `baseline`

**What it does:** computes the **control a model must beat** — popularity (Prec@K), repeat-last-item (Prec@1), and (when an amount column is present) a next-amount last-value MAE — via leave-one-last-out per entity. Marked `capability=searchable` (a declared property; nothing actually searches in v0.1).

**Flags:**

| Flag | Meaning | Default |
|---|---|---|
| `--in IN` | input `Corpus/<n>` or `IngestDataset/<n>`; in practice it reads the rows payload, so an **IngestDataset** is what works | — |
| `--task TASK` | `next-item` \| `fraud-auprc` — **accepted but not branched on**; the verb always runs the leave-one-last-out controls (`fraud-auprc` not separately implemented) | — |
| `--k K` | Prec@K cutoff | 5 |
| `--kind KIND` | `popularity` \| `repeat-last-item` \| `both` | `both` |
| `--eval-split EVAL_SPLIT` | `temporal` \| `entity-disjoint` — recorded on the object, but only the **temporal** leave-one-last-out hold-out is implemented | `temporal` |
| `--confirm-token` | inert in v0.1 | — |

#### Example — baseline over an ingested dataset

```bash
loom baseline --in IngestDataset/1 --experiment demo-exp
```

```
✓ baseline  status=OK  verdict=PASS  experiment=demo-exp
  tier=workspace-write  capability=searchable
  Baseline/1  repeat-last-item prec@1=0.0  popularity prec@5=0.0  next-amount MAE=753.755  (n=2, split=temporal)
  → Baseline/1
```

Columns are **auto-inferred** (for this demo: `entity_col=cust, item_col=mcc, amount_col=amount, time_col=datetime, side_col=null`). It uses leave-one-last-out per entity and only counts entities with **≥2 events** (`n_entities_eval=2` here). The `--json` `data.metrics` for this run:

```json
{"popularity": {"metric": "prec@5", "value": 0.0, "topk": ["5411", "4111"]},
 "repeat-last-item": {"metric": "prec@1", "value": 0.0},
 "next-amount-last-value": {"metric": "mae", "value": 753.755, "n": 2}}
```

Baseline **refuses** with `REFUSED_CONTRACT` / C6 if: no input, an unresolvable input, no rows, no entity+item columns, or zero multi-event entities.

---

## 6. Using Loom from an agent

### 6.1 The tool face

One verb declaration produces two faces. `loom.tools.tool_schema(verb)` emits an Anthropic-style tool schema (it takes the `Verb` object, e.g. `tool_schema(loom.registry.get("tokenize"))`; `loom.tools.all_tool_schemas()` returns one per registered verb); `loom.tools.dispatch(name, input_json, …)` runs the *same* verb function with an agent context (`driver="agent", interactive=False`). The real schema for `loom.tokenize` (each property carries its own `type`/`enum`/`description` — abbreviated here only in formatting, the values are verbatim):

```json
{
  "name": "loom.tokenize",
  "description": "compile a declarative tokenizer spec to a Corpus (C1/C2/C3 checked, <1s, no GPU)",
  "input_schema": {
    "type": "object",
    "properties": {
      "in": {"type": "string", "description": "input IngestDataset/<n> pathspec"},
      "preset": {"type": "string", "description": "tokenizer preset: financial | chain", "enum": ["financial", "chain"]},
      "include_time_delta": {"type": "boolean", "description": "add the TDIF time-delta field (T1)"},
      "merchant_hash_size": {"type": "integer", "description": "merchant hash buckets (financial)"},
      "amount_strategy": {"type": "string", "enum": ["fixed", "quantile", "kmeans"], "description": "amount binning; quantile/kmeans is a C2 fitted artifact"},
      "drop_step": {"type": "string", "description": "drop a named step (e.g. cust) — T2"},
      "reorder_step": {"type": "string", "description": "reorder a step (e.g. card:first) — T5"},
      "no_identity_token": {"type": "boolean", "description": "chain: keep wallet identity out of the vocab (T2)"},
      "eval_split": {"type": "string", "description": "temporal | entity-disjoint (C6)"},
      "context_len": {"type": "integer", "description": "model context length (default 4096)"},
      "confirm_token": {"type": "string", "description": "agent second-call confirm token (§5.3)"}
    }
  },
  "_loom": {"tier": "workspace-write", "capability_mode": "none", "disable_model_invocation": false}
}
```

`_loom.disable_model_invocation` is `true` only for IRREVERSIBLE or LAUNCH_AND_TRACK verbs — so it is **false for all three Phase-0 verbs**. Tool capability modes: `loom.tokenize` / `loom.ingest` → `none`; `loom.baseline` → `searchable`. `dispatch` accepts either `"tokenize"` or `"loom.tokenize"`; an unknown verb returns a `FAIL` envelope (`status=FAIL`, summary `unknown verb: '<name>'`).

### 6.2 The `--json` envelope

`--json` (CLI) and the dispatched tool result emit the same envelope, keys in this stable order:

| key | type | meaning |
|---|---|---|
| `verb` | str | verb name |
| `status` | str | `OK` / `REFUSED_*` / `FAIL` (Phase-0 emits `OK`, `REFUSED_CONTRACT`, `FAIL`) |
| `verdict` | str | `PASS` / `REVIEW` / `FAIL` / `INCOMPLETE` |
| `tier` | str | all three verbs = `workspace-write` |
| `capability_mode` | str | tokenize/ingest = `none`; baseline = `searchable` |
| `summary` | str | the one-line human summary (the card body) |
| `outputs` | list[str] | output pathspecs, e.g. `["Corpus/1"]`; empty on a refusal |
| `diagnostics` | list[obj] | named-diff cards: `{contract, severity, message, fix, data}` (contract ∈ C1/C2/C3/C6/EDA; severity ∈ info/warning/error) |
| `data` | obj | verb-specific derived block (see below) |
| `experiment` | str/null | the `--experiment` join key |
| `cost_plan` | obj/null | placeholder — all `None`/`false`/`{}` for these CPU verbs |
| `confirm_token` | str/null | always `null` in v0.1 (gating not wired) |

Per-verb `data` keys:
- **tokenize:** `preset, vocab_size, vocab_hash, tokens_per_txn, chunk_size, context_len, step_names, has_fitted_artifact, pathspec, wrote_corpus, content_id, n_lines, n_txns, parents`.
- **ingest:** `pathspec, name, entity, event, content_id, schema{n_rows,n_cols,columns{…}}, eda{verdict,n_flags,flags[]}, provenance{source,source_kind,files[],n_rows}`.
- **baseline:** `pathspec, input, entity_col, item_col, side_col, amount_col, time_col, eval_split, n_entities_eval, n_rows, k, metrics{…}`.

A real tokenize `--json` envelope (`loom tokenize --preset financial --json`):

```json
{"verb": "tokenize", "status": "OK", "verdict": "PASS", "tier": "workspace-write", "capability_mode": "none", "summary": "Corpus/1 verdict=PASS vocab=6251 tokens/txn=12 chunk_size=315 sig=sha256:ba0e0daa6c1…", "outputs": ["Corpus/1"], "diagnostics": [{"contract": "C2", "severity": "info", "message": "C2 determinism OK — vocab is config-only, no fitted artifact.", "fix": null, "data": {"amount_strategy": "fixed"}}, {"contract": "C3", "severity": "info", "message": "C3: chunk_size 315 = 4096 // (12 + 1) — derived & announced.", "fix": null, "data": {"context_len": 4096, "tokens_per_txn": 12, "chunk_size": 315}}, {"contract": "C3", "severity": "info", "message": "no input rows materialized (dataset missing/empty) — Corpus carries the compiled vocab + signature only.", "fix": "pass an IngestDataset/<n> with readable rows to emit corpus lines.", "data": {}}], "data": {"preset": "financial", "vocab_size": 6251, "vocab_hash": "sha256:ba0e0daa6c1d64a1028e428b7981a82a69fe45a42cc42161277df04aa9152ce4", "tokens_per_txn": 12, "chunk_size": 315, "context_len": 4096, "step_names": ["amt", "merch", "cat", "mcc", "hour", "dow", "month", "card", "chip", "zip3", "state", "cust"], "has_fitted_artifact": false, "pathspec": "Corpus/1", "wrote_corpus": true, "content_id": "344bc8d9bdfde52acb441cf72f69dc20188a14a9ea50296e80ce9fd41ec47aa3", "n_lines": 0, "n_txns": 0, "parents": []}, "experiment": null, "cost_plan": {"derived": false, "usd": null, "confidence": null, "tokens": null, "params": null, "seq_len": null, "gpu_target": null, "envelope": null, "inputs": {}}, "confirm_token": null}
```

### 6.3 The CLI == tool guarantee

The dual-driver invariant is real and load-bearing:

```
dispatch("loom.tokenize", {"preset": "chain"}).to_json()
  is BYTE-IDENTICAL to
loom tokenize --preset chain --json
```

(Verified with `diff` → IDENTICAL.) Whatever you test on the CLI, an agent gets exactly the same result.

---

## 7. Inspecting your work

> **There are no inspector commands in v0.1.** No `loom ls`, `loom show`, `loom cat`, or `loom inspect` — only `tokenize`, `ingest`, `baseline` are registered. (`ObjectStore.list()` exists in `store.py` and its docstring calls itself "the `loom ls` backend", but no verb wires it.) You inspect objects directly on disk.

The store is rooted at `$LOOM_WORKSPACE` (or the current directory) under `.loom/objects/`. After one ingest + one tokenize-with-input:

```
.loom/objects/_counters.json        {"IngestDataset": 1, "Corpus": 1}
.loom/objects/_counters.lock
.loom/objects/_index.json           content_id -> pathspec map
.loom/objects/_index.lock
.loom/objects/IngestDataset/1/object.json
.loom/objects/IngestDataset/1/payload/rows.csv     # ingest writes rows as CSV (no parquet dep)
.loom/objects/Corpus/1/object.json
.loom/objects/Corpus/1/payload/corpus.json         # vocab + grammar + corpus_lines
```

- **Pathspec:** `Type/<n>`; `<n>` is minted by an atomic per-kind counter starting at 1 (POSIX `flock` + `os.replace`).
- **`object.json` keys:** `content_id, cost_actuals, created_at, envelope, experiment, extras, kind, parents, payload_path, producer_args, producer_verb, ref, signatures, status, verdict`.
- **Idempotency:** `put()` looks up `content_id` in `_index.json`; on a hit it returns the existing object and writes no twin.
- **Lineage:** when `tokenize` gets a real `--in IngestDataset/1`, it materializes corpus lines (C3 grammar) and records `parents: ["IngestDataset/1"]` (here `n_lines=3` — one line per entity with events). The first real line from the demo data (cust `0`, two transactions):
  `<bos> AMT_1 MERCH_240 CAT_RETAIL MCC_5411 HOUR_09 DOW_4 MONTH_01 CARD_0 CHIP_SWIPE ZIP3_941 STATE_CA CUST_0 <sep> AMT_0 MERCH_482 CAT_MISC_STORES MCC_5814 HOUR_13 DOW_4 MONTH_01 CARD_1 CHIP_CHIP ZIP3_941 STATE_CA CUST_0 <eos>`.
  With no `--in`, `n_lines=0` and the Corpus carries vocab + signature only.

To read an object's metadata you point your tools at the JSON directly, e.g.:

```bash
python -m json.tool .loom/objects/Corpus/1/object.json
```

---

## 8. Troubleshooting & FAQ

**`status=REFUSED_CONTRACT verdict=FAIL`, `outputs: []` — what happened?**
A typed contract failed (C1/C3) and the verb **refused to write any object** — by design (compile before you spend). Read the `diagnostics[]` named diff: its `fix` field tells you exactly what to change. For a C1 injectivity collision: two steps emit the same token string — rename the prefix on one step so every `(step, value)` maps to a unique token. The process exits **1**.

**`ingest` says `verdict=REVIEW` with EDA warnings — did it fail?**
No. EDA is **advisory** and never a hard FAIL. `REVIEW` means "inspect these columns": id-shaped names and near-unique columns are flagged for possible leakage. If a flagged column is your grouping entity, pass it as `--entity` (so it's never tokenized as a feature, T2); otherwise drop it before tokenizing. The object is still written.

**`baseline` refused / produced no metric.**
Baseline refuses (`REFUSED_CONTRACT` / C6) when there's no input, an unresolvable input, no rows, no entity+item columns, or **zero multi-event entities** (it needs entities with ≥2 events for leave-one-last-out). Make sure your `--in` resolves to an `IngestDataset/<n>` whose rows have an inferrable entity and item column, and that some entities have at least two events.

**`tokenize` says "no input rows materialized".**
That's an INFO, not an error. Without `--in`, the Corpus carries the compiled vocab + signature only (`n_lines=0`). To emit corpus lines, pass `--in IngestDataset/<n>` with readable rows.

**I passed `--reorder-step` / `--eval-split` to tokenize and nothing changed.**
Correct — both are **surface-only** in v0.1. `tokenize --reorder-step` is a no-op (`_build_spec` never reads it) and `tokenize --eval-split` is ignored by tokenize. Likewise `baseline --task` always runs the leave-one-last-out controls (`fraud-auprc` is not implemented) and `baseline --eval-split entity-disjoint` is recorded but only the temporal hold-out runs.

**`loom ls` / `loom show` is "unrecognized".**
There are no inspector commands in v0.1 — only `tokenize`, `ingest`, `baseline`. Inspect objects on disk under `.loom/objects/**/object.json` (§7).

**Why is `cost_plan` all null and `--confirm-token` ignored?**
Cost/launch gating is not wired in Phase-0. `cost_plan` is an all-`None` placeholder, `confirm_token` is always `null`, and `--confirm-token` is inert (the underlying `make/validate_confirm_token` are stubs). These come online with the GPU verbs (§9).

**Re-running a verb didn't create a new object.**
Working as intended — objects are content-addressed, so identical inputs dedupe to the same pathspec. The `--experiment` id is *not* part of the content address, so the same spec dedupes across experiments. Use `ingest --force` to deliberately re-pull a moving source.

---

## 9. What's next

v0.1 is the data half of the lifecycle. The roadmap (see [`DESIGN.md`](./DESIGN.md), Phase 1 = the GPU verbs / **D5 next-trade**) adds:

- The model/eval verbs: **`pretrain`, `embed`, `evaluate`, `report`**.
- Real **GPU / NeMo** training (Phase-0 is CPU-only, every verb <1s).
- **Metaflow** execution (the local content-addressed `ObjectStore` is the v0.2 seam stand-in) and **AIDE tree-search** (today's `capability_mode=searchable` on `baseline` is a declared property only).
- A **live TUI** (today: one-shot pretty cards or `--json`).
- **Launch / cost gating**: `cost_plan`, spend caps, the `PLAN`/confirm round-trip, and the `--confirm-token` flow (today all inert placeholders).

Until then: tokenize your contract, ingest your data, and beat your baseline.
