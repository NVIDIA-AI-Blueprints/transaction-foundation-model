---
description: Compile a declarative tokenizer spec into a contract-checked training corpus (Corpus/<n>), deriving the vocab and chunking and enforcing C1/C2/C3.
argument-hint: <IngestDataset/n pathspec, or a preset>
---

# /loom-tokenize — compile a tokenizer spec into a Corpus (workspace-write)

This is the contract-bearing verb. It compiles a **declarative tokenizer spec**
into a **Corpus** — deriving `vocab_size`, `vocab_hash`, `tokens_per_txn`, and
`chunk_size`, and checking the three load-bearing contracts before any expensive
work. Getting tokenization right is the whole game.

Tokenize: $@

## 1. Intake — pick the spec
- `--preset financial` (the TabFormer 12-field recipe) or `--preset chain` (the
  DEX next-trade field set). Default `financial`.
- `--in <IngestDataset/n>` to materialize corpus lines from a registered dataset;
  without it, the Corpus carries the compiled vocab + signature only (a dry
  compile). Take the pathspec from a prior `/loom-ingest`.
- Optional shape: `--include-time-delta` (adds the TDIF inter-event token),
  `--drop-step <step>` (e.g. `cust` — the deployability ablation), `--amount-strategy
  fixed|quantile|kmeans`, `--merchant-hash-size`, `--context-len`, `--no-identity-token`.

## 2. Plan — workspace-write / cheap (compiles in <1s, no GPU)
State what you're compiling and what it implies, *before* running: a preset/flag
change changes the vocabulary, which means a retrain later. Honor the prior
`/loom-ingest` leakage flags — a flagged non-entity column should be dropped; the
grouping entity should never be tokenized as a feature.

## 3. Run — call the `loom_tokenize` tool
Call `loom_tokenize` with the chosen `preset`/`in`/flags. It compiles the spec and
checks: **C1** (the vocabulary is injective and dense — no two tokens share an id),
**C2** (determinism — config-only vocab; a fitted `amount_strategy` is flagged),
**C3** (`chunk_size = context_len // (tokens_per_txn + 1)`, plus the corpus grammar).

## 4. Verify — read the contracts, stop on a violation
- On **PASS**: report the derived `vocab_size`, `tokens_per_txn`, `chunk_size`, and
  the `Corpus/<n>` pathspec + its signature.
- On **`status=REFUSED_CONTRACT` / `verdict=FAIL`**: a contract failed and **no
  Corpus was written**. Read the named diff in `diagnostics` — surface the
  `contract`, the `message`, and the `fix` plainly, and **stop**. Do not work
  around a contract refusal; it is the engine protecting you from a silently-wrong
  corpus.

## 5. Deliver — return the Corpus
- Give the **`Corpus/<n>`** pathspec and the derived vocab/chunk in plain words.
- **Next step:** offer `/loom-baseline --in <IngestDataset/n>` to set the bar the
  model must beat. (Pretraining on the Corpus is a later version.)

## Composition / exit gate
Produces a `Corpus/<n>` (with its tokenizer signature) on PASS; on a contract
violation it writes nothing and returns the named diff — the chain stops there.
