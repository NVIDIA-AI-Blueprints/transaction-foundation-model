# Loom engineering notes

## The reference bug Loom fixes (C1 injectivity)

The reference tokenizer (`src/tokenizer/fixed_vocab.py`) keyed `idx_to_token` by
the **raw value** (`range(min_val, max_val + 1)`). For a field with
`min_val = 1` (MONTH), that assigned local ids `1..12` while the running offset
only advanced by the *count* (12). The result: MONTH's id block overlapped the
next field (CARD) by one — **MONTH_12 and CARD_0 both resolved to id 2179**, and
id 2167 was left dead. Training would run and produce a garbage-but-plausible
loss — the silent-failure class.

**Loom's fix:** every step assigns **0-based local indices** within its block
(`id = offset + local_index`, `local_index ∈ 0..count-1`), and the offset
advances by exactly `count`. Blocks are therefore contiguous and disjoint, and
the vocab is dense `0..vocab_size-1`. The **C1 check** asserts this (blocks
disjoint, total to `vocab_size`, every `(step, value)` → unique id) and
**refuses to write a Corpus on any collision**, surfacing a named diff (not a
stack trace). A test constructs a deliberately-overlapping spec and asserts C1
FAILs with the named diagnostic (the `--reorder-step card:first` case, §7.2a).

## Conformance oracle vs product

The reference `src/tokenizer/*.py` is GPU-only (cuDF/cupy). Loom reimplements the
tokenizer **clean on CPU (pandas/numpy)** and uses the reference only as a
correctness oracle:

- **Conformance is on:** `vocab_size` (financial preset = **6251**, +32 → 6283
  with time-delta), the contiguous/dense/injective vocab layout, the corpus
  grammar (`<bos> txn (<sep> txn)* <eos>`, `chunk_size = context_len //
  (tokens_per_txn + 1)`), and the per-field token strings.
- **Conformance is NOT on:** the merchant-bucket hash. The reference uses cuDF's
  GPU `hash_values()`; Loom uses a stable hashlib-based hash (`% buckets`) that is
  reproducible across runs/processes but **will differ** from the cuDF bucketing.
  This is fine — Loom is the product; merchant-bucket identity is not a
  correctness contract.

## Determinism (C2)

The default path is config-only (no fitted artifacts). `amount_strategy =
quantile | kmeans` is a **fitted artifact** — C2 flags it, the binner state is
persisted into the Corpus (`fitted_state`), and it is not allowed silently on the
default path.

## Reconciliation (2026-06-15)

Loom's clean-compile `tokenize` **replaces** hand-writing
`src/tokenizer/chain_pipeline.py` for Phase 1 (confirmed direction). The repo's
`src/` stays the notebook/CI path and the conformance oracle.

## TODO markers (where the rest of the design attaches)

- `loom/store.py` — swap the local store for the Metaflow metadata/datastore
  adapter in v0.2 (the `put`/`get`/`new_ref` seam is stable).
- `loom/experiment.py` — `.loom` campaign parser + `REFUSED_NO_METRIC` enforced
  for launch verbs (not for tokenize/ingest).
- `loom/tools.py` — `make_confirm_token`/`validate_confirm_token` for gated
  launch verbs (§5.3); none gated in this slice.
- `pretrain`/`embed`/`evaluate`/`report` verbs, the cost-PLAN derivation
  (`CostPlan` already carries the fields), the binding envelope, and the live
  `loom top` TUI are out of this slice.
