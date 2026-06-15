# Loom

**An agent harness for training SOTA foundation models** — a small set of sharp,
typed verbs you *compile before you spend*, driven identically by a human at a
terminal (`loom <verb> …`) and a Claude/Codex agent (`loom.<verb>(…)`), where
contract violations surface as named diffs caught for free.

See [`DESIGN.md`](./DESIGN.md) for the authoritative product + UX spec. This
package builds to it.

## Phase-0 slice (this build — ZERO GPU)

The first three verbs' foundation plus the typed-contract core:

- **`tokenize`** — THE contract compiler: a declarative tokenizer field-spec
  compiled to a vocabulary + derived (`vocab_size`, `vocab_hash`,
  `tokens_per_txn`, `chunk_size`), with contract checks C1 (injective + dense
  vocab), C2 (determinism / fitted-artifact), C3 (grammar / chunk derivation), on
  a pandas/CPU backend (no cuDF/GPU).
- **`ingest`** — register a dataset as a versioned, content-addressed data-object
  with a schema sniff + EDA leakage gate + provenance; idempotent.
- **`baseline`** — popularity + repeat-last-item baselines (the control a model
  must beat).

Plus: the typed-contract narrow waist (`loom/registry.py`), the dual-driver
result envelope (`loom/types.py`), a local content-addressed object store
(`loom/store.py`, NO Metaflow), the engine API contract (`loom/engine/api.py`),
`--experiment` threading, and the CLI + agent-tool faces.

**Out of this slice:** `pretrain`/`embed`/`evaluate`/`report`, real GPU, NeMo,
Metaflow, AIDE `search`, the live TUI. TODO markers in the code show where they
attach.

## Install & run

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
loom --help
pytest -q
```

## Architecture — the narrow waist

Each verb is declared **once** as a typed contract (`@register(...)` in
`loom/verbs/<verb>.py`): an argument JSON-Schema + tier + capability mode + an
implementation `fn(args, ctx) -> VerbResult`. From that single declaration Loom
generates (a) the human CLI subcommand and (b) the agent tool schema. The
`--json` result envelope is byte-identical to the agent tool result
(`VerbResult.to_json()`).

## Reconciliation note (2026-06-15)

Loom's clean-compile `tokenize` **replaces hand-writing
`src/tokenizer/chain_pipeline.py`** for Phase 1 (confirmed direction
2026-06-15). The reference repo's `src/` remains the notebook/CI path **and the
conformance oracle** — Loom reimplements the tokenizer clean on CPU and uses
`src/tokenizer/*.py` only to check correctness (vocab size, grammar,
injectivity). Loom deliberately does **not** reproduce the reference's cuDF
merchant-bucket hash (its salted/GPU hash differs); conformance is on
vocab/grammar/injectivity, not merchant-bucket identity. See
[`NOTES.md`](./NOTES.md) for the MONTH_12/CARD_0 bug Loom fixes.
