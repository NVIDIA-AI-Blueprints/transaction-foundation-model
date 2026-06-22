# Loom

**An agent harness for building state-of-the-art foundation models** — starting
with the part you have to get right first: the **tokenizer**. Loom is a small set
of sharp, typed verbs you *compile before you spend*, driven identically by a
human at a terminal (`loom <verb> …`) and a Claude/Codex agent (`loom.<verb>(…)`),
where contract violations surface as named diffs caught for free.

Loom is **domain-agnostic**. It reasons about *column shapes* — categorical,
continuous, timestamp, high-cardinality, identifier, sequence-over-an-alphabet —
not about any one dataset. The same verbs build a tokenizer for a payments model,
a music model, or a genomics model. A finance preset ships in the box as one
example, not as the point.

> **Start here: [`TOKENIZATION.md`](./TOKENIZATION.md)** — the user manual for
> building a custom, contract-checked tokenizer for any foundation model, with
> worked, reproducible examples ([`examples/`](./examples/)) and a demo script.

## The verbs

Tokenizer **design** (local, CPU, instant, $0):

- **`ingest`** — register a raw file as a versioned, content-addressed dataset (schema sniff + identity/leakage scan).
- **`propose`** — analyze the columns and emit an *editable* tokenizer spec (which column → which strategy, and **why**).
- **`tokenize`** — compile a spec into a **Corpus** (vocabulary + grammar), checking contracts **C1** (injective + dense vocab), **C2** (determinism / no fitted artifact), **C3** (grammar / chunk derivation), and refusing — with a named diff, not a stack trace — if any fails.

Tokenizer **validation** (local, CPU, ~$0 — *before* a GPU-hour):

- **`baseline`** — the controls a model must beat (popularity, repeat-last-item) on a temporal hold-out.
- **`embed`** — fit a quick PPMI-SVD embedding of a Corpus (deterministic, torch-free).
- **`evaluate`** — score the tokenizer: does the vocab fit the data, and does the embedding beat the baselines? A `PASS`/`REVIEW`/`FAIL` verdict with a refine plan.

**Training** (gated):

- **`pretrain`** — plan + (human-confirmed) launch a model-builder run (local CPU rehearsal, or NeMo on a GPU target) over a Corpus → a portable Checkpoint. `prepare` is the generic representation→Corpus path `tokenize` is pinned to.

## The design principle

**Design local, train at scale.** The tokenizer is deterministic and config-only,
so a vocabulary you design and validate on a laptop **sample** is *bit-identical*
on the full cloud-scale corpus. You design and prove it here; you train it later,
elsewhere, unchanged.

**Dual driver.** Each verb is declared **once** as a typed contract
(`@register(...)` in `loom/verbs/<verb>.py`): an argument JSON-Schema + tier +
capability mode + an implementation `fn(args, ctx) -> VerbResult`. From that single
declaration Loom generates the human CLI subcommand **and** the agent tool schema;
the `--json` result envelope is byte-identical to the agent tool result.

## Install & run

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
loom --help
pytest -q
```

## Docs

- [`TOKENIZATION.md`](./TOKENIZATION.md) — the tokenizer user manual + demo (start here).
- [`DESIGN.md`](./DESIGN.md) — the product + UX spec.
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — how the harness is built (ports/adapters, the contract core).
- [`GPU-RUNBOOK.md`](./GPU-RUNBOOK.md) — running a real training job on a GPU target.
- [`NOTES.md`](./NOTES.md) — engineering notes.
