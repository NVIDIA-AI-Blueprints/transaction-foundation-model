# 03-sequence-embeddings -- build an embedding backbone from event sequences (CPU)

## Use case

This walks through Loom's **model-builder seam**: turning raw per-account *event
sequences* into a learned **embedding backbone**, on CPU, with no GPU and no
model key. It also shows the production GPU path **gating cleanly** when there is
no GPU target -- the safe-by-default heavy-launch posture.

The synthetic data (`make_data.py`) is a domain-neutral, per-account
event-sequence fixture with a **planted next-event signal**. It is deterministic,
seeded, and generated offline. The schema:

| column | meaning |
| --- | --- |
| `account` | the grouping key (`acct-NNN`); rows are grouped into per-account sequences |
| `t` | the within-account ordering step (0, 1, 2, ...) |
| `event` | an abstract categorical event in `{A, B, C, D, E}` -- **no domain semantics**, just opaque tokens whose *transition structure* carries the signal |
| `amount` | a generic numeric field (bucketized into the shared vocab) |
| `label` | the per-account binary target (`1` = follows the chain, `0` = random) |

The planted structure: **positive accounts (`label == 1`) follow a first-order
Markov chain** `A->B->C->D->E->A` with 85% probability at each step; **negative
accounts emit i.i.d. random events**. So adjacent-event co-occurrence (the
`next-event` objective) genuinely separates the classes -- which is exactly what
the pooled PPMI+SVD embedding backbone learns, and why those embeddings beat a
raw per-row baseline (the lift the conformance suite proves).

## Ask Loom

The natural-language line you would type at the `loom` agent -- the product UX:

```
loom "build an embedding backbone over these account event sequences (next-event), then finetune a head and report the lift over the raw baseline"
```

> Needs a model key. The agent plans, then runs the verbs in "Step by step"
> under the hood. The **finetune + lift report** and any `loom run` / `loom
> optimize` search over tokenization / heads / embedding dims are the key-gated
> steps -- they appear here in prose only and are **not** in the asserted
> `run.sh`. The CPU `train` build itself is keyless and *is* asserted below.

## Step by step

The explicit, **keyless** verb sequence the agent runs under the hood. Each line
is one `$LOOM <verb> ... --json` with a one-line explanation and the expected
outcome (the VERDICT / summary field `run.sh` asserts on).

1. **`$LOOM ingest --source <dir> --name <unique> --json`** -- the one
   external->Metaflow boundary; the event-sequence CSV becomes a versioned data
   object. Expected: `status == "ok"`, a `pathspec` (the `dataset_ref`
   everything downstream consumes).

2. **`LOOM_MODEL_BUILDER_PROVIDER=local $LOOM train --dataset <pathspec>
   --objective next-event --budget probe --json`** -- build the embedding
   backbone on CPU with the **torch-free `local` adapter** (PPMI + TruncatedSVD,
   sub-2s, zero GPU). `next-event` learns the planted adjacent-pair (first-order
   Markov) co-occurrence; `probe` is the cheapest embedding dimensionality.
   Expected: `status == "ok"`, **`VERDICT == "BUILT"`** (the CPU adapter actually
   builds), `summary.fingerprint` is a deterministic `sha256:...` content hash,
   and `pathspec` is the produced **backbone run** (a first-class `dataset_ref` a
   downstream `embed`/`finetune` loads as a frozen backbone).

3. **`$LOOM train --dataset <pathspec> --objective next-event --budget probe
   --json`** -- the **production path**. The default model-builder provider is
   `nemo` (the GPU lowering compiler), and there is no configured `gpu_target`,
   so the heavy launch is **refused up front** rather than run -- the
   safe-by-default posture (mirrors `deploy --apply` being off by default).
   Expected: `status == "ok"` (a clean refusal is a success, not a crash),
   **`VERDICT == "REFUSED_NO_GPU_TARGET"`**, no backbone artifact
   (`summary.artifact_pathspec` is empty), and `summary.launch == False` (the
   real GPU launch is off behind `--launch`).

_(Optional aside: in a real session a key-gated `loom optimize` would
tree-search the tokenization / head / embedding-dim recipe, and a `finetune` +
`evaluate` would fit a cheap head on the frozen backbone and report the
embeddings-vs-raw **lift**. Those are omitted from the asserted recipe because
the search/agentic flow needs a model key; the CPU backbone build above does
not.)_

## What this proves

The model-builder seam is exercised end to end **without a GPU or a model key**,
and the heavy-launch gate is honored:

* The **`local` (CPU) adapter actually builds** a backbone from event sequences
  -- `train` returns `VERDICT == "BUILT"` with a deterministic `sha256:`
  fingerprint and a real produced run `pathspec`. The build is reproducible (a
  seeded fixture + a pinned SVD seed => a stable fingerprint), so a drift in the
  backbone is a drift in the fingerprint.
* The **production GPU path (`nemo`) refuses cleanly** when there is no GPU
  target -- `VERDICT == "REFUSED_NO_GPU_TARGET"`, no artifact, `launch == False`.
  The expensive launch is **off by default**; it never silently consumes GPU.

This is the contract `tests/test_examples.py` guards by replaying `run.sh`: the
CPU backbone build stays green and the GPU launch stays gated.
