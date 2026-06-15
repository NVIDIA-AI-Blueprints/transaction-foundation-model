# 05 -- Building a model backbone with no GPU

> A beginner-friendly tutorial for Loom's **model-builder** seam (`train`). You
> will build a real embedding **backbone** from per-account event sequences --
> on your laptop's CPU, with **no GPU and no model key** -- and see the
> production GPU path refuse cleanly until you point it at a cluster.

## What you'll learn

- What a "backbone" is and why building one is a distinct step in the data
  science lifecycle.
- How to drive Loom's `train` verb **keyless** with the torch-free CPU adapter.
- How to read the `train` result: the `BUILT` status, the deterministic
  fingerprint, and the backbone pathspec you hand to downstream verbs.
- Why the expensive GPU launch is **off by default** and how it gates safely.

No prior Loom experience is needed. If you have run tutorial `01` (the core
ingest -> profile -> validate lifecycle), you already know the shape; this one
adds the model-builder step.

## The use case: learning structure from event sequences

Imagine you have a stream of **per-account events** -- a customer's clicks, an
account's transactions, a device's telemetry. Each account is an ordered
*sequence* of events over time. The interesting signal is often not in any
single row but in the **order**: which event tends to follow which.

A **backbone** is a learned, reusable representation of that structure -- an
embedding model that turns "what happened, in what order" into a dense vector
per account. Once you have a backbone, you can freeze it and snap a cheap task
head onto it (fraud, churn, propensity, ...) instead of re-learning the
sequence structure from scratch every time. Building the backbone *once* is the
expensive, foundational step; reusing it is cheap.

The lesson here: **you do not need a GPU to build and test a backbone end to
end.** Loom's `train` seam has a torch-free CPU stand-in (a pooled
PPMI + Truncated SVD embedding) that builds a *real* backbone deterministically
in under two seconds. That lets you develop, test, and wire up the whole
pipeline keyless and free, then flip to the GPU pretrain only when you're ready
to spend.

## The data

The tutorial generates its own data inline -- no downloads, fully deterministic,
domain-neutral. It is a per-account event-sequence table:

| column    | meaning |
| --------- | --- |
| `account` | the grouping key (`acct-NNN`); rows are grouped into per-account sequences |
| `t`       | the within-account ordering step (`0, 1, 2, ...`) |
| `event`   | an abstract categorical event in `{A, B, C, D, E}` -- **no domain semantics**, just opaque tokens whose *transition structure* carries the signal |
| `amount`  | a generic numeric field |
| `label`   | the per-account binary target (`1` = follows the chain, `0` = random) |

**The planted signal.** Positive accounts (`label == 1`) follow a first-order
Markov chain `A -> B -> C -> D -> E -> A` with 85% probability at each step.
Negative accounts (`label == 0`) emit purely random events. So the *adjacent
event co-occurrence* genuinely separates the two classes -- and that adjacency
structure is exactly what a `next-event` backbone learns to encode. A single
seeded NumPy RNG drives every draw, so a repeat run is byte-identical.

This is the right shape to drive the model-builder: there is real sequential
structure to capture, and it's reproducible so the backbone's fingerprint is
stable run to run.

## Run it

The script is self-contained. If your local Metaflow + minio datastore env file
exists at `/tmp/loom-cluster-env.sh`, the script sources it for you, so a plain:

```bash
cd /Users/anub/Work/Loom
bash tutorials/05-build-a-backbone/run.sh
```

is enough. It generates the data, ingests it under a **unique** dataset name
(so repeat and concurrent runs never collide), runs the keyless verb sequence
with `--json`, asserts each outcome, prints a clear `PASS`/`FAIL` line, and
**exits nonzero on any regression**.

## Step by step

Each step is one `loom <verb> ... --json` call. The script captures the JSON and
asserts the stable fields; here is what each step does and what to look for.

### 1. Ingest the sequences

```bash
loom ingest --source <dir> --name <unique-name> --json
```

This is the one boundary where external data crosses into Loom. The CSV becomes
a **versioned Metaflow data object**. Look for:

- `status == "ok"`
- a `pathspec` like `IngestDataset/1781045646150800` -- this is the
  `dataset_ref` every downstream verb consumes. The script saves it as
  `$DATASET`.

### 2. Build the backbone on CPU (the keyless model-builder)

```bash
LOOM_MODEL_BUILDER_PROVIDER=local \
  loom train --dataset <dataset_ref> --objective next-event --budget probe --json
```

This is the heart of the tutorial. Setting
`LOOM_MODEL_BUILDER_PROVIDER=local` selects the **torch-free CPU adapter**
(PPMI + Truncated SVD): no GPU, no model key, sub-2s, deterministic.

- `--objective next-event` tells it to learn the planted adjacent-pair
  (first-order Markov) co-occurrence -- the structure that separates the classes.
- `--budget probe` is the cheapest, fastest setting (the smallest embedding
  dimensionality). The budget knob is where the cost "physics" lives: `probe`
  costs zero GPU-hours on the CPU adapter; `small` / `full` cost more on the GPU
  path.
- No `--launch` flag, so the heavy GPU launch stays off -- this run consumes
  **zero GPU-hours**.

What to look for in the result:

- `status == "ok"` and `VERDICT == "BUILT"` -- the CPU adapter *actually built*
  a backbone (it isn't a dry run).
- `summary.status == "BUILT"`, `summary.objective == "next-event"`,
  `summary.budget == "probe"` -- the build did what you asked.
- `summary.fingerprint` is a `sha256:...` content hash -- the backbone is
  **deterministic**, so a change in the build shows up as a changed fingerprint.
- `pathspec` like `TrainFlow/1781045649586074` -- the produced **backbone run**.
  This is a first-class `dataset_ref`: a downstream `embed` / `finetune` loads it
  as a frozen backbone via `--backbone <pathspec>`.

The script saves this as `$BACKBONE`.

### 3. See the GPU production path gate cleanly

```bash
loom train --dataset <dataset_ref> --objective next-event --budget probe --json
```

Same command, but **without** `LOOM_MODEL_BUILDER_PROVIDER=local`. The default
provider is `nemo`, the GPU lowering compiler. Because no `LOOM_GPU_TARGET` is
configured, `train` **refuses up front** rather than launching anything -- the
safe-by-default heavy-launch posture (the same idea as `deploy --apply` being
off by default).

What to look for:

- `status == "ok"` -- a clean refusal is a **success**, not a crash. The run
  completes; it just declines to spend GPU.
- `VERDICT == "REFUSED_NO_GPU_TARGET"` and
  `summary.status == "REFUSED_NO_GPU_TARGET"`.
- `summary.artifact_pathspec` is empty -- nothing was built, because nothing
  launched.
- `summary.launch == False` -- the expensive launch stays behind `--launch`.

This is the guardrail: Loom never silently consumes GPU.

## What to expect

A successful run ends with:

```
== PASS: 05-build-a-backbone -- backbone built keyless, GPU gate honored
```

and exit code `0`. Every assertion line prints `ok: ...` as it passes. If any
verb regresses (a changed `--json` shape, a non-`BUILT` status, a missing
pathspec), the matching assertion fails, prints the offending JSON, and the
script exits nonzero.

## Next step: the real GPU pretrain (needs a key / a GPU target)

This tutorial stays entirely keyless. To run the **real** GPU pretrain instead
of the CPU stand-in, you would:

1. Point Loom at a GPU cluster by setting `LOOM_GPU_TARGET`.
2. Add the `--launch` flag to actually perform the heavy launch:

   ```bash
   LOOM_GPU_TARGET=<your-cluster> \
     loom train --dataset <dataset_ref> --objective next-event --budget full --launch --json
   ```

That path is **deferred** here on purpose -- it consumes GPU-hours (and, in a
real agentic session, the surrounding search would use a model key). The CPU
backbone you built above proves the whole seam end to end before you spend a
cent.

### "Ask Loom" (the natural-language UX -- needs a model key)

In the full product you would just type the goal at the `loom` agent:

```
loom "build an embedding backbone over these account event sequences (next-event), then finetune a head and report the lift over the raw baseline"
```

The agent plans and runs the verbs above under the hood. The **finetune + lift
report** and any `loom optimize` search over tokenization / heads / embedding
dimensions are the key-gated steps -- they need a model key and are *not* part
of this keyless tutorial. The CPU backbone build itself is keyless and is what
you ran above.

## Why this matters

- **You can build and test a backbone with zero GPU and zero spend.** The CPU
  adapter builds a real, deterministic backbone, so you can develop and
  regression-test the entire model-builder pipeline for free.
- **The expensive path is safe by default.** The GPU pretrain never launches
  without an explicit target *and* `--launch`; a missing target is a clean
  refusal, not a surprise bill.
- **The backbone is a first-class artifact.** Its pathspec is a `dataset_ref`
  and its fingerprint is a content hash, so downstream verbs can consume it and
  any drift in the build is visible as a changed fingerprint.
