---
name: loom-train
description: Build the model the lifecycle needs through Loom — EXPENSIVE / MUTATE (escalates to irreversible/external at the real GPU launch). Stated in DS-intent terms (objective / budget / backbone / metric); the adapter hides ALL backend vocabulary — you never name NeMo, Megatron, a GPU-count, or a checkpoint file. Use when the user says "pretrain a backbone", "train an embedding model", "build the foundation model", "fine-tune a head on the backbone". pretrain is launch-and-track — AIDE NEVER tree-searches it (use loom-optimize for cheap scalars). The real heavy GPU launch is OFF by default (--launch); with no GPU target it refuses cleanly. NEVER auto-fire — the model proposes; only the user fires.
when_to_use: "pretrain a sequence backbone, embed via a frozen backbone, fine-tune a cheap head, build the model the lifecycle needs"
when_not_to_use: "to tree-search a cheap scalar (a head / tokenization / data-prep) — use loom-optimize; to ship a validated model — use loom-deploy; to validate held-out quality — use loom-validate; to engineer features — use loom-features."
argument-hint: "<a dataset_ref pathspec (+ objective, budget, optional backbone); --launch only to really launch GPU work>"
disable-model-invocation: true
---

# loom-train

Build the model the downstream lifecycle needs — the **expensive / mutate** verb (it
escalates to **irreversible / external** at the real GPU launch). This is a **planned,
always-gated run through Loom's MLOps interface**, never a loose training script,
because Loom draws the abstraction boundary in **DS-intent vocabulary**: you state an
`objective` (`next-event` / `masked-field` / `contrastive`), a `budget`
(`probe` / `small` / `full`), an optional frozen `backbone`, and the `metric` — and
the model-builder adapter is a *compiler* that lowers that intent into a backend
config. **Every backend noun lives inside the adapter and nowhere else** — you never
name NeMo, Megatron, a GPU-count, or a checkpoint file. The abstraction hides
**vocabulary, not physics**: `budget="full"` still costs real GPU-hours, surfaced at
the gate. **The metric is the spec.** Stay domain-neutral — never assume a task type,
column meaning, or vertical.

Two backends sit behind the seam, swapped purely by config (`model_builder_provider`):
the default **`nemo`** lowering compiler (which *plans* the real GPU launch in v0.1 and
gates it behind `--launch`, refusing cleanly with no GPU target), and a torch-free CPU
**`local`** PPMI+TruncatedSVD stand-in (the testable default that actually builds a
backbone and embeddings end-to-end on one machine). Swapping `local → nemo` changes
only weight quality — never the seam, gate, mode-typing, or lineage.

## When to use

- The user says "**pretrain a backbone**", "**train an embedding model**", "**build the
  foundation model**", or "**fine-tune a head on the backbone**" against a `dataset_ref`.
- They want the **model the lifecycle needs** (a backbone / embeddings) built as a
  versioned, lineage-grounded Metaflow run — not a notebook training cell.

## When NOT to use

- To **tree-search a cheap scalar** (a head / tokenization / data-prep probe) — use
  **`loom-optimize`** (AIDE). `pretrain` is `launch-and-track`; AIDE must never
  tree-search the expensive backbone. `loom-train` refuses a `searchable` capability
  and points you here.
- To **ship a validated model** — use **`loom-deploy`** (it gates on a `loom-validate`
  `VERDICT==PASS`).
- To **validate held-out quality** before promotion — use **`loom-validate`**.
- To **engineer features** into a new data object — use **`loom-features`**.

## 1. Intake — pin the spec (refuse if it is not measurable)

Pin the inputs in the user's own terms and write them back for confirmation:

- **Data** — the `dataset_ref` (a Metaflow **pathspec**, e.g. `IngestDataset/123`,
  produced by `loom ingest`). Take a pathspec, never a raw S3 URI or a loose local
  file as the source of truth.
- **Objective** — one of `next-event` / `masked-field` / `contrastive` (DS-intent;
  never a backend recipe name). For `finetune` / `embed`, a frozen **`backbone`**
  pathspec.
- **Budget** — one of `probe` / `small` / `full`. This is **physics**: a bigger budget
  costs more GPU-hours, surfaced at the gate (never faked away).
- **Metric** — one natural-language/standard metric the build is steered toward (e.g.
  `fraud-pr-auc`), stated so the direction is unambiguous.

**Refuse to start without a measurable `(dataset_ref, objective, metric)` spec** — a
build with no objective or metric has nothing to train toward, exactly as
`loom-validate` refuses a missing target. If any required piece is missing or vague,
**ask — do not guess** (a wrong objective silently trains the wrong thing).

## 2. Plan — show the plan + tier (EXPENSIVE/MUTATE, ALWAYS gate)

`loom-train` is the **expensive / mutate** tier of the approval matrix (see
`CONVENTIONS.md`) and **escalates to irreversible/external** at the real GPU launch, so
it **always gates** and is **never model-auto-invoked** (`disable-model-invocation:
true` — the model proposes, only the user fires). Propose the plan and **stop at the
gate**:

- **What will run** — the capability (`pretrain` by default), the objective, the
  budget, and that the backbone is built as ONE Metaflow run (no unbounded loop).
- **Cost shape (physics at the gate)** — show the **cost PLAN**: hours / **$** /
  **GPU-count** for the chosen budget (e.g. `budget="full"` ⇒ 8 GPU × 12 h ≈ 96
  GPU-hours ≈ $288). On a remote GPU profile this consumes the user's own compute.
- **Launch posture** — state plainly that the **real GPU launch is OFF** (`--launch`
  off by default): the default run produces a **PLAN** (the `nemo` adapter stages it;
  the `local` CPU stand-in actually builds), and that **with no GPU target configured
  it refuses cleanly** (`REFUSED_NO_GPU_TARGET`) without launching anything.
- **Mode** — `pretrain` is `launch-and-track` (AIDE never tree-searches it); a
  `searchable` capability is redirected to `loom-optimize`.

**Do not pass `--launch` until the user explicitly confirms after seeing the plan and
the cost.** Re-plan and re-present the gate if the user adjusts the budget / objective
/ metric.

## 3. Run — call Loom's MLOps INTERFACE (the `loom` CLI), never the backend

Speak **only Loom's interface** — shell out to the `loom` CLI, which resolves the MLOps
provider by name (default **Metaflow**, swappable by config) and runs the train flow
through the interface's `run_flow` seam. The model-builder backend (`local` / `nemo`)
is resolved **inside the flow** from config. **Never call Metaflow / NeMo / AIDE
directly, never touch raw S3**, and **never name a backend noun** (Megatron, a
GPU-count, a `.nemo` file) — the datastore and the backend recipe are the interface's
opaque concern.

```bash
loom train --dataset <PATHSPEC> --objective <next-event|masked-field|contrastive> \
           --budget <probe|small|full> [--backbone <PATHSPEC>] [--metric fraud-pr-auc] \
           [--capability pretrain|tokenize|finetune|embed]            # default: PLAN / CPU build, NO GPU launch
loom train --dataset <PATHSPEC> --objective next-event --budget full --launch   # real GPU launch — ONLY after the user confirms
```

- The work executes as a **Metaflow run**; inputs are read via the Client API. With
  `--launch` OFF the run PLANs (`nemo`) or builds on CPU (`local`); `--launch` performs
  the real launch **only when a `gpu_target` is configured** — otherwise it refuses.
- Lifecycle flows need the **metaflow** MLOps provider — the `local` dev provider
  cannot run them (it says so, pointing at `--mlops metaflow`).
- The GPU target + cluster credentials come from the **environment** only
  (`LOOM_GPU_TARGET` / the matching `METAFLOW_*` creds) — never on the command line,
  never in the plan or transcript.

## 4. Verify — assert lineage; spill large output to an Artifact

- Confirm the run reported success and read the **STATUS** the model-builder seam
  produced: `BUILT` (the `local` adapter produced a backbone/embeddings),
  `PLANNED` (a staged `nemo` plan, no mutation), or `REFUSED_NO_GPU_TARGET` (the gate
  refused — no launch). For `pretrain`, assert the mode was `launch-and-track` (AIDE
  never searched it).
- **Assert lineage:** the produced `ArtifactRef` carries a valid **`<FlowName>/<run_id>`
  pathspec** (the backbone / embeddings ref) plus a **data fingerprint** — never a
  `.nemo`/`.pt` file, never an S3 URI. The large weight matrix stays a **Metaflow
  Artifact** referenced by pathspec, never inlined; cap any inline output at ~25k
  tokens.

## 5. Deliver — narrate the @card, return run + summary + STATUS, append a learnings row

- **Narrate the `@card`:** lead with the **STATUS** (`BUILT` / `PLANNED` /
  `REFUSED_NO_GPU_TARGET`) and the **cost PLAN** (the physics the gate showed); then the
  produced ref (the backbone / embeddings pathspec + fingerprint) and the lineage. Make
  crystal clear whether this was a **CPU build / staged PLAN** (the default) or a **real
  GPU launch**.
- **Hand back the mandated artifact:** the versioned **Metaflow run + `@card`** plus the
  typed JSON summary the CLI prints, with the headline `STATUS`/`VERDICT` a downstream
  verb can consume (the embeddings pathspec is a first-class `dataset_ref` for
  `loom-validate` / `loom-optimize`).
- **Learnings:** the run appends one `command="train"` row to the flywheel corpus
  (`learnings/rollouts.jsonl`) — `dataset_ref` · capability · objective · budget ·
  resolved backend · gate STATUS · cost (GPU-hours / $) · produced pathspec ·
  fingerprint — sanitized, no raw rows, no secrets, no backend noun beyond the resolved
  backend name. The CLI does this every run; do not hand-write the row.
- **Next step:** if `BUILT`, offer `loom-validate --dataset <embeddings_ref>` (or
  `loom-optimize` to tree-search a cheap head on the frozen backbone); if `PLANNED` /
  `REFUSED_NO_GPU_TARGET`, point at the missing `gpu_target` (set `LOOM_GPU_TARGET`, or
  use the `local` CPU stand-in) and the explicit `--launch` once a target is configured.

## Composition — machine-checkable exit gate (executable self-test)

- **Consumes:** a `dataset_ref` pathspec (the sequences data object), and for
  `finetune` / `embed` a frozen `--backbone` pathspec from a prior `pretrain` run.
- **Produces:** a backbone / embeddings `ArtifactRef` whose pathspec is a first-class
  `dataset_ref` the downstream verbs consume (a `loom-validate` / `loom-optimize`
  `--dataset`), plus the typed `STATUS`/`VERDICT` line.
- **Exit gate:** the produced `ArtifactRef` has a **valid pathspec with a measurable
  gain** (the `local` adapter) **OR** a clean `STATUS == PLANNED` / `REFUSED_NO_GPU_TARGET`
  (the `nemo` adapter, no GPU) — and **`pretrain` is mode `launch-and-track`** so AIDE
  never tree-searched the backbone. The launch gate is plain Python (`f(gpu_target is
  None, launch)`), never a prompt the model can talk past: it **fails closed** — no GPU
  target ⇒ no launch.
- **Self-test (ships with the verb):** two executable self-tests assert the gate —
  `tests/test_train.py::test_train_local_roundtrip_gives_lift` (the torch-free `local`
  adapter builds a backbone whose embeddings **beat the raw baseline** on a planted
  fixture — a valid pathspec + a real positive lift) and
  `tests/test_train.py::test_train_nemo_refuses_without_gpu_target` (with
  `gpu_target=None` the `nemo` `pretrain` returns a clean `REFUSED_NO_GPU_TARGET`
  `ArtifactRef`, `pathspec is None`, and **never launches** — even with `--launch`). The
  full `ModelBuilderProvider` contract is pinned by
  `tests/test_model_builder_conformance.py` (the 8-test golden suite, parametrized over
  every backend), including mode test 5 (`pretrain` is `launch-and-track`) and the
  source-scan test 8 (no backend noun leaks the seam). "Guards failing open" is exactly
  the failure mode these guard against.

---

## Acceptance test (the bar before this verb joins the pack)

1. **Speaks only the interface** — shells out to `loom train` (the MLOps interface,
   provider-by-name), never Metaflow/NeMo/AIDE directly, never raw S3, and never names a
   backend noun; the model-builder backend is resolved by config inside the flow.
2. **Output is a versioned run + `@card`** — a backbone / embeddings build (or a staged
   PLAN), not a chat transcript or a loose training script; the weights stay a Metaflow
   Artifact referenced by pathspec.
3. **Approval tier is correct** — EXPENSIVE/MUTATE, **ALWAYS gates**, escalates to
   irreversible/external at the real launch; the heavy GPU launch is behind `--launch`
   and **OFF by default**, refuses cleanly with no `gpu_target`, and the skill sets
   `disable-model-invocation: true` so the model never auto-fires it.
4. **Writes a learnings row** — the run appends a sanitized `command="train"` row to
   `learnings/rollouts.jsonl` (the CLI does this every run).
5. **Exit gate has a self-test** — `tests/test_train.py::test_train_local_roundtrip_gives_lift`
   (valid pathspec + real lift) and `::test_train_nemo_refuses_without_gpu_target`
   (clean `REFUSED_NO_GPU_TARGET`, no launch), backed by the golden conformance suite.
6. **Single free-text arg** — one `dataset_ref` pathspec (plus the DS-intent
   `--objective` / `--budget` / `--backbone` / `--metric`), and the explicit `--launch`
   safety flag.
7. **Dual-invocation** — user-typed only by design (`/loom-train`); never
   model-auto-loaded (`disable-model-invocation: true`) because the action is
   expensive and escalates to irreversible/external.
