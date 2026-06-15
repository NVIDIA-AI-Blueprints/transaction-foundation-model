---
description: Build a model through the model-builder seam (gated) -> a run + @card. EXPENSIVE/MUTATE, escalates to irreversible at the real GPU launch (OFF unless --launch); never auto-fired.
argument-hint: <a dataset_ref pathspec (+ --objective, --budget, optional --backbone, --metric); --launch only to really launch GPU work>
---

# /loom-train — build a model (EXPENSIVE/MUTATE — always gate)

Build the model the downstream lifecycle needs. Loom draws the boundary in
**DS-intent vocabulary**: state an `objective` (`next-event`/`masked-field`/
`contrastive`), a `budget` (`probe`/`small`/`full`), an optional frozen `backbone`,
and the `metric` — the model-builder adapter compiles that intent into a backend
config. You never name a backend (NeMo, Megatron, a GPU-count, a checkpoint). The
abstraction hides vocabulary, not physics: `budget=full` still costs real
GPU-hours, surfaced at the gate. The metric is the spec.

Train: $@

## 1. Intake — pin the spec (refuse if not measurable)
- **Data** — a `dataset_ref` pathspec. Never a raw S3 URI / loose file.
- **Objective** — `next-event` / `masked-field` / `contrastive` (DS-intent, never a
  backend recipe). For `finetune` / `embed`, a frozen `backbone` pathspec.
- **Budget** — `probe` / `small` / `full` (this is physics — surfaced at the gate).
- **Metric** — one standard/natural metric the build is steered toward.

**Refuse without a measurable `(dataset, objective, metric)` spec.** If anything is
vague, **ask — do not guess**.

## 2. Plan — EXPENSIVE/MUTATE, always gate, never auto-fire
The expensive/mutate tier that **escalates to irreversible/external** at the real
GPU launch, so it **always gates** and is **never model-auto-invoked**. Propose the
plan and **stop**:
- **What runs** — the capability (`pretrain` default), objective, budget, as ONE
  Metaflow run (no unbounded loop).
- **Cost shape (physics)** — hours / $ / GPU-count for the chosen budget.
- **Launch posture** — the **real GPU launch is OFF** (`launch` off by default): the
  default PLANs (`nemo`) or builds on CPU (`local`); with no GPU target it refuses
  cleanly (`REFUSED_NO_GPU_TARGET`) without launching anything.

Do not set `launch` until the user explicitly confirms after seeing the plan + cost.

## 3. Run — call the `loom_train` tool
Call `loom_train` with `dataset`, `objective`, `budget` (+ `backbone`, `metric`,
`capability`). Default = PLAN / CPU build, no GPU launch. Only with explicit user
confirmation set `launch: true`. The GPU target + creds come from the environment
only. The harness will require a confirmation before this irreversible verb runs.

## 4. Verify — assert lineage
Read the **STATUS**: `BUILT` (the `local` adapter produced a backbone/embeddings),
`PLANNED` (a staged plan, no mutation), or `REFUSED_NO_GPU_TARGET` (the gate
refused). The produced ref is a `<FlowName>/<run_id>` pathspec + fingerprint — never
a `.nemo`/`.pt` file; weights stay a Metaflow Artifact by pathspec.

## 5. Deliver — narrate, return run + STATUS
- Lead with the **STATUS** and the cost PLAN; then the produced ref and lineage.
  Make crystal clear whether this was a CPU build / staged PLAN (default) or a real
  GPU launch.
- Hand back the run + `@card` + typed summary with the headline `STATUS`/`VERDICT`
  (the embeddings pathspec is a first-class `dataset_ref` for `/loom-validate` /
  `/loom-run`).
- **Next step:** if `BUILT`, offer `/loom-validate --dataset <embeddings_ref>`; if
  `PLANNED` / `REFUSED_NO_GPU_TARGET`, point at the missing GPU target and the
  explicit `launch` once configured.
