# Loom skill-pack (v0.1)

Claude Code skills that drive Loom conversationally. Loom is an agentic CLI for the
**full data-science lifecycle** (domain-neutral; the metric is the spec) — ML
modeling (the `loom-optimize` verb) is one slice; the other 97% is data access,
EDA, features, pipelines, training, validation, viz, reporting, deployment, ops,
and collaboration. These skills are the human-facing front door — they plan, gate
on cost/data, invoke the `loom` CLI through its provider interfaces, and narrate
lineage-grounded results. They do not reimplement any engine logic; they call the
same `loom` entrypoints a human would.

## Authoring a new verb

Every `/loom-*` verb is a Claude Code `SKILL.md`. Author one from
[`_TEMPLATE/SKILL.md`](_TEMPLATE/SKILL.md) (the canonical template +
its 7-point acceptance test) and obey [`CONVENTIONS.md`](CONVENTIONS.md) (the
cost/data approval matrix, the provider-interface discipline, no-S3, learnings
capture). These are repo invariants — see the root [`CLAUDE.md`](../CLAUDE.md).

## Command catalog (design-spec §3)

The flat, hyphenated lifecycle surface. **Built** verbs ship in this pack today;
**roadmap** verbs are designed (design-spec §3, build order §6) and not yet
implemented. The default search brain is AIDE (behind `loom-optimize` only); the
default MLOps muscle is Metaflow.

| Verb | Status | What it does | Reach for it when |
| --- | --- | --- | --- |
| [`loom-setup-metaflow`](loom-setup-metaflow/SKILL.md) | **built** | **Expensive/mutate — always-gate.** Stand up the verified minikube + minio + Metaflow datastore (drives `scripts/setup_metaflow_minikube.sh`, idempotent) so the MLOps interface has a backend, then verify with the read-only `loom doctor` (must end PASS). Local-dev + reversible (`minikube delete`); NOT `disable-model-invocation`. | You need to stand up (or repair) Loom's local MLOps stack so the lifecycle verbs have a datastore. |
| [`loom-connect`](loom-connect/SKILL.md) | **built** | Data access — register a source as a Metaflow **data object** by pathspec (`loom ingest`) and list ingested data objects (`loom datasets`). The #1 daily DS pain and the front door to every other verb. | You need to point Loom at file data and get a `dataset_ref`, or see what's already ingested. |
| [`loom-eda`](loom-eda/SKILL.md) | **built** | **Read-only** profile of a data object **through the MLOps interface** (`loom eda`) — shape, dtypes, missingness, target balance, top correlations, leakage flags — emitting a Metaflow run + `@card`, plus suggested goal/metric phrasing. | You point at a `dataset_ref` and ask "what's in here?" / "is this ready for Loom?" / "check for leakage". |
| [`loom-features`](loom-features/SKILL.md) | **built** | **Workspace-write** feature engineering **through the MLOps interface** (`loom features`) — domain-neutral transforms (scaling/interactions, encoding, datetime parts, aggregations) built into a **NEW** versioned data object (`FeaturesFlow/<id>`) every downstream verb consumes via `--dataset`; composes with `loom-eda` via `--from` (its leakage-flagged columns are dropped). Reads the source read-only, writes only its own workspace. | You want engineered features as a reusable, lineage-grounded data object. |
| [`loom-pipeline`](loom-pipeline/SKILL.md) | **built** | **Workspace-write → escalates to expensive** end-to-end lifecycle **through the MLOps interface** (`loom pipeline`) — profile → features → a bounded candidate/optimize step → validate chained into ONE gated Metaflow run; each stage asserts the prior stage's `VERDICT` (leakage blocks features; a sub-threshold validate marks the run FAIL). The optimize stage is the bounded EXPENSIVE step. | You want a reproducible, gated ingest→feature→train→eval run in one shot. |
| [`loom-train`](loom-train/SKILL.md) | **built** | **Expensive/mutate → escalates to irreversible/external** model build **through the MLOps interface** (`loom train`) over the third heavy backend (the `ModelBuilderProvider` seam) — stated in DS-intent vocabulary (objective/budget/backbone/metric); the adapter hides ALL backend vocabulary (you never name NeMo / a GPU-count / a checkpoint). `pretrain` is launch-and-track (AIDE never tree-searches it). The cost PLAN (hours/$/GPU-count) is surfaced at the gate; the real GPU launch is behind `--launch` (OFF by default) and refuses cleanly with no GPU target. The torch-free CPU `local` adapter actually builds a backbone + `IngestDataset`-shaped embeddings; the default `nemo` adapter plans it. Always gated; `disable-model-invocation: true`. | You want to pretrain a backbone, embed via a frozen backbone, or fine-tune a cheap head — build the model the lifecycle needs. |
| [`loom-optimize`](loom-optimize/SKILL.md) (AIDE) | **built** | Metric-is-the-spec entry → plan → **approval gate (cost/data)** → invoke `loom run` → narrate best metric + leaderboard. | You want Loom to optimize solution code against a measurable metric. |
| [`loom-validate`](loom-validate/SKILL.md) | **built** | **Workspace-write** rigorous validation of a baseline/solution against a data object **through the MLOps interface** (`loom validate`) — sealed holdout distinct from a stratified/purged K-fold CV, probability calibration (curve + Brier), per-slice / fairness metrics, and leakage flags — emitting a Metaflow run + `@card` with a `VERDICT` that blocks `loom-deploy` if sub-threshold/leaky. | You want to check a candidate is good enough before promotion. |
| [`loom-viz`](loom-viz/SKILL.md) | **built** | **Read-only** charts/plots **through the MLOps interface** (`loom viz`) — feature distributions, correlation heatmap, target-vs-feature from a data object, or metric-over-nodes / leaderboard from a run — emitted as `@card` images. | You want a visual of a dataset/result, source-grounded to a pathspec. |
| [`loom-report`](loom-report/SKILL.md) | **built** | **Read-only** assembly **through the MLOps interface** (`loom report`) of an experiment's runs + metrics + lineage (Flow/Run + tags + learnings rows) into a structured analysis/model-card + `@card`; the narrative prose is the skill's job. | You want a shareable write-up of what Loom did and why. |
| [`loom-deploy`](loom-deploy/SKILL.md) | **built** | **Irreversible/external** gated promotion **through the MLOps interface** (`loom deploy`) — asserts the upstream `loom-validate` `VERDICT==PASS` (the cross-verb exit gate) before deploying; a sub-threshold/leaky validate **BLOCKS** it. Default = a deployment PLAN + staged registry manifest (no external mutation); the real apply is behind `--apply` (OFF by default). Always gated; `disable-model-invocation: true`. | You want to ship a validated model to serving (and gate the promotion on a trustworthy held-out number). |
| [`loom-ops`](loom-ops/SKILL.md) | **built** | **Read-only** monitoring **through the MLOps interface** (`loom ops`) — recent run health (successes/failures, recency), the leaderboard, schedule/run health, and a simple data-object DRIFT check (vs a reference) — emitted as a Metaflow run + `@card`. Reads are free; never prompts. | You want to see what passed/failed, read the leaderboard, or check whether the data drifted. |
| [`loom-collab`](loom-collab/SKILL.md) | **built** | **Workspace-write to build / irreversible-external to send** a sanitized shareable bundle **through the MLOps interface** (`loom collab`) — report/card + a lineage manifest (pathspecs + fingerprints + commit) as a run + `@card`. Build-only by default (no data leaves the box); the off-box SEND is behind `--send` (OFF by default), gated, to an env/config-driven sink. `disable-model-invocation: true`. | You want to share or hand off a run/report to a teammate, lineage-grounded and sanitized. |
| [`loom-auto`](loom-auto/SKILL.md) | **built** | **Graduated — read-only/workspace-write → expensive at the optimize step.** Meta-skill (no new flow): orchestrates the existing verbs end-to-end — (ingest if a raw source) → `loom eda` → [leakage gate] `loom features` → `loom run`/optimize → `loom validate` → `loom report` — threading each artifact (`--from`/`--dataset`/`--solution`/`--runs`) and asserting each prior VERDICT, gating at the EXPENSIVE optimize step; a sub-threshold/leaky validate STOPS the chain. **Never auto-fires `loom-deploy` or `loom-collab --send`.** | You want the standard chain end-to-end without memorizing the verbs. |

## Typical flow

1. **`loom-connect`** — register a source as a Metaflow data object and get its
   `dataset_ref` pathspec (or list the data objects already ingested). Read-only/
   light; the front door to everything below.
2. **`loom-eda`** — profile the data object by pathspec, confirm the target, check
   for leakage, get suggested goal and metric sentences. Read-only; spends no
   budget; emits a Metaflow run + `@card`.
3. **`loom-optimize`** — pin the data/goal/metric, propose a run plan
   (providers, budget, models), **stop at the approval gate** (cost shape + data
   scope + the exact command), then run `loom run` and summarize the best
   solution, the leaderboard, and the artifact paths.

## Interface (v0.1)

Each verb is a plain Claude-Code `SKILL.md` file: YAML frontmatter
(`name` + `description` + `when_to_use`) followed by markdown instructions. They
shell out to the project's CLI (never importing a concrete backend):

```bash
bash scripts/setup_metaflow_minikube.sh      # loom-setup-metaflow: stand up the local datastore (idempotent, gated)
loom doctor [--config YAML]                  # loom-setup-metaflow: read-only stack health check (PASS/WARN/FAIL + VERDICT)
loom ingest --source PATH [--name NAME]      # loom-connect: register a data object
loom datasets                                # loom-connect: list ingested data objects
loom eda --dataset PATHSPEC [--target COL]   # loom-eda: read-only profile -> run + @card
loom features --dataset PATHSPEC [--target COL] [--from EDA-RUN] [--recipe minimal|full]  # loom-features: build a NEW feature data object -> run + @card
loom pipeline --dataset PATHSPEC --goal STR [--target COL]  # loom-pipeline: profile->features->optimize->validate in one gated run -> run + @card
loom train --dataset PATHSPEC --objective next-event|masked-field|contrastive --budget probe|small|full [--capability pretrain|tokenize|finetune|embed] [--backbone PATHSPEC] [--metric STR] [--launch]  # loom-train: build a model via the model-builder seam -> run + @card (launch OFF by default; refuses with no GPU target)
loom optimize ...  # via `loom run` (loom-optimize: AIDE tree-search)
loom validate --dataset PATHSPEC [--target COL] [--solution RUN] [--sensitive COL]  # loom-validate: CV+holdout+calibration+fairness+leakage -> run + @card
loom deploy (--validate RUN | --solution RUN) [--apply]  # loom-deploy: gate on validate VERDICT==PASS -> deploy PLAN/manifest (apply OFF by default)
loom ops (--flow NAME | --experiment ID | --dataset PATHSPEC --reference PATHSPEC)  # loom-ops: read-only run health / leaderboard / drift -> run + @card
loom collab (--run PATHSPEC | --experiment ID) [--send]  # loom-collab: sanitized shareable bundle -> run + @card (send OFF by default)
loom report (--experiment ID | --runs PATHSPEC,...)  # loom-report: assemble runs+metrics+lineage -> run + @card
loom viz (--dataset PATHSPEC | --run PATHSPEC) [--target COL] [--kind ...]  # loom-viz: standard plots -> run + @card images
loom run --dataset PATHSPEC --goal STR --metric STR [--steps N] [--mlops metaflow|local] [--search aide]
# loom-auto: NO new CLI verb — the meta-skill orchestrates the verbs above in sequence
#   (ingest if a raw source) -> eda -> [leakage gate] features -> run/optimize -> validate -> report,
#   threading --from/--dataset/--solution/--runs and asserting each VERDICT; gates at the optimize step;
#   never auto-fires `loom deploy` / `loom collab --send`.
```

**Composition edges (artifact handoff + `--from`/`--validate`/`--run`, machine-checkable exit gates):**
`loom-eda` leakage flags → `loom-features` (via `--from`, the flagged columns are dropped); a
`loom-features` data object → `loom-pipeline`/`loom-optimize`/`loom-validate` (via `--dataset`); a
`loom-train` backbone → `loom-train --capability embed` (via `--backbone`), and its
`IngestDataset`-shaped embeddings → `loom-validate`/`loom-optimize` (via `--dataset`); a
`loom-validate` `VERDICT==PASS` → `loom-deploy` (via `--validate`, a sub-threshold validate BLOCKS);
a `loom-report`/`loom-validate` run → `loom-collab` (via `--run`). Each gate ships an executable
self-test (train local-lift + nemo no-GPU refusal; deploy BLOCK-on-sub-threshold; pipeline stage-gate
ordering; collab send-off + sanitize; ops drift; features leakage-drop).

Providers are selected by name (search brain default `aide`; MLOps muscle default
`metaflow`, with `local` as a Metaflow-free dev path). See the repo
[`README.md`](../README.md) and [`docs/architecture.md`](../docs/architecture.md)
for the engine and provider model.

## Conventions

The full rules every verb obeys live in [`CONVENTIONS.md`](CONVENTIONS.md) (the
cost/data approval matrix, provider-interface discipline, no-S3, lineage +
mandated artifact, learnings capture). The headlines:

- **Speak the interface, not the backend.** Verbs call Loom's MLOps/search
  interface (via the `loom` CLI, which resolves providers by name) — never
  Metaflow/AIDE directly and never raw S3. MLOps default Metaflow, search default
  AIDE; both swappable by config.
- **Cost/data is gated by tier.** Read-only never prompts; workspace-write is
  light/auto within a budget; expensive/mutate always gates; irreversible/external
  always gates and is never model-auto-invoked. Prefer the cheap path first
  (`--mlops local`, small `--steps`).
- **Mandated artifact.** Every run returns a versioned Metaflow run + `@card` + a
  typed-JSON summary, lineage-grounded by a Verifier step, and appends a sanitized
  row to the flywheel corpus (`learnings/rollouts.jsonl`).
- **Secrets via env only.** Keys/endpoints come from `.env`/environment; skills
  never print or pass key material.
- **Domain-neutral.** No customer-, vertical-, or pricing-specific content — that
  strategy lives elsewhere, never in this repo.
