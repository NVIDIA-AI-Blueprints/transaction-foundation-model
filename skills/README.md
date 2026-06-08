# Loom skill-pack (v0.1)

Claude Code skills that drive Loom conversationally. Loom is a general-purpose,
domain-neutral automated ML engine (the metric is the spec); these skills are the
human-facing front door to it — they plan, gate on cost/data, invoke the `loom`
CLI, and narrate results. They do not reimplement any engine logic; they call the
same `loom run` entrypoint a human would.

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
| [`loom-setup-metaflow`](loom-setup-metaflow/) | **roadmap** (in progress) | Install the verified minikube + minio + Metaflow recipe so the MLOps interface has a backend. | You need to stand up Loom's local MLOps stack. |
| [`loom-connect`](loom-connect/SKILL.md) | **built** | Data access — register a source as a Metaflow **data object** by pathspec (`loom ingest`) and list ingested data objects (`loom datasets`). The #1 daily DS pain and the front door to every other verb. | You need to point Loom at file data and get a `dataset_ref`, or see what's already ingested. |
| [`loom-eda`](loom-eda/SKILL.md) | **built** | **Read-only** profile of a data object **through the MLOps interface** (`loom eda`) — shape, dtypes, missingness, target balance, top correlations, leakage flags — emitting a Metaflow run + `@card`, plus suggested goal/metric phrasing. | You point at a `dataset_ref` and ask "what's in here?" / "is this ready for Loom?" / "check for leakage". |
| `loom-features` | roadmap | Build features into a versioned feature set; gated by `loom-eda`'s leakage flags. | You want engineered features as a reusable, lineage-grounded artifact. |
| `loom-pipeline` | roadmap | Author/run the multi-stage DS pipeline (the recipe primitive; *is* a Metaflow `FlowSpec`). | You want a reproducible ingest→clean→feature→train→eval DAG. |
| [`loom-optimize`](loom-optimize/SKILL.md) (AIDE) | **built** | Metric-is-the-spec entry → plan → **approval gate (cost/data)** → invoke `loom run` → narrate best metric + leaderboard. | You want Loom to optimize solution code against a measurable metric. |
| [`loom-validate`](loom-validate/SKILL.md) | **built** | **Workspace-write** rigorous validation of a baseline/solution against a data object **through the MLOps interface** (`loom validate`) — sealed holdout distinct from a stratified/purged K-fold CV, probability calibration (curve + Brier), per-slice / fairness metrics, and leakage flags — emitting a Metaflow run + `@card` with a `VERDICT` that blocks `loom-deploy` if sub-threshold/leaky. | You want to check a candidate is good enough before promotion. |
| [`loom-viz`](loom-viz/SKILL.md) | **built** | **Read-only** charts/plots **through the MLOps interface** (`loom viz`) — feature distributions, correlation heatmap, target-vs-feature from a data object, or metric-over-nodes / leaderboard from a run — emitted as `@card` images. | You want a visual of a dataset/result, source-grounded to a pathspec. |
| [`loom-report`](loom-report/SKILL.md) | **built** | **Read-only** assembly **through the MLOps interface** (`loom report`) of an experiment's runs + metrics + lineage (Flow/Run + tags + learnings rows) into a structured analysis/model-card + `@card`; the narrative prose is the skill's job. | You want a shareable write-up of what Loom did and why. |
| `loom-deploy` | roadmap | Promote/serve a model — **irreversible/external**, always gated, never model-auto-invoked. | You want to ship a validated model to serving. |
| `loom-ops` | roadmap | Inspect/monitor running and past runs (jobs, status), killable; reads are free. | You want to see what's running or revisit a prior run. |
| `loom-collab` | roadmap | Collaboration / handoff around runs and cards. | You want to share or hand off a run to a teammate. |
| `loom-auto` | roadmap | Meta-skill: the one-command happy path (EDA→features→baseline→validate), surfacing only taste decisions. | You want the standard chain without memorizing the verbs. |

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
loom ingest --source PATH [--name NAME]      # loom-connect: register a data object
loom datasets                                # loom-connect: list ingested data objects
loom eda --dataset PATHSPEC [--target COL]   # loom-eda: read-only profile -> run + @card
loom validate --dataset PATHSPEC [--target COL] [--solution RUN] [--sensitive COL]  # loom-validate: CV+holdout+calibration+fairness+leakage -> run + @card
loom report (--experiment ID | --runs PATHSPEC,...)  # loom-report: assemble runs+metrics+lineage -> run + @card
loom viz (--dataset PATHSPEC | --run PATHSPEC) [--target COL] [--kind ...]  # loom-viz: standard plots -> run + @card images
loom run --dataset PATHSPEC --goal STR --metric STR [--steps N] [--mlops metaflow|local] [--search aide]
```

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
