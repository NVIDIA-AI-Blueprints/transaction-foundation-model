---
name: loom-auto
description: The one-command HAPPY PATH — orchestrate Loom's existing lifecycle verbs end-to-end so the user does not have to memorize the chain. From a dataset_ref (or a raw source to ingest) + a measurable goal + a metric, it runs (ingest if needed) -> loom eda -> [leakage gate] loom features -> loom run/optimize -> loom validate -> loom report, threading each step's artifact into the next via --from/--dataset/--solution and asserting each prior VERDICT before continuing. Graduated autonomy: read-only/workspace-write until the EXPENSIVE optimize step, where it STOPS at the cost gate. A sub-threshold/leaky validate STOPS the chain and surfaces it. It NEVER auto-fires deploy or collab-send — those stay user-fired. Use when the user says "just run Loom on this", "do the whole thing", "give me the standard pipeline", "run the happy path", "I don't want to memorize the verbs".
when_to_use: "run the standard end-to-end chain without naming each verb, just run Loom on a dataset toward a goal, the one-command happy path, profile->features->optimize->validate->report in sequence"
when_not_to_use: "to run ONE verb (only profile / only validate), use that verb directly; for a single gated Metaflow DAG instead of orchestrated CLI calls, use loom-pipeline; to ship a model, use loom-deploy (this never auto-deploys); to share a bundle off-box, use loom-collab --send (this never auto-sends)."
argument-hint: "<a dataset_ref pathspec OR a raw source to ingest + a goal sentence + a metric sentence>"
---

# loom-auto

The **one-command happy path**: drive Loom's *existing* lifecycle verbs in sequence so
the user gets a profiled, feature-engineered, optimized, validated, written-up result
without naming each verb. This is a **meta-skill, not a new flow** — it adds **no**
Metaflow flow and reimplements nothing; it orchestrates the same `loom` CLI verbs a
human would run, threading each step's artifact into the next and **asserting each
prior `VERDICT` before continuing**. It is **opinionated low, deferential high**:
mechanical sequencing is autonomous, but it **stops at the one expensive gate**
(optimize) and at any failed exit gate. **The metric is the spec** — it refuses to
start without a measurable goal/metric. Stay domain-neutral; reflect back only what the
data and the user's metric actually say.

## When to use

- The user says "just run Loom on this", "do the whole thing", "give me the standard
  pipeline", "run the happy path", or "I don't want to memorize the verbs".
- They have a dataset (or a raw source) and a measurable goal and want the standard
  chain end-to-end with the gates handled for them.

## When NOT to use

- To run **one** verb (only profile, only validate, …) — call that verb directly.
- For a **single gated Metaflow DAG** (one composite run, not orchestrated CLI calls) —
  use **`loom-pipeline`** (this meta-skill calls the *separate* verbs and threads their
  artifacts; pipeline fuses them into one run).
- To **ship** a model — use **`loom-deploy`** (this meta-skill **never** auto-deploys).
- To **share off-box** — use **`loom-collab --send`** (this meta-skill **never**
  auto-sends).

## 1. Intake — pin the spec (refuse without a measurable goal + metric)

Pin the inputs in the user's own terms and write them back for confirmation:

- **Data** — either a **`dataset_ref`** (a Metaflow **pathspec**, e.g.
  `IngestDataset/123`) to start from, **or** a **raw source** (a dir/CSV path) to
  register first via `loom ingest`. Take a pathspec or a source to ingest — **never a
  raw S3 URI** as the source of truth.
- **Goal** — one natural-language sentence of what a solution should achieve.
- **Metric** — one natural-language sentence of how a solution is scored, stated so the
  optimization direction is unambiguous (e.g. "Maximize ROC AUC on a held-out split").

This meta-skill spends a search budget at the optimize step, so it is an
optimize-class entry: **refuse to start without a measurable `(data, goal, metric)`
spec** — a wrong metric silently optimizes the wrong thing. If any piece is missing or
vague, **ask — do not guess**. Optionally accept a `--target` column (threaded into
features/validate) and a `--budget` for the optimize step.

## 2. Plan — show the verb sequence + graduated autonomy, STOP at the optimize gate

Show the full chain and its **graduated tier** before running anything. The composed
tier is the **max of its steps**: read-only / workspace-write through eda → features,
then it **escalates to EXPENSIVE at the optimize step** (see `CONVENTIONS.md`). The
plan, with the exit-gate composition:

1. **(ingest)** — `loom ingest` only if the user gave a raw source (→ a `dataset_ref`).
   *(boundary; skipped if a pathspec was given.)*
2. **`loom eda`** — read-only profile of the `dataset_ref`. **[leakage gate]** its
   `leakage` flags are threaded into features.
3. **`loom features`** — workspace-write; build a new feature data object,
   **`--from` the eda run** so the EDA-flagged leakage columns are dropped.
4. **`loom run` (optimize)** — **the EXPENSIVE step.** Search candidate solutions to
   the metric under a **declared budget** (`--steps`/budget cap). **This is the gate.**
5. **`loom validate`** — workspace-write; sealed holdout + CV + calibration + leakage,
   `--solution` the optimize run. Emits the `VERDICT` that gates what comes next.
6. **`loom report`** — read-only; assemble the runs + metrics + lineage into a
   model-card `@card`.

At the plan, **surface the cost shape** (the optimize budget — roughly how many
candidate executions / model calls, consuming the user's own compute) and the **data
scope** (the exact `dataset_ref`, staged into the user's own perimeter). **Stop at the
approval gate before the optimize step** — present the exact `loom run` command + its
budget and do not proceed until the user confirms. Re-plan and re-present the gate if
the user adjusts the budget / metric / target. Make explicit up front: **this chain
ends at the report; it NEVER auto-invokes `loom-deploy` or `loom-collab --send`** —
those stay user-fired.

## 3. Run — invoke each /loom-* verb via the `loom` CLI, threading artifacts + asserting VERDICTs

Speak only Loom's interface — shell out to the same `loom` CLI verbs a human would
(which resolve providers by name). **Never call Metaflow or AIDE directly, and never
touch raw S3.** Run the steps in sequence, **threading each prior artifact into the
next and asserting the prior step's `VERDICT`/status before invoking the next**:

```bash
# (0) Only if a raw source was given — register it, capture the dataset_ref.
loom ingest --source <PATH> --name <NAME>        # -> IngestDataset/<id>

# (1) Profile (read-only). Capture the EDA run pathspec + its leakage flags.
loom eda --dataset <DATASET_REF> [--target <COL>]

#     ASSERT: read the eda VERDICT/leakage. Thread the eda run into features.
# (2) Features (workspace-write) — drop EDA-flagged leakage via --from.
loom features --dataset <DATASET_REF> [--target <COL>] --from <EDA_RUN>
#     -> FeaturesFlow/<id>  (this becomes the --dataset for everything below)

#     === APPROVAL GATE (EXPENSIVE) — stop here, show cost, await confirm ===
# (3) Optimize (EXPENSIVE) — search against the metric under a declared budget.
loom run --dataset <FEATURES_REF> --goal "<goal>" --metric "<metric>" --steps <N>
#     -> the optimize run pathspec

#     ASSERT: the optimize run produced a best solution. Thread it into validate.
# (4) Validate (workspace-write) — sealed holdout; emits the gating VERDICT.
loom validate --dataset <FEATURES_REF> [--target <COL>] --solution <OPTIMIZE_RUN>

#     ASSERT: validate VERDICT. PASS -> continue; REVIEW/sub-threshold -> STOP (below).
# (5) Report (read-only) — assemble the chain into a model-card @card.
loom report --runs <EDA_RUN>,<FEATURES_RUN>,<OPTIMIZE_RUN>,<VALIDATE_RUN>
```

**Exit-gate composition (the chain is only as trustworthy as its gates):**

- **Leakage gate (eda → features):** if `loom eda` flags leakage, the columns are
  dropped via `--from` into features; if leakage cannot be resolved (no droppable
  columns), **stop and surface it** rather than building features on a leaky object.
- **Optimize gate:** the EXPENSIVE step never runs un-gated — it runs only after the
  user confirms at the approval gate above.
- **Validate gate (the chain terminator):** assert `loom validate`'s `VERDICT`. A
  **sub-threshold or leaky validate (`REVIEW` / FAIL) STOPS the chain** — surface
  exactly what failed (the holdout number vs the bar, or the leakage flag) and **do
  NOT proceed to deploy/collab**. A passing run continues to the (read-only) report.

Each step's bulk output stays in Metaflow (data objects / artifacts by pathspec); the
meta-skill threads only small *derived* references (pathspecs, VERDICTs) between steps.
Secrets/endpoints come from the **environment only** — never on the command line or in
the transcript.

## 4. Verify — assert the threaded lineage end-to-end

- Each verb returns a **run pathspec + an `@card`**; confirm each reported success
  before threading it forward. The chain's integrity *is* the per-step VERDICT
  assertions plus the artifact handoff (`--from` / `--dataset` / `--solution` /
  `--runs`) — every downstream step reads a real upstream pathspec, never a re-derived
  guess.
- The final **`loom report`** is the lineage capstone: it links every run + metric back
  to its pathspec + data fingerprint + commit. Large output already lives in Metaflow;
  the meta-skill inlines only the narrated summary (cap inline output at ~25k tokens).

## 5. Deliver — narrate the end-to-end run, return the report @card + the lineage

- **Narrate end-to-end:** walk the chain — the profile (shape / leakage handled), the
  feature object built, the optimize **best metric + leaderboard spread** (not just the
  winner), the validate **sealed-holdout `VERDICT`** (and calibration / fairness if
  surfaced), and the assembled report. If the chain **stopped at a gate** (leakage
  unresolved, or a sub-threshold validate), say exactly **where and why** it stopped and
  what to fix.
- **Hand back the mandated artifacts:** the **final report `@card`** (the shareable
  render) plus the **full lineage** — the EDA / features / optimize / validate run
  pathspecs threaded through the chain, each a versioned Metaflow run.
- **Learnings:** each underlying verb appends its own typed `command=...` row to the
  flywheel corpus (`learnings/rollouts.jsonl`) as it runs — the meta-skill does not
  hand-write rows; the per-verb capture covers the whole chain.
- **Next step (always user-fired):** if validate is `PASS` and the user wants to ship,
  point them at **`loom-deploy`** (which re-asserts the validate `VERDICT==PASS` gate);
  to hand off, **`loom-collab --send`**. **`loom-auto` never fires either** — it
  proposes; the user decides.

## Composition — machine-checkable exit gate (executable self-test)

- **Consumes:** a `dataset_ref` pathspec (or a raw source it ingests first) + a goal +
  a metric; threads each step's artifact into the next.
- **Exit gate:** it **composes the underlying verbs' exit gates** — the eda `leakage`
  flags gate `loom-features`; the validate `VERDICT` gates whether the chain continues
  (a sub-threshold/leaky validate STOPS it and forbids deploy/collab). The
  optimize step is the gated EXPENSIVE step. The meta-skill **never auto-fires** the
  irreversible verbs (deploy/collab-send).
- **Self-test:** the composed gates are the underlying verbs' own executable self-tests
  — the leakage gate by `tests/test_eda.py` (a known-leaky fixture is flagged) and the
  features drop by `tests/test_features.py`; the validate `VERDICT`/leakage gate by
  `tests/test_validate.py::test_validate_flags_leakage_and_sets_review_verdict`
  (leakage forces `REVIEW`, which this chain treats as STOP, never deploy). This
  meta-skill orchestrates those gated verbs, so it inherits their proven blocks; it
  ships no flow of its own to test.

---

## Acceptance test (the bar before this verb joins the pack)

1. **Speaks only the interface** — shells out to the existing `loom` verbs (`ingest` /
   `eda` / `features` / `run` / `validate` / `report`), never Metaflow/AIDE directly,
   never raw S3; inputs/handoffs are `dataset_ref` / run pathspecs read via the
   Client API.
2. **Output is versioned runs + an `@card`** — the chain hands back the final report
   `@card` + the threaded run lineage, not a chat transcript or a loose script. It adds
   no new flow.
3. **Approval tier is correct** — graduated; the composed tier is the max of its steps,
   read-only/workspace-write until it **escalates to EXPENSIVE at the optimize step**,
   where it **stops at the gate**. It **never auto-invokes** the irreversible verbs
   (deploy / collab-send) — no `disable-model-invocation` is needed on the meta-skill
   because it gates at the optimize step and never fires the irreversible verbs itself.
4. **Writes a learnings row** — the underlying verbs each append their own sanitized
   `command=...` row to `learnings/rollouts.jsonl` as the chain runs; the whole chain
   is captured.
5. **Exit gate has a self-test** — it composes the underlying verbs' gates, each
   covered by that verb's executable self-test (eda leakage, features drop, validate
   `VERDICT`); a sub-threshold/leaky validate STOPS the chain.
6. **Single free-text arg** — one noun: a `dataset_ref` (or a source) + a goal + a
   metric (plus optional `--target` / `--budget`), not flag soup.
7. **Dual-invocation** — works user-typed (`/loom-auto`) and model-auto-loaded on the
   `description` / `when_to_use` match; the model proposes the chain and **stops at the
   optimize gate**, and never auto-fires deploy / collab-send.
