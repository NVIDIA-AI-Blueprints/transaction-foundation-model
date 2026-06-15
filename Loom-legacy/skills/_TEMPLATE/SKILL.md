---
name: loom-<verb>
description: <One 3rd-person line describing the DS need this serves, rich enough that the model auto-loads it. Embed the LITERAL user phrases that should trigger it, e.g. "profile this data", "run Loom on X", "is this ready to deploy". State plainly whether it mutates anything.>
when_to_use: <Intent phrases for dual-invocation auto-load — e.g. "explore a dataframe", "check for leakage before features", "validate a model before deploy".>
when_not_to_use: <The neighbouring verb to defer to instead — e.g. "for an unprofiled dataset run loom-eda first", "for spending a search budget use loom-optimize". Keeps the catalog from overlapping.>
argument-hint: "<one free-text noun, e.g. 'dataset_ref + focus'>"
# disable-model-invocation: true   # REQUIRED for irreversible/costly verbs (deploy, optimize, anything that spends compute or hits prod) so the model never auto-fires them.
---

# loom-<verb>

> Copy this file to `skills/loom-<verb>/SKILL.md`, fill every section, then run
> the **acceptance test** at the bottom before adding the verb to the pack and
> to `skills/README.md`. Read `skills/CONVENTIONS.md` first — the approval
> matrix and the provider-interface discipline are repo invariants, not
> suggestions.

One paragraph: what DS lifecycle job this verb does and why it is a Loom tool
(a planned, gated, lineage-grounded run) rather than a loose script. **The
metric is the spec.** Stay domain-neutral — never assume a task type, column
meaning, or vertical; reflect back only what the data and the user's metric
actually say.

## When to use

- <The concrete situations + literal user phrases that should reach for this verb.>

## When NOT to use

- <The neighbouring verb to hand off to instead, so the catalog stays disjoint.>

## 1. Intake — pin the spec (refuse if it is not measurable)

Pin the inputs in the user's own terms and write them back for confirmation:

- **Data** — the `dataset_ref` (a Metaflow **pathspec**, e.g. `IngestDataset/123`,
  produced by `loom ingest`). Take a path/pathspec, never a raw S3 URI or a
  loose local file as the source of truth.
- **Goal** — one natural-language sentence of what a solution should achieve.
- **Metric** — one natural-language sentence of how a solution is scored, stated
  so the optimization direction is unambiguous (e.g. "Maximize ROC AUC on a
  held-out split").

If any required piece is missing or vague, **ask — do not guess**. For
optimize/validate-class verbs, **refuse to start without a measurable
`(dataset_ref, goal, eval-metric)` spec** — a wrong metric silently optimizes
the wrong thing. Pure read-only verbs may proceed with just a `dataset_ref` and
should auto-generate a data-preview `@card` first.

## 2. Plan — show the plan + cost/data tier, stop at the gate if tier > read-only

Propose a concrete plan and show it before doing anything that spends compute or
touches data:

- **What will run** — the steps, the provider(s) involved, and (for any loop) a
  **declared budget**: step cap / wall-clock cap / cost cap. No unbounded loops.
- **Cost shape** — roughly how many candidate executions / model calls the plan
  implies, and that on a remote MLOps profile it consumes the user's own compute.
- **Data scope** — exactly which `dataset_ref` will be read, and that on a remote
  profile the data is staged into the user's own perimeter, not exfiltrated.
- **Declare the tier** (see `CONVENTIONS.md`): `read-only` | `workspace-write` |
  `expensive/mutate` | `irreversible/external`. **Stop at the approval gate if
  the tier is greater than read-only** — present the exact command/operation and
  do not proceed until the user confirms. Surface only the taste decisions
  (which metric, which threshold, which features); everything mechanical is
  autonomous within the declared budget. Re-plan and re-present the gate if the
  user adjusts budget/providers/metric.

## 3. Run — call Loom's MLOps / search INTERFACE, never the backend

- Speak **only Loom's provider interfaces**: input is a data object by
  `dataset_ref` (pathspec) read via the MLOps interface's Client API; work
  **executes as a Metaflow run** through the `ExecutionProvider`; ML iteration
  goes through the `SearchProvider`. **Never call Metaflow or AIDE directly**,
  and **never touch raw S3** — the datastore is an opaque detail the MLOps
  interface owns. In v0.1 this means shelling out to the `loom` CLI (which
  resolves providers by name), not importing a concrete adapter.
- The MLOps default is **Metaflow** and the search default is **AIDE**; both are
  swappable by config. Never name a concrete backend where the interface will
  do, so the verb stays backend-swappable.
- Secrets/endpoints come from the **environment only** (`.env`/env) — never put
  keys on the command line, in the plan, or in the transcript.

## 4. Verify — assert lineage; spill large output to an Artifact

- Run a **Verifier step** (a STEP, not a prompt suffix) that asserts lineage
  integrity before emitting: every chart / metric / claim links back to its
  **pathspec + data fingerprint + commit**.
- **Large output → Artifact, not inline text.** Dataframes, logs, sweep results,
  big metrics tables become a Metaflow **Artifact** referenced by pathspec; cap
  any inline exec/tool output at **~25k tokens**.

## 5. Deliver — narrate, return run + @card + typed-JSON summary, append a learnings row

- **Narrate** the result for the user: what was found, whether it meets the goal,
  the relevant leaderboard/spread (not just the winner), and the next step.
- **Hand back the mandated artifact:** a versioned **Metaflow run + an `@card`**
  (the shareable render), plus a **typed (schema-conformant) JSON summary** with
  a `VERDICT`/status line that a downstream verb can consume via `--from`.
- **Append a learnings row** to the flywheel corpus (`learnings/rollouts.jsonl`)
  — a typed, versioned, **sanitized** record (task spec · data fingerprint ·
  exec result · metric · judge feedback · lineage · model + tokens). Sanitize
  anything ingested from notebooks/datasets; never write secrets. The moat
  compounds from run #1.

## Composition — machine-checkable exit gate (executable self-test)

- **Consumes:** `<named prior artifact via --from, e.g. a feature-set-id>`.
- **Exit gate:** emit the typed JSON `VERDICT`/status that the next verb asserts
  before it runs (e.g. `loom-eda` leakage flags gate `loom-features`; a
  sub-threshold `loom-validate` blocks `loom-deploy`).
- **Self-test:** the gate **must have an executable self-test** that runs on a
  known-bad fixture (e.g. a known-leaky dataset) and asserts the gate **BLOCKS**.
  "Guards failing open" is the cautionary tale — a declared gate without a test
  does not count.

---

## Acceptance test (the bar before this verb joins the pack)

1. **Speaks only the interface** — calls Loom's MLOps/search interface, never
   Metaflow/AIDE directly and never raw S3.
2. **Output is a versioned run + `@card`** — not a chat transcript or a loose
   script.
3. **Approval tier is correct** — declares its sandbox tier and gate per
   `CONVENTIONS.md`; irreversible/costly verbs set `disable-model-invocation`.
4. **Writes a learnings row** — appends a typed, sanitized record to the
   flywheel corpus on every run.
5. **Exit gate has a self-test** — any composition gate ships an executable test
   that proves it blocks on a bad fixture.
6. **Single free-text arg** — one quoted noun, not a flag soup.
7. **Dual-invocation** — works user-typed (`/loom-<verb>`) and model-auto-loaded
   (on `description`/`when_to_use` match), with `disable-model-invocation` set
   wherever auto-firing would be unsafe.
