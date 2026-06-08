# /loom-* skill conventions

Shared rules every `/loom-*` skill in this pack obeys. These are **repo
invariants** (see the root `CLAUDE.md`), not style preferences — a verb that
breaks them does not join the pack. Author a new verb from
[`_TEMPLATE/SKILL.md`](_TEMPLATE/SKILL.md), which operationalizes everything here.

A `/loom-*` skill is the human-facing front door to Loom's engine: it plans,
gates on cost/data, calls Loom's **provider interfaces** (in v0.1, by shelling
out to the `loom` CLI), and narrates a lineage-grounded result. It never
reimplements engine logic.

## 1. Cost / data approval matrix

Approval is **orthogonal to capability** and is enforced by the client/hook
layer beneath the model — not by prose the model can talk itself out of. Every
verb **declares its tier** in its plan and gates accordingly. Network is **off
by default** (notebooks can exfiltrate sensitive data).

| Tier | Examples | Gate |
| --- | --- | --- |
| **read-only** | EDA, profiling, leaderboard reads, `@card` reads | **Never prompts.** Reads are non-destructive; run free. |
| **workspace-write** | feature builds, candidate runs, viz, local/scratch writes, artifacts saved locally | **Light / auto** within a declared budget; network off by default. Auto-escalate to a prompt only on failure or when a cost/data boundary is crossed. |
| **expensive / mutate** | GPU training, large `foreach`, full-table scans, dataset/feature-store writes, registry edits | **Always gate.** Show estimated cost/rows and the exact operation before running. |
| **irreversible / external** | `loom-deploy`, prod schedule, dropping/overwriting a gold table, deleting a registry entry, sending data off-box | **Always gate, and never model-auto-invoke** — set `disable-model-invocation: true`. The model proposes; only the user fires. |

Principle: **be opinionated low, permissive high** — most opinionated at
compute/data/spend (the gated bottom), most deferential at modeling choices (the
free top). Surface only the *taste* decisions (which metric, which threshold,
which features); everything mechanical is autonomous within the declared budget.

**Every autonomous loop has a declared budget** — step cap, wall-clock cap, cost
cap. No unbounded loops. Cap exec/tool outputs at **~25k tokens**; spill larger
output to a Metaflow Artifact by pathspec.

## 2. Provider-interface discipline

- Skills speak **only Loom's provider interfaces** — the MLOps `ExecutionProvider`
  and the search `SearchProvider`. **Never call Metaflow or AIDE directly**, and
  never name a concrete backend where the interface will do. In v0.1 this means
  invoking the `loom` CLI (which resolves providers by name through the registry),
  not importing an adapter.
- The **MLOps default is Metaflow**, the **search default is AIDE** — both
  swappable purely by config. A verb must stay correct if the backend is swapped.
- Work **executes as a Metaflow run**. Input is a data object referenced by
  `dataset_ref` (a Metaflow **pathspec**, e.g. `IngestDataset/123`), read via the
  MLOps interface's Client API.

## 3. No raw S3 / no loose files as truth

- **No direct S3.** Below the MLOps interface there is **no raw S3** — Client API
  and Artifacts only. The datastore (local / minio / S3) is an opaque detail the
  interface owns; a skill never touches it.
- **Nothing ephemeral is the deliverable.** A loose local file or notebook is
  never the source of truth; the mandated artifact is a versioned **Metaflow run
  + `@card`** plus a typed JSON summary.

## 4. Secrets via environment only

Keys, tokens, and endpoints come from the **environment** (`.env`/env) at the
point of use. Never put key material on a command line, in a plan, in the
transcript, or in any persisted artifact or learnings row.

## 5. Lineage + the mandated artifact

- Every command returns a **Metaflow run + an `@card`** (the shareable, versioned
  render) and a **typed, schema-conformant JSON summary** with a `VERDICT`/status
  line for downstream verbs/CI.
- **Source-grounding = lineage:** every chart/metric/claim links to its
  **pathspec + data fingerprint + commit**, asserted by a **Verifier step** (a
  step, not a prompt suffix) before the artifact is emitted.

## 6. Composition + exit gates

- Composition is **artifact handoff on disk (Metaflow objects) + machine-checkable
  exit gates + explicit `--from`** — not pipe syntax.
- A step's typed summary carries a `VERDICT`/status the next step asserts before
  running (leakage flags from `loom-eda` gate `loom-features`; a sub-threshold
  `loom-validate` blocks `loom-deploy`).
- **Every exit gate ships an executable self-test** that runs on a known-bad
  fixture and asserts the gate BLOCKS. A gate without a test does not count.

## 7. Learnings capture (the flywheel / moat)

- **The primary central-data-collection mechanism is the Loom gateway**
  (`loom-proxy` + `loom proxy serve`; see the README). When a run routes through
  the gateway, Loom **owns the LLM egress** and logs every call — request system +
  messages, response text + usage, model, latency, tenant/owner tags — to one
  central JSONL corpus (`LOOM_PROXY_LOG_PATH`, default
  `learnings/proxy_calls.jsonl`). That central capture is the moat; the per-node
  corpus + `learnings/rollouts.jsonl` below are the complementary structured
  capture.
- Every run also **appends one typed row** to the flywheel corpus
  (`learnings/rollouts.jsonl`): task spec · data fingerprint · exec result ·
  metric · judge feedback · lineage · model + tokens. **The moat compounds from
  run #1** — capture is mandatory, not optional, even in v0.1 (optimization of the
  skills from this corpus spins up in v0.2+).
- **Bulk data stays in Metaflow — that is the real privacy line, and it holds for
  EVERY provider.** Datasets/transactions never go to the LLM; the LLM sees only
  small *derived* context (schema/preview/code/metrics), so keep raw rows OUT of
  prompts (**prompt hygiene**; AIDE injects a small data-preview, not the data).
  Prompts go to a third-party LLM (Anthropic) either way — `loom-proxy` is the
  *same* egress path, **no incremental leak** ("Claude or Us — no difference").
  So the provider choice is a *data-collection* choice, not a bulk-data-privacy
  one: `loom-proxy` logs the prompt traces (fuels the moat; default once hosted),
  BYO-key providers don't (the gateway sees nothing). A tenant that doesn't want
  Loom collecting its traces uses BYO-key — its bulk data is equally protected
  in both modes.
- **Sanitize** anything ingested from notebooks/datasets before recording it — DS
  context is full of untrusted strings. **Never persist secrets** (the gateway
  reads keys from the env at the point of use and logs none of them). Respect the
  multi-tenant IP boundary: tenant facts stay tenant-scoped (`owned_by: <tenant>` /
  `x-loom-owned-by`); the moat trains **only on consented / `general` records**.

## 8. Surface conventions

- **Flat, hyphenated, lifecycle-named verbs** (`/loom-eda`, not a nested
  `loom data list` grammar). The noun is a **single free-text arg**, not flag
  soup. Reserve a tiny meta flag set (`--from`, `--budget`, `--model`).
- **Dual surface:** every verb is a Claude Code Skill now and the same catalog
  becomes `loom <verb>` in the future binary — one verb table, both surfaces.
- **Dual-invocation:** rich `description` + `when_to_use` so the model can
  auto-load the verb, while the user can also type it. Reserve
  `disable-model-invocation: true` for irreversible/costly verbs.
- **Domain-neutral.** No customer-, vertical-, or pricing-specific content — that
  strategy lives elsewhere, never in this repo.
